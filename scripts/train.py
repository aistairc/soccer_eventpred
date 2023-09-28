import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
from _jsonnet import evaluate_file

from soccer_eventpred.data.sources.source import SoccerDataSource
from soccer_eventpred.env import OUTPUT_DIR, PROJECT_DIR
from soccer_eventpred.models.event_predictor import EventPredictor
from soccer_eventpred.modules.class_weight import ClassWeightBase
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.util import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=str, default="wyscout")
    parser.add_argument("--data-module", type=str, default="wyscout_single")
    parser.add_argument("--data-name", type=str, default="wyscout")
    parser.add_argument(
        "-c",
        "--config",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--prediction-method",
        type=str,
        default="sequence",
        help="prediction method to use",
    )
    parser.add_argument("-n", "--name", type=str, help="name of the run", required=True)
    parser.add_argument(
        "-e",
        "--exp-name",
        type=str,
        default="default",
        help="name of the experiment in mlflow",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=str,
        default=None,
        help="mapping of labels to events if any",
    )
    parser.add_argument(
        "--devices", type=int, default=-1, help="devices to use for training"
    )
    parser.add_argument(
        "--strategy", type=str, default=None, help="training strategy to use"
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="accelerator to use for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train for"
    )
    parser.add_argument(
        "--num-workers", type=int, default=16, help="number of workers for dataloader"
    )
    parser.add_argument("-g", "--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--class-weight-type", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss-function", type=str, default="cross_entropy_loss")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("-t", "--test-run", action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--ignore-tokens", type=str, default=None, nargs="+")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--sequence_encoder_dim", type=int, default=None)

    args = parser.parse_args()
    if args.encoder_dim is not None and args.sequence_encoder_dim is not None:
        params = json.loads(
            evaluate_file(
                str(args.config),
                ext_vars={
                    "encoder_dim": str(args.encoder_dim),
                    "sequence_encoder_dim": str(args.sequence_encoder_dim),
                },
            )
        )
    else:
        params = json.loads(evaluate_file(str(args.config)))
    if args.learning_rate is not None:
        params["optimizer"]["lr"] = args.learning_rate
    params_copy = deepcopy(params)

    logger.info("Loading data")
    if args.test_run:
        args.epochs = 1
        train_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "mini_train.jsonl",
            }
        )
        val_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "mini_dev.jsonl",
            }
        )
        test_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "mini_test.jsonl",
            }
        )

    else:
        train_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "train.jsonl",
            }
        )
        val_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "dev.jsonl",
            }
        )
        test_datasource = SoccerDataSource.from_params(
            params_={
                "type": args.data_source,
                "data_name": args.data_name,
                "subset": "test.jsonl",
            }
        )

    label2events = load_json(args.mapping) if args.mapping is not None else None
    datamodule = SoccerDataModule.from_params(
        params_={
            "type": args.data_module,
        },
        train_datasource=train_datasource,
        val_datasource=val_datasource,
        test_datasource=test_datasource,
        num_workers=args.num_workers,
        batch_size=params["batch_size"],
        label2events=label2events,
    )
    logger.info("Preparing datamodule...")
    datamodule.build_vocab()

    logger.info("Calculating class weights...")
    if args.class_weight_type is not None:
        if args.class_weight_type == "exponential":
            class_weight_fn = ClassWeightBase.from_params(
                params_={
                    "type": args.class_weight_type,
                    "beta": args.beta,
                }
            )
        else:
            class_weight_fn = ClassWeightBase.from_params(
                params_={
                    "type": args.class_weight_type,
                }
            )
        class_weight = class_weight_fn.calculate(
            dataset=datamodule._train_dataset,
            num_classes=datamodule.vocab.size("events"),
            ignore_indices=[
                int(datamodule.vocab.get(token, namespace="events"))
                for token in args.ignore_tokens
            ]
            if args.ignore_tokens is not None
            else None,
            class_counts=datamodule.event_counts,
        )
    else:
        class_weight = None

    # loss function
    if args.loss_function == "cross_entropy_loss":
        loss_function = {
            "type": "torch::CrossEntropyLoss",
        }
    elif args.loss_function == "focal_loss":
        loss_function = {"type": "FocalLoss", "gamma": args.focal_loss_gamma}
    else:
        raise ValueError(f"Loss function {args.loss_function} not supported")

    # model
    if args.prediction_method == "single":
        model_config = {
            "type": args.prediction_method,
            "seq2vec_encoder": params["seq2vec_encoder"],
        }
    elif args.prediction_method == "single_with_performer":
        model_config = {
            "type": args.prediction_method,
            "seq2vec_encoder": params["seq2vec_encoder"],
        }
    else:
        model_config = {
            "type": args.prediction_method,
            "seq2seq_encoder": params["seq2seq_encoder"],
        }
    model = EventPredictor.from_params(
        params_=model_config,
        time_encoder=params["time_encoder"],
        team_encoder=params["team_encoder"],
        event_encoder=params["event_encoder"],
        x_axis_encoder=params["x_axis_encoder"],
        y_axis_encoder=params["y_axis_encoder"],
        datamodule=datamodule,
        optimizer=params["optimizer"],
        loss_function=loss_function,
        player_encoder=params["player_encoder"] if "player_encoder" in params else None,
        scheduler=params["scheduler"] if "scheduler" in params else None,
        class_weight=class_weight,
    )
    output_dir = OUTPUT_DIR / args.name
    chackpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath=output_dir,
        mode="min",
        save_top_k=1,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid_loss",
        patience=5,
        mode="min",
        min_delta=0.0001,
    )
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=args.exp_name,
        run_name=args.name,
        save_dir=str(PROJECT_DIR / "mlruns"),
    )
    mlflow_logger.log_hyperparams(params_copy)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        callbacks=[chackpoint_callback, early_stopping_callback],
        logger=[mlflow_logger],
        deterministic=True,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=args.val_check_interval,
        detect_anomaly=True,
    )
    trainer.fit(model, datamodule=datamodule)
