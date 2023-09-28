import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from _jsonnet import evaluate_file

from soccer_eventpred.data.sources.wyscout import SoccerDataSource
from soccer_eventpred.env import OUTPUT_DIR
from soccer_eventpred.models.event_predictor import EventPredictor
from soccer_eventpred.modules.class_weight import ClassWeightBase
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.util import load_json, save_as_csv, save_formatted_json

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
    parser.add_argument(
        "-m",
        "--mapping",
        type=lambda p: Path(p).absolute(),
        default=None,
        help="mapping of labels to events if any",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=lambda p: Path(p).absolute(),
        help="path to output dir",
        default=None,
    )
    parser.add_argument("--class-weight-type", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n",
        "--run-name",
        type=str,
        help="name of the run",
    )
    group.add_argument(
        "-p",
        "--checkpoint-path",
        type=lambda p: Path(p).absolute(),
        help="path to checkpoint",
    )
    parser.add_argument(
        "--num-workers", type=int, default=16, help="number of workers for dataloader"
    )
    parser.add_argument("--loss-function", type=str, default="cross_entropy_loss")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--ignore-tokens", type=str, default=None, nargs="+")
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
    datamodule.prepare_data()

    # class weight
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

    if args.checkpoint_path is not None:
        best_checkpoint = torch.load(str(args.checkpoint_path))
    else:
        exp_dir = OUTPUT_DIR / args.run_name
        try:
            best_checkpoint = torch.load(str(exp_dir / "best.ckpt"))
        except FileNotFoundError:
            best_checkpoint_path = list(sorted(exp_dir.glob("*.ckpt")))[-1]
            best_checkpoint = torch.load(str(best_checkpoint_path))
    model.load_state_dict(state_dict=best_checkpoint["state_dict"])
    trainer = pl.Trainer(
        accelerator="gpu", deterministic=True, enable_progress_bar=False, logger=False
    )
    valid_results = trainer.validate(model, datamodule=datamodule)
    test_results = trainer.test(model, datamodule=datamodule)

    val_preds = []
    val_golds = []
    for batch in datamodule.val_dataloader():
        gold, pred = model.predict(batch)
        val_preds.extend(pred.flatten().tolist())
        val_golds.extend(gold.flatten().tolist())

    test_preds = []
    test_golds = []
    for batch in datamodule.test_dataloader():
        gold, pred = model.predict(batch)
        test_preds.extend(pred.flatten().tolist())
        test_golds.extend(gold.flatten().tolist())

    ## save
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = OUTPUT_DIR / args.run_name
    save_formatted_json(valid_results, output_dir / "valid_results.json")
    save_formatted_json(test_results, output_dir / "test_results.json")
    save_formatted_json(
        datamodule.vocab._token2index["events"], output_dir / "event2id.json"
    )
    save_as_csv(
        {
            "pred": val_preds,
            "gold": val_golds,
        },
        output_dir / "val_predictions.csv",
    )
    save_as_csv(
        {
            "pred": test_preds,
            "gold": test_golds,
        },
        output_dir / "test_predictions.csv",
    )
