import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from soccer_eventpred.env import OUTPUT_DIR
from soccer_eventpred.util import load_json, save_formatted_json


def output_confusion_matrix(args, event2id, split):
    pred_path = args.val_pred_path if split == "val" else args.test_pred_path
    output_path = args.val_output_path if split == "val" else args.test_output_path
    df = pd.read_csv(pred_path)
    df = df.query("gold != 1")  # ignore padding

    if args.ignore_tokens is not None:
        ignore_token_ids = [event2id[t] for t in args.ignore_tokens]
        df = df.query("gold not in @ignore_token_ids")

    classes = [
        k
        for k, v in event2id.items()
        if v in list(df["pred"].unique()) + list(df["gold"].unique())
    ]
    classes = [k.replace("Change of possession", "CoP") for k in classes]
    golds = df["gold"].values
    preds = df["pred"].values
    cf_matrix = confusion_matrix(y_true=golds, y_pred=preds)
    cf_matrix = cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=classes)
    disp.plot()
    accuracy = accuracy_score(golds, preds) * 100
    precision = precision_score(golds, preds, average="macro") * 100
    recall = recall_score(golds, preds, average="macro") * 100
    f1 = f1_score(golds, preds, average="macro") * 100
    print(f"Confusion matrix for {split} set")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    # output
    save_formatted_json(results, args.exp_dir / f"{split}_results_separate_eval.json")
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n",
        "--run-name",
        type=str,
        help="name of the run",
    )
    group.add_argument(
        "-e",
        "--exp-dir",
        type=lambda p: Path(p).absolute(),
        help="path to checkpoint",
    )
    parser.add_argument(
        "--val-pred-path", type=lambda p: Path(p).absolute(), default=None
    )
    parser.add_argument(
        "--test-pred-path", type=lambda p: Path(p).absolute(), default=None
    )
    parser.add_argument(
        "--val-output-path", type=lambda p: Path(p).absolute(), default=None
    )
    parser.add_argument(
        "--test-output-path", type=lambda p: Path(p).absolute(), default=None
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=lambda p: Path(p).absolute(),
        default=None,
    )
    parser.add_argument("--ignore-tokens", type=str, default=None, nargs="+")
    parser.add_argument(
        "--delete-unpredicted-classes",
        action="store_true",
        help="delete classes that are not predicted",
    )
    args = parser.parse_args()
    if args.exp_dir is None:
        args.exp_dir = OUTPUT_DIR / args.run_name
    if args.val_pred_path is None:
        args.val_pred_path = args.exp_dir / "val_predictions.csv"
    if args.test_pred_path is None:
        args.test_pred_path = args.exp_dir / "test_predictions.csv"
    if args.val_output_path is None:
        args.val_output_path = args.exp_dir / "val_confusion_matrix.png"
    if args.test_output_path is None:
        args.test_output_path = args.exp_dir / "test_confusion_matrix.png"
    if args.mapping is None:
        args.mapping = args.exp_dir / "event2id.json"

    event2id = load_json(args.mapping)
    output_confusion_matrix(args, event2id, "val")
    output_confusion_matrix(args, event2id, "test")
