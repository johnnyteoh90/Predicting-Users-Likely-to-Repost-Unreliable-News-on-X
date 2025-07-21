#!/usr/bin/env python3
import logging
import sys
import re

import pandas as pd
import numpy as np
import torch

import nltk
from nltk.corpus import stopwords

from simpletransformers.classification import ClassificationModel
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit

# — 0) Logging & NLTK setup ---------------------------------------------------------------------------------
logging.basicConfig(
    filename="error.log",
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger()

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    """Lowercase, replace URLs and @mentions, tokenize & remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"@\w+",    " USR ", text)
    tokens = re.findall(r"\w+", text)
    return " ".join([t for t in tokens if t not in STOPWORDS])

def main():
    # — 1) Load & aggregate per-user ----------------------------------------------------------------
    df = pd.read_csv("cleaned_dataset.csv")

    user_df = (
        df
        .groupby("username", as_index=False)
        .agg({
            "tweet_text": lambda texts: " ".join(texts.astype(str).fillna("")),
            "LABEL":      "first"
        })
    )
    user_df.rename(columns={"tweet_text": "text"}, inplace=True)
    user_df["labels"] = (
        user_df["LABEL"]
        .map({"reliable": 0, "unreliable": 1})
        .astype(int)
    )
    user_df = user_df[["username", "text", "labels"]]

    # save & preview
    user_df.to_csv("users_agg.csv", index=False)
    print("Aggregated to users_agg.csv; sample:")
    print(user_df.head(), "\n")

    # — 2) Stratified 70/10/20 split ----------------------------------------------------------------
    X = user_df.index.values
    y = user_df["labels"].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=555)
    train_idx, temp_idx = next(sss1.split(X, y))

    temp_y = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=555)
    dev_sub, test_sub = next(sss2.split(temp_idx, temp_y))
    dev_idx, test_idx = temp_idx[dev_sub], temp_idx[test_sub]

    train_df = user_df.loc[train_idx, ["text", "labels"]].copy()
    dev_df   = user_df.loc[dev_idx,   ["text", "labels"]].copy()
    test_df  = user_df.loc[test_idx,  ["text", "labels"]].copy()

    # save splits
    train_df.to_csv("train.csv", index=False)
    dev_df.to_csv("de1.csv",   index=False)
    test_df.to_csv("te1.csv",  index=False)
    print(f"Train size: {len(train_df)} | Dev size: {len(dev_df)} | Test size: {len(test_df)}\n")

    # — 3) Preprocess *before* passing to BERT -------------------------------------------------------
    for split in (train_df, dev_df, test_df):
        split["text"] = split["text"].astype(str).map(preprocess)

    # — 4) Training args ---------------------------------------------------------------------------
    max_len = 512
    train_args = {
        "sliding_window":           False,
        "stride":                   max_len,
        "reprocess_input_data":     True,
        "overwrite_output_dir":     True,
        "evaluate_during_training": True,
        "save_model_every_epoch":   True,
        "train_batch_size":         8,
        "eval_batch_size":          8,
        "learning_rate":            2e-5,
        "num_train_epochs":         4,
        "max_seq_length":           max_len,
        "process_count":            1,
        "use_multiprocessing":      False,
    }

    # — 5) Repeat 3 runs with different seeds ------------------------------------------------------
    seeds = [555, 666, 777]
    all_f1 = []

    for seed in seeds:
        print(f"\n===== Run with seed={seed} =====")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # update dirs per-run
        train_args["output_dir"]     = f"outputs/bert_seed_{seed}/"
        train_args["best_model_dir"] = f"outputs/bert_seed_{seed}/best_model/"

        model = ClassificationModel(
            "bert",
            "bert-base-uncased",
            args=train_args,
            use_cuda=False,
            cuda_device=-1,
        )

        print(f"Training with seed={seed}…")
        model.train_model(train_df, eval_df=dev_df)

        print(f"\nEvaluating with seed={seed} on test set…")
        # reload best checkpoint
        model = ClassificationModel(
            "bert",
            train_args["best_model_dir"],
            args=train_args,
            use_cuda=False,
            cuda_device=-1,
        )

        preds, _ = model.predict(test_df["text"].tolist())
        y_true   = test_df["labels"].tolist()

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="macro", zero_division=0
        )
        print(f"\nMacro-F1 for seed={seed}: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, preds, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, preds))

        all_f1.append(f1)

    # — 6) Summary ---------------------------------------------------------------------------------
    avg_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1, ddof=1)
    print(f"\n=== 3-run Average Macro-F1: {avg_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user — exiting.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled exception in bert.py")
        print(f"\n❌ Error: {e}. See error.log for details.")
        sys.exit(1)