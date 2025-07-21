#!/usr/bin/env python3
import os
import sys
import re
import logging

import numpy as np
import pandas as pd
from scipy import sparse

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import joblib  # for saving models

# ─────────────────────────────────────────────────────────────
# 0) Logging & NLTK setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="error.log",
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger()

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────
# 1) Preprocessing + LIWC loader
# ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"@\w+",    " USR ", text)
    tokens = re.findall(r"\w+", text)
    return " ".join([t for t in tokens if t not in STOPWORDS])

# (load_liwc and compute_liwc_feats same as before)
def load_liwc(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LIWC file not found: {path}")
    liwc_map = {}
    cats      = []
    with open(path, encoding="utf8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                word, catstr = line.split("\t", 1)
                clist = [c.strip() for c in catstr.split(",") if c.strip()]
                liwc_map[word.lower()] = clist
                for c in clist:
                    if c not in cats:
                        cats.append(c)
            else:
                m = re.search(r"\(([^)]+)\)", line)
                code = m.group(1) if m else line
                if code not in cats:
                    cats.append(code)
    if len(cats) != 93:
        raise ValueError(f"Expected 93 LIWC categories, got {len(cats)}")
    print(f"Loaded LIWC: {len(liwc_map)} words → {len(cats)} cats")
    return liwc_map, cats

# (load_cluster_map and compute_cluster_feats same as before)
def load_cluster_map(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cluster file not found: {path}")
    cmap = {}
    with open(path, encoding="utf8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.rstrip().split()
            w, cid = parts[0], int(parts[-1])
            cmap[w] = cid
    nclus = max(cmap.values()) + 1
    print(f"Loaded clusters: {len(cmap)} words → {nclus} clusters")
    return cmap, nclus

def compute_liwc_feats(texts, liwc_map, cat_list):
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    X = np.zeros((len(texts), len(cat_list)), dtype=float)
    for i, doc in enumerate(texts):
        tokens = doc.split()
        total  = len(tokens)
        if total == 0:
            continue
        for w in tokens:
            if w in liwc_map:
                for c in liwc_map[w]:
                    X[i, cat_to_idx[c]] += 1
        X[i, :] /= total
    return X

def compute_cluster_feats(texts, num_tweets, cmap, nclus):
    X = np.zeros((len(texts), nclus), dtype=float)
    for i, doc in enumerate(texts):
        nt = num_tweets[i]
        if nt == 0:
            continue
        for w in doc.split():
            if w in cmap:
                X[i, cmap[w]] += 1
        X[i, :] /= nt
    return X

# ─────────────────────────────────────────────────────────────
# 2) Stratified user‐level split
# ─────────────────────────────────────────────────────────────
def stratified_split(df, label_col="label", seed=555):
    X_idx = df.index.values
    y = df[label_col].values
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, temp_idx = next(s1.split(X_idx, y))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=seed)
    dev_sub, test_sub = next(s2.split(temp_idx, y[temp_idx]))
    return df.loc[tr_idx], df.loc[temp_idx[dev_sub]], df.loc[temp_idx[test_sub]]

# ─────────────────────────────────────────────────────────────
# 3) Main pipeline
# ─────────────────────────────────────────────────────────────
def main():
    try:
        # 3.1) Load & preprocess tweets
        df = pd.read_csv("cleaned_dataset.csv")
        # Map LABEL to numeric and preprocess text
        df["text"]  = df["tweet_text"].astype(str).map(preprocess)
        df["label"] = df["LABEL"].map({"reliable":0, "unreliable":1}).astype(int)

        # 3.2) Aggregate at user level, including new meta features
        user_df = (
            df.groupby("username").agg({
                "text":              " ".join,
                "label":             "first",
                "tweet_text":        "count",
                "account_age_years": "first",
                "total_posts":       "first",
                "following_count":   "first",
                "followers_count":   "first",
                "verified":          "first",
                "posting_frequency": "first",
            })
            .rename(columns={"tweet_text":"num_tweets"})
            .reset_index()
        )
        # Convert boolean verified→0/1
        user_df["verified"] = user_df["verified"].astype(int)

        # Save the preprocessed user‐level data
        user_df.to_csv("preprocessed_user_data.csv", index=False)
        print("✅ Saved preprocessed user data to preprocessed_user_data.csv")

        # 3.3) Compute static feature matrices
        liwc_map, liwc_cats = load_liwc("liwc2015.txt")
        X_liwc  = compute_liwc_feats(user_df["text"], liwc_map, liwc_cats)
        cmap, ncl = load_cluster_map("glove-200.txt")
        X_clust = compute_cluster_feats(
            user_df["text"], user_df["num_tweets"].values, cmap, ncl
        )
        # New meta features
        meta_cols = [
            "account_age_years", "total_posts",
            "following_count", "followers_count",
            "verified", "posting_frequency"
        ]
        X_meta = user_df[meta_cols].values

        # 3.4) Model tuning + evaluation
        p_grid = {"kernel":["rbf"], "C":[10,100,1e3,1e4,1e5],
                  "ngrams_range":[(1,1),(1,2),(1,3),(1,4)]}
        seeds = [555, 666, 777]
        test_scores = []

        for run_i, seed in enumerate(seeds):
            print(f"\n===== Run with seed={seed} =====")
            tr, dv, te = stratified_split(user_df, "label", seed)
            y_tr, y_dv, y_te = tr.label.values, dv.label.values, te.label.values

            # Fit TF‑IDF on text
            best = {"f1":0.0}
            for P in ParameterGrid(p_grid):
                vect = TfidfVectorizer(
                    ngram_range=P["ngrams_range"], max_features=20000,
                    min_df=5, max_df=0.4, lowercase=False
                )
                Xb_tr = vect.fit_transform(tr.text)
                Xb_dv = vect.transform(dv.text)

                # Combine all features
                idx_tr, idx_dv = tr.index.values, dv.index.values
                X_tr_all = sparse.hstack([
                    Xb_tr,
                    sparse.csr_matrix(X_liwc[idx_tr]),
                    sparse.csr_matrix(X_clust[idx_tr]),
                    sparse.csr_matrix(X_meta[idx_tr]),
                ])
                X_dv_all = sparse.hstack([
                    Xb_dv,
                    sparse.csr_matrix(X_liwc[idx_dv]),
                    sparse.csr_matrix(X_clust[idx_dv]),
                    sparse.csr_matrix(X_meta[idx_dv]),
                ])

                clf = SVC(kernel=P["kernel"], C=P["C"], random_state=seed)
                clf.fit(X_tr_all, y_tr)
                f1 = precision_recall_fscore_support(
                    y_dv, clf.predict(X_dv_all), average="macro"
                )[2]
                if f1 > best["f1"]:
                    best = {**P, "f1":f1}

            print(f"-- Best params on dev → {best}")
            vect_best = TfidfVectorizer(
                ngram_range=best["ngrams_range"], max_features=20000,
                min_df=5, max_df=0.4, lowercase=False
            )
            Xb_tr_b = vect_best.fit_transform(tr.text)
            Xb_te_b = vect_best.transform(te.text)
            idx_tr, idx_te = tr.index.values, te.index.values
            X_tr_all_b = sparse.hstack([
                Xb_tr_b,
                sparse.csr_matrix(X_liwc[idx_tr]),
                sparse.csr_matrix(X_clust[idx_tr]),
                sparse.csr_matrix(X_meta[idx_tr]),
            ])
            X_te_all_b = sparse.hstack([
                Xb_te_b,
                sparse.csr_matrix(X_liwc[idx_te]),
                sparse.csr_matrix(X_clust[idx_te]),
                sparse.csr_matrix(X_meta[idx_te]),
            ])

            clf_final = SVC(kernel=best["kernel"], C=best["C"], random_state=seed)
            clf_final.fit(X_tr_all_b, y_tr)
            joblib.dump(clf_final, f"svm_seed_{seed}.joblib")
            print(f"Saved trained model to svm_seed_{seed}.joblib")

            # Evaluation
            preds_te = clf_final.predict(X_te_all_b)
            print("Macro‐F1 (precision, recall, f1, support):")
            print(precision_recall_fscore_support(
                y_te, preds_te, average="macro"
            ))
            print("\nClassification Report:")
            print(classification_report(y_te, preds_te, digits=4, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(y_te, preds_te))

            test_scores.append(
                precision_recall_fscore_support(
                    y_te, preds_te, average="macro"
                )[2]
            )

        m, s = np.mean(test_scores), np.std(test_scores, ddof=1)
        print(f"\n=== 3-run Test macro-F1: {m:.4f} ± {s:.4f}")

    except Exception as e:
        logger.exception("Fatal")
        print(f"\n❌ {e}. See error.log for details.")
        sys.exit(1)

if __name__=="__main__":
    main()
