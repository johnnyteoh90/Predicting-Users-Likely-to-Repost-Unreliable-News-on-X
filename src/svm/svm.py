#!/usr/bin/env python3
import os, re, json, argparse, logging
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    make_scorer,
    f1_score,
)
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("svm_families_ablate")

# ---------------- fixed user feature list ----------------
USER_COLS = [
    "HU_TweetNum","HU_TweetPercent_Original","HU_TweetPercent_Retweet","HU_AverageInterval_days",
    "U_AccountAge_days","U_ListedNum","U_ProfileUrl","U_FollowerNumDay","U_FolloweeNumDay",
    "U_TweetNumDay","U_ListedNumDay","total_posts","followers_count","following_count",
    "verified","posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio"
]

EXCLUDE_ALWAYS = {"label","username","user","user_id","screen_name","aggr_text"}

# ---------------- SVM grids (Mu & Aletras) ----------------
NGRAMS = [(1,1), (1,2), (1,3), (1,4)]
CGRID  = [10, 100, 1000, 10000, 100000]

# ---------------- text utils (unchanged) ----------------
DEFAULT_STOP = {
    "the","a","an","and","or","but","if","while","is","are","was","were","be",
    "to","of","in","on","for","with","as","at","by","this","that","it","its",
    "from","we","you","i","me","my","our","your","they","their","he","she","his","her"
}
def preprocess(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"http\S+", " url ", t)
    t = re.sub(r"@\w+", " usr ", t)
    toks = re.findall(r"[a-z0-9_]+", t)
    return " ".join([w for w in toks if w not in DEFAULT_STOP])

# ---------------- cluster map loader (token ... <int cid>) ----------------
def load_cluster_map(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cluster map not found: {path}")
    cmap, n_lines = {}, 0
    with open(path, encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            try:
                cid = int(parts[-1])
            except ValueError:
                raise ValueError(
                    "Cluster map must be: token ... <int cluster id> (last token is an integer)."
                )
            token = parts[0]
            cmap[token] = cid
            n_lines += 1
            if n_lines >= 200000:
                break
    if not cmap:
        raise ValueError("No cluster entries loaded. Check the cluster file format.")
    nclus = max(cmap.values()) + 1
    log.info(f"Loaded cluster map: {len(cmap)} tokens -> {nclus} clusters")
    return cmap, nclus

def compute_topic_feats(texts, token_counts, cmap, nclus):
    # Returns a dense numpy array; we convert to sparse where needed.
    X = np.zeros((len(texts), nclus), dtype=float)
    for i, doc in enumerate(texts):
        nt = float(token_counts[i]) if token_counts[i] else 0.0
        if nt <= 0: continue
        for w in doc.split():
            cid = cmap.get(w)
            if cid is not None:
                X[i, cid] += 1.0
        if X[i].sum() > 0:
            X[i, :] /= nt
    return X

# ---------------- helpers ----------------
def to_csr(X):
    """Ensure a SciPy CSR matrix (used for topics to avoid ndarray.tolil errors)."""
    return X if sparse.issparse(X) else sparse.csr_matrix(X)

def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return p, r, f1

def build_numeric_matrix(df_sub: pd.DataFrame, cols: list):
    X = df_sub[cols].copy()
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X.values)
    return X_imp, imputer


def build_message_blocks(df_sub, vect_params, cmap, nclus):
    texts = df_sub["aggr_text"].astype(str).map(preprocess).fillna("")
    token_count = texts.map(lambda t: len(t.split()))
    vect = TfidfVectorizer(
        ngram_range=vect_params.get("ngram_range",(1,2)),
        max_features=vect_params.get("max_features",20000),
        min_df=vect_params.get("min_df",5),
        max_df=vect_params.get("max_df",0.4),
        lowercase=False,
    )
    X_bow = vect.fit_transform(texts.values)
    topics = compute_topic_feats(texts.values, token_count.values, cmap, nclus)
    X_topics = sparse.csr_matrix(topics)
    return X_bow, vect, X_topics

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="5-fold CV SVM (RBF): Users / Message / Fusion + LSA + ablations + importance + plots")
    ap.add_argument("--data", default="preprocessed_dataset_all.csv", help="Input CSV with all features + aggr_text")
    ap.add_argument("--cluster", default="glove-200.txt", help="Cluster map file (token ... <int cluster id>)")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--nrepeats", type=int, default=20, help="Permutation importance repeats (Users only)")
    ap.add_argument("--topics_topk", type=int, default=0, help="Top-K topic columns to permute for importance (0=disable)")
    # NEW: LSA / SVD grid and scaling
    ap.add_argument("--svd_grid", type=str, default="0,300,500,800",
                    help="Comma-separated SVD components for BoW; 0=off (raw BoW)")
    ap.add_argument("--scale_svd", action="store_true",
                    help="Z-score SVD components before stacking (recommended)")
    args = ap.parse_args()

    # Prepare output dirs
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "models"))
    ensure_dir(os.path.join(args.outdir, "diagnostics"))
    ensure_dir(os.path.join(args.outdir, "plots"))

    df = pd.read_csv(args.data)

    # checks
    missing_users = [c for c in USER_COLS if c not in df.columns]
    if missing_users:
        raise ValueError(f"Missing user feature columns in CSV: {missing_users}")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    if "aggr_text" not in df.columns:
        raise ValueError("CSV must contain 'aggr_text' for message features.")

    # encode label
    df["label"] = df["label"].map({"reliable":0, "unreliable":1}).astype(int)

    # message numeric columns = everything NOT in user list and not excluded
    msg_numeric_cols = [c for c in df.columns if c not in USER_COLS and c not in EXCLUDE_ALWAYS]
    log.info(f"Using {len(USER_COLS)} user columns, {len(msg_numeric_cols)} message numeric columns.")

    # load cluster map once
    cmap, nclus = load_cluster_map(args.cluster)

    # SVD grid
    svd_grid = [int(x) for x in args.svd_grid.split(",") if x.strip() != ""]
    if not svd_grid: svd_grid = [0]

    # CV
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    per_fold = []
    all_user_lofo, all_msg_block_abl = [], []
    agg_user_perm = {}          # Users permutation importance
    all_msg_num_perm = []       # message-numeric permutation rows
    all_topics_perm = []        # topic top-K permutation rows
    svd_dev_records = []        # (fold, k, dev_f1)
    svd_test_cmp = []           # (fold, k_selected, f1_selected, f1_k0)

    fold_idx = 0
    for trainval_idx, test_idx in skf.split(df, df["label"]):
        fold_idx += 1
        df_trainval = df.iloc[trainval_idx].copy()
        df_test = df.iloc[test_idx].copy()

        # small inner dev from trainval (for picking C / ngram_range / SVD k)
        train_df, dev_df = train_test_split(
            df_trainval, test_size=0.125, stratify=df_trainval["label"], random_state=args.seed
        )

        # common labels
        y_tr, y_dv, y_te = train_df["label"].values, dev_df["label"].values, df_test["label"].values

        # ===================== USERS =====================
        Xu_tr, imp_u = build_numeric_matrix(train_df, USER_COLS)
        Xu_dv = imp_u.transform(dev_df[USER_COLS].apply(pd.to_numeric, errors="coerce").values)
        Xu_te = imp_u.transform(df_test[USER_COLS].apply(pd.to_numeric, errors="coerce").values)

        best_users = {"C": None, "f1": -1}
        for Cval in CGRID:
            clf = SVC(kernel="rbf", C=Cval, random_state=args.seed)
            clf.fit(Xu_tr, y_tr)
            pred_dv = clf.predict(Xu_dv)
            _, _, f1 = macro_scores(y_dv, pred_dv)
            if f1 > best_users["f1"]:
                best_users = {"C": Cval, "f1": f1}

        clf_u = SVC(kernel="rbf", C=best_users["C"], random_state=args.seed).fit(
            np.vstack([Xu_tr, Xu_dv]), np.concatenate([y_tr, y_dv])
        )
        pred_u = clf_u.predict(Xu_te)
        Pu, Ru, F1u = macro_scores(y_te, pred_u)
        print(f"\n[FOLD {fold_idx}][Users]  P={Pu:.4f} R={Ru:.4f} F1={F1u:.4f}")
        print(classification_report(y_te, pred_u, digits=4, zero_division=0))
        joblib.dump(clf_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_svm_users.joblib"))
        joblib.dump(imp_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_users.joblib"))

        # --- Users LOFO ablation ---
        u_base_f1 = F1u
        Xu_tr_plus = np.vstack([Xu_tr, Xu_dv])
        y_tr_plus  = np.concatenate([y_tr, y_dv])

        lofo_rows = []
        for j, col in enumerate(USER_COLS):
            keep_idx = [i for i in range(len(USER_COLS)) if i != j]
            Xu_tr_lofo = Xu_tr_plus[:, keep_idx]
            Xu_te_lofo = Xu_te[:, keep_idx]
            clf_lofo = SVC(kernel="rbf", C=best_users["C"], random_state=args.seed).fit(Xu_tr_lofo, y_tr_plus)
            f1_lofo = macro_scores(y_te, clf_lofo.predict(Xu_te_lofo))[2]
            row = {"fold": fold_idx, "feature": col, "delta_F1": (f1_lofo - u_base_f1)}
            lofo_rows.append(row)
            all_user_lofo.append(row)

        pd.DataFrame(lofo_rows).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_ablation_users_lofo.csv"),
            index=False
        )

        # --- Users permutation importance (macro-F1 drop on test) ---
        scorer = make_scorer(f1_score, average="macro")
        perm = permutation_importance(
            clf_u, Xu_te, y_te, scoring=scorer, n_repeats=args.nrepeats,
            random_state=args.seed, n_jobs=4
        )
        pi_df = (pd.DataFrame({
            "feature": USER_COLS,
            "imp_mean": perm.importances_mean,  # mean F1 drop
            "imp_std":  perm.importances_std
        }).sort_values("imp_mean", ascending=False))
        pi_df.to_csv(os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_perm_users.csv"), index=False)
        for name, drop in zip(USER_COLS, perm.importances_mean):
            agg_user_perm[name] = agg_user_perm.get(name, []) + [float(drop)]

        # ===================== MESSAGE (with LSA / SVD grid) =====================
        best_msg = {"ngram_range": None, "C": None, "svd_k": 0, "f1": -1, "vect": None,
                    "svd": None, "svd_scaler": None,
                    "X_tr_msg_dev": None, "X_dv_msg_dev": None}
        best_by_k = {}  # k -> dict like best_msg (per-k best combo)
        dev_scores_by_k = {k: -1 for k in svd_grid}

        # build numeric message blocks once per split
        Xm_tr, imp_m = build_numeric_matrix(train_df, msg_numeric_cols)
        Xm_dv = imp_m.transform(dev_df[msg_numeric_cols].apply(pd.to_numeric, errors="coerce").values)
        Xm_te = imp_m.transform(df_test[msg_numeric_cols].apply(pd.to_numeric, errors="coerce").values)

        for ngr in NGRAMS:
            Xb_tr, vect_m_try, Xt_tr = build_message_blocks(train_df, {"ngram_range": ngr}, cmap, nclus)
            Xb_dv = vect_m_try.transform(dev_df["aggr_text"].astype(str).map(preprocess).values)
            # topics on dev (ensure sparse)
            Xt_dv = to_csr(compute_topic_feats(
                dev_df["aggr_text"].astype(str).map(preprocess).values,
                dev_df["aggr_text"].astype(str).map(lambda t: len(preprocess(t).split())).values, cmap, nclus
            ))

            log.info(f"[Fold {fold_idx}] ngram={ngr} TF-IDF n_features={Xb_tr.shape[1]} n_docs={Xb_tr.shape[0]}")

            for k in svd_grid:
                if k > 0:
                    nfeat = Xb_tr.shape[1]
                    if k >= nfeat:
                        log.warning(f"[Fold {fold_idx}] Skip k={k} (allowed <= {nfeat-1}) since n_features={nfeat}")
                        continue
                    svd = TruncatedSVD(n_components=k, random_state=args.seed)
                    Z_tr = svd.fit_transform(Xb_tr)
                    Z_dv = svd.transform(Xb_dv)
                    if args.scale_svd:
                        zsc = StandardScaler()
                        Z_tr = zsc.fit_transform(Z_tr)
                        Z_dv = zsc.transform(Z_dv)
                else:
                    svd, zsc = None, None
                    Z_tr, Z_dv = Xb_tr, Xb_dv

                X_tr_msg = sparse.hstack([sparse.csr_matrix(Z_tr) if k>0 else Z_tr,
                                          Xt_tr, sparse.csr_matrix(Xm_tr)]).tocsr()
                X_dv_msg = sparse.hstack([sparse.csr_matrix(Z_dv) if k>0 else Xb_dv,
                                          Xt_dv, sparse.csr_matrix(Xm_dv)]).tocsr()

                for Cval in CGRID:
                    clf = SVC(kernel="rbf", C=Cval, random_state=args.seed)
                    clf.fit(X_tr_msg, y_tr)
                    f1 = macro_scores(y_dv, clf.predict(X_dv_msg))[2]
                    if f1 > dev_scores_by_k[k]:
                        dev_scores_by_k[k] = f1
                        best_by_k[k] = {"ngram_range": ngr, "C": Cval, "svd_k": k, "f1": f1,
                                        "vect": vect_m_try, "svd": svd, "svd_scaler": zsc,
                                        "X_tr_msg_dev": X_tr_msg, "X_dv_msg_dev": X_dv_msg}
                    if f1 > best_msg["f1"]:
                        best_msg = {"ngram_range": ngr, "C": Cval, "svd_k": k, "f1": f1,
                                    "vect": vect_m_try, "svd": svd, "svd_scaler": zsc,
                                    "X_tr_msg_dev": X_tr_msg, "X_dv_msg_dev": X_dv_msg}

        # record dev sweep
        for k in svd_grid:
            svd_dev_records.append({"fold": fold_idx, "k": k, "dev_F1": float(dev_scores_by_k[k])})

        # rebuild with best msg settings and train on train+dev
        vect_m = best_msg["vect"]; ngr = best_msg["ngram_range"]; ksel = best_msg["svd_k"]
        Xb_tr_final, vect_m, Xt_tr_final = build_message_blocks(
            pd.concat([train_df, dev_df], axis=0), {"ngram_range": ngr}, cmap, nclus
        )
        y_tr_plus = np.concatenate([y_tr, y_dv])

        # IMPORTANT: refit imputer on train+dev, then re-transform test to avoid mismatch
        Xm_tr_plus = imp_m.fit_transform(pd.concat([train_df[msg_numeric_cols], dev_df[msg_numeric_cols]])
                                         .apply(pd.to_numeric, errors="coerce").values)
        Xb_te = vect_m.transform(df_test["aggr_text"].astype(str).map(preprocess).values)
        Xt_te = to_csr(compute_topic_feats(
            df_test["aggr_text"].astype(str).map(preprocess).values,
            df_test["aggr_text"].astype(str).map(lambda t: len(preprocess(t).split())).values, cmap, nclus
        ))
        Xm_te = imp_m.transform(df_test[msg_numeric_cols].apply(pd.to_numeric, errors="coerce").values)

        # Final Message matrices (best-k)
        if ksel > 0:
            nfeat_final = Xb_tr_final.shape[1]
            if ksel >= nfeat_final:
                log.warning(f"[Fold {fold_idx}] Adjust k={ksel} -> {nfeat_final-1} (final fit) since n_features={nfeat_final}")
                ksel = max(1, nfeat_final - 1)
            svd_m = TruncatedSVD(n_components=ksel, random_state=args.seed)
            Z_tr_final = svd_m.fit_transform(Xb_tr_final)
            Z_te = svd_m.transform(Xb_te)
            if args.scale_svd:
                zsc_m = StandardScaler()
                Z_tr_final = zsc_m.fit_transform(Z_tr_final)
                Z_te = zsc_m.transform(Z_te)
        else:
            svd_m, zsc_m = None, None
            Z_tr_final, Z_te = Xb_tr_final, Xb_te

        X_tr_msg_final = sparse.hstack([sparse.csr_matrix(Z_tr_final) if ksel>0 else Z_tr_final,
                                        Xt_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        X_te_msg_final = sparse.hstack([sparse.csr_matrix(Z_te) if ksel>0 else Xb_te,
                                        Xt_te,       sparse.csr_matrix(Xm_te)]).tocsr()

        clf_m = SVC(kernel="rbf", C=best_msg["C"], random_state=args.seed).fit(X_tr_msg_final, y_tr_plus)
        pred_m = clf_m.predict(X_te_msg_final)
        Pm, Rm, F1m = macro_scores(y_te, pred_m)
        print(f"\n[FOLD {fold_idx}][Message (k={ksel})] P={Pm:.4f} R={Rm:.4f} F1={F1m:.4f}")
        print(classification_report(y_te, pred_m, digits=4, zero_division=0))

        joblib.dump(clf_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_svm_message.joblib"))
        joblib.dump(vect_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_tfidf_message.joblib"))
        joblib.dump(imp_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_message.joblib"))
        if ksel > 0:
            joblib.dump(svd_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_svd_message.joblib"))
            if args.scale_svd:
                joblib.dump(zsc_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_svd_scaler_message.joblib"))

        # --- Message block ablations (best-k)
        m_base_f1 = F1m
        abl_rows = []
        # drop BoW (or SVD-BoW)
        Xtr_no_bow = sparse.hstack([Xt_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        Xte_no_bow = sparse.hstack([Xt_te,       sparse.csr_matrix(Xm_te)]).tocsr()
        clf_nb = SVC(kernel="rbf", C=best_msg["C"], random_state=args.seed).fit(Xtr_no_bow, y_tr_plus)
        f1_nb = macro_scores(y_te, clf_nb.predict(Xte_no_bow))[2]
        abl_rows.append({"fold": fold_idx, "svd_k": ksel, "block": "no_bow", "delta_F1": (f1_nb - m_base_f1)})
        # drop Topics
        Xtr_no_topics = sparse.hstack([sparse.csr_matrix(Z_tr_final) if ksel>0 else Xb_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        Xte_no_topics = sparse.hstack([sparse.csr_matrix(Z_te) if ksel>0 else Xb_te,             sparse.csr_matrix(Xm_te)]).tocsr()
        clf_nt = SVC(kernel="rbf", C=best_msg["C"], random_state=args.seed).fit(Xtr_no_topics, y_tr_plus)
        f1_nt = macro_scores(y_te, clf_nt.predict(Xte_no_topics))[2]
        abl_rows.append({"fold": fold_idx, "svd_k": ksel, "block": "no_topics", "delta_F1": (f1_nt - m_base_f1)})
        # drop message numeric (LIWC-style)
        Xtr_no_mnum = sparse.hstack([sparse.csr_matrix(Z_tr_final) if ksel>0 else Xb_tr_final, Xt_tr_final]).tocsr()
        Xte_no_mnum = sparse.hstack([sparse.csr_matrix(Z_te) if ksel>0 else Xb_te,             Xt_te]).tocsr()
        clf_nm = SVC(kernel="rbf", C=best_msg["C"], random_state=args.seed).fit(Xtr_no_mnum, y_tr_plus)
        f1_nm = macro_scores(y_te, clf_nm.predict(Xte_no_mnum))[2]
        abl_rows.append({"fold": fold_idx, "svd_k": ksel, "block": "no_msg_numeric", "delta_F1": (f1_nm - m_base_f1)})

        pd.DataFrame(abl_rows).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_ablation_message_blocks_bestk.csv"),
            index=False
        )
        all_msg_block_abl.extend(abl_rows)

        # --- Baseline: build and test k=0 (raw BoW) for paired tests & ablation comparison
        k0_conf = best_by_k.get(0, None)
        if k0_conf is not None:
            ngr0, C0 = k0_conf["ngram_range"], k0_conf["C"]
            Xb_tr0, vect0, Xt_tr0 = build_message_blocks(pd.concat([train_df, dev_df], axis=0), {"ngram_range": ngr0}, cmap, nclus)
            Xb_te0 = vect0.transform(df_test["aggr_text"].astype(str).map(preprocess).values)
            X_tr_k0 = sparse.hstack([Xb_tr0, Xt_tr0, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
            X_te_k0 = sparse.hstack([Xb_te0, Xt_te,  sparse.csr_matrix(Xm_te)]).tocsr()
            clf_k0 = SVC(kernel="rbf", C=C0, random_state=args.seed).fit(X_tr_k0, y_tr_plus)
            f1_k0 = macro_scores(y_te, clf_k0.predict(X_te_k0))[2]

            # ablations at k=0
            rows_k0 = []
            # no_bow
            Xtr0_no_bow = sparse.hstack([Xt_tr0, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
            Xte0_no_bow = sparse.hstack([Xt_te,  sparse.csr_matrix(Xm_te)]).tocsr()
            f1_nb0 = macro_scores(y_te, SVC(kernel="rbf", C=C0, random_state=args.seed).fit(Xtr0_no_bow, y_tr_plus).predict(Xte0_no_bow))[2]
            rows_k0.append({"fold": fold_idx, "svd_k": 0, "block": "no_bow", "delta_F1": (f1_nb0 - f1_k0)})
            # no_topics
            Xtr0_no_topics = sparse.hstack([Xb_tr0, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
            Xte0_no_topics = sparse.hstack([Xb_te0, sparse.csr_matrix(Xm_te)]).tocsr()
            f1_nt0 = macro_scores(y_te, SVC(kernel="rbf", C=C0, random_state=args.seed).fit(Xtr0_no_topics, y_tr_plus).predict(Xte0_no_topics))[2]
            rows_k0.append({"fold": fold_idx, "svd_k": 0, "block": "no_topics", "delta_F1": (f1_nt0 - f1_k0)})
            # no_msg_numeric
            Xtr0_no_mnum = sparse.hstack([Xb_tr0, Xt_tr0]).tocsr()
            Xte0_no_mnum = sparse.hstack([Xb_te0, Xt_te]).tocsr()
            f1_nm0 = macro_scores(y_te, SVC(kernel="rbf", C=C0, random_state=args.seed).fit(Xtr0_no_mnum, y_tr_plus).predict(Xte0_no_mnum))[2]
            rows_k0.append({"fold": fold_idx, "svd_k": 0, "block": "no_msg_numeric", "delta_F1": (f1_nm0 - f1_k0)})

            pd.DataFrame(rows_k0).to_csv(
                os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_ablation_message_blocks_k0.csv"),
                index=False
            )

            svd_test_cmp.append({"fold": fold_idx, "k_selected": ksel, "F1_selected": float(F1m), "F1_k0": float(f1_k0)})

        # --- Message-numeric permutation importance (per-column on test)
        rng = np.random.default_rng(args.seed + 1000 * fold_idx)
        mnum_dim = Xm_te.shape[1]
        mnum_rows = []
        mnum_names = [f"mnum:{c}" for c in msg_numeric_cols]
        for j in range(mnum_dim):
            Xm_te_perm = Xm_te.copy()
            Xm_te_perm[:, j] = rng.permutation(Xm_te_perm[:, j])
            X_te_perm = sparse.hstack([sparse.csr_matrix(Z_te) if ksel>0 else Xb_te, Xt_te, sparse.csr_matrix(Xm_te_perm)]).tocsr()
            f1p = macro_scores(y_te, clf_m.predict(X_te_perm))[2]
            mnum_rows.append({"fold": fold_idx, "feature": mnum_names[j], "drop_F1": float(m_base_f1 - f1p)})
        pd.DataFrame(mnum_rows).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_perm_message_numeric.csv"),
            index=False
        )
        all_msg_num_perm.extend(mnum_rows)

        # --- OPTIONAL: Topics top-K permutation importance
        if args.topics_topk > 0 and Xt_te.shape[1] > 0:
            Xt_te = to_csr(Xt_te)  # ensure sparse before .tolil()
            col_sums = np.asarray(Xt_te.sum(axis=0)).ravel()
            order = np.argsort(col_sums)[::-1]
            topk = int(min(args.topics_topk, Xt_te.shape[1]))
            top_idx = order[:topk]
            topic_rows = []
            for j in top_idx:
                Xt_te_perm = Xt_te.tolil()
                col = Xt_te[:, j].toarray().ravel()
                Xt_te_perm[:, j] = rng.permutation(col).reshape(-1, 1)
                Xt_te_perm = Xt_te_perm.tocsr()
                X_te_perm = sparse.hstack([sparse.csr_matrix(Z_te) if ksel>0 else Xb_te, Xt_te_perm, sparse.csr_matrix(Xm_te)]).tocsr()
                f1p = macro_scores(y_te, clf_m.predict(X_te_perm))[2]
                topic_rows.append({"fold": fold_idx, "feature": f"topic:{int(j)}", "drop_F1": float(m_base_f1 - f1p)})
            pd.DataFrame(topic_rows).to_csv(
                os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_perm_topics_topk.csv"),
                index=False
            )
            all_topics_perm.extend(topic_rows)

        # ===================== FUSION =====================
        # dev message matrices (for best-k)
        X_tr_msg_dev = best_msg["X_tr_msg_dev"]
        X_dv_msg_dev = best_msg["X_dv_msg_dev"]
        X_tr_fus_dev = sparse.hstack([X_tr_msg_dev, sparse.csr_matrix(Xu_tr)]).tocsr()
        X_dv_fus_dev = sparse.hstack([X_dv_msg_dev, sparse.csr_matrix(Xu_dv)]).tocsr()

        best_f = {"C": None, "f1": -1}
        for Cval in CGRID:
            clf = SVC(kernel="rbf", C=Cval, random_state=args.seed)
            clf.fit(X_tr_fus_dev, y_tr)
            pred = clf.predict(X_dv_fus_dev)
            _, _, f1 = macro_scores(y_dv, pred)
            if f1 > best_f["f1"]:
                best_f = {"C": Cval, "f1": f1}

        X_tr_fus_final = sparse.hstack([X_tr_msg_final, sparse.csr_matrix(np.vstack([Xu_tr, Xu_dv]))]).tocsr()
        X_te_fus_final = sparse.hstack([X_te_msg_final,  sparse.csr_matrix(Xu_te)]).tocsr()

        clf_f = SVC(kernel="rbf", C=best_f["C"], random_state=args.seed).fit(X_tr_fus_final, y_tr_plus)
        pred_f = clf_f.predict(X_te_fus_final)
        Pf, Rf, F1f = macro_scores(y_te, pred_f)
        print(f"\n[FOLD {fold_idx}][Fusion]  P={Pf:.4f} R={Rf:.4f} F1={F1f:.4f}")
        print(classification_report(y_te, pred_f, digits=4, zero_division=0))

        joblib.dump(clf_f, os.path.join(args.outdir, "models", f"fold{fold_idx}_svm_fusion.joblib"))
        joblib.dump(vect_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_tfidf_fusion.joblib"))

        # --- record per fold metrics ---
        per_fold += [
            {"fold": fold_idx, "family": "Users",   "P_macro": Pu, "R_macro": Ru, "F1_macro": F1u},
            {"fold": fold_idx, "family": "Message", "P_macro": Pm, "R_macro": Rm, "F1_macro": F1m},
            {"fold": fold_idx, "family": "Fusion",  "P_macro": Pf, "R_macro": Rf, "F1_macro": F1f},
        ]

    # ========== save per-fold and summary ==========
    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(os.path.join(args.outdir, "results_cv5_per_fold.csv"), index=False)

    summary_rows = []
    for fam in ["Users","Message","Fusion"]:
        fam_df = per_fold_df[per_fold_df["family"] == fam]
        summary_rows.append({
            "family": fam,
            "P_macro_mean": fam_df["P_macro"].mean(), "P_macro_std": fam_df["P_macro"].std(ddof=1),
            "R_macro_mean": fam_df["R_macro"].mean(), "R_macro_std": fam_df["R_macro"].std(ddof=1),
            "F1_macro_mean": fam_df["F1_macro"].mean(), "F1_macro_std": fam_df["F1_macro"].std(ddof=1),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.outdir, "results_cv5_summary.csv"), index=False)
    print("\n=== 5-fold Stratified CV (macro, SVM-RBF) ===")
    print(summary_df.to_string(index=False))

    # ===== Diagnostics aggregation =====
    # Users LOFO
    pd.DataFrame(all_user_lofo).to_csv(
        os.path.join(args.outdir, "diagnostics", "ablation_users_lofo.csv"), index=False
    )
    # Message block ablations (best-k only)
    pd.DataFrame(all_msg_block_abl).to_csv(
        os.path.join(args.outdir, "diagnostics", "ablation_message_blocks_bestk.csv"), index=False
    )

    # Users permutation importance (aggregate across folds)
    rows = []
    for k, vals in agg_user_perm.items():
        rows.append({
            "feature": k,
            "perm_drop_mean": float(np.mean(vals)),
            "perm_drop_std":  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        })
    out = pd.DataFrame(rows).sort_values("perm_drop_mean", ascending=False)
    out.to_csv(os.path.join(args.outdir, "diagnostics", "importance_users_agg.csv"), index=False)

    # Message-numeric permutation (aggregate)
    if len(all_msg_num_perm) > 0:
        mn_agg = (pd.DataFrame(all_msg_num_perm)
                    .groupby("feature")["drop_F1"]
                    .agg(["mean","std"])
                    .reset_index()
                    .rename(columns={"mean":"perm_drop_mean","std":"perm_drop_std"})
                    .sort_values("perm_drop_mean", ascending=False))
        mn_agg.to_csv(os.path.join(args.outdir, "diagnostics", "importance_message_numeric_agg.csv"), index=False)

    # Topics top-K permutation (aggregate)
    if len(all_topics_perm) > 0:
        tp_agg = (pd.DataFrame(all_topics_perm)
                    .groupby("feature")["drop_F1"]
                    .agg(["mean","std"])
                    .reset_index()
                    .rename(columns={"mean":"perm_drop_mean","std":"perm_drop_std"})
                    .sort_values("perm_drop_mean", ascending=False))
        tp_agg.to_csv(os.path.join(args.outdir, "diagnostics", "importance_topics_perm_agg.csv"), index=False)

    # --- SVD dev sweep (for plot Macro-F1 vs k)
    svd_dev_df = pd.DataFrame(svd_dev_records)
    svd_dev_df.to_csv(os.path.join(args.outdir, "diagnostics", "svd_dev_sweep.csv"), index=False)

    # --- Message best-k vs k=0 on TEST (for paired test)
    if len(svd_test_cmp) > 0:
        cmp_df = pd.DataFrame(svd_test_cmp)
        cmp_df.to_csv(os.path.join(args.outdir, "diagnostics", "m_f1_selected_vs_k0.csv"), index=False)
        try:
            stat = wilcoxon(cmp_df["F1_selected"], cmp_df["F1_k0"], zero_method="wilcox", alternative="two-sided")
            print(f"\n[Paired Wilcoxon] Message F1 (best-k) vs (k=0): W={stat.statistic:.3f}, p={stat.pvalue:.6f}")
            with open(os.path.join(args.outdir, "diagnostics", "paired_tests.json"), "w", encoding="utf-8") as f:
                json.dump({"wilcoxon_bestk_vs_k0": {"W": float(stat.statistic), "pvalue": float(stat.pvalue)}}, f, indent=2)
        except Exception as e:
            print(f"[Paired Wilcoxon] could not be computed: {e}")

    # ===== Plots =====
    try:
        # Plot: Macro-F1 vs k (DEV)
        dev_m = svd_dev_df.groupby("k")["dev_F1"].agg(["mean","std"]).reset_index()
        plt.figure()
        plt.errorbar(dev_m["k"], dev_m["mean"], yerr=dev_m["std"], fmt="-o")
        plt.xlabel("SVD components (k)"); plt.ylabel("Macro-F1 (dev)"); plt.title("Macro-F1 vs k (dev)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join(args.outdir, "plots", "macro_f1_vs_k_dev.png"), bbox_inches="tight", dpi=160)
        plt.close()

        # Plot: Ablations with vs without SVD (grouped means)
        bestk_abls = pd.read_csv(os.path.join(args.outdir, "diagnostics", "ablation_message_blocks_bestk.csv"))
        mean_bestk = bestk_abls.groupby("block")["delta_F1"].mean()
        # If a k0 ablation file exists (from any fold), aggregate them:
        k0_files = [f for f in os.listdir(os.path.join(args.outdir, "diagnostics")) if f.endswith("_ablation_message_blocks_k0.csv")]
        if k0_files:
            k0_all = pd.concat([pd.read_csv(os.path.join(args.outdir, "diagnostics", f)) for f in k0_files], ignore_index=True)
            mean_k0 = k0_all.groupby("block")["delta_F1"].mean()
            blocks = ["no_bow", "no_topics", "no_msg_numeric"]
            x = np.arange(len(blocks))
            w = 0.35
            plt.figure()
            plt.bar(x - w/2, [mean_k0.get(b, 0.0) for b in blocks], width=w, label="k=0 (raw BoW)")
            plt.bar(x + w/2, [mean_bestk.get(b, 0.0) for b in blocks], width=w, label="best-k (LSA)")
            plt.xticks(x, blocks); plt.ylabel("ΔF1 (ablation)"); plt.title("Ablation deltas: raw vs LSA")
            plt.legend()
            plt.grid(True, axis="y", linestyle="--", alpha=0.3)
            plt.savefig(os.path.join(args.outdir, "plots", "ablations_svd_vs_raw.png"), bbox_inches="tight", dpi=160)
            plt.close()
    except Exception as e:
        print(f"[Plotting] skipped due to: {e}")

    # ===== Print thesis-style blocks (Requirement 2 formatting) =====
    def block(df, fam_name: str, banner: str):
        fam = df[df["family"] == fam_name]
        return (f"\n[{banner}]\n"
                f"Precision: {fam['P_macro'].mean():.4f} ± {fam['P_macro'].std(ddof=1):.4f}\n"
                f"Recall   : {fam['R_macro'].mean():.4f} ± {fam['R_macro'].std(ddof=1):.4f}\n"
                f"F1       : {fam['F1_macro'].mean():.4f} ± {fam['F1_macro'].std(ddof=1):.4f}\n")

    print(block(per_fold_df, "Message", "TEXT").rstrip())
    print(block(per_fold_df, "Users", "USER").rstrip())
    print(block(per_fold_df, "Fusion", "FUSION").rstrip())

    # config snapshot
    with open(os.path.join(args.outdir, "svm_feature_config.json"), "w", encoding="utf8") as f:
        json.dump({
            "user_cols": USER_COLS,
            "message_numeric_cols": msg_numeric_cols,
            "kfolds": args.kfolds,
            "seed": args.seed,
            "cluster_file": args.cluster,
            "ngrams": NGRAMS,
            "C_grid": CGRID,
            "perm_n_repeats": args.nrepeats,
            "topics_topk": args.topics_topk,
            "svd_grid": svd_grid,
            "scale_svd": args.scale_svd
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
