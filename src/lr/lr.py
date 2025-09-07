#!/usr/bin/env python3
import os, re, json, argparse, logging, warnings
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import joblib

import nltk
from nltk.corpus import stopwords

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lr_families")

warnings.filterwarnings("once", category=ConvergenceWarning)

# ---------------- fixed user feature list ----------------
USER_COLS = [
    "HU_TweetNum","HU_TweetPercent_Original","HU_TweetPercent_Retweet","HU_AverageInterval_days",
    "U_AccountAge_days","U_ListedNum","U_ProfileUrl","U_FollowerNumDay","U_FolloweeNumDay",
    "U_TweetNumDay","U_ListedNumDay","total_posts","followers_count","following_count",
    "verified","posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio"
]
EXCLUDE_ALWAYS = {"label","username","user","user_id","screen_name","aggr_text"}

# ---------------- LR/text grids (same values as your original lr.py) ----------------
Cs          = [10, 100, 1e3, 1e4, 1e5]
NGRAMS      = [(1,1), (1,2), (1,3), (1,4)]
MIN_DFS     = [2, 3, 5]
L1_RATIOS   = [0.1, 0.5, 0.9]

# two grids: one for LR only (Users/Fusion tuning), one for Message (LR + text)
LR_GRID = (
    [{"penalty": "l1", "C": C, "l1_ratio": None} for C in Cs] +
    [{"penalty": "l2", "C": C, "l1_ratio": None} for C in Cs] +
    [{"penalty": "elasticnet", "C": C, "l1_ratio": r} for C in Cs for r in L1_RATIOS]
)
LR_TEXT_GRID = [
    {**g, "ngram_range": ng, "min_df": md}
    for g in LR_GRID for ng in NGRAMS for md in MIN_DFS
]

# ---------------- NLTK stopwords only (Mu & Aletras-style preprocessing) ----------------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"http\S+", " url ", t)
    t = re.sub(r"@\w+", " usr ", t)
    toks = re.findall(r"[a-z0-9_]+", t)
    return " ".join([w for w in toks if w not in STOPWORDS])

# ---------------- cluster map loader (token -> int cluster id) ----------------
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
            token = parts[0]
            try:
                cid = int(parts[-1])  # last token must be an integer cluster id
            except ValueError:
                raise ValueError(
                    "Cluster file must be 'token ... <int_cluster_id>' per line."
                )
            cmap[token] = cid
            n_lines += 1
            if n_lines >= 200000:  # safety cap
                break
    if not cmap:
        raise ValueError("No cluster entries loaded. Check the cluster file format.")
    nclus = max(cmap.values()) + 1
    log.info(f"Loaded cluster map: {len(cmap)} tokens -> {nclus} clusters")
    return cmap, nclus

def compute_topic_feats(texts, token_counts, cmap, nclus):
    X = np.zeros((len(texts), nclus), dtype=float)
    for i, doc in enumerate(texts):
        nt = float(token_counts[i]) if token_counts[i] else 0.0
        if nt <= 0: continue
        for w in doc.split():
            cid = cmap.get(w)
            if cid is not None:
                X[i, cid] += 1.0
        if X[i].sum() > 0:
            X[i, :] /= nt  # normalize by doc length
    return X  # dense (n_docs, nclus)

# ---------------- helpers ----------------
def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return p, r, f1

def build_logistic(penalty, C, l1_ratio, random_state):
    # same solvers/limits as your lr.py
    if penalty == "elasticnet":
        return LogisticRegression(
            penalty=penalty, C=C, solver="saga", l1_ratio=l1_ratio,
            max_iter=20000, tol=1e-4, random_state=random_state,
            class_weight="balanced"
        )
    elif penalty in ("l1", "l2"):
        return LogisticRegression(
            penalty=penalty, C=C, solver="liblinear",
            max_iter=10000, tol=1e-4, random_state=random_state,
            class_weight="balanced"
        )
    else:  # default to l2
        return LogisticRegression(
            penalty="l2", C=C, solver="saga",
            max_iter=10000, tol=1e-4, random_state=random_state,
            class_weight="balanced"
        )

def build_numeric_matrix_fit(df_sub: pd.DataFrame, cols: list):
    X = df_sub[cols].copy()
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X.values)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_imp)
    return X_scaled, imp, sc

def numeric_transform(df_sub: pd.DataFrame, cols: list, imp: SimpleImputer, sc: StandardScaler):
    X = df_sub[cols].copy()
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X_imp = imp.transform(X.values)
    return sc.transform(X_imp)

def lr_coeff_importance(model, feature_names):
    coef = model.coef_
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    elif coef.ndim == 2:
        # multiclass: mean abs across classes (not used here, binary)
        coef = np.mean(np.abs(coef), axis=0)
    imp = np.abs(coef)
    rows = list(zip(feature_names, imp))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="5-fold CV Logistic Regression: Users / Message / Fusion + importance & ablation")
    ap.add_argument("--data", default="preprocessed_dataset_all2.csv", help="Input CSV with all features + aggr_text")
    ap.add_argument("--cluster", default="glove-200.txt", help="Cluster map file (token ... <int cluster id>)")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--topk", type=int, default=100, help="Top-K features to save per fold")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "diagnostics"), exist_ok=True)

    df = pd.read_csv(args.data)

    # Checks
    missing_users = [c for c in USER_COLS if c not in df.columns]
    if missing_users:
        raise ValueError(f"Missing user feature columns in CSV: {missing_users}")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    if "aggr_text" not in df.columns:
        raise ValueError("CSV must contain 'aggr_text' for message features.")

    # Encode label
    df["label"] = df["label"].map({"reliable":0, "unreliable":1}).astype(int)

    # Message numeric columns = everything NOT in user list and not excluded
    msg_numeric_cols = [c for c in df.columns if c not in USER_COLS and c not in EXCLUDE_ALWAYS]
    log.info(f"Using {len(USER_COLS)} user columns, {len(msg_numeric_cols)} message numeric columns.")

    # Preprocess text and token counts
    df["aggr_text_processed"] = df["aggr_text"].astype(str).map(preprocess)
    df["token_count"] = df["aggr_text_processed"].map(lambda t: len(t.split()))

    # Load cluster map
    cmap, nclus = load_cluster_map(args.cluster)

    # Prepare CV
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    per_fold = []
    agg_user_imp, agg_msg_imp, agg_fus_imp = {}, {}, {}
    all_user_lofo, all_msg_block_abl = [], []

    fold_idx = 0
    for trainval_idx, test_idx in skf.split(df, df["label"]):
        fold_idx += 1
        df_trainval = df.iloc[trainval_idx].copy()
        df_test = df.iloc[test_idx].copy()

        # inner dev split (12.5% of trainval ~ 10% of full)
        train_df, dev_df = train_test_split(
            df_trainval, test_size=0.125, stratify=df_trainval["label"], random_state=args.seed
        )

        y_tr, y_dv, y_te = train_df["label"].values, dev_df["label"].values, df_test["label"].values

        # ======== USERS model ========
        Xu_tr, imp_u, sc_u = build_numeric_matrix_fit(train_df, USER_COLS)
        Xu_dv = numeric_transform(dev_df, USER_COLS, imp_u, sc_u)
        Xu_te = numeric_transform(df_test, USER_COLS, imp_u, sc_u)

        best_users = {"params": None, "f1": -1}
        for P in LR_GRID:
            clf = build_logistic(P["penalty"], P["C"], P["l1_ratio"], random_state=args.seed)
            with warnings.catch_warnings(record=True):
                clf.fit(Xu_tr, y_tr)
            f1 = macro_scores(y_dv, clf.predict(Xu_dv))[2]
            if f1 > best_users["f1"]:
                best_users = {"params": P, "f1": f1}

        # final train on train+dev
        Xu_tr_plus = np.vstack([Xu_tr, Xu_dv])
        y_tr_plus  = np.concatenate([y_tr, y_dv])
        clf_u = build_logistic(best_users["params"]["penalty"], best_users["params"]["C"], best_users["params"]["l1_ratio"], random_state=args.seed)
        clf_u.fit(Xu_tr_plus, y_tr_plus)
        pred_u = clf_u.predict(Xu_te)
        Pu, Ru, F1u = macro_scores(y_te, pred_u)
        print(f"\n[FOLD {fold_idx}][Users]  P={Pu:.4f} R={Ru:.4f} F1={F1u:.4f}")
        print(classification_report(y_te, pred_u, digits=4, zero_division=0))

        joblib.dump(clf_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_lr_users.joblib"))
        joblib.dump(imp_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_users.joblib"))
        joblib.dump(sc_u,  os.path.join(args.outdir, "models", f"fold{fold_idx}_scaler_users.joblib"))

        # importance (Users)
        u_imp = lr_coeff_importance(clf_u, USER_COLS)
        pd.DataFrame(u_imp, columns=["feature","abs_coef"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_users.csv"), index=False
        )
        for name, val in u_imp:
            agg_user_imp[name] = agg_user_imp.get(name, []) + [float(val)]

        # ======== MESSAGE model ========
        # message numeric (LIWC/meta) fit -> scale
        Xm_tr, imp_m, sc_m = build_numeric_matrix_fit(train_df, msg_numeric_cols)
        Xm_dv = numeric_transform(dev_df, msg_numeric_cols, imp_m, sc_m)
        Xm_te = numeric_transform(df_test, msg_numeric_cols, imp_m, sc_m)

        best_msg = {"ngr": None, "min_df": None, "params": None, "f1": -1,
                    "vect": None, "X_tr_msg_dev": None, "X_dv_msg_dev": None,
                    "sc_topics_dev": None}  # keep dev matrices for Fusion tuning

        for P in LR_TEXT_GRID:
            # BoW
            vect_try = TfidfVectorizer(
                ngram_range=P["ngram_range"], max_features=20000,
                min_df=P["min_df"], max_df=0.4, lowercase=False
            )
            Xb_tr = vect_try.fit_transform(train_df["aggr_text_processed"].values)
            Xb_dv = vect_try.transform(dev_df["aggr_text_processed"].values)

            # Topics (fit scaler on train topics)
            Xt_tr_raw = compute_topic_feats(
                train_df["aggr_text_processed"].values,
                train_df["token_count"].values, cmap, nclus
            )
            Xt_dv_raw = compute_topic_feats(
                dev_df["aggr_text_processed"].values,
                dev_df["token_count"].values, cmap, nclus
            )
            sc_topics = StandardScaler()
            Xt_tr = sc_topics.fit_transform(Xt_tr_raw)
            Xt_dv = sc_topics.transform(Xt_dv_raw)

            # concat: [BoW | Topics | MsgNumeric]
            X_tr_msg = sparse.hstack([Xb_tr, sparse.csr_matrix(Xt_tr), sparse.csr_matrix(Xm_tr)]).tocsr()
            X_dv_msg = sparse.hstack([Xb_dv, sparse.csr_matrix(Xt_dv), sparse.csr_matrix(Xm_dv)]).tocsr()

            clf = build_logistic(P["penalty"], P["C"], P["l1_ratio"], random_state=args.seed)
            with warnings.catch_warnings(record=True):
                clf.fit(X_tr_msg, y_tr)
            f1 = macro_scores(y_dv, clf.predict(X_dv_msg))[2]
            if f1 > best_msg["f1"]:
                best_msg = {"ngr": P["ngram_range"], "min_df": P["min_df"], "params": {k: P[k] for k in ["penalty","C","l1_ratio"]},
                            "f1": f1, "vect": vect_try, "X_tr_msg_dev": X_tr_msg, "X_dv_msg_dev": X_dv_msg,
                            "sc_topics_dev": sc_topics}

        # rebuild on train+dev with best settings
        vect_m = best_msg["vect"]
        Xb_trd = vect_m.fit_transform(pd.concat([train_df["aggr_text_processed"], dev_df["aggr_text_processed"]]).values)
        Xb_te  = vect_m.transform(df_test["aggr_text_processed"].values)

        # topics on train+dev and test with fresh scaler
        Xt_trd_raw = compute_topic_feats(
            pd.concat([train_df["aggr_text_processed"], dev_df["aggr_text_processed"]]).values,
            pd.concat([train_df["token_count"], dev_df["token_count"]]).values, cmap, nclus
        )
        Xt_te_raw = compute_topic_feats(
            df_test["aggr_text_processed"].values, df_test["token_count"].values, cmap, nclus
        )
        sc_topics_final = StandardScaler()
        Xt_trd = sc_topics_final.fit_transform(Xt_trd_raw)
        Xt_te  = sc_topics_final.transform(Xt_te_raw)

        # message numeric on train+dev with fresh imputer+scaler
        Xm_trd, imp_m_final, sc_m_final = build_numeric_matrix_fit(
            pd.concat([train_df, dev_df], axis=0), msg_numeric_cols
        )
        # test transform
        Xm_te_final = numeric_transform(df_test, msg_numeric_cols, imp_m_final, sc_m_final)

        X_tr_msg_final = sparse.hstack([Xb_trd, sparse.csr_matrix(Xt_trd), sparse.csr_matrix(Xm_trd)]).tocsr()
        X_te_msg_final = sparse.hstack([Xb_te,  sparse.csr_matrix(Xt_te),  sparse.csr_matrix(Xm_te_final)]).tocsr()

        Pm_params = best_msg["params"]
        clf_m = build_logistic(Pm_params["penalty"], Pm_params["C"], Pm_params["l1_ratio"], random_state=args.seed)
        clf_m.fit(X_tr_msg_final, np.concatenate([y_tr, y_dv]))
        pred_m = clf_m.predict(X_te_msg_final)
        Pm, Rm, F1m = macro_scores(y_te, pred_m)
        print(f"\n[FOLD {fold_idx}][Message] P={Pm:.4f} R={Rm:.4f} F1={F1m:.4f}")
        print(classification_report(y_te, pred_m, digits=4, zero_division=0))

        joblib.dump(clf_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_lr_message.joblib"))
        joblib.dump(vect_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_tfidf_message.joblib"))
        joblib.dump(imp_m_final, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_message.joblib"))
        joblib.dump(sc_m_final,  os.path.join(args.outdir, "models", f"fold{fold_idx}_scaler_mnum_message.joblib"))
        joblib.dump(sc_topics_final, os.path.join(args.outdir, "models", f"fold{fold_idx}_scaler_topics_message.joblib"))

        # importance (Message)
        bow_names   = [f"bow:{w}" for w in vect_m.get_feature_names_out()]
        topic_names = [f"topic:{i}" for i in range(nclus)]
        mnum_names  = [f"mnum:{c}" for c in msg_numeric_cols]
        msg_names   = bow_names + topic_names + mnum_names
        m_imp = lr_coeff_importance(clf_m, msg_names)
        pd.DataFrame(m_imp, columns=["feature","abs_coef"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_message.csv"), index=False
        )
        for name, val in m_imp:
            agg_msg_imp[name] = agg_msg_imp.get(name, []) + [float(val)]

        # ======== FUSION model ========
        # dev matrices from message + users (for tuning only)
        X_tr_fus_dev = sparse.hstack([best_msg["X_tr_msg_dev"], sparse.csr_matrix(Xu_tr)]).tocsr()
        X_dv_fus_dev = sparse.hstack([best_msg["X_dv_msg_dev"], sparse.csr_matrix(Xu_dv)]).tocsr()

        best_f = {"params": None, "f1": -1}
        for P in LR_GRID:
            clf = build_logistic(P["penalty"], P["C"], P["l1_ratio"], random_state=args.seed)
            with warnings.catch_warnings(record=True):
                clf.fit(X_tr_fus_dev, y_tr)
            f1 = macro_scores(y_dv, clf.predict(X_dv_fus_dev))[2]
            if f1 > best_f["f1"]:
                best_f = {"params": P, "f1": f1}

        # final fusion = final message (train+dev) + users (train+dev)
        Xu_tr_plus_final = Xu_tr_plus  # already vstacked
        X_tr_fus_final = sparse.hstack([X_tr_msg_final, sparse.csr_matrix(Xu_tr_plus_final)]).tocsr()
        X_te_fus_final = sparse.hstack([X_te_msg_final,  sparse.csr_matrix(Xu_te)]).tocsr()

        Pf_params = best_f["params"]
        clf_f = build_logistic(Pf_params["penalty"], Pf_params["C"], Pf_params["l1_ratio"], random_state=args.seed)
        clf_f.fit(X_tr_fus_final, y_tr_plus)
        pred_f = clf_f.predict(X_te_fus_final)
        Pf, Rf, F1f = macro_scores(y_te, pred_f)
        print(f"\n[FOLD {fold_idx}][Fusion]  P={Pf:.4f} R={Rf:.4f} F1={F1f:.4f}")

        joblib.dump(clf_f, os.path.join(args.outdir, "models", f"fold{fold_idx}_lr_fusion.joblib"))
        # Reuse/snapshot the TF-IDF used in message/fusion
        joblib.dump(vect_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_tfidf_fusion.joblib"))

        # importance (Fusion)
        fus_names = msg_names + [f"user:{c}" for c in USER_COLS]
        f_imp = lr_coeff_importance(clf_f, fus_names)
        pd.DataFrame(f_imp, columns=["feature","abs_coef"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_fusion.csv"), index=False
        )
        for name, val in f_imp:
            agg_fus_imp[name] = agg_fus_imp.get(name, []) + [float(val)]

        # record fold results
        per_fold += [
            {"fold": fold_idx, "family": "Users",   "P_macro": Pu, "R_macro": Ru, "F1_macro": F1u},
            {"fold": fold_idx, "family": "Message", "P_macro": Pm, "R_macro": Rm, "F1_macro": F1m},
            {"fold": fold_idx, "family": "Fusion",  "P_macro": Pf, "R_macro": Rf, "F1_macro": F1f},
        ]

        # ======== Ablations ========
        # Users: LOFO (drop one user feature; re-fit imputer+scaler on train+dev for fairness)
        u_base_f1 = F1u
        for j, col in enumerate(USER_COLS):
            keep_cols = [c for c in USER_COLS if c != col]
            Xu_trd_lofo, imp_u_lofo, sc_u_lofo = build_numeric_matrix_fit(pd.concat([train_df, dev_df], axis=0), keep_cols)
            Xu_te_lofo = numeric_transform(df_test, keep_cols, imp_u_lofo, sc_u_lofo)
            clf_lofo = build_logistic(best_users["params"]["penalty"], best_users["params"]["C"], best_users["params"]["l1_ratio"], random_state=args.seed)
            clf_lofo.fit(Xu_trd_lofo, y_tr_plus)
            f1_lofo = macro_scores(y_te, clf_lofo.predict(Xu_te_lofo))[2]
            all_user_lofo.append({"fold": fold_idx, "feature": col, "delta_F1": f1_lofo - u_base_f1})

        # Message: block ablations (drop BoW / drop Topics / drop MsgNumeric)
        m_base_f1 = F1m
        # no_bow
        Xtr_no_bow = sparse.hstack([sparse.csr_matrix(Xt_trd), sparse.csr_matrix(Xm_trd)]).tocsr()
        Xte_no_bow = sparse.hstack([sparse.csr_matrix(Xt_te),  sparse.csr_matrix(Xm_te_final)]).tocsr()
        clf_nb = build_logistic(Pm_params["penalty"], Pm_params["C"], Pm_params["l1_ratio"], random_state=args.seed)
        clf_nb.fit(Xtr_no_bow, y_tr_plus)
        f1_nb = macro_scores(y_te, clf_nb.predict(Xte_no_bow))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_bow", "delta_F1": f1_nb - m_base_f1})

        # no_topics
        Xtr_no_topics = sparse.hstack([Xb_trd, sparse.csr_matrix(Xm_trd)]).tocsr()
        Xte_no_topics = sparse.hstack([Xb_te,  sparse.csr_matrix(Xm_te_final)]).tocsr()
        clf_nt = build_logistic(Pm_params["penalty"], Pm_params["C"], Pm_params["l1_ratio"], random_state=args.seed)
        clf_nt.fit(Xtr_no_topics, y_tr_plus)
        f1_nt = macro_scores(y_te, clf_nt.predict(Xte_no_topics))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_topics", "delta_F1": f1_nt - m_base_f1})

        # no_msg_numeric
        Xtr_no_mnum = sparse.hstack([Xb_trd, sparse.csr_matrix(Xt_trd)]).tocsr()
        Xte_no_mnum = sparse.hstack([Xb_te,  sparse.csr_matrix(Xt_te)]).tocsr()
        clf_nm = build_logistic(Pm_params["penalty"], Pm_params["C"], Pm_params["l1_ratio"], random_state=args.seed)
        clf_nm.fit(Xtr_no_mnum, y_tr_plus)
        f1_nm = macro_scores(y_te, clf_nm.predict(Xte_no_mnum))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_msg_numeric", "delta_F1": f1_nm - m_base_f1})

    # save per-fold and summary
    per_fold_df = pd.DataFrame(per_fold)
    per_fold_path = os.path.join(args.outdir, "lr_results_cv5_per_fold.csv")
    per_fold_df.to_csv(per_fold_path, index=False)

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
    summary_path = os.path.join(args.outdir, "lr_results_cv5_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== 5-fold Stratified CV (macro, Logistic Regression) ===")
    print(summary_df.to_string(index=False))

    # ---------- FINAL: aggregate importances across folds (mean |coef|) ----------
    def aggregate_importances(dct):
        rows = []
        for k, vals in dct.items():
            rows.append({
                "feature": k,
                "abs_coef_mean": float(np.mean(vals)),
                "abs_coef_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            })
        out = pd.DataFrame(rows).sort_values("abs_coef_mean", ascending=False)
        return out

    # Compute and save to diagnostics (as before) AND to top-level with required names
    importance_users_agg_df = aggregate_importances(agg_user_imp)
    importance_message_agg_df = aggregate_importances(agg_msg_imp)
    importance_fusion_agg_df  = aggregate_importances(agg_fus_imp)

    # keep diagnostics copies (optional)
    importance_users_agg_df.to_csv(os.path.join(args.outdir, "diagnostics", "importance_users_agg.csv"), index=False)
    importance_message_agg_df.to_csv(os.path.join(args.outdir, "diagnostics", "importance_message_agg.csv"), index=False)
    importance_fusion_agg_df.to_csv(os.path.join(args.outdir, "diagnostics", "importance_fusion_agg.csv"), index=False)

    # required final outputs (top-level)
    importance_users_agg_df.to_csv(os.path.join(args.outdir, "importance_users_agg.csv"), index=False)
    importance_message_agg_df.to_csv(os.path.join(args.outdir, "importance_message_agg.csv"), index=False)
    importance_fusion_agg_df.to_csv(os.path.join(args.outdir, "importance_fusion_agg.csv"), index=False)

    # ---------- FINAL: save ablation results ----------
    ablation_users_lofo_df = pd.DataFrame(all_user_lofo)
    ablation_message_blocks_df = pd.DataFrame(all_msg_block_abl)

    # keep diagnostics copies (optional)
    ablation_users_lofo_df.to_csv(os.path.join(args.outdir, "diagnostics", "ablation_users_lofo.csv"), index=False)
    ablation_message_blocks_df.to_csv(os.path.join(args.outdir, "diagnostics", "ablation_message_blocks.csv"), index=False)

    # required final outputs (top-level)
    ablation_users_lofo_df.to_csv(os.path.join(args.outdir, "ablation_users_lofo.csv"), index=False)
    ablation_message_blocks_df.to_csv(os.path.join(args.outdir, "ablation_message_blocks.csv"), index=False)

    # ---------- config snapshot ----------
    with open(os.path.join(args.outdir, "lr_feature_config.json"), "w", encoding="utf8") as f:
        json.dump({
            "user_cols": USER_COLS,
            "message_numeric_cols": msg_numeric_cols,
            "kfolds": args.kfolds,
            "seed": args.seed,
            "cluster_file": args.cluster,
            "ngrams": NGRAMS,
            "min_df": MIN_DFS,
            "Cs": Cs,
            "l1_ratios": L1_RATIOS
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
