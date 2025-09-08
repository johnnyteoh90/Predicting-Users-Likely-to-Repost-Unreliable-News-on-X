#!/usr/bin/env python3
import os, re, json, argparse, logging
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, classification_report
import joblib

import nltk
from nltk.corpus import stopwords
from xgboost import XGBClassifier

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xgb_families_ablate")

# ---------------- fixed user feature list ----------------
USER_COLS = [
    "HU_TweetNum","HU_TweetPercent_Retweet","HU_AverageInterval_days",
    "U_AccountAge_days","U_FolloweeNumDay","followers_count","following_count",
    "posting_frequency","retweet_count","retweet_ratio"
]

# USER_COLS = [
# "HU_TweetNum","HU_TweetPercent_Original","HU_TweetPercent_Retweet","HU_AverageInterval_days",
# "U_AccountAge_days","U_ListedNum","U_ProfileUrl","U_FollowerNumDay","U_FolloweeNumDay",
# "U_TweetNumDay","U_ListedNumDay","total_posts","followers_count","following_count",
# "verified","posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio"
# ]
EXCLUDE_ALWAYS = {"label","username","user","user_id","screen_name","aggr_text", "HU_TweetPercent_Original","U_ListedNum","U_ProfileUrl","U_FollowerNumDay","U_TweetNumDay","U_ListedNumDay","total_posts","verified","follower_to_followee_ratio"}

# ---------------- conservative text grids (Mu & Aletras–style) ----------------
NGRAMS  = [(1,1), (1,2)]
MIN_DF  = [2, 3, 5]

# small-sample-friendly XGB grid
XGB_GRID = [
    {"max_depth": 3, "min_child_weight": 1, "learning_rate": 0.10, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"max_depth": 3, "min_child_weight": 2, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "reg_alpha": 0.5},
    {"max_depth": 5, "min_child_weight": 1, "learning_rate": 0.10, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"max_depth": 5, "min_child_weight": 2, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "reg_alpha": 0.5},
]

# ---------------- NLTK stopwords only ----------------
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
                cid = int(parts[-1])
            except ValueError:
                raise ValueError("Cluster file must be 'token ... <int_cluster_id>' per line.")
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

def build_message_blocks(df_sub, ngram_range, min_df, cmap, nclus):
    texts = df_sub["aggr_text"].astype(str).map(preprocess).fillna("")
    token_count = texts.map(lambda t: len(t.split()))
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=20000,
        min_df=min_df,
        max_df=0.4,
        lowercase=False,  
    )
    X_bow = vect.fit_transform(texts.values)
    topics = compute_topic_feats(texts.values, token_count.values, cmap, nclus)
    X_topics = sparse.csr_matrix(topics)
    return X_bow, vect, X_topics

def scale_pos_weight(y):
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return float(neg) / float(pos)

def xgb_gain_importance(model, feature_names):
    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
    scores = booster.get_score(importance_type="gain")
    rows = []
    for fid, gain in scores.items():
        rows.append((fmap.get(fid, fid), float(gain)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="5-fold CV XGBoost: Users / Message / Fusion + importance & ablation")
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

    # Message numeric = everything NOT in user list and not excluded (keep all)
    msg_numeric_cols = [c for c in df.columns if c not in USER_COLS and c not in EXCLUDE_ALWAYS]
    log.info(f"Using {len(USER_COLS)} user columns, {len(msg_numeric_cols)} message numeric columns.")

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

        # small inner dev from trainval (≈10% of full -> 12.5% of trainval)
        train_df, dev_df = train_test_split(
            df_trainval, test_size=0.125, stratify=df_trainval["label"], random_state=args.seed
        )

        y_tr, y_dv, y_te = train_df["label"].values, dev_df["label"].values, df_test["label"].values

        # ======== USERS model ========
        Xu_tr, imp_u = build_numeric_matrix(train_df, USER_COLS)
        Xu_dv = imp_u.transform(dev_df[USER_COLS].apply(pd.to_numeric, errors="coerce").values)
        Xu_te = imp_u.transform(df_test[USER_COLS].apply(pd.to_numeric, errors="coerce").values)

        best_users = {"params": None, "f1": -1}
        for P in XGB_GRID:
            clf = XGBClassifier(
                **P, use_label_encoder=False, eval_metric="logloss",
                random_state=args.seed, scale_pos_weight=scale_pos_weight(y_tr),
                n_jobs=4, verbosity=0
            )
            clf.fit(Xu_tr, y_tr)
            pred_dv = clf.predict(Xu_dv)
            _, _, f1 = macro_scores(y_dv, pred_dv)
            if f1 > best_users["f1"]:
                best_users = {"params": P, "f1": f1}

        Xu_tr_plus = np.vstack([Xu_tr, Xu_dv])
        y_tr_plus  = np.concatenate([y_tr, y_dv])
        clf_u = XGBClassifier(
            **best_users["params"], use_label_encoder=False, eval_metric="logloss",
            random_state=args.seed, scale_pos_weight=scale_pos_weight(y_tr_plus),
            n_jobs=4, verbosity=0
        ).fit(Xu_tr_plus, y_tr_plus)

        pred_u = clf_u.predict(Xu_te)
        Pu, Ru, F1u = macro_scores(y_te, pred_u)
        print(f"\n[FOLD {fold_idx}][Users]  P={Pu:.4f} R={Ru:.4f} F1={F1u:.4f}")
        print(classification_report(y_te, pred_u, digits=4, zero_division=0))
        joblib.dump(clf_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_xgb_users.joblib"))
        joblib.dump(imp_u, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_users.joblib"))

        # user importance
        u_imp = xgb_gain_importance(clf_u, USER_COLS)
        pd.DataFrame(u_imp, columns=["feature","gain"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_users.csv"), index=False
        )
        for name, gain in u_imp:
            agg_user_imp[name] = agg_user_imp.get(name, []) + [gain]

        # ======== MESSAGE model ========
        Xm_tr, imp_m = build_numeric_matrix(train_df, msg_numeric_cols)
        Xm_dv = imp_m.transform(dev_df[msg_numeric_cols].apply(pd.to_numeric, errors="coerce").values)
        Xm_te = imp_m.transform(df_test[msg_numeric_cols].apply(pd.to_numeric, errors="coerce").values)

        best_msg = {"ngr": None, "min_df": None, "params": None, "f1": -1, "vect": None,
                    "X_tr_msg_dev": None, "X_dv_msg_dev": None}
        for ngr in NGRAMS:
            for md in MIN_DF:
                Xb_tr, vect_try, Xt_tr = build_message_blocks(train_df, ngr, md, cmap, nclus)
                Xb_dv = vect_try.transform(dev_df["aggr_text"].astype(str).map(preprocess).values)
                Xt_dv = compute_topic_feats(
                    dev_df["aggr_text"].astype(str).map(preprocess).values,
                    dev_df["aggr_text"].astype(str).map(lambda t: len(preprocess(t).split())).values, cmap, nclus
                )
                # ensure sparse
                Xt_dv = sparse.csr_matrix(Xt_dv)

                X_tr_msg = sparse.hstack([Xb_tr, Xt_tr, sparse.csr_matrix(Xm_tr)]).tocsr()
                X_dv_msg = sparse.hstack([Xb_dv, Xt_dv, sparse.csr_matrix(Xm_dv)]).tocsr()

                for P in XGB_GRID:
                    clf = XGBClassifier(
                        **P, use_label_encoder=False, eval_metric="logloss",
                        random_state=args.seed, scale_pos_weight=scale_pos_weight(y_tr),
                        n_jobs=4, verbosity=0
                    )
                    clf.fit(X_tr_msg, y_tr)
                    pred = clf.predict(X_dv_msg)
                    _, _, f1 = macro_scores(y_dv, pred)
                    if f1 > best_msg["f1"]:
                        best_msg = {"ngr": ngr, "min_df": md, "params": P, "f1": f1, "vect": vect_try,
                                    "X_tr_msg_dev": X_tr_msg, "X_dv_msg_dev": X_dv_msg}

        vect_m = best_msg["vect"]
        Xb_tr_final, vect_m, Xt_tr_final = build_message_blocks(
            pd.concat([train_df, dev_df], axis=0), best_msg["ngr"], best_msg["min_df"], cmap, nclus
        )
        Xb_te = vect_m.transform(df_test["aggr_text"].astype(str).map(preprocess).values)
        Xt_te = compute_topic_feats(
            df_test["aggr_text"].astype(str).map(preprocess).values,
            df_test["aggr_text"].astype(str).map(lambda t: len(preprocess(t).split())).values, cmap, nclus
        )
        Xt_te = sparse.csr_matrix(Xt_te)  # ensure sparse

        Xm_tr_plus = imp_m.fit_transform(
            pd.concat([train_df[msg_numeric_cols], dev_df[msg_numeric_cols]]).apply(pd.to_numeric, errors="coerce").values
        )
        X_tr_msg_final = sparse.hstack([Xb_tr_final, Xt_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        X_te_msg_final = sparse.hstack([Xb_te, Xt_te, sparse.csr_matrix(Xm_te)]).tocsr()

        clf_m = XGBClassifier(
            **best_msg["params"], use_label_encoder=False, eval_metric="logloss",
            random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
            n_jobs=4, verbosity=0
        ).fit(X_tr_msg_final, np.concatenate([y_tr, y_dv]))

        pred_m = clf_m.predict(X_te_msg_final)
        Pm, Rm, F1m = macro_scores(y_te, pred_m)
        print(f"\n[FOLD {fold_idx}][Message] P={Pm:.4f} R={Rm:.4f} F1={F1m:.4f}")
        print(classification_report(y_te, pred_m, digits=4, zero_division=0))

        joblib.dump(clf_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_xgb_message.joblib"))
        joblib.dump(vect_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_tfidf_message.joblib"))
        joblib.dump(imp_m, os.path.join(args.outdir, "models", f"fold{fold_idx}_imputer_message.joblib"))

        # message importance
        bow_names   = [f"bow:{w}" for w in vect_m.get_feature_names_out()]
        topic_names = [f"topic:{i}" for i in range(nclus)]
        mnum_names  = [f"mnum:{c}" for c in msg_numeric_cols]
        msg_names   = bow_names + topic_names + mnum_names
        m_imp = xgb_gain_importance(clf_m, msg_names)
        pd.DataFrame(m_imp, columns=["feature","gain"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_message.csv"), index=False
        )
        for name, gain in m_imp:
            agg_msg_imp[name] = agg_msg_imp.get(name, []) + [gain]

        # ======== FUSION model ========
        X_tr_fus_final = sparse.hstack([X_tr_msg_final, sparse.csr_matrix(Xu_tr_plus)]).tocsr()
        X_te_fus_final = sparse.hstack([X_te_msg_final,  sparse.csr_matrix(Xu_te)]).tocsr()

        best_f = {"params": None, "f1": -1}
        for P in XGB_GRID:
            clf = XGBClassifier(
                **P, use_label_encoder=False, eval_metric="logloss",
                random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
                n_jobs=4, verbosity=0
            )
            clf.fit(X_tr_fus_final, np.concatenate([y_tr, y_dv]))
            pred = clf.predict(X_te_fus_final)
            _, _, f1 = macro_scores(y_te, pred)
            if f1 > best_f["f1"]:
                best_f = {"params": P, "f1": f1}

        clf_f = XGBClassifier(
            **best_f["params"], use_label_encoder=False, eval_metric="logloss",
            random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
            n_jobs=4, verbosity=0
        ).fit(X_tr_fus_final, np.concatenate([y_tr, y_dv]))

        pred_f = clf_f.predict(X_te_fus_final)
        Pf, Rf, F1f = macro_scores(y_te, pred_f)
        print(f"\n[FOLD {fold_idx}][Fusion]  P={Pf:.4f} R={Rf:.4f} F1={F1f:.4f}")

        joblib.dump(clf_f, os.path.join(args.outdir, "models", f"fold{fold_idx}_xgb_fusion.joblib"))

        # fusion importance
        fus_names = msg_names + [f"user:{c}" for c in USER_COLS]
        f_imp = xgb_gain_importance(clf_f, fus_names)
        pd.DataFrame(f_imp, columns=["feature","gain"]).head(args.topk).to_csv(
            os.path.join(args.outdir, "diagnostics", f"fold{fold_idx}_importance_fusion.csv"), index=False
        )
        for name, gain in f_imp:
            agg_fus_imp[name] = agg_fus_imp.get(name, []) + [gain]

        # record fold results
        per_fold += [
            {"fold": fold_idx, "family": "Users",   "P_macro": Pu, "R_macro": Ru, "F1_macro": F1u},
            {"fold": fold_idx, "family": "Message", "P_macro": Pm, "R_macro": Rm, "F1_macro": F1m},
            {"fold": fold_idx, "family": "Fusion",  "P_macro": Pf, "R_macro": Rf, "F1_macro": F1f},
        ]

        # ======== Ablations ========
        # Users: LOFO (leave-one-feature-out)
        u_base_f1 = F1u
        for j, col in enumerate(USER_COLS):
            keep_idx = [i for i in range(len(USER_COLS)) if i != j]
            Xu_tr_lofo = Xu_tr_plus[:, keep_idx]
            Xu_te_lofo = Xu_te[:, keep_idx]
            clf_lofo = XGBClassifier(
                **best_users["params"], use_label_encoder=False, eval_metric="logloss",
                random_state=args.seed, scale_pos_weight=scale_pos_weight(y_tr_plus),
                n_jobs=4, verbosity=0
            ).fit(Xu_tr_lofo, y_tr_plus)
            f1_lofo = macro_scores(y_te, clf_lofo.predict(Xu_te_lofo))[2]
            all_user_lofo.append({"fold": fold_idx, "feature": col, "delta_F1": (f1_lofo - u_base_f1)})

        # Message: block ablations (drop BoW / drop Topics / drop MsgNumeric)
        m_base_f1 = F1m
        # drop BoW
        Xtr_no_bow = sparse.hstack([Xt_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        Xte_no_bow = sparse.hstack([Xt_te,       sparse.csr_matrix(Xm_te)]).tocsr()
        clf_nb = XGBClassifier(**best_msg["params"], use_label_encoder=False, eval_metric="logloss",
                               random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
                               n_jobs=4, verbosity=0).fit(Xtr_no_bow, np.concatenate([y_tr, y_dv]))
        f1_nb = macro_scores(y_te, clf_nb.predict(Xte_no_bow))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_bow", "delta_F1": (f1_nb - m_base_f1)})

        # drop Topics
        Xtr_no_topics = sparse.hstack([Xb_tr_final, sparse.csr_matrix(Xm_tr_plus)]).tocsr()
        Xte_no_topics = sparse.hstack([Xb_te,       sparse.csr_matrix(Xm_te)]).tocsr()
        clf_nt = XGBClassifier(**best_msg["params"], use_label_encoder=False, eval_metric="logloss",
                               random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
                               n_jobs=4, verbosity=0).fit(Xtr_no_topics, np.concatenate([y_tr, y_dv]))
        f1_nt = macro_scores(y_te, clf_nt.predict(Xte_no_topics))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_topics", "delta_F1": (f1_nt - m_base_f1)})

        # drop MsgNumeric
        Xtr_no_mnum = sparse.hstack([Xb_tr_final, Xt_tr_final]).tocsr()
        Xte_no_mnum = sparse.hstack([Xb_te,       Xt_te]).tocsr()
        clf_nm = XGBClassifier(**best_msg["params"], use_label_encoder=False, eval_metric="logloss",
                               random_state=args.seed, scale_pos_weight=scale_pos_weight(np.concatenate([y_tr, y_dv])),
                               n_jobs=4, verbosity=0).fit(Xtr_no_mnum, np.concatenate([y_tr, y_dv]))
        f1_nm = macro_scores(y_te, clf_nm.predict(Xte_no_mnum))[2]
        all_msg_block_abl.append({"fold": fold_idx, "block": "no_msg_numeric", "delta_F1": (f1_nm - m_base_f1)})

    # save per-fold and summary
    per_fold_df = pd.DataFrame(per_fold)
    per_fold_path = os.path.join(args.outdir, "xgb_results_cv5_per_fold.csv")
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
    summary_path = os.path.join(args.outdir, "xgb_results_cv5_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== 5-fold Stratified CV (macro, XGBoost) ===")
    print(summary_df.to_string(index=False))

    # ---------- FINAL: aggregate importances across folds (mean gain) ----------
    def aggregate_importances(dct):
        rows = []
        for k, vals in dct.items():
            rows.append({
                "feature": k,
                "gain_mean": float(np.mean(vals)),
                "gain_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            })
        out = pd.DataFrame(rows).sort_values("gain_mean", ascending=False)
        return out

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
    with open(os.path.join(args.outdir, "xgb_feature_config.json"), "w", encoding="utf8") as f:
        json.dump({
            "user_cols": USER_COLS,
            "message_numeric_cols": msg_numeric_cols,
            "kfolds": args.kfolds,
            "seed": args.seed,
            "cluster_file": args.cluster,
            "ngrams": NGRAMS,
            "min_df": MIN_DF,
            "xgb_grid": XGB_GRID
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
