# README

This repository accompanies the study on *Predicting Users Likely to Repost Unreliable News on X (Formerly Twitter)*. It provides end-to-end pipelines to (a) collect user timelines, (b) construct two datasets—**Type‑1** (posts up to each user’s first news repost; early‑detection) and **Type‑2** (full tweet history excluding retweets), (c) extract rich features from timelines and profiles (e.g., bag‑of‑words, topics, psycholinguistic metrics; account age, activity, network reach), and (d) train/evaluate both classical ML (SVM, LR, XGBoost) and neural baselines (MLP, BiGru, Bert), including multimodal fusion of user + text signals. We compare three feature families—**User‑only**, **Message‑only (text)**, and **Fusion (Message + Users)**, across early vs. full‑history settings. For the experiments, user‑centric features are highly predictive, and an RBF‑SVM attains ~63–65% Macro‑F1 on balanced Type‑2 data, exceeding deep baselines. All commands assume Python ≥ 3.9 in a virtual environment; unless noted, results are reported as macro Precision/Recall/F1 under 5‑fold stratified cross‑validation.

## Assets: Embeddings & Cluster Map

**`glove.twitter.27B.200d.txt` — Pre‑trained Twitter word embeddings (200‑dimensional).**  
A plain‑text matrix where each line is `token` followed by **200 floating‑point values** (the embedding). These vectors were trained on ~2B tweets (≈27B tokens; ~1.2M vocabulary) and are uncased. Typical rows look like:

**`glove-200.txt` — Token→cluster map (topics) derived from GloVe‑Twitter‑200d.**  
A compact look‑up table that assigns each token to a **discrete cluster ID** (topic/bin) produced by clustering the 200‑d GloVe vectors (e.g., k‑means). Each line is:
- The **last field is an integer cluster ID**; scripts validate this and infer `n_clusters = max(cluster_id) + 1`.  
- The **middle float** is an auxiliary weight (e.g., frequency/score) and is **ignored** by most scripts; the mapping is driven by the final integer.

**Why both files exist.**
- The **embedding file** gives *dense, continuous* semantics for neural models and for producing interpretable clusters.  
- The **cluster map** turns those continuous vectors into *stable, discrete* features (“topics”) that work well with linear/SVM/XGB baselines.

---

## `crawler.py`

**Purpose.** Batch-fetch **user profiles** and **recent tweets** for a given list of accounts using Tweepy v2 client, with manual rate-limit handling and periodic checkpointing. Reads a seed CSV, fetches profiles and tweets (up to ~3,200 per user), and writes a consolidated CSV for downstream feature derivation. The script sets an internal **tweet quota** and sleeps on certain HTTP error codes to avoid bans.

**Inputs.**
- `raw_data.csv` in the working directory with (at least) columns:  
  `username` (or screen name), `user_found` (boolean), `Labels` (class label for that user), and (optionally) `total_posts` (used to cap tweets per user).  
  The script filters `user_found == True` as the crawl list.

**Auth.** Replace the placeholder **API key/secret, access token/secret, bearer token** in the file with your own. _Prefer reading these from environment variables; do not commit keys._

**How it works (high level).**
- Resolves handles → user IDs in batches (up to 100) and stores **profile snapshots** while crawling, with periodic **temp CSV** checkpoints.
- For each user: fetches timeline pages until **min(total_posts, 3,200)** tweets or **QUOTA_TWEETS** is reached, extracting: text, created time, URL entities, mentions, hashtags, and whether the tweet is a retweet/quote/reply. Saves incremental **temp outputs** between batches to avoid data loss on rate-limit.

**Run.**
```bash
python crawler.py
```

**Outputs.**
- `temp_profiles.csv` / `temp_output.csv` — intermediate snapshots while crawling
- `final_output.csv` — long format; one row per tweet with per-tweet text, timestamps, basic engagement fields, and (per-user) the label from the seed CSV.  
  Use this as input to `derive_hu_features.py` after light cleaning.

**Notes on stability.** Batch mode + paginated timelines; saves after each batch and sleeps on rate-limit. If stopped, it resumes from the last saved temp file.

---

## `derive_hu_features.py`

**Purpose.** Compute Human/User (HU) behavioral features over each user’s last-N tweets, and enrich with profile-derived features obtained via the X API (with a CSV fallback). Produces a single, per-user feature table for modeling.

**CLI.**
```bash
python derive_hu_features.py \
  data/final_output.csv \
  features/hu_features.csv \
  --window-size 100 \
  --rates-on-originals \
  --user-col username \
  --time-col tweet_created_at \
  --retweet-flag is_retweet \
  --quote-flag is_quote \
  --reply-flag is_reply \
  --retweet-count retweet_count \
  --quote-count quote_count \
  --reply-count reply_count \
  --text-col text
```

**Positional:** input (CSV or glob), output (CSV).

**Key options:** `--window-size` (N most-recent tweets per user; default 100), `--rates-on-originals` to compute mean engagement only on original tweets; explicit column overrides for user/time/flags/counts/text.

**Column auto-detection.** Tries common variants:
`username/user_id/screen_name` for user; `tweet_created_at/created_at/timestamp/time` for time; sensible defaults for RT/quote/reply flags & counts; optional CSV-level profile backup columns (followers_count, following_count, statuses_count/total_posts, listed_count, age_*).

**What it computes (per user).**
- Over last-N tweets: counts/proportions of originals, retweets, quotes, replies; fraction interactive (= (retweets+quotes+replies)/N); mean inter-tweet interval (days); average `retweet_count`, `quote_count`, `reply_count` (optionally only over originals).
- Profile features: account age (days); profile URL flag; followers/followees/tweets/listed per-day rates; plus raw follower/followee/tweet counts.

**API vs CSV fallback.** Fetches profile metrics via Bearer token; if unavailable, falls back to estimating profile features from CSV columns (if present).

**Output schema (columns).**
`user,
 HU_R_TweetNum, HU_R_TweetPercent_Original, HU_R_TweetPercent_Retweet,
 HU_R_TweetPercent_Quote, HU_R_TweetPercent_Reply, HU_R_RetweetPercent,
 HU_R_QuotePercent, HU_R_ReplyPercent, HU_R_InteractivePer, HU_R_AverageInterval_days,
 HU_R_RetweetedRate, HU_R_QuotedRate, HU_R_RepliedRate,
 U_R_AccountAge_days, U_R_ListedNum, U_R_ProfileUrl,
 U_R_FollowerNumDay, U_R_FolloweeNumDay, U_R_TweetNumDay, U_R_ListedNumDay,
 followers_count, following_count, tweet_count, username, id`

**Expected console line.**
```
[OK] Window size = <N>. Users = <num_users>. API used: <True|False>
```

---

## `lr.py` — Logistic Regression (Users / Message / Fusion)

**Purpose.** 5-fold CV for three families—Users, Message, Fusion—using Logistic Regression with grids over penalty/C (and text n-grams/min_df for Message/Fusion). Also computes feature importances, LOFO ablations (Users), and Message block ablations (drop BoW / Topics / numeric).

**Key expectations.**
- Input CSV must include label + `aggr_text` for Message, and a fixed set of user columns (see top).
- Label mapping: `reliable → 0`, `unreliable → 1` (handled internally).
- Text preprocessing replaces URLs/mentions and removes stopwords (NLTK).
- A cluster map is required for topic features (token→cluster id).

**Run (example).**
```bash
python lr.py \
  --data preprocessed_dataset_all2.csv \
  --cluster glove-200.txt \
  --kfolds 5 \
  --seed 555 \
  --outdir runs/lr_experiment
```

**What gets saved.**
- Per-fold metrics → `lr_results_cv5_per_fold.csv`
- Summary (means ± std) → `lr_results_cv5_summary.csv`
- Aggregated importances (`|coef|`):  
  `importance_users_agg.csv`, `importance_message_agg.csv`, `importance_fusion_agg.csv` (also in `diagnostics/`)
- Ablations:  
  `ablation_users_lofo.csv` (Users), `ablation_message_blocks.csv` (drop `no_bow`, `no_topics`, `no_msg_numeric`)
- Config snapshot → `lr_feature_config.json` (captures user columns, text grid, etc.)
- Models + vectorizers (per fold) under `outdir/models/` (e.g., `foldX_lr_*.joblib`, TF-IDF artifacts)

**Console output (indicative).**
```
=== 5-fold Stratified CV (macro, Logistic Regression) ===
family   P_macro_mean P_macro_std ... F1_macro_mean F1_macro_std
...
```

---

## `svm.py` — SVM (RBF) with LSA grid, permutations, and ablations

**Purpose.** 5-fold CV across Users/Message/Fusion using SVM-RBF. The Message family supports a TF-IDF encoder with optional LSA/SVD dimensionality reduction and dev-time selection of best-k, plus several diagnostics: user permutation importance, message-numeric permutation, topic permutation on top-K clusters, and LOFO ablations for Users.

**CLI.**
```bash
python svm.py \
  --data preprocessed_dataset_all2.csv \
  --cluster glove-200.txt \
  --kfolds 5 \
  --seed 555 \
  --outdir runs/svm_experiment \
  --topics_topk 50 \
  --svd_grid 0,300,500,800 \
  --scale_svd
```

- `--svd_grid`: comma-sep list; `0 = off` (raw BoW).
- `--scale_svd`: z-score SVD components before stacking (recommended).

Requires label, `aggr_text`, and the standard user columns (script validates presence).

**What gets saved.**
- Per-fold → `results_cv5_per_fold.csv`; Summary → `results_cv5_summary.csv`
- Models → `models/fold{K}_svm_*.joblib` (+ TF-IDF for fusion)
- Diagnostics → `diagnostics/`:
  - `ablation_users_lofo.csv` (Users)
  - `ablation_message_blocks_bestk.csv` (Message)
  - `importance_users_agg.csv` (Users permutation agg)
  - `importance_message_numeric_agg.csv` (Message numeric permutation)
  - `importance_topics_perm_agg.csv` (topic permutation agg)
  - `svd_dev_sweep.csv` (Macro-F1 vs k) and `m_f1_selected_vs_k0.csv` (comparative sweep; prints Wilcoxon test if available)

**Console output (indicative).**
```
=== 5-fold Stratified CV (macro, SVM-RBF) ===
...
```

---

## `xgb.py` — XGBoost (trees) over Users / Message / Fusion

**Purpose.** 5-fold CV using XGBClassifier, tuned on a small-sample-friendly grid, supporting Users/Message/Fusion families; includes message block ablations and exports feature importances aggregated across folds (by gain).

**Key bits.**
- Uses a concise set of user columns by default (can be expanded in the file); excludes label/username/text and some columns from “message-numeric” by default.
- Text: TF-IDF (Mu & Aletras style): ngram_range ∈ {(1,1),(1,2)}, min_df ∈ {2,3,5}.
- Class imbalance handling: scale_pos_weight computed on the fly using labels from the train(+dev) split.

**Run (example).**
```bash
python xgb.py \
  --data preprocessed_dataset_all2.csv \
  --cluster glove-200.txt \
  --kfolds 5 \
  --seed 555 \
  --outdir runs/xgb_experiment
```

**What gets saved.**
- Per-fold → `xgb_results_cv5_per_fold.csv`; Summary → `xgb_results_cv5_summary.csv`
- Aggregated importances (gain):  
  `importance_users_agg.csv`, `importance_message_agg.csv`, `importance_fusion_agg.csv` (top-level + in `diagnostics/`)
- Ablations (Message): `no_bow`, `no_topics`, `no_msg_numeric` deltas per fold (diagnostics CSVs)
- Models + vectorizers saved per fold under `outdir/models/` as joblib artifacts (see code for filenames)

**Console output (indicative).**
```
=== 5-fold Stratified CV (macro, XGBoost) ===
family   P_macro_mean ... F1_macro_mean ...
```

---

## `bigru.py` — BiGRU(+Attention) text encoder; MLP for users; late fusion

**Purpose.** Reproduces a Mu & Aletras-style BiGRU + Attention text encoder (optionally initialized with GloVe-Twitter 200d) and pairs it with an MLP (users) for a Fusion model. Runs 5-fold CV over text-only, user-only, and fusion; saves fold summaries, predictions, and checkpoints. Configuration is set via constants at the top of the file (no CLI).

**Edit these before running (top of `bigru.py`).**
- `CSV_PATH` (default: `preprocessed_dataset_all2.csv`) and `OUTPUT_ROOT` (default: `runs/`)
- `GLOVE_PATH` if you have GloVe-Twitter 200d; otherwise the model uses random init
- Model/sequence settings (aligned to the original Keras setup):  
  `MAX_SEQ_LEN=3000`, `EMBED_DIM=200`, `HIDDEN_SIZE=100`, batch 64, epochs 10, LR 1e-3, etc.

**Run.**
```bash
python bigru.py
```

**Outputs per fold (under `runs/`).**
- Text-only: `runs/text_fold{K}/pytorch_model.pt`, `metrics.json`, `predictions.csv`
- User-only: `runs/user_fold{K}/...` similarly
- Fusion: `runs/fusion_fold{K}/...` similarly
- Summaries: `runs/{mode_tag}_cv_summary.csv` for all arms and `runs/{mode_tag}_settings.json` (mode tag encodes T- vs H- settings)

**Console output (indicative).**
```
===== 5-fold CV Summary (Macro metrics) =====
[TEXT]
Precision: <mean> ± <std>
...
```

---

## `bert.py` — T-BERT / H-BERT (text-only or Fusion) + Users MLP (optional)

**Purpose.** A JSON-driven pipeline to run either:
1. MLP on user features only, or
2. BERT (text-only): T-BERT (flat) or H-BERT (hierarchical chunking), or
3. Fusion (BERT + Users),  
with feature importance for users (permutation, LOFO), token-cluster saliency/occlusion for message interpretability (requires cluster map), and stability aggregation across folds.

**CLI.**
```bash
# (A) MLP on users only
python bert.py --model mlp \
  --data preprocessed_dataset_all2.csv \
  --outdir runs_nn

# (B) T-BERT text-only
python bert.py --model bert \
  --data preprocessed_dataset_all2.csv \
  --bert_settings tbert_settings.json \
  --cluster glove-200.txt \
  --outdir runs_nn

# (C) H-BERT + Users fusion
python bert.py --model bert --fusion \
  --data preprocessed_dataset_all2.csv \
  --bert_settings hbert_settings.json \
  --cluster glove-200.txt \
  --outdir runs_nn
```

Required checks. Data columns + label mapping validated at startup. For BERT modes, transformers must be installed and a cluster map provided to enable interpretability exports.

**BERT settings JSON (minimal).**
```json
{
  "CSV_PATH": "preprocessed_dataset_all2.csv",
  "HIERARCHICAL": false,
  "MAX_SEQ_LEN": 512,
  "MAX_CHUNKS": 12,
  "OVERLAP_TOKENS": 0,
  "TRAIN_BATCH_SIZE": 8,
  "EVAL_BATCH_SIZE": 8,
  "NUM_EPOCHS": 4,
  "LEARNING_RATE": 2e-5,
  "WARMUP_RATIO": 0.1,
  "D_USER_LATENT": 16,
  "BERT_MODEL_NAME": "bert-base-uncased",
  "SEED": 555
}
```

**Outputs (under `runs_nn/` by default).**
- Directory scaffold: `cv/`, `diagnostics/`, `stable/`, `model/`
- Per-fold metrics (+ predictions printed to console) → `cv/results_per_fold.csv`, `cv/metrics_summary.csv`, and a formatted `cv/metrics_summary_formatted.csv`
- MLP (users) fold checkpoints: `model/mlp_users_fold{K}.pt` with metadata (feature names, imputers/scalers, metrics, label mapping, fold)
- User importance (users-only MLP): per-fold permutation/LOFO CSVs; aggregated stability tables in `stable/`, and diagnostic aggregates in `diagnostics/users_perm_agg.csv` and `diagnostics/users_lofo_agg.csv`
- BERT interpretability (text): aggregated saliency → `diagnostics/bert_saliency_agg.csv`; occlusion drops per cluster → `diagnostics/bert_occlusion_agg.csv`
- Fusion (BERT+Users): `diagnostics/fusion_user_perm_agg.csv` via permutation over user features

**Console output (indicative).**
```
=== 5-fold CV Summary (BERT) ===
family  n_folds  P            R            F1
...
```

---

## `mlp.py` — MLP (Users only) and (optionally) BERT modes too

**Purpose.** Contains both the Users-only MLP pipeline and the same BERT pipeline as `bert.py`. Use it like `bert.py` with `--model mlp` or `--model bert`. Required columns, label mapping, outputs, and diagnostics behave the same as documented in the `bert.py` section above.

**CLI (examples).**
```bash
# Users-only MLP
python mlp.py --model mlp \
  --data preprocessed_dataset_all2.csv \
  --outdir runs_nn

# T-BERT text-only (same flags as bert.py)
python mlp.py --model bert \
  --data preprocessed_dataset_all2.csv \
  --bert_settings tbert_settings.json \
  --cluster glove-200.txt \
  --outdir runs_nn
```

**What the MLP saves.**
- Fold checkpoints: `model/mlp_users_fold{K}.pt` (feature names + preprocessing) and metrics metadata
- Diagnostics & stability: permutation/LOFO per fold (`diagnostics/users_perm_fold*.csv`, `diagnostics/users_lofo_fold*.csv`), aggregated stability tables in `stable/` and aggregated diagnostics in `diagnostics/`
- CV summaries: `cv/results_per_fold.csv`, `cv/metrics_summary.csv`, `cv/metrics_summary_formatted.csv`

---

## Installation & Environment

Create and activate a virtual environment, then install requirements.
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt

# One-time NLTK data (for stopwords)
python - <<'PY'
import nltk
nltk.download("stopwords")
PY 
```
## Data Files & Purpose

This section documents the role of each CSV in the pipeline and clarifies the two timeline “Type” settings used to construct user histories.

### Timeline Types

- **Type‑1 (Early‑history).** For each user, retain only **original tweets** posted **before** the timestamp of their **first repost of any news source**; omit all earlier retweets (e.g., of friends or non‑news accounts) and any tweets after that first news repost. For reliable users, truncate at an equivalent point (e.g., first mainstream‑news repost or a time matched to the median span of unreliable users) so that features reflect a comparable pre‑exposure window, as in Mu and Aletras (2020). This simulates an **early detection** setting.

- **Type‑2 (Full‑history).** Use each user’s available timeline **up to a fixed collection date**, but **exclude all retweets and direct reposts** from feature computation to avoid label leakage. Thus, Type‑2 encodes a fuller view of **original language and activity**, independent of quoted/forwarded content.

---

### File‑by‑File Overview

- **`raw_data.csv`**  
  Source list from Mu & Aletras’ research repository. Contains a cleaned roster of existing X (formerly Twitter) users and the **classification label** of each user. Use this as the **seed/user‑label reference** for downstream collection and processing.

- **`master_raw_data.csv`**  
  Consolidated crawl output built by querying the X API (within rate limits) for all seed users. This is the **union of collected timelines/profiles** and serves as the **base input** for constructing Type‑1 and Type‑2 datasets.

- **`cleaned_data_filtered_newssource.csv`**  
  Processed dataset following the **Type‑1 (Early‑history)** rule. For each user, rows correspond to content retained **up to the user’s first repost of a (reliable/unreliable) news source**, forming the early‑detection window.

- **`Cleaned_data_set2.csv`**  
  Processed dataset following the **Type‑2 (Full‑history)** rule. User timelines are kept up to the collection cut‑off, while **retweets/direct reposts are excluded** from feature computation.

- **`preprocessed_dataset_all.csv` (Type‑1)**  
  **Model‑ready, per‑user** table derived from the Type‑1 window. Includes **aggregated text** (e.g., `aggr_text`), **user‑centric features**, and **LIWC features** for use by LR/SVM/XGB/BiGRU/BERT pipelines.

- **`preprocessed_dataset_all2.csv` (Type‑2)**  
  **Model‑ready, per‑user** table derived from the Type‑2 window. Includes **aggregated text**, **user‑centric features**, and **LIWC features**, analogous to the Type‑1 counterpart but using the full‑history window.

---

### Broad Overview for Data Files

`raw_data.csv` (seed users & labels)  
→ **collect** → `master_raw_data.csv` (API‑crawled union)  
→ **filter by timeline rule** → `cleaned_data_filtered_newssource.csv` (Type‑1) **and** `Cleaned_data_set2.csv` (Type‑2)  
→ **aggregate & featurize** → `preprocessed_dataset_all.csv` (Type‑1) **and** `preprocessed_dataset_all2.csv` (Type‑2)

> **Note.** All training/evaluation scripts assume consistent label mapping (e.g., `reliable → 0`, `unreliable → 1`) and expect the model‑ready datasets to expose the required columns (e.g., `username`, `label`, `aggr_text`, plus user features).


