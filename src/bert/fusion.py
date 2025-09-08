#!/usr/bin/env python3
"""
5-fold CV for: (1) Text-only (BERT), (2) User-only (MLP), (3) Fusion (BERT + MLP).
T-BERT by default (truncate to 512). Flip HIERARCHICAL=True for H-BERT (chunk + average).
Saves per-fold models, scalers, metrics, and predictions under runs/.
"""
import os, json, re, sys, logging, random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from joblib import dump as joblib_dump

from transformers import AutoTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# ─────────────────────────────────────────────────────────────
# Config & logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="fusion_cv.log", 
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fusion")

# ---- GLOBAL SETTINGS ----
CSV_PATH = "preprocessed_dataset_all2.csv"
OUTPUT_ROOT = "runs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 555

# ---- Tbert MODEL / TRAINING SETTINGS ----
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 512            # tokens per sequence
MAX_CHUNKS = 12              # cap for H-BERT memory
OVERLAP_TOKENS = 0          # e.g., 128 for sliding windows in H-BERT
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE  = 8
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
D_USER_LATENT = 16
NUM_LABELS = 2

# # ---- Hbert MODEL / TRAINING SETTINGS ----
# BERT_MODEL_NAME = "bert-base-uncased"
# MAX_SEQ_LEN = 512            # tokens per sequence
# MAX_CHUNKS = 4              # cap for H-BERT memory
# OVERLAP_TOKENS = 128          # e.g., 128 for sliding windows in H-BERT
# TRAIN_BATCH_SIZE = 2
# EVAL_BATCH_SIZE  = 2
# NUM_EPOCHS = 4
# LEARNING_RATE = 2e-5
# WARMUP_RATIO = 0.1
# D_USER_LATENT = 16
# NUM_LABELS = 2

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
HIERARCHICAL = False      # False = T‑BERT; True = H‑BERT
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# ─────────────────────────────────────────────────────────────
# Mu & Aletras–style text preprocessing for BERT
#   - replace URLs → 'url' and @mentions → 'usr'
#   - lowercase via TweetTokenizer
#   - remove English stopwords (NLTK)
# ─────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

try:
    STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP = set(stopwords.words("english"))

TT = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

def preprocess_for_bert(text: str) -> str:
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+|http\S+", " url ", t)
    t = re.sub(r"@\w+", " usr ", t)
    toks = TT.tokenize(t)
    toks = [w for w in toks if w not in STOP]
    out = " ".join(toks).strip()
    return out if out else "usr"  # guard against empty strings

# ─────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────
class BERTTextDataset(Dataset):
    """Text-only dataset: uses aggr_text → BERT tokens (T-BERT or H-BERT)."""
    def __init__(self, df: pd.DataFrame, tokenizer, hierarchical: bool):
        self.hierarchical = hierarchical
        self.labels = df["labels"].astype(int).values
        self.tokenizer = tokenizer
        self.texts = df["aggr_text"].astype(str).map(preprocess_for_bert).tolist()

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        text = self.texts[idx]

        if not self.hierarchical:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": y,
            }
        else:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_overflowing_tokens=True,
                stride=OVERLAP_TOKENS,
                return_tensors="pt",
            )
            # Already (num_chunks, L)
            ids = enc["input_ids"][:MAX_CHUNKS]
            attn = enc["attention_mask"][:MAX_CHUNKS]
            if ids.shape[0] == 0:
                ids = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
                attn = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": attn, "labels": y}

class UserDataset(Dataset):
    """User-only dataset: uses the 9 tabular user features."""
    def __init__(self, df: pd.DataFrame, numeric_cols: List[str], scaler: StandardScaler):
        X = df[numeric_cols].astype(float).fillna(0.0).values
        self.numeric = scaler.transform(X).astype(np.float32)
        self.labels = df["labels"].astype(int).values

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

class FusionDataset(Dataset):
    """Fusion dataset: aggr_text → BERT + user numeric → MLP."""
    def __init__(self, df: pd.DataFrame, tokenizer, numeric_cols: List[str],
                 scaler: StandardScaler, hierarchical: bool):
        self.text_ds = BERTTextDataset(df, tokenizer, hierarchical)
        X = df[numeric_cols].astype(float).fillna(0.0).values
        self.numeric = scaler.transform(X).astype(np.float32)

    def __len__(self): 
        return len(self.text_ds)

    def __getitem__(self, idx):
        item = self.text_ds[idx]
        item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)
        return item

# ─────────────────────────────────────────────────────────────
# Collate helpers
# ─────────────────────────────────────────────────────────────
def collate_text_tbert(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch]).to(DEVICE)
    attn      = torch.stack([b["attention_mask"] for b in batch]).to(DEVICE)
    labels    = torch.stack([b["labels"] for b in batch]).to(DEVICE)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def collate_text_hbert(batch):
    max_chunks = max(b["input_ids"].shape[0] for b in batch)
    L = batch[0]["input_ids"].shape[1]
    ids_list, attn_list, mask_list = [], [], []
    for b in batch:
        C = b["input_ids"].shape[0]
        padC = max_chunks - C
        if padC > 0:
            pad_ids  = torch.zeros((padC, L), dtype=torch.long)
            pad_attn = torch.zeros((padC, L), dtype=torch.long)
            ids  = torch.cat([b["input_ids"], pad_ids], dim=0)
            attn = torch.cat([b["attention_mask"], pad_attn], dim=0)
        else:
            ids, attn = b["input_ids"], b["attention_mask"]
        cm = torch.zeros(max_chunks, dtype=torch.float32); cm[:C] = 1.0
        ids_list.append(ids.unsqueeze(0)); attn_list.append(attn.unsqueeze(0)); mask_list.append(cm.unsqueeze(0))
    input_ids = torch.cat(ids_list, dim=0).to(DEVICE)     # (B, C, L)
    attn      = torch.cat(attn_list, dim=0).to(DEVICE)    # (B, C, L)
    chunk_mask= torch.cat(mask_list, dim=0).to(DEVICE)    # (B, C)
    labels    = torch.stack([b["labels"] for b in batch]).to(DEVICE)
    return {"input_ids": input_ids, "attention_mask": attn, "chunk_mask": chunk_mask, "labels": labels}

def collate_user(batch):
    numeric = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    labels  = torch.stack([b["labels"] for b in batch]).to(DEVICE)
    return {"numeric": numeric, "labels": labels}

def collate_fusion_tbert(batch):
    out = collate_text_tbert(batch)
    out["numeric"] = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    return out

def collate_fusion_hbert(batch):
    out = collate_text_hbert(batch)
    out["numeric"] = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    return out

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
class TextBERT(nn.Module):
    def __init__(self, hierarchical: bool):
        super().__init__()
        self.hier = hierarchical
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, NUM_LABELS)

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        if not self.hier:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.pooler_output                      # (B, 768)
        else:
            B, C, L = input_ids.shape
            flat_ids  = input_ids.view(B*C, L)
            flat_mask = attention_mask.view(B*C, L)
            out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
            pooled = out.pooler_output.view(B, C, -1)
            w = chunk_mask.unsqueeze(-1)
            pooled = (pooled * w).sum(1) / w.sum(1).clamp(min=1e-6)
        return self.classifier(self.dropout(pooled))

class UserOnlyMLP(nn.Module):
    def __init__(self, in_dim: int, latent: int = D_USER_LATENT):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, latent), nn.ReLU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(latent, NUM_LABELS)

    def forward(self, numeric):
        return self.classifier(self.tower(numeric))

class FusionBERTUser(nn.Module):
    def __init__(self, in_dim_user: int, hierarchical: bool, user_latent: int = D_USER_LATENT):
        super().__init__()
        self.hier = hierarchical
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        h_text = self.bert.config.hidden_size
        self.user = nn.Sequential(
            nn.Linear(in_dim_user, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, user_latent), nn.ReLU(), nn.Dropout(0.1),
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h_text + user_latent, NUM_LABELS)

    def forward(self, input_ids, attention_mask, numeric, chunk_mask=None):
        if not self.hier:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.pooler_output
        else:
            B, C, L = input_ids.shape
            flat_ids  = input_ids.view(B*C, L)
            flat_mask = attention_mask.view(B*C, L)
            out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
            pooled = out.pooler_output.view(B, C, -1)
            w = chunk_mask.unsqueeze(-1)
            pooled = (pooled * w).sum(1) / w.sum(1).clamp(min=1e-6)
        u = self.user(numeric)                               # (B, 16)
        return self.classifier(self.dropout(torch.cat([pooled, u], dim=1)))

# ─────────────────────────────────────────────────────────────
# Train / eval helpers
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, TextBERT):
            logits = model(batch["input_ids"], batch["attention_mask"], batch.get("chunk_mask"))
        elif isinstance(model, UserOnlyMLP):
            logits = model(batch["numeric"])
        else:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["numeric"], batch.get("chunk_mask"))
        loss = loss_fn(logits, batch["labels"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        if isinstance(model, TextBERT):
            logits = model(batch["input_ids"], batch["attention_mask"], batch.get("chunk_mask"))
        elif isinstance(model, UserOnlyMLP):
            logits = model(batch["numeric"])
        else:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["numeric"], batch.get("chunk_mask"))
        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.cpu().numpy().tolist())
        y_true.extend(batch["labels"].cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ─────────────────────────────────────────────────────────────
# Main: 5-fold CV over text-only, user-only, fusion
# ─────────────────────────────────────────────────────────────
def main():
    # Load
    df = pd.read_csv(CSV_PATH)

    needed = {"username", "label", "aggr_text",
              "U_AccountAge_days","total_posts","followers_count","following_count",
              "verified","posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio", "HU_TweetNum", "HU_TweetPercent_Original",
              "HU_TweetPercent_Retweet", "HU_AverageInterval_days", "U_ListedNum", "U_ProfileUrl", "U_FollowerNumDay", "U_FolloweeNumDay", "U_TweetNumDay", "U_ListedNumDay"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    # labels → ints
    df["labels"] = df["label"].map({"reliable": 0, "unreliable": 1}).astype(int)

    # user features
    numeric_cols = [
        "U_AccountAge_days","total_posts","followers_count","following_count",
        "verified","posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio", "HU_TweetNum", "HU_TweetPercent_Original",
        "HU_TweetPercent_Retweet", "HU_AverageInterval_days", "U_ListedNum", "U_ProfileUrl", "U_FollowerNumDay", "U_FolloweeNumDay", "U_TweetNumDay", "U_ListedNumDay"
    ]

    # tokenizer (fast)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=True)

    # folds
    y = df["labels"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mode_tag = "hbert" if HIERARCHICAL else "tbert"

    results = {"text": {"P": [], "R": [], "F1": []},
               "user": {"P": [], "R": [], "F1": []},
               "fusion": {"P": [], "R": [], "F1": []}}

    # For final summary CSV
    summary_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df.index.values, y), start=1):
        print(f"\n===== Fold {fold} / 5 =====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        fold_dir = os.path.join(OUTPUT_ROOT, f"{mode_tag}_fold{fold}")
        ensure_dir(fold_dir)

        # Fit scaler on TRAIN numeric only (no leakage)
        scaler = StandardScaler().fit(df_tr[numeric_cols].astype(float).fillna(0.0).values)
        joblib_dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
        save_json(os.path.join(fold_dir, "numeric_cols.json"), {"numeric_cols": numeric_cols})

        # ---------- TEXT-ONLY ----------
        text_dir = os.path.join(fold_dir, "text")
        ensure_dir(text_dir)
        ds_tr = BERTTextDataset(df_tr, tokenizer, hierarchical=HIERARCHICAL)
        ds_te = BERTTextDataset(df_te, tokenizer, hierarchical=HIERARCHICAL)
        coll_text = collate_text_hbert if HIERARCHICAL else collate_text_tbert
        dl_tr = DataLoader(ds_tr, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=coll_text, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=coll_text, num_workers=0)

        text_model = TextBERT(hierarchical=HIERARCHICAL).to(DEVICE)
        total_steps = max(1, len(dl_tr) * NUM_EPOCHS)
        opt = AdamW(text_model.parameters(), lr=LEARNING_RATE)
        sch = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO*total_steps), total_steps)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(text_model, dl_tr, opt, sch, loss_fn)
            print(f"[Text][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(text_model, dl_te)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[Text][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["text"]["P"].append(p); results["text"]["R"].append(r); results["text"]["F1"].append(f1)

        # save model + metrics + preds
        torch.save(text_model.state_dict(), os.path.join(text_dir, "pytorch_model.pt"))
        save_json(os.path.join(text_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({
            "username": df_te["username"].tolist(),
            "y_true": y_true, "y_pred": y_pred
        }).to_csv(os.path.join(text_dir, "predictions.csv"), index=False)

        summary_rows.append({"fold": fold, "arm": "text", "P": float(p), "R": float(r), "F1": float(f1)})

        # ---------- USER-ONLY ----------
        user_dir = os.path.join(fold_dir, "user")
        ensure_dir(user_dir)
        ds_tr_u = UserDataset(df_tr, numeric_cols, scaler)
        ds_te_u = UserDataset(df_te, numeric_cols, scaler)
        dl_tr_u = DataLoader(ds_tr_u, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=collate_user, num_workers=0)
        dl_te_u = DataLoader(ds_te_u, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=collate_user, num_workers=0)

        user_model = UserOnlyMLP(in_dim=len(numeric_cols)).to(DEVICE)
        opt_u = AdamW(user_model.parameters(), lr=1e-3, weight_decay=0.0)  # higher LR for MLP
        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(user_model, dl_tr_u, opt_u, None, loss_fn)
            print(f"[User][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(user_model, dl_te_u)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[User][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["user"]["P"].append(p); results["user"]["R"].append(r); results["user"]["F1"].append(f1)

        torch.save(user_model.state_dict(), os.path.join(user_dir, "pytorch_model.pt"))
        save_json(os.path.join(user_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({
            "username": df_te["username"].tolist(),
            "y_true": y_true, "y_pred": y_pred
        }).to_csv(os.path.join(user_dir, "predictions.csv"), index=False)

        summary_rows.append({"fold": fold, "arm": "user", "P": float(p), "R": float(r), "F1": float(f1)})

        # ---------- FUSION ----------
        fusion_dir = os.path.join(fold_dir, "fusion")
        ensure_dir(fusion_dir)
        ds_tr_f = FusionDataset(df_tr, tokenizer, numeric_cols, scaler, hierarchical=HIERARCHICAL)
        ds_te_f = FusionDataset(df_te, tokenizer, numeric_cols, scaler, hierarchical=HIERARCHICAL)
        coll_fusion = collate_fusion_hbert if HIERARCHICAL else collate_fusion_tbert
        dl_tr_f = DataLoader(ds_tr_f, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=coll_fusion, num_workers=0)
        dl_te_f = DataLoader(ds_te_f, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=coll_fusion, num_workers=0)

        fusion_model = FusionBERTUser(in_dim_user=len(numeric_cols), hierarchical=HIERARCHICAL).to(DEVICE)
        total_steps_f = max(1, len(dl_tr_f) * NUM_EPOCHS)
        opt_f = AdamW(fusion_model.parameters(), lr=LEARNING_RATE)
        sch_f = get_linear_schedule_with_warmup(opt_f, int(WARMUP_RATIO*total_steps_f), total_steps_f)

        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(fusion_model, dl_tr_f, opt_f, sch_f, loss_fn)
            print(f"[Fusion][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(fusion_model, dl_te_f)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[Fusion][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["fusion"]["P"].append(p); results["fusion"]["R"].append(r); results["fusion"]["F1"].append(f1)

        torch.save(fusion_model.state_dict(), os.path.join(fusion_dir, "pytorch_model.pt"))
        save_json(os.path.join(fusion_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({
            "username": df_te["username"].tolist(),
            "y_true": y_true, "y_pred": y_pred
        }).to_csv(os.path.join(fusion_dir, "predictions.csv"), index=False)

        summary_rows.append({"fold": fold, "arm": "fusion", "P": float(p), "R": float(r), "F1": float(f1)})

    # ---- Summary (print + CSV) ----
    def summarize(name: str, arr: List[float]) -> str:
        m = float(np.mean(arr)); s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return f"{name}: {m:.4f} ± {s:.4f}"

    print("\n===== 5-fold CV Summary (Macro metrics) =====")
    for arm in ["text", "user", "fusion"]:
        print(f"\n[{arm.upper()}]")
        print(summarize("Precision", results[arm]["P"]))
        print(summarize("Recall   ", results[arm]["R"]))
        print(summarize("F1       ", results[arm]["F1"]))

    # save final summary
    ensure_dir(OUTPUT_ROOT)
    mode_tag = "hbert" if HIERARCHICAL else "tbert"
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ROOT, f"{mode_tag}_cv_summary.csv"), index=False)
    save_json(os.path.join(OUTPUT_ROOT, f"{mode_tag}_settings.json"), {
        "CSV_PATH": CSV_PATH,
        "HIERARCHICAL": HIERARCHICAL,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "MAX_CHUNKS": MAX_CHUNKS,
        "OVERLAP_TOKENS": OVERLAP_TOKENS,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WARMUP_RATIO": WARMUP_RATIO,
        "D_USER_LATENT": D_USER_LATENT,
        "BERT_MODEL_NAME": BERT_MODEL_NAME,
        "SEED": SEED
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal")
        print(f"❌ {e}")
        sys.exit(1)
