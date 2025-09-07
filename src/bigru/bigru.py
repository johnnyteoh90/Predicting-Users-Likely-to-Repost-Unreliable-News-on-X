#!/usr/bin/env python3
"""
5-fold CV for: (1) Text-only (BiGRU+Attention), (2) User-only (MLP), (3) Fusion (BiGRU+Att + MLP).

This version aligns the BiGRU+Attention hyperparameters with Mu & Aletras' Keras setup:
- GloVe-Twitter 200d embeddings (frozen if available),
- max vocab = 50k,
- sequence length = 3000 tokens (truncate in T-mode),
- BiGRU (1 layer, bidirectional, hidden size = 100),
- dropout = 0.5 applied after the GRU and before attention,
- Adam LR=1e-3, batch size 64, 10 epochs, no warmup.

Everything else (preprocessing via TweetTokenizer + url/usr placeholders + stopword removal,
5-fold CV with per-fold vocab/scaler, user MLP, and fusion classifier) remains the same.
"""

import os, json, re, sys, logging, random
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from joblib import dump as joblib_dump

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

# ---- PATHS ----
CSV_PATH    = "preprocessed_dataset_all2.csv"
OUTPUT_ROOT = "runs"

# If you have GloVe-Twitter (200d), put the path here. File name is typically:
# "glove.twitter.27B.200d.txt"  (download once from Stanford NLP / other mirrors)
GLOVE_PATH  = "glove.twitter.27B.200d.txt"   # set to the real path or leave as-is to auto-fallback

# ---- DEVICE & REPRO ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 555
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(SEED)

# ─────────────────────────────────────────────────────────────
# Mu & Aletras–aligned TEXT ENCODER (BiGRU+Attention) SETTINGS
# ─────────────────────────────────────────────────────────────
MAX_SEQ_LEN    = 3000       # tokens per (sub)sequence  ← was 512
OVERLAP_TOKENS = 128        # unchanged; used only if HIERARCHICAL=True
MAX_CHUNKS     = 12
EMBED_DIM      = 200        # 200 to match GloVe-Twitter 200d
HIDDEN_SIZE    = 100        # Mu & Aletras grid tried {50,75,100}; we default to 100
NUM_LAYERS     = 1          # 1 layer (their model used single BiGRU layer)
BIDIRECTIONAL  = True
DROPOUT        = 0.5        # applied AFTER GRU and BEFORE attention (matches their model)

# Vocab building
MAX_VOCAB = 50000           # Mu & Aletras used max_features=50000
MIN_FREQ  = 1
RESERVE_SPECIALS = True     # ensure 'url' and 'usr' are in vocab

# ---- TRAINING SETTINGS (Mu & Aletras) ----
TRAIN_BATCH_SIZE = 64       # was 8
EVAL_BATCH_SIZE  = 64
NUM_EPOCHS       = 10       # was 4
LEARNING_RATE    = 1e-3     # Adam default in Keras
WARMUP_RATIO     = 0.0      # no warmup in their code
D_USER_LATENT    = 16
NUM_LABELS       = 2

# ---- MODE TO MATCH BERT MODES ----
HIERARCHICAL = False  # False = T‑BiGRU (truncate); True = H‑BiGRU (chunk + mean)

# ---- EMBEDDING INIT ----
USE_GLOVE_INIT       = True     # use pretrained word vectors if available
FINE_TUNE_EMBEDDINGS = False    # Mu & Aletras freeze embeddings (trainable=False)

# ─────────────────────────────────────────────────────────────
# Mu & Aletras–style text preprocessing (your original choices kept)
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

URL_TOK = "url"
USR_TOK = "usr"
PAD, UNK = "<pad>", "<unk>"

def normalize_and_tokenize(text: str) -> List[str]:
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+|http\S+", f" {URL_TOK} ", t)
    t = re.sub(r"@\w+", f" {USR_TOK} ", t)
    toks = TT.tokenize(t)
    toks = [w for w in toks if w not in STOP]
    return toks if len(toks) > 0 else [USR_TOK]

# ─────────────────────────────────────────────────────────────
# Vocab utilities
# ─────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.pad_index = self.stoi[PAD]
        self.unk_index = self.stoi[UNK]

    def encode(self, tokens: List[str]) -> List[int]:
        s = self.stoi; unk = self.unk_index
        return [s.get(t, unk) for t in tokens]

    def __len__(self): return len(self.itos)

def build_vocab(texts_tokens: List[List[str]], max_size: int = MAX_VOCAB, min_freq: int = MIN_FREQ) -> Vocab:
    cnt = Counter()
    for toks in texts_tokens: cnt.update(toks)

    specials = [PAD, UNK]
    if RESERVE_SPECIALS:
        for sp in (URL_TOK, USR_TOK):
            if sp not in specials:
                specials.append(sp)

    itos = list(dict.fromkeys(specials))  # keep order & uniqueness
    for sp in itos:
        if sp in cnt: del cnt[sp]

    frequent = [tok for tok, c in cnt.items() if c >= min_freq]
    frequent.sort(key=lambda x: (-cnt[x], x))
    if max_size is not None:
        frequent = frequent[: max(0, max_size - len(itos))]
    itos.extend(frequent)
    return Vocab(itos)

def save_vocab_json(vocab: Vocab, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, indent=2)

# ─────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────
class RNNTextDataset(Dataset):
    """
    Text dataset → token ids
    - HIERARCHICAL=False: returns a single id list (to be padded to MAX_SEQ_LEN).
    - HIERARCHICAL=True:  returns list of chunks (each ≤ MAX_SEQ_LEN) using sliding windows with overlap.
    """
    def __init__(self, df: pd.DataFrame, vocab: Vocab, hierarchical: bool):
        self.hier = hierarchical
        self.labels = df["labels"].astype(int).values
        self.vocab = vocab
        self.tokens = df["aggr_text"].astype(str).map(normalize_and_tokenize).tolist()
        self.ids_all = [self.vocab.encode(toks) for toks in self.tokens]

    def __len__(self): return len(self.labels)

    def _chunks(self, ids: List[int]) -> List[List[int]]:
        if len(ids) == 0:
            return [[self.vocab.stoi.get(USR_TOK, self.vocab.unk_index)]]
        step = max(1, MAX_SEQ_LEN - OVERLAP_TOKENS)
        out = [ids[i:i + MAX_SEQ_LEN] for i in range(0, len(ids), step)]
        if len(out) == 0:
            out = [[self.vocab.stoi.get(USR_TOK, self.vocab.unk_index)]]
        return out[:MAX_CHUNKS]

    def __getitem__(self, idx):
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        ids = self.ids_all[idx]
        if not self.hier:
            ids = ids[:MAX_SEQ_LEN]
            return {"input_ids_raw": torch.tensor(ids, dtype=torch.long), "labels": y}
        else:
            chunks = self._chunks(ids)
            chunks_t = [torch.tensor(c, dtype=torch.long) for c in chunks]
            return {"chunks_raw": chunks_t, "labels": y}

class UserDataset(Dataset):
    """User-only dataset: uses the 9 tabular user features."""
    def __init__(self, df: pd.DataFrame, numeric_cols: List[str], scaler: StandardScaler):
        X = df[numeric_cols].astype(float).fillna(0.0).values
        self.numeric = scaler.transform(X).astype(np.float32)
        self.labels = df["labels"].astype(int).values

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

class FusionDataset(Dataset):
    """Fusion dataset: text ids (+ hierarchical optional) + user numeric features."""
    def __init__(self, df: pd.DataFrame, vocab: Vocab, numeric_cols: List[str], scaler: StandardScaler, hierarchical: bool):
        self.text_ds = RNNTextDataset(df, vocab, hierarchical)
        X = df[numeric_cols].astype(float).fillna(0.0).values
        self.numeric = scaler.transform(X).astype(np.float32)

    def __len__(self): return len(self.text_ds)
    def __getitem__(self, idx):
        item = self.text_ds[idx]
        item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)
        return item

# ─────────────────────────────────────────────────────────────
# Collate helpers
# ─────────────────────────────────────────────────────────────
def pad_and_mask(batch_ids: List[torch.Tensor], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L = len(batch_ids), MAX_SEQ_LEN
    out_ids = torch.full((B, L), pad_idx, dtype=torch.long)
    mask    = torch.zeros((B, L), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        n = min(ids.numel(), L)
        out_ids[i, :n] = ids[:n]
        mask[i, :n] = 1
    return out_ids, mask

def collate_text_t(batch, pad_idx: int):
    ids_raw = [b["input_ids_raw"] for b in batch]
    input_ids, attn = pad_and_mask(ids_raw, pad_idx)
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids.to(DEVICE),
            "attention_mask": attn.to(DEVICE),
            "labels": labels.to(DEVICE)}

def collate_text_h(batch, pad_idx: int):
    max_chunks = max(len(b["chunks_raw"]) for b in batch)
    L = MAX_SEQ_LEN
    B = len(batch)
    input_ids = torch.full((B, max_chunks, L), pad_idx, dtype=torch.long)
    attn      = torch.zeros((B, max_chunks, L), dtype=torch.long)
    chunk_mask= torch.zeros((B, max_chunks), dtype=torch.long)
    labels    = torch.zeros((B,), dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i] = b["labels"]
        chunks = b["chunks_raw"]
        C = len(chunks)
        chunk_mask[i, :C] = 1
        for c, ids in enumerate(chunks):
            n = min(ids.numel(), L)
            input_ids[i, c, :n] = ids[:n]
            attn[i, c, :n] = 1
    return {"input_ids": input_ids.to(DEVICE),
            "attention_mask": attn.to(DEVICE),
            "chunk_mask": chunk_mask.to(DEVICE),
            "labels": labels.to(DEVICE)}

def collate_user(batch):
    numeric = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    labels  = torch.stack([b["labels"] for b in batch]).to(DEVICE)
    return {"numeric": numeric, "labels": labels}

def collate_fusion_t(batch, pad_idx: int):
    out = collate_text_t(batch, pad_idx)
    out["numeric"] = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    return out

def collate_fusion_h(batch, pad_idx: int):
    out = collate_text_h(batch, pad_idx)
    out["numeric"] = torch.stack([b["numeric"] for b in batch]).to(DEVICE)
    return out

# ─────────────────────────────────────────────────────────────
# Models: BiGRU + Additive Attention (text), MLP (user), Fusion
# ─────────────────────────────────────────────────────────────
class BiGRUAttEncoder(nn.Module):
    """
    Mu & Aletras–aligned text encoder:
      - Embedding (200d, padding_idx)
      - BiGRU (1 layer, hidden=HIDDEN_SIZE per direction)
      - Dropout(DROPOUT) applied to GRU outputs (matching their Dropout placement)
      - Additive attention (tanh + v) over time steps
    """
    def __init__(self, vocab_size: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
            dropout=0.0,  # single layer → no intra-GRU dropout
        )
        out_dim = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)

        # Dropout position aligned with Keras model: after GRU, before attention
        self.post_rnn_dropout = nn.Dropout(DROPOUT)

        # Additive attention
        self.attn_w = nn.Linear(out_dim, out_dim)
        self.attn_v = nn.Linear(out_dim, 1, bias=False)

        self.out_dim = out_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L) ; attention_mask: (B, L) with 1=real, 0=pad
        emb = self.embedding(input_ids)                              # (B, L, E)
        outputs, _ = self.gru(emb)                                   # (B, L, H*)
        outputs = self.post_rnn_dropout(outputs)                     # Dropout like Keras
        u = torch.tanh(self.attn_w(outputs))                         # (B, L, H*)
        scores = self.attn_v(u).squeeze(-1)                          # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e9)       # mask paddings
        alphas = torch.softmax(scores, dim=1)                        # (B, L)
        context = torch.bmm(alphas.unsqueeze(1), outputs).squeeze(1) # (B, H*)
        return context

class TextBiGRUAttClassifier(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, hierarchical: bool):
        super().__init__()
        self.hier = hierarchical
        self.encoder = BiGRUAttEncoder(vocab_size, pad_idx)
        # No extra dropout before classifier (Keras had only the post-GRU dropout)
        self.pre_fc_dropout = nn.Dropout(0.0)
        self.classifier = nn.Linear(self.encoder.out_dim, NUM_LABELS)

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        if not self.hier:
            rep = self.encoder(input_ids, attention_mask)                 # (B, H*)
        else:
            B, C, L = input_ids.shape
            flat_ids  = input_ids.view(B*C, L)
            flat_mask = attention_mask.view(B*C, L)
            reps = self.encoder(flat_ids, flat_mask).view(B, C, -1)       # (B, C, H*)
            w = chunk_mask.unsqueeze(-1).float()
            rep = (reps * w).sum(1) / w.sum(1).clamp(min=1e-6)            # mean over chunks
        return self.classifier(self.pre_fc_dropout(rep))

class UserOnlyMLP(nn.Module):
    def __init__(self, in_dim: int, latent: int = D_USER_LATENT):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, latent), nn.ReLU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(latent, NUM_LABELS)
    def forward(self, numeric): return self.classifier(self.tower(numeric))

class FusionBiGRUUserClassifier(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, in_dim_user: int, hierarchical: bool, user_latent: int = D_USER_LATENT):
        super().__init__()
        self.hier = hierarchical
        self.text = BiGRUAttEncoder(vocab_size, pad_idx)
        self.user = nn.Sequential(
            nn.Linear(in_dim_user, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, user_latent), nn.ReLU(), nn.Dropout(0.1),
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.text.out_dim + user_latent, NUM_LABELS)

    def forward(self, input_ids, attention_mask, numeric, chunk_mask=None):
        if not self.hier:
            t = self.text(input_ids, attention_mask)                       # (B, H*)
        else:
            B, C, L = input_ids.shape
            flat_ids  = input_ids.view(B*C, L)
            flat_mask = attention_mask.view(B*C, L)
            reps = self.text(flat_ids, flat_mask).view(B, C, -1)           # (B, C, H*)
            w = chunk_mask.unsqueeze(-1).float()
            t = (reps * w).sum(1) / w.sum(1).clamp(min=1e-6)               # mean over chunks
        u = self.user(numeric)                                             # (B, latent)
        return self.classifier(self.dropout(torch.cat([t, u], dim=1)))

# ─────────────────────────────────────────────────────────────
# Train / eval helpers
# ─────────────────────────────────────────────────────────────
def make_linear_warmup_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int):
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, TextBiGRUAttClassifier):
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
        if isinstance(model, TextBiGRUAttClassifier):
            logits = model(batch["input_ids"], batch["attention_mask"], batch.get("chunk_mask"))
        elif isinstance(model, UserOnlyMLP):
            logits = model(batch["numeric"])
        else:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["numeric"], batch.get("chunk_mask"))
        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.cpu().numpy().tolist())
        y_true.extend(batch["labels"].cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

# ─────────────────────────────────────────────────────────────
# Embedding init from GloVe-Twitter (optional)
# ─────────────────────────────────────────────────────────────
def maybe_load_glove_matrix(vocab: Vocab, glove_path: str, dim: int) -> Tuple[torch.Tensor, Dict]:
    info = {"used": False, "path": glove_path, "dim": dim, "coverage": 0.0, "found": 0, "total": len(vocab)}
    if not USE_GLOVE_INIT:
        return None, info
    if not os.path.isfile(glove_path):
        logger.warning(f"GloVe file not found at {glove_path}. Using random init (trainable embeddings).")
        return None, info

    wv = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            w = parts[0]; vec = np.asarray(parts[1:], dtype="float32")
            if vec.shape[0] == dim:
                wv[w] = vec

    mat = np.random.normal(scale=0.02, size=(len(vocab), dim)).astype("float32")
    mat[vocab.pad_index] = 0.0
    found = 0
    for i, tok in enumerate(vocab.itos):
        if tok in wv:
            mat[i] = wv[tok]
            found += 1
    info.update({"used": True, "coverage": float(found)/max(1,len(vocab)), "found": int(found)})
    return torch.tensor(mat), info

def apply_embedding_init(module_with_embedding: nn.Module, emb_matrix: torch.Tensor, trainable: bool):
    emb = None
    # Find the embedding attribute (for both Text and Fusion encoders)
    if hasattr(module_with_embedding, "encoder") and hasattr(module_with_embedding.encoder, "embedding"):
        emb = module_with_embedding.encoder.embedding
    elif hasattr(module_with_embedding, "text") and hasattr(module_with_embedding.text, "embedding"):
        emb = module_with_embedding.text.embedding
    elif hasattr(module_with_embedding, "embedding"):
        emb = module_with_embedding.embedding
    if emb is None:
        return
    with torch.no_grad():
        if emb.weight.shape == emb_matrix.shape:
            emb.weight.copy_(emb_matrix)
    emb.weight.requires_grad = bool(trainable)

# ─────────────────────────────────────────────────────────────
# Main: 5-fold CV over text-only, user-only, fusion
# ─────────────────────────────────────────────────────────────
def main():
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

    # folds
    y = df["labels"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mode_tag = "h_bigruatt" if HIERARCHICAL else "t_bigruatt"

    results = {"text": {"P": [], "R": [], "F1": []},
               "user": {"P": [], "R": [], "F1": []},
               "fusion": {"P": [], "R": [], "F1": []}}

    summary_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df.index.values, y), start=1):
        print(f"\n===== Fold {fold} / 5 =====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        fold_dir = os.path.join(OUTPUT_ROOT, f"{mode_tag}_fold{fold}")
        ensure_dir(fold_dir)

        # Build vocab on TRAIN text only (no leakage)
        train_tokens = df_tr["aggr_text"].astype(str).map(normalize_and_tokenize).tolist()
        vocab = build_vocab(train_tokens, max_size=MAX_VOCAB, min_freq=MIN_FREQ)
        save_vocab_json(vocab, os.path.join(fold_dir, "vocab.json"))

        # Fit scaler on TRAIN numeric only (no leakage)
        scaler = StandardScaler().fit(df_tr[numeric_cols].astype(float).fillna(0.0).values)
        joblib_dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
        save_json(os.path.join(fold_dir, "numeric_cols.json"), {"numeric_cols": numeric_cols})

        # Prepare collate fns
        pad_idx = vocab.pad_index
        coll_text = (lambda b: collate_text_h(b, pad_idx)) if HIERARCHICAL else (lambda b: collate_text_t(b, pad_idx))
        coll_fusion= (lambda b: collate_fusion_h(b, pad_idx)) if HIERARCHICAL else (lambda b: collate_fusion_t(b, pad_idx))

        # ---------- TEXT-ONLY ----------
        text_dir = os.path.join(fold_dir, "text"); ensure_dir(text_dir)
        ds_tr  = RNNTextDataset(df_tr, vocab, hierarchical=HIERARCHICAL)
        ds_te  = RNNTextDataset(df_te, vocab, hierarchical=HIERARCHICAL)
        dl_tr  = DataLoader(ds_tr, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=coll_text,  num_workers=0)
        dl_te  = DataLoader(ds_te, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=coll_text,  num_workers=0)

        text_model = TextBiGRUAttClassifier(vocab_size=len(vocab), pad_idx=pad_idx, hierarchical=HIERARCHICAL).to(DEVICE)

        # Optional GloVe init (freeze if loaded, per Mu & Aletras)
        emb_matrix, emb_info = maybe_load_glove_matrix(vocab, GLOVE_PATH, EMBED_DIM)
        save_json(os.path.join(text_dir, "embeddings_info.json"), emb_info)
        if emb_matrix is not None:
            apply_embedding_init(text_model, emb_matrix, FINE_TUNE_EMBEDDINGS)

        total_steps = max(1, len(dl_tr) * NUM_EPOCHS)
        opt = torch.optim.Adam(text_model.parameters(), lr=LEARNING_RATE)
        sch = make_linear_warmup_scheduler(opt, int(WARMUP_RATIO*total_steps), total_steps) if WARMUP_RATIO>0 else None
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(text_model, dl_tr, opt, sch, loss_fn)
            print(f"[Text][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(text_model, dl_te)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[Text][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["text"]["P"].append(p); results["text"]["R"].append(r); results["text"]["F1"].append(f1)

        torch.save(text_model.state_dict(), os.path.join(text_dir, "pytorch_model.pt"))
        save_json(os.path.join(text_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({"username": df_te["username"].tolist(), "y_true": y_true, "y_pred": y_pred}).to_csv(os.path.join(text_dir, "predictions.csv"), index=False)
        summary_rows.append({"fold": fold, "arm": "text", "P": float(p), "R": float(r), "F1": float(f1)})

        # ---------- USER-ONLY ----------
        user_dir = os.path.join(fold_dir, "user"); ensure_dir(user_dir)
        ds_tr_u = UserDataset(df_tr, numeric_cols, scaler)
        ds_te_u = UserDataset(df_te, numeric_cols, scaler)
        dl_tr_u = DataLoader(ds_tr_u, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=collate_user, num_workers=0)
        dl_te_u = DataLoader(ds_te_u, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=collate_user, num_workers=0)

        user_model = UserOnlyMLP(in_dim=len(numeric_cols)).to(DEVICE)
        opt_u = torch.optim.Adam(user_model.parameters(), lr=1e-3)
        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(user_model, dl_tr_u, opt_u, None, loss_fn)
            print(f"[User][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(user_model, dl_te_u)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[User][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["user"]["P"].append(p); results["user"]["R"].append(r); results["user"]["F1"].append(f1)

        torch.save(user_model.state_dict(), os.path.join(user_dir, "pytorch_model.pt"))
        save_json(os.path.join(user_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({"username": df_te["username"].tolist(), "y_true": y_true, "y_pred": y_pred}).to_csv(os.path.join(user_dir, "predictions.csv"), index=False)
        summary_rows.append({"fold": fold, "arm": "user", "P": float(p), "R": float(r), "F1": float(f1)})

        # ---------- FUSION ----------
        fusion_dir = os.path.join(fold_dir, "fusion"); ensure_dir(fusion_dir)
        ds_tr_f = FusionDataset(df_tr, vocab, numeric_cols, scaler, hierarchical=HIERARCHICAL)
        ds_te_f = FusionDataset(df_te, vocab, numeric_cols, scaler, hierarchical=HIERARCHICAL)
        dl_tr_f = DataLoader(ds_tr_f, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=coll_fusion, num_workers=0)
        dl_te_f = DataLoader(ds_te_f, batch_size=EVAL_BATCH_SIZE,  shuffle=False, collate_fn=coll_fusion, num_workers=0)

        fusion_model = FusionBiGRUUserClassifier(vocab_size=len(vocab), pad_idx=pad_idx, in_dim_user=len(numeric_cols), hierarchical=HIERARCHICAL).to(DEVICE)
        # GloVe init for fusion text encoder (freeze if loaded)
        if emb_matrix is not None:
            apply_embedding_init(fusion_model, emb_matrix, FINE_TUNE_EMBEDDINGS)

        total_steps_f = max(1, len(dl_tr_f) * NUM_EPOCHS)
        opt_f = torch.optim.Adam(fusion_model.parameters(), lr=LEARNING_RATE)
        sch_f = make_linear_warmup_scheduler(opt_f, int(WARMUP_RATIO*total_steps_f), total_steps_f) if WARMUP_RATIO>0 else None

        for ep in range(1, NUM_EPOCHS+1):
            loss = train_one_epoch(fusion_model, dl_tr_f, opt_f, sch_f, loss_fn)
            print(f"[Fusion][Fold {fold}] Epoch {ep} loss: {loss:.4f}")

        y_true, y_pred = evaluate(fusion_model, dl_te_f)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"[Fusion][Fold {fold}] Macro-P {p:.4f}  Macro-R {r:.4f}  Macro-F1 {f1:.4f}")
        results["fusion"]["P"].append(p); results["fusion"]["R"].append(r); results["fusion"]["F1"].append(f1)

        torch.save(fusion_model.state_dict(), os.path.join(fusion_dir, "pytorch_model.pt"))
        save_json(os.path.join(fusion_dir, "metrics.json"), {"macro_P": float(p), "macro_R": float(r), "macro_F1": float(f1)})
        pd.DataFrame({"username": df_te["username"].tolist(), "y_true": y_true, "y_pred": y_pred}).to_csv(os.path.join(fusion_dir, "predictions.csv"), index=False)
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

    ensure_dir(OUTPUT_ROOT)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ROOT, f"{mode_tag}_cv_summary.csv"), index=False)
    save_json(os.path.join(OUTPUT_ROOT, f"{mode_tag}_settings.json"), {
        "CSV_PATH": CSV_PATH,
        "HIERARCHICAL": HIERARCHICAL,
        "MAX_SEQ_LEN": MAX_SEQ_LEN, "OVERLAP_TOKENS": OVERLAP_TOKENS, "MAX_CHUNKS": MAX_CHUNKS,
        "EMBED_DIM": EMBED_DIM, "HIDDEN_SIZE": HIDDEN_SIZE, "NUM_LAYERS": NUM_LAYERS,
        "BIDIRECTIONAL": BIDIRECTIONAL, "DROPOUT": DROPOUT,
        "MAX_VOCAB": MAX_VOCAB, "MIN_FREQ": MIN_FREQ, "RESERVE_SPECIALS": RESERVE_SPECIALS,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE, "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS, "LEARNING_RATE": LEARNING_RATE, "WARMUP_RATIO": WARMUP_RATIO,
        "D_USER_LATENT": D_USER_LATENT, "SEED": SEED,
        "USE_GLOVE_INIT": USE_GLOVE_INIT, "FINE_TUNE_EMBEDDINGS": FINE_TUNE_EMBEDDINGS,
        "GLOVE_PATH": GLOVE_PATH
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal")
        print(f"❌ {e}")
        sys.exit(1)
