#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, logging, random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# transformers is required for --model bert
try:
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
except Exception:
    AutoTokenizer = None  # allows running mlp-only on systems without transformers

# -------------- logging --------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nn_bert_mlp")

# -------------- user features --------------
NEEDED_USER_COLS = [
    "U_AccountAge_days","total_posts","followers_count","following_count","verified",
    "posting_frequency","retweet_count","follower_to_followee_ratio","retweet_ratio",
    "HU_TweetNum","HU_TweetPercent_Original","HU_TweetPercent_Retweet","HU_AverageInterval_days",
    "U_ListedNum","U_ProfileUrl","U_FollowerNumDay","U_FolloweeNumDay","U_TweetNumDay","U_ListedNumDay"
]
HEAVY_TAILED_LOG1P = {"followers_count","following_count","retweet_count","HU_TweetNum"}

# -------------- text cleaning (for BERT tokenizer) --------------
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
    return out if out else "usr"

# -------------- cluster map (token -> int cluster_id) --------------
def load_cluster_map(path: str):
    if not path or not os.path.isfile(path):
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
                raise ValueError("Cluster map must be: token ... <int cluster id> (last token is an integer).")
            token = parts[0]
            cmap[token] = cid
            n_lines += 1
            if n_lines >= 200000:
                break
    if not cmap:
        raise ValueError("No entries in cluster file.")
    nclus = max(cmap.values()) + 1
    log.info(f"Loaded cluster map: {len(cmap)} tokens -> {nclus} clusters")
    return cmap, nclus

# -------------- utils --------------
def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def macro_scores(y_true, y_pred) -> Tuple[float,float,float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(p), float(r), float(f1)

def tokens_from_ids(tokenizer, ids: torch.Tensor) -> List[str]:
    toks = tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
    toks = [t.replace("##","").lower() for t in toks]
    toks = [re.sub(r"[^a-z0-9_]", "", t) for t in toks]
    return toks

# ---- helpers to export aggregated diagnostics & CV summaries ----
def save_cv_summary(per_rows: List[Dict], outdir: str):
    """
    Write per-fold results and a summary (mean/std across folds) by model family.
    Saves both raw and formatted summaries, and returns (raw_summary_df, formatted_df).
    Formatted summary shows means as percentages with 1 decimal place and std as 0.xxx (3 decimals).
    """
    cv_dir = os.path.join(outdir, "cv"); ensure_dir(cv_dir)
    df = pd.DataFrame(per_rows)
    df.to_csv(os.path.join(cv_dir, "results_per_fold.csv"), index=False)

    if df.empty:
        # still create empty files for consistency
        empty = pd.DataFrame(columns=["family","P_mean","P_std","R_mean","R_std","F1_mean","F1_std","n_folds"])
        empty.to_csv(os.path.join(cv_dir, "metrics_summary.csv"), index=False)
        fmt_empty = pd.DataFrame(columns=["family","n_folds","P","R","F1"])
        fmt_empty.to_csv(os.path.join(cv_dir, "metrics_summary_formatted.csv"), index=False)
        return empty, fmt_empty

    summary = df.groupby("family").agg(
        P_mean=("P","mean"), P_std=("P","std"),
        R_mean=("R","mean"), R_std=("R","std"),
        F1_mean=("F1","mean"), F1_std=("F1","std"),
        n_folds=("fold","nunique"),
    ).reset_index()

    # Save raw numeric summary
    summary.to_csv(os.path.join(cv_dir, "metrics_summary.csv"), index=False)

    # Build formatted display: mean as % with 1 decimal; std as 0.xxx (3 decimals)
    fmt = pd.DataFrame({
        "family":  summary["family"],
        "n_folds": summary["n_folds"],
        "P":  (summary["P_mean"]*100).map(lambda v: f"{v:.1f}%") + " ± " + summary["P_std"].map(lambda v: f"{v:.3f}"),
        "R":  (summary["R_mean"]*100).map(lambda v: f"{v:.1f}%") + " ± " + summary["R_std"].map(lambda v: f"{v:.3f}"),
        "F1": (summary["F1_mean"]*100).map(lambda v: f"{v:.1f}%") + " ± " + summary["F1_std"].map(lambda v: f"{v:.3f}")
    })
    fmt.to_csv(os.path.join(cv_dir, "metrics_summary_formatted.csv"), index=False)

    log.info("Saved CV summary → %s", os.path.join(cv_dir, "metrics_summary.csv"))
    log.info("Saved formatted CV summary → %s", os.path.join(cv_dir, "metrics_summary_formatted.csv"))
    return summary, fmt

def save_agg_from_rows(rows: List[Dict], key: str, val: str, out_csv: str,
                       sort_desc: bool=True, rename_prefix: Optional[str]=None,
                       extra_means: Optional[List[str]]=None):
    """
    Generic aggregator: group by `key`, aggregate `val` with mean/std/count.
    Optionally also average extra columns (e.g., base_F1, abl_F1).
    """
    diag_dir = os.path.dirname(out_csv); ensure_dir(diag_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame(columns=[key, f"{val}_mean", f"{val}_std", "n_folds"]).to_csv(out_csv, index=False)
        return
    agg_main = df.groupby(key)[val].agg(["mean","std","count"]).reset_index()
    if extra_means:
        extra = df.groupby(key)[extra_means].mean().reset_index()
        agg = pd.merge(agg_main, extra, on=key, how="left")
    else:
        agg = agg_main
    if rename_prefix:
        agg = agg.rename(columns={"mean": f"{rename_prefix}_mean", "std": f"{rename_prefix}_std", "count": "n_folds"})
    else:
        agg = agg.rename(columns={"mean": f"{val}_mean", "std": f"{val}_std", "count": "n_folds"})
    sort_col = f"{rename_prefix}_mean" if rename_prefix else f"{val}_mean"
    agg = agg.sort_values(sort_col, ascending=not sort_desc)
    agg.to_csv(out_csv, index=False)
    log.info("Saved aggregated diagnostics → %s", out_csv)

# -------------- model saving helper --------------
def save_checkpoint(model_obj: nn.Module, path: str, extra: Optional[dict] = None):
    """
    Save a PyTorch model checkpoint with state_dict moved to CPU plus any extra metadata.
    """
    state = {k: v.detach().cpu() for k, v in model_obj.state_dict().items()}
    payload = {"state_dict": state}
    if extra:
        payload.update(extra)
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)
    try:
        sz_mb = os.path.getsize(path) / 1e6
        log.info("Saved model checkpoint → %s (%.2f MB)", path, sz_mb)
    except Exception:
        log.info("Saved model checkpoint → %s", path)

# -------------- MLP for users --------------
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128,64,32], drop: float=0.1, n_classes: int=2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def build_user_matrix(df: pd.DataFrame, cols: List[str], fit_on: Optional[pd.DataFrame]=None,
                      imp: SimpleImputer=None, sc: StandardScaler=None):
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return np.zeros((len(df),0), dtype=np.float32), None, None, []
    Xraw = df[use_cols].apply(pd.to_numeric, errors="coerce").values
    name2idx = {c:i for i,c in enumerate(use_cols)}
    for c in use_cols:
        if c in HEAVY_TAILED_LOG1P:
            j = name2idx[c]; Xraw[:, j] = np.log1p(np.maximum(Xraw[:, j], 0.0))
    if imp is None:
        base = fit_on if fit_on is not None else df
        imp = SimpleImputer(strategy="median").fit(base[use_cols].apply(pd.to_numeric, errors="coerce").values)
    X = imp.transform(df[use_cols].apply(pd.to_numeric, errors="coerce").values)
    if sc is None:
        base = fit_on if fit_on is not None else df
        Xb = imp.transform(base[use_cols].apply(pd.to_numeric, errors="coerce").values)
        sc = StandardScaler().fit(Xb)
    X = sc.transform(X)
    names = [f"user:{c}" for c in use_cols]
    return X.astype(np.float32), imp, sc, names

def perm_importance_user(model, X_test, y_test, feature_names, repeats=20, batch_size=256, device="cpu"):
    ds = TabularDataset(X_test, y_test); dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    @torch.no_grad()
    def eval_once():
        model.eval(); y_true=[]; y_pred=[]
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb); pred = logits.argmax(1).cpu().numpy()
            y_true.append(yb.cpu().numpy()); y_pred.append(pred)
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        return macro_scores(y_true, y_pred)
    baseP, baseR, baseF1 = eval_once()
    rng = np.random.default_rng(2024)
    rows=[]
    for j, name in enumerate(feature_names):
        drops=[]
        for _ in range(repeats):
            Xp = X_test.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            dlp = DataLoader(TabularDataset(Xp, y_test), batch_size=batch_size, shuffle=False)
            @torch.no_grad()
            def eval_perm():
                model.eval(); y_true=[]; y_pred=[]
                for xb, yb in dlp:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb).argmax(1).cpu().numpy()
                    y_true.append(yb.cpu().numpy()); y_pred.append(pred)
                y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
                return macro_scores(y_true, y_pred)[2]
            f1p = eval_perm()
            drops.append(float(baseF1 - f1p))
        rows.append({"feature": name, "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops, ddof=1) if len(drops)>1 else 0.0)})
    return rows, baseF1

def lofo_user_ablation(model, X_test, y_test, feature_names, device="cpu", batch_size=256):
    ds = TabularDataset(X_test, y_test); dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    @torch.no_grad()
    def eval_dl(dloader):
        model.eval(); y_true=[]; y_pred=[]
        for xb, yb in dloader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            y_true.append(yb.numpy()); y_pred.append(pred)
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        return macro_scores(y_true, y_pred)[2]
    baseF1 = eval_dl(dl)
    rows=[]
    for j, name in enumerate(feature_names):
        Xz = X_test.copy(); Xz[:, j] = 0.0
        dlp = DataLoader(TabularDataset(Xz, y_test), batch_size=batch_size, shuffle=False)
        f1z = eval_dl(dlp)
        rows.append({"feature": name, "drop_F1": float(baseF1 - f1z), "base_F1": baseF1, "abl_F1": f1z})
    return rows

# -------------- BERT text models --------------
class TextBERT(nn.Module):
    """Text-only classifier. hierarchical=False → T-BERT; True → H-BERT (mean over chunk CLS)."""
    def __init__(self, model_name: str, n_classes: int=2, dropout: float=0.1, hierarchical: bool=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hierarchical = hierarchical
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h, n_classes)

    def forward(self, input_ids=None, attention_mask=None, chunk_mask=None, inputs_embeds=None):
        if not self.hierarchical:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask) if inputs_embeds is None \
                  else self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            cls = out.last_hidden_state[:,0,:]
            return self.fc(self.dropout(cls))
        else:
            B, C, L = input_ids.shape
            ids2d = input_ids.view(B*C, L)
            att2d = attention_mask.view(B*C, L)
            out = self.bert(input_ids=ids2d, attention_mask=att2d) if inputs_embeds is None \
                  else self.bert(inputs_embeds=inputs_embeds, attention_mask=att2d)
            cls2d = out.last_hidden_state[:,0,:]          # [B*C, H]
            cls = cls2d.view(B, C, -1)                    # [B, C, H]
            mask = chunk_mask.unsqueeze(-1).float()       # [B, C, 1]
            summed = (cls * mask).sum(1)                  # [B, H]
            denom = mask.sum(1).clamp(min=1e-6)           # [B, 1]
            pooled = summed / denom
            return self.fc(self.dropout(pooled))

class BertFusion(nn.Module):
    """BERT text + small MLP over users."""
    def __init__(self, model_name: str, user_dim: int, user_hidden: int=32, n_classes: int=2, dropout: float=0.1, hierarchical: bool=False):
        super().__init__()
        self.text = TextBERT(model_name=model_name, n_classes=n_classes, dropout=dropout, hierarchical=hierarchical)
        self.user_mlp = nn.Sequential(nn.Linear(user_dim, user_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fc = nn.Linear(self.text.bert.config.hidden_size + user_hidden, n_classes)
        self.hierarchical = hierarchical
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids=None, attention_mask=None, chunk_mask=None, user_x=None, inputs_embeds=None):
        if not self.hierarchical:
            out = self.text.bert(input_ids=input_ids, attention_mask=attention_mask) if inputs_embeds is None \
                  else self.text.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            cls = out.last_hidden_state[:,0,:]
        else:
            B, C, L = input_ids.shape
            ids2d = input_ids.view(B*C, L)
            att2d = attention_mask.view(B*C, L)
            out = self.text.bert(input_ids=ids2d, attention_mask=att2d) if inputs_embeds is None \
                  else self.text.bert(inputs_embeds=inputs_embeds, attention_mask=att2d)
            cls2d = out.last_hidden_state[:,0,:]
            cls = (cls2d.view(B, C, -1) * chunk_mask.unsqueeze(-1).float()).sum(1) / chunk_mask.sum(1, keepdim=True).clamp(min=1e-6)
        u = self.user_mlp(user_x)
        z = torch.cat([self.dropout(cls), u], dim=1)
        return self.fc(z)

# -------------- BERT datasets & collate --------------
class BERTTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, hierarchical: bool, overlap_tokens: int, max_chunks: int):
        self.hierarchical = hierarchical
        self.labels = df["label"].astype(int).values
        self.tokenizer = tokenizer
        self.texts = df["aggr_text"].astype(str).map(preprocess_for_bert).tolist()
        self.max_len = max_len; self.overlap = overlap_tokens; self.max_chunks = max_chunks
        self.row_indices = np.arange(len(df), dtype=np.int64)  # for aligning user features

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        ridx = torch.tensor(int(self.row_indices[idx]), dtype=torch.long)
        text = self.texts[idx]
        if not self.hierarchical:
            enc = self.tokenizer(
                text, add_special_tokens=True, max_length=self.max_len, padding="max_length",
                truncation=True, return_attention_mask=True, return_tensors="pt",
            )
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": y, "row_idx": ridx}
        else:
            enc = self.tokenizer(
                text, add_special_tokens=True, max_length=self.max_len, padding="max_length",
                truncation=True, return_attention_mask=True, return_overflowing_tokens=True,
                stride=self.overlap, return_tensors="pt",
            )
            ids = enc["input_ids"][:self.max_chunks]
            att = enc["attention_mask"][:self.max_chunks]
            if ids.shape[0] == 0:
                ids = torch.zeros((1, self.max_len), dtype=torch.long)
                att = torch.zeros((1, self.max_len), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": att, "labels": y, "row_idx": ridx}

def collate_tbert(batch):
    ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    att = torch.stack([b["attention_mask"] for b in batch], dim=0)
    y   = torch.stack([b["labels"] for b in batch], dim=0)
    ridx= torch.stack([b["row_idx"] for b in batch], dim=0)
    return {"input_ids": ids, "attention_mask": att, "labels": y, "row_idx": ridx}

def collate_hbert(batch):
    B = len(batch)
    n_chunks = [b["input_ids"].shape[0] for b in batch]
    C = max(n_chunks)
    L = batch[0]["input_ids"].shape[1]
    ids = torch.zeros((B, C, L), dtype=torch.long)
    att = torch.zeros((B, C, L), dtype=torch.long)
    cm  = torch.zeros((B, C), dtype=torch.long)
    y   = torch.stack([b["labels"] for b in batch], dim=0)
    ridx= torch.stack([b["row_idx"] for b in batch], dim=0)
    for i,b in enumerate(batch):
        c = b["input_ids"].shape[0]
        ids[i,:c,:] = b["input_ids"]
        att[i,:c,:] = b["attention_mask"]
        cm[i,:c] = 1
    return {"input_ids": ids, "attention_mask": att, "chunk_mask": cm, "labels": y, "row_idx": ridx}

# -------------- BERT attribution helpers --------------
@torch.no_grad()
def eval_text(model, loader, device, hierarchical):
    model.eval(); y_true=[]; y_pred=[]
    for batch in loader:
        if not hierarchical:
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=att)
        else:
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device); cm = batch["chunk_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=att, chunk_mask=cm)
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(batch["labels"].cpu().numpy()); y_pred.append(pred)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return (*macro_scores(y_true, y_pred), y_true, y_pred)

def bert_grad_times_input(model, tokenizer, loader, device, hierarchical, cluster_map: Dict[str,int]):
    model.eval()
    emb = model.text.bert.get_input_embeddings() if isinstance(model, BertFusion) else model.bert.get_input_embeddings()
    cluster_sal = {}
    for batch in loader:
        if not hierarchical:
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device)
            with torch.no_grad():
                logits = (model(input_ids=ids, attention_mask=att))
                target = logits.argmax(1)
            inp = emb(ids).detach().requires_grad_(True)
            logits2 = (model(inputs_embeds=inp, attention_mask=att))
        else:
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device); cm = batch["chunk_mask"].to(device)
            B,C,L = ids.shape
            with torch.no_grad():
                logits = (model(input_ids=ids, attention_mask=att, chunk_mask=cm))
                target = logits.argmax(1)
            inp = emb(ids.view(B*C, L)).detach().requires_grad_(True)
            if isinstance(model, BertFusion):
                out = model.text.bert(inputs_embeds=inp, attention_mask=att.view(B*C, L))
                cls2d = out.last_hidden_state[:,0,:].view(B, C, -1)
                mask = cm.unsqueeze(-1).float()
                pooled = (cls2d*mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                logits2 = model.fc(torch.cat([pooled, model.user_mlp(torch.zeros(B, model.user_mlp[0].in_features, device=device))], dim=1))
            else:
                out = model.bert(inputs_embeds=inp, attention_mask=att.view(B*C, L))
                cls2d = out.last_hidden_state[:,0,:].view(B, C, -1)
                mask = cm.unsqueeze(-1).float()
                pooled = (cls2d*mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                logits2 = model.fc(model.dropout(pooled))
        loss = logits2[torch.arange(target.size(0), device=device), target].sum()
        model.zero_grad(); loss.backward()
        grads = inp.grad
        sal = (grads * inp).abs().sum(-1)
        if hierarchical:
            sal = sal.view(ids.size(0), -1)
            ids2 = ids.view(ids.size(0), -1)
            for b in range(ids2.size(0)):
                toks = tokens_from_ids(tokenizer, ids2[b])
                vals = sal[b].detach().cpu().numpy()
                for t, v in zip(toks, vals):
                    if not t: continue
                    cid = cluster_map.get(t)
                    if cid is not None:
                        cluster_sal[cid] = cluster_sal.get(cid, 0.0) + float(v)
        else:
            for b in range(ids.size(0)):
                toks = tokens_from_ids(tokenizer, batch["input_ids"][b])
                vals = sal[b].detach().cpu().numpy()
                for t, v in zip(toks, vals):
                    if not t: continue
                    cid = cluster_map.get(t)
                    if cid is not None:
                        cluster_sal[cid] = cluster_sal.get(cid, 0.0) + float(v)
    return cluster_sal

@torch.no_grad()
def bert_cluster_occlusion(model, tokenizer, loader, device, hierarchical, clusters_to_test: List[int], cluster_map: Dict[str,int]):
    baseP, baseR, baseF1, _, _ = eval_text(model, loader, device, hierarchical)
    drops = {}
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    for cid in clusters_to_test:
        y_true=[]; y_pred=[]
        for batch in loader:
            if not hierarchical:
                ids = batch["input_ids"].clone()
                att = batch["attention_mask"].clone()
                for b in range(ids.size(0)):
                    toks = tokens_from_ids(tokenizer, ids[b])
                    for j, tok in enumerate(toks):
                        if cluster_map.get(tok) == cid:
                            ids[b, j] = mask_token_id
                ids=ids.to(device); att=att.to(device)
                logits = model(input_ids=ids, attention_mask=att)
            else:
                ids = batch["input_ids"].clone()
                att = batch["attention_mask"].clone()
                cm  = batch["chunk_mask"].clone()
                B,C,L = ids.shape
                for b in range(B):
                    for c in range(C):
                        toks = tokens_from_ids(tokenizer, ids[b,c])
                        for j, tok in enumerate(toks):
                            if cluster_map.get(tok) == cid:
                                ids[b, c, j] = mask_token_id
                ids=ids.to(device); att=att.to(device); cm=cm.to(device)
                logits = model(input_ids=ids, attention_mask=att, chunk_mask=cm)
            pred = logits.argmax(1).cpu().numpy()
            y_true.append(batch["labels"].cpu().numpy()); y_pred.append(pred)
        y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
        drops[cid] = float(baseF1 - macro_scores(y_true, y_pred)[2])
    return drops, baseF1

# -------------- stability aggregation --------------
def aggregate_stability(rows: List[Dict], kfolds: int, min_freq: float, min_sign: float, topk: int=0):
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    df["sign_bin"] = np.where(df["drop"] >= 0.0, 1, -1)
    freq = df.groupby("feature")["fold"].nunique().rename("n_folds")
    selected_frac = (freq / float(kfolds)).rename("selected_frac")
    scounts = df.groupby(["feature","sign_bin"])["fold"].nunique().unstack(fill_value=0)
    p1 = scounts[1] if 1 in scounts.columns else pd.Series(0, index=scounts.index)
    pneg = scounts[-1] if -1 in scounts.columns else pd.Series(0, index=scounts.index)
    sign_cons = (scounts.max(1) / scounts.sum(1).replace(0, np.nan)).fillna(0.0).rename("sign_consistency")
    maj_sign = pd.Series(np.where(p1 >= pneg, 1, -1), index=scounts.index)
    mag = df.groupby("feature")["drop"].agg(mean_abs_drop=lambda s: float(np.mean(np.abs(s))),
                                            mean_drop=lambda s: float(np.mean(s)))
    stable = pd.concat([freq, selected_frac, sign_cons, mag], axis=1).reset_index()
    stable["majority_sign"] = stable["feature"].map(maj_sign).astype(int)
    stable["score"] = stable["selected_frac"] * stable["sign_consistency"] * stable["mean_abs_drop"]
    filt = stable[(stable["selected_frac"] >= min_freq) & (stable["sign_consistency"] >= min_sign)].copy()
    filt = filt.sort_values(["score","selected_frac","mean_abs_drop"], ascending=False)
    if topk and topk > 0:
        filt = filt.head(int(topk))
    stable = stable.sort_values(["score","selected_frac","mean_abs_drop"], ascending=False)
    return stable, filt

# -------------- config loader --------------
def load_bert_settings(json_path: str):
    with open(json_path, "r", encoding="utf8") as f:
        cfg = json.load(f)
    return {
        "CSV_PATH": cfg.get("CSV_PATH", "preprocessed_dataset_all2.csv"),
        "HIERARCHICAL": bool(cfg.get("HIERARCHICAL", False)),
        "MAX_SEQ_LEN": int(cfg.get("MAX_SEQ_LEN", 512)),
        "MAX_CHUNKS": int(cfg.get("MAX_CHUNKS", 12)),
        "OVERLAP_TOKENS": int(cfg.get("OVERLAP_TOKENS", 0)),
        "TRAIN_BATCH_SIZE": int(cfg.get("TRAIN_BATCH_SIZE", 8)),
        "EVAL_BATCH_SIZE": int(cfg.get("EVAL_BATCH_SIZE", 8)),
        "NUM_EPOCHS": int(cfg.get("NUM_EPOCHS", 4)),
        "LEARNING_RATE": float(cfg.get("LEARNING_RATE", 2e-5)),
        "WARMUP_RATIO": float(cfg.get("WARMUP_RATIO", 0.1)),
        "D_USER_LATENT": int(cfg.get("D_USER_LATENT", 16)),
        "BERT_MODEL_NAME": cfg.get("BERT_MODEL_NAME", "bert-base-uncased"),
        "SEED": int(cfg.get("SEED", 555)),
    }

# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser(description="JSON‑driven NN pipeline: (1) MLP (users only), (2) BERT T‑/H‑BERT text‑only or fusion; with feature importance and ablations.")
    ap.add_argument("--model", choices=["mlp","bert"], required=True)
    ap.add_argument("--fusion", action="store_true", help="BERT+users fusion if --model bert")
    ap.add_argument("--data", required=True, help="CSV with columns: username,label,aggr_text and user features")
    ap.add_argument("--cluster", type=str, default="", help="token→cluster map for BERT saliency/occlusion")
    ap.add_argument("--outdir", default="runs_nn")
    ap.add_argument("--bert_settings", type=str, default="", help="JSON file with T‑BERT or H‑BERT settings")
    ap.add_argument("--perm_repeats", type=int, default=20)
    ap.add_argument("--occlude_topk", type=int, default=20, help="Number of top clusters (by saliency) to occlude")
    ap.add_argument("--stable_min_freq", type=float, default=0.60)
    ap.add_argument("--stable_min_sign", type=float, default=0.80)
    ap.add_argument("--stable_topk_users", type=int, default=0)
    ap.add_argument("--stable_topk_message", type=int, default=0)
    args = ap.parse_args()

    # ---------- load CSV ----------
    df = pd.read_csv(args.data)
    if "label" not in df.columns or "aggr_text" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'aggr_text'.")
    df["label"] = df["label"].map({"reliable":0,"unreliable":1}).astype(int)

    # ---------- output dirs ----------
    for d in ["cv","diagnostics","stable","model"]:   # ← removed 'models'
        ensure_dir(os.path.join(args.outdir, d))
    MODEL_DIR = os.path.join(args.outdir, "model")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================= MLP (users only) =======================
    if args.model == "mlp":
        user_cols = [c for c in NEEDED_USER_COLS if c in df.columns]
        if not user_cols:
            raise RuntimeError("No required user columns found in CSV.")
        set_all_seeds(555)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=555)
        per_rows=[]; stab_users=[]; lofo_rows_all=[]
        fold=0
        for tr_idx, te_idx in skf.split(df, df["label"]):
            fold += 1
            df_tr, df_te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            y_tr, y_te = df_tr["label"].values, df_te["label"].values
            X_tr, imp, sc, feat_names = build_user_matrix(df_tr, user_cols, fit_on=df_tr)
            X_te, _, _, _ = build_user_matrix(df_te, user_cols, imp=imp, sc=sc)
            # train/dev split
            X_tr2, X_dv, y_tr2, y_dv = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=555)
            model = MLP(X_tr.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
            bs = 64
            dl_tr = DataLoader(TabularDataset(X_tr2, y_tr2), batch_size=bs, shuffle=True)
            dl_dv = DataLoader(TabularDataset(X_dv, y_dv), batch_size=bs, shuffle=False)
            # early stopping on dev F1
            best_f1=-1; best_state=None
            for ep in range(40):
                model.train()
                for xb, yb in dl_tr:
                    xb=xb.to(device); yb=yb.to(device)
                    opt.zero_grad(); loss=F.cross_entropy(model(xb), yb); loss.backward(); opt.step()
                # dev
                model.eval(); y_true=[]; y_pred=[]
                for xb, yb in dl_dv:
                    xb=xb.to(device)
                    pred=model(xb).argmax(1).cpu().numpy()
                    y_true.append(yb.numpy()); y_pred.append(pred)
                y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
                f1=macro_scores(y_true,y_pred)[2]
                if f1>best_f1:
                    best_f1=f1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state)
            # test
            dl_te = DataLoader(TabularDataset(X_te, y_te), batch_size=bs, shuffle=False)
            @torch.no_grad()
            def eval_te():
                model.eval(); y_true=[]; y_pred=[]
                for xb, yb in dl_te:
                    xb=xb.to(device); pred=model(xb).argmax(1).cpu().numpy()
                    y_true.append(yb.numpy()); y_pred.append(pred)
                y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
                return (*macro_scores(y_true,y_pred), y_true, y_pred)
            P,R,F1, y_true, y_pred = eval_te()
            print(f"\n[FOLD {fold}][MLP-Users] P={P:.4f} R={R:.4f} F1={F1:.4f}")
            print(classification_report(y_true, y_pred, digits=4, zero_division=0))
            per_rows.append({"fold":fold,"family":"MLP-Users","P":P,"R":R,"F1":F1})

            # --- Save best model for this fold ---
            mlp_ckpt_path = os.path.join(MODEL_DIR, f"mlp_users_fold{fold}.pt")
            save_checkpoint(
                model,
                mlp_ckpt_path,
                extra={
                    "arch": {"type":"MLP", "in_dim": X_tr.shape[1], "hidden":[128,64,32], "drop":0.1, "n_classes":2},
                    "feature_names": feat_names,
                    "imputer": imp,
                    "scaler": sc,
                    "seed": 555,
                    "label_mapping": {"reliable":0, "unreliable":1},
                    "metrics": {"P":P, "R":R, "F1":F1},
                    "family": "MLP-Users",
                    "fold": fold,
                }
            )

            # permutation importance
            perm_rows, baseF1 = perm_importance_user(model, X_te, y_te, feat_names, repeats=args.perm_repeats, device=device, batch_size=bs)
            pd.DataFrame(perm_rows).to_csv(os.path.join(args.outdir,"diagnostics",f"users_perm_fold{fold}.csv"), index=False)
            for r in perm_rows:
                stab_users.append({"fold":fold,"feature":r["feature"],"drop":r["drop_mean"]})
            # LOFO ablation
            lofo_rows = lofo_user_ablation(model, X_te, y_te, feat_names, device=device, batch_size=bs)
            pd.DataFrame(lofo_rows).to_csv(os.path.join(args.outdir,"diagnostics",f"users_lofo_fold{fold}.csv"), index=False)
            for r in lofo_rows:
                r2 = r.copy(); r2["fold"]=fold
                lofo_rows_all.append(r2)

        # stability aggregation and diagnostics
        users_all, users_filt = aggregate_stability(stab_users, 5, args.stable_min_freq, args.stable_min_sign, args.stable_topk_users)
        users_all.to_csv(os.path.join(args.outdir,"stable","stable_users_all.csv"), index=False)
        (users_filt if users_filt is not None else pd.DataFrame()).to_csv(os.path.join(args.outdir,"stable","stable_users_filtered.csv"), index=False)
        save_agg_from_rows(rows=stab_users, key="feature", val="drop",
                           out_csv=os.path.join(args.outdir,"diagnostics","users_perm_agg.csv"),
                           sort_desc=True, rename_prefix="drop")
        save_agg_from_rows(rows=lofo_rows_all, key="feature", val="drop_F1",
                           out_csv=os.path.join(args.outdir,"diagnostics","users_lofo_agg.csv"),
                           sort_desc=True, rename_prefix="drop_F1", extra_means=["base_F1","abl_F1"])

        # ==== final summary print (formatted) ====
        summary_df, summary_fmt = save_cv_summary(per_rows, args.outdir)
        print("\n=== 5-fold CV Summary (MLP-Users) ===")
        if summary_fmt is not None and not summary_fmt.empty:
            print(summary_fmt.to_string(index=False))
        else:
            print("No results to summarize.")

    # ======================= BERT (text-only or fusion) =======================
    if args.model == "bert":
        if AutoTokenizer is None:
            raise RuntimeError("transformers must be installed to run BERT (--model bert).")
        if not args.bert_settings:
            raise RuntimeError("Please provide --bert_settings pointing to tbert_settings.json or hbert_settings.json.")
        bert_cfg = load_bert_settings(args.bert_settings)
        set_all_seeds(bert_cfg["SEED"])
        tokenizer = AutoTokenizer.from_pretrained(bert_cfg["BERT_MODEL_NAME"], use_fast=True)

        if not args.cluster:
            raise RuntimeError("Please provide --cluster (token→cluster map) to run BERT interpretability.")
        CLUSTER_MAP, NCLUS = load_cluster_map(args.cluster)

        user_cols = [c for c in NEEDED_USER_COLS if c in df.columns]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=bert_cfg["SEED"])
        per_rows=[]; saliency_rows=[]; occl_rows=[]; user_imp_rows=[]
        fold=0
        for tr_idx, te_idx in skf.split(df, df["label"]):
            fold += 1
            df_tr, df_te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            y_tr, y_te = df_tr["label"].values, df_te["label"].values

            ds_tr = BERTTextDataset(df_tr, tokenizer, bert_cfg["MAX_SEQ_LEN"], bert_cfg["HIERARCHICAL"], bert_cfg["OVERLAP_TOKENS"], bert_cfg["MAX_CHUNKS"])
            ds_te = BERTTextDataset(df_te, tokenizer, bert_cfg["MAX_SEQ_LEN"], bert_cfg["HIERARCHICAL"], bert_cfg["OVERLAP_TOKENS"], bert_cfg["MAX_CHUNKS"])
            coll = collate_hbert if bert_cfg["HIERARCHICAL"] else collate_tbert
            dl_tr = DataLoader(ds_tr, batch_size=bert_cfg["TRAIN_BATCH_SIZE"], shuffle=True, collate_fn=coll)
            dl_te = DataLoader(ds_te, batch_size=bert_cfg["EVAL_BATCH_SIZE"],  shuffle=False, collate_fn=coll)

            if args.fusion:
                Xu_tr, imp, sc, feat_names = build_user_matrix(df_tr, user_cols, fit_on=df_tr)
                Xu_te, _, _, _           = build_user_matrix(df_te, user_cols, imp=imp, sc=sc)
                model = BertFusion(bert_cfg["BERT_MODEL_NAME"], user_dim=Xu_tr.shape[1], user_hidden=bert_cfg["D_USER_LATENT"],
                                   n_classes=2, dropout=0.1, hierarchical=bert_cfg["HIERARCHICAL"]).to(device)
            else:
                model = TextBERT(bert_cfg["BERT_MODEL_NAME"], n_classes=2, dropout=0.1, hierarchical=bert_cfg["HIERARCHICAL"]).to(device)

            steps = max(1, len(dl_tr) * bert_cfg["NUM_EPOCHS"])
            opt = AdamW(model.parameters(), lr=bert_cfg["LEARNING_RATE"])
            sch = get_linear_schedule_with_warmup(opt, int(bert_cfg["WARMUP_RATIO"]*steps), steps)

            for ep in range(bert_cfg["NUM_EPOCHS"]):
                model.train()
                for batch in dl_tr:
                    y = batch["labels"].to(device)
                    ux = None
                    if args.fusion:
                        rows = batch["row_idx"].cpu().numpy()
                        ux = torch.tensor(Xu_tr[rows], dtype=torch.float32, device=device)
                    if not bert_cfg["HIERARCHICAL"]:
                        ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device)
                        logits = model(input_ids=ids, attention_mask=att, user_x=ux) if args.fusion \
                                 else model(input_ids=ids, attention_mask=att)
                    else:
                        ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device); cm = batch["chunk_mask"].to(device)
                        logits = model(input_ids=ids, attention_mask=att, chunk_mask=cm, user_x=ux) if args.fusion \
                                 else model(input_ids=ids, attention_mask=att, chunk_mask=cm)
                    loss = F.cross_entropy(logits, y)
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()

            if args.fusion:
                @torch.no_grad()
                def eval_fusion():
                    model.eval(); y_true=[]; y_pred=[]
                    for batch in dl_te:
                        rows = batch["row_idx"].cpu().numpy()
                        ux = torch.tensor(Xu_te[rows], dtype=torch.float32, device=device)
                        if not bert_cfg["HIERARCHICAL"]:
                            ids=batch["input_ids"].to(device); att=batch["attention_mask"].to(device)
                            logits=model(input_ids=ids, attention_mask=att, user_x=ux)
                        else:
                            ids=batch["input_ids"].to(device); att=batch["attention_mask"].to(device); cm=batch["chunk_mask"].to(device)
                            logits=model(input_ids=ids, attention_mask=att, chunk_mask=cm, user_x=ux)
                        pred=logits.argmax(1).cpu().numpy()
                        y_true.append(batch["labels"].cpu().numpy()); y_pred.append(pred)
                    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
                    return (*macro_scores(y_true, y_pred), y_true, y_pred)
                P,R,F1, y_true, y_pred = eval_fusion()
            else:
                P,R,F1, y_true, y_pred = eval_text(model, dl_te, device, bert_cfg["HIERARCHICAL"])

            mode = f"BERT{'-Fusion' if args.fusion else ''} ({'H' if bert_cfg['HIERARCHICAL'] else 'T'})"
            print(f"\n[FOLD {fold}][{mode}] P={P:.4f} R={R:.4f} F1={F1:.4f}")
            print(classification_report(y_true, y_pred, digits=4, zero_division=0))
            per_rows.append({"fold":fold,"family":mode,"P":P,"R":R,"F1":F1})

            # save model per fold
            mode_short = f"{'h' if bert_cfg['HIERARCHICAL'] else 't'}bert{'-fusion' if args.fusion else ''}"
            bert_ckpt_path = os.path.join(MODEL_DIR, f"{mode_short}_fold{fold}.pt")
            extra = {
                "arch": {
                    "type": "BertFusion" if args.fusion else "TextBERT",
                    "bert_model_name": bert_cfg["BERT_MODEL_NAME"],
                    "hierarchical": bool(bert_cfg["HIERARCHICAL"]),
                    "n_classes": 2,
                    "dropout": 0.1,
                    "d_user_latent": (bert_cfg["D_USER_LATENT"] if args.fusion else None),
                    "user_dim": (Xu_tr.shape[1] if args.fusion else None),
                },
                "label_mapping": {"reliable":0, "unreliable":1},
                "metrics": {"P":P, "R":R, "F1":F1},
                "family": mode,
                "fold": fold,
                "seed": bert_cfg["SEED"],
            }
            if args.fusion:
                extra.update({"user_feature_names": feat_names, "imputer": imp, "scaler": sc})
            save_checkpoint(model, bert_ckpt_path, extra=extra)

            # interpretability diagnostics
            sal_map = bert_grad_times_input(model, tokenizer, dl_te, device, bert_cfg["HIERARCHICAL"], CLUSTER_MAP)
            sal_items = sorted(sal_map.items(), key=lambda kv: kv[1], reverse=True)
            pd.DataFrame([{"fold":fold,"cluster":cid,"saliency":val} for cid,val in sal_items]).to_csv(
                os.path.join(args.outdir,"diagnostics",f"bert_saliency_fold{fold}.csv"), index=False)
            for cid,val in sal_items:
                saliency_rows.append({"fold":fold,"feature":f"topic:{cid}","drop":float(val)})

            test_cids = [cid for cid,_ in sal_items[:max(1, args.occlude_topk)]]
            drops_map, baseF1 = bert_cluster_occlusion(model, tokenizer, dl_te, device, bert_cfg["HIERARCHICAL"], test_cids, CLUSTER_MAP)
            pd.DataFrame([{"fold":fold,"cluster":cid,"drop_F1":drops_map[cid]} for cid in test_cids]).to_csv(
                os.path.join(args.outdir,"diagnostics",f"bert_occlusion_fold{fold}.csv"), index=False)
            for cid in test_cids:
                occl_rows.append({"fold":fold,"feature":f"topic:{cid}","drop":drops_map[cid]})

            if args.fusion and user_cols:
                from numpy.random import default_rng
                rng = default_rng(2024)
                baseF1 = F1
                for j,col in enumerate(user_cols):
                    drops=[]
                    for _ in range(args.perm_repeats):
                        Xp = Xu_te.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                        y_true_perm=[]; y_pred_perm=[]
                        model.eval()
                        for batch in dl_te:
                            rows = batch["row_idx"].cpu().numpy()
                            ux = torch.tensor(Xp[rows], dtype=torch.float32, device=device)
                            if not bert_cfg["HIERARCHICAL"]:
                                ids=batch["input_ids"].to(device); att=batch["attention_mask"].to(device)
                                logits=model(input_ids=ids, attention_mask=att, user_x=ux)
                            else:
                                ids=batch["input_ids"].to(device); att=batch["attention_mask"].to(device); cm=batch["chunk_mask"].to(device)
                                logits=model(input_ids=ids, attention_mask=att, chunk_mask=cm, user_x=ux)
                            pred=logits.argmax(1).cpu().numpy()
                            y_true_perm.append(batch["labels"].cpu().numpy()); y_pred_perm.append(pred)
                        y_true_perm=np.concatenate(y_true_perm); y_pred_perm=np.concatenate(y_pred_perm)
                        drops.append(float(baseF1 - macro_scores(y_true_perm,y_pred_perm)[2]))
                    user_imp_rows.append({"fold":fold,"feature":f"user:{col}","drop":float(np.mean(drops))})

        # stability & diagnostics
        msg_all, msg_filt = aggregate_stability(occl_rows, 5, args.stable_min_freq, args.stable_min_sign, args.stable_topk_message)
        msg_all.to_csv(os.path.join(args.outdir,"stable","stable_message_clusters_all.csv"), index=False)
        (msg_filt if msg_filt is not None else pd.DataFrame()).to_csv(os.path.join(args.outdir,"stable","stable_message_clusters_filtered.csv"), index=False)
        if args.fusion and user_imp_rows:
            users_all, users_filt = aggregate_stability(user_imp_rows, 5, args.stable_min_freq, args.stable_min_sign, args.stable_topk_users)
            users_all.to_csv(os.path.join(args.outdir,"stable","stable_users_all.csv"), index=False)
            (users_filt if users_filt is not None else pd.DataFrame()).to_csv(os.path.join(args.outdir,"stable","stable_users_filtered.csv"), index=False)

        pd.DataFrame(saliency_rows).to_csv(os.path.join(args.outdir,"stable","saliency_clusters_raw.csv"), index=False)

        # aggregated interpretability
        sal_df = pd.DataFrame(saliency_rows)
        if not sal_df.empty:
            agg = sal_df.rename(columns={"drop":"saliency"}).assign(cluster=lambda d: d["feature"].str.replace("topic:","", regex=False).astype(int)) \
                        .groupby("cluster")["saliency"].agg(["mean","std","count","sum"]).reset_index() \
                        .rename(columns={"mean":"saliency_mean","std":"saliency_std","count":"n_folds","sum":"saliency_sum"}) \
                        .sort_values("saliency_mean", ascending=False)
            agg.to_csv(os.path.join(args.outdir,"diagnostics","bert_saliency_agg.csv"), index=False)
            log.info("Saved aggregated saliency → %s", os.path.join(args.outdir,"diagnostics","bert_saliency_agg.csv"))
        else:
            pd.DataFrame(columns=["cluster","saliency_mean","saliency_std","n_folds","saliency_sum"]).to_csv(
                os.path.join(args.outdir,"diagnostics","bert_saliency_agg.csv"), index=False)

        occ_df = pd.DataFrame(occl_rows)
        if not occ_df.empty:
            agg = occ_df.assign(cluster=lambda d: d["feature"].str.replace("topic:","", regex=False).astype(int)) \
                        .groupby("cluster")["drop"].agg(["mean","std","count"]).reset_index() \
                        .rename(columns={"mean":"drop_F1_mean","std":"drop_F1_std","count":"n_folds"}) \
                        .sort_values("drop_F1_mean", ascending=False)
            agg.to_csv(os.path.join(args.outdir,"diagnostics","bert_occlusion_agg.csv"), index=False)
            log.info("Saved aggregated occlusion → %s", os.path.join(args.outdir,"diagnostics","bert_occlusion_agg.csv"))
        else:
            pd.DataFrame(columns=["cluster","drop_F1_mean","drop_F1_std","n_folds"]).to_csv(
                os.path.join(args.outdir,"diagnostics","bert_occlusion_agg.csv"), index=False)

        if args.fusion:
            save_agg_from_rows(rows=user_imp_rows, key="feature", val="drop",
                               out_csv=os.path.join(args.outdir,"diagnostics","fusion_user_perm_agg.csv"),
                               sort_desc=True, rename_prefix="drop")

        # ==== final summary print (formatted) ====
        summary_df, summary_fmt = save_cv_summary(per_rows, args.outdir)
        print("\n=== 5-fold CV Summary (BERT) ===")
        if summary_fmt is not None and not summary_fmt.empty:
            print(summary_fmt.to_string(index=False))
        else:
            print("No results to summarize.")

if __name__ == "__main__":
    main()
