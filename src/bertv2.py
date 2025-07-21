#!/usr/bin/env python3
import logging
import sys
import re

# Ensure accelerate is installed for PyTorch Trainer
try:
    import accelerate  # noqa: F401
except ImportError:
    raise ImportError(
        "Using the Trainer with PyTorch requires accelerate>=0.20.1.\n"
        "Please run `pip install accelerate -U` or `pip install transformers[torch]`"
    )

import pandas as pd
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from transformers import (
    BertTokenizerFast,
    BertModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import torch.nn as nn
from torch.utils.data import Dataset

# — 0) Logging & NLTK setup —————————————————————————————————————————————————
logging.basicConfig(
    filename="error.log",
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger()

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# — 1) Text preprocessing —————————————————————————————————————————————————
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"@\w+",    " USR ", text)
    tokens = re.findall(r"\w+", text)
    return " ".join([t for t in tokens if t not in STOPWORDS])

# — 2) Custom Dataset ———————————————————————————————————————————————————
class MetaTextDataset(Dataset):
    def __init__(self, df, tokenizer, num_cols, max_len=512):
        self.texts = df['text'].tolist()
        self.meta  = df[num_cols].astype(np.float32).values
        self.labels= df['labels'].astype(int).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['meta']   = torch.tensor(self.meta[idx], dtype=torch.float32)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# — 3) Model with Meta Fusion —————————————————————————————————————————————————
class BertWithMeta(nn.Module):
    def __init__(self, model_name, meta_dim, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size + meta_dim, num_labels)

    def forward(self, input_ids, attention_mask, meta, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.pooler_output.to(torch.float32)
        meta    = meta.to(pooled.dtype)
        x = torch.cat([pooled, meta], dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

# — 4) Main pipeline with 3-seed runs —————————————————————————————————————————
def main():
    try:
        # 4.1) Load & preprocess input
        df = pd.read_csv('preprocessed_user_data.csv')
        if 'label' in df.columns and 'labels' not in df.columns:
            df.rename(columns={'label':'labels'}, inplace=True)
        if 'labels' not in df.columns:
            raise KeyError("CSV must contain 'labels' column with 0/1 values.")
        df['text'] = df['text'].astype(str).map(preprocess)

        # 4.2) Scale meta features once
        num_cols = [
            'num_tweets', 'account_age_years', 'total_posts',
            'following_count','followers_count','posting_frequency'
        ]
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols].values)

        # 4.3) Create splits (fixed split across seeds)
        X_idxs, y = df.index.values, df['labels'].values
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=555)
        tr_idx, tmp_idx = next(s1.split(X_idxs, y))
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=555)
        dev_sub, test_sub = next(s2.split(tmp_idx, y[tmp_idx]))
        dv_idx, te_idx = tmp_idx[dev_sub], tmp_idx[test_sub]
        df_train, df_dev, df_test = df.loc[tr_idx], df.loc[dv_idx], df.loc[te_idx]

        print("Class distribution:")
        print("Train:", df_train['labels'].value_counts().to_dict())
        print("Dev:  ", df_dev['labels'].value_counts().to_dict())
        print("Test: ", df_test['labels'].value_counts().to_dict())

        # 4.4) Tokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # 4.5) Repeat runs
        seeds = [555, 666, 777]
        metrics = {'precision':[], 'recall':[], 'f1':[]}

        for seed in seeds:
            print(f"\n===== Run with seed={seed} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Datasets
            train_ds = MetaTextDataset(df_train, tokenizer, num_cols)
            dev_ds   = MetaTextDataset(df_dev,   tokenizer, num_cols)
            test_ds  = MetaTextDataset(df_test,  tokenizer, num_cols)

            # Model
            model = BertWithMeta('bert-base-uncased', meta_dim=len(num_cols))

            # Args
            args = TrainingArguments(
                output_dir=f'outputs/seed_{seed}',
                evaluation_strategy='epoch', save_strategy='epoch',
                num_train_epochs=4, per_device_train_batch_size=8,
                per_device_eval_batch_size=8, learning_rate=2e-5,
                logging_steps=50, load_best_model_at_end=True,
                metric_for_best_model='f1', greater_is_better=True,
                seed=seed
            )

            # Trainer
            trainer = Trainer(
                model=model, args=args,
                train_dataset=train_ds, eval_dataset=dev_ds,
                compute_metrics=lambda p: dict(zip(
                    ['precision','recall','f1'],
                    precision_recall_fscore_support(
                        p.label_ids, p.predictions.argmax(-1),
                        average='macro', zero_division=0
                    )[:3]
                ))
            )

            # Train & Evaluate
            trainer.train()
            preds = trainer.predict(test_ds)
            y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)
            p, r, f = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )[:3]
            print(f"Results seed {seed}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['f1'].append(f)

        # 4.6) Aggregate
        print("\n=== 3-run averages ± std ===")
        for m in metrics:
            arr = np.array(metrics[m])
            print(f"Macro {m.title()}: {arr.mean():.4f} ± {arr.std(ddof=1):.4f}")

    except Exception as e:
        logger.exception("Fatal error in script")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
