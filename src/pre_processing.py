import pandas as pd
import numpy as np
import re
import sys
from sklearn.model_selection import train_test_split

# --- User settings -----------------------------------------
INPUT_PATH   = 'cleaned_dataset.csv'
OUTPUT_TRAIN = 'train.csv'
OUTPUT_DEV   = 'de1.csv'
OUTPUT_TEST  = 'te1.csv'

USER_COL  = 'user_id'
TEXT_COL  = 'tweet_text'
LABEL_COL = 'LABEL'   # should be 0 (reliable) or 1 (unreliable)

TRAIN_RATIO = 0.7
DEV_RATIO   = 0.2
TEST_RATIO  = 0.1

# Sanity check
if abs((TRAIN_RATIO + DEV_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    sys.exit("ERROR: TRAIN+DEV+TEST ratios must sum to 1.0")
# -----------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lowercase + strip URLs + collapse whitespace."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1) Load & preprocess
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} tweets from {df[USER_COL].nunique()} users.")
    df['text'] = df[TEXT_COL].astype(str).apply(preprocess_text)

    # 2) Map string labels to binary if needed
    label_map = {'reliable': 0, 'unreliable': 1}
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].map(label_map)
        print("Mapped LABEL strings to 0/1.")

    # 3) Build user→label map
    user_labels = df.groupby(USER_COL)[LABEL_COL].first().astype(int)

    # 3b) Extract arrays for stratified split
    users  = user_labels.index.values
    labels = user_labels.values

    # 4) First split: train vs temp (dev+test)
    n_temp = DEV_RATIO + TEST_RATIO
    train_users, temp_users, train_labels, temp_labels = train_test_split(
        users,
        labels,
        test_size=n_temp,
        stratify=labels,
        random_state=42
    )

    # 5) Second split: dev vs test from temp
    dev_size = DEV_RATIO / (DEV_RATIO + TEST_RATIO)
    dev_users, test_users = train_test_split(
        temp_users,
        test_size=1-dev_size,
        stratify=temp_labels,
        random_state=42
    )

    # 6) Report splits
    def report_split(name, user_list):
        dist = user_labels.loc[user_list].value_counts().to_dict()
        print(f"{name}: {len(user_list)} users → label dist {dist}")

    report_split("TRAIN", train_users)
    report_split("DEV",   dev_users)
    report_split("TEST",  test_users)

    # 7) Slice original DF and save
    for split_name, user_set, out_path in [
        ('train', train_users, OUTPUT_TRAIN),
        ('dev',   dev_users,   OUTPUT_DEV),
        ('test',  test_users,  OUTPUT_TEST)
    ]:
        sub = df[df[USER_COL].isin(user_set)].copy()
        print(f"  • {split_name.upper()}: {len(sub)} tweets → {out_path}")
        sub[[USER_COL, 'text', LABEL_COL]].to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
