import pandas as pd
import re

# --- User settings -----------------------------------------

# Path to your data (xlsx or csv)
INPUT_PATH = 'fetched_data.xlsx'  # or 'fetched_data.csv'

# Column names in your file
USER_COL = 'user_id'
TIME_COL = 'tweet_created_at'
TEXT_COL = 'text'   # or wherever the tweet text lives

# (Optional) list of news‐source handles (lowercase, without @)
NEWS_SOURCES = {
    'cnn', 'bbcnews', 'nytimes', 'guardian',  # …etc.
}

# Output files
OUTPUT_CLEAN = 'filtered_tweets.csv'
OUTPUT_ARCHIVE = 'first_reposts.csv'

# -----------------------------------------------------------

# 1. Load data
if INPUT_PATH.endswith('.xlsx'):
    df = pd.read_excel(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH)

# 2. Normalize types
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str)

# Helper: detect RT and extract the retweeted handle
def extract_rt_handle(text):
    m = re.match(r'RT @([A-Za-z0-9_]+):', text)
    return m.group(1).lower() if m else None

df['rt_handle'] = df[TEXT_COL].apply(extract_rt_handle)
df['is_rt'] = df['rt_handle'].notna()

# 3. For each user: find first RT *of any* handle in NEWS_SOURCES
first_rts = (
    df[df['is_rt'] & df['rt_handle'].isin(NEWS_SOURCES)]
      .sort_values(TIME_COL)
      .groupby(USER_COL)
      .first()
      .reset_index()
      .rename(columns={TIME_COL: 'first_rt_time'})
      [[USER_COL, 'first_rt_time', 'rt_handle']]
)

# 4. Count original tweets before that time
#    Merge to get each user’s first RT timestamp
df = df.merge(first_rts[[USER_COL, 'first_rt_time']], on=USER_COL, how='left')

#    Only users who ever RT’ed a news source
df = df[~df['first_rt_time'].isna()].copy()

#    Compute count of non‐RT tweets before first RT
pre_rt = (
    df[~df['is_rt'] & (df[TIME_COL] < df['first_rt_time'])]
      .groupby(USER_COL)
      .size()
      .rename('orig_before_rt')
)

first_rts = first_rts.set_index(USER_COL).join(pre_rt, how='left').fillna(0)
first_rts['orig_before_rt'] = first_rts['orig_before_rt'].astype(int)

# 5. Filter users with ≥10 original tweets before first repost
keep_users = first_rts[first_rts['orig_before_rt'] >= 10].index

# Archive: all first reposts (including the handle they RT’ed)
archive = first_rts.loc[keep_users].reset_index()
archive.to_csv(OUTPUT_ARCHIVE, index=False)

# Cleaned dataset: keep only tweets by users in keep_users
cleaned = df[df[USER_COL].isin(keep_users)].drop(columns=['first_rt_time'])
cleaned.to_csv(OUTPUT_CLEAN, index=False)

print(f"Kept {len(keep_users)} users; "
      f"filtered dataset → {OUTPUT_CLEAN}; "
      f"archive of first RTs → {OUTPUT_ARCHIVE}")
