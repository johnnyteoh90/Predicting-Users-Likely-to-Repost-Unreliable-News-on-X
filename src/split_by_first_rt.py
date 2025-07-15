import pandas as pd

# --- User settings -----------------------------------------

# Path to your data (xlsx or csv)
INPUT_PATH = 'fetched_data.xlsx'   # or 'fetched_data.csv'

# Column names in your file
USER_COL = 'user_id'
TIME_COL = 'tweet_created_at'
TEXT_COL = 'tweet_text'   # or wherever the tweet text lives

# How many retweets to keep per user
MAX_RTS = 20

# Output files
OUTPUT_KEPT    = 'filtered_kept_users.csv'
OUTPUT_EXCL    = 'filtered_excluded_users.csv'

# -----------------------------------------------------------

# 1. Load data
if INPUT_PATH.lower().endswith('.xlsx'):
    df = pd.read_excel(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH)

# 2. Normalize types
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df[TEXT_COL]  = df[TEXT_COL].astype(str)

# 3. Identify retweets
df['is_rt'] = df[TEXT_COL].str.startswith('RT @')

# 4. Sort by time so that cumcounts are chronological
df = df.sort_values(TIME_COL)

# 5. Extract only the RT tweets, rank them per user, and pick first MAX_RTS
rt_only = df[df['is_rt']].copy()
rt_only['rt_rank'] = rt_only.groupby(USER_COL).cumcount() + 1
first_rts = rt_only[rt_only['rt_rank'] <= MAX_RTS]

# 6. Compute cutoff per user = timestamp of their Nth retweet
cutoff = first_rts.groupby(USER_COL)[TIME_COL] \
                  .max() \
                  .rename('cutoff_time')

# 7. Merge cutoff into the full dataframe
df = df.merge(cutoff, on=USER_COL, how='left')

# 8. Bring rt_rank back into df (NaN for non-RT or RT beyond MAX_RTS)
df = df.merge(
    first_rts[[USER_COL, TIME_COL, 'rt_rank']],
    on=[USER_COL, TIME_COL],
    how='left'
)

# 9. Filter:
#    - Keep RTs whose rank ≤ MAX_RTS
#    - Keep originals (is_rt=False) with timestamp < cutoff_time
cond_rt   = df['rt_rank'].between(1, MAX_RTS)
cond_orig = (~df['is_rt']) & (df[TIME_COL] < df['cutoff_time'])
filtered  = df[cond_rt | cond_orig].copy()

# 10. Count original tweets per user in the filtered set
orig_counts = (
    filtered[~filtered['is_rt']]
      .groupby(USER_COL)
      .size()
      .rename('orig_count')
)

# 11. Split users into kept (orig_count ≥ 10) and excluded (< 10)
orig_counts = orig_counts.reset_index()
kept_users  = orig_counts[orig_counts['orig_count'] >= 10][USER_COL]
excl_users  = orig_counts[orig_counts['orig_count'] < 10][USER_COL]

# 12. Build final DataFrames
kept_df = filtered[filtered[USER_COL].isin(kept_users)].copy()
excl_df = filtered[filtered[USER_COL].isin(excl_users)].copy()

# 13. Sort by user and time
kept_df = kept_df.sort_values([USER_COL, TIME_COL])
excl_df = excl_df.sort_values([USER_COL, TIME_COL])

# 14. Drop helper columns and save
for df_out, path in [(kept_df, OUTPUT_KEPT), (excl_df, OUTPUT_EXCL)]:
    df_out.drop(columns=['rt_rank', 'cutoff_time'], inplace=True)
    df_out.to_csv(path, index=False)
    print(f"Saved {len(df_out)} tweets → {path}")
