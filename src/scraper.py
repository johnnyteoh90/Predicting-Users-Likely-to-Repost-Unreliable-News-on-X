import tweepy
import pandas as pd
import time

# ----------------------
# 1. Authentication
# ----------------------
API_KEY            = "LhwNAB800EUy00GPY3RWDLP9q"
API_SECRET_KEY     = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN       = "AAAAAAAAAAAAAAAAAAAAACfP0QEAAAAArdPGFlxCc5Y67uj6rCePkVpAgHY%3DQGD9R8PE84Qn7iwLWkUY27o65GduujJOgPMwkOGsGGnTEPDqK6"
ACCESS_TOKEN       = "1861791142091657216-5z2nwwVmqVR3rbgOPViMSGPFdRp01O"
ACCESS_TOKEN_SECRET= "PKmIB7X1Q629HV3YL210mvRiBe8pTP0l50S4aLmyrWkJv"

# ----------------------
# 2. Initialize client
# ----------------------
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

# ----------------------
# 3. Load input CSV
# ----------------------
infile = "raw_data.csv"
df = pd.read_csv(infile)
total_users = len(df)
print(f"[INFO] Loaded {total_users} users from '{infile}'")

# ----------------------
# 4. Setup for progress/temp saving
# ----------------------
tweet_counts = {}
batch_size = 100
processed = 0

def save_temp():
    temp_df = pd.DataFrame({
        "username": list(tweet_counts.keys()),
        "total_posts": list(tweet_counts.values())
    })
    temp_df.to_csv("temp_user_post_counts.csv", index=False)
    print(f"[INFO] Saved temp file with {len(temp_df)} users' counts")

# ----------------------
# 5. Batch‐lookup tweet_count with progress
# ----------------------
for start in range(0, total_users, batch_size):
    end = min(start + batch_size, total_users)
    batch = df["username"].iloc[start:end].tolist()
    print(f"[INFO] Processing users {start+1}-{end} of {total_users}")
    while True:
        try:
            resp = client.get_users(
                usernames=batch,
                user_fields=["public_metrics"]
            )
            for u in resp.data or []:
                tweet_counts[u.username] = u.public_metrics["tweet_count"]
            processed += len(batch)
            print(f"[INFO] Fetched tweet_count for batch: {len(batch)} users (total processed: {processed}/{total_users})")
            save_temp()
            break
        except tweepy.TooManyRequests:
            print("[WARN] Rate limit hit; sleeping for 15 minutes...")
            save_temp()
            time.sleep(15*60)
        except Exception as e:
            print(f"[ERROR] Batch {start+1}-{end} failed: {e}")
            # mark missing as zero
            for uname in batch:
                tweet_counts.setdefault(uname, 0)
            save_temp()
            break

# ----------------------
# 6. Append to DataFrame and save final
# ----------------------
df["total_posts"] = df["username"].map(tweet_counts).fillna(0).astype(int)
outfile = "user_post_counts.csv"
df.to_csv(outfile, index=False)
print(f"[INFO] Completed. Wrote {total_users} rows to '{outfile}'")
