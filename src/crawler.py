import os
import tweepy
import pandas as pd
import time
import math
from datetime import datetime, timezone

# ----------------------
# 1. Authentication
# ----------------------
API_KEY            = "PeEYtuAaTwiCVZsdNeSHEKWgp"
API_SECRET_KEY     = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN       = "AAAAAAAAAAAAAAAAAAAAAO863AEAAAAAVdZk82RYhSY3n2CPWIX5LRxbMbk%3D11tcl7gR9jfSOZDFDUa0B1J89z21nkVUqbjkftnDKtxTVyf5lE"
ACCESS_TOKEN       = "1944709126032166912-ctFBl4Q2WGo034LJ6PNpBaxQ8vcZt"
ACCESS_TOKEN_SECRET= "0tiiVK7JJCYbZtkjkohkq13Fx4N5T4CLC3wmF85ZQXTHy"

client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=False   # disable automatic sleeping
)

# ----------------------
# 2. Handle sets
# ----------------------
RELIABLE_HANDLES = [
    "USATODAY", "BBC", "CNN", "Reuters", "MoonofA", "PressTV", "Independent",
    "AP", "VDARE", "Business", "BuzzFeedNews", "nytimes"
]
UNRELIABLE_HANDLES = [
    "ActivistPost", "infowars", "RT_com", "Antiwarcom", "Consortiumnews",
    "CorbettReport", "LibertyBlitz", "MintPressNews", "Ruptly", "RussiaInsider",
    "RussiaBeyond", "SputnikInt", "wikileaks", "TheRussophile", "HealthRanger",
    "YahooNews", "JoinGaia", "TheDCGazette", "ChroniclesU", "LewRockwell",
    "21WIRE", "TheBlaze", "anonews_co", "DiscloseTV"
]

RELIABLE_SET   = {h.lower() for h in RELIABLE_HANDLES}
UNRELIABLE_SET = {h.lower() for h in UNRELIABLE_HANDLES}

# ----------------------
# 3. Load users
# ----------------------
df_raw = pd.read_csv("raw_data2.csv")
df = df_raw[df_raw["user_found"] == True]
total_users = len(df)
print(f"[INFO] {total_users} users to process")

# ----------------------
# 4. Fetch profiles
# ----------------------
user_data = {}
for i in range(0, total_users, 100):
    batch = df["username"].iloc[i:i+100].tolist()
    print(f"[INFO] Fetching profiles {i+1}-{i+len(batch)}")
    while True:
        try:
            resp = client.get_users(
                usernames=batch,
                user_fields=["id","created_at","public_metrics","verified","verified_type"]
            )
            for u in (resp.data or []):
                days = (datetime.now(timezone.utc) - u.created_at).days
                user_data[u.username] = {
                    "user_id":           u.id,
                    "created_at":        u.created_at.isoformat(),
                    "account_age_years": days/365 if days>0 else None,
                    "followers_count":   u.public_metrics["followers_count"],
                    "following_count":   u.public_metrics["following_count"],
                    "verified":          u.verified,
                    "verified_type":     getattr(u,"verified_type",None),
                    "statuses_count":    u.public_metrics["tweet_count"],
                    "posting_frequency": u.public_metrics["tweet_count"]/(days/365) if days>0 else None
                }
            break
        except tweepy.TooManyRequests:
            print("[WARN] Profile rate limit, saving temp & sleeping 15m")
            # snapshot partial profiles
            pd.DataFrame.from_dict(user_data, orient="index").to_csv("temp_profiles.csv")
            time.sleep(15*60)
        except Exception as e:
            print(f"[ERROR] Profiles fetch failed: {e}")
            break

# ----------------------
# 5. Fetch tweets helper
# ----------------------
QUOTA_TWEETS   = 25000
tweets_fetched = 0

def save_temp():
    pd.DataFrame(processed_rows).to_csv("temp_output.csv", index=False)
    print(f"[INFO] Temp snapshot saved: {len(processed_rows)} rows; tweets_fetched={tweets_fetched}")

def fetch_user_tweets(user_id, to_fetch):
    global tweets_fetched
    all_tweets = []
    next_token = None
    fetched_for_user = 0

    while fetched_for_user < to_fetch and tweets_fetched < QUOTA_TWEETS:
        batch_size = min(100, to_fetch - fetched_for_user)
        try:
            resp = client.get_users_tweets(
                id=user_id,
                max_results=batch_size,
                pagination_token=next_token,
                tweet_fields=["created_at","text","referenced_tweets","entities","attachments"]
            )
        except tweepy.TooManyRequests:
            # save everything so far, then sleep and retry same page
            print("  [WARN] Tweet rate limit hit; saving temp & sleeping 15m")
            save_temp()
            time.sleep(15*60)
            continue
        except Exception as e:
            print(f"  [ERROR] Fetch tweets failed: {e}")
            break

        batch = resp.data or []
        if not batch:
            break

        all_tweets.extend(batch)
        fetched_for_user += len(batch)
        tweets_fetched   += len(batch)
        next_token = resp.meta.get("next_token")

        print(f"  [INFO] Fetched {fetched_for_user}/{to_fetch} for user; global tweets={tweets_fetched}")

        if not next_token:
            break

    return all_tweets

# ----------------------
# 6. Main loop
# ----------------------
processed_rows = []
temp_threshold = 5000

for idx, row in df.iterrows():
    username    = row["username"]
    label       = row["Labels"]
    total_posts = int(row["total_posts"])
    uf          = user_data.get(username)
    print(f"\n[INFO] ({idx+1}/{total_users}) {username}: total_posts={total_posts}")

    if uf is None:
        print("  [WARN] No profile data; skipping")
        continue

    to_fetch = min(total_posts, 3200)
    tweets = fetch_user_tweets(uf["user_id"], to_fetch)
    print(f"  [INFO] Completed fetch: got {len(tweets)}/{to_fetch}")

    handle_set = RELIABLE_SET if label=="reliable" else UNRELIABLE_SET

    for tw in tweets:
        ents = tw.entities or {}
        urls     = [u["expanded_url"] for u in ents.get("urls", []) if "expanded_url" in u]
        mentions = [m["username"] for m in ents.get("mentions", []) if "username" in m]
        hashtags = [h["tag"] for h in ents.get("hashtags", []) if "tag" in h]

        is_rt = bool(tw.referenced_tweets)
        mh, mc = "nil","no"
        if is_rt and mentions:
            orig = mentions[0].lower()
            if orig in handle_set:
                mh, mc = orig, "yes"

        processed_rows.append({
            "username":          username,
            "LABEL":             label,
            "created_at":        uf["created_at"],
            "user_id":           uf["user_id"],
            "account_age_years": uf["account_age_years"],
            "total_posts":       total_posts,
            "followers_count":   uf["followers_count"],
            "following_count":   uf["following_count"],
            "verified":          uf["verified"],
            "verified_type":     uf["verified_type"],
            "statuses_count":    uf["statuses_count"],
            "posting_frequency": uf["posting_frequency"],
            "tweet_text":        tw.text,
            "tweet_created_at":  tw.created_at,
            "is_retweet":        is_rt,
            "has_media":         bool(tw.attachments),
            "urls":              ",".join(urls),
            "mentions":          ",".join(mentions),
            "hashtags":          ",".join(hashtags),
            "matched_handler":   mh,
            "matched_condition": mc
        })

    # snapshot after each user
    save_temp()

    if tweets_fetched >= QUOTA_TWEETS:
        print("[INFO] Global quota reached; stopping.")
        break

# ----------------------
# 7. Final save
# ----------------------
pd.DataFrame(processed_rows).to_csv("final_output.csv", index=False)
print(f"\n[DONE] Wrote {len(processed_rows)} rows; total tweets fetched={tweets_fetched}")
