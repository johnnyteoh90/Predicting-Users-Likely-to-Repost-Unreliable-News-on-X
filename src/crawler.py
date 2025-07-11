import tweepy
import pandas as pd
import time
import random
from datetime import datetime, timezone

# ----------------------
# 1. Authentication
# ----------------------
API_KEY = "LhwNAB800EUy00GPY3RWDLP9q"
API_SECRET_KEY = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACfP0QEAAAAArdPGFlxCc5Y67uj6rCePkVpAgHY%3DQGD9R8PE84Qn7iwLWkUY27o65GduujJOgPMwkOGsGGnTEPDqK6"
ACCESS_TOKEN = "1861791142091657216-5z2nwwVmqVR3rbgOPViMSGPFdRp01O"
ACCESS_TOKEN_SECRET = "PKmIB7X1Q629HV3YL210mvRiBe8pTP0l50S4aLmyrWkJv"

client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET
)

# ----------------------
# 2. Mu & Aletras handles subset (updated to match Twitter casing)
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

# Convert handles to lowercase sets for case-insensitive matching
RELIABLE_HANDLES_SET = set(h.lower() for h in RELIABLE_HANDLES)
UNRELIABLE_HANDLES_SET = set(h.lower() for h in UNRELIABLE_HANDLES)

# ----------------------
# 3. Load & filter users
# ----------------------
df = pd.read_csv("user_creation_dates.csv")
df = df[df["user_found"] == True]

# ----------------------
# 4. Fetch user profiles
# ----------------------
user_data = {}
for i in range(0, len(df), 100):
    batch = df["username"].iloc[i:i + 100].tolist()
    while True:
        try:
            resp = client.get_users(
                usernames=batch,
                user_fields=["id", "created_at", "public_metrics", "verified", "verified_type"]
            )
            for u in resp.data or []:
                age_days = (datetime.now(timezone.utc) - u.created_at).days
                user_data[u.username] = {
                    "user_id": u.id,
                    "created_at": u.created_at.isoformat(),
                    "account_age_years": age_days / 365 if age_days > 0 else None,
                    "followers_count": u.public_metrics["followers_count"],
                    "following_count": u.public_metrics["following_count"],
                    "verified": u.verified,
                    "verified_type": getattr(u, "verified_type", None),
                    "statuses_count": u.public_metrics["tweet_count"],
                    "posting_frequency": u.public_metrics["tweet_count"] / (age_days / 365) if age_days > 0 else None
                }
            break
        except tweepy.TooManyRequests:
            print("[INFO] Rate limit on profile fetch; sleeping 15m")
            time.sleep(15 * 60)
        except Exception as ex:
            print(f"[ERROR] Profile batch failed: {ex}")
            break

# ----------------------
# 5. Prepare stratified pools
# ----------------------
reliable_pool = df[df["Labels"] == "reliable"]["username"].tolist()
unreliable_pool = df[df["Labels"] == "unreliable"]["username"].tolist()
random.seed(42)
random.shuffle(reliable_pool)
random.shuffle(unreliable_pool)

# ----------------------
# 6. Sampling & fetch settings
# ----------------------
TARGET = {"reliable": 110, "unreliable": 90}
QUOTA_TWEETS = 50000
tweets_fetched = 0
processed_rows_with_mentions = []
processed_rows_without_mentions = []
temp_threshold = 5000
last_user = None

def save_temp():
    pd.DataFrame(processed_rows_with_mentions).to_csv("temp_data_with_mentions.csv", index=False)
    pd.DataFrame(processed_rows_without_mentions).to_csv("temp_data_without_mentionsners.csv", index=False)
    print(f"[INFO] Temp saved: {len(processed_rows_with_mentions)} rows with mentions, {len(processed_rows_without_mentions)} rows without mentions")

def fetch_tweets(user_id, start_time, next_token=None):
    try:
        tr = client.get_users_tweets(
            id=user_id,
            start_time=start_time,
            max_results=100,
            tweet_fields=["created_at", "text", "referenced_tweets", "entities", "attachments"],
            pagination_token=next_token
        )
        return tr.data or [], tr.meta.get("next_token"), len(tr.data or [])
    except tweepy.TooManyRequests:
        print("[INFO] Rate limit on tweets; sleeping 15m")
        save_temp()
        time.sleep(15 * 60)
        return [], None, 0
    except Exception as ex:
        print(f"[ERROR] Fetching tweets for user {user_id}: {ex}")
        return [], None, 0

# ----------------------
# 7. Main sampling loop
# ----------------------
for label, pool in [("reliable", reliable_pool), ("unreliable", unreliable_pool)]:
    count = 0
    idx = 0
    handles_set = RELIABLE_HANDLES_SET if label == "reliable" else UNRELIABLE_HANDLES_SET
    while count < TARGET[label] and tweets_fetched < QUOTA_TWEETS and idx < len(pool):
        username = pool[idx]
        last_user = username
        idx += 1
        if username not in user_data:
            print(f"[INFO] Skipping user {username} (not in user_data)")
            continue
        uf = user_data[username]
        print(f"[INFO] Processing user: {username} ({label})")

        # Fetch first batch of 100 tweets from created_at
        tweets, next_token, num_fetched = fetch_tweets(uf["user_id"], uf["created_at"])
        tweets_fetched += num_fetched
        if tweets_fetched >= QUOTA_TWEETS:
            print(f"[INFO] Tweet limit reached. Last user: {last_user}")
            break
        if not tweets:
            print(f"[INFO] No tweets fetched for user: {username}")
            continue
        tweets_sorted = sorted(tweets, key=lambda t: t.created_at)
        print(f"[INFO] Fetched {len(tweets_sorted)} tweets for user: {username}")

        boundary = None
        # Check first 100 tweets
        for i, tweet in enumerate(tweets_sorted[:100]):
            entities = tweet.entities or {}
            mentions_lower = [m.get("username", "").lower() for m in entities.get("mentions", [])]
            if any(m in handles_set for m in mentions_lower):
                boundary = i
                break
        # If no mention in first 100, fetch next 100
        if boundary is None and next_token:
            next_tweets, _, next_num_fetched = fetch_tweets(uf["user_id"], uf["created_at"], next_token)
            tweets_fetched += next_num_fetched
            if tweets_fetched >= QUOTA_TWEETS:
                print(f"[INFO] Tweet limit reached. Last user: {last_user}")
                break
            if next_tweets:
                tweets_sorted.extend(next_tweets)
                tweets_sorted = sorted(tweets_sorted, key=lambda t: t.created_at)[:200]
                print(f"[INFO] Fetched additional {len(next_tweets)} tweets for user: {username}")
                # Check second batch (tweets 101-200)
                for i, tweet in enumerate(tweets_sorted[100:200]):
                    entities = tweet.entities or {}
                    mentions_lower = [m.get("username", "").lower() for m in entities.get("mentions", [])]
                    if any(m in handles_set for m in mentions_lower):
                        boundary = 100 + i
                        break

        if boundary is not None:
            print(f"[INFO] Found mention at tweet {boundary} for user: {username}")
            for tweet in tweets_sorted[:boundary]:
                entities = tweet.entities or {}
                urls = [u["expanded_url"] for u in entities.get("urls", []) if "expanded_url" in u]
                mentions = [m["username"] for m in entities.get("mentions", []) if "username" in m]
                hashtags = [h["tag"] for h in entities.get("hashtags", []) if "tag" in h]
                processed_rows_with_mentions.append({
                    "username": username,
                    "LABEL": label,
                    "user_id": uf["user_id"],
                    "account_age_years": uf["account_age_years"],
                    "followers_count": uf["followers_count"],
                    "following_count": uf["following_count"],
                    "verified": uf["verified"],
                    "verified_type": uf["verified_type"],
                    "statuses_count": uf["statuses_count"],
                    "posting_frequency": uf["posting_frequency"],
                    "tweet_text": tweet.text,
                    "tweet_created_at": tweet.created_at,
                    "is_retweet": any(ref.type == "retweeted" for ref in tweet.referenced_tweets) if tweet.referenced_tweets else False,
                    "has_media": bool(tweet.attachments),
                    "urls": ",".join(urls),
                    "mentions": ",".join(mentions),
                    "hashtags": ",".join(hashtags)
                })
            print(f"[INFO] Collected {boundary} tweets (before mention) for user: {username}")
            count += 1
            print(f"[INFO] Current count for {label}: {count}/{TARGET[label]}")
        else:
            print(f"[INFO] No mention found in tweets for user: {username}")
            for tweet in tweets_sorted:
                entities = tweet.entities or {}
                urls = [u["expanded_url"] for u in entities.get("urls", []) if "expanded_url" in u]
                mentions = [m["username"] for m in entities.get("mentions", []) if "username" in m]
                hashtags = [h["tag"] for h in entities.get("hashtags", []) if "tag" in h]
                processed_rows_without_mentions.append({
                    "username": username,
                    "LABEL": label,
                    "user_id": uf["user_id"],
                    "account_age_years": uf["account_age_years"],
                    "followers_count": uf["followers_count"],
                    "following_count": uf["following_count"],
                    "verified": uf["verified"],
                    "verified_type": uf["verified_type"],
                    "statuses_count": uf["statuses_count"],
                    "posting_frequency": uf["posting_frequency"],
                    "tweet_text": tweet.text,
                    "tweet_created_at": tweet.created_at,
                    "is_retweet": any(ref.type == "retweeted" for ref in tweet.referenced_tweets) if tweet.referenced_tweets else False,
                    "has_media": bool(tweet.attachments),
                    "urls": ",".join(urls),
                    "mentions": ",".join(mentions),
                    "hashtags": ",".join(hashtags)
                })
            print(f"[INFO] Collected {len(tweets_sorted)} tweets (no mention) for user: {username}")

        if tweets_fetched >= temp_threshold:
            save_temp()
            temp_threshold += 5000

# ----------------------
# 8. Final save & report
# ----------------------
pd.DataFrame(processed_rows_with_mentions).to_csv("final_data_with_mentions.csv", index=False)
pd.DataFrame(processed_rows_without_mentions).to_csv("final_data_without_mentions.csv", index=False)

users_with_mentions = len(set(r['username'] for r in processed_rows_with_mentions))
users_without_mentions = len(set(r['username'] for r in processed_rows_without_mentions))

print(f"[DONE] Tweets fetched: {tweets_fetched}")
print(f"Users with mentions: {users_with_mentions}")
print(f"Users without mentions: {users_without_mentions}")
print(f"Last user processed: {last_user}")