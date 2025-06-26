import tweepy
import pandas as pd
import time
from datetime import datetime, timezone

# Authentication
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

def main():
    input_csv = "dataset_trial.csv"
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded CSV file: '{input_csv}' with {len(df)} rows.")

    # Fetch user profiles in batches
    user_data = {}
    batch_size = 100
    for i in range(0, len(df), batch_size):
        usernames_batch = df['username'][i:i+batch_size].tolist()
        while True:
            try:
                response = client.get_users(
                    usernames=usernames_batch,
                    user_fields=["id", "created_at", "public_metrics", "verified", "verified_type"]
                )
                if response.data:
                    for user in response.data:
                        account_age_days = (datetime.now(timezone.utc) - user.created_at).days
                        user_data[user.username] = {
                            "user_id": user.id,
                            "created_at": user.created_at,
                            "account_age_years": account_age_days / 365 if user.created_at else None,
                            "followers_count": user.public_metrics["followers_count"],
                            "following_count": user.public_metrics["following_count"],
                            "verified": user.verified,
                            "verified_type": getattr(user, "verified_type", "N/A"),
                            "statuses_count": user.public_metrics["tweet_count"],
                            "posting_frequency": (user.public_metrics["tweet_count"] / (account_age_days / 365)) if user.created_at and account_age_days > 0 else None
                        }
                break  # Exit loop if successful
            except tweepy.errors.TooManyRequests:
                print("[INFO] Rate limit exceeded while fetching user profiles. Saving data and sleeping for 15 minutes...")
                # Since user_data is intermediate, we can proceed without saving it here
                time.sleep(15 * 60)  # Sleep for 15 minutes
            except Exception as e:
                print(f"[ERROR] Error fetching user batch {i}: {e}")
                break  # Skip this batch on other errors

    # Fetch all tweets (up to 100 per user)
    data_rows = []
    tweet_requests_made = 0
    for i, username in enumerate(df['username']):
        label = df.loc[df['username'] == username, 'LABEL'].values[0]
        
        if username not in user_data:
            row = {
                "username": username,
                "LABEL": label,
                "user_found": False,
                "error": "User not found",
                "account_age_years": None,
                "followers_count": None,
                "following_count": None,
                "verified": None,
                "verified_type": None,
                "statuses_count": None,
                "posting_frequency": None,
                "tweet_text": None,
                "tweet_created_at": None,
                "is_retweet": None,
                "has_media": None,
                "urls": None,
                "mentions": None,
                "hashtags": None
            }
            data_rows.append(row)
            continue

        user_features = user_data[username]
        while True:
            try:
                tweets_response = client.get_users_tweets(
                    id=user_features["user_id"],
                    max_results=100,
                    tweet_fields=["created_at", "text", "referenced_tweets", "entities", "attachments"],
                    expansions=["attachments.media_keys"]
                )
                tweet_requests_made += 1

                if tweets_response.data:
                    for tweet in tweets_response.data:
                        entities = tweet.entities if tweet.entities else {}
                        urls = [url["expanded_url"] for url in entities.get("urls", []) if "expanded_url" in url]
                        mentions = [mention["username"] for mention in entities.get("mentions", []) if "username" in mention]
                        hashtags = [hashtag["tag"] for hashtag in entities.get("hashtags", []) if "tag" in hashtag]
                        row = {
                            "username": username,
                            "LABEL": label,
                            "user_id": user_features["user_id"],
                            "account_age_years": user_features["account_age_years"],
                            "followers_count": user_features["followers_count"],
                            "following_count": user_features["following_count"],
                            "verified": user_features["verified"],
                            "verified_type": user_features["verified_type"],
                            "statuses_count": user_features["statuses_count"],
                            "posting_frequency": user_features["posting_frequency"],
                            "tweet_text": tweet.text,
                            "tweet_created_at": tweet.created_at,
                            "is_retweet": any(ref.type == "retweeted" for ref in tweet.referenced_tweets) if tweet.referenced_tweets else False,
                            "has_media": bool(tweet.attachments),
                            "urls": ",".join(urls),
                            "mentions": ",".join(mentions),
                            "hashtags": ",".join(hashtags)
                        }
                        data_rows.append(row)
                else:
                    # No tweets found
                    row = {
                        "username": username,
                        "LABEL": label,
                        "user_id": user_features["user_id"],
                        "account_age_years": user_features["account_age_years"],
                        "followers_count": user_features["followers_count"],
                        "following_count": user_features["following_count"],
                        "verified": user_features["verified"],
                        "verified_type": user_features["verified_type"],
                        "statuses_count": user_features["statuses_count"],
                        "posting_frequency": user_features["posting_frequency"],
                        "tweet_text": None,
                        "tweet_created_at": None,
                        "is_retweet": None,
                        "has_media": None,
                        "urls": None,
                        "mentions": None,
                        "hashtags": None,
                        "error": "No tweets found"
                    }
                    data_rows.append(row)
                break  # Exit loop if successful
            except tweepy.errors.TooManyRequests:
                print(f"[INFO] Rate limit exceeded while fetching tweets for user '{username}'. Saving data and sleeping for 15 minutes...")
                temp_df = pd.DataFrame(data_rows)
                temp_df.to_csv("temp_data.csv", index=False)
                time.sleep(15 * 60)  # Sleep for 15 minutes before retrying this user
            except Exception as e:
                print(f"[ERROR] Error fetching tweets for user '{username}': {e}")
                row = {
                    "username": username,
                    "LABEL": label,
                    "user_id": user_features["user_id"],
                    "account_age_years": user_features["account_age_years"],
                    "followers_count": user_features["followers_count"],
                    "following_count": user_features["following_count"],
                    "verified": user_features["verified"],
                    "verified_type": user_features["verified_type"],
                    "statuses_count": user_features["statuses_count"],
                    "posting_frequency": user_features["posting_frequency"],
                    "tweet_text": None,
                    "tweet_created_at": None,
                    "is_retweet": None,
                    "has_media": None,
                    "urls": None,
                    "mentions": None,
                    "hashtags": None,
                    "error": str(e)
                }
                data_rows.append(row)
                break  # Move to next user on other errors

        # Save temporary data every 100 users
        if (i + 1) % 100 == 0:
            temp_df = pd.DataFrame(data_rows)
            temp_df.to_csv("temp_data.csv", index=False)
            print(f"[INFO] Saved temporary data after processing {i+1} users.")

    # Save final data
    final_df = pd.DataFrame(data_rows)
    final_df.to_csv("final_data.csv", index=False)
    print("[INFO] Processing complete. Final data saved to 'final_data.csv'.")

if __name__ == "__main__":
    main()