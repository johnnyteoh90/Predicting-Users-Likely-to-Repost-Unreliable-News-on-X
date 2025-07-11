import tweepy
import pandas as pd
import time
import os

# Authentication
API_KEY = "LhwNAB800EUy00GPY3RWDLP9q"
API_SECRET_KEY = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACfP0QEAAAAArdPGFlxCc5Y67uj6rCePkVpAgHY%3DQGD9R8PE84Qn7iwLWkUY27o65GduujJOgPMwkOGsGGnTEPDqK6"
ACCESS_TOKEN = "1861791142091657216-5z2nwwVmqVR3rbgOPViMSGPFdRp01O"
ACCESS_TOKEN_SECRET = "PKmIB7X1Q629HV3YL210mvRiBe8pTP0l50S4aLmyrWkJv"

def create_client():
    return tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=API_KEY,
        consumer_secret=API_SECRET_KEY,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET,
        wait_on_rate_limit=False  # handle manually
    )


def main():
    # Load usernames
    input_csv = "dataset_twitter_raw.csv"
    df = pd.read_csv(input_csv)
    usernames = df['username'].astype(str).tolist()
    total_users = len(usernames)
    batch_size = 100
    total_batches = (total_users + batch_size - 1) // batch_size

    client = create_client()
    results = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_users)
        batch = usernames[start:end]
        print(f"[INFO] Processing batch {batch_idx+1}/{total_batches} users {start+1}-{end}")

        try:
            response = client.get_users(
                usernames=batch,
                user_fields=["id"]
            )
            found = {user.username.lower() for user in response.data} if response.data else set()
        except tweepy.errors.TooManyRequests:
            print(f"[WARNING] Rate limit hit at batch {batch_idx+1}. Sleeping for 15 minutes...")
            time.sleep(15 * 60)
            print("[INFO] Resuming after wait.")
            response = client.get_users(
                usernames=batch,
                user_fields=["id"]
            )
            found = {user.username.lower() for user in response.data} if response.data else set()
        except Exception as e:
            print(f"[ERROR] Unexpected error on batch {batch_idx+1}: {e}")
            found = set()

        # Record results
        for uname in batch:
            results.append({
                'username': uname,
                'user_found': uname.lower() in found
            })

        # Save a temp file after each batch
        temp_df = pd.DataFrame(results)
        temp_file = "temp_user_existence.csv"
        temp_df.to_csv(temp_file, index=False)
        print(f"[INFO] Saved progress to '{temp_file}' ({end}/{total_users} users checked)")

    # Final save
    final_file = "user_existence_check.csv"
    pd.DataFrame(results).to_csv(final_file, index=False)
    print(f"[INFO] Check complete. Final results saved to '{final_file}'.")

if __name__ == "__main__":
    main()
