#!/usr/bin/env python3
import csv
import time
import tweepy
from tweepy.errors import TooManyRequests

# ─── Your API credentials ─────────────────────────────────────────────────────
API_KEY             = "PeEYtuAaTwiCVZsdNeSHEKWgp"
API_SECRET_KEY      = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN        = "AAAAAAAAAAAAAAAAAAAAAO863AEAAAAAVdZk82RYhSY3n2CPWIX5LRxbMbk%3D11tcl7gR9jfSOZDFDUa0B1J89z21nkVUqbjkftnDKtxTVyf5lE"
ACCESS_TOKEN        = "1944709126032166912-ctFBl4Q2WGo034LJ6PNpBaxQ8vcZt"
ACCESS_TOKEN_SECRET = "0tiiVK7JJCYbZtkjkohkq13Fx4N5T4CLC3wmF85ZQXTHy"

# ─── Initialize the Tweepy v2 Client ──────────────────────────────────────────
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

INPUT_CSV   = 'raw_data.csv'
OUTPUT_CSV  = 'raw_data_with_posts.csv'
BATCH_SIZE  = 100

def main():
    # 1) Load users
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader     = csv.DictReader(f)
        users      = list(reader)
        fieldnames = reader.fieldnames + ['total_posts']

    # 2) Extract all usernames (in order), dedupe for efficiency
    all_usernames = [row.get('username', '').strip() for row in users]
    unique_usernames = list(dict.fromkeys(u for u in all_usernames if u))
    total_users = len(unique_usernames)

    # 3) Batch them into chunks of BATCH_SIZE
    batches = [
        unique_usernames[i : i + BATCH_SIZE]
        for i in range(0, total_users, BATCH_SIZE)
    ]

    # 4) Perform batch lookups with progress & back‑off
    total_posts_map = {}
    for batch_idx, batch in enumerate(batches, start=1):
        backoff = 1
        while True:
            try:
                print(f"[Batch {batch_idx}/{len(batches)}] Looking up {len(batch)} users…")
                resp = client.get_users(
                    usernames=batch,
                    user_fields=['public_metrics']
                )
                # Map successful lookups
                if resp.data:
                    for user_obj in resp.data:
                        total_posts_map[user_obj.username] = (
                            user_obj.public_metrics.get('tweet_count', '')
                        )
                # Map errors (e.g. suspended or not found)
                if hasattr(resp, 'errors') and resp.errors:
                    for err in resp.errors:
                        usr = err.get('value', '')
                        detail = err.get('detail', 'unknown error')
                        total_posts_map[usr] = ''
                        print(f"  [!] {usr}: {detail}")
                break

            except TooManyRequests:
                wait = backoff * 10
                print(f"  [!] Rate limit hit. Sleeping {wait}s…")
                time.sleep(wait)
                backoff = min(backoff * 2, 6)

            except Exception as e:
                print(f"  [!] Unhandled error on batch {batch_idx}: {e}")
                # assign blanks for this batch so you don't retry forever
                for usr in batch:
                    total_posts_map.setdefault(usr, '')
                break

        # small pause to be polite
        time.sleep(1)

    # 5) Write out with total_posts in original order
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in users:
            uname = row.get('username', '').strip()
            row['total_posts'] = total_posts_map.get(uname, '')
            writer.writerow(row)

    print(f"\n✅ Done! Augmented CSV written to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
