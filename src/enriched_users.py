import pandas as pd
import requests
import time
import os
import json

# ---------- Configuration ----------
EXCEL_FILE = 'mentioned.xlsx'
TEMP_DIR = 'temp'
OUTPUT_FILE = 'enriched_users.csv'
BATCH_SIZE = 100
SLEEP_ON_RATE_LIMIT = 60  # seconds

# X API Credentials
API_KEY = "PeEYtuAaTwiCVZsdNeSHEKWgp"
API_SECRET_KEY = "FExG9RIxPa5ybYFR8qrJwsC4i4yQq7QTii5ArZYpPVLw7L4nfZ"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAO863AEAAAAAVdZk82RYhSY3n2CPWIX5LRxbMbk%3D11tcl7gR9jfSOZDFDUa0B1J89z21nkVUqbjkftnDKtxTVyf5lE"
ACCESS_TOKEN = "1944709126032166912-ctFBl4Q2WGo034LJ6PNpBaxQ8vcZt"
ACCESS_TOKEN_SECRET = "0tiiVK7JJCYbZtkjkohkq13Fx4N5T4CLC3wmF85ZQXTHy"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "User-Agent": "MediaDetectorScript/1.0"
}


def extract_usernames():
    """Read the Excel file and extract a deduplicated list of @usernames."""
    df = pd.read_excel(EXCEL_FILE, dtype=str)
    mentions = df.astype(str).values.flatten()
    usernames = set()
    for cell in mentions:
        # split on commas, strip out any leading @ and whitespace
        for name in cell.split(','):
            clean = name.strip().lstrip('@')
            if clean:
                usernames.add(clean)
    return list(usernames)


def fetch_user_info(user_list):
    """Batch-fetch user info via X API, handling rate-limits and saving temp files."""
    all_users = []
    os.makedirs(TEMP_DIR, exist_ok=True)

    for idx in range(0, len(user_list), BATCH_SIZE):
        batch = user_list[idx:idx + BATCH_SIZE]
        print(f"[{idx + 1}–{idx + len(batch)}] Fetching {len(batch)} users…")
        params = {
            "usernames": ",".join(batch),
            "user.fields": "id,name,username,description,verified_type,public_metrics,created_at"
        }
        url = "https://api.twitter.com/2/users/by"
        resp = requests.get(url, params=params, headers=HEADERS)

        if resp.status_code == 429:
            reset = int(resp.headers.get('x-rate-limit-reset', time.time() + SLEEP_ON_RATE_LIMIT))
            wait = max(reset - time.time(), SLEEP_ON_RATE_LIMIT)
            print(f"→ Rate limit hit, sleeping for {int(wait)} seconds…")
            time.sleep(wait)
            resp = requests.get(url, params=params, headers=HEADERS)

        resp.raise_for_status()
        data = resp.json().get("data", [])
        all_users.extend(data)

        # write batch to temp JSON for progress tracking
        tmpfile = os.path.join(TEMP_DIR, f"batch_{idx//BATCH_SIZE + 1}.json")
        with open(tmpfile, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"→ Saved batch {idx//BATCH_SIZE + 1} to {tmpfile}")

        # small pause to be polite
        time.sleep(1)
    
    return all_users


def main():
    print("1) Extracting unique usernames from Excel…")
    users = extract_usernames()
    print(f"   → {len(users)} unique usernames found.\n")

    print("2) Enriching user data via X API…")
    enriched = fetch_user_info(users)
    print(f"   → Enriched data for {len(enriched)} users.\n")

    print("3) Writing final CSV output…")
    df_out = pd.DataFrame(enriched)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"   → Output file created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
