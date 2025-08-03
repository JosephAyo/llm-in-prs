import pandas as pd
import requests
import pickle
import time
import os
import sys
import json
import itertools
import random

# === Config ===
CSV_FILE = "./datasets/random_significant_prs.csv"
PICKLE_FILE = "progress.pkl"
TOKENS_FILE = "./env/tokens.txt"  # Each line should have one GitHub token
REQUEST_DELAY = 1.0  # seconds between requests


# === Load Tokens ===
def load_tokens(tokens_file):
    with open(tokens_file, "r") as file:
        tokens = file.read().splitlines()
    if not tokens:
        raise ValueError("No tokens found in token file.")
    start_index = random.randint(0, len(tokens) - 1)
    rotated_tokens = tokens[start_index:] + tokens[:start_index]
    return tokens, itertools.cycle(rotated_tokens)


try:
    all_tokens, token_cycle = load_tokens(TOKENS_FILE)
except Exception as e:
    print(f"Error loading tokens: {e}")
    sys.exit(1)

# === Load PR CSV ===
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found.")
    sys.exit(1)

# === Load previous progress ===
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        progress = pickle.load(f)
    print(f"Loaded progress for {len(progress)} PRs.")
else:
    progress = {}

# === Process each PR ===
for i, (_, row) in enumerate(df[:2].iterrows()):
    search_repository, pull_number = row["search_repository"], row["pull_number"]
    key = f"{search_repository}#{pull_number}"
    if key in progress:
        print(f"[{(i)+1}/{len(df)}] Skipping already processed {key}")
        continue

    url = f"https://api.github.com/repos/{search_repository}/pulls/{pull_number}/files"

    for _ in range(len(all_tokens)):
        token = next(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        print(f"[{(i)+1}/{len(df)}] Fetching {key} with token ...")
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                progress[key] = data
                with open(PICKLE_FILE, "wb") as f:
                    pickle.dump(progress, f)
                break  # successful
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                print("Rate limit hit, switching token...")
                continue  # try next token
            else:
                print(f"Failed ({resp.status_code}): {resp.json().get('message')}")
                break  # don't retry if not rate limit
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            continue

    time.sleep(REQUEST_DELAY)

print("\n✅ Done. Fetched data for", len(progress), "PRs.")

# === Save to JSON file ===
with open("pr_files_output.json", "w") as f:
    json.dump(progress, f, indent=2)

print("Saved results to pr_files_output.json")
