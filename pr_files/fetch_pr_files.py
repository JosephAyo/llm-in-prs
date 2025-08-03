import datetime
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
PICKLE_FILE = "./datasets/progress.pkl"
TOKENS_FILE = "./env/tokens.txt"  # Each line should have one GitHub token
REQUEST_DELAY = 1.0  # seconds between requests
LOG_PATH = f"./datasets/output.log"

def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)

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
    log_activity(f"Error loading tokens: {e}")
    sys.exit(1)

# === Load PR CSV ===
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    log_activity(f"Error: {CSV_FILE} not found.")
    sys.exit(1)

# === Load previous progress ===
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        progress = pickle.load(f)
    log_activity(f"Loaded progress for {len(progress)} PRs.")
else:
    progress = {}

# === Process each PR ===
for i, (_, row) in enumerate(df[:2].iterrows()):
    search_repository, pull_number = row["search_repository"], row["pull_number"]
    key = f"{search_repository}#{pull_number}"
    if key in progress:
        log_activity(f"[{(i)+1}/{len(df)}] Skipping already processed {key}")
        continue

    url = f"https://api.github.com/repos/{search_repository}/pulls/{pull_number}/files"

    for _ in range(len(all_tokens)):
        token = next(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        log_activity(f"[{(i)+1}/{len(df)}] Fetching {key} with token ...")
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                progress[key] = data
                with open(PICKLE_FILE, "wb") as f:
                    pickle.dump(progress, f)
                break  # successful
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                log_activity("Rate limit hit, switching token...")
                continue  # try next token
            else:
                log_activity(f"Failed ({resp.status_code}): {resp.json().get('message')}")
                break  # don't retry if not rate limit
        except requests.RequestException as e:
            log_activity(f"Request failed: {e}")
            continue

    time.sleep(REQUEST_DELAY)

log_activity(f"\n✅ Done. Fetched data for {len(progress)}, PRs.")

# === Save to JSON file ===
save_file = './datasets/pr_files_output.json'
with open(save_file, "w") as f:
    json.dump(progress, f, indent=2)

log_activity(f"Saved results to {save_file}")
