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
tag = "golden_dataset"
CSV_FILE = f"./datasets/{tag}_with_issues.csv"
PICKLE_FILE = f"./datasets/{tag}_files_progress.pkl"
TOKENS_FILE = "./env/tokens.txt"
REQUEST_DELAY = 1.0
LOG_PATH = f"./datasets/{tag}_files_output.log"
JSON_OUTPUT = f"./datasets/{tag}_with_issues_and_files.json"
CSV_OUTPUT = f"./datasets/{tag}_with_issues_and_files.csv"
SAVE_FILE_CONTENT = True  # Set to True to fetch and save file content


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


# === Get file size and content via contents_url ===
def get_file_size_and_content(contents_url, token_cycle, save_content=False):
    """Return file size in bytes and optionally decoded content from GitHub contents API."""
    token = next(token_cycle)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(contents_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            size = data.get("size")
            content = None
            if save_content and "content" in data and data.get("encoding") == "base64":
                import base64
                try:
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception as e:
                    log_activity(f"Failed to decode content for {contents_url}: {e}")
                    content = None
            return size, content
        elif resp.status_code == 403 and "rate limit" in resp.text.lower():
            log_activity(
                "Rate limit hit during contents_url request, switching token..."
            )
            return get_file_size_and_content(contents_url, token_cycle, save_content)  # retry with next token
        else:
            log_activity(f"Failed to get size ({resp.status_code}) for {contents_url}")
            return None, None
    except requests.RequestException as e:
        log_activity(f"Request failed for {contents_url}: {e}")
        return None, None


# === Fetch PR Files with Pagination ===
def fetch_pr_files(repo, pr_number, token_cycle):
    all_files = []
    page = 1
    while True:
        token = next(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"

        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                all_files.extend(data)
                if len(data) < 100:
                    break  # last page
                else:
                    page += 1
                    time.sleep(REQUEST_DELAY)
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                log_activity("Rate limit hit, switching token...")
                continue
            else:
                log_activity(
                    f"Failed ({resp.status_code}) on {repo}#{pr_number}: {resp.json().get('message')}"
                )
                break
        except requests.RequestException as e:
            log_activity(f"Request failed for {repo}#{pr_number}: {e}")
            break

    return all_files


# === Process PRs ===
for i, (_, row) in enumerate(df.iterrows()):
    search_repository, pull_number, id, title, description, state = (
        row["search_repository"],
        row["pull_number"],
        row["id"],
        row["title"],
        row["body"],
        row["state"],
    )
    # Extract issue-related columns
    issue_titles = row.get("issue_titles", None)
    issue_bodies = row.get("issue_bodies", None)
    issue_comments = row.get("issue_comments", None)
    key = f"{search_repository}#{pull_number}"
    if key in progress:
        log_activity(f"[{i + 1}/{len(df)}] Skipping already processed {key}")
        continue

    log_activity(f"[{i + 1}/{len(df)}] Fetching {key} ...")
    pr_files = fetch_pr_files(search_repository, pull_number, token_cycle)

    if pr_files:
        pr_total_size = 0
        for file_data in pr_files:
            contents_url = file_data.get("contents_url")
            if contents_url:
                size, content = get_file_size_and_content(contents_url, token_cycle, SAVE_FILE_CONTENT)
                file_data["file_size_bytes"] = size
                if SAVE_FILE_CONTENT:
                    file_data["file_content"] = content
                if size is not None:
                    pr_total_size += size
                time.sleep(REQUEST_DELAY)  # avoid hammering API

        progress[key] = {
            "id": id,
            "title": title,
            "description": description,
            "state": state,
            "repository": search_repository,
            "pull_number": pull_number,
            "files": pr_files,
            "pr_total_size_bytes": pr_total_size,
            "issue_titles": issue_titles,
            "issue_bodies": issue_bodies,
            "issue_comments": issue_comments,
        }

        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(progress, f)
        log_activity(
            f"✓ Saved progress for {key} ({len(pr_files)} files, {pr_total_size} bytes total)"
        )
    else:
        log_activity(f"⚠️ No files fetched for {key}")

    time.sleep(REQUEST_DELAY)

log_activity(f"✅ Done. Fetched data for {len(progress)} PRs.")

# === Save to JSON ===
with open(JSON_OUTPUT, "w") as f:
    json.dump(progress, f, indent=2)
log_activity(f"📁 Saved JSON results to {JSON_OUTPUT}")

# === Save to CSV ===
csv_rows = []
for key, pr_data in progress.items():
    files = pr_data.get("files", [])
    pr_id = pr_data.get("id")
    pr_title = pr_data.get("title")
    pr_description = pr_data.get("description")
    pr_state = pr_data.get("state")
    repo = pr_data.get("repository")
    pr_number = pr_data.get("pull_number")
    pr_total_size_bytes = pr_data.get("pr_total_size_bytes", None)
    issue_titles = pr_data.get("issue_titles", None)
    issue_bodies = pr_data.get("issue_bodies", None)
    issue_comments = pr_data.get("issue_comments", None)

    for f in files:
        csv_rows.append(
            {
                "id": pr_id,
                "title": pr_title,
                "description": pr_description,
                "state": pr_state,
                "repository": repo,
                "pr_number": pr_number,
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "changes": f.get("changes"),
                "sha": f.get("sha"),
                "blob_url": f.get("blob_url"),
                "raw_url": f.get("raw_url"),
                "patch": f.get("patch", None),
                "file_size_bytes": f.get("file_size_bytes"),
                "file_content": f.get("file_content") if SAVE_FILE_CONTENT else None,
                "pr_total_size_bytes": pr_total_size_bytes,
                "issue_titles": issue_titles,
                "issue_bodies": issue_bodies,
                "issue_comments": issue_comments,
            }
        )

pd.DataFrame(csv_rows).to_csv(CSV_OUTPUT, index=False)
log_activity(f"📄 Saved CSV results to {CSV_OUTPUT}")
