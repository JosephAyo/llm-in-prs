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
import re

# === Config ===
tag= 'few_shot_examples'
CSV_FILE = f"./datasets/{tag}.csv"
PICKLE_FILE = f"./datasets/{tag}_issues_progress.pkl"
TOKENS_FILE = "./env/tokens.txt"
REQUEST_DELAY = 1.0
LOG_PATH = f"./datasets/{tag}_issues_output.log"
JSON_OUTPUT = f"./datasets/{tag}_with_issues.json"
CSV_OUTPUT = f"./datasets/{tag}_with_issues.csv"


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

# === Load CSV ===
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    log_activity(f"Error: {CSV_FILE} not found.")
    sys.exit(1)

# Check if issue_numbers column exists
if 'issue_numbers' not in df.columns:
    log_activity(f"Error: 'issue_numbers' column not found in {CSV_FILE}")
    sys.exit(1)

# === Load previous progress ===
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        progress = pickle.load(f)
    log_activity(f"Loaded progress for {len(progress)} issues.")
else:
    progress = {}


# === Parse issue numbers from string ===
def parse_issue_numbers(issue_numbers_str):
    """Parse issue numbers from a string like '#762|#692|#685|#602|#225'"""
    if pd.isna(issue_numbers_str) or not issue_numbers_str:
        return []
    
    # Split by | and extract numbers, removing # prefix
    issues = issue_numbers_str.split('|')
    parsed_issues = []
    
    for issue in issues[:2]:  # Only take first two
        issue = issue.strip()
        # Extract number from #762 format
        match = re.search(r'#?(\d+)', issue)
        if match:
            parsed_issues.append(int(match.group(1)))
    
    return parsed_issues


# === Fetch issue data ===
def fetch_issue(repo, issue_number, token_cycle):
    """Fetch issue data from GitHub API"""
    token = next(token_cycle)
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403 and "rate limit" in resp.text.lower():
            log_activity("Rate limit hit while fetching issue, switching token...")
            return fetch_issue(repo, issue_number, token_cycle)  # retry with next token
        elif resp.status_code == 404:
            log_activity(f"Issue #{issue_number} not found in {repo}")
            return None
        else:
            log_activity(f"Failed to fetch issue #{issue_number} from {repo} ({resp.status_code}): {resp.json().get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        log_activity(f"Request failed for {repo}#{issue_number}: {e}")
        return None


# === Fetch issue comments ===
def fetch_issue_comments(comments_url, token_cycle):
    """Fetch all comments for an issue with pagination"""
    all_comments = []
    page = 1
    
    while True:
        token = next(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{comments_url}?per_page=100&page={page}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                all_comments.extend(data)
                if len(data) < 100:
                    break  # last page
                else:
                    page += 1
                    time.sleep(REQUEST_DELAY)
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                log_activity("Rate limit hit while fetching comments, switching token...")
                continue  # retry with next token
            else:
                log_activity(f"Failed to fetch comments ({resp.status_code}) from {comments_url}")
                break
        except requests.RequestException as e:
            log_activity(f"Request failed for comments {comments_url}: {e}")
            break
    
    return all_comments


# === Format comments ===
def format_comments(comments):
    """Format comments into a single string"""
    if not comments:
        return ""
    
    formatted_comments = []
    for i, comment in enumerate(comments, 1):
        author = comment.get('user', {}).get('login', 'Unknown')
        body = comment.get('body', '').strip()
        formatted_comments.append(f"Comment #{i} by {author} : {body}")
    
    return "\n\n".join(formatted_comments)


# === Process rows ===
for i, (_, row) in enumerate(df.iterrows()):
    # Get repository from the row (assuming it exists)
    if 'search_repository' in row:
        repository = row['search_repository']
    elif 'repository' in row:
        repository = row['repository']
    else:
        log_activity(f"[{i + 1}/{len(df)}] No repository column found, skipping row")
        continue
    
    issue_numbers_str = row['issue_numbers']
    issue_numbers = parse_issue_numbers(issue_numbers_str)
    
    if not issue_numbers:
        log_activity(f"[{i + 1}/{len(df)}] No valid issue numbers found in '{issue_numbers_str}', skipping")
        continue
    
    log_activity(f"[{i + 1}/{len(df)}] Processing issues {issue_numbers} from {repository}")
    
    # Process each issue number
    for issue_number in issue_numbers:
        key = f"{repository}#{issue_number}"
        
        if key in progress:
            log_activity(f"Skipping already processed {key}")
            continue
        
        log_activity(f"Fetching issue {key} ...")
        issue_data = fetch_issue(repository, issue_number, token_cycle)
        
        if issue_data:
            # Fetch comments
            comments_url = issue_data.get('comments_url')
            comments = []
            formatted_comments = ""
            
            if comments_url:
                log_activity(f"Fetching comments for {key} ...")
                comments = fetch_issue_comments(comments_url, token_cycle)
                formatted_comments = format_comments(comments)
                time.sleep(REQUEST_DELAY)
            
            # Store the processed issue
            progress[key] = {
                "repository": repository,
                "issue_number": issue_number,
                "id": issue_data.get('id'),
                "title": issue_data.get('title'),
                "body": issue_data.get('body'),
                "comments": formatted_comments,
                "comments_count": len(comments),
                "state": issue_data.get('state'),
                "created_at": issue_data.get('created_at'),
                "updated_at": issue_data.get('updated_at'),
                "closed_at": issue_data.get('closed_at'),
                "author": issue_data.get('user', {}).get('login') if issue_data.get('user') else None,
                "url": issue_data.get('html_url'),
            }
            
            # Save progress after each issue
            with open(PICKLE_FILE, "wb") as f:
                pickle.dump(progress, f)
            
            log_activity(f"✓ Saved progress for {key} ({len(comments)} comments)")
        else:
            log_activity(f"⚠️ Failed to fetch issue {key}")
        
        time.sleep(REQUEST_DELAY)

log_activity(f"✅ Done. Fetched data for {len(progress)} issues.")

# === Save to JSON ===
with open(JSON_OUTPUT, "w") as f:
    json.dump(progress, f, indent=2)
log_activity(f"📁 Saved JSON results to {JSON_OUTPUT}")

# === Save to CSV ===
csv_rows = []
for key, issue_data in progress.items():
    csv_rows.append({
        "repository": issue_data.get("repository"),
        "issue_number": issue_data.get("issue_number"),
        "id": issue_data.get("id"),
        "title": issue_data.get("title"),
        "body": issue_data.get("body"),
        "comments": issue_data.get("comments"),
        "comments_count": issue_data.get("comments_count"),
        "state": issue_data.get("state"),
        "created_at": issue_data.get("created_at"),
        "updated_at": issue_data.get("updated_at"),
        "closed_at": issue_data.get("closed_at"),
        "author": issue_data.get("author"),
        "url": issue_data.get("url"),
    })

pd.DataFrame(csv_rows).to_csv(CSV_OUTPUT, index=False)
log_activity(f"📄 Saved CSV results to {CSV_OUTPUT}")
