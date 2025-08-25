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
tag = "golden_dataset"
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
if "issue_numbers" not in df.columns:
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
    issues = issue_numbers_str.split("|")
    parsed_issues = []

    for issue in issues[:2]:  # Only take first two
        issue = issue.strip()
        # Extract number from #762 format
        match = re.search(r"#?(\d+)", issue)
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
            log_activity(
                f"Failed to fetch issue #{issue_number} from {repo} ({resp.status_code}): {resp.json().get('message', 'Unknown error')}"
            )
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
                log_activity(
                    "Rate limit hit while fetching comments, switching token..."
                )
                continue  # retry with next token
            else:
                log_activity(
                    f"Failed to fetch comments ({resp.status_code}) from {comments_url}"
                )
                break
        except requests.RequestException as e:
            log_activity(f"Request failed for comments {comments_url}: {e}")
            break

    return all_comments


# === Format comments ===
def format_comments(comments, issue_number):
    """Format comments into a single string with issue number"""
    if not comments:
        return ""

    formatted_comments = []
    for i, comment in enumerate(comments, 1):
        author = comment.get("user", {}).get("login", "Unknown")
        body = comment.get("body", "").strip()
        formatted_comments.append(f"Comment #{i} by {author} in issue #{issue_number}: {body}")

    return "\n\n".join(formatted_comments)


# === Process rows ===
for i, (_, row) in enumerate(df.iterrows()):
    # Get repository from the row (assuming it exists)
    if "search_repository" in row:
        repository = row["search_repository"]
    elif "repository" in row:
        repository = row["repository"]
    else:
        log_activity(f"[{i + 1}/{len(df)}] No repository column found, skipping row")
        continue

    issue_numbers_str = row["issue_numbers"]
    issue_numbers = parse_issue_numbers(issue_numbers_str)

    if not issue_numbers:
        log_activity(
            f"[{i + 1}/{len(df)}] No valid issue numbers found in '{issue_numbers_str}', skipping"
        )
        continue

    log_activity(
        f"[{i + 1}/{len(df)}] Processing issues {issue_numbers} from {repository}"
    )

    # Check if this row has already been processed (using row index as key)
    row_key = f"{repository}_row_{i}"
    if row_key in progress:
        log_activity(f"Skipping already processed row {i}")
        continue

    # Collect all issue data for this row
    all_issues_data = []
    all_comments = []
    
    # Process each issue number
    for issue_number in issue_numbers:
        log_activity(f"Fetching issue #{issue_number} from {repository} ...")
        issue_data = fetch_issue(repository, issue_number, token_cycle)

        if issue_data:
            # Fetch comments
            comments_url = issue_data.get("comments_url")
            comments = []

            if comments_url:
                log_activity(f"Fetching comments for {repository}#{issue_number} ...")
                comments = fetch_issue_comments(comments_url, token_cycle)
                time.sleep(REQUEST_DELAY)

            all_issues_data.append({
                "issue_number": issue_number,
                "issue_id": issue_data.get("id"),
                "issue_title": issue_data.get("title"),
                "issue_body": issue_data.get("body"),
                "issue_state": issue_data.get("state"),
                "issue_created_at": issue_data.get("created_at"),
                "issue_updated_at": issue_data.get("updated_at"),
                "issue_closed_at": issue_data.get("closed_at"),
                "issue_author": (
                    issue_data.get("user", {}).get("login")
                    if issue_data.get("user")
                    else None
                ),
                "issue_url": issue_data.get("html_url"),
                "comments": comments
            })
        else:
            log_activity(f"⚠️ Failed to fetch issue #{issue_number}")

        time.sleep(REQUEST_DELAY)

    if all_issues_data:
        # Combine all issue data into formatted strings
        issue_titles = []
        issue_bodies = []
        issue_states = []
        issue_authors = []
        issue_urls = []
        issue_ids = []
        issue_numbers_list = []
        issue_created_ats = []
        issue_updated_ats = []
        issue_closed_ats = []
        
        # Collect all comments with issue context
        all_formatted_comments = []
        total_comments_count = 0
        
        for issue_info in all_issues_data:
            issue_num = issue_info["issue_number"]
            issue_titles.append(f"issue #{issue_num}: {issue_info['issue_title'] or ''}")
            issue_bodies.append(f"issue #{issue_num}: {issue_info['issue_body'] or ''}")
            issue_states.append(f"issue #{issue_num}: {issue_info['issue_state'] or ''}")
            issue_authors.append(f"issue #{issue_num}: {issue_info['issue_author'] or ''}")
            issue_urls.append(f"issue #{issue_num}: {issue_info['issue_url'] or ''}")
            issue_ids.append(f"issue #{issue_num}: {issue_info['issue_id'] or ''}")
            issue_numbers_list.append(str(issue_num))
            issue_created_ats.append(f"issue #{issue_num}: {issue_info['issue_created_at'] or ''}")
            issue_updated_ats.append(f"issue #{issue_num}: {issue_info['issue_updated_at'] or ''}")
            issue_closed_ats.append(f"issue #{issue_num}: {issue_info['issue_closed_at'] or ''}")
            
            # Format comments with issue context
            if issue_info["comments"]:
                formatted_comments = format_comments(issue_info["comments"], issue_num)
                if formatted_comments:
                    all_formatted_comments.append(formatted_comments)
                total_comments_count += len(issue_info["comments"])

        # Store the processed issues data for this row
        progress[row_key] = {
            # Original row data (all columns from input CSV)
            "original_row_data": row.to_dict(),
            "row_index": i,
            # Combined issue-specific data
            "repository": repository,
            "issue_numbers": "|".join(issue_numbers_list),
            "issue_ids": " \n\n ".join(issue_ids),
            "issue_titles": " \n\n ".join(issue_titles),
            "issue_bodies": " \n\n ".join(issue_bodies),
            "issue_comments": "\n\n".join(all_formatted_comments),
            "issue_comments_count": total_comments_count,
            "issue_states": " \n\n ".join(issue_states),
            "issue_created_ats": " \n\n ".join(issue_created_ats),
            "issue_updated_ats": " \n\n ".join(issue_updated_ats),
            "issue_closed_ats": " \n\n ".join(issue_closed_ats),
            "issue_authors": " \n\n ".join(issue_authors),
            "issue_urls": " \n\n ".join(issue_urls),
        }

        # Save progress after each row
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(progress, f)

        log_activity(f"✓ Saved progress for row {i} ({len(all_issues_data)} issues, {total_comments_count} total comments)")
    else:
        log_activity(f"⚠️ No issues fetched for row {i}")

log_activity(f"✅ Done. Fetched data for {len(progress)} issues.")

# === Clean data for JSON serialization ===
def clean_for_json(obj):
    """Recursively clean data to remove NaN values and make it JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
        return str(obj) if not pd.isna(obj) else None
    else:
        return obj

# === Save to JSON ===
clean_progress = clean_for_json(progress)
with open(JSON_OUTPUT, "w") as f:
    json.dump(clean_progress, f, indent=2)
log_activity(f"📁 Saved JSON results to {JSON_OUTPUT}")

# === Save to CSV ===
csv_rows = []
for key, issue_data in progress.items():
    # Get original row data
    original_data = issue_data.get("original_row_data", {})

    # Create a new row combining original data with issue data
    csv_row = original_data.copy()  # Start with all original columns

    # Add issue-specific columns (prefixed to avoid conflicts)
    csv_row.update(
        {
            "issue_repository": issue_data.get("repository"),
            "issue_numbers": issue_data.get("issue_numbers"),
            "issue_ids": issue_data.get("issue_ids"),
            "issue_titles": issue_data.get("issue_titles"),
            "issue_bodies": issue_data.get("issue_bodies"),
            "issue_comments": issue_data.get("issue_comments"),
            "issue_comments_count": issue_data.get("issue_comments_count"),
            "issue_states": issue_data.get("issue_states"),
            "issue_created_ats": issue_data.get("issue_created_ats"),
            "issue_updated_ats": issue_data.get("issue_updated_ats"),
            "issue_closed_ats": issue_data.get("issue_closed_ats"),
            "issue_authors": issue_data.get("issue_authors"),
            "issue_urls": issue_data.get("issue_urls"),
        }
    )

    csv_rows.append(csv_row)

pd.DataFrame(csv_rows).to_csv(CSV_OUTPUT, index=False)
log_activity(f"📄 Saved CSV results to {CSV_OUTPUT}")
