import csv
import itertools
import json
import os
import pickle
import random
import sys
import time
from http import HTTPStatus

import pandas as pd
import requests
import datetime

# Increase the maximum field size limit
csv.field_size_limit(sys.maxsize)

# Constants / Paths
TOKENS_FILE = "./env/zero-gpt-tokens.txt"
DATASET_NAME = "jabref_prs"
INPUT_CSV_PATH = f"../generation/datasets/sample_by_state_with_generated.csv"
OUTPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection.csv"
PROGRESS_PKL_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection-progress.pkl"
LOG_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection-output.log"
API_URL = "https://api.zerogpt.com/api/detect/detectText"
IGNORE_IS_SIGNIFICANT = True


def load_tokens(tokens_file=TOKENS_FILE):
    with open(tokens_file, "r") as file:
        tokens = file.read().splitlines()
    start_index = random.randint(0, len(tokens) - 1)
    rotated_tokens = tokens[start_index:] + tokens[:start_index]
    return tokens, itertools.cycle(rotated_tokens)


def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)


def is_significant(text, min_chars=250):
    if IGNORE_IS_SIGNIFICANT:
        return True
    return text and len(text.strip()) >= min_chars


def save_progress(df, processed_entries, path=PROGRESS_PKL_PATH):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "df": df.to_dict(orient="records"),
                "processed_entries": list(processed_entries),
            },
            f,
        )


def load_progress(path):
    try:
        with open(path, "rb") as f:
            progress_data = pickle.load(f)
        if (
            not isinstance(progress_data, dict)
            or "df" not in progress_data
            or "processed_entries" not in progress_data
        ):
            return pd.DataFrame(), set()
        return pd.DataFrame(progress_data["df"]), set(
            progress_data["processed_entries"]
        )
    except FileNotFoundError:
        return pd.DataFrame(), set()


def create_detection_entries(row):
    """Create detection entries for original and generated descriptions."""
    row_id = str(row.get("id", ""))
    original_description = row.get("body", "")

    entries = []

    # Original description
    if is_significant(original_description):
        entries.append(
            {
                "id": row_id,
                "type": "original",
                "input_text": original_description,
                "entry_key": f"{row_id}_original",
            }
        )

    # Generated variants
    shot_types = ["zero_shot", "one_shot", "few_shot"]
    for shot_type in shot_types:
        gen_description = row.get(f"generated_description_{shot_type}", "")

        if is_significant(gen_description):
            entries.append(
                {
                    "id": row_id,
                    "type": f"generated_{shot_type}",
                    "input_text": gen_description,
                    "entry_key": f"{row_id}_generated_{shot_type}",
                }
            )

    return entries


def run_detection_on_jabref_prs(
    input_csv_path=INPUT_CSV_PATH,
    output_csv_path=OUTPUT_CSV_PATH,
    progress_pkl_path=PROGRESS_PKL_PATH,
    log_path=LOG_PATH,
    tokens_file=TOKENS_FILE,
    max_retries=20,
    retry_wait_seconds=900,
    batch_size=50,
):
    log_activity("Starting detection on JabRef PRs...")

    tokens, token_iterator = load_tokens(tokens_file)
    df, processed_entries = load_progress(progress_pkl_path)

    # Load input CSV
    log_activity(f"Loading input CSV: {input_csv_path}")
    with open(input_csv_path, mode="r", encoding="utf-8") as input_file:
        reader = list(csv.DictReader(input_file))

    log_activity(f"Loaded {len(reader)} rows from input CSV")

    # Create all detection entries (original + generated variants)
    all_entries = []
    for row in reader:
        entries = create_detection_entries(row)
        all_entries.extend(entries)

    log_activity(f"Created {len(all_entries)} detection entries from {len(reader)} PRs")

    # Filter out already processed entries
    entries_to_process = [
        entry for entry in all_entries if entry["entry_key"] not in processed_entries
    ]

    log_activity(f"Found {len(entries_to_process)} unprocessed entries")

    if not entries_to_process:
        log_activity("All entries already processed!")
        return

    attempt = 0
    while entries_to_process and attempt < max_retries:
        current_batch = entries_to_process[:batch_size]
        log_activity(
            f"Processing batch {attempt + 1}, {len(current_batch)} entries out of {len(entries_to_process)} remaining..."
        )

        new_rows = []
        processed_in_batch = []
        failed_in_batch = []

        for i, entry in enumerate(current_batch):
            entry_key = entry["entry_key"]
            pr_id = entry["id"]
            entry_type = entry["type"]
            input_text = entry["input_text"]

            current_token = next(token_iterator)
            payload = json.dumps({"input_text": input_text})
            headers = {
                "ApiKey": current_token,
                "Content-Type": "application/json",
            }

            try:
                log_activity(
                    f"Processing entry {i+1}/{len(current_batch)}: {entry_key}"
                )

                response = requests.post(API_URL, headers=headers, data=payload)
                response_data = response.json()
                zerogpt_response = json.dumps(response_data)
                code = response_data.get("code")

                if code != HTTPStatus.OK:
                    log_activity(f"❌ Entry {entry_key} failed with code {code}.")
                    failed_in_batch.append(entry)
                    continue  # Skip to next entry on failure

                log_activity(f"✅ Entry {entry_key} processed successfully.")

                # Build output row
                output_row = {
                    "pr_id": pr_id,
                    "entry_key": entry_key,
                    "entry_type": entry_type,
                    "input_text": input_text,
                    "zerogpt_response": zerogpt_response,
                }

                new_rows.append(output_row)
                processed_in_batch.append(entry_key)

                # Small delay between requests
                time.sleep(1)

            except Exception as e:
                zerogpt_response = f"Error: {e}"
                log_activity(f"❌ Error on entry {entry_key}: {e}")
                failed_in_batch.append(entry)
                continue  # Skip on error

        # Save progress after batch
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            processed_entries.update(processed_in_batch)

            # Save progress
            save_progress(df, processed_entries, progress_pkl_path)
            df.drop_duplicates(subset="entry_key", keep="first").to_csv(
                output_csv_path, index=False, encoding="utf-8"
            )

            log_activity(f"Saved progress: {len(new_rows)} new entries processed")

        # Remove processed entries from the list
        entries_to_process = [
            entry
            for entry in entries_to_process
            if entry["entry_key"] not in processed_entries
        ]

        # If we have failures and remaining entries, wait before retry
        if failed_in_batch and entries_to_process:
            log_activity(
                f"⌛ {len(failed_in_batch)} entries failed. Waiting {retry_wait_seconds//60} minutes before retry for {len(entries_to_process)} remaining entries."
            )
            time.sleep(retry_wait_seconds)
            attempt += 1
        elif entries_to_process:
            # Continue processing without waiting if no failures
            log_activity(
                f"Continuing to next batch. {len(entries_to_process)} entries remaining..."
            )
            # Small delay between batches
            time.sleep(5)

    log_activity(f"✅ Detection finished. Output saved to {output_csv_path}")

    # Print summary
    if not df.empty:
        summary = df["entry_type"].value_counts()
        log_activity("Summary by entry type:")
        for entry_type, count in summary.items():
            log_activity(f"  {entry_type}: {count}")


if __name__ == "__main__":
    # Run with larger batch size since API is working
    run_detection_on_jabref_prs(batch_size=20)
