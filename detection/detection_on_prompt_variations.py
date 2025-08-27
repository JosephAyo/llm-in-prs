import csv
import itertools
import json
import os
import pickle
import random
import sys
import time
from http import HTTPStatus
import glob

import pandas as pd
import requests
import datetime

# Increase the maximum field size limit
csv.field_size_limit(sys.maxsize)

# Constants / Paths
TOKENS_FILE = "./env/zero-gpt-tokens.txt"
DATASET_NAME = "prompt_variations"
INPUT_CSV_PATTERN = f"../generation/datasets/prompt_variation_*_generated.csv"
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


def create_detection_entries(row, csv_filename):
    """Create detection entries for original and generated descriptions."""
    row_id = str(row.get("id", ""))
    prompt_variation = row.get("prompt_variation", csv_filename.split("_generated.csv")[0].replace("prompt_variation_", ""))
    original_description = row.get("description", "")
    generated_description = row.get("generated_description", "")

    entries = []

    # Original description
    if is_significant(original_description):
        entries.append(
            {
                "id": row_id,
                "prompt_variation": prompt_variation,
                "type": "original",
                "input_text": original_description,
                "entry_key": f"{row_id}_{prompt_variation}_original",
            }
        )

    # Generated description
    if is_significant(generated_description):
        entries.append(
            {
                "id": row_id,
                "prompt_variation": prompt_variation,
                "type": "generated",
                "input_text": generated_description,
                "entry_key": f"{row_id}_{prompt_variation}_generated",
            }
        )

    return entries


def run_detection_on_prompt_variations(
    input_csv_pattern=INPUT_CSV_PATTERN,
    output_csv_path=OUTPUT_CSV_PATH,
    progress_pkl_path=PROGRESS_PKL_PATH,
    log_path=LOG_PATH,
    tokens_file=TOKENS_FILE,
    max_retries=20,
    retry_wait_seconds=900,
    batch_size=50,
):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    log_activity("Starting detection on prompt variation CSV files...")

    tokens, token_iterator = load_tokens(tokens_file)
    df, processed_entries = load_progress(progress_pkl_path)

    # Find all prompt variation CSV files
    csv_files = glob.glob(input_csv_pattern)
    log_activity(f"Found {len(csv_files)} prompt variation CSV files")

    # Create all detection entries (original + generated variants) from all CSV files
    all_entries = []
    for csv_file in csv_files:
        csv_filename = os.path.basename(csv_file)
        log_activity(f"Processing file: {csv_filename}")
        
        with open(csv_file, mode="r", encoding="utf-8") as input_file:
            reader = list(csv.DictReader(input_file))
            
        log_activity(f"  Loaded {len(reader)} rows from {csv_filename}")

        # Create detection entries for this file
        for row in reader:
            entries = create_detection_entries(row, csv_filename)
            all_entries.extend(entries)

    log_activity(f"Created {len(all_entries)} detection entries from {len(csv_files)} CSV files")

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
            prompt_variation = entry["prompt_variation"]
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
                    "prompt_variation": prompt_variation,
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
        summary_by_type = df["entry_type"].value_counts()
        summary_by_variation = df["prompt_variation"].value_counts()
        
        log_activity("Summary by entry type:")
        for entry_type, count in summary_by_type.items():
            log_activity(f"  {entry_type}: {count}")
        
        log_activity("Summary by prompt variation:")
        for variation, count in summary_by_variation.items():
            log_activity(f"  {variation}: {count}")


if __name__ == "__main__":
    # Run with smaller batch size to be conservative with API
    run_detection_on_prompt_variations(batch_size=20)
