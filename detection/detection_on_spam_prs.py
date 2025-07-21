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
DATASET_NAME = "spam_prs"
INPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}.csv"
OUTPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection.csv"
PROGRESS_PKL_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection-progress.pkl"
LOG_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection-output.log"
API_URL = "https://api.zerogpt.com/api/detect/detectText"


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
    return text and len(text.strip()) >= min_chars


def save_progress(df, path=PROGRESS_PKL_PATH):
    with open(path, "wb") as f:
        pickle.dump({"df": df.to_dict(orient="records")}, f)


def load_progress(path):
    try:
        with open(path, "rb") as f:
            progress_data = pickle.load(f)
        if (
            not isinstance(progress_data, dict)
            or "df" not in progress_data
            or "processed_ids" not in progress_data
        ):
            return pd.DataFrame(), set()
        return progress_data["df"], progress_data["processed_ids"]
    except FileNotFoundError:
        return pd.DataFrame(), set()


def run_detection(
    input_csv_path=INPUT_CSV_PATH,
    output_csv_path=OUTPUT_CSV_PATH,
    progress_pkl_path=PROGRESS_PKL_PATH,
    log_path=LOG_PATH,
    tokens_file=TOKENS_FILE,
    max_retries=20,
    retry_wait_seconds=900,
    batch_size=4,
):
    tokens, token_iterator = load_tokens(tokens_file)
    df, processed_ids = load_progress(progress_pkl_path)

    # Load and filter input
    with open(input_csv_path, mode="r", encoding="utf-8") as input_file:
        reader = list(csv.DictReader(input_file))

    significant_rows = [row for row in reader if is_significant(row.get("body", ""))]

    random.seed(12345)  # fixed seed for deterministic shuffle
    random.shuffle(significant_rows)

    significant_rows = significant_rows[:batch_size]
    rows_to_process = [
        row for row in significant_rows if str(row.get("id", "")) not in processed_ids
    ]

    attempt = 0
    while attempt < max_retries and rows_to_process:
        new_rows = []
        for row in rows_to_process:
            row_id = str(row.get("id", ""))
            description = row.get("body", "")
            zerogpt_response = ""

            current_token = next(token_iterator)
            payload = json.dumps({"input_text": description})
            headers = {
                "ApiKey": current_token,
                "Content-Type": "application/json",
            }

            try:
                response = requests.post(API_URL, headers=headers, data=payload)
                response_data = response.json()
                zerogpt_response = json.dumps(response_data)
                code = response_data.get("code")
                if code != HTTPStatus.OK:
                    log_activity(f"❌ Row ID {row_id} failed with code {code}.")
                    continue  # Skip to next row on failure

                log_activity(f"✅ Row ID {row_id} processed.")

            except Exception as e:
                zerogpt_response = f"Error: {e}"
                log_activity(f"❌ Error on row ID {row_id}: {e}")
                continue  # Skip on error

            output_row = {
                "id": row_id,
                "repository_name_with_owner": row.get("repository_name_with_owner", ""),
                "description": description,
                "url": row.get("url", ""),
                "created_at": row.get("created_at", ""),
                "updated_at": row.get("updated_at", ""),
                "zerogpt_response": zerogpt_response,
            }

            new_rows.append(output_row)

        # Append new rows and update processed_ids
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            processed_ids.update([r["id"] for r in new_rows])

            # Save progress
            save_progress(df, progress_pkl_path)
            df.drop_duplicates(subset="id", keep="first").to_csv(
                output_csv_path, index=False, encoding="utf-8"
            )

        # Prepare for retry: filter out processed rows
        rows_to_process = [
            row
            for row in rows_to_process
            if str(row.get("id", "")) not in processed_ids
        ]

        if rows_to_process:
            log_activity(
                f"⌛ Waiting {retry_wait_seconds//60} minutes before retry #{attempt + 1} for {len(rows_to_process)} remaining rows."
            )
            time.sleep(retry_wait_seconds)

        attempt += 1

    log_activity(f"✅ Detection finished. Output saved to {output_csv_path}")


if __name__ == "__main__":
    run_detection()
