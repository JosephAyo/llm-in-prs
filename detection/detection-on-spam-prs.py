# RUN detection tool on already labeled spam PRs
import csv
from http import HTTPStatus
import json
import requests
import itertools
import random
import os
import datetime
from http import HTTPStatus

# Read tokens from a text file
tokens_file = "./env/zero-gpt-tokens.txt"
with open(tokens_file, "r") as file:
    tokens = file.read().splitlines()

# Choose a random start index
start_index = random.randint(0, len(tokens) - 1)

# Rotate the tokens list starting at a random position
rotated_tokens = tokens[start_index:] + tokens[:start_index]

# Create an infinite cycle iterator from the rotated list
token_iterator = itertools.cycle(rotated_tokens)

# Define API endpoint
url = "https://api.zerogpt.com/api/detect/detectText"

# CSV input and output file paths
dataset_name = "spam_prs"
input_csv_path = f"./datasets/{dataset_name}/{dataset_name}.csv"
output_csv_path = f"./datasets/{dataset_name}/{dataset_name}-detection.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)


def log_activity(activity: str):
    log = f"{datetime.datetime.now()}: {activity}\n"
    # print(log)
    with open(
        f"./datasets/{dataset_name}/{dataset_name}-detection-output.log", "a"
    ) as log_file:
        log_file.write(log)


def is_significant(text, min_chars=250):
    return text and len(text.strip()) >= min_chars


# Read input and prepare to write output
with open(input_csv_path, mode="r", encoding="utf-8") as input_file, open(
    output_csv_path, mode="w", encoding="utf-8", newline=""
) as output_file:

    reader = csv.DictReader(input_file)

    # Filter only significant rows first, then take first 20
    significant_rows = (row for row in reader if is_significant(row.get("body", "")))
    rows = itertools.islice(significant_rows, 1)
    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "id",
            "repository_name_with_owner",
            "url",
            "created_at",
            "updated_at",
            "description",
            "zerogpt_response",
        ],
    )

    writer.writeheader()

    for row in rows:
        description = row.get("body", "")
        zerogpt_response = ""

        current_token = next(token_iterator)
        payload = json.dumps({"input_text": description})
        headers = {
            "ApiKey": current_token,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, data=payload)
            response_data = response.json()
            zerogpt_response = json.dumps(response_data)
            status_code = response_data.get("code")

            # Ensure status code is 200 OK
            if status_code != HTTPStatus.OK:
                raise Exception(f"Detection failed: {status_code}")

        except Exception as e:
            zerogpt_response = f"Error: {e}"
            log_activity(f"Error processing row id {row.get('id', '')}: {e}")
        else:
            log_activity(f"Successfully processed row id {row.get('id', '')}")

        # Write results to CSV
        output_row = {
            "id": row.get("id", ""),
            "repository_name_with_owner": row.get("repository_name_with_owner", ""),
            "description": description,
            "url": row.get("url", ""),
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
            "zerogpt_response": zerogpt_response,
        }
        writer.writerow(output_row)
