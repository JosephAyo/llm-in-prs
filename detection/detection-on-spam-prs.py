# RUN detection tool on already labeled spam PRs
import csv
from http import HTTPStatus
import json
import sys
import requests
import itertools
import random
import os
import datetime
from http import HTTPStatus
import csv
import json
import os
import pickle
import random
import requests
import datetime
import itertools
import pandas as pd
from http import HTTPStatus

# Increase the maximum field size limit
csv.field_size_limit(sys.maxsize)

# Load API tokens
tokens_file = "./env/zero-gpt-tokens.txt"
with open(tokens_file, "r") as file:
    tokens = file.read().splitlines()

# Rotate tokens randomly
start_index = random.randint(0, len(tokens) - 1)
rotated_tokens = tokens[start_index:] + tokens[:start_index]
token_iterator = itertools.cycle(rotated_tokens)

# API endpoint
url = "https://api.zerogpt.com/api/detect/detectText"

# Dataset paths
dataset_name = "spam_prs"
input_csv_path = f"./datasets/{dataset_name}/{dataset_name}.csv"
output_csv_path = f"./datasets/{dataset_name}/{dataset_name}-detection.csv"
progress_pkl_path = f"./datasets/{dataset_name}/{dataset_name}-detection-progress.pkl"
log_path = f"./datasets/{dataset_name}/{dataset_name}-detection-output.log"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Resume from previous progress if available
if os.path.exists(progress_pkl_path):
    with open(progress_pkl_path, "rb") as f:
        progress_data = pickle.load(f)
        df_list = progress_data["df"]

        # Deduplicate by 'id'
        if isinstance(df_list, list):
            df = pd.DataFrame(df_list)
        elif isinstance(df_list, pd.DataFrame):
            df = df_list
        else:
            df = pd.DataFrame()

        df = df.drop_duplicates(subset="id", keep="first")
        processed_ids = set(df["id"].astype(str))
    print(f"Resuming. {len(processed_ids)} entries already processed.")
else:
    df = pd.DataFrame()
    processed_ids = set()


# Logging helper
def log_activity(activity: str):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)


# Filter logic
def is_significant(text, min_chars=250):
    return text and len(text.strip()) >= min_chars


# Read input and write output
new_rows = []
# Set seed for reproducibility
random.seed(42)

# Load and shuffle rows
with open(input_csv_path, mode="r", encoding="utf-8") as input_file:
    reader = list(csv.DictReader(input_file))

    # Only keep significant rows
    significant_rows = [row for row in reader if is_significant(row.get("body", ""))]

    # Shuffle and pick first 100
    random.shuffle(significant_rows)
    significant_rows = significant_rows[:100]

    for row in significant_rows:
        row_id = str(row.get("id", ""))
        if row_id in processed_ids:
            continue

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
            code = response_data.get("code")
            if code != HTTPStatus.OK:
                raise Exception(f"Detection failed: {code}")

            log_activity(f"✅ Row ID {row_id} processed.")

        except Exception as e:
            zerogpt_response = f"Error: {e}"
            log_activity(f"❌ Error on row ID {row_id}: {e}")

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
        df = pd.concat([df, pd.DataFrame([output_row])], ignore_index=True)

        # Periodically save progress
        if len(new_rows) % 10 == 0:
            with open(progress_pkl_path, "wb") as f:
                pickle.dump({"df": df.to_dict(orient="records")}, f)
            log_activity("💾 Progress saved.")

# Final save
with open(progress_pkl_path, "wb") as f:
    pickle.dump({"df": df.to_dict(orient="records")}, f)

# Write full CSV
df.drop_duplicates(subset="id", keep="first").to_csv(
    output_csv_path, index=False, encoding="utf-8"
)

log_activity(f"✅ Detection finished. Output saved to {output_csv_path}")
