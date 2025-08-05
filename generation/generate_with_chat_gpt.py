import csv
import datetime
import openai
import time
from collections import defaultdict

openai.api_key = "your-api-key"

### === CONFIG === ###
MODE = "few"  # choose: "zero", "one", "few"
EXAMPLE_FILE = "../pr_files/datasets/few_shot_example_pr_files_output.csv"
TARGET_FILE = "../pr_files/datasets/pr_files_output.csv"
OUTPUT_FILE = "./datasets/generated_title_and_pr.csv"
MAX_EXAMPLES = {"zero": 0, "one": 1, "few": 3}[MODE]
LOG_PATH = "./output.log"


def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)


# Load few-shot examples
def load_examples(file_path):
    examples = defaultdict(list)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            examples[row["id"]].append(row)
    return list(examples.values())  # List of list of files per example PR


# Format a multi-file PR into a user prompt
def format_pr_prompt(pr_files):
    parts = []
    for f in pr_files:
        log_activity(f"here:>{f['status']}")
        part = f"""Filename: `{f['filename']}`
Status: {f['status']}
Additions: {f['additions']}, Deletions: {f['deletions']}, Changes: {f['changes']}
Patch:
```diff
{f['patch']}
```"""
        parts.append(part)
    return "Generate a title and description for this PR:\n\n" + "\n\n---\n\n".join(
        parts
    )


# Format the assistant reply for few-shot
def format_assistant_reply(title, description):
    return f"""**Title:** {title}

**Description:** {description}"""


# Build full chat messages
def build_messages(example_blocks, target_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that writes concise and informative GitHub pull request titles and descriptions based on multiple file diffs and metadata.",
        }
    ]
    for pr_files in example_blocks[:MAX_EXAMPLES]:
        user_msg = format_pr_prompt(pr_files)
        # Assumes all rows in pr_files have same title/description
        assistant_msg = format_assistant_reply(
            pr_files[0]["title"], pr_files[0]["description"]
        )
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": target_prompt})
    return messages


# Call ChatGPT
def call_chatgpt(messages):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        log_activity(f"Error: {e}")
        return None


# Group PR rows by pr id
def group_by_pr(csv_file):
    prs = defaultdict(list)
    with open(csv_file, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            prs[row["id"]].append(row)
    return prs, reader.fieldnames


# Parse model output
def extract_title_description(response):
    title = description = ""
    lines = response.split("\n")
    for i, line in enumerate(lines):
        if line.lower().startswith("**title:**"):
            title = line.split("**Title:**", 1)[-1].strip()
        elif line.lower().startswith("**description:**"):
            description = "\n".join(lines[i + 1 :]).strip()
            break
    return title, description


# Main pipeline
def process_all_prs():
    examples = load_examples(EXAMPLE_FILE)
    target_prs, fieldnames = group_by_pr(TARGET_FILE)

    # Add output fields
    out_fields = list(fieldnames or []) + ["generated_title", "generated_description"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for id, files in target_prs.items():
            log_activity(f"Processing PR {id} with {len(files)} files...")

            prompt = format_pr_prompt(files)
            messages = build_messages(examples, prompt)
            response = call_chatgpt(messages)
            time.sleep(1.5)

            if response:
                title, desc = extract_title_description(response)
            else:
                title, desc = "ERROR", "Failed to generate"

            for row in files:
                row["generated_title"] = title
                row["generated_description"] = desc
                writer.writerow(row)


# Run it
if __name__ == "__main__":
    process_all_prs()
