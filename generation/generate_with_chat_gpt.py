import csv
import datetime
import json
import openai
import time
from collections import defaultdict
import tiktoken
import math
import sys

MODEL_NAME = "gpt-4.1-mini-2025-04-14"
# Initialize encoding once (choose the same model as used in OpenAI calls)
ENCODER = tiktoken.encoding_for_model(MODEL_NAME)  # or "gpt-4.1" etc.

# 🔐 Load OpenAI API key
with open("./env/tokens.txt", "r") as f:
    openai.api_key = f.read().strip()

### === CONFIG === ###
MODE = "few"  # choose: "zero", "one", "few"
EXAMPLE_FILE = "../pr_files/datasets/sample_additional_pr_files_output.csv"
TARGET_FILE = "../pr_files/datasets/sample_by_state_pr_files_output.csv"
OUTPUT_FILE = f"./datasets/{MODE}_shot_generated.csv"
OUTPUT_JSON_FILE = f"./datasets/{MODE}_shot_generated.json"
LOG_PATH = f"./datasets/{MODE}_shot_output.log"
MAX_EXAMPLES = {"zero": 0, "one": 1, "few": 3}[MODE]
MAX_PROMPT_TOKENS = 8000  # Leave room below 10k TPM
USE_MOCK = False
INTERMEDIATE_FILE = "./datasets/intermediate_chunk_outputs.csv"


# === Utility: Logging ===
def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)


# === Utility: Estimate token count ===
def estimate_tokens(text, use_real=True):
    """
    Estimate token count.
    If use_real=True, use tiktoken. Otherwise, fall back to /4 heuristic.
    """
    if not text:
        return 0
    if use_real:
        return len(ENCODER.encode(text))
    return len(text) // 4


def estimate_tokens_bytes(file_size_bytes):
    """
    Estimate token count based on file size in bytes.
    Handles string, int, float, and NaN values gracefully.
    """
    if file_size_bytes is None:
        return 0
    try:
        # Convert to float first to handle both int and string representations
        size = float(file_size_bytes)
        if math.isnan(size):
            return 0
        return int(size // 3)
    except (ValueError, TypeError):
        return 0


# === Chunk PR files if total tokens too high ===
def chunk_pr_files(pr_files, max_tokens=MAX_PROMPT_TOKENS):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for f in pr_files:
        title_tokens = estimate_tokens(f["title"], True)
        patch_tokens = estimate_tokens(f["patch"], True)
        file_content_tokens = estimate_tokens_bytes(
            f["file_size_bytes"],
        )
        meta_tokens = estimate_tokens(f["filename"] + f["status"], True) + 10
        total_tokens = title_tokens + patch_tokens + file_content_tokens + meta_tokens

        if current_token_count + total_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(f)
        current_token_count += total_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# === Load few-shot examples grouped by PR ===
def load_examples(file_path):
    examples = defaultdict(list)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            examples[row["id"]].append(row)
    return list(examples.values())  # List of list of files per example PR


# === Format full PR as prompt ===
def format_pr_prompt(pr_files):
    parts = []
    for f in pr_files:
        patch_lines = f["patch"].splitlines()
        trimmed_patch = "\n".join(patch_lines)
        part = f"""
        Title: `{f['title']}`
        Filename: `{f['filename']}`
        Status: {f['status']}
        Additions: {f['additions']}, Deletions: {f['deletions']}, Changes: {f['changes']}
        File Content: ```{f['file_content']}```
        Patch:
        ```diff
        {trimmed_patch}
        ```"""
        parts.append(part)
    return "Generate a description for this PR:\n\n" + "\n\n---\n\n".join(parts)


# === Format assistant example ===
def format_assistant_reply(title, description):
    return json.dumps({"title": title, "description": description}, ensure_ascii=False)


# === Trim message content to fit token limits ===
def trim_message_content(content, max_tokens=2000):
    """Trim message content to stay within token limits for examples."""
    if estimate_tokens(content) <= max_tokens:
        return content

    lines = content.splitlines()
    trimmed_lines = []
    current_tokens = 0

    # Keep the first line (usually the instruction)
    if lines:
        trimmed_lines.append(lines[0])
        current_tokens += estimate_tokens(lines[0])

    # Add lines until we hit the limit
    for line in lines[1:]:
        line_tokens = estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens:
            trimmed_lines.append("... [content trimmed for brevity] ...")
            break
        trimmed_lines.append(line)
        current_tokens += line_tokens

    return "\n".join(trimmed_lines)


# === Build full message list ===
def build_messages(example_blocks, target_prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that writes concise and informative GitHub pull request descriptions based on title, multiple code file contents and file diffs and metadata. "
                "Respond ONLY with a valid JSON object in the following format (do not include any extra text):\n"
                '{"description": "<description>"}'
            ),
        }
    ]
    for pr_files in example_blocks[:MAX_EXAMPLES]:
        user_msg = format_pr_prompt(pr_files)
        trimmed_user_msg = trim_message_content(user_msg, max_tokens=2000)
        assistant_msg = format_assistant_reply(
            pr_files[0]["title"], pr_files[0]["description"]
        )
        messages.append({"role": "user", "content": trimmed_user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": target_prompt})
    return messages


def mock_chatgpt(messages):
    user_msg = messages[-1]["content"]
    file_count = user_msg.count("Filename:")
    filenames = [
        line.split("`")[1]
        for line in user_msg.splitlines()
        if line.startswith("Filename:")
    ]
    joined_files = ", ".join(filenames[:3]) + ("..." if len(filenames) > 3 else "")
    return json.dumps(
        {
            "description": f"This mock PR updates {file_count} file(s) including {joined_files}. It includes code changes such as additions, deletions, and modifications. Use this mock description to verify formatting and CSV outputs.",
        },
        ensure_ascii=False,
    )


# === Call GPT API ===
def call_chatgpt(messages, max_retries=3):
    if USE_MOCK:
        return mock_chatgpt(messages)

    retries = 0
    while retries < max_retries:
        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            log_activity(f"Error: {error_msg}")
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                wait_time = 65  # 1 minute and a bit
                log_activity(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                break
    return None


def save_intermediate_chunks(pr_id, chunk_outputs, output_file=INTERMEDIATE_FILE):
    fieldnames = ["pr_id", "chunk_index", "chunk_title", "chunk_description"]
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for i, (title, desc) in enumerate(chunk_outputs):
            writer.writerow(
                {
                    "pr_id": pr_id,
                    "chunk_index": i,
                    "chunk_title": title,
                    "chunk_description": desc,
                }
            )


def merge_descriptions_with_gpt(descriptions):
    if len(descriptions) == 1:
        # If only one description, return it directly as JSON
        return json.dumps({"description": descriptions[0]}, ensure_ascii=False)
    prompt = "\n\nDescriptions from PR chunks to merge into a single, clear, PR description. Use the original PR title as the final title.\n\nDescriptions:\n"
    for i, d in enumerate(descriptions):
        prompt += f"Part {i+1}: {d}\n\n"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You are given multiple descriptions generated from chunks of a pull request. "
                "Your task is to merge them into a single, clear, PR description. Respond ONLY with a valid JSON object in the following format (do not include any extra text):\n"
                '{"description": "<merged description>"}'
            ),
        },
        {"role": "user", "content": prompt.strip()},
    ]
    return call_chatgpt(messages)


# === Group files by PR ID ===
def group_by_pr(csv_file):
    prs = defaultdict(list)
    try:
        # Increase CSV field size limit to handle large fields
        csv.field_size_limit(sys.maxsize)
        with open(csv_file, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                log_activity(f"Error: CSV file {csv_file} has no header/fieldnames.")
                return prs, []
            row_count = 0
            for row in reader:
                if not row or not row.get("id"):
                    continue
                prs[row["id"]].append(row)
                row_count += 1
            if row_count == 0:
                log_activity(
                    f"Warning: CSV file {csv_file} is empty or has no valid rows."
                )
    except Exception as e:
        log_activity(f"Error reading CSV file {csv_file}: {e}")
        return prs, []
    return prs, reader.fieldnames


# === Extract title/description from GPT response ===
def extract_title_description(response, original_title=None):
    try:
        data = json.loads(response)
        # If only description is present, use original_title
        title = data.get("title", original_title or "")
        description = data.get("description", "")
        return title, description
    except Exception:
        # fallback to old extraction if not valid JSON
        title = original_title or ""
        description = ""
        lines = response.splitlines()
        for i, line in enumerate(lines):
            if "**Description:**" in line or "Description:" in line:
                desc_part = line.split("Description:", 1)[-1].replace("**", "").strip()
                rest = [desc_part] if desc_part else []
                rest += lines[i + 1 :]
                description = "\n".join(rest).strip()
                break
        return title, description


# === MAIN PIPELINE ===
def process_all_prs():
    examples = load_examples(EXAMPLE_FILE)
    target_prs, fieldnames = group_by_pr(TARGET_FILE)
    out_fields = list(fieldnames or []) + ["generated_description"]

    log_activity(f"target_prs len {len(target_prs)}")
    # Collect all data for JSON output
    all_data = []

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for pr_idx, (pr_id, files) in enumerate(list(target_prs.items())):
            log_activity(f"Processing PR {pr_id} with {len(files)} files...")

            # Access pr_total_size_bytes from target_prs if present
            pr_total_size_bytes = (
                target_prs[pr_id][0].get("pr_total_size_bytes")
                if target_prs[pr_id] and "pr_total_size_bytes" in target_prs[pr_id][0]
                else None
            )
            log_activity(f"PR {pr_id} total size (bytes): {pr_total_size_bytes}")

            file_chunks = chunk_pr_files(files)
            chunk_outputs = []

            for i, chunk in enumerate(file_chunks):
                log_activity(f"  Chunk {i+1}/{len(file_chunks)} for PR {pr_id}")
                prompt = format_pr_prompt(chunk)
                messages = build_messages(examples, prompt)
                # Log the messages as JSON
                log_activity(
                    f"Messages for PR {pr_id} chunk {i+1}:\n"
                    + json.dumps(messages, indent=2, ensure_ascii=False)
                )
                response = call_chatgpt(messages)
                time.sleep(1.5)

                if response:
                    title, desc = extract_title_description(response)
                else:
                    title, desc = "ERROR", "Failed to generate"

                chunk_outputs.append((title, desc))

            save_intermediate_chunks(pr_id, chunk_outputs)

            # Combine chunk outputs
            all_titles = [title for title, _ in chunk_outputs]
            all_descriptions = [desc for _, desc in chunk_outputs]

            merged_response = merge_descriptions_with_gpt(all_descriptions)

            # Use the original title from the first chunk
            original_title = all_titles[0] if all_titles else ""
            if merged_response:
                final_title, combined_desc = extract_title_description(
                    merged_response, original_title=original_title
                )
            else:
                combined_desc = "\n\n".join(all_descriptions)

            for row in files:
                row["generated_description"] = combined_desc
                filtered_row = {k: row.get(k, "") for k in out_fields}
                writer.writerow(filtered_row)
                all_data.append(dict(row))  # Add to JSON data collection

    # Save as JSON
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as json_file:
        json.dump(all_data, json_file, indent=2, ensure_ascii=False)

    log_activity(f"Saved CSV output to {OUTPUT_FILE}")
    log_activity(f"Saved JSON output to {OUTPUT_JSON_FILE}")


# Run it
if __name__ == "__main__":
    process_all_prs()
