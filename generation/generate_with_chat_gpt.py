import csv
import datetime
import json
import openai
import time
from collections import defaultdict

# 🔐 Load OpenAI API key
with open("./env/tokens.txt", "r") as f:
    openai.api_key = f.read().strip()

### === CONFIG === ###
MODE = "few"  # choose: "zero", "one", "few"
EXAMPLE_FILE = "../pr_files/datasets/few_shot_example_pr_files_output.csv"
TARGET_FILE = "../pr_files/datasets/pr_files_output.csv"
OUTPUT_FILE = f"./datasets/generated_title_and_pr_{MODE}_shot.csv"
OUTPUT_JSON_FILE =f"./datasets/generated_title_and_pr_{MODE}_shot.json"
LOG_PATH = f"./datasets/output_{MODE}_shot.log"
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
def estimate_tokens(text):
    return len(text) // 4  # Approximate: 1 token ≈ 4 characters


# === Chunk PR files if total tokens too high ===
def chunk_pr_files(pr_files, max_tokens=MAX_PROMPT_TOKENS):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for f in pr_files:
        patch_tokens = estimate_tokens(f["patch"])
        meta_tokens = estimate_tokens(f["filename"] + f["status"]) + 10
        file_tokens = patch_tokens + meta_tokens

        if current_token_count + file_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(f)
        current_token_count += file_tokens

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
        part = f"""Filename: `{f['filename']}`
Status: {f['status']}
Additions: {f['additions']}, Deletions: {f['deletions']}, Changes: {f['changes']}
Patch:
```diff
{trimmed_patch}
```"""
        parts.append(part)
    return "Generate a title and description for this PR:\n\n" + "\n\n---\n\n".join(
        parts
    )


# === Format assistant example ===
def format_assistant_reply(title, description):
    return f"""**Title:** {title}

**Description:** {description}"""


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
            "content": "You are a helpful assistant that writes concise and informative GitHub pull request titles and descriptions based on multiple file diffs and metadata.",
        }
    ]
    for pr_files in example_blocks[:MAX_EXAMPLES]:
        user_msg = format_pr_prompt(pr_files)
        # Trim the user message for few-shot examples to prevent token overflow
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

    return f"""**Title:** Update {file_count} files: {joined_files}

**Description:** This mock PR updates {file_count} file(s) including {joined_files}. It includes code changes such as additions, deletions, and modifications. Use this mock description to verify formatting and CSV outputs."""


# === Call GPT API ===
def call_chatgpt(messages):
    if USE_MOCK:
        return mock_chatgpt(messages)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        log_activity(f"Error: {e}")
        return None

def save_intermediate_chunks(pr_id, chunk_outputs, output_file=INTERMEDIATE_FILE):
    fieldnames = ["pr_id", "chunk_index", "chunk_title", "chunk_description"]
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for i, (title, desc) in enumerate(chunk_outputs):
            writer.writerow({
                "pr_id": pr_id,
                "chunk_index": i,
                "chunk_title": title,
                "chunk_description": desc,
            })


def merge_titles_and_descriptions_with_gpt(titles, descriptions):
    prompt = (
        "You are a helpful assistant. You are given multiple titles and descriptions generated from chunks of a pull request.\n"
        "Your task is to merge them into a single, clear, professional PR title and description.\n\n"
        "### Titles:\n"
    )

    for i, t in enumerate(titles):
        prompt += f"- {t}\n"

    prompt += "\n### Descriptions:\n"

    for i, d in enumerate(descriptions):
        prompt += f"Part {i+1}:\n{d}\n\n"

    prompt += "\nRespond in the following format:\n\n**Title:** <merged title>\n\n**Description:** <merged description>"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that merges multiple PR titles and descriptions into one cohesive summary.",
        },
        {"role": "user", "content": prompt.strip()},
    ]

    return call_chatgpt(messages)


# === Group files by PR ID ===
def group_by_pr(csv_file):
    prs = defaultdict(list)
    with open(csv_file, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            prs[row["id"]].append(row)
    return prs, reader.fieldnames

# === Extract title/description from GPT response ===
def extract_title_description(response):
    title = ""
    description = ""
    lines = response.splitlines()
    log_activity(f"lines:>>{lines}")
    for i, line in enumerate(lines):
        if "**Title:**" in line or "Title:" in line:
            title = line.split("Title:", 1)[-1].replace("**", "").strip()
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
    out_fields = list(fieldnames or []) + ["generated_title", "generated_description"]
    
    # Collect all data for JSON output
    all_data = []

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for pr_idx, (pr_id, files) in enumerate(list(target_prs.items())):
            log_activity(f"Processing PR {pr_id} with {len(files)} files...")

            file_chunks = chunk_pr_files(files)
            chunk_outputs = []

            for i, chunk in enumerate(file_chunks):
                log_activity(f"  Chunk {i+1}/{len(file_chunks)} for PR {pr_id}")
                prompt = format_pr_prompt(chunk)
                messages = build_messages(examples, prompt)
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

            merged_response = merge_titles_and_descriptions_with_gpt(all_titles, all_descriptions)

            if merged_response:
                final_title, combined_desc = extract_title_description(merged_response)
            else:
                final_title = all_titles[0]
                combined_desc = "\n\n".join(all_descriptions)

            for row in files:
                row["generated_title"] = final_title
                row["generated_description"] = combined_desc
                writer.writerow(row)
                all_data.append(dict(row))  # Add to JSON data collection

    # Save as JSON
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as json_file:
        json.dump(all_data, json_file, indent=2, ensure_ascii=False)
    
    log_activity(f"Saved CSV output to {OUTPUT_FILE}")
    log_activity(f"Saved JSON output to {OUTPUT_JSON_FILE}")


# Run it
if __name__ == "__main__":
    process_all_prs()
