import csv
import datetime
import json
import openai
import os
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

### === PROMPT VARIATION MATRIX === ###
# Updated prompt variations based on research matrix - September 2025
# This matrix defines 6 progressive prompt variations for PR description generation
# Each variation builds upon the previous one by adding more context components
# 
# Matrix components:
# - repo_name_and_path: Repository name and file pathnames (always included)
# - pr_template_content: PR template guidelines 
# - pr_title: Pull request title
# - pr_diffs: Code diffs/changes
# - pr_file_contents: Full file contents
# - pr_issue_context: Related issue information
#
# Progression: P-1 (minimal) → P-6 (full context)
PROMPT_VARIATIONS = {
    # P-1: Minimal - Only repository name and file pathnames
    "P-1": {
        "repo_name_and_path": True,
        "pr_template_content": False,
        "pr_title": False,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
    # P-2: Add PR template content
    "P-2": {
        "repo_name_and_path": True,
        "pr_template_content": True,
        "pr_title": False,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
    # P-3: Add PR title
    "P-3": {
        "repo_name_and_path": True,
        "pr_template_content": True,
        "pr_title": True,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
    # P-4: Add PR diffs
    "P-4": {
        "repo_name_and_path": True,
        "pr_template_content": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
    # P-5: Add PR file contents
    "P-5": {
        "repo_name_and_path": True,
        "pr_template_content": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": False,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
    # P-6: Full context - Add PR issue context
    "P-6": {
        "repo_name_and_path": True,
        "pr_template_content": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": True,
        "few_shot_examples": False,
        "one_shot_examples": False,
    },
}

### === CONFIG === ###
# Modified to support updated prompt variations (P-1 through P-6)
PROMPT_VARIATION = "P-1"  # Change this to test different variations (P-1, P-2, P-3, P-4, P-5, P-6)
EXAMPLE_FILE = "../pr_files/datasets/few_shot_examples_with_issues_and_files.csv"
TARGET_FILE = "../pr_files/datasets/golden_dataset_with_issues_and_files.csv"
OUTPUT_FILE = f"./datasets/prompt_variation_{PROMPT_VARIATION}_generated.csv"
OUTPUT_JSON_FILE = f"./datasets/prompt_variation_{PROMPT_VARIATION}_generated.json"
LOG_PATH = f"./datasets/prompt_variation_{PROMPT_VARIATION}_output.log"
MAX_EXAMPLES = 3  # Will be overridden based on variation
MAX_PROMPT_TOKENS = 32000
USE_MOCK = False
INTERMEDIATE_FILE = (
    f"./datasets/prompt_variation_{PROMPT_VARIATION}_intermediate_chunks.csv"
)


# === Utility: Logging ===
def log_activity(activity: str):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(LOG_PATH, "a") as log_file:
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
    try:
        # Increase CSV field size limit to handle large fields
        csv.field_size_limit(sys.maxsize)
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                examples[row["id"]].append(row)
    except Exception as e:
        log_activity(f"Error loading examples from {file_path}: {e}")
        return []
    return list(examples.values())  # List of list of files per example PR


# === Extract repository name and path from PR files ===
def extract_repo_info(pr_files):
    """Extract repository name, path, and file statistics from PR files."""
    if not pr_files:
        return "Repository: Unknown"
    
    repo = pr_files[0].get("repository", "")
    repo_info = [f"Repository: {repo}"]
    
    # Add file statistics for each file
    file_stats = []
    for f in pr_files:
        filename = f.get("filename", "")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        changes = f.get("changes", 0)
        file_stats.append(f"Filename: `{filename}`, Additions: {additions}, Deletions: {deletions}, Changes: {changes}")
    
    if file_stats:
        repo_info.append("Files:")
        repo_info.extend(file_stats)
    
    return "\n".join(repo_info)


# === Format PR template content ===
def format_template_content():
    """Format template content (PR template placeholder)."""
    return """PR Template Guidelines:
- Describe the changes you have made here: what, why
- Link the issue that will be closed
- Add tests for changes if applicable
- Manually test changed features
- Update documentation if needed"""


# === Format issue context ===
def format_issue_context(pr_files):
    """Format issue context from issue_titles and issue_bodies (ignore issue_comments)."""
    if not pr_files:
        return ""

    # Get issue context from the first file (assuming all files in PR have same issue info)
    first_file = pr_files[0]
    issue_titles = first_file.get("issue_titles", "")
    issue_bodies = first_file.get("issue_bodies", "")

    context_parts = []
    if issue_titles:
        context_parts.append(f"Related Issues:\n{issue_titles}")
    if issue_bodies:
        context_parts.append(f"Issue Details:\n{issue_bodies}")

    return "\n\n".join(context_parts) if context_parts else ""


# === Format PR prompt with variation support ===
def format_pr_prompt_with_variation(pr_files, prompt_variation_key):
    """
    Format PR prompt based on the specified variation.
    Only includes components that are True in the variation matrix.
    """
    if prompt_variation_key not in PROMPT_VARIATIONS:
        raise ValueError(f"Unknown prompt variation: {prompt_variation_key}")

    variation = PROMPT_VARIATIONS[prompt_variation_key]
    parts = []

    # --- Prompt Structure and Delimiters ---
    # Minimal prompt (P-1): simple instruction, no extra structure
    if prompt_variation_key == "P-1":
        repo_info = (
            extract_repo_info(pr_files) if variation["repo_name_and_path"] else ""
        )
        return f"Generate a concise description for this PR.\n\n{repo_info}"

    # For all other prompts, use delimiters and explicit instructions
    # Always repeat instructions before and after context
    # Add planning/outline step and negative prompt
    instructions = []
    instructions.append(
        """### INSTRUCTIONS
You are to write a concise, clear, and informative GitHub pull request description based on the provided context.
Follow these steps:
1. Read all provided context and code carefully.
2. First, list the major changes or bullet points summarizing the PR (planning step).
3. Then, write a final JSON object with a single, cohesive description.
4. If available, use the PR title, issue context, and template guidelines to inform your summary.
5. Do not copy code; describe what it does.
6. Do not speculate on motivation or add information not present in the files, title, or issues.
7. Respond ONLY with a valid JSON object: {\"description\": \"<description>\"}
### END INSTRUCTIONS"""
    )

    # Adaptive chunking note
    chunk_note = ""
    # If chunked, add a note for the model
    # (Assume pr_files has an attribute or pass chunk info; here, check for a 'chunk_index' or similar)
    chunk_count = getattr(pr_files, 'chunk_count', None) or None
    chunk_index = getattr(pr_files, 'chunk_index', None) or None
    if chunk_count and chunk_count > 1:
        part_num = str(chunk_index + 1) if chunk_index is not None else '?'
        chunk_note = f"\n\nNOTE: This is part {part_num} of {chunk_count}. At the end, you will be asked to merge all summaries into a single, cohesive PR description."

    # --- Context Section ---
    context_parts = []
    if variation["repo_name_and_path"]:
        repo_info = extract_repo_info(pr_files)
        context_parts.append(f"<repo>\n{repo_info}\n</repo>")
    if variation.get("pr_title") and pr_files:
        title = pr_files[0]["title"]
        context_parts.append(f"<title>\n{title}\n</title>")
    if variation.get("pr_issue_context"):
        issue_context = format_issue_context(pr_files)
        if issue_context:
            context_parts.append(f"<issue_context>\n{issue_context}\n</issue_context>")
    if variation.get("pr_template_content"):
        template_content = format_template_content()
        context_parts.append(f"<template>\n{template_content}\n</template>")

    # File-specific content
    file_parts = []
    for f in pr_files:
        file_part_components = []
        file_part_components.append(f"Filename: `{f['filename']}`")
        file_part_components.append(f"Status: {f['status']}")
        file_part_components.append(
            f"Additions: {f['additions']}, Deletions: {f['deletions']}, Changes: {f['changes']}"
        )
        if variation.get("pr_file_contents"):
            file_content = f.get("file_content", "")
            if file_content:
                file_part_components.append(
                    f"File Content:\n" + '"""' + f"\n{file_content}\n" + '"""'
                )
        if variation.get("pr_diffs"):
            patch_lines = f["patch"].splitlines()
            trimmed_patch = "\n".join(patch_lines)
            file_part_components.append(f"Patch:\n```diff\n{trimmed_patch}\n```")
        file_parts.append("\n".join(file_part_components))
    if file_parts:
        context_parts.append(
            "<files>\n" + "\n\n---\n\n".join(file_parts) + "\n</files>"
        )

    # Combine all
    prompt_sections = []
    prompt_sections.append(instructions[0])
    if chunk_note:
        prompt_sections.append(chunk_note)
    prompt_sections.append(
        """### CONTEXT
"""
        + "\n\n".join(context_parts)
        + "\n### END CONTEXT"
    )
    # Always repeat instructions after context for all variations
    prompt_sections.append(instructions[0])

    return "\n\n".join(prompt_sections)


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


# === Build full message list with variation support ===
def build_messages_with_variation(example_blocks, target_prompt, prompt_variation_key):
    """Build messages with support for prompt variations."""
    variation = PROMPT_VARIATIONS[prompt_variation_key]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that writes concise and informative GitHub pull request descriptions based on the provided information. "
                "Respond ONLY with a valid JSON object in the following format (do not include any extra text):\n"
                '{"description": "<description>"}'
            ),
        }
    ]

    # Add examples based on variation type
    if variation.get("few_shot_examples"):
        max_examples = MAX_EXAMPLES  # Use 3 examples for few-shot
        for pr_files in example_blocks[:max_examples]:
            user_msg = format_pr_prompt_with_variation(pr_files, prompt_variation_key)
            trimmed_user_msg = trim_message_content(user_msg, max_tokens=2000)
            assistant_msg = format_assistant_reply(
                pr_files[0]["title"], pr_files[0]["description"]
            )
            messages.append({"role": "user", "content": trimmed_user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    elif variation.get("one_shot_examples"):
        max_examples = 1  # Use 1 example for one-shot
        for pr_files in example_blocks[:max_examples]:
            user_msg = format_pr_prompt_with_variation(pr_files, prompt_variation_key)
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
            "description": f"This mock PR (variation: {PROMPT_VARIATION}) updates {file_count} file(s) including {joined_files}. It includes code changes such as additions, deletions, and modifications. Use this mock description to verify formatting and CSV outputs.",
        },
        ensure_ascii=False,
    )


# === Call GPT API ===
def call_chatgpt(messages, max_retries=3):
    if USE_MOCK:
        # For mock, calculate estimated token usage
        input_tokens = sum(len(ENCODER.encode(str(msg))) for msg in messages)
        output_tokens = len(
            ENCODER.encode(
                "Mock generated PR description with detailed technical content covering multiple aspects of the implementation."
            )
        )
        return mock_chatgpt(messages), input_tokens, output_tokens

    retries = 0
    while retries < max_retries:
        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
            )
            # Extract token usage from response
            if response and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                log_activity(
                    f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}"
                )

                return response.choices[0].message.content, input_tokens, output_tokens
            else:
                # Fallback if usage info is not available
                log_activity(
                    "Warning: No usage information available from API response"
                )
                return response.choices[0].message.content if response else None, 0, 0
        except Exception as e:
            error_msg = str(e)
            log_activity(f"Error: {error_msg}")
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = 65  # 1 minute and a bit
                log_activity(
                    f"Rate limit hit. Waiting {wait_time} seconds before retrying..."
                )
                time.sleep(wait_time)
                retries += 1
            else:
                break
    return None, 0, 0


def save_intermediate_chunks(pr_id, chunk_outputs, output_file=None):
    # Use current INTERMEDIATE_FILE if no specific file is provided
    if output_file is None:
        output_file = INTERMEDIATE_FILE

    fieldnames = [
        "pr_id",
        "chunk_index",
        "chunk_title",
        "chunk_description",
        "input_tokens",
        "output_tokens",
    ]

    # Check if file exists to determine if we need to write header
    file_exists = False
    try:
        with open(output_file, "r") as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for i, chunk_data in enumerate(chunk_outputs):
            # Handle both old format (title, desc) and new format (title, desc, input_tokens, output_tokens)
            if len(chunk_data) == 2:
                title, desc = chunk_data
                input_tokens, output_tokens = 0, 0
            else:
                title, desc, input_tokens, output_tokens = chunk_data

            writer.writerow(
                {
                    "pr_id": pr_id,
                    "chunk_index": i,
                    "chunk_title": title,
                    "chunk_description": desc,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )


def merge_descriptions_with_gpt(descriptions):
    if len(descriptions) == 1:
        # If only one description, return it directly as JSON with zero tokens
        return json.dumps({"description": descriptions[0]}, ensure_ascii=False), 0, 0
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
def process_all_prs_with_variation(prompt_variation_key=PROMPT_VARIATION):
    """Process all PRs using the specified prompt variation."""

    # Create variation-specific file paths
    variation_output_file = (
        f"./datasets/prompt_variation_{prompt_variation_key}_generated.csv"
    )
    variation_json_file = (
        f"./datasets/prompt_variation_{prompt_variation_key}_generated.json"
    )
    variation_log_path = (
        f"./datasets/prompt_variation_{prompt_variation_key}_output.log"
    )
    intermediate_file = (
        f"datasets/prompt_variation_{prompt_variation_key}_intermediate_chunks.csv"
    )

    # Create a local logging function that uses the variation-specific log file
    def log_variation_activity(activity: str):
        log = f"{datetime.datetime.now()}: {activity}\n"
        with open(variation_log_path, "a") as log_file:
            log_file.write(log)

    log_variation_activity(
        f"Starting processing with prompt variation: {prompt_variation_key}"
    )
    log_variation_activity(
        f"Variation settings: {PROMPT_VARIATIONS[prompt_variation_key]}"
    )

    # Clear intermediate file for this variation to ensure fresh start
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
        log_variation_activity(
            f"Cleared existing intermediate file: {intermediate_file}"
        )

    examples = load_examples(EXAMPLE_FILE)
    target_prs, fieldnames = group_by_pr(TARGET_FILE)
    out_fields = list(fieldnames or []) + [
        "generated_description",
        "prompt_variation",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
    ]

    log_variation_activity(f"target_prs len {len(target_prs)}")
    # Collect all data for JSON output
    all_data = []

    with open(variation_output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for pr_idx, (pr_id, files) in enumerate(list(target_prs.items())):
            log_variation_activity(f"Processing PR {pr_id} with {len(files)} files...")

            # Access pr_total_size_bytes from target_prs if present
            pr_total_size_bytes = (
                target_prs[pr_id][0].get("pr_total_size_bytes")
                if target_prs[pr_id] and "pr_total_size_bytes" in target_prs[pr_id][0]
                else None
            )
            log_variation_activity(
                f"PR {pr_id} total size (bytes): {pr_total_size_bytes}"
            )

            file_chunks = chunk_pr_files(files)
            chunk_outputs = []

            for i, chunk in enumerate(file_chunks):
                log_variation_activity(
                    f"  Chunk {i+1}/{len(file_chunks)} for PR {pr_id}"
                )
                prompt = format_pr_prompt_with_variation(chunk, prompt_variation_key)
                messages = build_messages_with_variation(
                    examples, prompt, prompt_variation_key
                )
                # Log the messages as JSON
                log_variation_activity(
                    f"Messages for PR {pr_id} chunk {i+1}:\n"
                    + json.dumps(messages, indent=2, ensure_ascii=False)
                )
                response, input_tokens, output_tokens = call_chatgpt(messages)
                time.sleep(1.5)

                if response:
                    title, desc = extract_title_description(response)
                else:
                    title, desc = "ERROR", "Failed to generate"

                chunk_outputs.append((title, desc, input_tokens, output_tokens))

            save_intermediate_chunks(pr_id, chunk_outputs, intermediate_file)

            # Combine chunk outputs and calculate token totals
            all_titles = [title for title, _, _, _ in chunk_outputs]
            all_descriptions = [desc for _, desc, _, _ in chunk_outputs]
            total_chunk_input_tokens = sum(
                input_tokens for _, _, input_tokens, _ in chunk_outputs
            )
            total_chunk_output_tokens = sum(
                output_tokens for _, _, _, output_tokens in chunk_outputs
            )

            merged_response, merge_input_tokens, merge_output_tokens = (
                merge_descriptions_with_gpt(all_descriptions)
            )

            # Calculate total token usage for this PR
            total_input_tokens = total_chunk_input_tokens + merge_input_tokens
            total_output_tokens = total_chunk_output_tokens + merge_output_tokens
            total_tokens = total_input_tokens + total_output_tokens

            log_variation_activity(
                f"PR {pr_id} total tokens - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}"
            )

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
                row["prompt_variation"] = prompt_variation_key
                row["total_input_tokens"] = total_input_tokens
                row["total_output_tokens"] = total_output_tokens
                row["total_tokens"] = total_tokens
                filtered_row = {k: row.get(k, "") for k in out_fields}
                writer.writerow(filtered_row)
                all_data.append(dict(row))  # Add to JSON data collection

    # Save as JSON
    with open(variation_json_file, "w", encoding="utf-8") as json_file:
        json.dump(all_data, json_file, indent=2, ensure_ascii=False)

    log_variation_activity(f"Saved CSV output to {variation_output_file}")
    log_variation_activity(f"Saved JSON output to {variation_json_file}")


def run_all_variations():
    """Run all prompt variations sequentially."""
    log_activity("Starting batch processing of all prompt variations...")

    for variation_key in PROMPT_VARIATIONS.keys():
        log_activity(f"Processing variation: {variation_key}")

        # Update global config for this variation
        global PROMPT_VARIATION, OUTPUT_FILE, OUTPUT_JSON_FILE, LOG_PATH, INTERMEDIATE_FILE
        PROMPT_VARIATION = variation_key
        OUTPUT_FILE = f"./datasets/prompt_variation_{variation_key}_generated.csv"
        OUTPUT_JSON_FILE = f"./datasets/prompt_variation_{variation_key}_generated.json"
        LOG_PATH = f"./datasets/prompt_variation_{variation_key}_output.log"
        INTERMEDIATE_FILE = (
            f"./datasets/prompt_variation_{variation_key}_intermediate_chunks.csv"
        )

        try:
            process_all_prs_with_variation(variation_key)
            log_activity(f"Completed variation: {variation_key}")
        except Exception as e:
            log_activity(f"Error processing variation {variation_key}: {e}")

    log_activity("Completed all variations!")


# Run it
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PR descriptions with prompt variations"
    )
    parser.add_argument(
        "--variation",
        type=str,
        default=PROMPT_VARIATION,
        help="Prompt variation to use (e.g., P-1, P-2, P-3, P-4, P-5, P-6)",
    )
    parser.add_argument("--all", action="store_true", help="Run all prompt variations")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock responses for testing"
    )

    args = parser.parse_args()

    if args.mock:
        USE_MOCK = True
        log_activity("Running in MOCK mode")

    if args.all:
        run_all_variations()
    else:
        if args.variation not in PROMPT_VARIATIONS:
            print(f"Error: Unknown variation '{args.variation}'")
            print(f"Available variations: {list(PROMPT_VARIATIONS.keys())}")
            sys.exit(1)
        process_all_prs_with_variation(args.variation)
