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
PROMPT_VARIATIONS = {
    "P-1_Minimal": {
        "repo_name_and_path": True,
        "pr_title": False,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-2_Basic": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-3_Diffs_Only": {
        "repo_name_and_path": True,
        "pr_title": False,
        "pr_diffs": True,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-4_Diffs_Plus_Title": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-5_Code_Only": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": False,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-6_Issue_Only": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": True,
        "pr_template_content": False,
        "few_shot_examples": False,
    },
    "P-7_Template_Plus_Title": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": True,
        "few_shot_examples": False,
    },
    "P-8_Full_Context": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": True,
        "pr_template_content": True,
        "few_shot_examples": False,
    },
    "P-9_Basic_One_Shot": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": False,
        "pr_file_contents": False,
        "pr_issue_context": False,
        "pr_template_content": False,
        "one_shot_examples": True,
    },
    "P-10_Full_Plus_One_Shot": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": True,
        "pr_template_content": True,
        "one_shot_examples": True,
    },
    "P-11_Full_Plus_Few_Shot": {
        "repo_name_and_path": True,
        "pr_title": True,
        "pr_diffs": True,
        "pr_file_contents": True,
        "pr_issue_context": True,
        "pr_template_content": True,
        "few_shot_examples": True,
    },
}

### === CONFIG === ###
EXAMPLE_FILE = "../pr_files/datasets/few_shot_examples_with_issues_and_files.csv"
MAX_EXAMPLES = 3  # Will be overridden based on variation
MAX_PROMPT_TOKENS = 32000
USE_MOCK = False


# === Utility: Logging ===
def log_activity(activity: str, log_path):
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
    try:
        # Increase CSV field size limit to handle large fields
        csv.field_size_limit(sys.maxsize)
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                examples[row["id"]].append(row)
    except Exception as e:
        print(f"Error loading examples from {file_path}: {e}")
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
    if prompt_variation_key == "P-1_Minimal":
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


def mock_chatgpt(messages, prompt_variation):
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
            "description": f"This mock PR (variation: {prompt_variation}) updates {file_count} file(s) including {joined_files}. It includes code changes such as additions, deletions, and modifications. Use this mock description to verify formatting and CSV outputs.",
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
        return mock_chatgpt(messages, "retry"), input_tokens, output_tokens

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

                return response.choices[0].message.content, input_tokens, output_tokens
            else:
                # Fallback if usage info is not available
                return response.choices[0].message.content if response else None, 0, 0
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = 65  # 1 minute and a bit
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                break
    return None, 0, 0


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
                print(f"Error: CSV file {csv_file} has no header/fieldnames.")
                return prs, []
            row_count = 0
            for row in reader:
                if not row or not row.get("id"):
                    continue
                prs[row["id"]].append(row)
                row_count += 1
            if row_count == 0:
                print(f"Warning: CSV file {csv_file} is empty or has no valid rows.")
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
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


def find_entries_with_missing_descriptions(file_path):
    """Find entries with empty generated descriptions."""
    missing_entries = []
    
    if file_path.endswith('.json'):
        # Handle JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data:
                if entry.get("generated_description", "").strip() == "":
                    missing_entries.append(entry)
                    
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return []
    
    elif file_path.endswith('.csv'):
        # Handle CSV file
        try:
            csv.field_size_limit(sys.maxsize)
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get("generated_description", "").strip() == "":
                        missing_entries.append(row)
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return []
    
    return missing_entries


def retry_missing_descriptions_for_variation(prompt_variation_key, dry_run=False):
    """Retry generating descriptions for entries with missing descriptions."""
    
    # Create file paths for this variation
    variation_output_file = f"./datasets/prompt_variation_{prompt_variation_key}_generated.csv"
    variation_json_file = f"./datasets/prompt_variation_{prompt_variation_key}_generated.json"
    variation_log_path = f"./datasets/prompt_variation_{prompt_variation_key}_retry_output.log"
    
    print(f"\n=== Processing variation: {prompt_variation_key} ===")
    
    # Check if files exist
    if not os.path.exists(variation_json_file):
        print(f"JSON file not found: {variation_json_file}")
        return
    
    if not os.path.exists(variation_output_file):
        print(f"CSV file not found: {variation_output_file}")
        return
    
    # Find entries with missing descriptions
    missing_entries = find_entries_with_missing_descriptions(variation_json_file)
    
    if not missing_entries:
        print(f"No entries with missing descriptions found for {prompt_variation_key}")
        return
    
    print(f"Found {len(missing_entries)} entries with missing descriptions")
    
    # Load examples for this variation
    examples = load_examples(EXAMPLE_FILE)
    
    # Group missing entries by PR
    missing_prs = defaultdict(list)
    for entry in missing_entries:
        missing_prs[entry["id"]].append(entry)
    
    print(f"Missing descriptions span {len(missing_prs)} unique PRs")
    
    if dry_run:
        print("DRY RUN - Would retry the following PRs:")
        for pr_id, files in missing_prs.items():
            print(f"  PR: {pr_id} ({len(files)} files)")
        return
    
    # Log retry activity
    log_activity(f"Starting retry for variation: {prompt_variation_key}", variation_log_path)
    log_activity(f"Found {len(missing_entries)} entries with missing descriptions", variation_log_path)
    
    retried_entries = []
    
    for pr_idx, (pr_id, files) in enumerate(missing_prs.items()):
        print(f"Processing PR {pr_idx + 1}/{len(missing_prs)}: {pr_id}")
        log_activity(f"Processing PR {pr_id} with {len(files)} files...", variation_log_path)
        
        file_chunks = chunk_pr_files(files)
        chunk_outputs = []
        
        for i, chunk in enumerate(file_chunks):
            log_activity(f"  Chunk {i+1}/{len(file_chunks)} for PR {pr_id}", variation_log_path)
            prompt = format_pr_prompt_with_variation(chunk, prompt_variation_key)
            messages = build_messages_with_variation(examples, prompt, prompt_variation_key)
            
            response, input_tokens, output_tokens = call_chatgpt(messages)
            time.sleep(1.5)  # Be nice to the API
            
            if response:
                title, desc = extract_title_description(response)
            else:
                title, desc = "ERROR", "Failed to generate"
                
            chunk_outputs.append((title, desc, input_tokens, output_tokens))
        
        # Combine chunk outputs
        all_titles = [title for title, _, _, _ in chunk_outputs]
        all_descriptions = [desc for _, desc, _, _ in chunk_outputs]
        total_chunk_input_tokens = sum(input_tokens for _, _, input_tokens, _ in chunk_outputs)
        total_chunk_output_tokens = sum(output_tokens for _, _, _, output_tokens in chunk_outputs)
        
        merged_response, merge_input_tokens, merge_output_tokens = merge_descriptions_with_gpt(all_descriptions)
        
        # Calculate total token usage for this PR
        total_input_tokens = total_chunk_input_tokens + merge_input_tokens
        total_output_tokens = total_chunk_output_tokens + merge_output_tokens
        
        log_activity(f"PR {pr_id} total tokens - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}", variation_log_path)
        
        # Use the original title from the first chunk
        original_title = all_titles[0] if all_titles else ""
        if merged_response:
            final_title, combined_desc = extract_title_description(merged_response, original_title=original_title)
        else:
            combined_desc = "\n\n".join(all_descriptions)
        
        # Update entries with new descriptions
        for entry in files:
            entry["generated_description"] = combined_desc
            entry["total_input_tokens"] = total_input_tokens
            entry["total_output_tokens"] = total_output_tokens
            entry["total_tokens"] = total_input_tokens + total_output_tokens
            retried_entries.append(entry)
    
    # Now update the files with the new descriptions
    print(f"Updating files with {len(retried_entries)} retry results...")
    
    # Update JSON file
    try:
        with open(variation_json_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Create a mapping of entries by a unique key for faster lookup
        retry_map = {}
        for entry in retried_entries:
            # Use id + filename as unique key
            key = entry["id"] + "|" + entry["filename"]
            retry_map[key] = entry
        
        # Update matching entries
        updated_count = 0
        for entry in all_data:
            key = entry["id"] + "|" + entry["filename"]
            if key in retry_map:
                entry["generated_description"] = retry_map[key]["generated_description"]
                entry["total_input_tokens"] = retry_map[key]["total_input_tokens"]
                entry["total_output_tokens"] = retry_map[key]["total_output_tokens"]
                entry["total_tokens"] = retry_map[key]["total_tokens"]
                updated_count += 1
        
        # Write back to JSON file
        with open(variation_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated {updated_count} entries in JSON file")
        
    except Exception as e:
        print(f"Error updating JSON file: {e}")
    
    # Update CSV file
    try:
        # Read all data from CSV
        csv.field_size_limit(sys.maxsize)
        with open(variation_output_file, 'r', newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            all_rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Update matching rows
        retry_map = {}
        for entry in retried_entries:
            key = entry["id"] + "|" + entry["filename"]
            retry_map[key] = entry
        
        updated_count = 0
        for row in all_rows:
            key = row["id"] + "|" + row["filename"]
            if key in retry_map:
                row["generated_description"] = retry_map[key]["generated_description"]
                row["total_input_tokens"] = retry_map[key]["total_input_tokens"]
                row["total_output_tokens"] = retry_map[key]["total_output_tokens"]
                row["total_tokens"] = retry_map[key]["total_tokens"]
                updated_count += 1
        
        # Write back to CSV file
        with open(variation_output_file, 'w', newline="", encoding="utf-8") as csvfile:
            if fieldnames:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        
        print(f"Updated {updated_count} entries in CSV file")
        
    except Exception as e:
        print(f"Error updating CSV file: {e}")
    
    log_activity(f"Completed retry for variation: {prompt_variation_key}", variation_log_path)
    print(f"✓ Completed retry for {prompt_variation_key}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Retry generating missing PR descriptions")
    parser.add_argument(
        "--variation",
        type=str,
        help="Prompt variation to retry (e.g., P-9_Basic_One_Shot). If not specified, will check all variations."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be retried, don't actually retry"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock responses for testing"
    )
    
    args = parser.parse_args()
    
    if args.mock:
        global USE_MOCK
        USE_MOCK = True
        print("Running in MOCK mode")
    
    if args.variation:
        if args.variation not in PROMPT_VARIATIONS:
            print(f"Error: Unknown variation '{args.variation}'")
            print(f"Available variations: {list(PROMPT_VARIATIONS.keys())}")
            return
        retry_missing_descriptions_for_variation(args.variation, dry_run=args.dry_run)
    else:
        print("Checking all variations for missing descriptions...")
        for variation_key in PROMPT_VARIATIONS.keys():
            try:
                retry_missing_descriptions_for_variation(variation_key, dry_run=args.dry_run)
            except Exception as e:
                print(f"Error processing variation {variation_key}: {e}")
    
    print("\nRetry process completed!")


if __name__ == "__main__":
    main()
