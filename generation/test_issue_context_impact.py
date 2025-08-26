import csv
import datetime
import json
import openai
import time
import tiktoken
import math
import sys

MODEL_NAME = "gpt-4o-mini-2024-07-18"
# Initialize encoding once
ENCODER = tiktoken.encoding_for_model(MODEL_NAME)

# 🔐 Load OpenAI API key
with open("./env/tokens.txt", "r") as f:
    openai.api_key = f.read().strip()

### === CONFIG === ###
INPUT_FILE = "../pr_files/datasets/few_shot_examples_with_issues_and_files.csv"
OUTPUT_CSV = "./datasets/three_way_issue_context_comparison.csv"
OUTPUT_JSON = "./datasets/three_way_issue_context_comparison.json"
LOG_PATH = "./datasets/three_way_issue_context_test.log"
MAX_PROMPT_TOKENS = 32000
MAX_ISSUE_CONTEXT_TOKENS = 4000  # Reserve tokens for issue context
USE_MOCK = False


# === Utility: Logging ===
def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)


# === Utility: Estimate token count ===
def estimate_tokens(text, use_real=True):
    """Estimate token count using tiktoken."""
    if not text:
        return 0
    if use_real:
        return len(ENCODER.encode(text))
    return len(text) // 4


def estimate_tokens_bytes(file_size_bytes):
    """Estimate token count based on file size in bytes."""
    if file_size_bytes is None:
        return 0
    try:
        size = float(file_size_bytes)
        if math.isnan(size):
            return 0
        return int(size // 3)
    except (ValueError, TypeError):
        return 0


# === Smart truncation for issue context ===
def truncate_text_smart(text, max_tokens):
    """Truncate text while preserving structure and meaning."""
    if not text:
        return ""
    
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text
    
    # Split into sentences and keep the most important ones
    sentences = text.split('. ')
    truncated = []
    current_count = 0
    
    # Always keep the first sentence
    if sentences:
        truncated.append(sentences[0])
        current_count = estimate_tokens(sentences[0])
    
    # Add more sentences until we hit the limit
    for sentence in sentences[1:]:
        sentence_tokens = estimate_tokens(sentence)
        if current_count + sentence_tokens > max_tokens - 50:  # Leave buffer
            break
        truncated.append(sentence)
        current_count += sentence_tokens
    
    result = '. '.join(truncated)
    if len(sentences) > len(truncated):
        result += "... [truncated for token limit]"
    
    return result


def truncate_comments_smart(comments, max_tokens):
    """Truncate comments while keeping the most relevant ones."""
    if not comments:
        return ""
    
    # Split comments by actual comment boundaries
    comment_blocks = []
    current_tokens = 0
    
    # Simple splitting - could be improved with better parsing
    lines = comments.split('\n')
    current_block = []
    
    for line in lines:
        if line.startswith('Comment #') and current_block:
            # Save previous block
            block_text = '\n'.join(current_block)
            block_tokens = estimate_tokens(block_text)
            if current_tokens + block_tokens <= max_tokens:
                comment_blocks.append(block_text)
                current_tokens += block_tokens
            else:
                break
            current_block = [line]
        else:
            current_block.append(line)
    
    # Add the last block if there's room
    if current_block:
        block_text = '\n'.join(current_block)
        block_tokens = estimate_tokens(block_text)
        if current_tokens + block_tokens <= max_tokens:
            comment_blocks.append(block_text)
    
    result = '\n'.join(comment_blocks)
    if len(result) < len(comments):
        result += "\n... [additional comments truncated for token limit]"
    
    return result


# === Format issue context ===
def format_issue_context(issue_titles, issue_bodies, issue_comments, max_tokens=MAX_ISSUE_CONTEXT_TOKENS, include_comments=True):
    """Format issue context with smart truncation."""
    context_parts = []
    used_tokens = 0
    
    # Reserve tokens for structure
    structure_tokens = 100
    available_tokens = max_tokens - structure_tokens
    
    if include_comments:
        # Allocate tokens: 20% for titles, 40% for bodies, 40% for comments
        title_tokens = int(available_tokens * 0.2)
        body_tokens = int(available_tokens * 0.4)
        comment_tokens = available_tokens - title_tokens - body_tokens
    else:
        # Allocate tokens: 30% for titles, 70% for bodies (no comments)
        title_tokens = int(available_tokens * 0.3)
        body_tokens = available_tokens - title_tokens
        comment_tokens = 0
    
    if issue_titles:
        truncated_titles = truncate_text_smart(issue_titles, title_tokens)
        if truncated_titles:
            context_parts.append(f"**Related Issue Titles:**\n{truncated_titles}")
            used_tokens += estimate_tokens(truncated_titles)
    
    if issue_bodies:
        truncated_bodies = truncate_text_smart(issue_bodies, body_tokens)
        if truncated_bodies:
            context_parts.append(f"**Related Issue Description:**\n{truncated_bodies}")
            used_tokens += estimate_tokens(truncated_bodies)
    
    if include_comments and issue_comments:
        truncated_comments = truncate_comments_smart(issue_comments, comment_tokens)
        if truncated_comments:
            context_parts.append(f"**Related Issue Discussion:**\n{truncated_comments}")
            used_tokens += estimate_tokens(truncated_comments)
    
    if not context_parts:
        return "", 0
    
    full_context = "**RELATED ISSUE CONTEXT:**\n\n" + "\n\n".join(context_parts) + "\n\n---\n\n"
    actual_tokens = estimate_tokens(full_context)
    
    return full_context, actual_tokens


# === Format PR prompt ===
def format_pr_prompt(pr_files, include_issue_context=False, issue_context=""):
    """Format PR prompt with optional issue context."""
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
    
    base_prompt = "Generate a description for this PR:\n\n"
    if include_issue_context and issue_context:
        base_prompt = issue_context + base_prompt
    
    return base_prompt + "\n\n---\n\n".join(parts)


# === Build messages ===
def build_messages(target_prompt):
    """Build messages without examples for this test."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that writes concise and informative GitHub pull request descriptions based on code file contents, diffs, and any provided context. "
                "Respond ONLY with a valid JSON object in the following format (do not include any extra text):\n"
                '{"description": "<description>"}'
            ),
        },
        {"role": "user", "content": target_prompt}
    ]
    return messages


def mock_chatgpt(messages, version=""):
    """Mock ChatGPT response for testing."""
    user_msg = messages[-1]["content"]
    file_count = user_msg.count("Filename:")
    has_issue_context = "RELATED ISSUE CONTEXT" in user_msg
    has_discussion = "Related Issue Discussion" in user_msg
    
    mock_desc = f"[MOCK {version}] This PR updates {file_count} file(s). "
    if has_issue_context:
        if has_discussion:
            mock_desc += "Based on full issue context with discussion, this addresses privacy filtering concerns. "
        else:
            mock_desc += "Based on partial issue context, this addresses privacy filtering concerns. "
    mock_desc += "Mock description for testing purposes."
    
    return json.dumps({"description": mock_desc}, ensure_ascii=False)


# === Call GPT API ===
def call_chatgpt(messages, max_retries=3, version=""):
    """Call ChatGPT API with retry logic."""
    if USE_MOCK:
        return mock_chatgpt(messages, version)

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
            log_activity(f"Error in {version}: {error_msg}")
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                wait_time = 65
                log_activity(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                break
    return None


# === Extract description from GPT response ===
def extract_description(response):
    """Extract description from GPT response."""
    try:
        data = json.loads(response)
        return data.get("description", "")
    except Exception:
        # Fallback parsing
        if "description" in response.lower():
            lines = response.splitlines()
            for line in lines:
                if "description" in line.lower() and ":" in line:
                    return line.split(":", 1)[-1].strip().strip('"')
        return response.strip()


# === Load first row ===
def load_first_row():
    """Load the first row from the CSV file."""
    try:
        csv.field_size_limit(sys.maxsize)
        with open(INPUT_FILE, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            first_row = next(reader, None)
            if not first_row:
                log_activity("Error: No data found in CSV file")
                return None
            return first_row
    except Exception as e:
        log_activity(f"Error reading CSV file: {e}")
        return None


# === Main test function ===
def test_issue_context_impact():
    """Test the impact of including different levels of issue context."""
    log_activity("Starting three-way issue context impact test...")
    
    # Load first row
    row = load_first_row()
    if not row:
        log_activity("Failed to load test data")
        return
    
    log_activity(f"Testing PR: {row['id']} - {row['title']}")
    
    # Prepare issue context
    issue_titles = row.get('issue_titles', '')
    issue_bodies = row.get('issue_bodies', '')  
    issue_comments = row.get('issue_comments', '')
    
    # Prepare PR data (treating the single row as one file)
    pr_files = [row]
    
    # === Version A: Without issue context ===
    log_activity("Generating description WITHOUT issue context...")
    prompt_without = format_pr_prompt(pr_files, include_issue_context=False)
    messages_without = build_messages(prompt_without)
    
    prompt_without_tokens = estimate_tokens(prompt_without)
    log_activity(f"Prompt without issue context tokens: {prompt_without_tokens}")
    
    response_without = call_chatgpt(messages_without, version="WITHOUT_CONTEXT")
    time.sleep(2)  # Rate limiting
    
    description_without = extract_description(response_without) if response_without else "Failed to generate"
    
    # === Version B: With partial issue context (titles + bodies only) ===
    log_activity("Generating description WITH PARTIAL issue context (titles + bodies)...")
    partial_context, partial_context_tokens = format_issue_context(
        issue_titles, issue_bodies, issue_comments, include_comments=False
    )
    log_activity(f"Partial issue context tokens: {partial_context_tokens}")
    
    prompt_partial = format_pr_prompt(pr_files, include_issue_context=True, issue_context=partial_context)
    messages_partial = build_messages(prompt_partial)
    
    prompt_partial_tokens = estimate_tokens(prompt_partial)
    log_activity(f"Prompt with partial issue context tokens: {prompt_partial_tokens}")
    
    response_partial = call_chatgpt(messages_partial, version="PARTIAL_CONTEXT")
    time.sleep(2)  # Rate limiting
    
    description_partial = extract_description(response_partial) if response_partial else "Failed to generate"
    
    # === Version C: With full issue context (titles + bodies + comments) ===
    log_activity("Generating description WITH FULL issue context (titles + bodies + comments)...")
    full_context, full_context_tokens = format_issue_context(
        issue_titles, issue_bodies, issue_comments, include_comments=True
    )
    log_activity(f"Full issue context tokens: {full_context_tokens}")
    
    prompt_full = format_pr_prompt(pr_files, include_issue_context=True, issue_context=full_context)
    messages_full = build_messages(prompt_full)
    
    prompt_full_tokens = estimate_tokens(prompt_full)
    log_activity(f"Prompt with full issue context tokens: {prompt_full_tokens}")
    
    response_full = call_chatgpt(messages_full, version="FULL_CONTEXT")
    
    description_full = extract_description(response_full) if response_full else "Failed to generate"
    
    # Calculate metrics
    desc_without_tokens = estimate_tokens(description_without)
    desc_partial_tokens = estimate_tokens(description_partial)
    desc_full_tokens = estimate_tokens(description_full)
    
    # Check if contexts were truncated
    total_raw_tokens = estimate_tokens(issue_titles + issue_bodies + issue_comments)
    partial_raw_tokens = estimate_tokens(issue_titles + issue_bodies)
    
    # Prepare results
    results = {
        "pr_id": row['id'],
        "pr_title": row['title'],
        "original_description": row['description'],
        "description_without_context": description_without,
        "description_partial_context": description_partial,
        "description_full_context": description_full,
        "prompt_tokens_without": prompt_without_tokens,
        "prompt_tokens_partial": prompt_partial_tokens,
        "prompt_tokens_full": prompt_full_tokens,
        "issue_context_tokens_partial": partial_context_tokens,
        "issue_context_tokens_full": full_context_tokens,
        "response_tokens_without": desc_without_tokens,
        "response_tokens_partial": desc_partial_tokens,
        "response_tokens_full": desc_full_tokens,
        "token_increase_partial": prompt_partial_tokens - prompt_without_tokens,
        "token_increase_full": prompt_full_tokens - prompt_without_tokens,
        "issue_titles_length": len(issue_titles) if issue_titles else 0,
        "issue_bodies_length": len(issue_bodies) if issue_bodies else 0,
        "issue_comments_length": len(issue_comments) if issue_comments else 0,
        "partial_context_truncated": partial_context_tokens < partial_raw_tokens,
        "full_context_truncated": full_context_tokens < total_raw_tokens
    }
    
    # Save CSV output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = results.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
    
    # Save JSON output with detailed comparison
    detailed_results = {
        "experiment_info": {
            "date": datetime.datetime.now().isoformat(),
            "model": MODEL_NAME,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_issue_context_tokens": MAX_ISSUE_CONTEXT_TOKENS,
            "experiment_type": "three_way_comparison"
        },
        "pr_info": {
            "id": row['id'],
            "title": row['title'],
            "repository": row.get('repository', ''),
            "state": row.get('state', ''),
            "original_description": row['description']
        },
        "issue_context_info": {
            "has_titles": bool(issue_titles),
            "has_bodies": bool(issue_bodies), 
            "has_comments": bool(issue_comments),
            "titles_char_count": len(issue_titles) if issue_titles else 0,
            "bodies_char_count": len(issue_bodies) if issue_bodies else 0,
            "comments_char_count": len(issue_comments) if issue_comments else 0,
            "partial_context_tokens_used": partial_context_tokens,
            "full_context_tokens_used": full_context_tokens,
            "partial_context_was_truncated": results["partial_context_truncated"],
            "full_context_was_truncated": results["full_context_truncated"]
        },
        "generation_results": {
            "without_context": {
                "description": description_without,
                "prompt_tokens": prompt_without_tokens,
                "response_tokens": desc_without_tokens
            },
            "partial_context": {
                "description": description_partial,
                "prompt_tokens": prompt_partial_tokens,
                "response_tokens": desc_partial_tokens
            },
            "full_context": {
                "description": description_full,
                "prompt_tokens": prompt_full_tokens,
                "response_tokens": desc_full_tokens
            }
        },
        "comparison_metrics": {
            "token_increase_partial_absolute": results["token_increase_partial"],
            "token_increase_partial_percentage": (results["token_increase_partial"] / prompt_without_tokens * 100) if prompt_without_tokens > 0 else 0,
            "token_increase_full_absolute": results["token_increase_full"],
            "token_increase_full_percentage": (results["token_increase_full"] / prompt_without_tokens * 100) if prompt_without_tokens > 0 else 0,
            "description_length_without": len(description_without),
            "description_length_partial": len(description_partial),
            "description_length_full": len(description_full),
            "description_length_diff_partial": len(description_partial) - len(description_without),
            "description_length_diff_full": len(description_full) - len(description_without)
        }
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jsonfile:
        json.dump(detailed_results, jsonfile, indent=2, ensure_ascii=False)
    
    log_activity(f"Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
    log_activity("Three-way test completed successfully!")
    
    # Print summary
    print("\n=== THREE-WAY ISSUE CONTEXT IMPACT TEST RESULTS ===")
    print(f"PR: {row['id']}")
    print(f"Title: {row['title']}")
    print(f"\nToken Usage:")
    print(f"  Without context: {prompt_without_tokens} tokens")
    print(f"  Partial context: {prompt_partial_tokens} tokens (+{results['token_increase_partial']} | {(results['token_increase_partial'] / prompt_without_tokens * 100):.1f}%)")
    print(f"  Full context: {prompt_full_tokens} tokens (+{results['token_increase_full']} | {(results['token_increase_full'] / prompt_without_tokens * 100):.1f}%)")
    print(f"\nIssue Context:")
    print(f"  Partial context tokens: {partial_context_tokens}")
    print(f"  Full context tokens: {full_context_tokens}")
    print(f"  Partial truncated: {results['partial_context_truncated']}")
    print(f"  Full truncated: {results['full_context_truncated']}")
    print(f"\nDescription Lengths:")
    print(f"  Without context: {len(description_without)} chars")
    print(f"  Partial context: {len(description_partial)} chars ({len(description_partial) - len(description_without):+d})")
    print(f"  Full context: {len(description_full)} chars ({len(description_full) - len(description_without):+d})")
    print(f"\nFiles saved:")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    test_issue_context_impact()
