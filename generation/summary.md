
# Prompt Variations Update Summary - September, 09, 2025

## Changes Made

### What was changed:
- **Updated PROMPT_VARIATIONS dictionary**: Replaced the existing 11 prompt variations (P-1_Minimal through P-11_Full_Plus_Few_Shot) with 6 new progressive variations (P-1 through P-6) based on the research matrix.
- **Simplified naming convention**: Changed from descriptive names like "P-1_Minimal" to simple numerical format "P-1".
- **Updated default configuration**: Changed default PROMPT_VARIATION from "P-9_Basic_One_Shot" to "P-1".
- **Updated special handling**: Modified the minimal prompt handling to use "P-1" instead of "P-1_Minimal".
- **Updated help text**: Updated argument parser help text to reflect the new variation names.

### Why it was changed:
- **Research alignment**: The new variations follow a specific research matrix that progressively adds context components in a controlled manner.
- **Cleaner progression**: The new matrix provides a cleaner progression from minimal context (P-1) to full context (P-6).
- **Standardized approach**: Each variation builds upon the previous one by adding exactly one new component, making it easier to analyze the impact of each component.

### New Prompt Variation Matrix:
- **P-1**: Repository name and file pathnames only (minimal)
- **P-2**: P-1 + PR template content
- **P-3**: P-2 + PR title
- **P-4**: P-3 + PR diffs
- **P-5**: P-4 + PR file contents
- **P-6**: P-5 + PR issue context (full context)

### How to extend it in the future:
1. **Adding new components**: Add new boolean fields to the variation dictionaries (e.g., `"pr_comments": True`).
2. **Adding new variations**: Add new entries to the PROMPT_VARIATIONS dictionary following the same structure.
3. **Modifying component handling**: Update the `format_pr_prompt_with_variation()` function to handle new components.
4. **Testing new variations**: Use the `--variation` flag to test individual variations or `--all` to test all variations.

### Usage Examples:
```bash
# Test a specific variation
pyenv exec python generate_with_prompt_variations.py --variation P-3

# Test all variations
pyenv exec python generate_with_prompt_variations.py --all

# Test with mock responses
pyenv exec python generate_with_prompt_variations.py --variation P-1 --mock
```

## Commit Message (Conventional Commits)

```
feat(generation): update prompt variations to progressive research matrix

- Replace 11 complex variations with 6 progressive variations (P-1 to P-6)
- Simplify naming from descriptive to numerical format
- Each variation progressively adds one component for controlled analysis
- Update default configuration to use P-1 (minimal variation)
- Maintain backward compatibility in core functionality

BREAKING CHANGE: Old prompt variation names (P-1_Minimal, P-2_Basic, etc.) are no longer available
```

---

# Previous Summary of prompt engineering enhancements

## What was changed

- Instructions are now always repeated before and after the context for all prompt variations (not just long-context).
- Added a planning/outline step: the model is first asked to list major changes or bullet points before writing the final JSON description.
- Clarified non-goals with a "do not" section (negative prompt) to reduce hallucination and speculation.
- Added an adaptive chunking note: if a PR is split into multiple chunks, the prompt now tells the model which part it is and to merge all summaries at the end.
- Maintained clear delimiters and explicit, step-by-step instructions for all prompt levels except minimal (P-1).
- Minimal prompt (P-1) remains simple and concise.

## Why

- To further improve the quality, reliability, and consistency of generated PR descriptions by leveraging advanced prompt engineering techniques and reducing model hallucination.

## How

- Refactored the prompt assembly logic to always repeat instructions, add a planning step, clarify non-goals, and include adaptive chunking notes when needed.

---

### Previous Conventional Commit Message

feat(prompt): reinforce instructions, add planning step, clarify non-goals, and support chunked PRs

- Always repeat instructions before and after context
- Add planning/outline step before final description
- Add negative prompt to reduce hallucination
- Add adaptive chunking note for multi-part PRs
- **P-8_Full_Context**: All components except examples
- **P-9_Basic_One_Shot**: Basic components (repo + title + diffs) with 1 example
- **P-10_Full_Plus_One_Shot**: All components with 1 example
- **P-11_Full_Plus_Few_Shot**: All components with 3 example

### 2. Component Implementation

Each prompt component is conditionally included based on the variation:

- **repo_name_and_path**: Repository name and path from the PR data
- **pr_title**: The pull request title
- **pr_diffs**: Git diffs/patches for modified files
- **pr_file_contents**: Full file contents (when available)
- **pr_issue_context**: Issue titles and issue bodies (NOTE: issue_comments are ignored as requested)
- **pr_template_content**: Standard PR template guidelines
- **few_shot_examples**: Example PRs with descriptions (3 examples)
- **one_shot_examples**: Single example PR with description (1 example)

### 3. Core Functions Modified

#### `format_pr_prompt_with_variation(pr_files, prompt_variation_key)`

- Replaced the original `format_pr_prompt()` function
- Conditionally includes components based on the variation matrix
- Maintains clean separation between different content types

#### `build_messages_with_variation(example_blocks, target_prompt, prompt_variation_key)`

- Enhanced message building to support both few-shot and one-shot example inclusion
- Only adds examples when `few_shot_examples` or `one_shot_examples` is True in the variation
- Dynamically adjusts the number of examples (3 for few-shot, 1 for one-shot)

#### `format_issue_context(pr_files)`

- New function to extract issue context from `issue_titles` and `issue_bodies`
- **Specifically ignores `issue_comments` as requested**

### 4. Configuration & Usage

#### Single Variation Mode

```bash
# Run specific variation
python generate_with_prompt_variations.py --variation P-1_Minimal

# Test with mock responses
python generate_with_prompt_variations.py --variation P-5_Code_Only --mock
```

#### Batch Processing Mode

```bash
# Run all 11 variations sequentially
python generate_with_prompt_variations.py --all
```

#### Output Files

Each variation generates separate output files:

- `prompt_variation_{VARIATION}_generated.csv`
- `prompt_variation_{VARIATION}_generated.json`
- `prompt_variation_{VARIATION}_output.log`
- `prompt_variation_{VARIATION}_intermediate_chunks.csv`

### 5. Data Structure

The `PROMPT_VARIATIONS` dictionary defines the matrix:

```python
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
    # ... 8 more variations
}
```

### 6. Issue Context Handling

As specifically requested:

- **Included**: `issue_titles` and `issue_bodies`
- **Excluded**: `issue_comments` (completely ignored)

The `format_issue_context()` function only processes the title and body fields, providing relevant issue information without the potentially noisy comment threads.

### 7. Key Improvements

1. **Modular Design**: Each component can be independently enabled/disabled
2. **Systematic Testing**: All 11 variations can be tested consistently
3. **Backward Compatibility**: Maintains the same core logic as the original script
4. **Enhanced Logging**: Tracks which variation is being processed
5. **Flexible Execution**: Single variation or batch processing modes

## Usage Examples

### Test a single variation with mock data

```bash
python generate_with_prompt_variations.py --variation P-3_Diffs_Only --mock
```

### Test one-shot variations

```bash
# Basic one-shot with minimal context
pyenv exec python generate_with_prompt_variations.py --variation P-9_Basic_One_Shot --mock

# Full context with one example
pyenv exec python generate_with_prompt_variations.py --variation P-11_Full_Plus_Few_Shot
```

### Generate descriptions for all variations

```bash
pyenv exec python generate_with_prompt_variations.py --all
```

### Generate with specific variation

```bash
pyenv exec python generate_with_prompt_variations.py --variation P-8_Full_Context
```

## File Structure

- Original script: `generate_with_chat_gpt.py` (unchanged)
- New script: `generate_with_prompt_variations.py`
- Output directory: `./datasets/` (with variation-specific files)

This implementation allows for systematic evaluation of how different prompt components affect the quality of generated PR descriptions, enabling data-driven optimization of the prompting strategy.
