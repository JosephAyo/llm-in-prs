# GitHub Copilot Instructions for LLM-in-PRs Research

This repository analyzes LLM-generated pull request descriptions through a comprehensive research pipeline. Understanding the three-phase workflow is essential for effective contributions.

## Architecture Overview

**Three-Phase Research Pipeline:**
1. **Generation** (`generation/`) - Generate PR descriptions using 6 prompt variations (P-1 to P-6)
2. **Detection** (`detection/`) - Run AI detection analysis using ZeroGPT API
3. **Analysis** (`analysis/`) - Statistical analysis, cost evaluation, and textual feature extraction

**Data Flow:** PR metadata → Prompt variations → LLM generation → AI detection → Statistical analysis

## Essential Workflows

### Python Environment Setup
```bash
# Python version is pinned via .python-version
# Always create and activate the pyenv-managed environment
pyenv install --skip-existing $(cat .python-version)
pyenv virtualenv $(cat .python-version) llm-prs-env
pyenv local llm-prs-env

# Install dependencies
pip install -r requirements.txt

# ALWAYS use pyenv exec for Python scripts
pyenv exec python script.py

### Generation Pipeline
```bash
# Single variation generation
pyenv exec python generate_with_prompt_variations.py --variation P-1

# Generate all 6 variations (P-1 through P-6)
pyenv exec python generate_with_prompt_variations.py --all

# Test with mock responses (no API calls)
pyenv exec python generate_with_prompt_variations.py --variation P-2 --mock
```

### Detection Pipeline
```bash
# Run detection on generated content
pyenv exec python detection_on_jabref_prs.py

# Detection supports progress resumption via pickle files
# Check datasets/*/detection-progress.pkl for resumable state
```

### Analysis Execution
**Primary Analysis Notebooks:**
- `context_cost_analysis.ipynb` - Token cost analysis with statistical testing
- `prompt-variations-analysis.ipynb` - Comprehensive prompt variation comparison
- `textual-features-on-prompt-variations.ipynb` - Textual feature extraction

All analysis results (CSV, Markdown tables, plots) must be saved under: 
`analysis/outputs/{analysis-type}/`

## Project-Specific Patterns

### Prompt Variation Matrix (P-1 to P-6)
```python
# Progressive context addition strategy
P-1: repo_name_and_path only (minimal baseline)
P-2: + pr_template_content  
P-3: + pr_title
P-4: + pr_diffs
P-5: + pr_file_contents
P-6: + pr_issue_context (full context)
```

### Data Structure Conventions
- **CSV naming**: `prompt_variation_P-X_generated.csv` for generation outputs
- **Progress tracking**: `.pkl` files for resumable API operations
- **Logging**: All scripts write to `*_output.log` files in datasets directories
- **Entry keys**: Format `{pr_id}_{entry_type}` for detection deduplication

### API Integration Patterns
- **Token rotation**: ZeroGPT API uses cycling token iterator from `env/zero-gpt-tokens.txt`
- **Rate limiting**: Built-in delays and retry mechanisms (15-minute waits on failures)
- **Batch processing**: Configurable batch sizes with progress persistence
- **Error handling**: Continue on individual failures, comprehensive logging

### Statistical Analysis Approach
- **Non-parametric methods**: Kruskal-Wallis tests (data not normally distributed), but ANOVA if assumptions pass.
- **Effect sizes**: Cohen's d for practical significance assessment  
- **Multiple comparisons**: Welch t-tests for pairwise analysis
- **Publication tables**: Automated markdown/CSV table generation

## Critical Integration Points

### Dataset Dependencies
```
pr_files/datasets/ → generation/datasets/ → detection/datasets/ → analysis/
```

### Cross-Component Communication
- Generation outputs feed directly into detection scripts
- Detection results merge with generation data for analysis on `pr_id`, `prompt_variation`
- All components share common PR ID indexing scheme

### External Dependencies
- **OpenAI API**: GPT-4 for PR description generation
- **ZeroGPT API**: AI content detection scoring
- **Jupyter**: Analysis notebooks require kernel configuration

## Development Guidelines

### File Organization
- Module-specific instructions in `{module}/instructions.md`
- Progress summaries in `{module}/summary.md`
- Environment tokens in `{module}/env/` (never commit)

### Testing Patterns
- Mock responses available for generation testing (`--mock` flag)
- Progress resumption testing via pickle manipulation
- Statistical validation through notebook re-execution

### Logging Standards
```python
def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)
```

### Note 
- new variations (P-7+) or detectors can be added without breaking architecture.

This research codebase prioritizes reproducibility, resumable operations, and comprehensive statistical analysis. Always verify environment setup before executing pipelines and check existing progress files before restarting long-running operations.
