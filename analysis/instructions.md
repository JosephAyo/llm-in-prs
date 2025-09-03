# Instructions: Prompt Variations Analysis

## Overview

This guide walks you through using the `prompt-variations-analysis.ipynb` notebook to analyze different prompt variations for generating PR descriptions and evaluate their detectability by AI detection tools (ZeroGPT).

## Prerequisites

Before running this analysis, ensure you have:

### Required Data Files

- **Generation Data**: Multiple `generation/datasets/prompt_variation_P-*_generated.csv` files containing PR descriptions generated with different prompt variations and their token usage metrics
- **Detection Data**: `detection/datasets/prompt_variations/prompt_variations-detection.csv` containing ZeroGPT detection results for all variations

### Setup Requirements

1. Ensure all prompt variation CSV files are present in `generation/datasets/`
2. Verify the detection results file exists at `detection/datasets/prompt_variations/prompt_variations-detection.csv`
3. Make sure both datasets can be merged using `pr_id` and `prompt_variation` as common keys

## How to Run the Analysis

Follow these steps to execute the complete analysis:

### Step 1: Data Loading and Preprocessing

1. **Load Detection Results**: The notebook will read the ZeroGPT detection results from the CSV file
2. **Discover Prompt Variations**: Automatically find all `prompt_variation_P-*_generated.csv` files in the generation datasets folder
3. **Combine Generation Data**: Merge all prompt variation files into a single dataset
4. **Deduplicate Records**: Remove duplicate entries by grouping on PR ID and prompt variation

### Step 2: Data Merging and Validation

1. **Merge Datasets**: Combine generation and detection data using PR ID and prompt variation as keys
2. **Validate Merge**: Verify data integrity and check for missing records
3. **Prepare Analysis Dataset**: Create the final dataset with all required metrics

### Step 3: Calculate Metrics

1. **Token Usage Analysis**: Compute prompt and completion token statistics for cost analysis
2. **Text Quality Metrics**: Calculate description length, word count, and sentence count statistics
3. **AI Detection Performance**: Evaluate true positive rates and AI probability scores
4. **Extended Features**: Generate additional textual features for comprehensive analysis

### Step 4: Generate Results

1. **Create Summary Tables**: Generate comprehensive analysis tables with all metrics
2. **Export Results**: Save analysis results to CSV files for further use
3. **Generate Comparisons**: Create side-by-side description comparisons for presentation

## What You'll Get

The analysis will provide you with several key capabilities:

### Analysis Features

- **Cost Efficiency Comparison**: Identify the most token-efficient prompts to minimize API costs
- **Detection Evasion Assessment**: Evaluate how well each prompt variation avoids AI detection
- **Content Quality Evaluation**: Analyze generated content characteristics like length, complexity, and structure
- **AI Detection Scoring**: Get detailed ZeroGPT confidence scores for both generated and original content
- **Presentation Materials**: Generate side-by-side comparisons suitable for research presentations

### Output Files You'll Receive

- `prompt_variations_analysis_results.csv`: Core analysis metrics for all prompt variations
- `comprehensive_prompt_variations_analysis.csv`: Extended analysis including textual features
- `side_by_side_comparison_PR_*.md`: Markdown files with description comparisons ready for presentations

## Understanding Your Results

The analysis generates a comprehensive table with 18 key metrics. Here's how to interpret each column:

### Token Usage Metrics (Cost Analysis)

| Metric | What It Tells You |
|--------|-------------------|
| **mean_prompt_tokens** | Average input tokens consumed - use for cost estimation |
| **median_prompt_tokens** | Median input tokens - shows typical usage per prompt |
| **mean_completion_tokens** | Average output tokens generated - affects response cost |
| **median_completion_tokens** | Median output tokens - typical generation length |

### Content Quality Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **mean_description_length** | Average character count - indicates response verbosity |
| **median_description_length** | Median character count - typical response size |
| **mean_word_count** | Average words per description - content density measure |
| **median_word_count** | Median words per description - typical content length |
| **mean_sentence_count** | Average sentences - indicates structural complexity |
| **median_sentence_count** | Median sentences - typical structural complexity |

### AI Detection Performance

| Metric | What It Tells You |
|--------|-------------------|
| **true_positive_pct** | Percentage of AI content correctly detected (higher = more detectable) |
| **false_negative_pct** | Percentage of human content incorrectly flagged as AI (should be ~0%) |
| **mean_ai_score_generated** | Average AI confidence for generated content (0-100%) |
| **median_ai_score_generated** | Median AI confidence for generated content |
| **mean_ai_score_original** | Average AI confidence for human content (baseline ~13.5%) |
| **median_ai_score_original** | Median AI confidence for human content |

### Sample Size

| Metric | What It Tells You |
|--------|-------------------|
| **total_samples** | Number of descriptions analyzed - indicates result reliability |

## How to Interpret Key Insights

Use these analysis results to make informed decisions:

### 1. Choosing the Best Prompt for Your Needs

- **For Maximum Stealth**: Look for prompts with the lowest `true_positive_pct` (hardest to detect)
- **For Cost Efficiency**: Choose prompts with low `mean_prompt_tokens` but acceptable content quality
- **For Quality Output**: Balance `mean_word_count` and `mean_sentence_count` with detection rates

### 2. Understanding Detection Performance

- **Baseline Reference**: Original human content averages 13.5% AI score with 0% false positives
- **Good Performance**: Generated content should ideally score below 50% AI probability
- **Detection Threshold**: ZeroGPT uses 50% as the AI/Human classification boundary

### 3. Cost-Benefit Analysis

- **High Token Usage**: P-10 and P-11 use ~64k-75k prompt tokens but achieve higher detection rates (28%)
- **Low Token Usage**: P-1 and P-2 use ~632-681 prompt tokens with minimal detection (3-6%)
- **Balanced Options**: P-6 and P-9 offer moderate token usage with reasonable detection evasion

### 4. Research Insights

- **Most Detectable**: P-9 and P-10 variations (28.1% true positive rate)
- **Most Stealthy**: P-1 Minimal variation (3.1% true positive rate)
- **Most Efficient**: P-1 Minimal uses only 632 average prompt tokens
- **Baseline Accuracy**: ZeroGPT correctly identifies 100% of human content (0% false positive rate)
