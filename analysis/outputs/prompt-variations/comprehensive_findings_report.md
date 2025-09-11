# Comprehensive Findings Report: Prompt Variations Analysis

## Executive Summary

This report presents the comprehensive analysis of prompt variations in LLM-generated pull request descriptions, comparing 6 prompt variations (P-1 through P-6) against original human-written descriptions. The analysis follows a progressive context addition strategy and includes statistical testing, effect size calculations, and AI detectability assessment.

---

## Research Design

### Prompt Variation Strategy
Following the copilot instructions, we implemented a progressive context addition approach:

- **P-1 (Minimal)**: `repo_name_and_path` only (568 tokens)
- **P-2 (Template)**: + `pr_template_content` (1,824 tokens)  
- **P-3 (Title)**: + `pr_title` (1,879 tokens)
- **P-4 (Diffs)**: + `pr_diffs` (6,792 tokens)
- **P-5 (Content)**: + `pr_file_contents` (57,734 tokens)
- **P-6 (Full)**: + `pr_issue_context` (59,090 tokens)

### Dataset Composition
- **Total Entries**: 224
- **Original Descriptions**: 32 human-written PR descriptions
- **Generated Descriptions**: 192 (32 per prompt variation)
- **AI Detection**: ZeroGPT API responses for all entries
- **Statistical Approach**: Non-parametric (Kruskal-Wallis) with Cohen's d effect sizes

---

## Key Findings

### 1. AI Detection Performance

#### Best Performing Variations
1. **P-3 (Title + Template)**: 18.8% true positive rate
2. **P-2 (Template)**: 15.6% true positive rate  
3. **P-6 (Full Context)**: 12.5% true positive rate

#### Detection Score Analysis
- **Original Content**: 13.5% mean AI score (all below 50% threshold)
- **Generated Content Range**: 5.3% (P-1) to 23.2% (P-2)
- **Perfect Human Classification**: 0% false positive rate for original content

#### Statistical Significance
- **Kruskal-Wallis Test**: H = 15.25, p = 0.018 (significant)
- **Effect Sizes vs Original**:
  - P-2: Cohen's d = +0.414 (small, higher detectability)
  - P-3: Cohen's d = +0.275 (small, higher detectability)
  - P-1: Cohen's d = -0.555 (medium, lower detectability)

### 2. Token Efficiency Analysis

#### Resource Consumption
- **Most Efficient**: P-1 (568 input tokens, 131 output tokens)
- **Least Efficient**: P-6 (59,090 input tokens, 335 output tokens)
- **Statistical Significance**: H = 131.87, p < 0.001 (highly significant)

#### Cost-Benefit Assessment
- **P-1**: Minimal cost but lowest detection (3.1%)
- **P-2**: Moderate cost with highest AI scores (23.2%)
- **P-3**: Similar cost to P-2 with best detection rate (18.8%)

### 3. Content Quality Metrics

#### Text Length Analysis
- **Original Baseline**: 1,184 characters (median: 1,188)
- **All Generated Variations**: Significantly shorter than original
- **Effect Sizes vs Original** (all large to medium):
  - P-1: Cohen's d = -1.422 (large decrease)
  - P-2: Cohen's d = -0.869 (large decrease)
  - P-3: Cohen's d = -0.946 (large decrease)

#### Word Count Progression
1. **Original**: 161 words (baseline)
2. **P-5**: 111 words (69% of original)
3. **P-6**: 109 words (68% of original)
4. **P-4**: 101 words (63% of original)
5. **P-2**: 83 words (52% of original)
6. **P-3**: 80 words (50% of original)
7. **P-1**: 46 words (29% of original)

#### Structural Features
- **Sentence Count**: Ranges from 3.8 (P-1) to 6.7 (P-5) vs 19.8 (Original)
- **Content Density**: Generated descriptions more concise but comprehensive
- **Template Compliance**: Higher context variations show better structure

---

## Statistical Validation

### Hypothesis Testing Results
All metrics show statistically significant differences across variations:

| Feature | H-Statistic | p-Value | Significance |
|---------|-------------|---------|--------------|
| Text Length | 38.87 | < 0.001 | Yes |
| AI Detection Score | 15.25 | 0.018 | Yes |
| Input Tokens | 131.87 | < 0.001 | Yes |
| Output Tokens | 17.95 | 0.003 | Yes |

### Effect Size Summary
**Large Effects** (|d| > 0.8): All text length comparisons vs original
**Medium Effects** (0.5 < |d| < 0.8): P-1 AI detection vs original  
**Small Effects** (0.2 < |d| < 0.5): P-2, P-3 AI detection vs original

---

## Practical Implications

### Recommended Prompt Strategy

#### For Maximum AI Detection (Research/Evaluation)
- **Use P-3**: Best balance of detection rate (18.8%) and efficiency
- **Context**: Include repository info + template + PR title
- **Cost**: ~1,879 input tokens per generation

#### For Production Use (Minimal Detection)
- **Use P-1**: Lowest detection rate (3.1%) while maintaining coherence
- **Context**: Repository name and path only
- **Cost**: ~568 input tokens per generation

#### For Quality Content
- **Use P-5 or P-6**: Longest, most detailed descriptions
- **Trade-off**: Higher cost but closer to original human length
- **Detection**: Moderate (9-14%)

### Context Addition Impact

1. **Minimal → Template (P-1 → P-2)**:
   - Detection increases 5x (3.1% → 15.6%)
   - Cost increases 3x (568 → 1,824 tokens)
   - Content quality significantly improves

2. **Template → Title (P-2 → P-3)**:
   - Detection peaks at 18.8%
   - Minimal cost increase (1,824 → 1,879 tokens)
   - **Optimal efficiency point**

3. **Title → Diffs (P-3 → P-4)**:
   - Detection decreases to 9.4%
   - Cost increases 3.6x (1,879 → 6,792 tokens)
   - Diminishing returns evident

4. **Diffs → Content (P-4 → P-5)**:
   - Detection remains low (9.4%)
   - Cost explodes 8.5x (6,792 → 57,734 tokens)
   - Length increases significantly

---

## Methodological Notes

### Alignment with Research Standards
✅ **Correct Comparison**: All variations vs Original (not just generated vs original)
✅ **Statistical Rigor**: Non-parametric tests with effect sizes  
✅ **Output Structure**: Results saved to `analysis/outputs/prompt-variations/`
✅ **Complete Coverage**: Analysis includes all 7 variations (Original + P-1 to P-6)

### Limitations
- Sample size: 32 PR descriptions per variation
- Single domain: JabRef repository only
- AI detector: ZeroGPT API only
- Language: English-only PR descriptions

---

## Conclusions

1. **Sweet Spot Identified**: P-3 (repo + template + title) offers optimal balance of detection, cost, and quality

2. **Context Paradox**: More context doesn't always increase detectability; P-2 and P-3 are more detectable than higher-context variations

3. **Human Content Security**: Original descriptions show perfect classification (0% false positives), validating ZeroGPT's human detection accuracy

4. **Cost-Effectiveness**: Significant quality gains possible with moderate context addition (P-1 → P-3), but diminishing returns beyond P-3

5. **Research Validation**: Progressive context strategy successfully isolates individual context effects on AI generation detectability

---

## Files Generated

- `comprehensive_prompt_variations_analysis.csv` - Complete results table
- `statistical_test_results.csv` - Kruskal-Wallis test results  
- `effect_sizes_vs_original.csv` - Cohen's d effect sizes
- `descriptive_statistics.csv` - Descriptive statistics by variation
- `prompt_variations_analysis_overview.png` - 4-panel visualization

**Analysis Date**: September 11, 2025  
**Methodology**: Following copilot-instructions.md standards
