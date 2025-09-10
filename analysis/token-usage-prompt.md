# Instructions: Context Cost Analysis with Statistical Evaluation (P-1 to P-6)

## Objective

You need to create a notebook similar to `prompt-variations-analysis.ipynb` notebook that performs a **comprehensive cost analysis** of six context variations (P-1 through P-6).  
The notebook must produce **tables, statistical results, and figures** that can be directly included in a research paper.

---

## Required Inputs

- **Generation Data**: Six files named:
  - `generation/datasets/prompt_variation_P-1_generated.csv`
  - `generation/datasets/prompt_variation_P-2_generated.csv`
  - `generation/datasets/prompt_variation_P-3_generated.csv`
  - `generation/datasets/prompt_variation_P-4_generated.csv`
  - `generation/datasets/prompt_variation_P-5_generated.csv`
  - `generation/datasets/prompt_variation_P-6_generated.csv`
- Each file must include at least:
  - `pr_id`
  - `prompt_variation`
  - `prompt_tokens`
  - `completion_tokens`
  - `total_tokens`
  - (optional but recommended) `description`, `word_count`, `sentence_count`

---

## Analysis Workflow

### Step 1: Data Loading
1. Load all six CSV files.
2. Merge into a single dataset with `pr_id` and `prompt_variation` as identifiers.
3. Deduplicate entries if needed.

### Step 2: Descriptive Statistics
For each variation (P-1 … P-6), calculate:

- **Prompt Tokens (input cost)**  
  - mean, median, min, max, std
- **Completion Tokens (output cost)**  
  - mean, median, min, max, std
- **Total Tokens (overall cost)**  
  - mean, median, min, max, std
- **Ratios & Efficiency**  
  - Prompt-to-Completion Ratio = prompt_tokens / completion_tokens  
  - Efficiency Score = completion_tokens / total_tokens  
- **Normalized Costs**  
  - Tokens per word = total_tokens / word_count  
  - Tokens per sentence = total_tokens / sentence_count  

Export as:  
- `context_cost_analysis_results.csv`

### Step 3: Impact & Rankings
1. Choose **P-1 as baseline**.  
2. Compute **% increase/decrease** in mean total tokens relative to P-1.  
3. Rank variations from cheapest to most expensive.  

Export as:  
- `context_cost_ranking.csv`

### Step 4: Statistical Testing
1. Perform **one-way ANOVA** on total tokens across P-1 … P-6.  
   - If data is not normally distributed, use **Kruskal–Wallis test**.  
2. If significant differences exist, run **post-hoc Tukey’s HSD** (or Dunn’s test for non-parametric).  
3. Perform **pairwise t-tests** (Welch’s correction) for robustness.  
4. Compute **effect sizes**:  
   - Eta-squared (η²) for ANOVA  
   - Cohen’s d for pairwise differences  

Export as:  
- `context_cost_significance.csv` (p-values for ANOVA, Tukey/Dunn, t-tests)  
- `context_cost_effect_sizes.csv` (η², Cohen’s d)

### Step 5: Visualizations
Generate publication-ready plots:

1. **Boxplots** of token usage distributions per variation  
   - `token_usage_boxplot.png`  
2. **Bar chart with error bars** (mean ± SE) of total tokens  
   - `mean_token_cost_bars.png`  
3. **Heatmap of pairwise effect sizes or significance**  
   - `pairwise_effect_size_heatmap.png`  
4. (Optional) **Scatterplot of tokens per word vs. tokens per sentence**  
   - `normalized_cost_efficiency.png`

All plots should have:
- Titles, axis labels, and legends  
- Readable font sizes for publication  
- Consistent color scheme  

---

## Reporting Guidelines (for Research Paper)

### Table 1: Descriptive Statistics
| Variation | Mean Prompt Tokens | Mean Completion Tokens | Mean Total Tokens | Prompt/Completion Ratio | Efficiency Score | Tokens/Word | Tokens/Sentence |
|-----------|--------------------|------------------------|-------------------|-------------------------|------------------|-------------|-----------------|

### Table 2: Relative Costs (Baseline = P-1)
| Variation | Mean Total Tokens | % Difference vs. P-1 | Ranking (1=Cheapest) |
|-----------|-------------------|-----------------------|----------------------|

### Table 3: Statistical Significance
| Comparison | Test Used | p-value | Significant (p<0.05) | Effect Size (Cohen’s d) | Interpretation |
|------------|-----------|---------|-----------------------|--------------------------|----------------|

---

## Interpretation Guidelines

- **Statistical Significance**:  
  - p < 0.05 → meaningful difference between variations  
  - p ≥ 0.05 → no significant difference  

- **Effect Size (Cohen’s d)**:  
  - 0.2 = small  
  - 0.5 = medium  
  - 0.8+ = large  

- **Efficiency Trade-offs**:  
  - High prompt-to-completion ratio → costly setup, less payoff  
  - High efficiency score → more tokens spent on generation (better value)  

---

## Deliverables

The notebook must produce the following outputs automatically:

- CSV Files:  
  - `context_cost_analysis_results.csv`  
  - `context_cost_ranking.csv`  
  - `context_cost_significance.csv`  
  - `context_cost_effect_sizes.csv`

- Figures:  
  - `token_usage_boxplot.png`  
  - `mean_token_cost_bars.png`  
  - `pairwise_effect_size_heatmap.png`  
  - (optional) `normalized_cost_efficiency.png`

- Ready-to-use tables for the paper, saved as markdown (`.md`) and CSV:  
  - `table1_descriptive_stats.md`  
  - `table2_relative_costs.md`  
  - `table3_statistical_results.md`

---

## Notes

- Keep analysis modular so new variations (P-7, P-8, …) can be added later.  
- Ensure all outputs are **self-contained** (no manual processing needed).  
- Code must be fully reproducible.