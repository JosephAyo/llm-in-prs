
# Textual Features Analysis Summary Report

## Dataset Overview
- Total entries analyzed: 384
- Valid entries with complete features: 384
- Original content entries: 32
- Generated content entries: 352
- Prompt variations analyzed: 11

## Key Features Extracted
- Basic text statistics (length, word count, sentence count)
- Stop word ratios
- Code element detection
- Question and sentence pattern analysis
- Readability scores
- Cosine similarity with 5 code categories

## Key Insights
- Generated content is -44.6% different in word count compared to original
- Generated content has -16.4% different stop word usage
- Generated content has -99.6% different code element usage
- Generated content has 0.0363 avg similarity vs 0.0404 for original

## Prompt Variations Analyzed
- P-10_Full_Plus_One_Shot
- P-11_Full_Plus_Few_Shot
- P-1_Minimal
- P-2_Basic
- P-3_Diffs_Only
- P-4_Diffs_Plus_Title
- P-5_Code_Only
- P-6_Issue_Only
- P-7_Template_Plus_Title
- P-8_Full_Context
- P-9_Basic_One_Shot


## Output Files Generated
- prompt_variation_textual_features_summary.csv
- original_vs_generated_comparison.csv  
- prompt_variation_detailed_analysis.csv
- textual_features_analysis_visualization.png
- prompt_variation_comparison.png
- textual_features_full_dataset.csv (complete dataset with features)

## Methodology
This analysis used the unified TextualFeatureExtractor pipeline to extract comprehensive
textual features from PR descriptions across different prompt variations. The analysis
compares original human-written content with AI-generated content to identify patterns
and differences in textual characteristics.

Generated on: 2025-09-04 01:43:48
