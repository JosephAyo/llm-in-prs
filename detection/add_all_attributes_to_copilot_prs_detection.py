import pandas as pd

# File paths
DATASET_NAME = "copilot-prs"
INPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}.csv"               # CSV A
OUTPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection.csv"    # CSV B
MERGED_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-merged.csv"       # CSV C

# Load both CSVs
df_a = pd.read_csv(INPUT_CSV_PATH)
df_b = pd.read_csv(OUTPUT_CSV_PATH)

# Merge on 'id', with suffixes to detect duplicates
df_merged = df_b.merge(df_a, on='id', how='left', suffixes=('', '_a'))

# Drop duplicated columns from df_a (those that got suffix '_a')
# Keep only the column from df_b (already unsuffixed)
for col in df_b.columns:
    if col != 'id' and f"{col}_a" in df_merged.columns:
        df_merged.drop(columns=[f"{col}_a"], inplace=True)

# Save the cleaned merged result
df_merged.to_csv(MERGED_CSV_PATH, index=False)
print(f"Merged CSV saved to {MERGED_CSV_PATH}")
