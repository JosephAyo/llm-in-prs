import pandas as pd

# File paths
DATASET_NAME = "spam_prs"
INPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}.csv"               # CSV A
OUTPUT_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-detection.csv"    # CSV B
MERGED_CSV_PATH = f"./datasets/{DATASET_NAME}/{DATASET_NAME}-merged.csv"       # CSV C

# Load only 'id' and 'labels' from CSV A
df_a = pd.read_csv(INPUT_CSV_PATH, usecols=['id', 'labels'])
df_b = pd.read_csv(OUTPUT_CSV_PATH)

# Merge based on 'id', keeping all rows from df_b
df_merged = df_b.merge(df_a, on='id', how='left')

# Save the result to a new CSV
df_merged.to_csv(MERGED_CSV_PATH, index=False)

print(f"Merged CSV saved to {MERGED_CSV_PATH}")
