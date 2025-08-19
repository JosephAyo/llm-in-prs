import pandas as pd
import os
import datetime

# File paths
BASE_CSV = "../pr_files/datasets/sample_by_state.csv"
ZERO_SHOT_CSV = "./datasets/zero_shot_generated.csv"
ONE_SHOT_CSV = "./datasets/one_shot_generated.csv"
FEW_SHOT_CSV = "./datasets/few_shot_generated.csv"
OUTPUT_CSV = "./datasets/sample_by_state_with_generated.csv"
LOG_PATH = "./datasets/merge_output.log"

# === Utility: Logging ===
def log_activity(activity: str, log_path=LOG_PATH):
    log = f"{datetime.datetime.now()}: {activity}\n"
    with open(log_path, "a") as log_file:
        log_file.write(log)

def load_generated_data(csv_file, suffix):
    """Load generated data and extract unique records per PR ID."""
    log_activity(f"Loading {csv_file}...")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        log_activity(f"  Warning: {csv_file} not found, skipping...")
        return pd.DataFrame(columns=['id', f'generated_description_{suffix}'])
    
    df = pd.read_csv(csv_file)
    log_activity(f"  Loaded {len(df)} total rows")
    
    # Group by ID and take the first occurrence (since all files in a PR have same generated title/desc)
    unique_df = df.groupby('id').first().reset_index()
    log_activity(f"  Found {len(unique_df)} unique PR IDs (removed {len(df) - len(unique_df)} duplicate rows)")
    
    # Select only the columns we need and rename them with suffix
    result_df = unique_df[['id', 'generated_description']].copy()
    result_df.rename(columns={
        'generated_description': f'generated_description_{suffix}'
    }, inplace=True)
    
    return result_df

def merge_all_data():
    """Merge all generated data into the base CSV."""
    log_activity("Starting merge process...")
    
    # Load base data
    log_activity(f"Loading base CSV: {BASE_CSV}")
    base_df = pd.read_csv(BASE_CSV)
    log_activity(f"  Base CSV has {len(base_df)} rows")
    
    # Load generated data from each shot type
    zero_shot_df = load_generated_data(ZERO_SHOT_CSV, "zero_shot")
    one_shot_df = load_generated_data(ONE_SHOT_CSV, "one_shot")
    few_shot_df = load_generated_data(FEW_SHOT_CSV, "few_shot")
    
    # Merge with base data
    log_activity("Merging data...")
    merged_df = base_df.copy()
    
    # Merge zero shot
    if not zero_shot_df.empty:
        merged_df = pd.merge(merged_df, zero_shot_df, on='id', how='left')
        log_activity(f"  After zero shot merge: {len(merged_df)} rows")
    
    # Merge one shot
    if not one_shot_df.empty:
        merged_df = pd.merge(merged_df, one_shot_df, on='id', how='left')
        log_activity(f"  After one shot merge: {len(merged_df)} rows")
    
    # Merge few shot
    if not few_shot_df.empty:
        merged_df = pd.merge(merged_df, few_shot_df, on='id', how='left')
        log_activity(f"  After few shot merge: {len(merged_df)} rows")
    
    # Check merge success
    zero_count = merged_df['generated_description_zero_shot'].notna().sum() if 'generated_description_zero_shot' in merged_df.columns else 0
    one_count = merged_df['generated_description_one_shot'].notna().sum() if 'generated_description_one_shot' in merged_df.columns else 0
    few_count = merged_df['generated_description_few_shot'].notna().sum() if 'generated_description_few_shot' in merged_df.columns else 0
    
    log_activity(f"Merge results:")
    log_activity(f"  Zero shot matches: {zero_count}")
    log_activity(f"  One shot matches: {one_count}")
    log_activity(f"  Few shot matches: {few_count}")
    
    # Save the merged data
    log_activity(f"Saving merged data to: {OUTPUT_CSV}")
    merged_df.to_csv(OUTPUT_CSV, index=False)
    
    log_activity(f"✅ Successfully saved merged data with {len(merged_df)} rows")
    log_activity(f"   Added columns: generated_description_zero_shot")
    log_activity(f"                  generated_description_one_shot")
    log_activity(f"                  generated_description_few_shot")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_all_data()
    
    # Display a sample of the merged data to console
    print("\nSample of merged data:")
    sample_cols = ['id', 'title', 'generated_description_zero_shot', 'generated_description_one_shot', 'generated_description_few_shot']
    available_cols = [col for col in sample_cols if col in merged_data.columns]
    print(merged_data[available_cols].head(3).to_string(index=False))
    
    # Also log the completion
    log_activity("Script completed successfully!")