import pandas as pd
import os

def append_csv_files():
    """
    Append additional CSV files to their respective main CSV files.
    """
    base_path = "/Users/ayo/Documents/research-workspace/llm-in-prs/generation/datasets"
    
    # Define the file pairs to append
    file_pairs = [
        ("generated_title_and_pr_zero_shot.csv", "generated_title_and_pr_zero_shot-additional.csv"),
        ("generated_title_and_pr_one_shot.csv", "generated_title_and_pr_one_shot-additional.csv"),
        ("generated_title_and_pr_few_shot.csv", "generated_title_and_pr_few_shot-additional.csv")
    ]
    
    for main_file, additional_file in file_pairs:
        main_path = os.path.join(base_path, main_file)
        additional_path = os.path.join(base_path, additional_file)
        
        # Check if both files exist
        if not os.path.exists(main_path):
            print(f"Warning: Main file {main_file} does not exist, skipping...")
            continue
        
        if not os.path.exists(additional_path):
            print(f"Warning: Additional file {additional_file} does not exist, skipping...")
            continue
        
        try:
            # Read the main CSV file
            main_df = pd.read_csv(main_path)
            print(f"Main file {main_file} has {len(main_df)} rows")
            
            # Read the additional CSV file
            additional_df = pd.read_csv(additional_path)
            print(f"Additional file {additional_file} has {len(additional_df)} rows")
            
            # Append the additional data to the main data
            combined_df = pd.concat([main_df, additional_df], ignore_index=True)
            print(f"Combined data has {len(combined_df)} rows")
            
            # Save the combined data back to the main file
            combined_df.to_csv(main_path, index=False)
            print(f"Successfully appended {additional_file} to {main_file}")
            
            # Optionally, you can rename or remove the additional file after appending
            # os.rename(additional_path, additional_path + ".backup")
            
        except Exception as e:
            print(f"Error processing {main_file} and {additional_file}: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("Starting CSV file appending process...")
    append_csv_files()
    print("CSV file appending process completed.")