import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from compare import calculate_ece, calculate_brier_score, calculate_f1_score, extract_model_name_and_type

# Assuming the required functions from metrics are already defined, including calculate_ece, calculate_brier_score, calculate_f1_score

# Define a custom color palette
custom_palette = ["#cde6f8", "#5fa4e4", "#1c6bcb", "#aaa3f2", "#e1e1ff", "#cce6ff"]

# Create a function to generate the table with all permutations and metrics
def generate_metrics_table(folder_path, num_bins=10):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return

    # Initialize DataFrame for all permutations
    all_metrics_df = pd.DataFrame(columns=["Model", "Confidence Type", "Dataset", "ECE", "Brier Score", "F1 Score"])

    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)

        # Extract model name and confidence type
        model_name, confidence_type = extract_model_name_and_type(csv_file)

        # Read CSV data
        data = pd.read_csv(csv_path)

        # Remove rows with missing confidence values
        data = data[data['confidence'].notna()]

        # Calculate metrics by dataset
        for ds_name, subset in data.groupby("ds_name"):
            ece = calculate_ece(subset, num_bins=num_bins)
            brier = calculate_brier_score(subset)
            f1 = calculate_f1_score(subset)

            # Append row to all_metrics_df
            all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([{
                "Model": model_name,
                "Confidence Type": confidence_type,
                "Dataset": ds_name,
                "ECE": round(ece, 3),
                "Brier Score": round(brier, 3),
                "F1 Score": round(f1, 3)
            }])], ignore_index=True)

    print("\nAll Permutations of Metrics Summary:")
    print(all_metrics_df)
    return all_metrics_df

# Call the function to generate the table and store it
folder_path = "../results"  # Update this path to your CSV directory
all_metrics_df = generate_metrics_table(folder_path, num_bins=10)

# Save as csv
all_metrics_df.to_csv("../reports/summarized_metrics.csv")
