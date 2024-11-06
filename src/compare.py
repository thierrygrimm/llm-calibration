import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, f1_score

# Import the function for ECE calculation
from metrics import expected_calibration_error

# Function to extract model name and type remains the same
def extract_model_name_and_type(file_name):
    patterns = [
        (r"llama-v3p(\d)-(\d+)b-instruct_(absolute|relative)_confidence",
         lambda m: (f"Llama 3.{m.group(1)}: {m.group(2)}B Instruct", m.group(3))),
        (r"claude-(\d+)-(\d+)-haiku-\d{8}_(absolute|relative)_confidence",
         lambda m: (f"Claude {m.group(1)}.{m.group(2)} Haiku", m.group(3))),
        (r"gpt-4o-mini_(absolute|relative)_confidence",
         lambda m: ("GPT-4o-mini", m.group(1))),
    ]
    for pattern, formatter in patterns:
        match = re.search(pattern, file_name)
        if match:
            return formatter(match)
    return (file_name, "unknown")


def calculate_ece(data, num_bins=10):
    predicted_probs = data['confidence'].values
    true_labels = data['is_correct'].values.astype(int)
    return expected_calibration_error(predicted_probs, true_labels, num_bins=num_bins)


def calculate_brier_score(data):
    predicted_probs = data['confidence'].values
    true_labels = data['is_correct'].values.astype(int)
    return brier_score_loss(true_labels, predicted_probs)


def calculate_f1_score(data):
    predicted_classes = data['confidence'].round().astype(int)
    true_labels = data['is_correct'].values.astype(int)
    return f1_score(true_labels, predicted_classes)


def calculate_metrics_by_dataset(data, num_bins=10):
    metrics_by_ds = []
    for ds_name, subset in data.groupby("ds_name"):
        ece = calculate_ece(subset, num_bins=num_bins)
        brier = calculate_brier_score(subset)
        f1 = calculate_f1_score(subset)
        metrics_by_ds.append((ds_name, ece, brier, f1))
    return pd.DataFrame(metrics_by_ds, columns=["Dataset", "ECE", "Brier Score", "F1 Score"])


def check_missing_values(data, model_name, calib_type):
    """
    Check for missing values in confidence and is_correct columns for a specific model.
    """
    missing_values = {
        "Model": model_name,
        "Calibration Type": calib_type,
        "Missing Confidence": data['confidence'].isna().sum(),
        "Missing is_correct": data['is_correct'].isna().sum()
    }
    return missing_values


def main(folder_path, num_bins=10):
    """
    Main function to process all CSV files in a folder and display ECE metrics.
    """
    # Define custom color palette
    custom_palette = ["#cde6f8", "#5fa4e4", "#1c6bcb", "#aaa3f2", "#e1e1ff", "#cce6ff"]


    # Get a list of CSV files in the specified folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return

    # Initialize DataFrames for ECE values and missing values
    ece_table = pd.DataFrame(columns=["Model", "Calibration Type", "ECE"])
    missing_table = pd.DataFrame(columns=["Model", "Calibration Type", "Missing Confidence", "Missing is_correct"])
    metrics_by_dataset_table = pd.DataFrame(
        columns=["Model", "Calibration Type", "Dataset", "ECE", "Brier Score", "F1 Score"])

    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        print(f"Processing file: {csv_file}")

        # Extract model name and calibration type
        model_name, calib_type = extract_model_name_and_type(csv_file)

        # Read the data
        data = pd.read_csv(csv_path)

        # Check for missing values and add to missing_table
        missing_values = check_missing_values(data, model_name, calib_type)
        missing_table = pd.concat([missing_table, pd.DataFrame([missing_values])], ignore_index=True)

        # Sanitize data by removing rows with missing confidence
        data = data[data['confidence'].notna()]

        # Calculate ECE for the sanitized data and add to ece_table
        ece = calculate_ece(data, num_bins=num_bins)
        if ece is not None:
            ece_table = pd.concat([ece_table,
                                   pd.DataFrame([[model_name, calib_type, ece]],
                                                columns=["Model", "Calibration Type", "ECE"])],
                                  ignore_index=True)

        # Calculate metrics grouped by dataset and add to metrics_by_dataset_table
        metrics_by_dataset = calculate_metrics_by_dataset(data, num_bins=num_bins)
        metrics_by_dataset["Model"] = model_name
        metrics_by_dataset["Calibration Type"] = calib_type
        metrics_by_dataset_table = pd.concat([metrics_by_dataset_table, metrics_by_dataset], ignore_index=True)

    # Display missing values table
    print("\nMissing Values Summary:")
    print(missing_table)

    # Check if missing_table has any data before plotting
    if not missing_table.empty and missing_table[["Missing Confidence", "Missing is_correct"]].sum().sum() > 0:
        # Plot missing values by calibration type if missing data exists
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid", context="paper")
        sns.barplot(
            x="Model",
            y="Missing Confidence",
            hue="Calibration Type",
            data=missing_table,
            palette=custom_palette[:2],
            errorbar=None,
            dodge=True
        )
        plt.ylabel("Number of Missing Values")
        plt.xlabel("Model")
        plt.title("Missing Values by Model and Calibration Type")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Calibration Type")
        plt.tight_layout()
        plt.show()
    else:
        print("All models have complete data; no missing values to plot.")

    print("\nExpected Calibration Error (ECE) by Dataset for each model:")
    print(metrics_by_dataset_table)

    plt.figure(figsize=(12, 7))
    sns.barplot(x="Model", y="ECE", hue="Dataset", data=metrics_by_dataset_table, palette=custom_palette, errorbar=None)
    plt.legend(title="Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("ECE")
    plt.xlabel("Model")
    plt.title("ECE by Dataset for each Model")
    plt.tight_layout()
    plt.show()

    ece_grouped = ece_table.groupby(["Calibration Type", "Model"], as_index=False).mean()
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Model", y="ECE", hue="Calibration Type", data=ece_grouped, palette=custom_palette[:2], errorbar=None)
    plt.ylabel("ECE")
    plt.xlabel("Model")
    plt.title("ECE Grouped by Calibration Type Across All Models")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Calibration Type")
    plt.tight_layout()
    plt.show()

    # Aggregate metrics across all models
    aggregated_metrics = metrics_by_dataset_table.groupby("Dataset")[["ECE", "Brier Score", "F1 Score"]].mean().reset_index()

    # Plotting Brier Score and F1 Score for each dataset
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x="Dataset", y="Brier Score", data=aggregated_metrics, palette=custom_palette[:3])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Brier Score")
    plt.title("Average Brier Score by Dataset")

    plt.subplot(1, 2, 2)
    sns.barplot(x="Dataset", y="F1 Score", data=aggregated_metrics, palette=custom_palette[:3])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("F1 Score")
    plt.title("Average F1 Score by Dataset")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ECE, Brier, and F1 from CSV files and plot values.")
    parser.add_argument('--num_bins', type=int, default=10, help="Number of bins for calibration calculation.")
    args = parser.parse_args()

    main("../results", args.num_bins)
