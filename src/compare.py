import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the function for ECE calculation
from metrics import expected_calibration_error


def extract_model_name_and_type(file_name):
    """
    Extract and format model name and type (absolute/relative) from the file name.

    Parameters:
    - file_name (str): The name of the file.

    Returns:
    - tuple: (Formatted model name, calibration type)
    """
    # Define regex patterns for different models
    patterns = [
        (r"llama-v3p(\d)-(\d+)b-instruct_(absolute|relative)_confidence",
         lambda m: (f"Llama 3.{m.group(1)}: {m.group(2)}B Instruct", m.group(3))),
        (r"claude-(\d+)-(\d+)-haiku-\d{8}_(absolute|relative)_confidence",
         lambda m: (f"Claude {m.group(1)}.{m.group(2)} Haiku", m.group(3))),
        (r"gpt-4o-mini_(absolute|relative)_confidence",
         lambda m: ("GPT-4o-mini", m.group(1))),
    ]

    # Try each pattern and return the formatted name and type if there's a match
    for pattern, formatter in patterns:
        match = re.search(pattern, file_name)
        if match:
            return formatter(match)

    # Default to the original file name if no pattern matches
    return (file_name, "unknown")


def calculate_ece(data, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for the given DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing predictions.
    - num_bins (int): Number of bins for ECE calculation.

    Returns:
    - ece (float): The calculated ECE value.
    """
    # Extract confidence and is_correct data
    predicted_probs = data['confidence'].values
    true_labels = data['is_correct'].values.astype(int)  # Ensure binary labels (0 or 1)

    # Calculate the ECE
    ece = expected_calibration_error(predicted_probs, true_labels, num_bins=num_bins)
    return ece


def check_missing_values(data, model_name, calib_type):
    """
    Check for missing values in confidence and is_correct columns for a specific model.

    Parameters:
    - data (pd.DataFrame): The dataframe containing data for a model.
    - model_name (str): Name of the model for labeling in the plot.
    - calib_type (str): Calibration type (absolute/relative).

    Returns:
    - dict: Number of missing values in confidence and is_correct columns.
    """
    missing_values = {
        "Model": model_name,
        "Calibration Type": calib_type,
        "Missing Confidence": data['confidence'].isna().sum(),
        "Missing is_correct": data['is_correct'].isna().sum()
    }
    return missing_values


def calculate_ece_by_dataset(data, num_bins=10):
    """
    Calculate ECE values grouped by dataset name for each subset.

    Parameters:
    - data (pd.DataFrame): The data containing predictions.
    - num_bins (int): Number of bins for ECE calculation.

    Returns:
    - pd.DataFrame: ECE values for each dataset subset.
    """
    ece_by_ds = []
    for ds_name, subset in data.groupby("ds_name"):
        predicted_probs = subset['confidence'].values
        true_labels = subset['is_correct'].values.astype(int)
        ece = expected_calibration_error(predicted_probs, true_labels, num_bins=num_bins)
        ece_by_ds.append((ds_name, ece))
    return pd.DataFrame(ece_by_ds, columns=["Dataset", "ECE"])


def main(folder_path, num_bins=10):
    """
    Main function to process all CSV files in a folder and display ECE metrics.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - num_bins (int): Number of bins for ECE calculation.
    """
    # Get a list of CSV files in the specified folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return

    # Initialize DataFrames for ECE values and missing values
    ece_table = pd.DataFrame(columns=["Model", "Calibration Type", "ECE"])
    missing_table = pd.DataFrame(columns=["Model", "Calibration Type", "Missing Confidence", "Missing is_correct"])
    ece_by_dataset_table = pd.DataFrame(columns=["Model", "Calibration Type", "Dataset", "ECE"])

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

        # Calculate ECE grouped by dataset and add to ece_by_dataset_table
        ece_by_dataset = calculate_ece_by_dataset(data, num_bins=num_bins)
        ece_by_dataset["Model"] = model_name
        ece_by_dataset["Calibration Type"] = calib_type
        ece_by_dataset_table = pd.concat([ece_by_dataset_table, ece_by_dataset], ignore_index=True)

    # Display missing values table
    print("\nMissing Values Summary:")
    print(missing_table)

    # Plot missing values
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid", context="paper")

    # Create a grouped bar plot for missing values
    sns.barplot(x="Model", y="Missing Confidence", hue="Calibration Type", data=missing_table,
                palette=["lightblue", "lightcoral"], ci=None, dodge=True)
    plt.ylabel("Number of Missing Values")
    plt.xlabel("Model")
    plt.title("Missing Values by Model and Calibration Type")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Calibration Type")
    plt.tight_layout()
    plt.show()

    # Display ECE by dataset table
    print("\nExpected Calibration Error (ECE) by Dataset for each model:")
    print(ece_by_dataset_table)

    # Plot ECE by dataset
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Model", y="ECE", hue="Dataset", data=ece_by_dataset_table, palette="Set2", ci=None)
    plt.legend(title="Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("ECE")
    plt.xlabel("Model")
    plt.title("ECE by Dataset for each Model")
    plt.tight_layout()
    plt.show()

    # New plot for ECE grouped by Calibration Type
    ece_grouped = ece_table.groupby(["Calibration Type", "Model"], as_index=False).mean()
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Model", y="ECE", hue="Calibration Type", data=ece_grouped, palette="colorblind", ci=None)
    plt.ylabel("ECE")
    plt.xlabel("Model")
    plt.title("ECE Grouped by Calibration Type Across All Models")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Calibration Type")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ECE from CSV files and plot ECE values.")
    parser.add_argument('--num_bins', type=int, default=10, help="Number of bins for calibration calculation.")
    args = parser.parse_args()

    # Call main function with the provided arguments
    main("../results", args.num_bins)
