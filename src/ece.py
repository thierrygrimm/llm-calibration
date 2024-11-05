from metrics import expected_calibration_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import os


def main(csv_file):
    # Load the data
    data = pd.read_csv(csv_file)

    # Shuffle the rows
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle and reset index

    # Extract confidence and is_correct columns
    confidence = data['confidence'].to_numpy()
    is_correct = data['is_correct'].to_numpy()

    ece_values = []
    sample_sizes = range(20, len(data) + 1, 5)  # Sample sizes starting from 20, incrementing by 5

    for size in sample_sizes:
        ece = expected_calibration_error(confidence[:size], is_correct[:size])
        ece_values.append(ece)

    # Set Seaborn style
    sns.set(style="whitegrid")  # Use a white grid background
    plt.figure(figsize=(12, 7))

    # Plotting as a line without points
    plt.plot(sample_sizes, ece_values, linestyle='-', color='b', label='Expected Calibration Error (ECE)')  # No markers
    plt.title('Expected Calibration Error vs. Sample Size', fontsize=16)
    plt.xlabel('Sample Size', fontsize=14)
    plt.ylabel('Expected Calibration Error (ECE)', fontsize=14)

    # Automatically adjust y-limits to fit the ECE data
    plt.ylim(bottom=min(ece_values), top=max(ece_values))  # Adjust y-limits based on ECE values

    # Draw a horizontal line for the final ECE
    plt.axhline(y=ece_values[-1], color='r', linestyle='--', label=f'Final ECE ({ece_values[-1]:.4f})', linewidth=1.5)

    # Adjust x-ticks to reduce clutter
    plt.xticks(np.arange(20, len(data) + 1, step=400), fontsize=10)  # Set ticks every 400 samples starting from 20

    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(fontsize=12)
    plt.tight_layout()  # Automatically adjust subplot parameters for better fit
    plt.savefig('ece_vs_sample_size.png', dpi=300)  # Save the plot as a high-resolution image
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate ECE from a CSV file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing confidence and is_correct columns.')
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: The file {args.csv_file} does not exist.")
    else:
        main(args.csv_file)
