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

    # Extract confidence and is_correct columns
    confidence = data['confidence'].to_numpy()
    is_correct = data['is_correct'].to_numpy()

    ece_values = []
    sample_sizes = range(5, len(data) + 1, 5)  # Sample sizes from 10 to total length in steps of 10

    for size in sample_sizes:
        ece = expected_calibration_error(confidence[:size], is_correct[:size])
        ece_values.append(ece)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, ece_values, marker='o', linestyle='-', color='b')
    plt.title('Expected Calibration Error vs. Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.grid()
    plt.xticks(sample_sizes)
    plt.ylim(0, max(ece_values) + 0.05)

    # Draw a horizontal line for the final ECE
    final_ece = ece_values[-1]  # Last ECE value
    plt.axhline(y=final_ece, color='r', linestyle='--', label='Final ECE')
    plt.legend()
    plt.savefig('ece_vs_sample_size.png')  # Save the plot as an image
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate ECE from a CSV file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing confidence and is_correct columns.')
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: The file {args.csv_file} does not exist.")
    else:
        main(args.csv_file)
