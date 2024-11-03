import numpy as np

def expected_calibration_error(probabilities, labels, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for a model's predictions.

    Parameters:
    - probabilities (np.ndarray): Array of predicted probabilities for the positive class.
    - labels (np.ndarray): Array of true binary labels (0 or 1).
    - num_bins (int): Number of bins to use for calibration. Default is 10.

    Returns:
    - ece (float): Expected Calibration Error.
    """
    # Initialize variables
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples within the current bin
        in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
        prop_in_bin = np.mean(in_bin)


        if prop_in_bin > 0:  # Avoid division by zero
            # Calculate average predicted probability and actual accuracy in the bin
            avg_confidence = np.mean(probabilities[in_bin])
            avg_accuracy = np.mean(labels[in_bin])

            print(f"There are {np.sum(in_bin)} samples in the bin {bin_lower} to {bin_upper}")
            print(f"Average confidence: {avg_confidence}")

            # Accumulate the weighted difference between accuracy and confidence
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece
