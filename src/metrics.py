"""
Description:
Functions for formatting and constructing prompts for answering questions.

Functions:
- expected_calibration_error: Calculates the Expected Calibration Error (ECE) for a model's predictions.
- plot_calibration_with_density: Plots a calibration curve with ECE and a density plot of predicted probabilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

            # Accumulate the weighted difference between accuracy and confidence
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return ece


def plot_calibration_with_density(predicted_probs, true_labels, num_bins=5):
    """
    Plots a calibration curve with Expected Calibration Error (ECE) and a density plot of predicted probabilities.

    Parameters:
    - predicted_probs (np.ndarray): Array of predicted probabilities for the positive class.
    - true_labels (np.ndarray): Array of true binary labels (0 or 1).
    - num_bins (int): Number of bins to use for calibration. Default is 5.
    """
    # Calculate ECE
    ece = expected_calibration_error(predicted_probs, true_labels, num_bins=num_bins)

    # Create bins for the predicted probabilities
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1

    # Calculate true probabilities and predicted probabilities for each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    prob_true = []
    prob_pred = []

    for i in range(num_bins):
        in_bin = (bin_indices == i)
        if np.any(in_bin):
            prob_true.append(true_labels[in_bin].mean())
            prob_pred.append(predicted_probs[in_bin].mean())
        else:
            prob_true.append(0)
            prob_pred.append(0)

    prob_true = np.array(prob_true)
    prob_pred = np.array(prob_pred)

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 8), sharex=True)

    # Plot calibration curve
    ax1.plot(bin_centers, prob_true, marker='o', label='True Probability', color='blue')

    # Fill areas above and below the diagonal with lighter colors
    ax1.fill_between(bin_centers, prob_pred, prob_true, where=(prob_pred > prob_true),
                     interpolate=True, color='lightcoral', alpha=0.3, label='Overestimated')  # Light Red
    ax1.fill_between(bin_centers, prob_pred, prob_true, where=(prob_pred <= prob_true),
                     interpolate=True, color='lightblue', alpha=0.3, label='Underestimated')  # Light Blue

    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # Diagonal line
    ax1.set_title("Calibration Curve with ECE")
    ax1.legend(loc="upper left")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(-0.05, 1.05)  # Adjust y limits for better visibility
    ax1.grid()

    # Plot density of predicted probabilities
    sns.histplot(predicted_probs, bins=20, kde=True, color='skyblue', ax=ax2)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.grid()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# Execute if this script is called directly
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    num_samples = 1000

    # Generate true labels
    true_labels = np.random.randint(0, 2, size=num_samples)  # Random binary labels

    # Generate predicted probabilities with noise
    predicted_probs = np.zeros(num_samples)

    # Assign predicted probabilities with a mix of random data
    for i in range(num_samples):
        if true_labels[i] == 1:
            # Predicted probabilities for positive class with some noise
            predicted_probs[i] = np.clip(np.random.normal(loc=0.7, scale=0.3), 0, 1)
        else:
            # Predicted probabilities for negative class with some noise
            predicted_probs[i] = np.clip(np.random.normal(loc=0.3, scale=0.15), 0, 1)

    # Call the function to plot
    plot_calibration_with_density(predicted_probs, true_labels, num_bins=10)
