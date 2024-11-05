import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from pydantic import (
    ValidationError,
)
from tqdm import tqdm

current_directory = os.getcwd()  # Get the current working directory
src_path = os.path.join(current_directory, "../")  # Navigate to the src folder
sys.path.insert(0, src_path)

from format import *
from templates import *
from eval import *

# Function to map answer letters to indices
ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


# Define a function that returns two values
def pred_and_conf(row, client, model="llama3.1:8b", confidence_type="absolute", encoding_type="uppercase"):
    """
    Make a prediction and calculate the associated confidence from the given context, question, and options.

    Args:
        row (dict): A dictionary containing the context, question, options, and possibly other information.
        confidence_type (str): Whether to calculate the confidence as "absolute" or "relative". Defaults to "absolute".
        model (str): The model to use for making the prediction. Defaults to "llama3".

    Returns:
        tuple: A tuple containing the predicted answer and the calculated confidence.
    """
    json_resp = infer(
        row["context"], row["question"], row["options"], client, model, confidence_type, encoding_type
    )
    pred = json_resp["answer"]
    if json_resp["answer"] != "NaN":
        # For 5 options, there are 5 confidence values
        if "conf_e" in json_resp:
            conf = relative_confidence(
                [
                    json_resp["conf_a"],
                    json_resp["conf_b"],
                    json_resp["conf_c"],
                    json_resp["conf_d"],
                    json_resp["conf_e"],
                ],
                json_resp["answer"],
            )
        # For 4 options, there are 4 confidence values
        elif "conf_d" in json_resp:
            conf = relative_confidence(
                [
                    json_resp["conf_a"],
                    json_resp["conf_b"],
                    json_resp["conf_c"],
                    json_resp["conf_d"],
                ],
                json_resp["answer"],
            )
        # For absolute confidence, there is only 1 confidence value
        else:
            conf = json_resp["confidence"]
    else:
        print("No success in this row")
        return np.nan, np.nan
    return pred, conf


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run predictions with confidence calculation.")
    parser.add_argument("--data_path", type=str, default="../data/subset1000.csv", help="Path to the CSV data file.")
    parser.add_argument("--confidence_type", type=str, choices=["absolute", "relative"], default="absolute",
                        help="Type of confidence calculation to use.")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name to use for predictions.")
    parser.add_argument("--limit", type=int, default=0, help="Number of rows to process (0 for all rows).")

    args = parser.parse_args()

    # Set the mode based on the model name
    mode = "gpt-4o" if args.model == "gpt-4o" else "ollama"
    if mode not in ["gpt-4o", "ollama"]:
        raise ValueError(f"Unsupported mode '{mode}'. Only 'gpt-4o' and 'ollama' are supported.")

    if mode == "ollama":
        # enables `response_model` in create call
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
    else:
        openai_key = os.getenv("OLLAMA_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=openai_key))

    predictions = []
    confidences = []

    # Load data
    train_df = pd.read_csv(args.data_path)

    # Preprocess data: remove options with more than 5 choices
    train_df["options"] = train_df["options"].apply(ast.literal_eval)
    train_df = train_df[train_df["options"].apply(len) <= 5].reset_index()
    # Convert context to string to avoid AttributeError
    train_df['context'] = train_df['context'].astype(str)

    # Limit the number of rows to process
    if args.limit > 0:
        train_df = train_df.head(args.limit)

    # Use tqdm to show progress
    for index, row in tqdm(
            train_df.iterrows(), total=train_df.shape[0], desc="Processing rows"
    ):
        pred, conf = pred_and_conf(row, client, confidence_type=args.confidence_type, model=args.model)
        predictions.append(pred)
        confidences.append(conf)

    # Save predictions and confidences to a CSV file
    os.makedirs("../results", exist_ok=True)

    # Sanitize model name by replacing slashes with underscores and save to CSV
    safe_model_name = args.model.replace("/", "_")
    output_filename = f"../results/predictions_{safe_model_name}_{args.confidence_type}_confidence.csv"
    output_df = train_df.copy()
    output_df["prediction"] = predictions
    output_df["confidence"] = confidences

    # Map predictions to indices and calculate is_correct column
    output_df["is_correct"] = (output_df["prediction"].map(ANSWER_MAP) == output_df["correct_answer"]).astype(int)

    # Save to CSV
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

    print(f"Average confidence: {output_df['confidence'].mean() * 100:.2f}%")
    print(f"Accuracy: {output_df['is_correct'].mean() * 100:.2f}%")
    print("Done!")
