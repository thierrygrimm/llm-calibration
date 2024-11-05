import argparse
import ast
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import numpy as np
import pandas as pd
from anthropic import Anthropic
from pydantic import (
    ValidationError,
)
from tqdm import tqdm
from dotenv import load_dotenv

# API key from .env
load_dotenv()

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
    parser.add_argument("--num_threads", type=int, default=15, help="Number of threads to use for processing.")

    args = parser.parse_args()

    # Set the mode based on the model name
    mode = "gpt-4o" if args.model in ["gpt-4o", "gpt-4o-mini"] else "fireworks"
    if mode not in ["gpt-4o", "ollama", "litellm", "fireworks", "claude"]:
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
    elif mode == "gpt-4o":
        openai_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=openai_key))
    elif mode == "litellm":
        # litellm
        client = instructor.from_openai(OpenAI(api_key="anything",base_url="http://0.0.0.0:4000"))
    elif mode == "fireworks":
        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        client = instructor.from_openai(
            OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=fireworks_api_key,
            ),
        mode = instructor.Mode.JSON,
        )
    else:
        client = instructor.from_anthropic(Anthropic())

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

    predictions = [None] * len(train_df)  # Initialize list to preserve order
    confidences = [None] * len(train_df)  # Initialize list to preserve order

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(pred_and_conf, row, client, args.model, args.confidence_type): index
            for index, row in train_df.iterrows()
        }

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            index = futures[future]
            try:
                pred, conf = future.result()
                predictions[index] = pred  # Place result at the correct index
                confidences[index] = conf  # Place result at the correct index
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                predictions[index] = None  # Handle the error appropriately
                confidences[index] = None  # Handle the error appropriately

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
