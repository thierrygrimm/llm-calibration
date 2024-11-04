"""
Description:
Evaluate the model's predictions.

Functions:
- evaluate_answer: Evaluate whether the model's answer is correct.
- relative_confidence: Calculate the relative confidence of a given answer.
- infer: Let the model infer the answer based on the provided context, question, options, and confidence type.
"""
import os
import instructor
import numpy as np
from instructor.exceptions import InstructorRetryException
from openai import OpenAI
from pydantic import ValidationError

from format import construct_prompt
from templates import *


def evaluate_answer(model_answer, correct_answer):
    """
        Evaluate whether the model's answer is correct.

        Args:
            model_answer (str): The answer predicted by the model.
            correct_answer (int): The correct answer.

        Returns:
            bool: Whether the model's answer is correct.
        """
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return answer_map.get(model_answer) == correct_answer


def relative_confidence(confidence_list, given_answer):
    """
    Calculate the relative confidence of a given answer.

    :param confidence_list: A list of confidence values for each answer option (A-E).
    :type confidence_list: list
    :param given_answer: The answer for which to calculate the relative confidence (A-E).
    :type given_answer: str
    :return: The relative confidence of the given answer.
    :rtype: float
    :raises KeyError: If the given answer is not a valid option (A-E).
    """
    total_confidence = sum(confidence_list)
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return confidence_list[answer_map.get(given_answer)] / total_confidence


def infer(context, question, options, client, model="llama3.1:8b", confidence_type="absolute", encoding_type="uppercase"):
    """
    Let the model infer the answer based on the provided context, question, options, and confidence type.

        Args:
            :param context: (str) The context for the question.
            :param question: (str) The question to be answered.
            :param options: (list) List of answer options.
            :param client: The LLM client.
            :param model: (str) The model to use for inference. Default is "llama3.1:8b".
            :param confidence_type: (str) Type of confidence ("absolute" or "relative"). Default is "absolute".
            :param encoding_type: (str) Type of encoding ("uppercase", "lowercase" or "numeric"). Default is "uppercase".
        Returns:
            :return: (dict) The inferred answer.
        """
    prompt = construct_prompt(context, question, options, confidence_type)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            max_retries=10,
            response_model=get_template(len(options), confidence_type, encoding_type),
            temperature=0,
        )
        response = resp.model_dump()

    except (ValidationError, InstructorRetryException) as exc:
        response = {
            "answer": "NaN",
            "conf_a": np.nan,
            "conf_b": np.nan,
            "conf_c": np.nan,
            "conf_d": np.nan,
        }

    return response
