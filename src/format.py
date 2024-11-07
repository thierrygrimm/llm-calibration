"""
Description:
Functions for formatting and constructing prompts for answering questions.

Functions:
- format_choices: Formats a list of choices into a string with each choice prefixed by specified style.
- get_example_format: Determines the answer format example based on format_style.
- format_prompt: Formats a prompt with specified style.
- construct_prompt: Constructs a prompt for answering a question based on provided context and choices.
"""


def format_choices(choices, format_style="uppercase", separator="\n"):
    """
    Format a list of choices into a string with each choice prefixed by specified style.

    Parameters:
        choices (list of str): The choices to format.
        format_style (str): The formatting style. Options are "uppercase", "lowercase",
                            or "numeric". Defaults to "uppercase".
        separator (str): The string to separate choices. Defaults to newline.

    Returns:
        str: A formatted string of choices.

    Raises:
        ValueError: If choices is not a non-empty list or if format_style is invalid.
    """
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list.")

    if format_style not in ["uppercase", "lowercase", "numeric"]:
        raise ValueError("format_style must be 'uppercase', 'lowercase', or 'numeric'.")

    if format_style == "numeric":
        choices_text = separator.join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    elif format_style == "lowercase":
        choices_text = separator.join([f"{chr(97 + i)}) {choice}" for i, choice in enumerate(choices)])
    else:  # Default to uppercase letters
        choices_text = separator.join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])

    return choices_text


def get_example_format(format_style):
    """
    Determine the answer format example based on format_style.

    Parameters:
        format_style (str): The formatting style. Options are "uppercase", "lowercase", or "numeric".

    Returns:
        str: The answer format example.

    Raises:
        ValueError: If format_style is invalid.
    """
    formats = {"uppercase": "A, B, C, D", "lowercase": "a, b, c, d", "numeric": "1, 2, 3, 4"}

    try:
        return formats[format_style]
    except KeyError:
        raise ValueError("Invalid format style specified.")


def format_prompt(context, question, choices_text, mode="1-hop", confidence_type="absolute", format_style="uppercase"):
    """
    Format a prompt with specified style.

    Parameters:
        context (str): The context for the question.
        question (str): The question to be answered.
        choices_text (str): The formatted choices to include in the prompt.
        format_style (str): The formatting style. Options are "uppercase", "lowercase",
                            or "numeric". Defaults to "uppercase".

    Returns:
        str: The formatted prompt.

    Raises:
        ValueError: If format_style is invalid.
    """
    # Determine the answer format example based on format_style
    example_format = get_example_format(format_style)

    # Check if context is provided and not NaN
    context_part = f"Context: {context}\n" if context.strip() and context.strip().lower() != "nan" else ""

    # Base prompt components
    base_prompt = f"""Question: {question}\nOptions:\n{choices_text}\n\nFirst think about it and give a logical chain of thought.\n"""

    confidence_prompt = {
        "relative": f"""Then provide the likelihood that each answer out of ({example_format}) is correct.\nGive ONLY the probabilities between 0 and 1 for each option, no other words or explanation.""",
        "absolute": f"""Then choose the best answer by returning one letter ({example_format}) corresponding to the correct choice.\nFinally, please provide your confidence that your guess is correct. Give ONLY the probability between 0 and 1, no other words or explanation."""}

    # Construct final prompt
    prompt = f"""{context_part}{base_prompt}{confidence_prompt[confidence_type]}""".strip()
    # elif mode == "2-hop":
    #     if relative_confidence: # Relative confidence 2-hop with context
    #         prompt = f"""
    #         Context: {context}
    #         Question: {question}
    #         Options: {choices_text}
    #
    #         First think about it and give a logical chain of thought.
    #         Finally, please provide your confidence that your guess is correct. Give ONLY the probability between 0 and 1, no other words or explanation.
    #         """
    #     else: # Absolute confidence 2-hop with context
    #         prompt = f"""
    #         Context: {context}
    #         Question: {question}
    #         Options: {choices_text}
    #
    #         First think about it and give a logical chain of thought.
    #         Then choose the best answer by returning one letter ({example_format}) corresponding to the correct choice.
    #         Finally, please provide your confidence that your guess is correct. Give ONLY the probability between 0 and 1, no other words or explanation.
    #         """

    return prompt


def construct_prompt(context, question, choices, mode="1-hop", confidence_type="absolute", format_style="uppercase"):
    """
    Construct a prompt for answering a question based on provided context and choices.

    Parameters:
    context (str): The context for the question. If empty, the context will be omitted.
    question (str): The question to be answered.
    choices (list of str): A list of answer choices.
    format_style (str): The formatting style for the choices. Options are "uppercase", "lowercase", or "numeric". Defaults to "uppercase".

    Returns:
    str: The constructed prompt, formatted according to the specified context, question, and choices.
    """
    # Format the choices using the format_choices function
    choices_text = format_choices(choices, format_style)

    # Construct the prompt using the format_prompt function
    prompt = format_prompt(context, question, choices_text, mode, confidence_type, format_style)

    return prompt
