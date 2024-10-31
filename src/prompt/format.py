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
        choices_text = separator.join(
            [f"{i + 1}. {choice}" for i, choice in enumerate(choices)]
        )
    elif format_style == "lowercase":
        choices_text = separator.join(
            [f"{chr(97 + i)}) {choice}" for i, choice in enumerate(choices)]
        )
    else:  # Default to uppercase letters
        choices_text = separator.join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )

    return choices_text
