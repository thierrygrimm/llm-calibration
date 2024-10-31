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


