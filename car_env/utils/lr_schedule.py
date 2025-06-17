def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: Function that computes the learning rate given the current progress.
    """

    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        :param progress_remaining: Current progress remaining (from 1 to 0).
        :return: Current learning rate.
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule
