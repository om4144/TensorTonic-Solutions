import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """

    ans = np.power((np.array(y_pred) - np.array(y_true)), 2)
    return np.sum(ans) / len(ans)