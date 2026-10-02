import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    e = np.array(y_true) - np.array(y_pred)

    ans = np.where(
        abs(e) <= delta, 
        np.power(e, 2) / 2, 
        delta * (abs(e) - (delta / 2))
    )
  
    return float(np.sum(ans) / len(e))