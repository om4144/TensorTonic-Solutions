import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    n = len(y)
    loss = - (np.power((np.ones(n) - np.array(p)), gamma) * np.array(y) * np.log(p)) - (np.power(p, gamma) * (np.ones(n) - np.array(y)) * np.log(np.ones(n) - p)) 

    return float(np.sum(loss) / n)