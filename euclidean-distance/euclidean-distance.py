import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)

    dis = np.sqrt(np.sum(np.pow(x - y, 2)))

    return float(dis)