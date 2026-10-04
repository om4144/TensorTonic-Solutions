import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if type(x) != list:
        return 1 / (1 + np.exp(-x))
        
    n = len(x)
    x = np.array(x)

    ones = np.ones(n)
    exp = np.exp(-x)
    
    return ones / (ones + exp)