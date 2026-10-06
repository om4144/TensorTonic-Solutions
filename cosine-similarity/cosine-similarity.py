import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)

    mod_a = abs(np.sqrt(np.sum(np.pow(a, 2))))
    mod_b = abs(np.sqrt(np.sum(np.pow(b, 2))))

    if mod_a == 0 or mod_b == 0:
        return float(0)
        
    cos = np.dot(a, b.T) / (mod_a * mod_b)

    return float(cos)
    