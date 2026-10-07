import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    # Write code here
    def cosine_similarity(x, y):
        mod_x = 0
        mod_y = 0

        prod = 0
        
        for a, b in zip(x, y):
            mod_x += math.pow(a, 2)
            mod_y += math.pow(b, 2)

            prod += a * b

        return prod / (math.sqrt(mod_x) * math.sqrt(mod_y))
            
    if label == 1:
        loss = 1 - cosine_similarity(x1, x2)
        
    else:
        loss = max(0, cosine_similarity(x1, x2) - margin)

    return float(loss)