import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    loss = []
    for y, p in zip(y_true, y_pred):
        p_hat = min((1 - eps), max(eps, p))
        
        loss.append(
            -(y * math.log(p_hat) +((1 - y) * math.log(1 - p_hat)))
        )
        
    
    return loss