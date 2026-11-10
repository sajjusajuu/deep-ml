import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:

    Yh = X @ w
    n, d = X.shape
    loss = np.sum(np.power(y_true - Yh, 2))/n + alpha*(np.sum(np.power(w, 2)))
    return loss

