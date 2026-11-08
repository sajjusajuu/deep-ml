# Cell 1
import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    # y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros(n)  # Initialize weights to zeros
    
    for i in range(iterations):
        H = X @ theta
        # print("H", H)
        # print("y", y)
        gradient = ((H - y).T @ X)*alpha / m
        # print("gradient", gradient)
        theta = theta - gradient
        # print("theta", theta)
    

    return theta.flatten()