import numpy as np

def gaussian(x, mean, sd):
    coefficient = 1 / (sd * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / sd) ** 2
    return coefficient * np.exp(exponent)

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)
    
    Returns:
        Predicted class labels for X_test (shape: N_test)
    """

    N, D = X_train.shape
    values, counts = np.unique(y_train, return_counts=True)

    K = len(values)
    prior_prob = counts / N  
    
    
    mean = np.empty((K, D))
    sd = np.empty((K, D))


    for k in range(K):
        rows = X_train[y_train == values[k]]
        mean[k] = np.mean(rows, axis=0)
        sd[k] = np.std(rows, axis=0) + 1e-9 

    N_test, d = X_test.shape
    y_test = np.empty(N_test, dtype=values.dtype)


    for i in range(N_test):
        xtest = X_test[i]
        yi = prior_prob.copy()
        for k in range(K):
            yi[k] = yi[k] * np.prod(gaussian(xtest, mean[k], sd[k]))
        y_test[i] = values[np.argmax(yi)]


    return y_test
    
