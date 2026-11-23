import numpy as np


def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:

    (n, d) = data.shape
    alpha = np.zeros(n)
    b = 0

    kernel_mat = np.zeros((n,n))
    for i in range(0,n):
        if kernel == 'linear':
            kernel_mat[i] = np.sum(data*data[i], axis=1)
        else:
            sq_dist = np.sum((data - data[i]) ** 2, axis=1)
            kernel_mat[i] = np.exp(-sq_dist / (2 * (sigma**2)))



    for t in range(1, iterations+1):
        for i in range(0, n):
            psyt = 1/ (lambda_val * t)
           
            gi = np.sum(alpha * labels * kernel_mat[i]) + b
            if labels[i]*gi < 1:
                alpha[i] = alpha[i] + psyt*(1-lambda_val*alpha[i])
                b = b + psyt*labels[i]
            else:
                alpha[i] = alpha[i] - psyt*(lambda_val*alpha[i])


    return (alpha, b)