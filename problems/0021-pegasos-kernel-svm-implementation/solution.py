import numpy as np

def linear_kernal(x: np.ndarray, y: np.ndarray):
    return np.sum(x*y)

def rbf_kernal(x: np.ndarray, y: np.ndarray, sigma):
    sq_dist = np.sum((x - y) ** 2)
    return np.exp(-sq_dist / (2 * (sigma**2)))


def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:

    (n, d) = data.shape
    alpha = np.zeros(n)
    b = 0



    for t in range(1, iterations+1):
        for i in range(0, n):
            psyt = 1/ (lambda_val * t)
            kernal_vector = np.zeros(n)
            if kernel == 'linear':
                kernal_vector = np.sum(data*data[i], axis=1)
            else:
                sq_dist = np.sum((data - data[i]) ** 2, axis=1)
                kernal_vector = np.exp(-sq_dist / (2 * (sigma**2)))

            gi = np.sum(alpha * labels * kernal_vector) + b
            if labels[i]*gi < 1:
                alpha[i] = alpha[i] + psyt*(1-lambda_val*alpha[i])
                b = b + psyt*labels[i]
            else:
                alpha[i] = alpha[i] - psyt*(lambda_val*alpha[i])


    return (alpha, b)