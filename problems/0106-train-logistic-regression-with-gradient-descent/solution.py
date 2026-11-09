import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	
	m, n = X.shape
	bias_col = np.ones(m)
	X = np.column_stack((bias_col, X))
	m, n = X.shape

	theta = np.zeros(n)

	losses = []

	for i in range(iterations):
		Yh = X @ theta
		Yh = 1 / (1 + np.exp(-Yh))
		loss = -(np.dot(y, np.log(Yh)) + np.dot(1-y, np.log(1-Yh)))
		losses.append(loss)
		gradient = ((Yh - y).T @ X)*learning_rate
		theta = theta - gradient


	return (theta, losses)