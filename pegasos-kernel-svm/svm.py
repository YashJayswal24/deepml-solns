import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    n_samples, n_features = data.shape
    alphas = np.zeros(n_samples)
    bias = 0.0
    
    def linear_kernel(x1, x2):
        return np.sum(x1 * x2, axis=1)
    
    def rbf_kernel(x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1 - x2, axis=1) ** 2 / (2 * sigma ** 2)).reshape(-1)
    
    kernel_fun = linear_kernel if kernel == 'linear' else lambda x1, x2: rbf_kernel(x1, x2, sigma)
    
    for t in range(1, iterations + 1):
        lr = 1 / (lambda_val * t)
        for i in range(n_samples):
            pred = np.sum(alphas * labels * kernel_fun(data[i].reshape((1, -1)), data)) + bias
            if labels[i] * pred < 1:
                alphas[i] = (1 - 1/t) * alphas[i] + lr * labels[i]
                bias += lr * labels[i]
    
    return alphas.tolist(), bias

# Test Case 1
if __name__ == "__main__":
    alphas, b = pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
        np.array([1, 1, -1, -1]), 
        kernel='linear', 
        lambda_val=0.01, 
        iterations=100
    )
    print("Test Case 1 (Linear Kernel):")
    print(([round(a, 4) for a in alphas], round(b, 4)))
    # Expected: ([100.0, 0.0, -100.0, -100.0], -937.4755)

    # Test Case 2
    alphas, b = pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
        np.array([1, 1, -1, -1]), 
        kernel='rbf', 
        lambda_val=0.01, 
        iterations=100, 
        sigma=0.5
    )
    print("\nTest Case 2 (RBF Kernel):")
    print(([round(a, 4) for a in alphas], round(b, 4)))
    # Expected: ([100.0, 99.0, -100.0, -100.0], -115.0)
