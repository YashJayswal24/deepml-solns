import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image as described in the AlexNet paper.
    
    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components
    
    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    H, W, _ = image.shape
    # Reshape image to (H*W, 3) where each row is a pixel
    img = image.reshape((-1, 3))
    
    # Center the data by subtracting the mean
    mu = np.mean(img, axis=0, keepdims=True)
    A = img - mu
    
    # Compute 3x3 covariance matrix
    cov = A.T @ A / (H * W - 1)
    
    # Compute eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(cov)
    
    # Compute the noise vector: p_i * (alpha_i * lambda_i)
    # alpha is random variable (input), lambda is sqrt(eigenvalue)
    # Note: AlexNet paper adds multiples of found principal components, 
    # with magnitudes proportional to corresponding eigenvalues times a random variable.
    noise = eig_vec @ np.multiply(alpha, np.sqrt(eig_val))
    
    # Add noise to original image
    aug_img = image + noise
    
    return np.clip(aug_img, 0, 255)
