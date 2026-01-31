import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # 1. Compute A^T * A. This symmetric matrix has eigenvalues s^2.
    A_processed = A.T @ A
    
    # 2. Find eigenvalues and eigenvectors of A^T * A
    eig_val, V = np.linalg.eig(A_processed)
    
    # 3. Sort descending (standard SVD convention)
    sort_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_idx]
    V = V[:, sort_idx]
    
    # 4. Singular values are sqrt(eigenvalues)
    sigma = np.sqrt(eig_val)
    
    # 5. Compute U using the relation A * v_i = s_i * u_i  => u_i = (A * v_i) / s_i
    # Note: If sigma is small/zero, this is numerically unstable, but for this problem
    # we assume non-zero singular values or standard cases.
    U = A @ V / sigma
    
    return U, sigma, V.T
