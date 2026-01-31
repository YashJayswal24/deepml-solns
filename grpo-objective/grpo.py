import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    A = np.array(A)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)
    rhos = np.array(rhos)
    
    # Calculate current policy probability: pi_theta = rho * pi_theta_old
    pi_theta = pi_theta_old * rhos
    
    G = A.shape[0]
    total_loss = 0.0
    
    for i in range(G):
        # 1. PPO-style clipped surrogate objective
        surrogate = min(rhos[i] * A[i], np.clip(rhos[i], 1 - epsilon, 1 + epsilon) * A[i])
        
        # 2. KL Divergence penalty (unbiased estimator)
        # D_KL = (pi_theta / pi_ref) - log(pi_theta / pi_ref) - 1
        # Note: log(a/b) = log(a) - log(b), but here we compute the ratio directly first
        ratio = pi_theta[i] / pi_theta_ref[i]
        kl_div = ratio - np.log(ratio) - 1
        
        # Combine terms: objective - beta * KL
        total_loss += surrogate - beta * kl_div
        
    # Start: The final objective is the average
    return total_loss / G
