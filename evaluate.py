import numpy as np

def frobenius_loss(H_true: np.ndarray, H_pred: np.ndarray) -> float:
    """
    Frobenius norm loss: ||H_true - H_pred||_F^2
    """
    diff = H_true - H_pred
    return np.linalg.norm(diff, ord="fro")**2


def stein_loss(H_true, H_pred, eps=1e-6):
    n = H_true.shape[0]
    
    H_true = (H_true + H_true.T) * 0.5
    H_pred = (H_pred + H_pred.T) * 0.5
    H_true_reg = H_true + eps * np.eye(n)

    A = np.linalg.solve(H_true_reg, H_pred)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return np.inf

    return float(np.trace(A) - logdet - n)


def stein_loss_scaled(H_true, H_pred, eps=1e-6):
    n = H_true.shape[0]
    H_true = (H_true + H_true.T) / 2
    H_pred = (H_pred + H_pred.T) / 2
    H_true_reg = H_true + eps * np.eye(n)

    # optimal scaling
    trPTinv = np.trace(H_pred @ np.linalg.inv(H_true_reg))
    alpha = n / (trPTinv + 1e-12)
    H_pred_scaled = alpha * H_pred
    
    return stein_loss(H_true, H_pred_scaled, eps)



def correlation_loss(H_true: np.ndarray, H_pred: np.ndarray, fisher_z: bool = False) -> float:
    """
    Frobenius loss between correlation matrices.
    Optionally applies Fisher-z transform for scale adjustment.
    """
    # convert to correlation matrices
    D_true = np.sqrt(np.diag(H_true))
    D_pred = np.sqrt(np.diag(H_pred))
    
    R_true = H_true / np.outer(D_true, D_true)
    R_pred = H_pred / np.outer(D_pred, D_pred)

    if fisher_z:
        # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
        R_true = np.arctanh(np.clip(R_true, -0.999999, 0.999999))
        R_pred = np.arctanh(np.clip(R_pred, -0.999999, 0.999999))

    diff = R_true - R_pred
    return np.linalg.norm(diff, ord="fro")**2

def portfolio_aligned_loss(H_pred, H_true, weights):
    return ((weights.T @ H_pred @ weights) - (weights.T @ H_true @ weights))**2

