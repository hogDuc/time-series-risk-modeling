import torch
import random
import math
import numpy as np
from scipy.optimize import minimize

def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def NLL(returns, covariance):
    '''
    Negative log-likelihood function, use for evaluation
    Params:
        returns: [batch, T-1, n_assets]
        covariance: [batch, T-1, n_assets, n_assets]
    Returns scalar loss
    logL = -1/2 * (log(H_t.abs()) + r_t.T * H_t^(-1) * r_t)
    '''

    batch_size, T, N = returns.shape
    eps = 1e-3 # To avoid singular matrices
    # nll = torch.tensor(0.0, device=returns.device)
    nll_list = []

    for t in range(T): # Elevate the likelihood at time T
        r_t = returns[:, t, :].unsqueeze(2)
        H_t = covariance[:, t, :, :]

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(H_t + eps * torch.eye(N).to(H_t.device))
        except:
            print(f"Cholesky failed at t = {t}")
            continue

        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)

        # Solve
        Linv_r = torch.cholesky_solve(r_t, L)
        quad_form = torch.bmm(r_t.transpose(1, 2), Linv_r).squeeze()

        # Final NLL for time t
        nll_t = 0.5 * (log_det + quad_form + N * math.log(2 * math.pi))
        nll_list.append(nll_t)
        if len(nll_list) == 0:
            raise RuntimeError("All Cholesky decompositions failed - check covariance")

    return torch.mean(torch.stack(nll_list))

def split_train_test(df, train_ratio=0.8):
    '''
    Params:
        df: input df
        train_ratio: ratio to split the train data
    Output:
        Train set and Test set
    '''
    T, _ = df.shape
    train_T = int(T * train_ratio)
    return df.iloc[:train_T, :], df.iloc[train_T - 1:, :]

def minimum_variance_portfolio(covariance_matrix, data):

    n_assets = covariance_matrix.shape[1]
    
    def portfolio_variance(weights, covariance_matrix=covariance_matrix):
        return (weights @ covariance_matrix @ weights.T)
    
    def weights_constraints(weights):
        return np.sum(weights) - 1

    bounds = [(0, 1) for x in range(n_assets)] # Bounds for weights
    init_weights = [(1/n_assets) for x in range(n_assets)] # Initial weights, equal weighted

    optimal_weights = minimize(
        fun=portfolio_variance,
        x0=init_weights,
        bounds=bounds,
        constraints={
            "type":'eq',
            "fun":weights_constraints,
            'method':"SLSQP"
        }
    )

    return optimal_weights.x, {
        str(col): float(optimal_weights.x[i]) for i, col in enumerate(data.columns)
    }
