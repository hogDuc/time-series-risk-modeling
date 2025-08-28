import torch.nn as nn
import torch
from torch.optim import RMSprop
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import pandas as pd
import math
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class LSTM_BEKK_MODEL:

    def to_tensor(x, device):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=torch.float64)
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float64)
        raise TypeError("Unsupported array type")

    def lower_triangle_index(n_assets: int):
        '''
        Get index of lower triangular matrix
        '''
        rows, cols = torch.tril_indices(n_assets, n_assets)
        diag_mask = rows == cols
        off_mask = ~diag_mask
        return rows, cols, diag_mask, off_mask

    def vec_to_lower_tri(vec: torch.Tensor, n_assets: int) -> torch.Tensor:
        """
        Map vector of length n_assets(n_assets+1)/2 to a lower-triangular matrix
        Args:
            vec: shape (..., n_assets(n_assets+1)/2)
        Output:
            shapeL (..., n_assets,n_assets)
        """

        vec_length = n_assets * (n_assets + 1) // 2
        assert vec.shape[-1] == vec_length, f"Expected vector length {vec_length} for number of assets = {n_assets}, got {vec.shape[-1]}"
        rows, cols, _, _ = LSTM_BEKK_MODEL.lower_triangle_index(n_assets)
        out = vec.new_zeros(*vec.shape[:-1], n_assets, n_assets) # Initialize
        out[..., rows, cols] = vec

        return out

    class Static_C(nn.Module):
        '''
        Parameterization of the static lower-triangular C with positive diagonal to ensure semipositive definitenes of C'C
        '''
        def __init__(self, n_assets: int):
            super().__init__()
            self.n_assets = n_assets
            self.off_idx = torch.tril_indices(n_assets, n_assets, offset=-1)
            num_off = self.off_idx.shape[1]
            
            self.off_params = nn.Parameter(
                torch.zeros(num_off, dtype=torch.float64)
            )
            self.diag_params = nn.Parameter(
                torch.zeros(n_assets, dtype=torch.float64)
            )
        
        def forward(self) -> torch.Tensor:
            C = self.off_params.new_zeros(self.n_assets, self.n_assets)
            # Off diagonals parameteres
            C[self.off_idx[0], self.off_idx[1]] = self.off_params
            # Diagonal parameteres
            C[range(self.n_assets), range(self.n_assets)] = torch.nn.functional.softplus(self.diag_params) + 1e-6

            return C

    class LSTM_Dynamic_C(nn.Module):
        """
        Map past returns to dynamic element C_t, map LSTM output to lower triangular matrix C_t, and regularize the diagonal with a Swish function Swish(x) = x*sigmopoid(Beta*x), with learnable x.
        The Swish function helps stability without forcing diagonals strictly positive
        """

        def __init__(self, n_assets: int, hidden_size: Optional[int] = None, num_layers: int = 1, dropout:float = 0.1):
            super().__init__()
            self.n_assets = n_assets
            self.length = n_assets * (n_assets + 1) // 2
            self.hidden_size = hidden_size or max(8, n_assets) # At least equal to number of assets
            self.num_layers = num_layers

            # LSTM that takes in r_{t-1} and outputs a vector length = self.length at each step
            self.lstm = nn.LSTM(
                input_size=n_assets,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                dtype=torch.float64
            )

            self.out = nn.Linear(
                self.hidden_size, 
                self.length, 
                dtype=torch.float64
            )
            self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float64)) # Swish parameters beta, is learnable

        def swish(self, x: torch.Tensor) -> torch.Tensor:
            # Swish(x) = x * sigmoid(beta * x)
            return x * torch.sigmoid(self.beta * x)

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            """
            Args:
                returns: TxM returns
            Outputs:
                C_t for t = 1,... T shape (T, M, M)
            Convention: at step t we feed r_{t-1}, for t=0 we won't form C_0
            """

            n_periods, n_assets = returns.shape
            assert n_assets == self.n_assets

            # FIX: build inputs = [0, r_0, r_1, ..., r_{T-2}] so that output at index t uses r_{t-1}
            zeros = torch.zeros(1, n_assets, dtype=returns.dtype, device=returns.device)
            inputs = torch.cat([zeros, returns[:-1, :]], dim=0)  # shape (T, M)

            x = inputs.unsqueeze(0) # Add new dimension at the beginning to (1, T, M) to be suitable with torch
            h, _ = self.lstm(x) # (1, T, H)
            z = self.out(h) # (1, T, L)
            z = z.squeeze(0) # (T, L)
            C_full = LSTM_BEKK_MODEL.vec_to_lower_tri(z, self.n_assets)

            # Apply Swish to only diagonal elements
            rows, cols, diag_mask, off_mask = LSTM_BEKK_MODEL.lower_triangle_index(self.n_assets)
            diag_vals = C_full[..., rows[diag_mask], cols[diag_mask]] # Get all diagonal values
            diag_vals = self.swish(diag_vals) # Apply Swish
            C_full[..., rows[diag_mask], cols[diag_mask]] = diag_vals # Assign Swish-applied values to the lower-triangular matrix C

            return C_full # C_t
        
    class params(nn.Module):
        """
        Static scalars a, b with constraints: a, b >= 0; a + b < 1
        Use positive reparameterization via softplus -> normalize:
            u = softplus(u0), v = softplus(v0); s = u + v + 1; a = u/s, b = v/s
        Which makes a, b in (0,1) and a+b <1
        """

        def __init__(self):
            super().__init__()
            self.u0 = nn.Parameter(
                torch.tensor(0.2, dtype=torch.float64)
            )
            self.v0 = nn.Parameter(
                torch.tensor(0.7, dtype=torch.float64)
            )
        
        def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
            u = torch.nn.functional.softplus(self.u0)
            v = torch.nn.functional.softplus(self.v0)
            s = u + v + 1.0
            a = u/s
            b = v/s
            return a, b
        

    @dataclass
    class LSTM_BEKK_config:
        hidden_size: Optional[int] = None
        num_layers: int = 1
        dropout: float = 0.1
        lr: float = 0.001
        weight_decay: float = 0.0
        epochs: int = 500
        grad_clip: float = 10.0
        val_split: float = 0.1
        early_stopping_patience: int = 20
        device: str = "cpu"
        jitter: float = 1e-6 # for Cholesky stability
        seed: int = 1

    class LSTM_BEKK(nn.Module):
        """
        H_t = C*C' + C_t*C_t' + a*r_{t-1}*r_{t-1}' + b*H_{t-1}
        with Gaussian log-likelihood
        """

        def __init__(self, n_assets: int, config: Optional["LSTM_BEKK_MODEL.LSTM_BEKK_config"]=None):
            super().__init__()
            self.n_assets = n_assets
            self.config = config or LSTM_BEKK_MODEL.LSTM_BEKK_config()
            torch.manual_seed(self.config.seed) # Set seed

            self.C = LSTM_BEKK_MODEL.Static_C(n_assets)
            self.C_dynamic = LSTM_BEKK_MODEL.LSTM_Dynamic_C(
                n_assets=n_assets, 
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
            self.ab = LSTM_BEKK_MODEL.params()

            # Buffers created at fit-time 
            # NOTE: wtf is this ==================================================================
            self.H0_: Optional[torch.Tensor] = None
            self.mu_: Optional[torch.Tensor] = None
            
            self.to(
                dtype=torch.float64,
                device=self.config.device
            )
        
        def forward_sequence(
                self,
                returns: torch.Tensor,
                init_cov_matrix: Optional[torch.Tensor]=None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Implement the recursion for covariance matrix H_t. Compute covariance matrix H_t and per-step negative log-likelihood terms, given returns TxM
            Output:
                H: (T, M, M)
                nll_terms: (T,) with 0.5 * (logdet(cov_matrix) + r_t' * cov_matrix **(-1)*r_t)
            """
            config = self.config
            device = config.device
            n_periods, n_assets = returns.shape
            assert n_assets == self.n_assets

            # Static C and static scalar a, b
            C = self.C()
            a, b = self.ab()

            # Dynamic C_t
            C_t_all = self.C_dynamic(returns)

            # Initialize H_0
            if init_cov_matrix is None:
                cov = torch.cov(returns.T)
                cov = cov + config.jitter * torch.eye(
                    n_assets,
                    dtype=torch.float64,
                    device=device
                )
                prev_cov_matrix = cov
            else:
                prev_cov_matrix = init_cov_matrix.to(device=device, dtype=torch.float64)
            
            cov_matrix_list = []
            nll_terms = []

            eye_assets = torch.eye(n_assets, dtype=torch.float64, device=device)

            for t in range(n_periods):
                prev_returns = returns[t-1].unsqueeze(0).T if t>0 else torch.zeros(n_assets, 1, dtype=torch.float64, device=device)
                C_t = C_t_all[t]

                C_static = C @ C.T
                C_dynamic = C_t @ C_t.T
                arch = a * (prev_returns @ prev_returns.T)
                garch = b * prev_cov_matrix

                cov_matrix = C_static + C_dynamic + arch + garch

                # Stabilize for Cholesky
                # NOTE wtf is this? -==========================
                cov_matrix = cov_matrix + config.jitter * eye_assets

                # Cholesky-based log-likelihood
                L = torch.linalg.cholesky(cov_matrix)
                logdet = 2.0 * torch.log(torch.diag(L)).sum()

                # Solve H^{-1}*r via 2 triangular solves
                returns_t = returns[t].unsqueeze(0).T
                y = torch.cholesky_solve(returns_t, L) # Solves H*y=r
                quad = (returns_t.T @ y).squeeze() # r' * H^{t-1} * r

                nll_t = 0.5 * (logdet + quad)
                nll_terms.append(nll_t)
                cov_matrix_list.append(cov_matrix)

                prev_cov_matrix = cov_matrix
            
            cov_matrices = torch.stack(cov_matrix_list, dim=0)
            nll_terms = torch.stack(nll_terms, dim=0)
            

            return cov_matrices, nll_terms, C_t_all
        
        def negative_loglik(
                self, 
                returns: torch.Tensor,
                lambda_reg: float = 0.0,
                tau: Optional[float] = None
            ) -> torch.Tensor:

            _, nll_terms, C_t_all = self.forward_sequence(returns, init_cov_matrix=self.H0_)
            nll = nll_terms.sum()

            if lambda_reg > 0.0 and tau is not None:
                reg_terms = []
                for C_t in C_t_all:
                    C_tC_t_trace = torch.trace(C_t @ C_t.T)
                    reg_terms.append(torch.relu(C_tC_t_trace - tau)**2)
                reg_penalty = torch.stack(reg_terms).mean()
                nll = nll + lambda_reg * reg_penalty
            
            return nll

        
        def fit(
                self,
                returns_df: pd.DataFrame,
                verbose: bool = True
        ) -> Dict[str, float]:
            """
            Fit the model by minimizing the Gaussian negative log-likelihood
            returns_df: TxM of demeaned returns
            """

            config = self.config
            device = config.device

            returns_np = returns_df.to_numpy(dtype=float)
            returns_tensor = LSTM_BEKK_MODEL.to_tensor(returns_np, device=device)
            n_periods = returns_tensor.shape[0]

            # split train, valuate set
            t_valuate = max(1, int(math.floor(config.val_split * n_periods)))
            t_train = n_periods - t_valuate
            r_train = returns_tensor[:t_train]
            r_valuate = returns_tensor[t_train:]

            # set H_0 from training sample
            cov = torch.cov(r_train.T)
            cov = cov + config.jitter * torch.eye(
                self.n_assets,
                dtype=torch.float64,
                device=device
            )
            # self.init_cov_matrix = cov.detach()
            self.H0_ = cov.detach()

            params = list(self.parameters())
            opt = RMSprop(params, lr=config.lr, weight_decay=config.weight_decay)
            best_val = float("inf")
            best_state = None
            patience = config.early_stopping_patience
            epochs_no_improve = 0

            for epoch in range(config.epochs):
                self.train()
                opt.zero_grad(set_to_none=True)
                if epoch == 0:
                    tau = torch.trace(torch.cov(r_train.T)).item()
                nll = self.negative_loglik(r_train, lambda_reg=1e-3, tau=tau)
                nll.backward()
                if config.grad_clip is not None and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_clip)
                opt.step()

                # Evaluate on validation
                self.eval()
                with torch.no_grad():
                    val_nll = self.negative_loglik(r_valuate).item()

                if verbose and (epoch % 10 == 0 or epoch == config.epochs-1):
                    print(f"[{epoch:04d}] train NLL : {nll.item():.3f} | val NLL : {val_nll:.3f}")

                # Early stopping
                if val_nll + 1e-9 < best_val:
                    best_val = val_nll
                    best_state = {
                        k: v.detach().clone() for k, v in self.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}, best val NLL: {best_val:.3f}")
                        break
            
            # Restore best
            if best_state is not None:
                self.load_state_dict(best_state)

            return {"best_val_nll":best_val}

        @torch.no_grad()
        def covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
            """
            Compute fitted covariance matrix for the full series
            Args:
                returns_df: demeaned returns
            Output:
                array of shape (n_periods, n_assets, n_assets)
            """
            returns_np = returns_df.to_numpy(dtype=float)
            returns_tensor = LSTM_BEKK_MODEL.to_tensor(returns_np, self.config.device)
            returns_tensor = LSTM_BEKK_MODEL.to_tensor(returns_np, self.config.device)
            cov_matrix, _, _ = self.forward_sequence(returns_tensor, init_cov_matrix=self.H0_)
            return cov_matrix.cpu().numpy()
        
        def get_params(self) -> Dict[str, torch.Tensor]:
            a, b = self.ab()
            return {
                "C_static":self.C().detach().cpu(),
                "a": a.detach().cpu(),
                "b": b.detach().cpu(),
                "beta_swish":self.C_dynamic.beta.detach().cpu()
            }

        @torch.no_grad()
        def forecast_one_step(
            self,
            last_returns: np.ndarray,
            last_cov: np.ndarray
        ) -> np.ndarray:
            """
            One-step ahead forecast with r_T and H_T
            Args: 
                last_returns: shape (n_assets, )
                last_cov: shape (n_assets, n_assets) (H_T)
            """
            device = self.config.device
            returns = LSTM_BEKK_MODEL.to_tensor(last_returns.reshape(1, -1), device)
            C = self.C()
            a, b = self.ab()
            C_dynamic = self.C_dynamic(returns)[0]

            cov_matrix_forecast = C @ C.T + C_dynamic @ C_dynamic.T + a * (returns @ returns.T) + b * LSTM_BEKK_MODEL.to_tensor(last_cov, device)
            cov_matrix_forecast = cov_matrix_forecast + self.config.jitter * torch.eye(self.n_assets, dtype=torch.float64, device=device)

            return cov_matrix_forecast.cpu().numpy()
        
        @torch.no_grad()
        def forecast_multi_step(
            self,
            last_returns: np.ndarray,
            last_cov: np.ndarray,
            steps: int = 20,
            method: str = "zero"
        ) -> np.ndarray:
            """
            Args:
                method:
                - "zero": feed zeros for future returns
                - "simulate": simulate paths of future returns
            """
            device = self.config.device
            n_assets = self.n_assets
            forecasts = []

            r_curr = LSTM_BEKK_MODEL.to_tensor(last_returns.reshape(1, -1), device)
            H_curr = LSTM_BEKK_MODEL.to_tensor(last_cov, device)

            for step in range(steps):
                C = self.C()
                a, b = self.ab()
                
                # Dynamic C
                C_t = self.C_dynamic(r_curr)[0]

                if method == "zero" and step > 0:
                    arch = torch.zeros(
                        n_assets, n_assets, dtype=torch.float64, device=device
                    )
                else:
                    arch = a * (r_curr.T @ r_curr)

                H_next = C @ C.T + C_t @ C_t.T + arch + b * H_curr
                H_next = H_next + self.config.jitter * torch.eye(n_assets, dtype=torch.float64, device=device)

                forecasts.append(H_next.cpu().numpy())

                # Update for next iteration
                H_curr = H_next
                if method == "zero":
                    r_curr = torch.zeros_like(r_curr) # feed zeros
                elif method == "simulate":
                    # sample one return path
                    L = torch.linalg.cholesky(H_next)
                    z = torch.randn(n_assets, 1, dtype=torch.float64, device=device)
                    r_curr = (L @ z).T 
            
            return np.stack(forecasts, axis=0)
        
    def fit_lstm_bekk(
            returns_df: pd.DataFrame,
            hidden_size: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.1,
            lr: float = 0.001,
            epochs: int = 500,
            device: str = "cpu"
    ) -> LSTM_BEKK:
        n_assets = returns_df.shape[1]
        config = LSTM_BEKK_MODEL.LSTM_BEKK_config(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            device=device
        )
        model = LSTM_BEKK_MODEL.LSTM_BEKK(n_assets=n_assets, config=config)
        model.fit(returns_df, verbose=True)

        return model
    
    @staticmethod
    def load_model(
        path: str, 
        n_assets: int, 
        config: Optional["LSTM_BEKK_MODEL.LSTM_BEKK_config"]=None
    ):
        model = LSTM_BEKK_MODEL.LSTM_BEKK(n_assets=n_assets, config=config)
        state_dict = torch.load(path, map_location=config.device if config else "cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    
class BEKK_GARCH_MODEL:
    def vech_to_matrix(params, n_assets):
        '''
        Convert vech C params to lower-triangular matrix C
        '''

        C = np.zeros((n_assets, n_assets))
        index = np.tril_indices(n_assets) # index of lower-triangular part of the matrix
        C[index] = params[:len(index[0])]
        
        return C

    def unpack_params(params, n_assets):
        """
        Unpack parameters to C, A, B
        """
        nC = n_assets * (n_assets + 1) // 2
        C = BEKK_GARCH_MODEL.vech_to_matrix(params[:nC], n_assets)
        A = params[nC:nC+n_assets*n_assets].reshape(n_assets, n_assets)
        B = params[nC+n_assets*n_assets:].reshape(n_assets, n_assets)
        
        return C, A, B

    def bekk_loglikelihood(params, returns):
        n_periods, n_assets = returns.shape
        C, A, B = BEKK_GARCH_MODEL.unpack_params(params, n_assets)
        cov_matrix = np.cov(returns.T) # initialize sample covariance matrix
        loglikelihood = 0

        for t in range(n_periods):
            residual = returns[t].reshape(-1, 1)
            cov_matrix = C@C.T + A@(residual@residual.T)@A.T + B@cov_matrix@B.T
            sign, logdet = np.linalg.slogdet(cov_matrix)

            if sign <= 0:
                return 1e6 # Ensure positive definiteness
            loglikelihood += 0.5 * (n_assets*np.log(2*np.pi) + logdet + residual.T@np.linalg.inv(cov_matrix)@residual)
        
        return loglikelihood.flatten()[0]

    def fit_bekk(returns):
        n_periods, n_assets = returns.shape
        nC = n_assets * (n_assets+1) // 2
        n_params = nC + 2*n_assets*n_assets
        x0 = 0.05 * np.random.randn(n_params) # initialize first value

        bekk = minimize(
            BEKK_GARCH_MODEL.bekk_loglikelihood, x0,
            args=(returns,),
            method="L-BFGS-B",
            options={"maxiter":500}
        )

        C, A, B = BEKK_GARCH_MODEL.unpack_params(bekk.x, n_assets)

        return C, A, B, bekk

    def bekk_forecast(C, A, B, returns, horizon=1):
        n_periods, n_assets = returns.shape
        last_cov_matrix = np.cov(returns.T)
        residual = returns[-1].reshape(-1,1)
        forecasts = []

        cov_matrix_f1 = C @ C.T + A @ (residual @ residual.T) @ A.T + B @ last_cov_matrix @ B.T
        forecasts.append(cov_matrix_f1) # Use actual shocks

        prev_cov_matrix = cov_matrix_f1.copy()
        for t in range(2, horizon+1):
            cov_matrix_t = C @ C.T + (A @ A.T + B @ B.T) @ prev_cov_matrix
            forecasts.append(cov_matrix_t)
            prev_cov_matrix = cov_matrix_t
        
        return forecasts
    
    def bekk_fitted_covariances(params, returns):
        """
        Compute fitted conditional covariance matrices H_t for the full series
        """
        n_periods, n_assets = returns.shape
        C, A, B = BEKK_GARCH_MODEL.unpack_params(params, n_assets)

        cov_matrices = np.zeros((n_periods, n_assets, n_assets))
        cov_matrix = np.cov(returns.T)  # initialize with sample covariance

        for t in range(n_periods):
            residual = returns[t].reshape(-1, 1)
            cov_matrix = C @ C.T + A @ (residual @ residual.T) @ A.T + B @ cov_matrix @ B.T
            cov_matrices[t] = cov_matrix

        return cov_matrices

    

class DCC_GARCH_MODEL:
    def negative_log_likelihood(params, returns):
        """
        Negative log-likelihood (Gaussian QML) for a single return series. Used to find the optimize parameters
        """

        # omega, alpha, beta = params
        # Try fixing omega to prevent covariance exploding
        _, alpha, beta = params
        omega = np.var(returns)


        if (omega <= 0) or (alpha < 0) or (beta < 0) or (alpha + beta >= 0.9999): # Check condition omega > 0, alpha, beta > 0 and alpha + beta < 1
            return np.inf 
        
        n_period = returns.size
        variances = np.empty(n_period) # Array of variance

        variance_0 = np.var(returns) if np.var(returns) > 1e-12 else 1.0 # To ensure positive definiteness

        variances[0] = variance_0 # Use sample variance as the first variance

        for t in range(1, n_period):
            # Diagonal matrix of variances
            variances[t] = omega + alpha*returns[t-1]**2 + beta*variances[t-1] # Univariate GARCH
            if not np.isfinite(variances[t]) or variances[t] <= 1e-16:
                return np.inf # Ensure positive definiteness
            
        log_likelihood = -0.5 * (np.log(2*np.pi) + np.log(variances) + (returns**2)/variances)

        return -np.sum(log_likelihood)


    def univariate_garch(returns: np.ndarray, x0=(1e-6, 0.05, 0.9)):
        """
        Fit Univariate GARCH
        Args:
            returns: np.array of return
            x0: inital parameters
        Return:
            DCC input parameters
            "omega":omega,
            "alpha":alpha,
            "beta":beta,
            "variances": variances,
            "residuals":resid_standardized,
            "success":ugarch.success
        """

        returns = np.asanyarray(returns).astype(float)
        returns = returns - np.mean(returns) # Demean return
        
        bounds = [
            (1e-12, None), # Must be positive
            (0.0, 1.0), # Can be semipositive
            (0.0, 1.0) # Can be semipositive
        ]
        constraints = (
            {
                "type":'ineq',
                "fun": lambda p: 0.999 - (p[1] + p[2]) # Ensure alpha + beta < 1
            },
        )
        
        ugarch = minimize(
            DCC_GARCH_MODEL.negative_log_likelihood, x0,
            args=(returns,),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP"
        )

        omega, alpha, beta = ugarch.x

        # Conditional variances and standardized residuals
        n_period = returns.size
        variances = np.empty(n_period) # Initialize diagonal matrix of variances
        variances[0] = np.var(returns) if np.var(returns) > 1e-12 else 1.0 # Assign first variance
        for t in range(1, n_period):    
            # Univariate GARCH
            variances[t] = omega + alpha*returns[t-1]**2 + beta*variances[t-1]
        
        resid_standardized = returns / np.sqrt(np.clip(variances, 1e-12, None)) # Standardize residuals

        return {
            "omega":omega,
            "alpha":alpha,
            "beta":beta,
            "variances": variances,
            "residuals":resid_standardized,
            "success":ugarch.success
        }


    def fit_univariate_garch(df:pd.DataFrame):
        """
        Fit GARCH(1, 1) for each stock column
        Args:
            df: pd.DataFrame. Should be cleaned off NA
        """

        n_period, n_asset = df.shape # Get period length and number of asset
        variance_mtrx = np.zeros((n_period, n_asset)) # Initalize matrix of conditional variances for each asset
        residual_mtrx = np.zeros((n_period, n_asset)) # Initialize matrix of standardized residuals for each assset
        params = {}

        # Fit Univariate GARCH to each stock returns
        for i, col in enumerate(df.columns): 
            ugarch = DCC_GARCH_MODEL.univariate_garch(df[col].values)
            params[col] = {
                key:ugarch[key] for key in ["omega", "alpha", "beta", "success"]
            }
            variance_mtrx[:, i] = ugarch["variances"]
            residual_mtrx[:, i] = ugarch["residuals"]

        return variance_mtrx, residual_mtrx, params # D_t^2, eps_t, params

    # DCC estimation
    def dcc_NLL(params, residuals):
        '''
        Return negative correlation log-likelihood for DCC(1,1). Residuals is a matrix of TxM
        ''' 
        alpha, beta = params
        if (alpha < 0) or (beta < 0) or (alpha + beta >= 0.9999): # conditions
            return np.inf
        
        n_period, n_asset = residuals.shape
        # Unconditional correlation of residuals
        S = np.corrcoef(residuals.T) # Initialize correlation matrix between assets return residuals aka unconditional correlation matrix of the standardized residuals

        # Initialize Q with S
        Q = S.copy()
        NLL = 0.0
        for t in range(n_period):
            # Update Q_t (if t>=1 use residual[t-1], if t=0, use previous Q)
            if t > 0:
                prev_resid = residuals[t-1:t, :].T # Matrix M x 1
                # Correlation matrix of residuals
                Q = (1-alpha-beta)*S + alpha*(prev_resid@prev_resid.T) + beta*Q # DCC estimator

            # Get diagonal matrix of conditional standard deviation
            D_t = np.sqrt( 
                np.clip(
                    np.diag(Q),
                    1e-12, # Ensure no division by zero
                    None
                )
            )

            # Correlation matrix of the standardized residuals at time t
            R_t = np.diag(1.0/D_t) @ Q @ np.diag(1.0/D_t) # R_t = D_t^{-1} * H_t * D_t^{-1} (Engel, 2002)

            resid = residuals[t]

            try:
                # Solve R_t * x = e
                sol = np.linalg.solve(R_t, resid) # solve R_t * x = resid
                quadratic = resid @ sol
                sign, logdet = np.linalg.slogdet(R_t) # Returns the sign and the natural log of determinant of R_t
                if sign <= 0:
                    return np.inf
            except np.linalg.LinAlgError:
                return np.inf
            
            # Correlation loglikelihood contribution up to constant
            NLL += 0.5 * (logdet + quadratic)
        
        return NLL


    def fit_dcc(residuals, x0=(0.2, 0.97-0.02)):
        """
        Fit DCC(1,1) by minimizing negative log-likelihood
        """

        residuals = np.asarray(residuals)
        bounds = [
            (1e-8, 0.999999),
            (1e-8, 0.999999)
        ] # NOTE: optimize this part

        constraints = (
            {
                "type":"ineq",
                "fun": lambda p: 0.9999 - (p[0] + p[1])
            },
        )

        dcc = minimize(
            DCC_GARCH_MODEL.dcc_NLL,
            (0.05, 0.9),
            args=(residuals, ),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        alpha, beta = dcc.x
        # Reconstruct Q_t and R_t paths
        n_period, n_asset = residuals.shape
        S = np.corrcoef(residuals.T)
        Q = S.copy()
        Q_list, R_list = [], []
        for t in range(n_period):
            if t > 0:
                prev_resid = residuals[(t-1):t, :].T
                Q = (1-alpha-beta)*S + alpha*(prev_resid@prev_resid.T) + beta*Q
            diag_std = np.sqrt(
                np.clip(
                    np.diag(Q),
                    1e-12,
                    None
                )
            )
            R = np.diag(1.0/diag_std) @ Q @ np.diag(1.0/diag_std)
            Q_list.append(Q.copy())
            R_list.append(R.copy())
        
        return {
            "a":alpha,
            "b":beta,
            "Qt":Q_list,
            "Rt":R_list,
            "S":S,
            "success":dcc.success
        }
    # Build covariance matrix H_t and forecast

    def build_covmatrix(var_matrix, R_list):
        '''
        Args:
            var_matrix: T x M matrix of conditional variances from Univariate GARCH
            R_list: List of correlation matrices from DCC
        Return list of H_t = D_t * R_t * D_t from univariate GARCH and DCC R_t
        '''
        n_period, n_asset = var_matrix.shape
        covmatrix_list = []
        for t in range(n_period):
            D = np.diag(
                np.sqrt(var_matrix[t, :])
            ) # diagonal matrix of conditional standard deviation
            cov_matrix = D @ R_list[t] @ D # Conditional covariance matrix
            covmatrix_list.append(cov_matrix)
        
        return covmatrix_list


    def forecast_dcc_multi_step(
            h_last, r_last, garch_params,
            eps_last, Q_last, dcc_params, S,
            horizon=1
        ):
        """
        Multi-step forecast of conditional covariance matrices under DCC-GARCH(1,1).

        Args:
            h_last : (M,) last conditional variances
            r_last : (M,) last observed returns
            garch_params : dict of {asset: {'omega','alpha','beta'}}
            eps_last : (M,) last standardized residuals
            Q_last : (M,M) last Q matrix from DCC recursion
            dcc_params : dict {'a':..., 'b':...}
            S : (M,M) unconditional correlation matrix of eps
            horizon : int, number of steps ahead

        Returns
            H_path : list of (M,M) covariance forecasts
            h_path : (horizon, M) variance forecasts
            R_path : list of (M,M) correlation forecasts
            Q_path : list of (M,M) Q matrices
        """
        M = len(h_last)
        a, b = dcc_params["a"], dcc_params["b"]

        # ---------- Step 1: variance forecasts ----------
        h_path = np.empty((horizon, M))

        # 1-step-ahead: needs actual r_last
        for j, (name, p) in enumerate(garch_params.items()):
            omega, alpha, beta = p['omega'], p['alpha'], p['beta']
            h_path[0, j] = omega + alpha * (r_last[j]**2) + beta * h_last[j]

        # Multi-step expectation (replace r^2 with expected h)
        for k in range(1, horizon):
            for j, (name, p) in enumerate(garch_params.items()):
                omega, alpha, beta = p['omega'], p['alpha'], p['beta']
                phi = alpha + beta
                h_path[k, j] = omega + phi * h_path[k-1, j]

        # ---------- Step 2: correlation forecasts ----------
        Q_path, R_path = [], []
        
        # 1-step-ahead
        Q_next = (1 - a - b) * S + a * np.outer(eps_last, eps_last) + b * Q_last
        Q_path.append(Q_next.copy())
        dq = np.sqrt(np.clip(np.diag(Q_next), 1e-12, None))
        R_path.append(Q_next / np.outer(dq, dq))

        # 2..H: use expected recursion (E[eps eps'] ~ S)
        for k in range(1, horizon):
            Q_next = (1 - a - b) * S + a * S + b * Q_path[-1]   # simplifies to S + b*(Q_{k-1}-S)
            Q_path.append(Q_next.copy())
            dq = np.sqrt(np.clip(np.diag(Q_next), 1e-12, None))
            R_path.append(Q_next / np.outer(dq, dq))

        # ---------- Step 3: combine into H ----------
        H_path = []
        for k in range(horizon):
            D_k = np.diag(np.sqrt(h_path[k, :]))
            H_path.append(D_k @ R_path[k] @ D_k)

        return H_path, h_path, R_path, Q_path

    def forecast_dcc_one_step(residuals, dcc_fit):
        '''
        One-step ahead forecast using last residual and last covariance matrix of standardized residuals (not a true correlation matrix)
        '''
        alpha, beta = dcc_fit["a"], dcc_fit["b"]
        Q_last = dcc_fit["Qt"][-1].copy()
        S = np.corrcoef(residuals.T)
        resid = residuals[-1][:, None] # Mx1
        Q_forecast = (1-alpha-beta)*S + alpha*(resid@resid.T) + beta*Q_last
        diag_std = np.sqrt(
            np.clip(
                np.diag(Q_forecast),
                1e-12, 
                None
            )
        )
        R_forecast = np.diag(1.0/diag_std) @ Q_forecast @ np.diag(1.0/diag_std)

        return Q_forecast, R_forecast

    def forecast_H_one_step(h_last, garch_params, r_last):
        '''
        One-step ahead diagonal vol forecast
        '''
        n_asset = len(h_last)
        variance_forecast = np.empty(n_asset) # initialize variance
        for j, (name, p) in enumerate(garch_params.items()):
            omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
            variance_forecast[j] = omega + alpha*(r_last[j]**2) + beta*h_last[j]

        return variance_forecast


class SVR_MODEL:
    def parkinson_variance(high, low):
        """
        Parkinson variance estimator with high and low prices
        var_{Pt} = [ln(H_t/L_t)^2]/(4*ln2)
        """
        return (np.log(high/low)**2) / (4*np.log(2))

    def range_based_covariance_matrix(data:pd.DataFrame) -> pd.DataFrame:
        """
        Compute range-based covariance matrix for multiple assets
        Args:
            data: TxM dataframe in MultiIndex format (asset, price_type). Example: ('HPG', 'high), ('HPG', 'low'),...
        Returns:
            Range-based covariance matrix for multiple assets
        """

        tickers = data.columns.get_level_values(0).unique(0)
        n_assets = len(tickers)
        cov_matrices = {}

        for date, row in data.iterrows():
            variances = {}
            for asset in tickers:
                high_price, low_price = row[asset, 'high'], row[asset, 'low']
                variances[asset] = SVR_MODEL.parkinson_variance(high_price, low_price)

            cov_matrix = pd.DataFrame(
                np.zeros((n_assets, n_assets)),
                index=tickers,
                columns=tickers
            ) # Initialize covariance matrix

            # Fill diagonal with variances estimated with Parkinson
            for asset in tickers:
                cov_matrix.loc[asset, asset] = variances[asset]

            # Off-diagonals
            for i, asset_i in enumerate(tickers):
                for j, asset_j in enumerate(tickers):
                    if j>i: # Only get the upper triangular
                        high_sum = row[asset_i, "high"] + row[(asset_j, 'high')]
                        low_sum = row[asset_i, "low"] + row[asset_j, 'low']
                        var_sum = SVR_MODEL.parkinson_variance(high_sum, low_sum)

                        cov = 0.5 * (var_sum - variances[asset_i] - variances[asset_j])
                        cov_matrix.loc[asset_i, asset_j] = cov
                        cov_matrix.loc[asset_j, asset_i] = cov
                    
            cov_matrices[date] = cov_matrix
        
        return cov_matrices

    def cholesky_decomposition(
        G: np.ndarray,
        tol=1e-12,
        jitter_start=1e-12,
        jitter_max=1e-3
    ):
        # Symmetrize
        Gs = 0.5 * (G + G.T)

        try:
            upper_triang = np.linalg.cholesky(Gs, upper=True)
            # return upper_triang
        except np.linalg.LinAlgError:
            # Eigenvalue correction
            w, Q = np.linalg.eigh(Gs)
            w_clipped = np.maximum(w, tol)
            G_corr = (Q * w_clipped) @ Q.T

            try:
                upper_triang = np.linalg.cholesky(G_corr, upper=True)
                # return upper_triang
            except np.linalg.LinAlgError:
                # Final fallback: escalating diagonal jitter
                jitter = jitter_start
                I = np.eye(G.shape[0])
                while jitter <= jitter_max:
                    try:
                        upper_triang = np.linalg.cholesky(G_corr + jitter * I, upper=True)
                        # return upper_triang
                    except np.linalg.LinAlgError:
                        jitter *= 10.0
                raise np.linalg.LinAlgError("Cholesky failed: matrix far from positive definite even after eigenvalue clipping and jitter")
        
        return upper_triang
    # Step 3: For each entry of the cholesky factor, construct and train the autoregressive SVR model

    def get_cholesky_series(chol_factors):
        dates = list(chol_factors.keys())
        P0 = chol_factors[dates[0]] # The first upper triangular matrix
        assets = P0.columns
        series_dict = {}
        n_assets = len(assets)

        for i in range(n_assets):
            for j in range(i, n_assets):
                series_dict[(i, j)] = pd.Series(
                    [chol_factors[d].iloc[i, j] for d in dates],
                    index=dates
                )
        
        return series_dict

    def fit_SVR(series, lags=15):
        """
        Fit SVR to Cholesky entry
        Returns:
            Fitted series
        """
        y = series.values
        X = np.column_stack([np.roll(y, k) for k in range(1, lags + 1)])
        X, y = X[lags:], y[lags:]
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel='rbf', C=1.0, epsilon=0.01)
        )
        model.fit(X, y)

        return model

    def forecast_svr(model, hist, steps=1, lags=15):
        """
        Forecast with SVR
        Args:
            model: Fitted SVR model
            hist: Historical data
            steps: Forecast steps
            lags: Days to input into training
        Returns Cholesky entries
        """
        
        preds = []
        h = hist.copy()
        for _ in range(steps):
            x = h[-lags:].reshape(1, -1)
            pred = model.predict(x)[0]
            preds.append(pred)
            h = np.append(h, pred)
        
        return preds

    def forecast_covariance(chol_factors, horizon=1, lags=20):
        series_dict = SVR_MODEL.get_cholesky_series(chol_factors)
        models = {
            k: SVR_MODEL.fit_SVR(v, lags=lags) for k, v in series_dict.items()
        }
        forecasts = {
            k: SVR_MODEL.forecast_svr(models[k], series_dict[k].values
        , steps=horizon, lags=lags) for k in series_dict.keys()
        }
        n_assets = len(chol_factors[next(iter(chol_factors))]) # number of assets
        pred_covs = []
        for step in range(horizon):
            # Build forecasted P_t
            P_fc = np.zeros((n_assets, n_assets)) # Initialize matrix
            for (i, j), vals in forecasts.items():
                P_fc[i, j] = vals[step]
            
            # Covariance forecast
            G_fc = P_fc.T @ P_fc
            pred_covs.append(G_fc)
        
        return pred_covs

    def svr_model_forecast(
            train_data, 
            horizon=1, 
            lags=30
        ):

        cov_matrices = SVR_MODEL.range_based_covariance_matrix(train_data)

        # Step 2: The matrices are decomposed using Cholesky decomposition in the form G_t = P_t' P_t
        # This is to ensure the covariance matrix is always positive definite
        chol_factors = {}
        for date, cov in cov_matrices.items():
            upper_triang = SVR_MODEL.cholesky_decomposition(cov.values)
            chol_factors[date] = pd.DataFrame(
                upper_triang,
                index=cov.index,
                columns=cov.columns
            )

        # Step 3: Predict covariances
        pred_covs = SVR_MODEL.forecast_covariance(
            chol_factors=chol_factors,
            horizon=horizon,
            lags=lags # Experiment to find the best lags
        )

        return pred_covs