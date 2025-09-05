import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import RMSprop
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import pandas as pd
import math
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy.lib.stride_tricks import sliding_window_view
import math
from dataclasses import dataclass
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


class LSTM_BEKK_MODEL:

    class LSTMBEKKModel(nn.Module):
        """
        LSTM-BEKK model for multivariate volatility modeling
        """
        def __init__(
            self,
            n_assets,
            hidden_size=None,
            num_layers=3,
            dropout=0.1,
            beta=1.0
        ):
            """
            Initialize LSTM-BEKK Model
            Args: 
                n_assets: Number of stocks in the portfolio
                hidden_size: LSTM hidden size, recommended to set as equal to the number of stocks
                num_layers: Number of LSTM layers. Recommended 3-5
                dropout: Dropout rate for regularization. Recommended 0.1-0.2
                beta: Swish activation parameter
            """
            super(LSTM_BEKK_MODEL.LSTMBEKKModel, self).__init__()

            self.n_assets = n_assets
            self.hidden_size = hidden_size or n_assets
            self.num_layers = num_layers
            self.beta = nn.Parameter(torch.tensor(beta))

            # Static lower triangular matrix C
            self.C = nn.Parameter(torch.randn(n_assets, n_assets))

            # Scalar parameters for BEKK component
            self.log_a = nn.Parameter(torch.tensor(-3.0))
            self.log_b = nn.Parameter(torch.tensor(-0.1))

            # LSTM network for dynamic component
            self.lstm = nn.LSTM(
                input_size=n_assets,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )

            # Output layer to generate lower triangular matrix element
            n_lower_triangular = n_assets * (n_assets + 1) // 2
            self.output_layer = nn.Linear(self.hidden_size, n_lower_triangular)

        def swish_activation(self, x):
            '''Swish activation function: x * sigmoid(beta * x)'''
            return x * torch.sigmoid(self.beta * x)
        
        def make_lower_triangular(self, vec):
            '''
            Convert vector to lower triangular matrix using completely functional approach
            '''
            batch_size = vec.shape[0]
            n = self.n_assets
            device = vec.device
            
            L = torch.zeros(batch_size, n, n, device=device) # Initialize lower-triangular matrix

            k = 0
            for i in range(n):
                for j in range(i + 1):
                    L[:, i, j] = vec[:, k]
                    k += 1

            # Create position encodings for lower triangular matrix
            positions = []
            for i in range(n):
                for j in range(i + 1):
                    positions.append((i, j))
            
            # Apply Swish activation to diagonal elements
            diag_elements = torch.diagonal(L, dim1=1, dim2=2)
            active_L = L.clone()
            for i in range(n):
                active_L[:, i, i] = self.swish_activation(diag_elements[:, i])

            return active_L
        
        def get_static_C(self):
            """Get static lower triangular matrix C"""
            # Make lower triangular
            C_lower = torch.tril(self.C)
            
            # Ensure positive diagonal elements using functional approach
            diag_values = torch.diagonal(C_lower, dim1=-2, dim2=-1)
            abs_diag_values = torch.abs(diag_values) + 1e-6
            
            # Create diagonal matrix
            diag_matrix = torch.diag_embed(abs_diag_values)
            
            # Create off-diagonal part
            off_diag = C_lower - torch.diag_embed(diag_values)
            
            # Combine
            result = off_diag + diag_matrix
            
            return result
        
        def get_bekk_params(self):
            """Get positive BEKK parameters with stationarity constraint"""
            a = torch.exp(self.log_a)
            b = torch.exp(self.log_b)

            # Ensure stationarity: a + b < 1
            total = a + b
            scale_factor = torch.where(total >= 0.999, 0.998 / total, torch.tensor(1.0, device=total.device))
            
            a_scaled = a * scale_factor
            b_scaled = b * scale_factor

            return a_scaled, b_scaled

        def forward(self, returns):
            """
            Forward pass using completely functional approach
            """
            batch_size, seq_len, n_assets = returns.shape
            device = returns.device

            if seq_len == 0:
                empty_cov = torch.zeros(batch_size, 0, n_assets, n_assets, device=device)
                return empty_cov, torch.tensor(0.0, device=device)

            # Get static components
            C_static = self.get_static_C()
            CC_static = torch.mm(C_static, C_static.t())
            a, b = self.get_bekk_params()

            # LSTM forward pass
            lstm_out, _ = self.lstm(returns)
            C_t_vec = self.output_layer(lstm_out)

            # Generate dynamic matrices
            # Reshape for batch processing
            C_t_vec_reshaped = C_t_vec.view(-1, C_t_vec.shape[-1])
            C_t_matrices_flat = self.make_lower_triangular(C_t_vec_reshaped)
            C_t_matrices = C_t_matrices_flat.view(batch_size, seq_len, n_assets, n_assets)

            # Process sequence using functional approach
            all_H = []
            all_nll = []
            
            # Initial H
            H_current = CC_static.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            
            for t in range(seq_len):
                # Get current dynamic component
                C_t = C_t_matrices[:, t]
                CC_dynamic = torch.bmm(C_t, C_t.transpose(-2, -1))

                # BEKK component
                if t > 0:
                    r_prev = returns[:, t-1].unsqueeze(-1)
                    rr_prev = torch.bmm(r_prev, r_prev.transpose(-2, -1))
                    bekk_term = a * rr_prev + b * H_current
                else:
                    bekk_term = b * H_current

                # Compute new H
                H_new = CC_static.unsqueeze(0) + CC_dynamic + bekk_term
                
                # Add regularization
                reg_term = torch.eye(n_assets, device=device) * 1e-6
                H_new = H_new + reg_term.unsqueeze(0)
                
                all_H.append(H_new)

                # Compute likelihood
                try:
                    L = torch.linalg.cholesky(H_new)
                    log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
                    
                    r_t = returns[:, t].unsqueeze(-1)
                    y = torch.triangular_solve(r_t, L, upper=False)[0]
                    quad_form = torch.sum(y ** 2, dim=(-2, -1))
                    
                    nll_t = 0.5 * (log_det + quad_form)
                    all_nll.append(torch.mean(nll_t))
                    
                except Exception as e:
                    print(f"Cholesky failed at t={t}: {e}")
                    all_nll.append(torch.tensor(1e6, device=device))

                # Update H_current - create completely new tensor
                H_current = H_new.clone().detach()

            # Combine results
            covariance_matrices = torch.stack(all_H, dim=1)
            total_nll = sum(all_nll) if all_nll else torch.tensor(0.0, device=device)
            
            return covariance_matrices, total_nll

        def forecast(self, past_returns, n_steps=20, sampling=False):
            """
            Multi-step ahead forecast
            Args:
                past_returns: Tensor (1, seq_len, n_assets)
                n_steps: number of steps ahead
                sampling: if True, sample future returns, else assume zero mean
            Returns:
                forecasts: list of covariance matrices for n_steps
            """

            self.eval()
            device = past_returns.device
            with torch.no_grad():
                # Run LSTM to get dynamic component
                lstm_out, (h_n, c_n) = self.lstm(past_returns)
                hidden = lstm_out[:, -1:] # Last step hidden state

                # Get static components
                C_static = self.get_static_C()
                CC_static = torch.mm(C_static, C_static.t())
                a, b = self.get_bekk_params()

                # Last observed covariance
                covariances, _ = self.forward(past_returns)
                H_current = covariances[:, -1]

                # Last observed return
                r_prev = past_returns[:, -1]

                forecasts = []
                for step in range(n_steps):
                    # Dynamic matrix from hidden state
                    C_vec = self.output_layer(hidden).view(-1, self.output_layer.out_features)
                    C_mat = self.make_lower_triangular(C_vec)[0]
                    CC_dynamic = C_mat @ C_mat.T

                    rr_prev = r_prev @ r_prev.T
                    H_new = CC_static + CC_dynamic + a * rr_prev + b * H_current
                    H_new = H_new + torch.eye(self.n_assets, device=device) * 1e-6

                    forecasts.append(H_new)

                    # Update for next iteration
                    H_current = H_new.clone()
                    if sampling:
                        r_prev = torch.distributions.MultivariateNormal(
                            loc=torch.zeros(self.n_assets, device=device), covariance_matrix=H_new
                        ).sample()
                    else:
                        r_prev = torch.zeros(self.n_assets, device=device)

                    # Evolve LSTM hidden state with zero input
                    zero_input = torch.zeros(1, 1, self.n_assets, device=device)
                    hidden, (h_n, c_n) = self.lstm(zero_input, (h_n, c_n))

                return forecasts

    class ReturnDataset(Dataset):
        """Dataset class for multivariate return data"""
        def __init__(self, returns, seq_len=50):
            self.returns = torch.tensor(returns, dtype=torch.float32)
            self.seq_len = seq_len
            self.T, self.n_assets = self.returns.shape

        def __len__(self):
            return max(0, self.T - self.seq_len + 1)
        
        def __getitem__(self, index):
            return self.returns[index:index + self.seq_len]


    class LSTMBEKKTrainer:
        """Trainer class for LSTM-BEKK Model"""
        
        def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
            self.model = model
            self.optimizer = LSTM_BEKK_MODEL.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=1e-8
            )
            self.scheduler = LSTM_BEKK_MODEL.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )

        def train_epoch(self, train_loader):
            """Train for one epoch"""

            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch_returns in enumerate(train_loader):
                batch_returns = batch_returns.to(next(self.model.parameters()).device)

                # Zero gradients
                self.optimizer.zero_grad()

                try:
                    # Forward pass
                    _, loss = self.model(batch_returns)

                    # Regularization
                    reg_loss = torch.tensor(0.0, device=loss.device)
                    for param in self.model.parameters():
                        reg_loss = reg_loss + torch.sum(torch.abs(param))
                    
                    total_loss_batch = loss + 1e-6 * reg_loss
                    
                    # Backward pass
                    total_loss_batch.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as error:
                    print(f"Error in training batch {batch_idx}: {error}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return total_loss / max(num_batches, 1)
        
        def validate(self, val_loader):
            """Validate the model"""
            self.model.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_returns in val_loader:
                    batch_returns = batch_returns.to(next(self.model.parameters()).device)

                    try:
                        _, loss = self.model(batch_returns)
                        total_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
                
            return total_loss / max(num_batches, 1)

        def train(self, train_loader, val_loader, epochs=100, patience=20):
            """Train the LSTM-BEKK model"""
            
            best_val_loss = float("inf")
            patience_counter = 0
            train_losses = []
            val_losses = []

            print("Starting LSTM-BEKK training...")
            a_param, b_param = self.model.get_bekk_params()
            print(f"Model parameters: a = {a_param:.4f}, b = {b_param:.4f}")

            for epoch in range(epochs):
                # Training
                train_loss = self.train_epoch(train_loader)
                train_losses.append(train_loss)

                # Validation
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                print(f"Epoch {epoch + 1}/{epochs}: Train loss = {train_loss:.6f}; Val loss = {val_loss:.6f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "best_lstm_bekk_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Load best model
            self.model.load_state_dict(torch.load('best_lstm_bekk_model.pth'))
            print("Training completed!")

            return train_losses, val_losses
        
        
    def evaluate_model(model, test_loader):
        """Evaluate model on test data"""
        model.eval()
        total_nll = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_returns in test_loader:
                batch_returns = batch_returns.to(next(model.parameters()).device)

                try:
                    _, nll = model(batch_returns)
                    total_nll += nll.item()
                    num_batches += 1 
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
                    
        avg_nll = total_nll / max(num_batches, 1)
        return avg_nll

    
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
            cov_matrix = C @ C.T + A @ (residual @ residual.T) @ A.T + B @ cov_matrix @ B.T
            
            # Check for numerical overflow
            if np.any(np.isinf(cov_matrix)) or np.any(np.isnan(cov_matrix)):
                return np.inf
            
            if np.linalg.norm(cov_matrix, 'fro') > 1e15:
                return np.inf

            sign, logdet = np.linalg.slogdet(cov_matrix)

            if sign <= 0:
                return 1e6 # Ensure positive definiteness
            loglikelihood += 0.5 * (n_assets*np.log(2*np.pi) + logdet + residual.T@np.linalg.inv(cov_matrix)@residual)
        
        return loglikelihood.flatten()[0]

    def fit_bekk(returns, x0):
        n_periods, n_assets = returns.shape

        def stability_constraint(params, n_assets):
            _, A, B = BEKK_GARCH_MODEL.unpack_params(params, n_assets)
            stability_matrix = A @ A.T + B @ B.T
            max_eigenvalue = np.max(np.linalg.eigvals(stability_matrix))

            return 1 - max_eigenvalue

        bekk = minimize(
            BEKK_GARCH_MODEL.bekk_loglikelihood, x0,
            args=(returns,),
            method="SLSQP",
            options={"maxiter":500},
            constraints= [{
                'type':'ineq',
                'fun':stability_constraint,
                'args':(n_assets,)
            }]
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

            # Bound the values to prevent explosion
            cov_matrix_t = np.clip(cov_matrix_t, -1e10, 1e10)

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
    # -------------------------
    # 1) Parkinson variance
    # -------------------------
    def parkinson_variance(high, low):
        """Parkinson variance estimator: (ln(H/L))^2 / (4 ln 2).
        Assumes prices > 0. If you have zeros / negatives, handle beforehand."""
        return (np.log(high / low) ** 2) / (4.0 * np.log(2.0))


    # -------------------------
    # 2) Range-based covariance matrices
    # -------------------------
    def range_based_covariance_matrix(data: pd.DataFrame) -> dict:
        """
        data: TxM DataFrame with MultiIndex columns (ticker, price_type) where price_type in {'high','low'}
        Returns: dict mapping date -> covariance DataFrame (index & columns = tickers)
        """
        # get tickers reliably
        tickers = data.columns.get_level_values(0).unique()
        n_assets = len(tickers)
        cov_matrices = {}

        for date, row in data.iterrows():
            # compute Parkinson variances
            variances = {}
            for asset in tickers:
                high_price = row[(asset, 'high')]
                low_price = row[(asset, 'low')]
                # ensure positivity
                if high_price <= 0 or low_price <= 0:
                    raise ValueError(f"Nonpositive price for {asset} on {date}")
                variances[asset] = SVR_MODEL.parkinson_variance(high_price, low_price)

            cov_matrix = pd.DataFrame(
                np.zeros((n_assets, n_assets)),
                index=tickers,
                columns=tickers
            )

            # diagonal
            for asset in tickers:
                cov_matrix.loc[asset, asset] = variances[asset]

            # off-diagonal using range-of-sum formula (Eq.9)
            for i, asset_i in enumerate(tickers):
                for j in range(i + 1, n_assets):
                    asset_j = tickers[j]
                    high_sum = row[(asset_i, "high")] + row[(asset_j, 'high')]
                    low_sum = row[(asset_i, "low")] + row[(asset_j, 'low')]
                    var_sum = SVR_MODEL.parkinson_variance(high_sum, low_sum)
                    cov = 0.5 * (var_sum - variances[asset_i] - variances[asset_j])
                    cov_matrix.iat[i, j] = cov
                    cov_matrix.iat[j, i] = cov

            cov_matrices[date] = cov_matrix

        return cov_matrices


    # -------------------------
    # 3) Robust Cholesky decomposition (returns upper triangular P such that G = P.T @ P)
    # -------------------------
    def cholesky_decomposition(G: np.ndarray,
                            tol=1e-12,
                            jitter_start=1e-12,
                            jitter_max=1e-3):
        """
        Returns an upper-triangular matrix P such that G ≈ P.T @ P.
        Uses eigenvalue clipping + jitter fallback for near-singular / non-PD matrices.
        """
        # symmetrize
        Gs = 0.5 * (G + G.T)

        # first try: standard cholesky (numpy returns lower L)
        try:
            L = np.linalg.cholesky(Gs)          # lower-triangular L
            P = L.T                             # upper triangular P such that Gs = L @ L.T = P.T @ P
            return P
        except np.linalg.LinAlgError:
            # eigenvalue clipping
            w, Q = np.linalg.eigh(Gs)
            w_clipped = np.maximum(w, tol)
            G_corr = Q @ np.diag(w_clipped) @ Q.T

            try:
                L = np.linalg.cholesky(G_corr)
                return L.T
            except np.linalg.LinAlgError:
                # escalate diagonal jitter
                jitter = jitter_start
                I = np.eye(G.shape[0])
                while jitter <= jitter_max:
                    try:
                        L = np.linalg.cholesky(G_corr + jitter * I)
                        return L.T
                    except np.linalg.LinAlgError:
                        jitter *= 10.0
                raise np.linalg.LinAlgError(
                    "Cholesky failed: matrix far from positive definite even after eigenvalue clipping and jitter"
                )


    # -------------------------
    # 4) Extract series of Cholesky entries
    # -------------------------
    def get_cholesky_series(chol_factors: dict) -> dict:
        """
        chol_factors: dict[date] -> DataFrame (upper-triangular) with columns indexed by tickers.
        Returns dict mapping (i,j) -> pd.Series indexed by sorted dates.
        """
        dates = sorted(chol_factors.keys())
        P0 = chol_factors[dates[0]]
        assets = list(P0.columns)
        n_assets = len(assets)
        series_dict = {}
        for i in range(n_assets):
            for j in range(i, n_assets):
                series_dict[(i, j)] = pd.Series(
                    [chol_factors[d].iloc[i, j] for d in dates],
                    index=dates
                )
        return series_dict


    # -------------------------
    # 5) Lagged matrix builder (robust)
    # -------------------------
    def lagged_matrix(y: np.ndarray, lags: int):
        n = len(y)
        if n <= lags:
            raise ValueError(f"Series too short (len={n}) for lags={lags}")
        X = sliding_window_view(y, lags)[:-1]
        y_target = y[lags:]
        return X, y_target


    # -------------------------
    # 6) fit_SVR (returns model + normalization)
    # -------------------------
    def fit_SVR(series, scaler:StandardScaler, lags=30, kernel='linear', C=1.0, epsilon=0.01, standardize: bool = True):
        """
        Returns: (fitted_model, y_mean, y_std)
        """
        y = np.asarray(series, dtype=float)

        # target normalization
        y_mean, y_std = 0.0, 1.0
        if standardize:
            y_mean, y_std = y.mean(), y.std()
            if y_std < 1e-8:
                y_std = 1.0
            y = (y - y_mean) / y_std

        # build X, y_target
        X, y_target = SVR_MODEL.lagged_matrix(y, lags)

        # model = make_pipeline(
        #     scaler,  # standardize features
        # )
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X, y_target)
        return model, y_mean, y_std


    # -------------------------
    # 7) forecast_svr (recursive multi-step)
    # -------------------------
    def forecast_svr(model, hist, steps=1, lags=30, y_mean=0.0, y_std=1.0):
        """
        hist: raw historical series (1D array) in original scale
        Returns list of predicted values in original scale (length == steps)
        """
        hist = np.asarray(hist, dtype=float)
        if len(hist) < lags:
            raise ValueError("Not enough history for forecasting with given lags")

        # keep standardized history for inputs
        h_std = (hist - y_mean) / y_std
        preds = []
        for _ in range(steps):
            x = h_std[-lags:].reshape(1, -1)           # standardized features
            pred_std = model.predict(x)[0]             # model predicts in standardized target space
            pred = float(pred_std * y_std + y_mean)    # back to original scale
            preds.append(pred)
            # append standardized prediction to standardized history (for next step)
            h_std = np.append(h_std, pred_std)
        return preds


    # -------------------------
    # 8) forecast_covariance (wires everything together)
    # -------------------------
    def forecast_covariance(chol_factors: dict, horizon: int = 20, lags: int = 20,
                            kernel='linear', C=1.0, epsilon=0.01, standardize: bool = True, scaler=StandardScaler):
        """
        Fit SVR per Cholesky entry and produce horizon-step forecasts of covariance matrices.
        Returns: list of horizon numpy arrays (n_assets x n_assets), each symmetric PD (in practice).
        """
        series_dict = SVR_MODEL.get_cholesky_series(chol_factors)

        # train models (store per-entry model + normalization)
        models = {}
        for k, series in series_dict.items():
            model, y_mean, y_std = SVR_MODEL.fit_SVR(series, lags=lags, kernel=kernel, C=C, epsilon=epsilon, standardize=standardize, scaler=scaler)
            models[k] = {'model': model, 'mean': y_mean, 'std': y_std}

        # forecast each entry
        forecasts = {}
        for k, meta in models.items():
            series_hist = series_dict[k].values
            preds = SVR_MODEL.forecast_svr(meta['model'], series_hist, steps=horizon, lags=lags, y_mean=meta['mean'], y_std=meta['std'])
            forecasts[k] = preds

        # assemble P matrices and reconstruct covariances
        # get number of assets from first chol_factors entry
        n_assets = len(chol_factors[next(iter(sorted(chol_factors.keys())))])
        pred_covs = []
        for step in range(horizon):
            P_fc = np.zeros((n_assets, n_assets))
            for (i, j), vals in forecasts.items():
                P_fc[i, j] = vals[step]
            # reconstruct covariance: G = P.T @ P
            G_fc = P_fc.T @ P_fc
            # enforce symmetry numerically
            G_fc = 0.5 * (G_fc + G_fc.T)
            pred_covs.append(G_fc)
        return pred_covs


    # -------------------------
    # 9) wrapper for full pipeline
    # -------------------------
    def svr_model_forecast(train_data: pd.DataFrame, horizon=20, lags=30, scaler=StandardScaler,**svr_kwargs):
        cov_matrices = SVR_MODEL.range_based_covariance_matrix(train_data)
        # cholesky factors dict
        chol_factors = {}
        for date, cov in cov_matrices.items():
            P = SVR_MODEL.cholesky_decomposition(cov.values)
            chol_factors[date] = pd.DataFrame(P, index=cov.index, columns=cov.columns)
        pred_covs = SVR_MODEL.forecast_covariance(chol_factors=chol_factors, horizon=horizon, lags=lags, scaler=scaler, **svr_kwargs)
        return pred_covs
