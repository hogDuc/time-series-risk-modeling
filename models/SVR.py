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
            variances[asset] = parkinson_variance(high_price, low_price)

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
                var_sum = parkinson_variance(high_sum, low_sum)
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
    X, y_target = lagged_matrix(y, lags)

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
    series_dict = get_cholesky_series(chol_factors)

    # train models (store per-entry model + normalization)
    models = {}
    for k, series in series_dict.items():
        model, y_mean, y_std = fit_SVR(series, lags=lags, kernel=kernel, C=C, epsilon=epsilon, standardize=standardize, scaler=scaler)
        models[k] = {'model': model, 'mean': y_mean, 'std': y_std}

    # forecast each entry
    forecasts = {}
    for k, meta in models.items():
        series_hist = series_dict[k].values
        preds = forecast_svr(meta['model'], series_hist, steps=horizon, lags=lags, y_mean=meta['mean'], y_std=meta['std'])
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
    cov_matrices = range_based_covariance_matrix(train_data)
    # cholesky factors dict
    chol_factors = {}
    for date, cov in cov_matrices.items():
        P = cholesky_decomposition(cov.values)
        chol_factors[date] = pd.DataFrame(P, index=cov.index, columns=cov.columns)
    pred_covs = forecast_covariance(chol_factors=chol_factors, horizon=horizon, lags=lags, scaler=scaler, **svr_kwargs)
    return pred_covs
