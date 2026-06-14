import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def adf_statistic(series, max_lag=5):
    """
    Compute ADF t‑statistic for a single series (no breaks).
    Returns t‑statistic (more negative = mean‑reverting).
    """
    n = len(series)
    if n < 10:
        return 0.0
    y = np.diff(series)
    y_lag = series[:-1]
    # Build X matrix: constant, lagged level, lagged differences
    X = [np.ones(len(y_lag)), y_lag]
    for lag in range(1, max_lag + 1):
        if len(y) > lag:
            X.append(np.roll(y, lag)[lag:])
    # Align lengths
    min_len = min(len(x) for x in X)
    X = np.column_stack([x[:min_len] for x in X])
    y_aligned = y[:min_len]
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_aligned))
    X = X[mask]
    y_aligned = y_aligned[mask]
    if len(y_aligned) < 10:
        return 0.0
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_aligned, rcond=None)
        resid = y_aligned - X @ beta
        sigma2 = np.sum(resid**2) / (len(resid) - X.shape[1])
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(sigma2 * XtX_inv[1, 1])
        t_stat = beta[1] / se_beta1
        return t_stat
    except:
        return 0.0

def macro_adf_score(returns, macro_series, max_lag=5):
    """
    Compute ADF statistic separately for high and low macro regimes,
    then return the difference (how much mean reversion changes with macro).
    """
    if len(returns) < 20 or len(macro_series) < 20:
        return 0.0
    # Align
    min_len = min(len(returns), len(macro_series))
    returns = returns[:min_len]
    macro_series = macro_series[:min_len]
    # Split at median
    median = np.median(macro_series)
    high_idx = macro_series > median
    low_idx = macro_series <= median
    if np.sum(high_idx) < 10 or np.sum(low_idx) < 10:
        return 0.0
    t_high = adf_statistic(returns[high_idx], max_lag)
    t_low = adf_statistic(returns[low_idx], max_lag)
    # Score = difference in mean‑reversion strength between regimes
    # More positive means ETF becomes more mean‑reverting when macro is high
    score = (t_low - t_high) / (abs(t_low) + abs(t_high) + 1e-8)
    return float(score)

def adf_score(returns, macro_df, max_lag=5):
    """
    Compute combined score across all macro variables.
    For each macro, compute the regime‑sensitive ADF difference.
    Then weighted average using ridge regression weights (predicting next‑day return).
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Compute per‑macro scores
    per_macro_scores = []
    for col in macro_df.columns:
        macro_series = macro_df[col].values
        s = macro_adf_score(returns, macro_series, max_lag)
        per_macro_scores.append(s)
    # Estimate macro importance via ridge regression (predict next‑day return)
    # Use lagged macro to predict next return
    X = macro_df.iloc[:-1].values
    y = returns[1:]
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(y) < 10:
        return np.mean(per_macro_scores)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    weights = np.abs(ridge.coef_)
    weights = weights / (weights.sum() + 1e-8)
    # Weighted average of per‑macro scores
    combined_score = np.sum(weights * np.array(per_macro_scores))
    return float(combined_score)
