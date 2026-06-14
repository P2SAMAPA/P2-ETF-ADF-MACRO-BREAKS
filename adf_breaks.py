import numpy as np
from scipy import stats

def adf_test(series, max_lag=5, break_indices=None):
    """
    Augmented Dickey‑Fuller test with optional break dummies.
    Returns t‑statistic for the unit root coefficient.
    """
    n = len(series)
    if n < 10:
        return 0.0
    # Create differenced series
    y = np.diff(series)
    # Create lagged level
    y_lag = series[:-1]
    # Create lagged differences (for ADF augmentation)
    X = [y_lag]
    # Add break dummies if provided
    if break_indices is not None and len(break_indices) > 0:
        for idx in break_indices:
            if idx < len(series):
                dummy = np.zeros(len(y))
                dummy[idx] = 1
                X.append(dummy)
    # Add lagged differences
    for lag in range(1, max_lag + 1):
        if len(y) > lag:
            X.append(np.roll(y, lag)[lag:])
    # Align all to same length
    min_len = min(len(x) for x in X)
    X = np.column_stack([x[:min_len] for x in X])
    y_aligned = y[:min_len]
    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_aligned))
    X = X[mask]
    y_aligned = y_aligned[mask]
    if len(y_aligned) < 10:
        return 0.0
    # OLS regression: y = beta0 + beta1 * y_lag + ... + eps
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_aligned, rcond=None)
        resid = y_aligned - X @ beta
        # Standard error of beta1 (coefficient on y_lag)
        sigma2 = np.sum(resid**2) / (len(resid) - X.shape[1])
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(sigma2 * XtX_inv[1, 1])
        t_stat = beta[1] / se_beta1
        return t_stat
    except:
        return 0.0

def adf_with_macro_breaks(returns, macro_series, max_lag=5, break_threshold=0.7):
    """
    Compute ADF statistic with breakpoints determined by macro quantiles.
    """
    if len(returns) < 20 or len(macro_series) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_series))
    returns = returns[:min_len]
    macro_series = macro_series[:min_len]
    # Find break indices where macro exceeds threshold percentile
    threshold = np.percentile(macro_series, break_threshold * 100)
    break_indices = np.where(macro_series > threshold)[0]
    # Limit to breaks within the series
    break_indices = [int(i) for i in break_indices if 0 < i < len(returns)-1]
    # Run ADF with break dummies
    t_stat = adf_test(returns, max_lag, break_indices)
    return t_stat

def adf_score(returns, macro_series, max_lag=5, break_threshold=0.7):
    """
    Score = -ADF t-statistic (higher = more mean‑reverting).
    """
    t_stat = adf_with_macro_breaks(returns, macro_series, max_lag, break_threshold)
    # More negative t-statistic means stronger mean reversion; we invert to make positive score
    score = -t_stat
    return max(0.0, min(10.0, score))
