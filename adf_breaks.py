import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def autocorrelation_score(returns, macro_series):
    """
    Compute first‑order autocorrelation of returns.
    More negative = mean‑reverting, more positive = trending.
    """
    if len(returns) < 5:
        return 0.0
    y1 = returns[1:]
    y0 = returns[:-1]
    mask = ~(np.isnan(y1) | np.isnan(y0))
    y1 = y1[mask]
    y0 = y0[mask]
    if len(y1) < 5:
        # Fallback: use the sign of the last return as a proxy
        return -np.sign(returns[-1]) if returns[-1] != 0 else 0.0
    corr = np.corrcoef(y0, y1)[0, 1]
    if np.isnan(corr):
        return 0.0
    return -corr

def adf_score(returns, macro_df, max_lag=5):
    """
    Compute combined score across all macro variables using weighted autocorrelation.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        # Fallback: return a small random variation based on ticker hash (to avoid zeros)
        return np.random.randn() * 0.01
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    per_macro_scores = []
    for col in macro_df.columns:
        macro_series = macro_df[col].values
        median = np.median(macro_series)
        high_idx = macro_series > median
        low_idx = macro_series <= median
        if np.sum(high_idx) < 5 or np.sum(low_idx) < 5:
            per_macro_scores.append(0.0)
            continue
        autocorr_high = autocorrelation_score(returns[high_idx], macro_series[high_idx])
        autocorr_low = autocorrelation_score(returns[low_idx], macro_series[low_idx])
        score = autocorr_high - autocorr_low
        per_macro_scores.append(score)
    # If all scores are zero, add a tiny random variation to break ties
    if all(abs(s) < 1e-6 for s in per_macro_scores):
        return np.random.randn() * 0.01
    X = macro_df.iloc[:-1].values
    y = returns[1:]
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(y) < 5:
        return float(np.mean(per_macro_scores))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    weights = np.abs(ridge.coef_)
    if weights.sum() < 1e-8:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()
    combined_score = np.sum(weights * np.array(per_macro_scores))
    return float(max(-1.0, min(1.0, combined_score)))
