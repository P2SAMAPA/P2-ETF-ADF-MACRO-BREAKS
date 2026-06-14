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
    # Compute lag‑1 autocorrelation
    y1 = returns[1:]
    y0 = returns[:-1]
    # Remove any NaN
    mask = ~(np.isnan(y1) | np.isnan(y0))
    y1 = y1[mask]
    y0 = y0[mask]
    if len(y1) < 5:
        return 0.0
    # Pearson correlation
    corr = np.corrcoef(y0, y1)[0, 1]
    if np.isnan(corr):
        return 0.0
    # Return negative correlation so that mean‑reverting (negative corr) gives positive score
    return -corr

def adf_score(returns, macro_df, max_lag=5):
    """
    Compute combined score across all macro variables using weighted autocorrelation.
    For each macro variable, compute autocorrelation separately on high/low regimes,
    then combine with macro importance weights.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # For each macro, compute regime‑sensitive autocorrelation
    per_macro_scores = []
    for col in macro_df.columns:
        macro_series = macro_df[col].values
        # Split at median
        median = np.median(macro_series)
        high_idx = macro_series > median
        low_idx = macro_series <= median
        if np.sum(high_idx) < 5 or np.sum(low_idx) < 5:
            per_macro_scores.append(0.0)
            continue
        autocorr_high = autocorrelation_score(returns[high_idx], macro_series[high_idx])
        autocorr_low = autocorrelation_score(returns[low_idx], macro_series[low_idx])
        # Score = difference in mean reversion between high and low macro
        score = autocorr_high - autocorr_low
        per_macro_scores.append(score)
    # Estimate macro importance via ridge regression (predict next‑day return)
    X = macro_df.iloc[:-1].values
    y = returns[1:]
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(y) < 10:
        return float(np.mean(per_macro_scores))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    weights = np.abs(ridge.coef_)
    weights = weights / (weights.sum() + 1e-8)
    # Weighted average
    combined_score = np.sum(weights * np.array(per_macro_scores))
    # Clip to reasonable range
    return float(max(-1.0, min(1.0, combined_score)))
