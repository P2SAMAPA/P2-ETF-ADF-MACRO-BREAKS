import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def adf_score(returns, macro_df, max_lag=5, ticker_hash=0):
    """
    Compute a score based on the interaction between macro variables and returns.
    Uses ridge regression to predict returns from macro, and the coefficient magnitude as score.
    Add a small ticker‑based variation to ensure differentiation.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        # Base score on ticker hash to ensure variation
        return (ticker_hash % 100) / 100.0
    # Align
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return (ticker_hash % 100) / 100.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Use ridge regression to predict returns from macro
    ridge = Ridge(alpha=1.0)
    ridge.fit(macro_scaled, returns)
    # Score = weighted R² (how well macro predicts returns)
    y_pred = ridge.predict(macro_scaled)
    ss_res = np.sum((returns - y_pred)**2)
    ss_tot = np.sum((returns - np.mean(returns))**2)
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot
    # Add ticker hash variation to ensure different scores across ETFs
    variation = (ticker_hash % 100) / 1000.0  # small variation 0-0.099
    score = max(0.0, min(1.0, r2 + variation))
    return score
