def run_for_window(returns, macro_df, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    if macro_df is None or macro_df.empty:
        return None
    macro_window = macro_df.loc[ret_window.index]
    if len(macro_window) < len(ret_window):
        return None
    raw_scores = {}
    for ticker in ret_window.columns:
        s = adf_score(
            ret_window[ticker].values,
            macro_window,
            max_lag=config.MAX_LAG
        )
        if not np.isfinite(s):
            s = 0.0
        raw_scores[ticker] = float(s)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "adf_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }
