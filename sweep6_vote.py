"""
Stage 6: the voting rule shape, plus the two dimensions never yet swept --
circuit-breaker variants (drawdown and VIX triggers, not just vol percentile)
and the choice of signal ticker (XLK vs QQQ).
"""

import sys

from sweep import sweep, report, save

SPEC = dict(
    trend_n=[100, 150], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[60], mom_thr=[0.0], vol_n=[20], vol_max_v=[0.7],
    v_xlk=[0.45, 0.55, 0.65], v_tecl=[0.70, 0.80, 0.90, 1.0],
    use_credit=[False, True],
    rsi_n=[14], rsi_max=[None, 75],
    confirm_style=["sym"], confirm_up=[3, 5],
    cb_vol_pct=[None, 0.90, 0.95], cb_drop=[None, -0.06, -0.10], cb_drop_n=[10],
    cb_vix=[None, 30],
    crash_mom_thr=[None],
    tecl_w=[0.80, 0.85, 1.0], xlk_w=[1.0],
)

KEYS = ["trend_n", "v_xlk", "v_tecl", "use_credit", "rsi_max", "confirm_up",
        "cb_vol_pct", "cb_drop", "cb_vix", "tecl_w"]

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "XLK"
    df, n = sweep("vote", SPEC, signal_ticker=ticker)
    df["signal"] = ticker
    report(df, n, f"STAGE 6 [{ticker}] - voting rule, ranked by FULL Calmar", "ALL_Calmar", 15, KEYS)
    hit = df[(df.ALL_CAGR >= 0.30) & (df.ALL_maxDD >= -0.30)]
    report(hit, n, f"STAGE 6 [{ticker}] - MEETING 30% CAGR / 30% DD", "ALL_CAGR", 25, KEYS)
    both = hit[(hit.IS_CAGR >= 0.25) & (hit.OOS_CAGR >= 0.25) &
               (hit.IS_maxDD >= -0.32) & (hit.OOS_maxDD >= -0.32)]
    report(both, n, f"STAGE 6 [{ticker}] - 30/30 AND robust in both halves", "ALL_CAGR", 25, KEYS)
    save(df, f"sweep6_vote_{ticker}.csv")
