"""
Stage 5: trend-QUALITY gates.

Stages 1-4 could reach ~32% CAGR at ~31% DD but the CAGR/DD trade was close to
linear, so simply de-scaling could not reach 30/30 -- the core itself had to get
better. The diagnosis said the losses were chop, so this stage gates TECL on how
clean the trend is (Kaufman efficiency ratio, MACD sign, RSI ceiling) rather than
only on how high or how calm the tape is.
"""

from sweep import sweep, report, save

SPEC = dict(
    trend_n=[100], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[60], mom_thr=[0.0], vol_n=[20], vol_max=[0.7],
    abs_vol_max=[None], vix_max=[None], require_slope=[True],
    confirm_style=["sym"], confirm_up=[5],
    cb_vol_pct=[0.90, 0.95], cb_drop=[None], cb_drop_n=[10], cb_vix=[None],
    crash_mom_thr=[None],
    er_n=[10, 20, 30, 60],
    er_min=[None, 0.20, 0.30, 0.40, 0.50],
    er_cash=[None, 0.15, 0.25],
    rsi_n=[14], rsi_max=[None, 75, 85],
    macd_pos=[False, True],
    sizing=["fixed"], target_vol=[None], max_leverage=[1.0], min_weight=[0.0],
    tecl_w=[0.85, 1.0], xlk_w=[1.0],
)

KEYS = ["cb_vol_pct", "er_n", "er_min", "er_cash", "rsi_max", "macd_pos", "tecl_w"]

if __name__ == "__main__":
    df, n = sweep("v2vol", SPEC)
    report(df, n, "STAGE 5 - quality gates, ranked by FULL Calmar", "ALL_Calmar", 15, KEYS)
    hit = df[(df.ALL_CAGR >= 0.30) & (df.ALL_maxDD >= -0.30)]
    report(hit, n, "STAGE 5 - MEETING 30% CAGR / 30% DD (full sample)", "ALL_CAGR", 25, KEYS)
    both = hit[(hit.IS_CAGR >= 0.25) & (hit.OOS_CAGR >= 0.25) &
               (hit.IS_maxDD >= -0.32) & (hit.OOS_maxDD >= -0.32)]
    report(both, n, "STAGE 5 - meeting 30/30 AND robust in both halves", "ALL_CAGR", 25, KEYS)
    save(df, "sweep5_quality.csv")
