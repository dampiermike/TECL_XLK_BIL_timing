"""
Stage 3: does the TECS sleeve EVER help?

Holds the stage-2 long core fixed and sweeps the crash gate across every
dimension that could define "poised for a crash": momentum depth and window,
position vs the long MA, realised-vol percentile, drawdown from high, VIX level,
trend slope, and how many days the gate must persist. If a profitable short
configuration exists, this grid should contain it.
"""

from sweep import sweep, report, save

CORE = dict(
    trend_n=[100], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[60], mom_thr=[0.0], vol_n=[20], vol_max=[0.70],
    abs_vol_max=[None], vix_max=[None], require_slope=[True],
    confirm_style=["sym"], confirm_up=[5],
    cb_vol_pct=[0.95], cb_drop=[None], cb_drop_n=[10], cb_vix=[None],
)

SPEC = dict(
    CORE,
    crash_mom_n=[10, 20, 60],
    crash_mom_thr=[-0.03, -0.06, -0.10, -0.15],
    crash_below_n=[None, 50, 200],
    crash_vol_min=[None, 0.60, 0.85],
    crash_dd_n=[120],
    crash_dd=[None, -0.10, -0.20],
    crash_vix_min=[None, 25, 35],
    crash_slope_n=[None, 100],
    crash_confirm=[1, 3, 5],
)

if __name__ == "__main__":
    df, n = sweep("v2", SPEC)
    keys = ["crash_mom_n", "crash_mom_thr", "crash_below_n", "crash_vol_min",
            "crash_dd", "crash_vix_min", "crash_slope_n", "crash_confirm"]
    base = df[df.crash_mom_thr.isna()] if df.crash_mom_thr.isna().any() else None
    report(df, n, "STAGE 3 - TECS overlay, ranked by FULL-sample Calmar", "ALL_Calmar", 20, keys)
    report(df, n, "STAGE 3 - TECS overlay, ranked by FULL-sample CAGR", "ALL_CAGR", 15, keys)

    used = df[df.get("pct_TECS", 0).fillna(0) > 0.005] if "pct_TECS" in df else df
    print(f"\n\nconfigs that actually took a TECS position: {len(used)} of {len(df)}")
    report(used, n, "STAGE 3 - among configs that DID short, best Calmar", "ALL_Calmar", 20, keys)
    save(df, "sweep3_tecs.csv")
