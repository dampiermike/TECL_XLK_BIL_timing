"""Stage 2: v2 long side -- absolute vol gates, VIX gates, circuit breakers,
both confirmation styles. TECS still off, so the long engine is tuned alone."""

from sweep import sweep, report, save

SPEC = dict(
    trend_n=[100, 150, 200],
    trend_buffer=[0.0],
    prox_n=[20, 60],
    prox_thr=[0.93, 0.96],
    mom_n=[20, 60],
    mom_thr=[0.0],
    vol_n=[20],
    vol_max=[0.5, 0.7, None],
    abs_vol_max=[None, 0.18, 0.24, 0.30],
    vix_max=[None, 20, 25, 30],
    require_slope=[False, True],
    confirm_style=["sym", "asym"],
    confirm_up=[1, 3, 5],
    cb_vol_pct=[None, 0.95],
    cb_drop=[None, -0.07],
    cb_drop_n=[10],
    cb_vix=[None],
    crash_mom_thr=[None],
    crash_mom_n=[20],
)

if __name__ == "__main__":
    df, n = sweep("v2", SPEC)
    keys = ["trend_n", "prox_n", "prox_thr", "mom_n", "vol_max", "abs_vol_max",
            "vix_max", "require_slope", "confirm_style", "confirm_up", "cb_vol_pct", "cb_drop"]
    report(df, n, "STAGE 2 - v2 long side, ranked by IS Calmar", "IS_Calmar", 15, keys)
    ok = df[(df.ALL_CAGR > 0.30) & (df.ALL_maxDD > -0.35)]
    report(ok, n, "STAGE 2 - CAGR>30% and maxDD<35% over FULL sample", "ALL_Calmar", 25, keys)
    both = df[(df.IS_CAGR > 0.28) & (df.OOS_CAGR > 0.28) &
              (df.IS_maxDD > -0.35) & (df.OOS_maxDD > -0.35)]
    report(both, n, "STAGE 2 - robust in BOTH halves (CAGR>28%, DD<35%)", "ALL_Calmar", 25, keys)
    save(df, "sweep2_gates.csv")
