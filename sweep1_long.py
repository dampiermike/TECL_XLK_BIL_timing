"""Stage 1: the long side only (XLK / TECL / BIL). TECS gate disabled."""

from sweep import sweep, report, save

SPEC = dict(
    trend_n=[100, 150, 200, 250],
    trend_buffer=[0.0, 0.02],
    prox_n=[20, 40, 60, 90, 120, 250],
    prox_thr=[0.93, 0.95, 0.97, 0.99],
    mom_n=[10, 20, 60, 120],
    mom_thr=[0.0],
    vol_n=[20],
    vol_max=[0.5, 0.7, 0.9, None],
    confirm=[1, 3, 5],
    # TECS switched off for this stage
    crash_mom_n=[20], crash_mom_thr=[-99.0], crash_vol_min=[None],
    crash_dd_n=[60], crash_dd=[None],
)

if __name__ == "__main__":
    df, n = sweep("hier", SPEC)
    keys = ["trend_n", "trend_buffer", "prox_n", "prox_thr", "mom_n", "vol_max", "confirm"]
    report(df, n, "STAGE 1 - long side, ranked by IS Calmar", "IS_Calmar", 15, keys)
    report(df, n, "STAGE 1 - long side, ranked by IS Sharpe", "IS_Sharpe", 15, keys)
    ok = df[(df.IS_CAGR > 0.30) & (df.IS_maxDD > -0.30)]
    report(ok, n, "STAGE 1 - meeting 30/30 IN-SAMPLE", "IS_Calmar", 20, keys)
    save(df, "sweep1_long.csv")
