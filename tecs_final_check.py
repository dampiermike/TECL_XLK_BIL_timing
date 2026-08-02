"""
Re-test the TECS sleeve against the FINAL (stronger) core.

Stage 3 rejected TECS on the weaker stage-2 core. That is not sufficient -- a
better long engine leaves different residual risk, so the crash sleeve deserves
a second, independent hearing on the core that was actually chosen.
"""

import pandas as pd

from sweep import sweep, report
from final import PARAMS

CORE = {k: [v] for k, v in PARAMS.items() if k != "crash_mom_thr"}

SPEC = dict(
    CORE,
    crash_mom_n=[10, 20, 60],
    crash_mom_thr=[None, -0.03, -0.06, -0.10, -0.15],
    crash_below_n=[None, 50, 200],
    crash_vol_min=[None, 0.60, 0.85],
    crash_dd_n=[120], crash_dd=[None, -0.15],
    crash_vix_min=[None, 25, 35],
    crash_slope_n=[None, 100],
    crash_confirm=[1, 3, 5],
)

KEYS = ["crash_mom_n", "crash_mom_thr", "crash_below_n", "crash_vol_min",
        "crash_vix_min", "crash_slope_n", "crash_confirm"]

if __name__ == "__main__":
    df, n = sweep("v2vol", SPEC)
    off = df[df.crash_mom_thr.isna()]
    on = df[df.crash_mom_thr.notna()]
    base = off.iloc[0] if len(off) else None

    if base is not None:
        print(f"\nFINAL core with TECS OFF:  CAGR {base.ALL_CAGR:.1%}  "
              f"maxDD {base.ALL_maxDD:.1%}  Sharpe {base.ALL_Sharpe:.2f}")

    print(f"\nTECS-enabled configs: {len(on)}")
    if "pct_TECS" in on:
        used = on[on.pct_TECS.fillna(0) > 0.002]
        print(f"  ...that actually took a short position: {len(used)}")
        if base is not None and len(used):
            better_both = used[(used.ALL_CAGR > base.ALL_CAGR) &
                               (used.ALL_maxDD > base.ALL_maxDD)]
            better_sharpe = used[used.ALL_Sharpe > base.ALL_Sharpe]
            print(f"  ...that beat TECS-OFF on BOTH CAGR and maxDD: {len(better_both)}")
            print(f"  ...that beat TECS-OFF on Sharpe:              {len(better_sharpe)}")

            bins = [0.002, 0.005, 0.01, 0.02, 0.05, 1.0]
            used = used.copy()
            used["bucket"] = pd.cut(used.pct_TECS, bins=bins)
            print("\n  by realised TECS exposure:")
            print(used.groupby("bucket", observed=True).agg(
                n=("ALL_CAGR", "size"), CAGR=("ALL_CAGR", "mean"),
                maxDD=("ALL_maxDD", "mean"), Sharpe=("ALL_Sharpe", "mean"),
                best_Sharpe=("ALL_Sharpe", "max")).to_string(
                    float_format=lambda v: f"{v:,.3f}"))

    report(on, n, "TECS-enabled, best full-sample Sharpe", "ALL_Sharpe", 10, KEYS)
