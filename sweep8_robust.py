"""
Stage 8: robustness-aware selection.

Validation showed the stage-7 pick sits on a spike -- confirm_up=5 gives -29% DD
while 4 and 6 both give about -35%. A parameter set that is only good at an exact
value is a fitting artifact, not a strategy.

So this stage stops ranking by a config's own score. It builds a dense grid,
then scores each config by its NEIGHBOURHOOD: the configs one grid step away in
each dimension. A config wins only if the whole region around it works, which is
what "this will still work next year" actually requires.
"""

import itertools

import numpy as np
import pandas as pd

from sweep import sweep, save

AXES = {
    "trend_n": [90, 100, 110, 125],
    "prox_thr": [0.94, 0.95, 0.96, 0.97],
    "mom_n": [45, 60, 90],
    "vol_max": [0.65, 0.70, 0.75, 0.80],
    "rsi_max": [75, 80, None],
    "confirm_up": [4, 5, 6, 7],
    "cb_vol_pct": [0.85, 0.88, 0.90, 0.93],
    "tecl_w": [0.70, 0.75, 0.80, 0.85],
}

SPEC = dict(
    {k: v for k, v in AXES.items()},
    trend_buffer=[0.0], prox_n=[60], mom_thr=[0.0], vol_n=[20],
    abs_vol_max=[None], vix_max=[None], require_slope=[True],
    er_min=[None], er_cash=[None], rsi_n=[14], macd_pos=[False],
    confirm_style=["sym"], cb_drop=[None], cb_drop_n=[10], cb_vix=[None],
    crash_mom_thr=[None],
    sizing=["fixed"], target_vol=[None], max_leverage=[1.0], min_weight=[0.0],
    xlk_w=[1.0],
)


def neighbourhood_scores(df):
    """For each config, aggregate the metrics of configs one grid step away."""
    keys = list(AXES)
    idx = {}
    for i, row in df.iterrows():
        idx[tuple(row[k] if pd.notna(row[k]) else None for k in keys)] = i

    pos = {k: {v: j for j, v in enumerate(AXES[k])} for k in keys}
    rob = []
    for i, row in df.iterrows():
        key = tuple(row[k] if pd.notna(row[k]) else None for k in keys)
        neigh = [i]
        for k in keys:
            j = pos[k].get(key[keys.index(k)])
            if j is None:
                continue
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(AXES[k]):
                    alt = list(key)
                    alt[keys.index(k)] = AXES[k][jj]
                    hit = idx.get(tuple(alt))
                    if hit is not None:
                        neigh.append(hit)
        sub = df.loc[neigh]
        rob.append({
            "n_neigh": len(neigh),
            "nb_CAGR_mean": sub.ALL_CAGR.mean(),
            "nb_CAGR_min": sub.ALL_CAGR.min(),
            "nb_maxDD_mean": sub.ALL_maxDD.mean(),
            "nb_maxDD_worst": sub.ALL_maxDD.min(),
            "nb_Sharpe_mean": sub.ALL_Sharpe.mean(),
            "nb_frac_hit": float(((sub.ALL_CAGR >= 0.30) & (sub.ALL_maxDD >= -0.30)).mean()),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rob)], axis=1)


if __name__ == "__main__":
    df, n = sweep("v2vol", SPEC)
    print(f"evaluated {len(df)} of {n} combos")
    r = neighbourhood_scores(df)

    print("\n" + "=" * 110)
    print("Configs whose ENTIRE NEIGHBOURHOOD averages 30%+ CAGR and DD inside 30%")
    print("=" * 110)
    good = r[(r.nb_CAGR_mean >= 0.30) & (r.nb_maxDD_mean >= -0.30) & (r.n_neigh >= 10)]
    good = good.sort_values("nb_frac_hit", ascending=False)
    cols = list(AXES) + ["ALL_CAGR", "ALL_maxDD", "ALL_Sharpe", "nb_CAGR_mean",
                         "nb_CAGR_min", "nb_maxDD_mean", "nb_maxDD_worst", "nb_frac_hit", "n_neigh"]
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(good[cols].head(25).to_string(index=False) if len(good) else "none")

    print("\n" + "=" * 110)
    print("Most robust overall (highest share of neighbours meeting 30/30)")
    print("=" * 110)
    top = r[r.n_neigh >= 10].sort_values(["nb_frac_hit", "nb_CAGR_mean"], ascending=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(top[cols].head(20).to_string(index=False))
    save(r, "sweep8_robust.csv")
