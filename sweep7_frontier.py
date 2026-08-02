"""
Stage 7: the CAGR / drawdown frontier.

Every rule shape tried so far tops out near Calmar ~1.0-1.06, so where a config
lands on the CAGR-vs-DD line is set mostly by how much TECL it carries. This
stage fixes the best cores from stages 5 and 6 and sweeps the sizing finely, to
find configurations that actually clear 30% CAGR with drawdown inside 30% --
and to show how much margin each one has rather than picking a knife-edge fit.
"""

import pandas as pd

from sweep import sweep, report, save

W = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50 .. 1.00

# --- core A: stage-5 AND-chain gate ------------------------------------------
A = dict(
    trend_n=[100], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[60], mom_thr=[0.0], vol_n=[20], vol_max=[0.7],
    abs_vol_max=[None], vix_max=[None], require_slope=[True],
    confirm_style=["sym"], confirm_up=[5],
    cb_vol_pct=[0.90, 0.95], cb_drop=[None], cb_drop_n=[10], cb_vix=[None],
    crash_mom_thr=[None], er_min=[None], er_cash=[None],
    rsi_n=[14], rsi_max=[75], macd_pos=[False],
    sizing=["fixed"], target_vol=[None], max_leverage=[1.0], min_weight=[0.0],
    tecl_w=W, xlk_w=[0.7, 1.0],
)

# --- core B: stage-6 voting gate ---------------------------------------------
B = dict(
    trend_n=[100], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[60], mom_thr=[0.0], vol_n=[20], vol_max_v=[0.7],
    v_xlk=[0.45, 0.65], v_tecl=[0.70], use_credit=[True],
    rsi_n=[14], rsi_max=[75],
    confirm_style=["sym"], confirm_up=[3, 5],
    cb_vol_pct=[0.90, 0.95], cb_drop=[None, -0.10], cb_drop_n=[10], cb_vix=[None],
    crash_mom_thr=[None],
    tecl_w=W, xlk_w=[0.7, 1.0],
)

TARGET = "CAGR >= 30% and maxDD <= 30%"


def check(df, name):
    hit = df[(df.ALL_CAGR >= 0.30) & (df.ALL_maxDD >= -0.30)]
    both = hit[(hit.IS_CAGR >= 0.25) & (hit.OOS_CAGR >= 0.25) &
               (hit.IS_maxDD >= -0.32) & (hit.OOS_maxDD >= -0.32)]
    print(f"\n[{name}] full-sample 30/30 hits: {len(hit)} | also robust in both halves: {len(both)}")
    return hit, both


if __name__ == "__main__":
    keys_a = ["cb_vol_pct", "tecl_w", "xlk_w"]
    keys_b = ["v_xlk", "confirm_up", "cb_vol_pct", "cb_drop", "tecl_w", "xlk_w"]

    da, na = sweep("v2vol", A)
    da["core"] = "A-andchain"
    report(da, na, "FRONTIER core A - by full-sample Calmar", "ALL_Calmar", 10, keys_a)
    ha, ba = check(da, "A")
    report(ha, na, f"core A - {TARGET}", "ALL_CAGR", 20, keys_a)

    db, nb = sweep("vote", B)
    db["core"] = "B-vote"
    report(db, nb, "FRONTIER core B - by full-sample Calmar", "ALL_Calmar", 10, keys_b)
    hb, bb = check(db, "B")
    report(hb, nb, f"core B - {TARGET}", "ALL_CAGR", 20, keys_b)

    both = pd.concat([ba, bb], ignore_index=True)
    if len(both):
        report(both, na + nb, f"BOTH CORES - {TARGET} and robust in each half",
               "ALL_CAGR", 30, ["core"] + keys_b)
    save(pd.concat([da, db], ignore_index=True), "sweep7_frontier.csv")
