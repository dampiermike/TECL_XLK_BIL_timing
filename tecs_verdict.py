"""
Is the TECS sleeve worth having? Two independent tests.

A) Across the whole stage-3 grid, bucket every config by how much TECS it held
   and look at what happens to CAGR / maxDD / Sharpe as that exposure rises.
B) Take the identical long core and toggle ONLY the crash gate, so the TECS
   sleeve's contribution is isolated with nothing else moving.
"""

import numpy as np
import pandas as pd

from data import load_all
from engine import run, metrics, fmt
from strategy import build_features
import strategy2

df = pd.read_csv("results/sweep3_tecs.csv")
df["pct_TECS"] = df.get("pct_TECS", pd.Series(0.0, index=df.index)).fillna(0.0)

print("A) stage-3 grid bucketed by realised TECS exposure\n")
bins = [-1e-9, 1e-6, 0.005, 0.01, 0.02, 0.05, 1.0]
lbl = ["never", "<0.5%", "0.5-1%", "1-2%", "2-5%", ">5%"]
df["bucket"] = pd.cut(df["pct_TECS"], bins=bins, labels=lbl)
agg = df.groupby("bucket", observed=True).agg(
    n=("ALL_CAGR", "size"),
    CAGR=("ALL_CAGR", "mean"), best_CAGR=("ALL_CAGR", "max"),
    maxDD=("ALL_maxDD", "mean"), best_DD=("ALL_maxDD", "max"),
    Sharpe=("ALL_Sharpe", "mean"), best_Sharpe=("ALL_Sharpe", "max"))
print(agg.to_string(float_format=lambda v: f"{v:,.3f}"))

live = df[df.pct_TECS > 1e-6]
if len(live) > 10:
    print("\ncorrelation of TECS exposure with outcome (configs that did short):")
    for c in ["ALL_CAGR", "ALL_maxDD", "ALL_Sharpe", "ALL_Calmar"]:
        print(f"   {c:<12} r = {live['pct_TECS'].corr(live[c]):+.3f}")

print("\n\nB) same long core, crash gate toggled on/off\n")
prices, returns = load_all(include_tecs=True)   # this script exists to test TECS
cash = returns["BIL"]
f = build_features(prices, "XLK")

CORE = dict(trend_n=100, trend_buffer=0.0, prox_n=60, prox_thr=0.96, mom_n=60,
            mom_thr=0.0, vol_n=20, vol_max=0.70, abs_vol_max=None, vix_max=None,
            require_slope=True, confirm_style="sym", confirm_up=5,
            cb_vol_pct=0.95, cb_drop=None, cb_drop_n=10, cb_vix=None)

VARIANTS = [
    ("long only (no TECS)", dict(crash_mom_thr=None)),
    ("TECS: best-of-grid", dict(crash_mom_n=10, crash_mom_thr=-0.03, crash_below_n=None,
                                crash_vol_min=0.85, crash_dd=None, crash_vix_min=25,
                                crash_slope_n=100, crash_confirm=5)),
    ("TECS: moderate gate", dict(crash_mom_n=20, crash_mom_thr=-0.06, crash_below_n=200,
                                 crash_vol_min=0.60, crash_dd=None, crash_vix_min=None,
                                 crash_slope_n=100, crash_confirm=3)),
    ("TECS: loose gate", dict(crash_mom_n=20, crash_mom_thr=-0.03, crash_below_n=200,
                              crash_vol_min=None, crash_dd=None, crash_vix_min=None,
                              crash_slope_n=None, crash_confirm=1)),
    ("TECS: deep-crash only", dict(crash_mom_n=20, crash_mom_thr=-0.15, crash_below_n=200,
                                   crash_vol_min=0.85, crash_dd=-0.20, crash_vix_min=35,
                                   crash_slope_n=100, crash_confirm=3)),
]

rows = []
for label, extra in VARIANTS:
    p = dict(CORE, **extra)
    tgt = strategy2.make_target(f, p)
    bt = run(returns, tgt, start="2009-01-02")
    m = metrics(bt, cash, label=label)
    m["pct_TECS"] = m.get("pct_TECS", 0.0)
    rows.append(m)
print(fmt(rows))
print("\nTECS exposure and the P&L it generated:")
for label, extra in VARIANTS:
    p = dict(CORE, **extra)
    tgt = strategy2.make_target(f, p)
    bt = run(returns, tgt, start="2009-01-02")
    held = bt["target"].shift(1)
    days = int((held == "TECS").sum())
    if days == 0:
        print(f"  {label:<24} never held")
        continue
    pnl = (1 + bt.loc[held == "TECS", "ret"]).prod() - 1
    hit = float((bt.loc[held == "TECS", "ret"] > 0).mean())
    print(f"  {label:<24} {days:>4} days ({days/len(bt):.1%})  "
          f"compounded while short: {pnl:+.1%}  win rate {hit:.0%}")
