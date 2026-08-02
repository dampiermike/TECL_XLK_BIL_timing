"""
Head-to-head on the robustness-aware finalists from stage 8.

Each is tested on the things it was NOT selected for: costs, execution lag,
walk-forward, the post-2009 subsample, and the confirm_up neighbourhood that
exposed the stage-7 pick as a spike.
"""

import numpy as np
import pandas as pd

from data import load_all
from engine import run_weights, metrics, buy_hold, fmt
from strategy import build_features
import strategy2

BASE = dict(
    trend_buffer=0.0, prox_n=60, mom_thr=0.0, vol_n=20,
    abs_vol_max=None, vix_max=None, require_slope=True,
    er_min=None, er_cash=None, rsi_n=14, macd_pos=False,
    confirm_style="sym", cb_drop=None, cb_drop_n=10, cb_vix=None,
    crash_mom_thr=None,
    sizing="fixed", target_vol=None, max_leverage=1.0, min_weight=0.0, xlk_w=1.0,
)

FINALISTS = {
    "S7 peak (old)": dict(BASE, trend_n=100, prox_thr=0.96, mom_n=60, vol_max=0.70,
                          rsi_max=75, confirm_up=5, cb_vol_pct=0.90, tecl_w=0.80),
    "R1 widest plateau": dict(BASE, trend_n=100, prox_thr=0.95, mom_n=60, vol_max=0.65,
                              rsi_max=75, confirm_up=5, cb_vol_pct=0.85, tecl_w=0.75),
    "R2 best Sharpe": dict(BASE, trend_n=100, prox_thr=0.96, mom_n=60, vol_max=0.75,
                           rsi_max=75, confirm_up=5, cb_vol_pct=0.85, tecl_w=0.75),
    "R3 lowest DD": dict(BASE, trend_n=90, prox_thr=0.96, mom_n=60, vol_max=0.75,
                         rsi_max=75, confirm_up=5, cb_vol_pct=0.85, tecl_w=0.70),
    "R4 R3 sized up": dict(BASE, trend_n=90, prox_thr=0.96, mom_n=60, vol_max=0.75,
                           rsi_max=75, confirm_up=5, cb_vol_pct=0.85, tecl_w=0.75),
}

START = "2009-01-02"
prices, returns = load_all()
cash = returns["BIL"]
F = build_features(prices, "XLK")


def ev(p, cost_bps=10.0, lag=0, start=START, end=None, label=""):
    w = strategy2.vol_target_weights(F, p, returns)
    if lag:
        w = w.shift(lag).dropna(how="all")
    bt = run_weights(returns, w, cost_bps=cost_bps, start=start, end=end)
    return metrics(bt, cash, label=label), bt


print("=" * 92)
print("FULL SAMPLE 2009-2026  (10 bps/side)")
print("=" * 92)
print(fmt([buy_hold(returns, t, cash, start=START) for t in ("XLK", "TECL")] +
          [ev(p, label=k)[0] for k, p in FINALISTS.items()]))

print("\n" + "=" * 92)
print("EXCLUDING 2009  (the +149% rebound year the full-sample CAGR leans on)")
print("=" * 92)
print(fmt([buy_hold(returns, "XLK", cash, start="2010-01-04")] +
          [ev(p, start="2010-01-04", label=k)[0] for k, p in FINALISTS.items()]))

print("\n" + "=" * 92)
print("COST AND LAG STRESS   (CAGR / maxDD)")
print("=" * 92)
hdr = f"{'config':<20}{'20bps':>16}{'40bps':>16}{'lag+1':>16}{'lag+2':>16}"
print(hdr + "\n" + "-" * len(hdr))
for k, p in FINALISTS.items():
    cells = []
    for kw in (dict(cost_bps=20), dict(cost_bps=40), dict(lag=1), dict(lag=2)):
        m, _ = ev(p, **kw)
        cells.append(f"{m['CAGR']:.1%}/{m['maxDD']:.0%}")
    print(f"{k:<20}" + "".join(f"{c:>16}" for c in cells))

print("\n" + "=" * 92)
print("confirm_up NEIGHBOURHOOD   (spike test -- CAGR / maxDD)")
print("=" * 92)
hdr = f"{'config':<20}" + "".join(f"{'cu=' + str(c):>15}" for c in (3, 4, 5, 6, 7, 8))
print(hdr + "\n" + "-" * len(hdr))
for k, p in FINALISTS.items():
    cells = []
    for c in (3, 4, 5, 6, 7, 8):
        m, _ = ev(dict(p, confirm_up=c))
        cells.append(f"{m['CAGR']:.0%}/{m['maxDD']:.0%}")
    print(f"{k:<20}" + "".join(f"{c:>15}" for c in cells))

print("\n" + "=" * 92)
print("WALK-FORWARD 2013-2026   (tecl_w re-picked each year on prior data only)")
print("=" * 92)
grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
for k, p in FINALISTS.items():
    pieces = []
    for y in range(2013, 2027):
        best, best_w = -9e9, None
        for wgt in grid:
            m, _ = ev(dict(p, tecl_w=wgt), start=START, end=f"{y-1}-12-31")
            if not m:
                continue
            s = m["CAGR"] / abs(m["maxDD"]) - (1.0 if m["maxDD"] < -0.30 else 0.0)
            if s > best:
                best, best_w = s, wgt
        _, bt = ev(dict(p, tecl_w=best_w), start=f"{y}-01-01", end=f"{y}-12-31")
        pieces.append(bt["ret"])
    wf = pd.concat(pieces).sort_index()
    we = (1 + wf).cumprod()
    c = we.iloc[-1] ** (252 / len(we)) - 1
    d = (we / we.cummax() - 1).min()
    ex = wf - cash.reindex(wf.index).fillna(0)
    print(f"  {k:<20} CAGR {c:>6.1%}   maxDD {d:>6.1%}   Sharpe {ex.mean()/ex.std()*np.sqrt(252):>4.2f}")
