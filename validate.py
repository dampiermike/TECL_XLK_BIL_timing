"""
Robustness testing for the selected config.

The config was chosen out of ~75,000 evaluations, so the only question that
matters is whether it survives conditions it was not selected under:
  1. higher trading costs
  2. slower execution (signal acted on a day later)
  3. one-at-a-time parameter perturbation (is it a peak or a plateau?)
  4. rolling 3-year windows (is the record continuous or one lucky stretch?)
  5. a walk-forward re-fit that never sees the future
  6. a different signal ticker (QQQ) and a start-date shift
"""

import numpy as np
import pandas as pd

from data import load_all
from engine import run_weights, metrics, buy_hold, fmt
from strategy import build_features
import strategy2
from final import PARAMS, START

prices, returns = load_all()
cash = returns["BIL"]
F = {t: build_features(prices, t) for t in ("XLK", "QQQ")}


def evaluate(params, signal="XLK", cost_bps=10.0, lag=0, start=START, end=None, label=""):
    w = strategy2.vol_target_weights(F[signal], params, returns)
    if lag:
        w = w.shift(lag).dropna(how="all")
    bt = run_weights(returns, w, cost_bps=cost_bps, start=start, end=end)
    return metrics(bt, cash, label=label), bt


print("=" * 92)
print("BASELINE (as selected)")
print("=" * 92)
base, bt = evaluate(PARAMS, label="FINAL 80% TECL")
rows = [buy_hold(returns, t, cash, start=START) for t in ("XLK", "TECL")] + [base]
print(fmt(rows))

print("\n" + "=" * 92)
print("1. COST SENSITIVITY   (selected at 10 bps per side)")
print("=" * 92)
rows = [evaluate(PARAMS, cost_bps=c, label=f"{c:.0f} bps/side")[0] for c in (0, 5, 10, 20, 40, 75)]
print(fmt(rows))

print("\n" + "=" * 92)
print("2. EXECUTION LAG   (0 = trade at next close, as designed)")
print("=" * 92)
rows = [evaluate(PARAMS, lag=l, label=f"lag +{l} day(s)")[0] for l in (0, 1, 2, 3)]
print(fmt(rows))

print("\n" + "=" * 92)
print("3. PARAMETER PERTURBATION   (one at a time; is this a peak or a plateau?)")
print("=" * 92)
PERTURB = {
    "trend_n": [80, 90, 100, 110, 125, 150],
    "prox_n": [40, 50, 60, 75, 90],
    "prox_thr": [0.94, 0.95, 0.96, 0.97, 0.98],
    "mom_n": [30, 45, 60, 90, 120],
    "vol_max": [0.6, 0.65, 0.70, 0.75, 0.8],
    "rsi_max": [70, 75, 80, 85, None],
    "confirm_up": [3, 4, 5, 6, 8],
    "cb_vol_pct": [0.85, 0.88, 0.90, 0.93, 0.95],
    "tecl_w": [0.70, 0.75, 0.80, 0.85, 0.90],
}
for k, vals in PERTURB.items():
    rows = []
    for v in vals:
        p = dict(PARAMS, **{k: v})
        m, _ = evaluate(p, label=f"{k}={v}{'  <-- chosen' if v == PARAMS[k] else ''}")
        rows.append(m)
    print(fmt(rows))
    cg = [r["CAGR"] for r in rows]
    dd = [r["maxDD"] for r in rows]
    n_ok = sum(1 for a, b in zip(cg, dd) if a >= 0.30 and b >= -0.30)
    print(f"   -> {n_ok}/{len(rows)} neighbours still meet 30/30\n")

print("=" * 92)
print("4. ROLLING 3-YEAR WINDOWS")
print("=" * 92)
eq = bt["equity"]
yrs = sorted(set(bt.index.year))
print(f"{'window':<14}{'CAGR':>9}{'maxDD':>9}{'vs XLK':>10}")
for y in yrs[:-2]:
    a, b = f"{y}-01-01", f"{y+2}-12-31"
    seg = bt.loc[a:b]
    if len(seg) < 400:
        continue
    e = (1 + seg["ret"]).cumprod()
    c = e.iloc[-1] ** (252 / len(seg)) - 1
    d = (e / e.cummax() - 1).min()
    bh = buy_hold(returns, "XLK", cash, start=a, end=b)
    print(f"{y}-{y+2:<9}{c:>8.1%}{d:>9.1%}{c - bh['CAGR']:>+10.1%}")

print("\ncalendar years:")
yr = bt["ret"].groupby(bt.index.year).apply(lambda s: (1 + s).prod() - 1)
ydd = bt["equity"].groupby(bt.index.year).apply(lambda s: (s / s.cummax() - 1).min())
xlkyr = returns["XLK"].reindex(bt.index).groupby(bt.index.year).apply(lambda s: (1 + s).prod() - 1)
for y in yr.index:
    print(f"  {y}  strat={yr[y]:+7.1%}  intra-yr DD={ydd[y]:+6.1%}   XLK={xlkyr[y]:+7.1%}")

print("\n" + "=" * 92)
print("5. WALK-FORWARD   (re-pick tecl_w each year using only prior data)")
print("=" * 92)
grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pieces, picks = [], []
for y in range(2013, 2027):
    tr_end = f"{y-1}-12-31"
    best, best_w = -9e9, None
    for wgt in grid:
        m, _ = evaluate(dict(PARAMS, tecl_w=wgt), start=START, end=tr_end)
        if not m:
            continue
        score = m["CAGR"] / abs(m["maxDD"]) if m["maxDD"] < 0 else -9e9
        if m["maxDD"] < -0.30:
            score -= 1.0                      # penalise breaching the DD limit in training
        if score > best:
            best, best_w = score, wgt
    m_oos, bt_oos = evaluate(dict(PARAMS, tecl_w=best_w), start=f"{y}-01-01", end=f"{y}-12-31")
    picks.append((y, best_w, m_oos["CAGR"], m_oos["maxDD"]))
    pieces.append(bt_oos["ret"])
    print(f"  {y}  trained through {tr_end}  picked tecl_w={best_w:.2f}  "
          f"-> live {m_oos['CAGR']:+7.1%}  DD {m_oos['maxDD']:+6.1%}")

wf = pd.concat(pieces).sort_index()
we = (1 + wf).cumprod()
wf_cagr = we.iloc[-1] ** (252 / len(we)) - 1
wf_dd = (we / we.cummax() - 1).min()
ex = wf - cash.reindex(wf.index).fillna(0)
print(f"\n  stitched walk-forward 2013-2026: CAGR {wf_cagr:.1%}  maxDD {wf_dd:.1%}  "
      f"Sharpe {ex.mean()/ex.std()*np.sqrt(252):.2f}")

print("\n" + "=" * 92)
print("6. SIGNAL TICKER AND START DATE")
print("=" * 92)
rows = [evaluate(PARAMS, signal=s, label=f"signal={s}")[0] for s in ("XLK", "QQQ")]
for s in ("2010-01-04", "2011-01-03", "2012-01-03", "2013-01-02", "2015-01-02"):
    rows.append(evaluate(PARAMS, start=s, label=f"start {s[:4]}")[0])
print(fmt(rows))
