"""Where does the drawdown actually come from? Worst episodes + what was held."""

import pandas as pd

from data import load_all
from engine import run, metrics
from strategy import build_features, make_target

prices, returns = load_all()
cash = returns["BIL"]
f = build_features(prices, "XLK")

P = dict(trend_n=100, trend_buffer=0.0, prox_n=20, prox_thr=0.93, mom_n=20, mom_thr=0.0,
         vol_n=20, vol_max=0.70, confirm=3,
         crash_mom_n=20, crash_mom_thr=-99.0, crash_vol_min=None, crash_dd_n=60, crash_dd=None)

tgt = make_target(f, "hier", P)
bt = run(returns, tgt, start="2009-01-02")
m = metrics(bt, cash, "stage1 best")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

eq = bt["equity"]
dd = eq / eq.cummax() - 1

# identify separate drawdown episodes deeper than 15%
under = dd < -0.001
grp = (under != under.shift()).cumsum()
eps = []
for g, seg in dd.groupby(grp):
    if not under.loc[seg.index[0]]:
        continue
    trough = seg.idxmin()
    if seg.min() > -0.15:
        continue
    eps.append((seg.index[0].date(), trough.date(), seg.index[-1].date(), seg.min(),
                (trough - seg.index[0]).days, (seg.index[-1] - trough).days))
eps.sort(key=lambda e: e[3])
print(f"\n{'start':>12}{'trough':>12}{'end':>12}{'depth':>8}{'d_down':>8}{'d_rec':>7}   holdings during decline")
for s, t, e, d, dn, rc in eps[:10]:
    mix = bt.loc[str(s):str(t), "target"].value_counts(normalize=True)
    mixs = " ".join(f"{k}:{v:.0%}" for k, v in mix.items())
    print(f"{str(s):>12}{str(t):>12}{str(e):>12}{d:>7.1%}{dn:>8}{rc:>7}   {mixs}")

print("\nworst 15 single days of the strategy:")
w = bt["ret"].nsmallest(15)
for d, v in w.items():
    prev = bt["target"].shift(1).loc[d]
    print(f"  {d.date()}  {v:+.2%}  held={prev}  XLK={returns['XLK'].loc[d]:+.2%}")

print("\ncalendar-year returns:")
yr = bt["ret"].groupby(bt.index.year).apply(lambda s: (1 + s).prod() - 1)
ydd = bt["equity"].groupby(bt.index.year).apply(lambda s: (s / s.cummax() - 1).min())
mixy = bt.groupby(bt.index.year)["target"].apply(lambda s: " ".join(
    f"{k}:{v:.0%}" for k, v in s.value_counts(normalize=True).items()))
for y in yr.index:
    print(f"  {y}  ret={yr[y]:+7.1%}  intra-yr DD={ydd[y]:+6.1%}   {mixy[y]}")
