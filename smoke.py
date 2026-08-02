"""Smoke test: a few hand-picked parameter sets, before any sweeping."""

from data import load_all
from engine import run, metrics, buy_hold, fmt
from strategy import build_features, make_target

prices, returns = load_all()
cash = returns["BIL"]
f = build_features(prices, "XLK")
START = "2009-01-02"

CASES = [
    ("hier baseline", "hier", dict(
        trend_n=200, prox_n=60, prox_thr=0.97, mom_n=20, mom_thr=0.0,
        vol_n=20, vol_max=0.60, crash_mom_n=20, crash_mom_thr=-0.05,
        crash_vol_min=0.70, crash_dd_n=60, crash_dd=-0.10, confirm=2)),
    ("hier looser breakout", "hier", dict(
        trend_n=200, prox_n=90, prox_thr=0.95, mom_n=60, mom_thr=0.0,
        vol_n=20, vol_max=0.80, crash_mom_n=20, crash_mom_thr=-0.08,
        crash_vol_min=0.80, crash_dd_n=60, crash_dd=-0.12, confirm=2)),
    ("hier no TECS", "hier", dict(
        trend_n=200, prox_n=60, prox_thr=0.97, mom_n=20, mom_thr=0.0,
        vol_n=20, vol_max=0.60, crash_mom_n=20, crash_mom_thr=-99.0,
        crash_vol_min=None, crash_dd_n=60, crash_dd=None, confirm=2)),
    ("dualma", "dualma", dict(
        fast_n=50, slow_n=200, vol_n=20, vol_max=0.60,
        crash_mom_n=20, crash_mom_thr=-0.06, confirm=2)),
    ("score", "score", dict(
        trend_n=200, mom_n=60, vol_n=20, w_trend=1.0, w_mom=1.0, w_calm=0.5,
        s_crash=-1.5, s_side=-0.2, s_break=1.0, confirm=2)),
]

rows = [buy_hold(returns, t, cash, start=START) for t in ["XLK", "TECL"]]
for label, kind, p in CASES:
    tgt = make_target(f, kind, p)
    bt = run(returns, tgt, start=START)
    m = metrics(bt, cash, label=label)
    rows.append(m)

print(f"Smoke test, {START} -> {returns.index[-1].date()}  (10bps/side)\n")
print(fmt(rows))
print("\nexposure mix:")
for m in rows[2:]:
    mix = {k[4:]: f"{v:.0%}" for k, v in m.items() if k.startswith("pct_")}
    print(f"  {m['label']:<24} {mix}  switches/yr={m['trades_per_yr']:.1f}")
