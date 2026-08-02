"""Buy-and-hold baselines over the common TECL/TECS window, for reference."""

from data import load_all
from engine import buy_hold, fmt

prices, returns = load_all()
cash = returns["BIL"]
START = "2009-01-02"  # first full day all of XLK/TECL/TECS/BIL trade

rows = [buy_hold(returns, t, cash, start=START) for t in ["XLK", "TECL", "TECS", "BIL", "QQQ", "SPY"]]
print(f"Buy & hold, {START} -> {returns.index[-1].date()}\n")
print(fmt(rows))

print("\nSame, by sub-period (XLK / TECL):")
for a, b in [("2009-01-02", "2014-12-31"), ("2015-01-01", "2019-12-31"),
             ("2020-01-01", "2022-12-31"), ("2023-01-01", "2026-07-31")]:
    sub = [buy_hold(returns, t, cash, start=a, end=b) for t in ["XLK", "TECL"]]
    for m in sub:
        m["label"] = f"{m['label']}  {a[:4]}-{b[:4]}"
    print(fmt(sub))
