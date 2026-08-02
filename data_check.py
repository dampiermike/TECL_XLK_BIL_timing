"""
Validate the downloaded EODHD json before any backtest trusts it.

Checks, per ticker:
  1. duplicate / unsorted dates
  2. non-positive or NaN prices
  3. extreme adjusted_close jumps (split-fabrication tell)
  4. calendar gaps vs. the union trading calendar (holiday/stale rows)
  5. leverage sanity: TECL vs 3*XLK, TECS vs -3*XLK daily-return regression
"""

import json
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
TICKERS = ["XLK", "TECL", "TECS", "BIL", "QQQ", "SPY", "TLT", "HYG", "SHY", "VIX", "VXN", "IRX"]


def load(t):
    with open(os.path.join(DIR, f"{t}.json")) as f:
        df = pd.DataFrame(json.load(f))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def main():
    data = {t: load(t) for t in TICKERS}
    print(f"{'ticker':<7}{'rows':>7}{'start':>13}{'end':>13}{'dupes':>7}{'bad_px':>8}{'|jump|>40%':>12}")
    print("-" * 70)
    for t, df in data.items():
        px = df["adjusted_close"]
        dupes = int(df.index.duplicated().sum())
        bad = int((~np.isfinite(px)).sum() + (px <= 0).sum())
        r = px.pct_change()
        jumps = r[r.abs() > 0.40]
        print(f"{t:<7}{len(df):>7}{str(df.index[0].date()):>13}{str(df.index[-1].date()):>13}"
              f"{dupes:>7}{bad:>8}{len(jumps):>12}")
        for d, v in jumps.items():
            print(f"          !! {d.date()}  {v:+.1%}  close={px.loc[d]:.4f}")

    # calendar coverage against SPY (the reference US equity calendar)
    print("\ncalendar gaps vs SPY (equities only, over each ticker's own span):")
    cal = data["SPY"].index
    for t in ["XLK", "TECL", "TECS", "BIL", "QQQ", "TLT", "HYG", "SHY"]:
        idx = data[t].index
        span = cal[(cal >= idx[0]) & (cal <= idx[-1])]
        missing = span.difference(idx)
        extra = idx.difference(cal)
        print(f"  {t:<6} missing={len(missing):>4}  extra={len(extra):>4}"
              + (f"   first missing {missing[0].date()}" if len(missing) else ""))

    # leverage sanity on the overlap
    print("\nleverage check (daily returns, TECL/TECS vs XLK, 2009+):")
    xlk = data["XLK"]["adjusted_close"].pct_change()
    for t, target in [("TECL", 3.0), ("TECS", -3.0)]:
        lev = data[t]["adjusted_close"].pct_change()
        j = pd.concat([xlk, lev], axis=1, join="inner").dropna()
        j = j[j.index >= "2009-01-01"]
        j.columns = ["x", "y"]
        beta = np.polyfit(j["x"], j["y"], 1)[0]
        corr = j["x"].corr(j["y"])
        print(f"  {t:<6} beta={beta:+.3f} (expect {target:+.1f})  corr={corr:+.3f}  n={len(j)}")


if __name__ == "__main__":
    main()
