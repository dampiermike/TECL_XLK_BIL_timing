"""
Freshness gate for the downloaded EODHD json, run between download_data.py and
anything that trusts the files. Exits 0 if current, 1 (with detail) if stale, so
that run_daily.sh's failure trap fires instead of publishing a stale signal.

Two independent staleness modes are checked:

  1. RELATIVE — one ticker lagging the rest. The nine US equities share the NYSE
     calendar, so the newest bar among them is the expected last bar; any equity
     behind that is a partial update. Indices (VIX/VXN/IRX) settle later and are
     allowed one trading day of slack.

  2. ABSOLUTE — the whole feed frozen. There is no independent trading calendar
     here (every file comes from the same vendor), so the test is the number of
     weekdays strictly between the last bar and today: 0 is normal, 1 is allowed
     because it is indistinguishable from a market holiday, 2+ is stale.

Note the second test cannot tell a one-day outage from a holiday. That is
tolerable: a stale feed reproduces yesterday's signal and commits nothing.

Usage:
    python3 validate_freshness.py
    python3 validate_freshness.py --today 2026-08-04   # for testing
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")

# TECS is absent on purpose: not downloaded, not traded. See download_data.py.
EQUITIES = ["XLK", "TECL", "BIL", "QQQ", "SPY", "TLT", "HYG", "SHY"]
INDICES = ["VIX", "VXN", "IRX"]

INDEX_SLACK = 1      # trading days an index may lag the equity consensus
MAX_WEEKDAY_GAP = 1  # weekdays allowed between the last bar and today (holiday)


def last_date(ticker):
    """Most recent bar date in json/<ticker>.json, or None if unreadable."""
    path = os.path.join(DIR, f"{ticker}.json")
    try:
        with open(path) as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    return date.fromisoformat(max(r["date"] for r in rows))


def weekdays_between(start, end):
    """Weekdays strictly after `start` and strictly before `end`."""
    n, d = 0, start + timedelta(days=1)
    while d < end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--today", help="Override today's date (YYYY-MM-DD).")
    args = ap.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    dates = {t: last_date(t) for t in EQUITIES + INDICES}

    missing = [t for t, d in dates.items() if d is None]
    equity_dates = [dates[t] for t in EQUITIES if dates[t] is not None]
    if not equity_dates:
        print("  FRESHNESS FAIL - no readable equity files", file=sys.stderr)
        return 1
    expected = max(equity_dates)

    print("validate_freshness")
    print(f"  json dir:     {DIR}")
    print(f"  expected bar: {expected}  (newest across {len(EQUITIES)} US equities)")
    print(f"  today:        {today}\n")
    print(f"  {'ticker':<8}{'last bar':<13}{'weekdays behind':>16}   status")
    print(f"  {'-' * 46}")

    fails = []
    for t in EQUITIES + INDICES:
        d = dates[t]
        if d is None:
            print(f"  {t:<8}{'(unreadable)':<13}{'?':>16}   FAIL")
            fails.append((t, "missing or unparseable file"))
            continue
        behind = weekdays_between(d, expected) + (1 if d < expected else 0)
        slack = INDEX_SLACK if t in INDICES else 0
        ok = behind <= slack
        print(f"  {t:<8}{str(d):<13}{behind:>16}   {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append((t, f"{behind} weekday(s) behind {expected}, slack {slack}"))

    gap = weekdays_between(expected, today)
    if gap > MAX_WEEKDAY_GAP:
        fails.append(("<feed>", f"newest bar {expected} is {gap} weekdays before {today}"))
    elif gap == MAX_WEEKDAY_GAP:
        print(f"\n  WARNING: {gap} weekday between {expected} and {today} - "
              f"market holiday, or the feed is one session behind.")

    if fails:
        print(f"\n  FRESHNESS FAIL - {len(fails)} problem(s):", file=sys.stderr)
        for name, why in fails:
            print(f"    {name}: {why}", file=sys.stderr)
        return 1

    print(f"\n  all {len(dates) - len(missing)} files current through {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
