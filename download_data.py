"""
Download the tickers this project needs into ./json/, using the EODHD end-of-day API.

Every run downloads the FULL history for each ticker in one request and overwrites
the file. Never append/merge incremental pulls: EODHD's adjusted_close is restated
on every split, so welding an old file to a new tail fabricates split-sized bars.

Requires EODHD_API_TOKEN in the environment (it's in ~/.bash_profile).

Traded:   XLK (1x tech), TECL (3x tech long), BIL (T-bills)
Signals:  QQQ, SPY, TLT, HYG, and the VIX / VXN / IRX indices

TECS is deliberately not downloaded. The strategy never holds it (see final.py),
and on 2026-08-18 EODHD began serving a corrupted history for it -- ~4,400 bars
flattened to a 0.2 close -- which would overwrite the good json/TECS.json kept
for the archived TECS research. Add it to EQUITIES only if that feed is fixed
AND the short sleeve is being reconsidered.
"""

import json
import os
import sys
import time

import requests

EQUITIES = ["XLK", "TECL", "BIL", "QQQ", "SPY", "TLT", "HYG", "SHY"]
INDICES = ["VIX", "VXN", "IRX"]  # EODHD serves these on the INDX exchange
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
START = sys.argv[1] if len(sys.argv) > 1 else "1999-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else None  # None = latest

TOKEN = os.environ.get("EODHD_API_TOKEN")
EXPECTED = {"date", "open", "high", "low", "close", "adjusted_close", "volume"}


def fetch(symbol):
    params = {"api_token": TOKEN, "fmt": "json", "from": START, "period": "d"}
    if END:
        params["to"] = END
    r = requests.get(f"https://eodhd.com/api/eod/{symbol}", params=params, timeout=90)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"unexpected response: {str(data)[:150]}")
    missing = EXPECTED - set(data[0])
    if missing:
        raise RuntimeError(f"missing fields {missing}")
    return data


def main():
    if not TOKEN:
        sys.exit("ERROR: EODHD_API_TOKEN not set (source ~/.bash_profile).")
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(t, f"{t}.US", t) for t in EQUITIES] + [(t, f"{t}.INDX", t) for t in INDICES]
    print(f"Downloading {len(jobs)} symbols -> {OUT_DIR} (from {START} to {END or 'latest'})\n")
    failures = []
    for name, symbol, fname in jobs:
        try:
            rec = fetch(symbol)
        except Exception as e:
            print(f"  {name:<6} FAILED: {e}")
            failures.append(name)
            continue
        with open(os.path.join(OUT_DIR, f"{fname}.json"), "w") as f:
            json.dump(rec, f)
        print(f"  {name:<6} {len(rec):>6} rows  {rec[0]['date']} -> {rec[-1]['date']}")
        time.sleep(0.2)
    print("\nDone." if not failures else f"\nDone with failures: {failures}")


if __name__ == "__main__":
    main()
