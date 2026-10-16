"""
The daily signal, written out the way the Nitro runner writes its own: the live
value of every indicator, then each gate with its threshold and whether it is
met, so the emailed signal can be audited without opening the repo.

Nothing here re-derives the strategy. Weights come from the same
strategy2.vol_target_weights call that export.py and plot_final.py use, so the
report cannot drift from the backtest; the gate breakdown below is a *display*
of the components that produced those weights.

  python3 signal_report.py              full report to stdout
  python3 signal_report.py --subject    one-line email subject, nothing else
"""

import sys

import numpy as np
import pandas as pd

from data import load_all
from engine import run_weights
from strategy import build_features
import strategy2
from final import PARAMS, START

TRADES = "results/trades.csv"
WIDTH = 70

MIX = {"TECL": "TECL 75% / BIL 25%", "XLK": "XLK 100%", "BIL": "BIL 100%"}


def build():
    prices, returns = load_all()
    f = build_features(prices, "XLK")
    w = strategy2.vol_target_weights(f, PARAMS, returns)
    bt = run_weights(returns, w, start=START)
    return f, w, bt


def state_of(row):
    """Dominant sleeve of a weight row. BIL 100% reads as BIL, not a TECL stub."""
    risk = {a: float(row.get(a, 0.0) or 0.0) for a in ("TECL", "XLK")}
    return max(risk, key=risk.get) if max(risk.values()) > 0 else "BIL"


def collect(f, w, bt):
    """Everything the report prints, resolved to the last completed session."""
    d = w.index[-1]
    x = f.loc[d]
    p = PARAMS

    lvl = strategy2.long_ladder(f, p)
    raw = strategy2.INV[int(lvl.loc[d])]
    # how many consecutive sessions the raw ladder has read the same level --
    # this is exactly what the 5-day confirmation is counting
    s = lvl.dropna()
    run = int(s.groupby((s != s.shift(1)).cumsum()).cumcount().loc[d]) + 1

    # w is shifted into force the next day, so yesterday's signal is both what is
    # held right now and the thing today's signal would be changing away from
    sig = state_of(w.loc[d])
    held = prev = state_of(w.iloc[-2])

    last = bt.loc[d]
    eq = bt["equity"]
    return dict(
        date=d, x=x, raw=raw, run=run, signal=sig, prev=prev, held=held,
        cb=bool(strategy2.circuit_breaker(f, p).loc[d]),
        ret=float(last["ret"]), equity=float(last["equity"]),
        dd=float(eq.loc[d] / eq.loc[:d].max() - 1),
    )


# ------------------------------------------------------------------ rendering
def report(f, w, bt):
    c = collect(f, w, bt)
    x, p = c["x"], PARAMS
    out, conds = [], []

    def rule(ch="-"):
        return ch * WIDTH

    def cond(ok, text, current):
        conds.append(f"     {'🟢' if ok else '🔴'} {text} (Current = {current})")

    def block(num, title, note=""):
        out.append(f"  {num}. {title}" + (f"  {note}" if note else ""))
        out.extend(conds)
        out.append("")
        conds.clear()

    sma = float(x[f"sma{p['trend_n']}"])
    px = float(x["px"])
    prox = float(x[f"prox{p['prox_n']}"])
    mom = float(x[f"roc{p['mom_n']}"])
    volp = float(x[f"volpct{p['vol_n']}"])
    vol = float(x[f"vol{p['vol_n']}"])
    slope = float(x[f"slope{p['trend_n']}"])
    rsi = float(x[f"rsi{p.get('rsi_n', 14)}"])
    day = str(c["date"].date())

    out += ["=" * WIDTH,
            f"===  TECL / XLK / BIL REGIME TIMING · DAILY SIGNAL · {day}  ===".center(WIDTH),
            "=" * WIDTH, ""]

    changed = c["signal"] != c["prev"]
    tag = f"   ** CHANGED from {c['prev']} **" if changed else "   (unchanged)"
    out += [f"  TOMORROW'S POSITION :  {MIX[c['signal']]}{tag}",
            f"  HELD TODAY          :  {MIX[c['held']]}", ""]

    # ------------------------------------------------------------ indicators
    out += ["  " + rule(), f"  MARKET INDICATORS  (XLK close {day})", "  " + rule()]

    def fmt(label, val):
        out.append(f"  {label:<21}:  {val}")

    fmt("XLK Close", f"{px:,.2f}")
    fmt(f"{p['trend_n']}-day SMA", f"{sma:,.2f}   (XLK/SMA = {px / sma:.3f})")
    fmt(f"{p['trend_n']}-day slope", f"{slope * 100:+.3f}% per day")
    fmt(f"{p['prox_n']}-day high prox", f"{prox:.3f}   ({(1 - prox) * 100:.1f}% below the {p['prox_n']}d high)")
    fmt(f"{p['mom_n']}-day return", f"{mom:+.2%}")
    fmt(f"{p['vol_n']}-day realised vol", f"{vol:.1%}   (percentile {volp:.2f} of last 252d)")
    fmt(f"RSI({p.get('rsi_n', 14)})", f"{rsi:.1f}")
    fmt("Raw state today", c["raw"])
    fmt("Consecutive days", f"{c['run']}   (confirmation needs {p['confirm_up']})")
    fmt("Circuit breaker", "TRIPPED — forcing BIL" if c["cb"] else "off")
    out.append("")

    # ------------------------------------------------------------ conditions
    out += ["  " + rule(), "  STATE CONDITIONS", "  " + rule(), ""]

    cond(px > sma * (1 + p.get("trend_buffer", 0.0)),
         f"XLK > {p['trend_n']}-day SMA", f"{px:,.2f} vs {sma:,.2f}")
    block(1, "TREND — is XLK above its long trend?", "-> at least XLK")

    cond(prox >= p["prox_thr"], f"proximity to {p['prox_n']}d high >= {p['prox_thr']:.2f}", f"{prox:.3f}")
    cond(mom > p["mom_thr"], f"{p['mom_n']}-day return > {p['mom_thr']:.0%}", f"{mom:+.2%}")
    cond(volp <= p["vol_max"], f"{p['vol_n']}d vol percentile <= {p['vol_max']:.2f}", f"{volp:.2f}")
    cond(slope > 0, f"{p['trend_n']}-day log-price slope > 0", f"{slope * 100:+.3f}%/day")
    cond(rsi <= p["rsi_max"], f"RSI({p.get('rsi_n', 14)}) <= {p['rsi_max']}", f"{rsi:.1f}")
    block(2, "BREAKOUT — all five must hold to lever up", f"-> TECL {p['tecl_w']:.0%}")

    cond(not c["cb"], f"{p['vol_n']}d vol percentile < {p['cb_vol_pct']:.2f}",
         f"{volp:.2f}" + ("" if c["cb"] else " — not tripped"))
    block(3, "CIRCUIT BREAKER — forces BIL regardless of the above")

    cond(c["run"] >= p["confirm_up"],
         f"raw state has persisted {p['confirm_up']} consecutive days",
         f"{c['raw']} for {c['run']} day{'s' if c['run'] != 1 else ''}")
    block(4, f"CONFIRMATION — a new state must persist {p['confirm_up']} days")

    # ---------------------------------------------------------------- action
    if changed:
        action = f"MOVE TO {MIX[c['signal']]}   (from {MIX[c['prev']]})"
        why = (f"raw state {c['raw']} confirmed after {c['run']} consecutive days"
               if not c["cb"] else "circuit breaker tripped — force BIL")
    else:
        action = f"HOLD {MIX[c['signal']]}"
        why = "no confirmed state change."
    out += ["  " + "=" * (WIDTH - 2),
            f"  > TOMORROW'S ACTION: {action}",
            f"     Reason: {why}",
            "  " + "=" * (WIDTH - 2), ""]

    # ------------------------------------------------------------ position
    out += ["  " + rule(), "  POSITION / PERFORMANCE", "  " + rule(),
            f"  Day return : {c['ret']:+.2%}",
            f"  Equity     : {c['equity']:,.2f}x since {START}",
            f"  Drawdown   : {c['dd']:.1%} from high", ""]

    out += _recent_trades()
    out += _glossary(rule)
    return "\n".join(out)


def _recent_trades(n=5):
    try:
        tr = pd.read_csv(TRADES)
    except FileNotFoundError:
        return []
    out = ["  " + "-" * WIDTH,
           f"  LAST {n} HOLDING PERIODS  (of {len(tr)} total)",
           "  " + "-" * WIDTH]
    # the final row is the period still in force -- its return is mark-to-market,
    # not a closed result, so say so rather than letting it read as a finished trade
    for i, (_, t) in enumerate(tr.tail(n).iterrows()):
        open_now = i == n - 1
        out.append(f"    {t.entry_date} {t.asset:<4} -> {t.exit_date} "
                   f"{t.ret_net:+.2%}  [{t.days}d]"
                   + ("   << open, marked to today" if open_now else ""))
    out.append("")
    return out


def _glossary(rule):
    """Plain-English definition of every number above, in the order printed."""
    items = [
        (f"{PARAMS['trend_n']}-day SMA",
         "Average XLK close over the last 90 sessions. Above it = risk-on, and the",
         "strategy holds at least XLK. Below it = BIL. This is the master switch."),
        (f"{PARAMS['trend_n']}-day slope",
         "Slope of a straight line fitted to 90 days of log price, in % per day.",
         "Positive means the trend is still rising, not just sitting above a flat MA."),
        (f"{PARAMS['prox_n']}-day high prox",
         "Today's close divided by the highest close of the last 60 sessions.",
         "1.000 = at the high; 0.960 = 4% below it, the breakout threshold."),
        (f"{PARAMS['mom_n']}-day return",
         "Plain 60-session price change. Must be positive to lever up — it blocks",
         "a bounce that is near a *falling* 60-day high."),
        (f"{PARAMS['vol_n']}-day realised vol",
         "Annualised standard deviation of the last 20 daily returns, shown with its",
         "percentile against the trailing year. Percentile, not level, is what gates:",
         "<= 0.75 to hold TECL, >= 0.85 trips the circuit breaker."),
        (f"RSI({PARAMS.get('rsi_n', 14)})",
         "Wilder's relative strength index. Capped at 75 so the strategy does not buy",
         "3x leverage into a blow-off top."),
        ("Raw state",
         "What today's gates say in isolation, before confirmation. This is NOT the",
         "position — it is the input the confirmation counter is watching."),
        ("Consecutive days",
         "How long the raw state has read the same value. A new state is only acted on",
         "once this reaches 5. This single rule is the largest contributor to returns:",
         "without it the same gates yield 21.8% CAGR / -32.9% DD instead of 34.3% / -24.9%."),
        ("Circuit breaker",
         "An override, independent of confirmation. If 20-day vol reaches its 85th",
         "percentile the position goes to BIL immediately, however the ladder reads."),
    ]
    out = ["  " + rule(), "  WHAT THE INDICATORS MEAN", "  " + rule(), ""]
    for term, *lines in items:
        out.append(f"  {term}")
        out += [f"      {ln}" for ln in lines]
        out.append("")
    out += ["  Signals are computed on XLK's close at day t and held over day t+1, so",
            "  today's action is placed at or near tomorrow's open. Each day of delay",
            "  costs roughly 3 points of CAGR in backtest.", ""]
    return out


def subject(f, w, bt):
    c = collect(f, w, bt)
    mark = " ** CHANGED **" if c["signal"] != c["prev"] else ""
    return f"✅ TECL/XLK {c['date'].date()}: {MIX[c['signal']]}{mark}"


if __name__ == "__main__":
    f, w, bt = build()
    print(subject(f, w, bt) if "--subject" in sys.argv else report(f, w, bt))
