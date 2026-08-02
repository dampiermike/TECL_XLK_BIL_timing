"""
Export the daily equity curve and the trade blotter for the final config.

A "trade" is a contiguous run holding the same state. Because engine.run_weights
holds day t's weights over t+1, the position actually in force on a given day is
`weights.shift(1)` -- so entry_date is the first day the position EARNS a return,
not the day the signal fired. signal_date is reported alongside it, since that is
the close at which the order would be placed.

Cash (BIL) stretches are included as rows so the record is continuous and the
holding periods reconcile to 100% of the calendar.
"""

import numpy as np
import pandas as pd

from data import load_all
from engine import run_weights, metrics
from strategy import build_features
import strategy2
from final import PARAMS, START

OUT_EQUITY = "results/equity_curve.csv"
OUT_TRADES = "results/trades.csv"


def build():
    prices, returns = load_all()
    cash = returns["BIL"]
    f = build_features(prices, "XLK")
    w = strategy2.vol_target_weights(f, PARAMS, returns)
    bt = run_weights(returns, w, start=START)
    return returns, cash, w, bt


def equity_frame(returns, cash, w, bt):
    idx = bt.index
    held = w.reindex(idx).shift(1)          # what is actually owned that day
    eq = bt["equity"]
    # Two different days' information, kept explicitly separate:
    #   state_held   -- what is owned during this day (matches the w_* columns
    #                   and earns this day's ret_net)
    #   state_signal -- what this day's CLOSE says to own tomorrow; it becomes
    #                   the next row's state_held. These differ by one row by
    #                   construction, which is the no-lookahead rule made visible.
    invested = held.sum(axis=1) > 0
    held_state = pd.Series(pd.NA, index=idx, dtype=object)
    held_state[invested] = held.round(4)[invested].idxmax(axis=1)
    out = pd.DataFrame({
        "date": idx,
        "state_held": held_state,
        "state_signal": bt["target"],
        "w_TECL": held["TECL"].round(4),
        "w_XLK": held["XLK"].round(4),
        "w_BIL": held["BIL"].round(4),
        "ret_gross": bt["gross"].round(6),
        "cost": bt["cost"].round(6),
        "ret_net": bt["ret"].round(6),
        "equity": eq.round(4),
        "drawdown": (eq / eq.cummax() - 1).round(4),
        "XLK_ret": returns["XLK"].reindex(idx).round(6),
        "TECL_ret": returns["TECL"].reindex(idx).round(6),
        "BIL_ret": cash.reindex(idx).round(6),
    })
    return out.set_index("date")


def trade_blotter(returns, w, bt):
    """One row per contiguous holding period."""
    idx = bt.index
    held = w.reindex(idx).shift(1).fillna(0.0)
    # label each day by the position in force, rounded so tiny float noise
    # does not split a run into two trades
    label = held.round(4).astype(str).agg("|".join, axis=1)
    runs = (label != label.shift(1)).cumsum()

    rows = []
    for _, seg in bt.groupby(runs):
        d0, d1 = seg.index[0], seg.index[-1]
        pos = held.loc[d0]
        if pos.sum() == 0:                    # the pre-first-signal stub
            continue
        asset = pos.idxmax()
        wt = float(pos.max())
        gross = float((1 + seg["gross"]).prod() - 1)
        net = float((1 + seg["ret"]).prod() - 1)
        sig = idx[idx.get_loc(d0) - 1] if idx.get_loc(d0) > 0 else d0
        under = returns[asset].reindex(seg.index) if asset in returns else None
        rows.append({
            "signal_date": sig.date(),         # close at which the order is placed
            "entry_date": d0.date(),           # first day the position earns
            "exit_date": d1.date(),
            "asset": asset,
            "weight": round(wt, 3),
            "days": len(seg),
            "ret_gross": round(gross, 5),
            "cost": round(float(seg["cost"].sum()), 5),
            "ret_net": round(net, 5),
            "equity_in": round(float(seg["equity"].iloc[0] / (1 + seg["ret"].iloc[0])), 4),
            "equity_out": round(float(seg["equity"].iloc[-1]), 4),
            "underlying_ret": round(float((1 + under).prod() - 1), 5) if under is not None else np.nan,
            "max_adverse": round(float(((1 + seg["ret"]).cumprod() /
                                        (1 + seg["ret"]).cumprod().cummax() - 1).min()), 4),
        })
    return pd.DataFrame(rows)


def summarise(tr):
    print(f"\n{len(tr)} holding periods "
          f"({tr.entry_date.min()} -> {tr.exit_date.max()})\n")
    g = tr.groupby("asset")
    summary = pd.DataFrame({
        "trades": g.size(),
        "days_held": g["days"].sum(),
        "pct_of_time": (g["days"].sum() / tr["days"].sum()),
        "avg_days": g["days"].mean(),
        "win_rate": g["ret_net"].apply(lambda s: (s > 0).mean()),
        "avg_ret": g["ret_net"].mean(),
        "median_ret": g["ret_net"].median(),
        "best": g["ret_net"].max(),
        "worst": g["ret_net"].min(),
        "total_cost": g["cost"].sum(),
    })
    print(summary.to_string(float_format=lambda v: f"{v:,.4f}"))

    risk = tr[tr.asset != "BIL"]
    print(f"\nrisk-on trades only: {len(risk)}  win rate {(risk.ret_net > 0).mean():.1%}  "
          f"avg {risk.ret_net.mean():+.2%}  "
          f"payoff {risk[risk.ret_net>0].ret_net.mean()/abs(risk[risk.ret_net<0].ret_net.mean()):.2f}x")

    print("\n10 biggest winners:")
    print(tr.nlargest(10, "ret_net")[
        ["entry_date", "exit_date", "asset", "weight", "days", "ret_net", "underlying_ret"]
    ].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n10 biggest losers:")
    print(tr.nsmallest(10, "ret_net")[
        ["entry_date", "exit_date", "asset", "weight", "days", "ret_net", "underlying_ret"]
    ].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    returns, cash, w, bt = build()
    m = metrics(bt, cash, "final")
    print(f"CAGR {m['CAGR']:.1%}  maxDD {m['maxDD']:.1%}  Sharpe {m['Sharpe']:.2f}  "
          f"final {m['final_x']:.1f}x")

    eq = equity_frame(returns, cash, w, bt)
    eq.to_csv(OUT_EQUITY)
    print(f"\nwrote {OUT_EQUITY}  ({len(eq)} daily rows)")

    tr = trade_blotter(returns, w, bt)
    tr.to_csv(OUT_TRADES, index=False)
    print(f"wrote {OUT_TRADES}  ({len(tr)} trades)")

    # the blotter must reconcile to the backtest, or it is decoration
    chain = float((1 + tr["ret_net"]).prod())
    print(f"\nreconciliation: compounded trade returns {chain:.4f}x  vs  "
          f"equity curve {bt['equity'].iloc[-1]:.4f}x  "
          f"(diff {abs(chain / bt['equity'].iloc[-1] - 1):.2e})")
    print(f"days in blotter {tr['days'].sum()} vs backtest days {len(bt)}")

    summarise(tr)
