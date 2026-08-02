"""Chart the final strategy: growth, drawdown, and what it was holding."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_all
from engine import run_weights, metrics
from strategy import build_features
import strategy2
from final import PARAMS, START

# validated categorical slots: blue / orange / aqua  (see dataviz palette.md)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
SURF = "#fcfcfb"

prices, returns = load_all()
cash = returns["BIL"]
f = build_features(prices, "XLK")
w = strategy2.vol_target_weights(f, PARAMS, returns)
bt = run_weights(returns, w, start=START)
m = metrics(bt, cash, "final")

idx = bt.index
strat = bt["equity"]
xlk = (1 + returns["XLK"].reindex(idx).fillna(0)).cumprod()
tecl = (1 + returns["TECL"].reindex(idx).fillna(0)).cumprod()

fig, axes = plt.subplots(3, 1, figsize=(13.5, 12), height_ratios=[3, 1.5, 1.4],
                         sharex=True, facecolor=SURF)
fig.subplots_adjust(hspace=0.22, right=0.855, top=0.885, bottom=0.06)

# ---------------------------------------------------- 1. growth of $1 (log)
ax = axes[0]
ax.set_facecolor(SURF)
for series, color, name in ((tecl, ORANGE, "TECL buy & hold"),
                            (xlk, AQUA, "XLK buy & hold"),
                            (strat, BLUE, "Strategy")):
    ax.plot(idx, series, color=color, lw=2.0, label=name,
            zorder=3 if name == "Strategy" else 2)
    ax.annotate(f" {name}\n {series.iloc[-1]:,.0f}x", (idx[-1], series.iloc[-1]),
                color=color, fontsize=9.5, fontweight="bold", va="center",
                xytext=(6, 0), textcoords="offset points", annotation_clip=False)
ax.set_yscale("log")
ax.set_ylabel("growth of $1  (log scale)", color=INK2, fontsize=10)
ax.set_title("TECL / XLK / BIL regime strategy — 2009-2026", color=INK,
             fontsize=15, fontweight="bold", loc="left", pad=30)
ax.text(0, 1.035, f"CAGR {m['CAGR']:.1%}   max drawdown {m['maxDD']:.1%}   "
        f"Sharpe {m['Sharpe']:.2f}   {m['trades_per_yr']:.0f} switches/yr   "
        f"(10 bps/side, cash earns BIL)",
        transform=ax.transAxes, color=INK2, fontsize=10)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2,
          bbox_to_anchor=(0.0, 0.99))

# ---------------------------------------------------- 2. drawdown
ax = axes[1]
ax.set_facecolor(SURF)
dd_s = strat / strat.cummax() - 1
dd_x = xlk / xlk.cummax() - 1
ax.fill_between(idx, dd_x * 100, 0, color=AQUA, alpha=0.30, lw=0, label="XLK")
ax.plot(idx, dd_s * 100, color=BLUE, lw=1.8, label="Strategy")
ax.axhline(-30, color=ORANGE, lw=1.4, ls="--", zorder=1)
ax.annotate(" −30% limit", (idx[-1], -30), color=ORANGE, fontsize=9,
            fontweight="bold", va="bottom", xytext=(6, 2),
            textcoords="offset points", annotation_clip=False)
ax.annotate(f" worst {m['maxDD']:.1%}", (dd_s.idxmin(), dd_s.min() * 100), color=BLUE,
            fontsize=9, fontweight="bold", va="top", xytext=(4, -4),
            textcoords="offset points")
ax.set_ylabel("drawdown", color=INK2, fontsize=10)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_ylim(-36, 2)
ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2, ncol=2,
          bbox_to_anchor=(0.0, -0.30))

# ---------------------------------------------------- 3. exposure
ax = axes[2]
ax.set_facecolor(SURF)
expo = w.reindex(idx).fillna(0.0)
order = [("TECL", ORANGE), ("XLK", AQUA), ("BIL", BLUE)]
ax.stackplot(idx, *[expo[a] * 100 for a, _ in order],
             colors=[c for _, c in order], labels=[a for a, _ in order],
             edgecolor=SURF, linewidth=0.4)
ax.set_ylim(0, 100)
ax.set_ylabel("allocation", color=INK2, fontsize=10)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
share = {a: expo[a].mean() for a, _ in order}
leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2, ncol=3,
                bbox_to_anchor=(0.0, -0.12),
                title=f"time-weighted:  TECL {share['TECL']:.0%}   XLK {share['XLK']:.0%}   BIL {share['BIL']:.0%}")
leg.get_title().set_color(INK2)
leg.get_title().set_fontsize(9)
leg._legend_box.align = "left"

for a in axes:
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9.5)

fig.savefig("strategy_final.png", dpi=140, facecolor=SURF, bbox_inches="tight")
print("wrote strategy_final.png")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
       if k in ("CAGR", "maxDD", "Sharpe", "Sortino", "Calmar", "vol", "final_x")})
