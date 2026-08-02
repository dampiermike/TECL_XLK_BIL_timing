"""
The selected strategy, in one place.

Signal ticker: XLK. Traded: TECL / XLK / BIL. TECS is deliberately absent --
see tecs_verdict.py for the evidence that it subtracts value at every exposure
level above noise.

Rules, evaluated on XLK's close at day t and held over day t+1:

  TREND      XLK > its 90-day SMA                           -> at least XLK
  BREAKOUT   ...AND within 4% of its 60-day high
             ...AND 60-day return positive
             ...AND 20-day realised vol below its 75th pctile
             ...AND the 90-day log-price OLS slope is positive
             ...AND 14-day RSI <= 75 (do not buy leverage into a blow-off)
                                                             -> TECL at 75%
  CONFIRM    a new state must persist 5 consecutive days before it is acted on
  BREAKER    20-day realised vol at/above its 85th pctile forces BIL
  ELSE                                                       -> BIL

Sizing: the breakout state holds 0.75 TECL / 0.25 BIL. That fraction is the one
knob that sets where the strategy sits on the CAGR-vs-drawdown line.

These values were NOT chosen as the single best-scoring cell of the sweep. They
were chosen because the whole neighbourhood around them works (stage 8) -- the
earlier best-scoring pick turned out to sit on a spike that vanished one grid
step away.
"""

from data import load_all
from strategy import build_features
import strategy2

START = "2009-01-02"

PARAMS = dict(
    trend_n=90, trend_buffer=0.0,
    prox_n=60, prox_thr=0.96,
    mom_n=60, mom_thr=0.0,
    vol_n=20, vol_max=0.75,
    abs_vol_max=None, vix_max=None,
    require_slope=True,
    rsi_n=14, rsi_max=75, macd_pos=False,
    er_min=None, er_cash=None,
    confirm_style="sym", confirm_up=5,
    cb_vol_pct=0.85, cb_drop=None, cb_drop_n=10, cb_vix=None,
    crash_mom_thr=None,                      # TECS disabled -- see tecs_verdict.py
    sizing="fixed", target_vol=None, max_leverage=1.0, min_weight=0.0,
    tecl_w=0.75, xlk_w=1.0,
)

# Same rules, sized down: 32.4% CAGR / -23.5% DD. Use when the drawdown limit
# is hard rather than a target.
PARAMS_CONSERVATIVE = dict(PARAMS, tecl_w=0.70)


def build(signal_ticker="XLK", params=None):
    """Return (returns, cash, weights) for the strategy."""
    prices, returns = load_all()
    f = build_features(prices, signal_ticker)
    w = strategy2.vote_weights(f, params or PARAMS, returns) if (params or PARAMS).get("_vote") \
        else strategy2.vol_target_weights(f, params or PARAMS, returns)
    return returns, returns["BIL"], w
