"""
Stage 4: position sizing, to close the last few points of drawdown.

The long core is already at 35% CAGR / -34.8% DD, so CAGR has headroom and DD
does not. Two sizing schemes are swept against the same signals:

  voltarget -- hold TECL scaled to a volatility budget, remainder in BIL. When
               3x tech vol runs hot the position shrinks automatically, which is
               exactly the chop regime the diagnosis blamed for the drawdowns.
  fixed     -- a constant fractional TECL weight, as the naive control.

TECS is excluded here: stage 3 showed its contribution is negative at every
exposure level above noise.
"""

from sweep import sweep, report, save

BASE = dict(
    trend_n=[100, 150], trend_buffer=[0.0], prox_n=[60], prox_thr=[0.96],
    mom_n=[20, 60], mom_thr=[0.0], vol_n=[20], vol_max=[0.7, None],
    abs_vol_max=[None], vix_max=[None], require_slope=[True, False],
    confirm_style=["sym"], confirm_up=[3, 5],
    cb_vol_pct=[0.90, 0.95, None], cb_drop=[None], cb_drop_n=[10], cb_vix=[None],
    crash_mom_thr=[None],
)

VOL = dict(BASE, sizing=["voltarget"],
           target_vol=[0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
           max_leverage=[1.0], min_weight=[0.0], tecl_w=[1.0], xlk_w=[1.0])

FIXED = dict(BASE, sizing=["fixed"], target_vol=[None], max_leverage=[1.0],
             min_weight=[0.0], tecl_w=[0.4, 0.5, 0.6, 0.7, 0.85, 1.0], xlk_w=[1.0])

KEYS = ["trend_n", "mom_n", "vol_max", "require_slope", "confirm_up", "cb_vol_pct",
        "sizing", "target_vol", "tecl_w"]

if __name__ == "__main__":
    for name, spec in (("VOL-TARGET", VOL), ("FIXED FRACTION", FIXED)):
        df, n = sweep("v2vol", spec)
        report(df, n, f"STAGE 4 {name} - ranked by FULL Calmar", "ALL_Calmar", 12, KEYS)
        hit = df[(df.ALL_CAGR >= 0.30) & (df.ALL_maxDD >= -0.30)]
        report(hit, n, f"STAGE 4 {name} - MEETING 30% CAGR / 30% DD (full sample)",
               "ALL_CAGR", 25, KEYS)
        both = hit[(hit.IS_CAGR >= 0.25) & (hit.OOS_CAGR >= 0.25) &
                   (hit.IS_maxDD >= -0.32) & (hit.OOS_maxDD >= -0.32)]
        report(both, n, f"STAGE 4 {name} - and robust in BOTH halves", "ALL_CAGR", 25, KEYS)
        save(df, f"sweep4_{spec['sizing'][0]}.csv")
