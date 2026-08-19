#!/bin/bash
# TECL / XLK / BIL regime timing — Daily runner
# Refreshes the full EODHD histories, gates on freshness, re-verifies the TECS
# split repair, re-exports the equity curve + trade blotter, reports the current
# signal, and publishes the data to GitHub. Logs to logs/daily-*.log and sends a
# heartbeat on success / failure email on any error, exactly like the Nitro runner.

set -eo pipefail

# Load credentials (EODHD_API_TOKEN, GOOGLE_EMAIL/GOOGLE_APP_PASSWORD, GIT_TOKEN)
source "$HOME/.bash_profile"

# Self-locate: this script lives in the project root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PROJECT_NAME="TECL/XLK"

# Prefer a project venv if present, else python3 from PATH, else the system one.
if   [ -x ".venv/bin/python3" ];  then PY=".venv/bin/python3"
elif command -v python3 >/dev/null; then PY="$(command -v python3)"
else                                   PY="/usr/bin/python3"
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$LOG_DIR/daily-${TS}.log"
SIGNAL_FILE="$LOG_DIR/latest_signal.txt"

# ── Email alert helper ────────────────────────────────────────────────────────
# Usage: send_alert "subject" "/path/to/body_file"
send_alert() {
    local subject="$1"
    local body_file="$2"
    "$PY" -c '
import os, sys, smtplib
from email.mime.text import MIMEText
user = os.environ.get("GOOGLE_EMAIL", "")
pw   = os.environ.get("GOOGLE_APP_PASSWORD", "")
if not (user and pw):
    print("send_alert: GOOGLE_EMAIL/GOOGLE_APP_PASSWORD not set — skipping", file=sys.stderr)
    sys.exit(0)
subject = sys.argv[1]
with open(sys.argv[2], "r") as f:
    body = f.read()
msg = MIMEText(body)
msg["Subject"] = subject
msg["From"]    = user
msg["To"]      = user
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, [user], msg.as_string())
    print(f"alert sent: {subject}")
except Exception as e:
    print(f"alert email failed: {e}", file=sys.stderr)
' "$subject" "$body_file"
}

# ── Git publish helper ────────────────────────────────────────────────────────
# Stages json/ + results/, commits, and pushes to origin/main using GOOGLE_EMAIL
# + GIT_TOKEN. Returns non-zero on failure so the caller can warn without aborting.
git_publish() {
    if [ -z "$GIT_TOKEN" ];    then echo "WARN: GIT_TOKEN not set — skipping git push";    return 1; fi
    if [ -z "$GOOGLE_EMAIL" ]; then echo "WARN: GOOGLE_EMAIL not set — skipping git push"; return 1; fi

    git add json results || return 1
    if git diff --cached --quiet; then
        echo "No data changes to commit."
        return 0
    fi
    git -c user.email="$GOOGLE_EMAIL" -c user.name="$GOOGLE_EMAIL" \
        commit -m "Daily data update — $(date '+%Y-%m-%d')" || return 1

    local encoded_user="${GOOGLE_EMAIL//@/%40}"
    local push_url="https://${encoded_user}:${GIT_TOKEN}@github.com/dampiermike/TECL_XLK_BIL_timing.git"
    git -c credential.helper= push "$push_url" HEAD:main || return 1
    echo "Pushed to origin/main."
}

# ── Failure trap ──────────────────────────────────────────────────────────────
on_failure() {
    local rc=$?
    local alert_body="/tmp/TECL_XLK_fail_$$.txt"
    {
        echo "$PROJECT_NAME daily run FAILED"
        echo "Exit code: $rc"
        echo "Timestamp: $(date)"
        echo "Log file:  $ROOT_DIR/$LOG_FILE"
        echo ""
        echo "─── Last 50 log lines ───"
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "(no log available)"
    } > "$alert_body"
    send_alert "❌ $PROJECT_NAME daily FAILED (rc=$rc)" "$alert_body"
    rm -f "$alert_body"
    exit $rc
}
trap on_failure ERR

# ── Pipeline (all output tee'd to log file) ───────────────────────────────────
{
    echo "=========================================="
    echo "  $PROJECT_NAME Daily Run — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    echo
    echo "── Step 1/5: Full-history EODHD download (overwrites json/) ──"
    $PY download_data.py

    echo
    echo "── Step 2/5: Freshness gate (aborts run if any file stale) ──"
    $PY validate_freshness.py

    echo
    echo "── Step 3/5: Loader sanity (TECL beta must hold +3 vs XLK) ──"
    # TECS is not loaded here: it is never held (final.PARAMS crash_mom_thr=None),
    # so its feed must not be able to halt the run. `python data.py --tecs` runs
    # the old split-repair check on demand.
    $PY data.py

    echo
    echo "── Step 4/5: Export equity curve + trade blotter ──"
    $PY export.py

    echo
    echo "── Step 5/5: Current signal ──"
    # state_signal on the last row is what that close says to hold next session;
    # state_held is what the position was during that day. See README "Outputs".
    # NB: no single quotes below — the program is inside a single-quoted shell string.
    $PY -c '
import pandas as pd
eq = pd.read_csv("results/equity_curve.csv", index_col=0, parse_dates=True)
last, prev = eq.iloc[-1], eq.iloc[-2]
sig, prev_sig, held = last["state_signal"], prev["state_signal"], last["state_held"]
w = {c[2:]: float(last[c] or 0) for c in ("w_TECL", "w_XLK", "w_BIL")}
mix = "  ".join(f"{k} {v:.0%}" for k, v in w.items() if v > 0)
note = f"** CHANGED from {prev_sig} **" if sig != prev_sig else "(unchanged)"
ret, eq_x, dd = float(last["ret_net"]), float(last["equity"]), float(last["drawdown"])
print(f"  as of close:   {last.name.date()}")
print(f"  signal:        {sig}   {note}")
print(f"  held that day: {held}   [{mix}]")
print(f"  day return:    {ret:+.2%}    equity {eq_x:.2f}x    drawdown {dd:.1%}")
' | tee "$SIGNAL_FILE"

    echo
    echo "── Commit & push data ──"
    git_publish || echo "WARN: git commit/push failed (continuing)"

    echo
    echo "=========================================="
    echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} 2>&1 | tee "$LOG_FILE"

# ── Heartbeat on success ──────────────────────────────────────────────────────
alert_body="/tmp/TECL_XLK_ok_$$.txt"
{
    echo "$PROJECT_NAME daily run completed successfully."
    echo "Timestamp: $(date)"
    echo "Log file:  $ROOT_DIR/$LOG_FILE"
    echo ""
    echo "─── Signal ───"
    cat "$SIGNAL_FILE" 2>/dev/null || echo "(no signal captured)"
    echo ""
    echo "─── Last 30 log lines ───"
    tail -n 30 "$LOG_FILE"
} > "$alert_body"
send_alert "✅ $PROJECT_NAME daily OK" "$alert_body"
rm -f "$alert_body"
