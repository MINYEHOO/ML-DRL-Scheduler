#!/bin/bash
# HOLD-vs-watchdog integration test (audit round 8) -- MANUAL, ~6 minutes.
#
# Verifies that a pin-failed wrapper HOLD survives the hang-watchdog's
# HANG_MIN kill rule (because the HOLD touches the run's .wd_marker every
# 30 s), using a NEGATIVE CONTROL that reproduces the pre-fix churn.
#
# Rig (all under a scratch dir; the real repo wrapper/watchdog are used as
# templates via sed, so the tested logic is the shipped logic):
#   A = copy of Run4/_wrap_queuepostrzf.sh with PIN_BASE=deadbeef...
#       (forces PIN FAIL -> HOLD) and test-local RUN_DIR/LOG.
#   B = same, but the HOLD's marker touch removed and sleep 30 -> 600
#       (the pre-round-8 behavior).
#   watchdog = copy of Run3/_watchdog.sh watching A and B, HANG_MIN=1,
#       CHECK_SEC=10.
#
# PASS (observed 2026-07-13): after 5 minutes,
#   A: tmux session ALIVE, exactly 1 "PIN FAIL" line, zero watchdog actions;
#   B: killed+relaunched ~5x by the watchdog, >=5 "PIN FAIL" lines.
#
# Run from the repo root:  bash Run4/_analysis/scripts/audit_probes/hold_watchdog_integration_test.sh
set -u
cd /home/MYH/ML_DRL_Scheduler || exit 1
T=$(mktemp -d /tmp/holdtest.XXXX)
mkdir -p "$T/runA" "$T/runB"
trap 'tmux kill-session -t holdtestA 2>/dev/null; tmux kill-session -t holdtestB 2>/dev/null; tmux kill-session -t watchdogtest 2>/dev/null; rm -rf "$T"' EXIT

sed -e "s|RUN_DIR=Run4/QueuePostRZF|RUN_DIR=$T/runA|" \
    -e "s|LOG=Run4/QueuePostRZF_run.log|LOG=$T/A.log|" \
    -e "s|^PIN_BASE=.*|PIN_BASE=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef|" \
    Run4/_wrap_queuepostrzf.sh > "$T/wrapA.sh"
sed -e 's|sleep 30|sleep 600|' -e '/touch "\$RUN_DIR\/.wd_marker"/d' \
    "$T/wrapA.sh" | sed -e "s|RUN_DIR=$T/runA|RUN_DIR=$T/runB|" \
    -e "s|LOG=$T/A.log|LOG=$T/B.log|" > "$T/wrapB.sh"
sed -e "s|^RUNS=(|RUNS=(\n  \"$T/runA:holdtestA:$T/wrapA.sh\"\n  \"$T/runB:holdtestB:$T/wrapB.sh\"\n)\nDISABLED_RUNS=(|" \
    -e "s|LOG=Run3/_watchdog.log|LOG=$T/wd.log|" \
    -e "s|bash /home/MYH/ML_DRL_Scheduler/\$2|bash \$2|" \
    Run3/_watchdog.sh > "$T/wd.sh"

tmux new-session -d -s holdtestA "bash $T/wrapA.sh"
tmux new-session -d -s holdtestB "bash $T/wrapB.sh"
sleep 3
tmux new-session -d -s watchdogtest "HANG_MIN=1 CHECK_SEC=10 bash $T/wd.sh"
echo "rigs up; waiting 5 minutes..."
sleep 300

A_ALIVE=$(tmux has-session -t holdtestA 2>/dev/null && echo yes || echo no)
A_FAILS=$(grep -c "PIN FAIL" "$T/A.log" 2>/dev/null || echo 0)
B_FAILS=$(grep -c "PIN FAIL" "$T/B.log" 2>/dev/null || echo 0)
WD_A=$(grep -c "runA.*HUNG" "$T/wd.log" 2>/dev/null || echo 0)
WD_B=$(grep -c "runB.*HUNG" "$T/wd.log" 2>/dev/null || echo 0)
echo "A: alive=$A_ALIVE pin_fails=$A_FAILS watchdog_kills=$WD_A"
echo "B: pin_fails=$B_FAILS watchdog_kills=$WD_B (negative control)"
[ "$A_ALIVE" = "yes" ] && [ "$A_FAILS" -eq 1 ] && [ "$WD_A" -eq 0 ] \
  && [ "$WD_B" -ge 2 ] && [ "$B_FAILS" -ge 2 ] \
  && echo "HOLD-WATCHDOG INTEGRATION TEST PASSED" \
  || { echo "TEST FAILED"; exit 1; }
