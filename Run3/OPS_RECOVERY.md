# Ops: container-restart recovery architecture (2026-07-15)

## The problem

The training host's platform watchdog restarts this dev container when the
GPU driver (NVML) health-check fails. A restart wipes everything outside
`/home/MYH` (overlay filesystem): tmux and its sessions, gh, cron entries,
any daemon we start. Only `/home/MYH` (ext4 mount) survives — repo,
checkpoints, wrappers, ssh config. After a restart the ONLY process the
image starts by itself is `sshd`, so recovery code can run only when an
inbound connection arrives.

Observed restart history (host kernel itself stayed up — container-level):

| restart | recovered | gap |
|---|---|---|
| 2026-06-30 06:39 | same day | — |
| 2026-07-06 02:55 | same day (resurrected 3 retired runs — zombie lesson) | — |
| 2026-07-13 17:40 | 07-14 00:55 (login-triggered) | ~7 h dead |
| 2026-07-14 12:30 | 07-15 00:10 (login-triggered) | ~11.7 h dead |

## Recovery layers (current)

1. **`Run3/_autorecover.sh` — the single source of recovery logic.**
   Idempotent: if `tmux` exists the container is healthy → exit instantly
   (dead-session handling inside a healthy container belongs to the
   watchdog). On a fresh restart: mkdir-lock against concurrent callers →
   apt-install tmux+gh → per run: `touch <run>/.wd_marker` **before**
   launching (grace for the watchdog — prevents the 07-14/15 double-fire
   churn where the watchdog killed just-revived sessions whose CSVs were
   hours stale) → relaunch the run sessions and the watchdog. Log:
   `Run3/_autorecover.log`.
2. **`~/.ssh/rc`** (outside the repo, in /home — survives restarts): sshd
   executes it on EVERY incoming session — interactive terminals, **Cursor
   remote (including its automatic background reconnects)**, scp/sftp. It
   invokes `_autorecover.sh` detached and fully silenced (stdout must stay
   clean or sftp breaks). Net effect: while the owner's Cursor/laptop is
   merely open, any reconnect attempt after a container restart triggers
   recovery within the reconnect interval — no manual terminal needed.
3. **`~/.bashrc`** hook (interactive logins) — now a thin delegate to the
   same script; kept as a fallback and for the human-visible message.
4. **`Run3/_watchdog.sh`** (inside tmux, dies with the container): within a
   healthy container it relaunches dead sessions and kill+resumes hung runs
   (env_metrics mtime older than HANG_MIN=60 min).

**Run-list sync rule**: the active-run list lives in BOTH
`Run3/_autorecover.sh` (RECOVER_RUNS) and `Run3/_watchdog.sh` (RUNS).
Retiring or adding a run = edit both in the same breath + restart the
watchdog session (a running bash script never re-reads its file).

## Coverage and the remaining gap

- Covered: any moment SOMETHING connects — owner's terminal, Cursor open in
  the background, scp. With Cursor left open this is near-immediate.
- NOT covered (deliberately deferred, owner decision 2026-07-15): restarts
  while no machine of the owner is connected at all (laptop off,
  overnight). Runs stay dead until the next connection. The trainings
  themselves need no connection to keep running — the gap exists only
  between a container restart and the next inbound connection.

## Deferred options for full unattended recovery (documented for later)

1. **Admin boot hook (cleanest, permanent)**: ask the platform admin to add
   to the container entrypoint: *"if `/home/MYH/.on_boot.sh` exists, run it
   in the background at container start"*. Then point that file at
   `_autorecover.sh`. No SSH involved at all.
2. **Always-on poker machine**: any 24/7 box (lab server, RPi) with a
   crontab line every 5–10 min:
   `ssh -o BatchMode=yes MYH@<host> 'bash ~/ML_DRL_Scheduler/Run3/_autorecover.sh'`
   Security: generate a NEW key on that box and register it in
   `authorized_keys` with a **forced command** so the key can do nothing
   but run the recovery script:
   `command="bash /home/MYH/ML_DRL_Scheduler/Run3/_autorecover.sh",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding ssh-ed25519 AAAA...`
   The personal-laptop key keeps full access; the poker key cannot open a
   shell even if leaked. (GitHub-Actions-as-poker was considered and
   rejected: it requires depositing a server key with an external service.)

## Related facts worth remembering

- Interrupted updates resume faithfully: checkpoints are update-boundary
  snapshots (model+optimizer+RNG), so a kill between "CSV row written" and
  "checkpoint saved" makes the resumed process REDO those updates and
  append duplicate CSV rows — **bit-identical to the originals** (verified
  on updates 70–75 and 140–141 of QueuePostRZF). Analysis must dedupe by
  update index; training math is unaffected.
- `pkill`/`pgrep -f` self-match trap: a pattern that appears literally in
  your own command line matches your own shell. Use character-class
  patterns (`QueuePostRZ[F]`).
