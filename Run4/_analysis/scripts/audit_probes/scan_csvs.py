#!/usr/bin/env python3
"""Read-only scan of run CSVs: duplicate update indices, backward jumps,
mixed column counts (schema drift). Never writes into the repo."""
import csv, glob, os

ROOT = "/home/MYH/ML_DRL_Scheduler"
pats = ["Run2/*/csv_logs/env_metrics.csv", "Run3/*/csv_logs/env_metrics.csv",
        "Run4/*/csv_logs/env_metrics.csv", "runs/*/csv_logs/env_metrics.csv",
        "Run2/*/csv_logs/ppo_metrics.csv", "Run3/*/csv_logs/ppo_metrics.csv",
        "Run4/*/csv_logs/ppo_metrics.csv"]
files = sorted(set(f for p in pats for f in glob.glob(os.path.join(ROOT, p))))
for f in files:
    ncols = {}
    updates = []
    with open(f, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if not row:
                continue
            ncols.setdefault(len(row), 0)
            ncols[len(row)] += 1
            if row[0] != "update":
                try:
                    updates.append(int(float(row[0])))
                except ValueError:
                    pass
    dups = 0
    back = 0
    prev = None
    seen = set()
    for u in updates:
        if u in seen:
            dups += 1
        seen.add(u)
        if prev is not None and u < prev:
            back += 1
        prev = u
    rel = os.path.relpath(f, ROOT)
    flag = []
    if len(ncols) > 1:
        flag.append(f"MIXED-COLS {sorted(ncols.items())}")
    if dups:
        flag.append(f"DUP-UPDATES n={dups}")
    if back:
        flag.append(f"BACKWARD-JUMPS n={back}")
    if flag:
        print(f"{rel}: rows={len(updates)} :: " + " | ".join(flag))
    else:
        print(f"{rel}: rows={len(updates)} :: clean (cols={list(ncols)[0] if ncols else 0})")
