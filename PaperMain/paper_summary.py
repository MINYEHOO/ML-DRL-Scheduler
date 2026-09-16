#!/usr/bin/env python3
"""Summarize preserved paper evidence; no simulation or checkpoint loading.

Run from any directory: python /path/to/PaperMain/paper_summary.py
Outputs are written under PaperMain/reports/evidence_summary by default.
The 20000 headline, reused 21000 comparison and 30000 OOD bands stay separate.
Historical NARROW D26 is inventoried but excluded from strict comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent
RECIPES = ("base", "lrann", "narrow")
WORLDS = ("P055", "P010", "V60max", "CSI02", "D26", "STORM2", "K8", "K48", "K60")
METRICS = ("reward", "throughput_mbps", "goodput_mbps", "mean_sinr_db",
           "completion_rate", "deadline_miss_rate", "retx_drop_rate", "mu_depth", "jain")
HIGHER_IS_BETTER = {m: m not in {"deadline_miss_rate", "retx_drop_rate"} for m in METRICS}
HIGHER_IS_BETTER["mu_depth"] = None  # Descriptive mechanism, not an objective.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_path(relative: str) -> Path:
    """Never follow a source-manifest path or symlink outside this package."""
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT)
    return result


def read_rows(pattern: str) -> list[dict]:
    result = []
    for path in sorted(ROOT.glob(pattern)):
        path.resolve().relative_to(ROOT)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path.relative_to(ROOT).as_posix()
                row["seed"] = int(row["seed"])
                world = row.get("world", row.get("config", "in_distribution"))
                row["world"] = "K" + world if world in {"8", "48", "60"} else world
                row["sched"] = row["sched"].replace("+Random", "+Rnd")
                result.append(row)
    return result


def select(rows: list[dict], sched: str, world: str = "in_distribution") -> dict[int, dict]:
    selected = {}
    for row in rows:
        if row["sched"] == sched and row["world"] == world:
            if row["seed"] in selected:
                raise ValueError(f"Duplicate episode {world}/{sched}/{row['seed']}")
            selected[row["seed"]] = row
    return selected


def coverage(rows: dict[int, dict], start: int, stop: int) -> dict:
    expected = set(range(start, stop))
    actual = set(rows)
    return {"n": len(actual), "expected_n": len(expected),
            "complete": actual == expected, "missing_seeds": sorted(expected - actual),
            "unexpected_seeds": sorted(actual - expected)}


def paired(band: str, world: str, lhs: str, rhs: str,
           left: dict[int, dict], right: dict[int, dict], notes: str = "") -> list[dict]:
    if not left or not right:
        return []
    if left.keys() != right.keys():
        raise ValueError(f"Unpaired episode sets: {band}/{world}/{lhs}/{rhs}")
    n = len(left)
    if n < 2:
        raise ValueError("At least two paired episodes are required")
    crit = float(student_t.ppf(0.975, n - 1))
    result = []
    sources = sorted({r["_source"] for rows in (left, right) for r in rows.values()})
    for metric in METRICS:
        if not all(metric in r and r[metric] != "" for r in (*left.values(), *right.values())):
            continue
        a = [float(left[k][metric]) for k in sorted(left)]
        b = [float(right[k][metric]) for k in sorted(left)]
        if not all(math.isfinite(x) for x in a + b):
            raise ValueError(f"Nonfinite {metric}: {band}/{world}/{lhs}/{rhs}")
        diffs = [x - y for x, y in zip(a, b)]
        mean = statistics.mean(diffs)
        half = crit * statistics.stdev(diffs) / math.sqrt(n)
        direction = HIGHER_IS_BETTER[metric]
        result.append({"band": band, "world": world, "lhs": lhs, "rhs": rhs,
                       "metric": metric, "n": n, "lhs_mean": statistics.mean(a),
                       "rhs_mean": statistics.mean(b), "mean_difference": mean,
                       "ci95_low": mean - half, "ci95_high": mean + half,
                       "positive_differences": sum(x > 0 for x in diffs),
                       "negative_differences": sum(x < 0 for x in diffs),
                       "ties": sum(x == 0 for x in diffs),
                       "higher_is_better": direction, "notes": notes,
                       "source_files": sources})
    return result


def verify_provenance() -> dict:
    manifest_path = package_path("provenance/source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checked, missing, changed = [], [], []
    recorded = set()
    for entry in manifest["entries"]:
        relative = entry["path"]
        if not relative.startswith(("artifacts/", "evidence/")):
            continue  # Executable copies may intentionally be repaired after extraction.
        path = package_path(relative)
        recorded.add(relative)
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            changed.append(relative)
        else:
            checked.append(relative)
    unrecorded = sorted(p.relative_to(ROOT).as_posix()
                        for folder in ("artifacts", "evidence")
                        for p in package_path(folder).rglob("*")
                        if p.is_file() and p.relative_to(ROOT).as_posix() not in recorded)
    return {"manifest": "provenance/source_manifest.json", "source_head": manifest["source_head"],
            "source_root_informational_only": manifest["source_root"],
            "verified_file_count": len(checked), "missing_files": missing,
            "changed_files": changed, "unrecorded_files": unrecorded,
            "passed": not (missing or changed or unrecorded)}


def build_summary() -> dict:
    provenance = verify_provenance()
    stats, matrix, issues = [], [], []
    datasets = {}

    def record(recipe, task, world, data, start, stop, note="", invalid=False):
        c = coverage(data, start, stop)
        status = "invalid_protocol" if invalid and c["n"] else (
            "available" if c["complete"] else "missing" if not c["n"] else "incomplete")
        matrix.append({"recipe": recipe, "evaluation": task, "world": world,
                       "status": status, **c, "notes": note})
        return c["complete"] and not invalid

    headline = read_rows("evidence/queue_s40hl_cqi4_final100.csv")
    head_ppo = select(headline, "PPO")
    headline_ok = record("base", "historical_headline_20000_20099", "in_distribution",
                         head_ppo, 20000, 20100, "Historical headline; already inspected, not a new blind test.")
    if headline_ok:
        for sched in sorted({r["sched"] for r in headline} - {"PPO"}):
            control = select(headline, sched)
            if coverage(control, 20000, 20100)["complete"]:
                stats += paired("historical_headline_20000_20099", "in_distribution",
                                "base", sched, head_ppo, control)
            else:
                issues.append(f"Historical headline control incomplete: {sched}")
    for recipe in ("lrann", "narrow"):
        record(recipe, "historical_headline_20000_20099", "in_distribution", {}, 20000, 20100,
               "No preserved result in this band; compare recipes in the separate reused21000 band.")

    common, controls = {}, {}
    for recipe, pattern in (("base", "seedreplicate_final100_h21_2024_[abc].csv"),
                            ("lrann", "seedreplicate_final100_h21_LRANN.csv"),
                            ("narrow", "seedreplicate_final100_h21_NARROW.csv")):
        rows = read_rows("evidence/" + pattern)
        ppo, control = select(rows, "PPO"), select(rows, "SUS+CQI")
        ok = record(recipe, "reused_comparison_21000_21099", "in_distribution", ppo, 21000, 21100,
                    "Previously used comparison band; fixed policies, not fresh blind evaluation.")
        if ok:
            common[recipe] = ppo
            datasets[f"common_{recipe}"] = {"reward_mean": statistics.mean(float(r["reward"]) for r in ppo.values()),
                                          "n": len(ppo), "best_updates": sorted({int(r["best_update"]) for r in ppo.values()})}
            if coverage(control, 21000, 21100)["complete"]:
                controls[recipe] = control
                stats += paired("reused_comparison_21000_21099", "in_distribution", recipe,
                                "SUS+CQI_same_job", ppo, control)
            else:
                issues.append(f"Same-job SUS control missing/incomplete: {recipe}")
    datasets["common_control_integrity"] = {}
    if "base" in controls:
        for recipe in ("lrann", "narrow"):
            if recipe not in controls:
                continue
            mismatches = [(seed, metric) for seed in controls["base"] for metric in METRICS
                          if controls["base"][seed][metric] != controls[recipe][seed][metric]]
            datasets["common_control_integrity"][recipe] = {"compared_metric_values": 100 * len(METRICS),
                                                            "mismatches": len(mismatches)}
            if mismatches:
                issues.append(f"Same-job common-world SUS controls differ: {recipe} ({len(mismatches)} values)")
    common_note = ("Fixed-policy episode uncertainty, not across-training-seed recipe uncertainty. "
                   "Historical Base used serial replay; LRANN/NARROW used batched replay. "
                   "This comparison band was already used during analysis.")
    for a, b in (("lrann", "base"), ("narrow", "base"), ("lrann", "narrow")):
        if a in common and b in common:
            stats += paired("reused_comparison_21000_21099", "in_distribution", a, b,
                            common[a], common[b], common_note)

    extended = read_rows("evidence/holdout21_baselines_w[0-4].csv")
    for sched in sorted({r["sched"] for r in extended}):
        control = select(extended, sched)
        if coverage(control, 21000, 21100)["complete"]:
            for recipe, ppo in common.items():
                stats += paired("reused_comparison_21000_21099_extended_baselines", "in_distribution",
                                recipe, sched, ppo, control,
                                "Separate evaluation jobs/CPU thread counts; same-job SUS controls differ slightly numerically.")
        else:
            issues.append(f"Extended baseline incomplete: {sched}")

    replicate_means = []
    for seed in (2024, 3024, 4024, 5024):
        rows = read_rows(f"evidence/seedreplicate_final100_s{seed}_[abc].csv")
        ppo, control = select(rows, "PPO"), select(rows, "SUS+CQI")
        if coverage(ppo, 20000, 20100)["complete"] and coverage(control, 20000, 20100)["complete"]:
            recipe = f"base_s{seed}"
            stats += paired("historical_replication_20000_20099", "in_distribution", recipe,
                            "SUS+CQI_same_job", ppo, control)
            replicate_means.append({"training_seed": seed,
                                    "reward_mean": statistics.mean(float(r["reward"]) for r in ppo.values())})
    datasets["base_training_seed_means"] = replicate_means
    if len(replicate_means) > 1:
        values = [r["reward_mean"] for r in replicate_means]
        datasets["base_across_training_seed"] = {"n_training_seeds": len(values),
                                                "mean_reward": statistics.mean(values),
                                                "sample_std_reward": statistics.stdev(values)}
    for recipe in RECIPES:
        n = len(replicate_means) if recipe == "base" else 1
        matrix.append({"recipe": recipe, "evaluation": "independent_training_seeds",
                       "world": "in_distribution", "status": "available" if n > 1 else "single_seed_only",
                       "n": n, "expected_n": None, "complete": n > 1, "missing_seeds": [],
                       "unexpected_seeds": [], "notes": "Episode CIs cannot replace independent training-seed replication."})

    ood_base = read_rows("evidence/OOD/results/ood_all_n100.csv")
    ood_narrow = read_rows("evidence/ood_policy_narrow_g[0-2].csv")
    ood_extra = read_rows("evidence/ood_newbaselines_*.csv")
    for world in WORLDS:
        base, narrow = select(ood_base, "PPO", world), select(ood_narrow, "PPO", world)
        base_ok = record("base", "historical_ood_30000_30099", world, base, 30000, 30100,
                         "Merged Base evidence stores reward rounded to0.1; other metrics unavailable in this table.")
        narrow_ok = record("narrow", "historical_ood_30000_30099", world, narrow, 30000, 30100,
                           "D26 changes policy deadline normalization to/6; strict /12 rerun required."
                           if world == "D26" else "Preserved NARROW evaluation, already inspected.",
                           invalid=world == "D26")
        record("lrann", "historical_ood_30000_30099", world, {}, 30000, 30100,
               "No packaged LRANN OOD result; shared evaluator can generate it.")
        if base_ok and narrow_ok:
            stats += paired("historical_ood_30000_30099", world, "narrow", "base", narrow, base,
                            "Base reward rounded to0.1; separate evaluation jobs, fixed-policy contrast.")
        for recipe, ppo, ok in (("base", base, base_ok), ("narrow", narrow, narrow_ok)):
            if not ok:
                continue
            for source, suffix in ((ood_base, ""), (ood_extra, "_extended_baselines")):
                for sched in sorted({r["sched"] for r in source if r["world"] == world} - {"PPO"}):
                    control = select(source, sched, world)
                    if coverage(control, 30000, 30100)["complete"]:
                        stats += paired("historical_ood_30000_30099" + suffix, world, recipe,
                                        sched, ppo, control,
                                        "Historical same-world episodes; source precision/thread settings differ across jobs.")
                    else:
                        issues.append(f"OOD control incomplete: {world}/{sched}")

    artifacts = {}
    for recipe in (*RECIPES, "base_s3024", "base_s4024", "base_s5024"):
        required = ("config.json", "ckpt/best.pt", "ckpt/latest.pt", "csv_logs/eval_metrics.csv",
                    "csv_logs/ppo_metrics.csv", "csv_logs/env_metrics.csv")
        absent = [p for p in required if not package_path(f"artifacts/{recipe}/{p}").is_file()]
        artifacts[recipe] = {"complete": not absent, "missing": absent}
    pilot_paths = ("evidence/queue_s40hl_cqi4_final40.out",
                   *(f"evidence/OOD/results/ood_pilot40k_{tag}.csv" for tag in "abc"),
                   *(f"evidence/OOD/results/ood_userscale_k{k}.csv" for k in (8, 48, 60)))
    pilot_missing = [p for p in pilot_paths if not package_path(p).is_file()]
    datasets["threshold_pilot_lineage"] = {
        "expected_files": list(pilot_paths), "missing_files": pilot_missing,
        "complete": not pilot_missing,
        "notes": "Independent 40000-band threshold selection supporting all recipes' common evaluation worlds. "
                 "The older queue_sus_threshold_sweep.csv does not replace the CQI4 pilot."}
    return {"provenance": provenance, "artifacts": artifacts, "data_issues": issues,
            "datasets": datasets, "paired_statistics": stats, "missing_data_matrix": matrix,
            "interpretation": [
                "All input tables are historical evidence. The reused21000 band is not a new blind holdout.",
                "Paired Student-t95% CIs describe episode differences for fixed policies; no multiple-comparison adjustment.",
                "No strict NARROW D26 result is emitted. Its historical100rows have a preprocessing mismatch.",
                "LRANN OOD and independent LRANN/NARROW training-seed replications are absent.",
                "NARROW fixed speed20/p.30 differ from wide-world means22.5/.325; diversity is not the only changed factor.",
                "Integrity checks establish copied-artifact provenance, not physical-model correctness."]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/evidence_summary",
                        help="Output directory relative to this package, or an absolute directory inside it.")
    args = parser.parse_args()
    try:
        out = package_path(args.output_dir)
    except ValueError:
        parser.error("Output directory must stay inside this package")
    relative = out.relative_to(ROOT)
    if not relative.parts or relative.parts[0] in {"artifacts", "evidence", "provenance"}:
        parser.error("Output must be a separate directory; historical evidence may not be overwritten")
    summary = build_summary()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(out / "paired_statistics.csv", summary["paired_statistics"])
    write_csv(out / "missing_data_matrix.csv", summary["missing_data_matrix"])
    passed = summary["provenance"]["passed"] and not summary["data_issues"] and all(
        a["complete"] for a in summary["artifacts"].values())
    print(f"Historical artifact integrity: {'PASS' if passed else 'CHECK REQUIRED'}")
    print(f"Verified {summary['provenance']['verified_file_count']} files; "
          f"wrote {len(summary['paired_statistics'])} paired metric rows to {out}")
    print("Missing/invalid: LRANN OOD; LRANN/NARROW training-seed replication; strict NARROW D26.")
    if summary["datasets"]["threshold_pilot_lineage"]["missing_files"]:
        print("Threshold pilot lineage is incomplete; see summary.json for missing preserved files.")
    print("The 21000 band is reused evidence, not a fresh blind test.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
