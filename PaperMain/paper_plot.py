#!/usr/bin/env python3
"""Render the historical Base result from archived episodes 20000–20099.

This figure contains one trained Base policy and the original baseline grid.
It is separate from the Base/LRANN/NARROW comparison on reused episodes 21000+.
Paths are resolved against this package; historical artifacts are never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parent
MAIN7 = ("PPO", "SUS+CQI", "SUS+PF", "SUS+Rnd", "SU+CQI", "SU+PF", "SU+Rnd")
ALL9 = ("PPO", "SUS+CQI", "SUS+PF", "SUS+DPF", "SUS+Rnd",
        "SU+CQI", "SU+PF", "SU+DPF", "SU+Rnd")
PANELS = (("(a) Episode return", "reward", "{:.0f}"),
          ("(b) Goodput [Mbit/s]", "goodput_mbps", "{:.1f}"),
          ("(c) Deadline miss rate", "deadline_miss_rate", "{:.2f}"),
          ("(d) MU depth", "mu_depth", "{:.2f}"))


def local_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_data(path: Path, order: tuple[str, ...]) -> dict:
    selected = {name: {} for name in order}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["sched"]
            if name not in selected:
                continue
            seed = int(row["seed"])
            if seed in selected[name]:
                raise ValueError(f"Duplicate scheduler/episode row: {name}/{seed}")
            values = {metric: float(row[metric]) for _, metric, _ in PANELS}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Nonfinite metric: {name}/{seed}")
            if row.get("train_seed") and name == "PPO" and int(row["train_seed"]) != 2024:
                raise ValueError("This plot is reserved for the historical Base training seed 2024")
            selected[name][seed] = values
    expected = set(range(20000, 20100))
    for name, episodes in selected.items():
        if set(episodes) != expected:
            raise ValueError(f"{name}: expected exactly archived episode indexes 20000–20099; "
                             f"missing={sorted(expected - episodes.keys())}, "
                             f"unexpected={sorted(episodes.keys() - expected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="evidence/queue_s40hl_cqi4_final100.csv",
                        help="Historical Base CSV inside this package; relative to the package root")
    parser.add_argument("--out", required=True, help="New output directory inside this package's results/")
    parser.add_argument("--appendix", action="store_true", help="Plot nine schedulers including the two DPF variants")
    args = parser.parse_args()
    try:
        source = local_path(args.input)
        out = local_path(args.out)
    except ValueError:
        parser.error("Input and output paths must remain inside this package")
    results = (ROOT / "results").resolve()
    if out == results or not out.is_relative_to(results):
        parser.error("--out must be a new subdirectory below this package's results/")
    if out.exists():
        parser.error(f"Refusing to overwrite existing output: {out}")
    order = ALL9 if args.appendix else MAIN7
    data = load_data(source, order)

    from scipy.stats import t as student_t
    n = 100
    critical = float(student_t.ppf(0.975, n - 1))
    stats = {}
    for name in order:
        stats[name] = {}
        for _, metric, _ in PANELS:
            values = [data[name][seed][metric] for seed in sorted(data[name])]
            stats[name][metric] = {"mean": statistics.mean(values),
                                   "ci95_halfwidth": critical * statistics.stdev(values) / math.sqrt(n)}

    # A disposable writable cache makes plotting independent of home-directory setup.
    with tempfile.TemporaryDirectory(prefix="paper-plot-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.0,
                             "axes.titlesize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
        colors = ["#c1272d" if name == "PPO" else "#2b6ca3" if name.startswith("SUS")
                  else "#2e8b57" for name in order]
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.7))
        fig.suptitle("Historical Base · training seed 2024", fontsize=7, y=0.985)
        for ax, (title, metric, fmt) in zip(axes.flat, PANELS):
            means = [stats[name][metric]["mean"] for name in order]
            halves = [stats[name][metric]["ci95_halfwidth"] for name in order]
            positions = list(range(len(order)))
            ax.bar(positions, means, color=colors, width=0.72, yerr=halves,
                   error_kw={"elinewidth": 0.6, "capsize": 1.4, "capthick": 0.6, "ecolor": "#333333"})
            ax.set_title(title, pad=3)
            ax.set_xticks(positions, order, rotation=60, ha="right", fontsize=5.0)
            ax.tick_params(axis="y", labelsize=5.2, length=2, pad=1)
            ax.tick_params(axis="x", length=0, pad=1)
            ax.margins(x=0.03, y=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(0.0, color="#888888", lw=0.5)
            for x, mean, half in zip(positions, means, halves):
                padding = 1 + (5 if args.appendix and x % 2 else 0)
                ax.annotate(fmt.format(mean), (x, mean + (half if mean >= 0 else -half)),
                            ha="center", va="bottom" if mean >= 0 else "top", fontsize=4.4,
                            xytext=(0, padding if mean >= 0 else -padding), textcoords="offset points")
        fig.text(0.5, 0.012, "Archived episodes 20000–20099 · means ± 95% CI", ha="center", fontsize=5.5)
        fig.tight_layout(pad=0.45, h_pad=0.85, w_pad=0.65, rect=(0, 0.035, 1, 0.96))
        out.mkdir(parents=True)
        stem = "historical_base_appendix9" if args.appendix else "historical_base_main7"
        fig.savefig(out / f"{stem}.png", dpi=300)
        fig.savefig(out / f"{stem}.pdf", metadata={"Title": "Historical Base result, training seed 2024"})
        plt.close(fig)

    caption = (
        "Historical Base policy (training seed 2024, best checkpoint update 409) and the original "
        f"{'nine' if args.appendix else 'seven'}-scheduler display set on 100 archived episode indexes "
        "20000–20099. Bars show means; error bars are Student-t 95% confidence intervals of each "
        "scheduler's mean absolute performance. PPO-versus-baseline conclusions require the paired "
        "episode-difference statistics, not overlap of these bars. This is a reproduction of the "
        "historical Base figure; it is separate from the Base/LRANN/NARROW comparison on the reused "
        "21000–21099 band. Extended feasibility-aware baselines are outside this historical display "
        "set; use paper_summary.py for those comparisons.\n")
    (out / "caption.txt").write_text(caption)
    manifest = {"figure": stem, "historical": True, "recipe": "base", "training_seed": 2024,
                "best_update": 409, "episode_indexes": [20000, 20099], "n_episodes": n,
                "input": source.relative_to(ROOT).as_posix(),
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "schedulers": list(order), "confidence_interval": "Student-t, 95%, absolute scheduler means",
                "statistics": stats, "caption": caption.strip(), "arguments": vars(args)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Historical Base figure: {out / (stem + '.pdf')}")
    print(f"PNG preview: {out / (stem + '.png')}")
    print("Three-recipe comparisons remain separate; see paper_summary.py.")


if __name__ == "__main__":
    main()
