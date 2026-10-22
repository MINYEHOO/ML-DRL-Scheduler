# Run 30 / 34 / 35: phase-invariant PPO evaluation

This patch enables variant-aware evaluation; it does not establish final training
completion or a performance gain. The inspected upstream revision is
`b5fdd83dfc844b33086ad88ce5fa24edab89f45f` (2026-09-17 06:52:16 UTC).

## What the committed evidence establishes

| Run | Saved run directory below `runs/` | Recipe | Initial LR | Published launch state |
| --- | --- | --- | --- | --- |
| 30 | `30_raw_invariant_lr1000_s2024_20260916_gpu3` | `raw_invariant` | 3e-4 | `running (resumed)` |
| 34 | `34_raw_phase_inv_lr1000_s2024_20260917_gpu5` | `raw_phase_inv` | 3e-4 | `running` |
| 35 | `35_raw_phase_inv_lr6_lr1000_s2024_20260917_gpu4` | `raw_phase_inv_lr6` | 6e-4 | `running` |

The corresponding `review/logs/<run-directory>_launch.json` files contain
`target_completed: false` and no successful final checkpoint-update record.
The older GPU 0/1 launch records for 34/35 are superseded by the GPU 5/4
assignments in the latest commit. Directory GPU suffixes are historical names;
evaluation can use any available device.

The launch records show identical complete configuration dictionaries for 30
and 34. Run 35 differs only in `ppo_learning_rate`. They share seed 2024,
1,000-update training targets, linear LR decay to zero over 1,000 updates,
1,000-slot episodes, Bernoulli arrivals, 16–32 active UEs out of 32, arrival
probability range 0.15–0.5, speed range 5–40 km/h, eight RBGs, rank limit four,
32 BS antennas, `nr4bit` CQI, `p_csi=0.6`, and post-RZF link adaptation.
All three use the same calibrated depth betas
`[1.0162, 0.8509, 0.7732, 0.728]`, the same core-source hashes, and the same
calibration-profile document hash
`0a8a6ff47855f24c33b8e348019ab30a4f73e05d7ef02e553eae18f5ee72e39e`.

`paper_train_rawpi.py` reports the 39/40 action-change diagnostic for Run 30.
That is a recorded diagnostic, not a measurement reproduced by this patch.
The source implements adjacent autocorrelations
`8*h[m]*conj(h[m+1])`, whose global phase cancels algebraically. Synthetic
invariance tests verify the model path but cannot establish trained goodput.
Autocorrelation input also changes dimension/representation (73 to 71 channels);
Run 34 versus 30 tests that complete representation change at fixed LR.
Run 35 versus 34 tests the LR change with the same architecture.

## Evaluation contract

Use `paper_run_eval.py` for these live-run checkpoints. It requires the actual
`runs/<name>/paper_manifest.json`, `config.json`, `ckpt/best.pt` (or `latest.pt`),
the authentic completed launch record, and the calibration reference snapshot.
Do not rewrite saved hashes, manufacture completion records, or use the launch
JSON as a substitute for a run manifest. Missing or incomplete inputs must fail.

Variant dispatch retains manifest/configuration/provenance validation and strict
state-dictionary loading. Source-digest checks apply to the frozen training and
variant sources. The patch adds evaluation files without changing those sources.
`paper_eval.py` remains the frozen-artifact evaluator. A new variant archive
must include its exact original `artifacts/<recipe>/paper_manifest.json` beside
the checkpoint; the evaluator retains its Base common/sensitivity-world
protocol. `paper_ood_eval.py` supports variant loading within its explicit OOD
protocol. The separate OOD campaign/merge tools remain the existing fixed
Run 16/21/22 campaign, so this patch does not declare a Run 30/34/35 OOD campaign.
Use live-run ID evaluation for the primary comparison.

The machine-readable proposal is `evaluation/raw_phase_30_34_35.json`:

| Setting | Value |
| --- | --- |
| Checkpoint policy | `best` for all three, selected only by saved internal validation |
| Full evaluation | episode IDs 150000–150099; 1,000 slots each |
| Diagnostic smoke | episode IDs 149000–149001; 40 slots; excluded from results |
| Sharding | 2 disjoint shards of 50 episodes per run |
| Action selection | deterministic scheduler; frozen parameters and normalization |
| Scheduling methods | PPO and the same six baselines from `paper_run_eval.SCHEDULERS` for every run |
| Main comparisons | Run 34 minus 30; Run 35 minus 34; optionally 35 minus 30 |
| Primary metric | goodput; report throughput, completion/miss/overflow, fairness and MU depth alongside |

These episode bands are **proposals**, not a claim that private/local experiment
history has been audited. Check the complete experiment history before use.
The preflight requires a real audit file at the path in the specification,
with `status: "validated"` and `allowed_episode_ranges` covering the smoke band
with purpose `"smoke"` and the full band with purpose `"id"`. The existing
evaluator separately rejects training, checkpoint-selection and calibration
overlap. No prevalidated audit file is supplied.

`paper_raw_compare.py` requires all configuration fields to match except the
predeclared LR values; this includes seed, reward weights, queue, channel,
traffic, policy and normalization settings. It also checks common training
budget/schedule, calibration, source hashes, episode IDs, scheduler set and
checkpoint policy, and then validates each checkpoint and strict CPU model
loading. Variant architecture/provenance is validated per run rather than
required to be equal across different architectures. No rollout is performed.

`best` updates may differ: report each update and SHA256. If equal final
training budget at a common update is the primary question, change the common
policy to `latest` before inspecting results; preflight then requires every update to equal 999.
Never select the better of `best` and `latest` using the held-out evaluation.

## Commands

Run from `PaperMain/` in the existing PyTorch/Sionna evaluation environment.
The command renderer works before real run inputs are available and explicitly
labels its output `commands_only_not_validated`:

```bash
python paper_raw_compare.py --print-commands --device cuda --gpu 5
```

After acquiring authentic completed run inputs and auditing the episode bands:

```bash
python paper_raw_compare.py --device cuda --gpu 5
```

This read-only preflight prints checkpoint hashes/updates and executable
`dry_run`, `smoke`, `evaluation`, and `merge` commands for all three runs. Execute
the printed dry runs first, then smoke runs, then full shards. Each command runs
in its own process, preventing variant-global setup from crossing models.
Outputs must be new directories. The same audit covers the two disjoint bands.
For example, the first full shard is:

```bash
python paper_run_eval.py \
  --run runs/30_raw_invariant_lr1000_s2024_20260916_gpu3 \
  --checkpoint best --episode-start 150000 --episodes 100 \
  --num-shards 2 --shard-index 0 --device cuda --gpu 5 --threads 4 \
  --history-audit results/raw_phase_30_34_35/inputs/history_audit.json \
  --out results/raw_phase_30_34_35/run30/shard0 --dry-run
```

Remove `--dry-run` only when running that shard. The command renderer supplies
all other runs and shard indices with the same conditions. Merge shards within
each run using `paper_run_eval_merge.py`; do not pass different runs to one merge
because their checkpoint/protocol identities intentionally differ.

For the cross-run comparison, pair the `ppo` rows of each run's
`mergedmetrics.csv` by `episode_idx`, verify exactly the declared 100 episodes,
and compute per-episode metric differences. Bootstrap whole paired episodes
(10,000 replicates, fixed seed 20261022), reporting the mean difference and a
95% percentile interval. Do not infer the cross-run interval by subtracting
the independently reported confidence bounds. The existing merger computes
PPO-versus-baseline intervals within each run, not cross-run intervals.

Same seed and episode IDs pair channel/topology/load conditions; scheduler
decisions can change queue admissions and shared RNG consumption. They do not
guarantee identical packet or CSI traces. These intervals describe evaluation
variation for one training seed, not variation across independently trained
models. The 40-slot smoke result is never a full-episode performance estimate.

## Verification and remaining evidence

```bash
python -m unittest discover -s tests -p 'test_paper_raw_compare.py'
python -m unittest discover -s tests -p 'test_paper_model_loader.py'
python -m unittest discover -s tests -p 'test_paper_run_eval.py'
python -m unittest discover -s tests -p 'test_paper_eval.py'
python -m unittest discover -s tests -p 'test_paper_ood_eval.py'
```

Tests use synthetic metadata/checkpoints and observations where appropriate;
they establish loading, rejection behavior, invariance and existing-path
compatibility. The actual three checkpoints, successful completion records,
and full held-out rollouts are still required to determine learning completion,
action invariance of those trained checkpoints, and comparative performance.
