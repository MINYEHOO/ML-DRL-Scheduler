# PaperMain execution audit and packaging contract

Scope: Base `QueuePostRZF_S40HL_CQI4`, LRANN, NARROW; combined NARROW_LRANN excluded by current user instruction. Original research files were not modified. Training integration changes are confined to `work/PaperMain/paper_train.py`, `work/PaperMain/train_phase2.py`, and `work/PaperMain/tests/test_paper_train.py`.

## Executed checks in this subtask

- Historical base checkpoint `best.pt` update 409 strictly loads in its complete main Config.
- Base archived config and checkpoint lack `ppo_batched_replay`; current explicit default resolves to **False**, consistent with its original wrapper. Preserve sequential updates for Base.
- The same base tensors also strictly load into `phase4_queue_config()` despite incompatible environment/algorithm settings. Thus successful tensor loading cannot establish correct resume semantics.
- Two-step GAE analytic calculation with time-limit bootstrap passed (`[4.4826, 3.83]` advantages; `[4.9826, 4.08]` returns).
- 14 focused portable-launcher guard tests passed (including CPU skip, real-allocation/synchronize CUDA preflight and busy-GPU refusal): recipe configuration retention; base sequential vs LRANN batched; LRANN horizon retained when shortening total execution; NARROW's actual fixed world; resume restores configuration and schedule; explicit seed/threads/smoke changes rejected; source changes rejected; run-config changes rejected; schedule changes rejected; archives and older best checkpoints cannot be resumed in-place; path escapes/collisions rejected; numerical thread environment configured before Config's NumPy import.
- Real LRANN archived configuration successfully passed portable launcher `--dry-run --smoke-slots 32 --num-updates 1`; this writes no run files.

Evidence: `work/paper_execution_checks.py`, `work/paper_execution_checks.json`, `work/paper_train_dryrun.json`. Server execution tests are performed by parent task and must be combined separately with this audit.

## Added launch behavior

Fresh:

```bash
python paper_train.py --recipe base --name base_seed2024 --device cuda --gpu 0 --threads 4
python paper_train.py --recipe lrann --name smoke_lrann --smoke-slots 32 --num-updates 1
```

Resume:

```bash
python paper_train.py --recipe lrann --resume runs/smoke_lrann/ckpt/latest.pt --num-updates 2
```

The requested number of updates is the total target, not additional updates. The launcher accepts only `base|lrann|narrow` and only new outputs under `PaperMain/runs/<name>`. Frozen `artifacts/` remain input-only. Resume accepts only latest checkpoints from new package runs, never an archived `artifacts/*` or `best.pt`. Names/paths, recipe, config, source hashes, schedule, explicit CLI overrides and runtime are validated before writes. A symlinked `runs/` is rejected. Checkpoint config is compared against the run manifest before model loading.

`paper_manifest.json` records the complete resolved Config (including defaults absent from old configs), original artifact configuration hash, hashes for all 15 training sources, fixed schedule, target updates, execution device/GPU/threads, Python/library/platform/runtime metadata and whether the run is an execution smoke. `config.json` and checkpoint configuration must agree with it on resume. Source or runtime changes require a separate run. Increasing total updates is allowed without resetting a stored LR schedule.

`train_phase2.main(config_override=None)` now supports a complete Config injection after its historical CLI adjustments and before constructing environment/model. The launcher supplies that Config and aligns the driver's seed. The original control flow, environment, policy, PPO, loss, checkpoint tensor format and optimizer are unchanged. The launcher replaces the incidental Git stamp with an aggregate training-source SHA256 so copying the package beneath another Git root cannot silently misidentify its source.

Numerical libraries are imported after CPU thread/GPU setup. TensorFlow memory growth is enabled so TensorFlow and PyTorch can share the selected GPU. If CUDA is explicitly requested, the launcher performs a real allocation and synchronization before writing run files; an unavailable or Exclusive_Process-busy GPU produces a controlled error before creating run outputs. Exact runtime information is restored on resume.

Recipe details:

| Recipe | World | Replay | LR | Original launch target |
|---|---|---|---|---|
| Base | active UE 16..32; speed U(5,40); arrivals U(.15,.50) | sequential | fixed 3e-4 | 1500 updates |
| LRANN | same wide world | batched | 3e-4 to 0, horizon 866 | 866 updates |
| NARROW | active UE 24; speed20; arrivals .30 | batched | fixed 3e-4 | 866 updates |

Base's historical run was stopped at update866; 1500 is the original wrapper's intended target, not an assertion that archived Base trained1500updates. These sources/configs preserve a recipe, not bitwise historical retraining under changed hardware/library versions. A smoke shortens episode length, epochs, minibatch size and validation episode count while retaining all32UE and main PHY dimensions. Smoke results are not paper-performance evidence.

## Confirmed defects and limitations still relevant

### E1 — P1 original resume silently changes the world (mitigated by launcher)

`train_phase2.py:572–599` loads model/optimizer/RNG but merely prints warnings about changed Config. Original Config is built from CLI presets at394–474 and existing config.json retained at519–526. Base checkpoint weights load into queue-default with `nr4bit→continuous`, `post_rzf→legacy`, `rbg_major→layer_major`, wide traffic→fixed. New launcher restores and validates full Config. Direct historical CLI remains unsafe; document portable launcher as supported training entrypoint.

Original schedule CLI at688–729 is not independently persisted in checkpoints. New launcher restores fixed recipe schedule and validates manifest. LRANN's horizon stays866 even when a smoke only executes1or2updates.

### E2 — P2 failed batched verification still applies faulty gradient

`ppo.py:307–321` computes batched backward, then on verification failure changes local `batched=False` but skips sequential recomputation. `optimizer.step` later applies that gradient. At the next update288–290 re-reads unchanged cfg and batching reactivates. This applies to LRANN/NARROW; historical Base is sequential. Earlier fault-injection evidence remains applicable because copied PPO source is unchanged. No algorithm change was made by this subtask. This needs correcting or explicit unresolved reporting before making resilience claims for those recipes.

### E3 — P2 final baseline set must exceed training-eval grid

`all_baselines` at815–842 returns the historic12-baseline queue grid. It excludes later paper primaries `SUS-RPS`, `SUS+CQI-Feasible`, `PF-Greedy-SDS`. The standalone paper evaluator must instantiate those explicitly. Copying only training validation CSVs would omit the current strongest controls and overstate PPO's advantage. The parent evaluator handles this separately.

### E4 — P2 inconsistent evaluator thread settings

Original `seedreplicate_final100.py:36–45` uses6 threads; `holdout21_baselines.py:15–19` uses4. The archived common SUS+CQI results consequently are not identical (the earlier audit measured98/100different rows; thread causality itself was not experimentally isolated). A single evaluator must use one environment/config/seed family/thread setting for all policy/baseline rows; record that runtime in its own evaluation manifest. Use seed2024 as worldseed for all learned training-seed replicas; otherwise each policy faces a different testworld.

### E5 — scientific scope of NARROW comparison

NARROW speed20 and arrival.30 are below wide means22.5 and.325. It changes average load/mobility as well as diversity. Preserving the actual recipe is correct for reproducing existing research; changing it to the mean would create a new experiment. Do not describe existing NARROW as an isolated domain-randomization ablation.

### E6 — archival status vs evaluation reproducibility

The original Base wrapper hardcodes `_cqi4dev`, GPU4, full path, old Git pin and1000restart loop. LRANN/NARROW wrappers hardcode home path/GPU and repeatedly restore command-line flags. These are provenance only; they should not be the runnable package entrypoints. The new launcher removes that requirement without deleting original evidence.

## Focused runtime test recommendations

1. Each frozen best checkpoint must strictly load into its own policy Config, then run a short32UE episode. Evaluate in a fixed Base world when making wide comparisons; keep policy preprocessing from its training config.
2. Run one fresh update plus evaluation/checkpoint and resume to update2 through the real launcher. Verify config equality, LRANN update1 LR=`3e-4*(1-1/866)`, manifest horizon866, checkpointupdate1, normalizer/optimizer/RNG retained, CSV appended.
3. Stronger continuation check: compare a same-seed two-update uninterrupted LRANN smoke with one-update+resume to2. Compare model, Adam states and torch RNG. This tests continuation semantics rather than only successful loading.
4. Existing `tests/test_replay_batch.py` and `tests/test_ppo_update_batched.py` are useful mathematical checks; point TEST_RUN to copied Base artifact. The latter exercises short final minibatch and KL-frozen actor paths. `test_cross_tree_identity.py` is irrelevant to a standalone package because it assumes an external `_batchdev` tree.
5. Explicitly evaluate current primary baseline set under a single pinned execution protocol. Random baselines should be instantiated with reproducible episode-specific policy RNG if results must be independent of seed batching/order.
6. The calibration validity/first-ACK issue, common traffic trace dependence and wireless modelling claims are covered by the wireless agent; do not replace those checks with successful PPO training smoke.
