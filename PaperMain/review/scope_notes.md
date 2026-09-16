# Paper extraction scope — read-only analysis (2026-09-09)

Source inspected: `work/ML-DRL-Scheduler`, HEAD `1564c45cbc56aabe341b896558b9ba2bbea699ff`. Original files were not modified.

## Decision (updated to user scope: Base + LRANN + NARROW)

The existing paper's main experiment is **Run4/QueuePostRZF_S40HL_CQI4**, training seed 2024, checkpoint **best@409**. Evidence: its `RUN_MANIFEST.json`; `docs/RESEARCH_LOG.md` 2026-08-10 main-result entry; `docs/PPO_PAPER_NOTES.md` 2026-08-24 four-training-seed section; the exact file paths in `paper_fig_main_result.py` and `seedreplicate_final100.py`.

The three identically trained seed replicates **3024/4024/5024** are part of main-result reproducibility, not exploratory runs. They ran **866 updates** to match the realized 2024 budget (0..865; original nominal budget was 1500). Their best updates are 489/849/389. Configs/checkpoints/logs are remote-only in current Git snapshot: fetch them from the server.

**User explicitly included LRANN and NARROW in the new paper workspace.** Treat Base, LRANN, NARROW as three separate recipes; Base remains the origin of the existing main paper figure. **NARROW_LRANN, FullRank, NUS*, ANN1500 remain excluded.** The latest commit/result being NARROW_LRANN does not make it main. Retain at most small historical pointers to excluded variants.

World: K=32, active count U{16..32}, speed U(5,40) km/h, arrival probability U(0.15,0.50), queue capacity 8, deadlines U{3..12}, p_csi .6, Type-II-like sparse56-bit PMI, CQI nr4bit, post-RZF LA, rbg_major decode, frozen beta=(1.0018,.7499,.6592,.6058), PPO entropy .02, no LR/entropy annealing, fresh init.

## Bounded primary manifest

### 14 required root source modules

`config.py`, `channel.py`, `codebook.py`, `csi.py`, `phy.py`, `traffic.py`, `transmission.py`, `la_planner.py`, `env.py`, `baselines.py`, `metrics.py`, `policy.py`, `ppo.py`, `train_phase2.py`.

Do **not** expose old `run_phase1.py` or `eval_phase2.py` as paper entry points. README explicitly says eval_phase2 only reconstructs Run1 presets and cannot reproduce Run4. Keep all necessary config dataclass fields even if their historical presets are not publicly documented; deleting old fields can break checkpoint/config compatibility.

### Six selected run artifacts

For each of the following run directories, copy `config.json`, `RUN_MANIFEST.json` if present, `ckpt/best.pt`, `ckpt/latest.pt` if available, and complete `csv_logs/`. Checkpoints/logs are relatively small compared with old campaign histories; best supports evaluation, latest supports resumption. Distinguish preserved historical results from future execution output directories.

- `Run4/QueuePostRZF_S40HL_CQI4`
- `Run4/QueuePostRZF_S40HL_CQI4_s3024`
- `Run4/QueuePostRZF_S40HL_CQI4_s4024`
- `Run4/QueuePostRZF_S40HL_CQI4_s5024`
- `Run4/QueuePostRZF_S40HL_CQI4_LRANN` (best@539)
- `Run4/QueuePostRZF_S40HL_CQI4_NARROW` (best@739)

LRANN: same wide training world, 866 updates, batched replay, LR linearly .0003→0 over866updates (`--lr_final 0.0 --lr_decay_updates 866`). NARROW: 866 updates, batched replay, constant LR, fixed speed20 km/h, p_arrival=.30, n_active=24; evaluation deliberately uses the wide Base world. Both configs/checkpoints are remote-only. Original recipe wrappers `Run4/_wrap_LRANN.sh` and `_wrap_NARROW.sh` are provenance only. Do not label the NARROW world as the wide-world mean: wide means are22.5 km/h and p=.325, not20 and.30.

Preserve `Run4/QueuePostRZF_S40HL_CQI4/paper_main_result{,_appendix9}.{pdf,png}` under results/figures if helpful. Other PNGs in this run are optional diagnostics, not necessary for executable reproducibility.

Do not directly use `_wrap_*.sh` in the new tree. Original seed2024 wrapper points to the removed `_cqi4dev` worktree and all wrappers hardcode server paths/pinned Git history, GPU IDs, watchdog markers and restart loops. Preserve wrapper text only as provenance (6 files) and create one portable training entry point with explicit seed, budget, device, run destination and resume settings.

### Primary evaluation inputs and scripts

Required historical inputs:
- `Run4/_analysis/queue_s40hl_cqi4_final100.csv` (900 rows; 9 schedulers x 100 reserved episodes)
- `Run4/_analysis/queue_s40hl_cqi4_final40.out` (contains independent 40000..40007 SUS-threshold pilot; final100 uses T*=.75)
- `Run4/_analysis/seedreplicate_final100_s{2024,3024,4024,5024}_{a,b,c}.csv` (12 shards, main replicate assessment on **20000..20099**)
- **Required common comparison**: `seedreplicate_final100_h21_{2024,3024,4024,5024}_{a,b,c}.csv`, `seedreplicate_final100_h21_LRANN.csv`, `seedreplicate_final100_h21_NARROW.csv`, `eval_LRANN.log`, `eval_NARROW.log`. This21000..21099band is the common Base/LRANN/NARROW comparison; do not mix it with Base20000..20099headline rows. Base2024 mean4786.68, LRANN4839.36, NARROW4796.54 per previous audit; calculate paired recipe differences on these identical episodes. Raw across-seed Basereplicates help contextualize small recipe differences. OOD data also exist for NARROW in `ood_policy_narrow_g{0,1,2}.csv` (100 episodes x9worlds). NARROW D26 uses changed /6 policy preprocessing and requires strict /12 rerun before paper comparison. LRANN OOD evidence is absent.
- Optional main mechanism instrumentation: `queue_s40hl_cqi4_final100_rejection.csv`, `queue_s40hl_cqi4_final100_aug.csv` (and their generating scripts below) if rejection/resource diagnostics enter the paper.

Required original script references to adapt:
- `Run4/_analysis/scripts/queue_s40hl_cqi4_final50.py`: despite name, takes start/end and produced the 100-episode final result in shards.
- `Run4/_analysis/scripts/queue_s40hl_cqi4_final40.py`: threshold pilot + validation protocol, also a source of held-out leakage cautions.
- `Run4/_analysis/scripts/seedreplicate_final100.py`: fixed eval-world seed2024, policy checkpoint varies; only compares PPO to SUS+CQI.
- `Run4/_analysis/scripts/paper_fig_main_result.py`: main7 and appendix9, 2x2 reward/goodput/miss/depth.
- Optional `queue_s40hl_cqi4_final100_rejection.py`, `queue_s40hl_cqi4_rejection_stats.py`.

**World seed must stay 2024 for comparisons across training seeds**; loading each policy's seed-specific cfg directly as the evaluation world yields different episodes. Seed integers in filenames are episode indexes; the simulator combines cfg.seed with episode index internally. `seedreplicate_final100.py` contains a guard for this. Training runs have nonoverlapping episode ranges because 1000-spaced training seeds exceed 866-update budgets.

Paper display set is **PPO + {SUS,SU} x {CQI,PF,Random}**, total7. DPF pair gives appendix9. Full12 heuristic grid (+PPO=13) should remain available for fair baseline verification, even when not all are plotted. Some figure/eval docstrings still incorrectly say DPF is in the main7 and Random appendix-only: executable `MAIN7` and user-final 2026-08-10 log establish the actual final set.

### Calibration (essential, not a discarded Run4 variant)

- `Run4/_analysis/scripts/audit_probes/beta_m_calib_queue_s40hl_cqi4.py`
- `Run4/_analysis/scripts/audit_probes/beta_m_calib_holdout.py` (helper import)
- `Run4/_analysis/beta_m_calib_queue_s40hl_cqi4.out`

Current CQI4 calibration script reads old `QueuePostRZF_S40HighLoad/config.json` and overrides cqi_mode. In extracted code, read main CQI4 config and clear beta fields instead; verify equivalent world config. No need to bring the continuous run checkpoint/whole directory solely for this dependency. Calibration seeds50000..50035 (36 episodes), holdout70000..70011 (12), random grouping seed777/20250713. Main frozen beta values must be retained as historical provenance even if re-audit finds calibration defects.

### OOD supporting the same main policy

Recommended include as one bounded supporting evaluation, because it demonstrates the selected policies' generalization; these are not separate training runs. Final table `Run4/_analysis/OOD/results/ood_all_n100.csv` has 6300 rows =9worlds x7schedulers x100episodes (30000..30099). Fresh SUS pilot40000..40007, same main best@409, frozen beta.

Worlds/T*: P055 .75; P010 .80; V60max .75; CSI02 .70; D26 .80; STORM2 .75; K8 .80; K48 .75; K60 .80. D26 must use environment deadline U[2,6] while policy preprocessing retains training normalization /12. Population K changes also offered load; avoid claiming isolated K scaling.

Minimal original script references (prefer consolidate into one portable configurable OOD evaluator):
- `OOD/scripts/ood_grid_extend100.py` (latest6world definitions, strict D26 intent; extend seed range for all100)
- `OOD/scripts/ood_userscale_ext100.py` (K8/48/60, seed-range arguments)
- `OOD/scripts/ood_pilot40k_cqi4.py`, `ood_pilot40k_musweep.py` (12-grid pilot/champion evidence)
- `OOD/scripts/ood_userscale_cqi4.py` (K pilot before test)
- `OOD/scripts/ood_d26_strict_rerun.py` (strict fix reference)

Required supporting evidence beyond merged100 CSV: `OOD/results/ood_pilot40k_{a,b,c}.csv`, `ood_pilot40k_musweep.csv`, `ood_userscale_k{8,48,60}.csv` (contain K pilot), and their `.out` where command context is needed. For complete lineage of merged table, also preserve raw final pieces (small): `ood_paper_grid_cqi4_{a,b,c}.csv`, `ood_reeval_sus_t{1,2}.csv`, `ood_grid_fill_{a,b,c}.csv`, `ood_d26_strict.csv`, `ood_grid_ext100_g{1..5}.csv`, `ood_userscale_ext100_{a,b,c}.csv`. Original merge script is not clearly preserved; new extraction should supply a reproducible merge/validation utility or simply generate full100 in one evaluator.

Omit old exploratory OOD probes/scaling_zeroshot/load_response and legacy ood_p_arrival from executable paper area; retain small history pointers if necessary. Existing OOD README mixes final100 with stale n40/tie and pilot reservation text, so write fresh protocol documentation instead of adopting it verbatim as authoritative.

### Documentation/runtime

- Original `requirements.txt` (not fully pinned) as provenance; record exact remote runtime versions and make an environment lock for reconstruction. Plotting additionally needs matplotlib, currently not explicitly listed.
- `docs/SYSTEM_MODEL.md`, `docs/CQI_QUANTIZATION.md`, `docs/BETA_M_RATIONALE.md`, `docs/PPO_PAPER_NOTES.md` may be preserved as reference but contain stale sections and claims that must be re-audited before paper use.
- Replace sprawling README/RESEARCH_LOG/RUNS with a concise new paper README, manifest, provenance/historical notes, executable commands, output map and unresolved audit findings.
- Existing tests `tests/test_replay_batch.py` / `test_ppo_update_batched.py` concern optional acceleration (paper default serial), while `test_cross_tree_identity.py` depends on original tree paths. Keep if adapting as optional checks; paper subset needs focused standalone execution/isolation/protocol checks first.

## Isolation warning

Every inspected analysis script hardcodes `/home/MYH/ML_DRL_Scheduler` in chdir and/or sys.path; copied files can silently read and execute originals unless rebased. Verify running new scripts from an unrelated cwd, confirm imported module paths point inside new folder, ensure no symlinks/absolute original-tree accesses, and verify historical results are never default write destinations. Shell wrappers additionally require the original Git object history and are not standalone.

The prior audit's CQI0 calibration and D26 preprocessing findings apply to this main subset. Do not label extracted files as validated simply because most previous runs were removed.

## Common-band recipe statistics (direct CSV recomputation)

Using exactly100 episode indexes21000..21099, paired reward differences and Student-t critical t99=.975=1.9842169515:

| Recipe | Mean reward |
|---|---:|
| Base2024 | 4786.676244 |
| LRANN2024 | 4839.359654 |
| NARROW2024 | 4796.543784 |

| Paired contrast | Mean difference | 95% CI | Episode wins |
|---|---:|---|---:|
| LRANN − Base | +52.6834 | [+36.7425,+68.6243] |76/100|
| NARROW − Base | +9.8675 | [−10.1722,+29.9073] |54/100|
| LRANN − NARROW | +42.8159 | [+26.2780,+59.3537] |67/100|

These CIs describe held-out episode uncertainty for the fixed trained policies. They do not establish improvement across random training initializations: LRANN/NARROW are each single-training-seed and have batched execution/threads different from historical Base. NARROW does not show a resolved reward change from Base in this band. Keep actual NARROW training constants rather than relabeling them as exactly matched means.
