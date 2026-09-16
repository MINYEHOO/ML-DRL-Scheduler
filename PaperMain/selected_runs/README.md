# Selected additional training runs

This directory is the paper-facing shortlist of useful training runs produced after the
`PaperMain` split. Originals remain under `../runs/`; files here are immutable snapshots
that can be cloned without the server-local run tree.

## Selected runs

| Run | Role | Frozen checkpoint | Why retained |
|---|---|---:|---|
| 16 | Primary Bernoulli model | best@1989 | Broadest evidence: best PPO goodput in 7/10 evaluated worlds and gains over the strongest tested conventional baseline in every world. |
| 21 | High-load complement | best@649 | Better reward/deadline behavior at high offered load; PPO goodput lead in P055, STORM2, and K48. |
| 22 | NARROW generalization ablation | best@1469 | Shows that a narrower training distribution still generalizes competitively across the OOD suite. |
| 18 | FTP3 extension | best@639 | Completed 100-episode frozen evaluation; useful traffic-model extension with clear gains over SUS+CQI and PF-Greedy-SDS. |

Run 16 was continued from run 14. Its `parent_run14/` directory preserves the exact
parent checkpoint and metadata needed to audit that lineage.

## Evidence

- `evidence/bernoulli_ood/analysis_report.md`: interpretation and paired comparisons for runs 16, 21, and 22 across ID plus nine OOD worlds.
- `evidence/bernoulli_ood/combined_summary.*`: frozen-policy aggregate statistics.
- `evidence/ftp3_run18/summary.*`: 100-episode FTP3 evaluation for run 18.
- `selection.json`: source paths, checkpoint hashes, training endpoints, and selection decisions.

Training-time evaluation values across different seeds are not used to rank policies.
The shortlist requires a completed shared fixed-input evaluation and a distinct paper role.

## Deliberately not selected

- Runs 01–13, 15, 17, and 20 are audits, smoke checks, or intermediate versions.
- Run 14 is retained only as run 16 lineage because run 16 supersedes it as a result.
- Run 19 is a useful negative result: its FTP3 curve peaked early and degraded substantially by update 1999; its best checkpoint has no matching fixed evaluation.
- Runs 23–28 form the independent-seed study. They stay pending until all jobs finish and their frozen checkpoints are evaluated on the same inputs.
