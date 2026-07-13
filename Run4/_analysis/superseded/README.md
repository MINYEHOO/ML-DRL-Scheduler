# Superseded analysis artifacts (kept for audit history)

- `new_la_baselines.global_beta.invalid-depth-metrics.csv` -- global-beta
  world summary whose per-depth `first_ack_m*` columns are INVALID: they
  were computed as the mean over episodes of per-episode rates with empty
  depth bins counted as 0 (the "SUS m1 = 36%" artifact, audit round 6).
  Scalar columns (reward etc.) are fine. Superseded by
  `../new_la_baselines_global.csv` (pooled sum-acks/sum-units + raw
  per-seed file).
- `new_la_baselines_betam.pre-round8.csv` -- beta_m world summary generated
  before the round-8 script consolidation; numbers match the replacement
  `../new_la_baselines_betam.csv` (pooled), kept only so the audit trail of
  file generations is complete.
