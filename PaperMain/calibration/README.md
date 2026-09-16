# CQI4 calibration

`cqi4.py` implements `cqi4-feasible-v1`. The default `paper_calibration.py`
entrypoint uses the frozen Base HL CQI4 world, neutralizes the old beta, samples
only feedback-eligible UEs before precoding, and retains genuine zero-MI outages.
It samples two uniform groups per depth every seven slots. A selected RBG with
fewer than m eligible UEs is skipped, not redrawn. Queue state and true CSI do
not influence membership. This defines an all-new full-capacity reference, not
the distribution of transmissions chosen by a learned policy or HARQ.

The fit uses the 10th percentile (NumPy linear interpolation), rounds to four
decimals, and is saved before holdout starts. Every sampled beta-scaled capacity
must pass runtime epsilon in both phases; otherwise the run fails closed.
Thus final membership, depth and power remain consistent with the sampled group.
A future setting that fails this condition requires a separately specified
closure-aware protocol; this implementation does not clip beta or discard such
holdout samples. Actual MI is saved and ACK uses the runtime 1e-6-bit tolerance.

Default ranges: calibration 50000..50035 (sampler seed 777), holdout
70000..70011 (sampler seed 20250713), world seed 2024, 1000 slots per episode.
The episode-cluster percentile bootstrap uses 10000 draws, seed 42. Empty
resampled denominators raise an error. The 90% target is a calibration objective,
not a holdout acceptance rule or a promise of policy ACK/NR BLER performance.
No holdout-based retuning takes place. Shortened runs cannot become deployable.

`profile.py` validates protocol, complete phases, disjoint ranges, beta, reference
config and source hashes. Training/evaluation require explicit
`--calibration-profile results/cqi4_hl_corrected_v2/summary.json` to opt in.
Training stores the immutable profile in its manifest and restores it on resume.
Existing archived results remain unchanged. Source hashes include calibration
support modules; changes to them require a new calibration profile.

`beta_m_calib_holdout.py` is unchanged historical source from commit
1564c45cbc56aabe341b896558b9ba2bbea699ff. Use `paper_calibration.py --mode archived`
for the old CQI4 diagnostic. Its sampler includes zero-capacity members and
converts undefined ratios to zero, with the wrong nominal group power/depth.
Removing zero rows afterward cannot correct that sampling procedure.
