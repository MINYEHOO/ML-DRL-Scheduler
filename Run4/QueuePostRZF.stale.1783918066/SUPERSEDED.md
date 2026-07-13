# SUPERSEDED — QueuePostRZF attempt 1 (archived, do not resume)

This is the FIRST launch attempt of the official post-RZF run
(launched 2026-07-13 02:46 KST on launch HEAD `22bda54`, root-Python
baseline `efcfac6`, ~10 updates). It was stopped and archived — NOT
resumed — when audit round 8 landed a root-Python fix (the `snr_m`
routing bug in `env.py`, commit `9ed28d0`), so that the official run's
entire history lives on a single immutable code baseline.

The snr_m fix does not touch the post_rzf path (behavior-identical for
this configuration); the restart was for provenance cleanliness only.

Official run: `Run4/QueuePostRZF/` (fresh start, same seed 2024, launch
HEAD `a67aac4`, root-Python baseline `9ed28d0`) — see its
`RUN_MANIFEST.json`.
