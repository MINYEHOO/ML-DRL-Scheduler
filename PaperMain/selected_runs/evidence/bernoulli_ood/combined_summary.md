# Common-world Bernoulli OOD evaluation

10 worlds × 100 episodes × 9 schedulers; 9000 validated rows.

Three frozen best checkpoints; six historical baselines. Intervals are pointwise 95% Student-t intervals over episodes.

- Run 16: `runs/16_base_cqi4_lrrestart2000_s2024_20260910_gpu5`, best update 1989, SHA-256 `04ce34d239f4ac377b4fd27f40a39640d966094dc9d01fa0eb4d17e2c69dac45`.
- Run 21: `runs/21_base_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu3`, best update 649, SHA-256 `d1980adc99ec4cd14ea332fc8fb2ac4a2a5ef0339043ff4a9d49f161555f91a2`.
- Run 22: `runs/22_narrow_mean_hl_cqi4_bernoulli_lr2000_s2024_20260911_gpu5`, best update 1469, SHA-256 `539e72c35cd40699e17d1738b1a606d0a17c7d3b7239c71b41a91506967f9bda`.

Rate columns below are percentages; raw rate differences in the CSV/JSON remain fractions.

## ID

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 5348.27 | 77.576 [72.889, 82.264] | 32.910 | 0.008 | 3.064 |
| PPO/run21 | 5347.60 | 77.407 [72.658, 82.156] | 33.054 | 0.008 | 3.226 |
| PPO/run22 | 5318.79 | 77.412 [72.706, 82.117] | 33.124 | 0.008 | 3.112 |
| SUS+CQI | 4845.98 | 75.032 [70.362, 79.702] | 35.442 | 0.008 | 3.844 |
| SU+CQI | 2068.59 | 65.130 [62.292, 67.968] | 41.953 | 0.013 | 1.000 |
| SUS+PF | 2998.27 | 66.546 [63.207, 69.885] | 40.692 | 0.009 | 3.873 |
| SU+PF | -815.52 | 51.457 [49.957, 52.957] | 51.039 | 0.013 | 1.000 |
| SUS+Random | 2298.67 | 63.382 [60.189, 66.575] | 43.234 | 0.010 | 3.901 |
| SU+Random | -3233.89 | 40.356 [39.195, 41.517] | 60.117 | 0.015 | 1.000 |

## P055

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 4384.24 | 115.336 [111.726, 118.946] | 43.201 | 0.109 | 3.381 |
| PPO/run21 | 4541.13 | 115.759 [112.096, 119.421] | 42.935 | 0.109 | 3.463 |
| PPO/run22 | 4418.71 | 115.516 [111.863, 119.170] | 43.170 | 0.109 | 3.403 |
| SUS+CQI | 3827.09 | 112.273 [108.579, 115.966] | 44.842 | 0.108 | 3.960 |
| SU+CQI | -3892.12 | 84.015 [82.399, 85.632] | 57.879 | 0.153 | 1.000 |
| SUS+PF | -2136.34 | 85.579 [83.181, 87.977] | 56.084 | 0.126 | 3.983 |
| SU+PF | -10275.17 | 54.344 [53.184, 55.504] | 71.146 | 0.162 | 1.000 |
| SUS+Random | -2685.97 | 83.164 [80.774, 85.554] | 57.402 | 0.128 | 3.990 |
| SU+Random | -12933.20 | 42.332 [41.082, 43.581] | 76.838 | 0.181 | 1.000 |

## P010

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 3173.35 | 29.924 [28.593, 31.255] | 19.522 | 0.000 | 2.065 |
| PPO/run21 | 3153.16 | 29.813 [28.488, 31.138] | 19.774 | 0.000 | 2.302 |
| PPO/run22 | 3162.70 | 29.855 [28.527, 31.183] | 19.765 | 0.000 | 2.056 |
| SUS+CQI | 3010.98 | 29.184 [27.894, 30.474] | 21.791 | 0.000 | 2.940 |
| SU+CQI | 3075.03 | 29.646 [28.370, 30.922] | 20.810 | 0.000 | 1.000 |
| SUS+PF | 2956.91 | 28.901 [27.637, 30.165] | 22.371 | 0.000 | 2.963 |
| SU+PF | 2925.13 | 28.857 [27.652, 30.062] | 22.387 | 0.000 | 1.000 |
| SUS+Random | 2844.94 | 28.361 [27.155, 29.568] | 23.601 | 0.000 | 3.049 |
| SU+Random | 2509.70 | 26.761 [25.784, 27.738] | 27.075 | 0.000 | 1.000 |

## V60max

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 5259.06 | 77.187 [72.512, 81.861] | 33.231 | 0.008 | 3.067 |
| PPO/run21 | 5260.49 | 77.015 [72.288, 81.741] | 33.363 | 0.008 | 3.232 |
| PPO/run22 | 5232.89 | 77.032 [72.346, 81.718] | 33.419 | 0.008 | 3.116 |
| SUS+CQI | 4775.90 | 74.742 [70.078, 79.407] | 35.672 | 0.008 | 3.884 |
| SU+CQI | 1991.94 | 64.813 [61.984, 67.641] | 42.225 | 0.012 | 1.000 |
| SUS+PF | 2845.65 | 65.903 [62.620, 69.186] | 41.092 | 0.011 | 3.908 |
| SU+PF | -899.58 | 51.098 [49.618, 52.579] | 51.325 | 0.013 | 1.000 |
| SUS+Random | 2093.42 | 62.513 [59.393, 65.633] | 43.847 | 0.011 | 3.927 |
| SU+Random | -3357.96 | 39.860 [38.704, 41.017] | 60.563 | 0.014 | 1.000 |

## CSI02

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 4289.90 | 73.021 [68.588, 77.454] | 36.227 | 0.009 | 3.055 |
| PPO/run21 | 4248.75 | 72.650 [68.182, 77.118] | 36.450 | 0.009 | 3.282 |
| PPO/run22 | 4254.56 | 72.818 [68.381, 77.255] | 36.437 | 0.009 | 3.127 |
| SUS+CQI | 3796.22 | 70.539 [66.120, 74.958] | 38.390 | 0.009 | 3.867 |
| SU+CQI | 1396.84 | 62.414 [59.594, 65.235] | 44.343 | 0.013 | 1.000 |
| SUS+PF | 1605.55 | 60.658 [57.701, 63.614] | 44.321 | 0.011 | 3.893 |
| SU+PF | -2197.26 | 45.855 [44.552, 47.158] | 55.418 | 0.014 | 1.000 |
| SUS+Random | 1029.70 | 58.072 [55.136, 61.008] | 46.716 | 0.012 | 3.915 |
| SU+Random | -4395.91 | 35.817 [34.693, 36.941] | 63.977 | 0.015 | 1.000 |

## D26strict

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 5233.68 | 74.779 [70.276, 79.281] | 35.223 | 0.000 | 2.994 |
| PPO/run21 | 5200.17 | 74.505 [69.958, 79.052] | 35.453 | 0.000 | 3.160 |
| PPO/run22 | 5224.24 | 74.722 [70.190, 79.254] | 35.348 | 0.000 | 2.990 |
| SUS+CQI | 4634.57 | 71.868 [67.401, 76.335] | 38.083 | 0.000 | 3.760 |
| SU+CQI | 2116.62 | 63.561 [60.716, 66.406] | 43.507 | 0.000 | 1.000 |
| SUS+PF | 2760.53 | 63.401 [60.276, 66.526] | 43.315 | 0.000 | 3.796 |
| SU+PF | -782.16 | 50.125 [48.646, 51.605] | 52.433 | 0.000 | 1.000 |
| SUS+Random | 1960.70 | 59.592 [56.709, 62.476] | 46.159 | 0.000 | 3.830 |
| SU+Random | -3450.38 | 37.597 [36.492, 38.702] | 62.497 | 0.000 | 1.000 |

## STORM2

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 2101.15 | 105.645 [102.113, 109.177] | 47.609 | 0.118 | 3.407 |
| PPO/run21 | 2189.87 | 105.804 [102.219, 109.389] | 47.488 | 0.116 | 3.521 |
| PPO/run22 | 2116.66 | 105.707 [102.141, 109.273] | 47.643 | 0.117 | 3.418 |
| SUS+CQI | 1471.52 | 102.385 [98.795, 105.975] | 49.239 | 0.118 | 3.938 |
| SU+CQI | -4659.23 | 80.890 [79.121, 82.660] | 59.442 | 0.156 | 1.000 |
| SUS+PF | -5582.32 | 71.323 [68.777, 73.868] | 62.239 | 0.141 | 3.976 |
| SU+PF | -12701.54 | 44.968 [43.760, 46.177] | 75.683 | 0.178 | 1.000 |
| SUS+Random | -5064.89 | 73.686 [71.160, 76.212] | 61.629 | 0.139 | 3.982 |
| SU+Random | -14672.23 | 35.940 [34.682, 37.197] | 79.962 | 0.194 | 1.000 |

## K8

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 3224.69 | 31.624 [29.696, 33.552] | 20.524 | 0.008 | 2.256 |
| PPO/run21 | 3218.88 | 31.577 [29.646, 33.507] | 20.594 | 0.008 | 2.363 |
| PPO/run22 | 3221.12 | 31.599 [29.673, 33.525] | 20.602 | 0.008 | 2.347 |
| SUS+CQI | 3144.10 | 31.228 [29.309, 33.148] | 21.451 | 0.008 | 2.521 |
| SU+CQI | 3052.64 | 30.960 [29.154, 32.767] | 22.072 | 0.008 | 1.000 |
| SUS+PF | 3115.09 | 31.078 [29.163, 32.992] | 21.731 | 0.008 | 2.525 |
| SU+PF | 2970.95 | 30.510 [28.733, 32.288] | 22.876 | 0.008 | 1.000 |
| SUS+Random | 3063.28 | 30.845 [28.970, 32.720] | 22.180 | 0.008 | 2.584 |
| SU+Random | 2699.28 | 29.194 [27.613, 30.775] | 25.384 | 0.009 | 1.000 |

## K48

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 3189.30 | 127.812 [122.766, 132.857] | 44.651 | 0.012 | 3.715 |
| PPO/run21 | 3388.82 | 128.099 [122.972, 133.226] | 44.424 | 0.012 | 3.770 |
| PPO/run22 | 3234.54 | 127.960 [122.847, 133.074] | 44.668 | 0.012 | 3.705 |
| SUS+CQI | 2966.34 | 126.218 [121.183, 131.253] | 45.629 | 0.012 | 3.995 |
| SU+CQI | -7523.94 | 87.422 [85.673, 89.172] | 60.749 | 0.016 | 1.000 |
| SUS+PF | -7549.31 | 78.747 [77.547, 79.947] | 62.561 | 0.015 | 3.999 |
| SU+PF | -15457.55 | 50.272 [49.599, 50.944] | 75.255 | 0.018 | 1.000 |
| SUS+Random | -7822.15 | 77.190 [75.580, 78.800] | 63.952 | 0.015 | 4.000 |
| SU+Random | -18188.02 | 37.623 [36.977, 38.269] | 80.816 | 0.019 | 1.000 |

## K60

| Scheduler | Reward | Goodput Mbps [95% CI] | Deadline miss % | Overflow % | MU depth |
|---|---:|---:|---:|---:|---:|
| PPO/run16 | 254.79 | 145.378 [139.948, 150.808] | 49.339 | 0.015 | 3.889 |
| PPO/run21 | 396.42 | 145.053 [139.705, 150.401] | 49.202 | 0.015 | 3.899 |
| PPO/run22 | 238.62 | 145.176 [139.725, 150.627] | 49.439 | 0.015 | 3.856 |
| SUS+CQI | 104.10 | 144.286 [138.934, 149.638] | 49.962 | 0.014 | 3.999 |
| SU+CQI | -14280.65 | 90.666 [89.111, 92.221] | 67.103 | 0.019 | 1.000 |
| SUS+PF | -15175.32 | 76.121 [74.698, 77.543] | 70.181 | 0.018 | 4.000 |
| SU+PF | -23314.70 | 48.567 [47.868, 49.266] | 80.635 | 0.020 | 1.000 |
| SUS+Random | -14765.80 | 76.691 [75.099, 78.282] | 70.619 | 0.018 | 4.000 |
| SU+Random | -25756.84 | 36.518 [35.875, 37.161] | 84.744 | 0.021 | 1.000 |

All 18 policy–baseline comparisons and three policy–policy comparisons per world are in `paired_statistics.csv` and `summary.json`.

Interpretation limits:

- Intervals use independent evaluation episodes as the statistical unit, not slots or packets.
- Pointwise 95% Student-t intervals are not simultaneous bounds across worlds, schedulers or metrics.
- All three fixed checkpoints share one training seed; these intervals do not measure training-seed variation. Run16 continues earlier training and is not an independent training replicate.
- Within each world, differences pair the same episode IDs and initial conditions. Shared RNG consumption can change realized traffic/CSI after scheduler-dependent queue admissions; identical packet traces are not claimed.
- K scaling changes both user population and total offered load. Different K worlds do not share identical topology and are not paired against one another.
- D26strict changes environment deadlines to 2..6 while policy deadline normalization remains /12. Variable K uses the unchanged policy formulas with their native current-K count fractions.
- SUS thresholds were selected by mean reward on separate pilot episodes. Only the six declared historical baselines are compared; this does not establish superiority over every heuristic.
- Completion, deadline-miss and retransmission-drop rates use admitted arrivals; buffer-overflow rates use offered arrivals. Unfinished packets remain at the finite episode horizon.
- First-ACK rates pool raw counts across episodes. Empty depth bins are undefined and stored as null, not zero.
