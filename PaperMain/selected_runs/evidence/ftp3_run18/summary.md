# Frozen-checkpoint FTP3 ID evaluation

Run: `runs/18_base_cqi4_ftp3_lr1000_s2024_20260910_gpu4`; checkpoint: `best` at update 639; SHA-256: `9afca2081ee8a5db08fe40460528e253456974b2d53abd39ca1cf0ace30ce329`.

100 episodes; 7 schedulers; 700 validated rows. Episode IDs: 120000–120099. Diagnostic only: False.

Values are episode means. Goodput includes its pointwise 95% bootstrap interval; all metric intervals are retained in `summary.json`.

| Scheduler | Goodput Mbps [95% CI] | Reward | Completion % | Deadline miss % | Overflow % | MU depth | Jain |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 76.2956 [72.0532, 80.6886] | 5060.59 | 65.765 | 33.929 | 0.146 | 3.025 | 0.7160 |
| SUS+CQI | 74.4696 [70.2808, 78.8710] | 4700.45 | 63.811 | 35.858 | 0.144 | 3.861 | 0.7294 |
| SUS+CQI@m=2 | 74.6004 [70.7125, 78.5849] | 4581.52 | 64.297 | 35.407 | 0.145 | 1.991 | 0.7132 |
| SUS+CQI@m=3 | 75.5543 [71.3717, 79.9251] | 4912.92 | 64.767 | 34.926 | 0.143 | 2.956 | 0.7284 |
| SUS-RPS | 76.1683 [72.1695, 80.3012] | 4938.49 | 65.937 | 33.777 | 0.134 | 2.408 | 0.7214 |
| PF-Greedy-SDS | 71.0531 [67.4220, 74.7470] | 3911.96 | 61.688 | 37.955 | 0.160 | 3.185 | 0.7309 |
| SumRate-Greedy-SDS | 76.2778 [72.0970, 80.6428] | 4951.89 | 65.333 | 34.378 | 0.143 | 2.172 | 0.7135 |

Paired PPO minus baseline differences, with pointwise 95% episode-bootstrap intervals. Positive goodput/reward differences favor PPO; negative miss/overflow differences favor PPO. Rate differences in the following table are fractions, not percentage points.

| Baseline | Goodput Mbps difference | Reward difference | Deadline miss difference | Overflow difference |
|---|---:|---:|---:|---:|
| SUS+CQI | 1.8260 [1.6314, 2.0208] | 360.1415 [318.5174, 401.6937] | -0.0193 [-0.0212, -0.0174] | 0.0000 [-0.0000, 0.0001] |
| SUS+CQI@m=2 | 1.6952 [1.2329, 2.2230] | 479.0720 [357.5354, 616.3478] | -0.0148 [-0.0172, -0.0125] | 0.0000 [-0.0000, 0.0001] |
| SUS+CQI@m=3 | 0.7413 [0.5689, 0.9138] | 147.6685 [111.2756, 184.6697] | -0.0100 [-0.0112, -0.0087] | 0.0000 [-0.0000, 0.0001] |
| SUS-RPS | 0.1273 [-0.2831, 0.6363] | 122.1016 [26.2725, 242.0881] | 0.0015 [-0.0009, 0.0036] | 0.0001 [0.0001, 0.0002] |
| PF-Greedy-SDS | 5.2425 [4.4703, 6.0851] | 1148.6301 [981.2956, 1330.9760] | -0.0403 [-0.0439, -0.0368] | -0.0001 [-0.0002, -0.0001] |
| SumRate-Greedy-SDS | 0.0178 [-0.1414, 0.1795] | 108.7020 [72.9709, 145.4130] | -0.0045 [-0.0057, -0.0033] | 0.0000 [-0.0000, 0.0001] |

Raw episode counter totals (packets). Offered includes arrivals rejected by a full queue; admitted excludes those arrivals. Terminal queued packets have not been drained.

| Scheduler | Offered | Admitted | Completed | Deadline drops | Retx-limit drops | Retx-overflow drops | Buffer-overflow drops | Terminal queued |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 758461 | 756989 | 482161 | 272494 | 125 | 0 | 1472 | 2209 |
| SUS+CQI | 758348 | 756913 | 468408 | 286013 | 192 | 0 | 1435 | 2300 |
| SUS+CQI@m=2 | 758332 | 756868 | 468470 | 286125 | 63 | 0 | 1464 | 2210 |
| SUS+CQI@m=3 | 758229 | 756792 | 474891 | 279587 | 128 | 0 | 1437 | 2186 |
| SUS-RPS | 758330 | 756969 | 481901 | 272833 | 65 | 0 | 1361 | 2170 |
| PF-Greedy-SDS | 758267 | 756666 | 447519 | 306407 | 369 | 0 | 1601 | 2371 |
| SumRate-Greedy-SDS | 758283 | 756842 | 478586 | 276054 | 73 | 0 | 1441 | 2129 |

First-ACK rates below pool raw counts across episodes: `sum(ACKs) / sum(new units)`. Each cell gives the rate and its count denominator; an empty bin is undefined.

| Scheduler | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| PPO | 93.358% (187798/201158) | 96.064% (246880/256996) | 95.050% (321267/337998) | 92.505% (1141406/1233880) |
| SUS+CQI | 94.681% (12068/12746) | 97.716% (67543/69122) | 97.891% (234335/239384) | 96.219% (2469028/2566039) |
| SUS+CQI@m=2 | 94.274% (13516/14337) | 95.756% (1435695/1499323) | undefined (0/0) | undefined (0/0) |
| SUS+CQI@m=3 | 94.530% (12839/13582) | 97.751% (75594/77333) | 96.110% (2060036/2143424) | undefined (0/0) |
| SUS-RPS | 92.518% (308994/333981) | 95.590% (354890/371264) | 94.790% (358903/378628) | 93.814% (403023/429596) |
| PF-Greedy-SDS | 92.718% (92395/99652) | 96.056% (315637/328597) | 95.818% (525412/548344) | 95.585% (1180230/1234738) |
| SumRate-Greedy-SDS | 92.448% (341677/369587) | 95.070% (408347/429522) | 93.327% (267615/286749) | 92.198% (263198/285471) |

Interpretation limits:

- Evaluation episodes are the resampling unit; slots and packets are not independent replicates.
- Paired differences match episode IDs and underlying episode conditions. Scheduler-dependent shared RNG consumption can change realized traffic/CSI, so identical packet traces are not claimed.
- Intervals are pointwise 95% percentile bootstrap intervals, not simultaneous confidence bounds across schedulers or metrics.
- One frozen checkpoint from one training seed is evaluated. Intervals do not quantify training-seed variation, and a positive interval is not a universal performance guarantee.
- Completion and deadline-miss rates use admitted arrivals; buffer-overflow rate uses offered arrivals. Finite episodes can retain unfinished packets at the horizon.
- First-ACK rates pool raw ACK counts and new-unit counts. Empty depth bins are undefined, not zero.
