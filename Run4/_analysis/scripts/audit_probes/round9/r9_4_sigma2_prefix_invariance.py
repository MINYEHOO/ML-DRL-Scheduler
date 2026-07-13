from config import Config, phase2_hard_main_config, phase2_hetero_config, phase4_queue_config
import numpy as np
from phy import sigma2_from_gain

for name, cfg in [("default Config", Config()),
                  ("Run2 phase2_hard_main", phase2_hard_main_config()),
                  ("Run3 phase2_hetero", phase2_hetero_config()),
                  ("Run4 phase4_queue", phase4_queue_config())]:
    print(f"{name:26s} noise_mode={cfg.noise_mode!r} target_snr_db={cfg.target_snr_db}")

# demonstrate: sigma2 is median over the FULL [T,K,R] array -> changing a
# future slot's gains changes sigma2 (hence slot-0 CQI)
rng = np.random.default_rng(0)
cfg = phase4_queue_config()
g = rng.gamma(2.0, 1.0, size=(1000, 32, 8))   # stand-in beamformed gains [T,K,R]
s_full = sigma2_from_gain(g, cfg)
g2 = g.copy(); g2[500:] *= 4.0                # perturb only slots 500..999
s_pert = sigma2_from_gain(g2, cfg)
cqi0 = np.log2(1.0 + g[0] / s_full)
cqi0_p = np.log2(1.0 + g[0] / s_pert)
print(f"sigma2 full={s_full:.6f}  sigma2 after future-only perturb={s_pert:.6f}")
print(f"slot-0 CQI mean shift: {cqi0.mean():.4f} -> {cqi0_p.mean():.4f} "
      f"(prefix-invariance violated: {not np.allclose(cqi0, cqi0_p)})")
