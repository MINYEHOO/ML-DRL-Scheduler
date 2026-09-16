"""Analytic wireless invariants; NumPy only, no channel generation or GPU."""
from pathlib import Path
from dataclasses import fields
import json
import os
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config
from csi import NR_CQI_TABLE_256QAM, quantize_cqi
from phy import rzf_precoder, _rbg_sinr, mi_bits
from la_planner import SlotAllocationPlanner, predict_group_link_adaptation


class WirelessMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get('PAPER_MAIN_CONFIG', ROOT/'artifacts/base/config.json'))
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(Config)}
        cls.cfg = Config(**{k:v for k,v in raw.items() if k in known})
        assert cls.cfg.num_ue == 32 and cls.cfg.cqi_mode == 'nr4bit'

    def test_post_rzf_against_explicit_independent_inverse(self):
        rng = np.random.default_rng(913)
        cfg = self.cfg
        for depth in range(1, 5):
            for _ in range(50):
                h = rng.normal(size=(depth, cfg.num_bs_ant))+1j*rng.normal(size=(depth,cfg.num_bs_ant))
                # Independent mathematical implementation, rather than using
                # the shared rzf_precoder in both prediction and truth paths.
                raw = h.conj().T @ np.linalg.inv(h@h.conj().T + np.eye(depth))
                oracle_w = raw/np.linalg.norm(raw,axis=0)*np.sqrt(cfg.p_rbg/depth)
                powers = abs(h@oracle_w)**2
                oracle_s = np.diag(powers)/(powers.sum(axis=1)-np.diag(powers)+1.)
                w, pred, _ = predict_group_link_adaptation(h,1.,1.,cfg.p_rbg,cfg)
                np.testing.assert_allclose(w,oracle_w,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(pred,oracle_s,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(_rbg_sinr(h,h,1.,cfg.p_rbg,1.),oracle_s,rtol=1e-10)
                self.assertAlmostEqual(float(np.sum(abs(w)**2)),cfg.p_rbg,places=12)

    def test_cqi_floor_boundaries(self):
        ladder = NR_CQI_TABLE_256QAM
        np.testing.assert_array_equal(quantize_cqi(ladder,'nr4bit'),ladder)
        np.testing.assert_array_equal(quantize_cqi(np.nextafter(ladder[1:],-np.inf),'nr4bit'),ladder[:-1])
        np.testing.assert_array_equal(quantize_cqi(np.array([0.,100.]),'nr4bit'),ladder[[0,-1]])

    def test_archived_zero_cap_ratio_is_not_runtime_group(self):
        cfg = self.cfg
        h = np.zeros((cfg.num_ue,cfg.num_rbg,cfg.num_bs_ant),complex)
        h[0,:,0] = np.sqrt(10.)
        _,sp,_ = predict_group_link_adaptation(h[[0,1],0],1.,1.,cfg.p_rbg,cfg)
        cap = cfg.eta_data*cfg.n_re_rbg*cfg.beta_rate*np.log2(1+sp)
        mi = mi_bits(_rbg_sinr(h[[0,1],0],h[[0,1],0],1.,cfg.p_rbg,1.),cfg)
        self.assertEqual(float(cap[1]),0.)
        self.assertEqual(float((mi/np.maximum(cap,1e-12))[1]),0.)
        planner = SlotAllocationPlanner(cfg,h,1.,1.,np.full(cfg.num_ue,100000.))
        closed = planner.solve_closure(0,[],[0,1])
        self.assertEqual(closed['survivors'],[0])
        self.assertAlmostEqual(float(closed['sinr'][0]),2*float(sp[0]))

    def test_zero_cqi_fixed_retx_keeps_declared_outage(self):
        cfg = self.cfg
        h = np.zeros((4,cfg.num_bs_ant),complex)
        for i in (0,1,3): h[i,i] = 1.
        w = rzf_precoder(h,1.,cfg.p_rbg)
        self.assertTrue(np.isfinite(w).all())
        self.assertEqual(float(np.linalg.norm(w[:,2])),0.)
        self.assertAlmostEqual(float(np.sum(abs(w)**2)),.75*cfg.p_rbg)


if __name__ == '__main__':
    unittest.main()
