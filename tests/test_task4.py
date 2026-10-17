"""Ellipse physics, train-only probes, and strict acceptance checks."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from src.data import build_dataset_ellip_task4,build_peri_apo_task4,instantaneous_probe_targets
from src.train import ellip_horizon_grid,elliptic_training_losses
from experiments.exp3_elliptic_kepler import config_for,make_model,oos_probes,phase_gap,acceptance


class Task4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_peri_apo_physics_and_first_sample(self):
        pair=build_peri_apo_task4(raw_samples=24)
        for name,rf,vf in [('peri',.5,3.),('apo',1.5,1/3)]:
            batch=pair[name];a=batch['meta']['a'];x=batch['trajectories'][:,0]
            np.testing.assert_allclose(np.linalg.norm(x[:,:2],axis=1),rf*a,rtol=1e-13)
            np.testing.assert_allclose(np.linalg.norm(x[:,2:],axis=1),np.sqrt(vf/a),rtol=1e-13)
            targets=instantaneous_probe_targets(batch)
            np.testing.assert_allclose(targets['log_abs_energy'],-np.log(2)-targets['log_a'])

    def test_split_ranges_labels_and_physical_conservation(self):
        for split,n,lo,hi in [('train_ellip',200,.05,.70),('test_id_ellip',30,.1,.65),('test_ood_ecc',30,.72,.85),('test_iso_energy',45,.05,.75)]:
            batch=build_dataset_ellip_task4(split,42,raw_samples=120)
            self.assertEqual(batch['trajectories'].shape,(n,120,4))
            self.assertTrue(np.all((batch['meta']['e']>=lo)&(batch['meta']['e']<=hi)))
            x=batch['trajectories'];r=np.linalg.norm(x[...,:2],axis=-1)
            energy=(x[...,2:]**2).sum(-1)/2-1/r
            np.testing.assert_allclose(energy,np.broadcast_to(batch['meta']['energy_true'][:,None],energy.shape),atol=2e-13)
            if split in ('test_id_ellip','test_ood_ecc'):
                self.assertLess(abs(batch['meta']['log_a_e_pearson']),.001)

    def test_random_phases_repeatable_and_seed_changes(self):
        a=build_dataset_ellip_task4('test_id_ellip',42,raw_samples=16)
        b=build_dataset_ellip_task4('test_id_ellip',43,raw_samples=16)
        np.testing.assert_array_equal(a['meta']['e'],b['meta']['e'])
        self.assertFalse(np.array_equal(a['trajectories'],b['trajectories']))
        np.testing.assert_array_equal(a['trajectories'],build_dataset_ellip_task4('test_id_ellip',42,raw_samples=16)['trajectories'])

    def test_sparse_horizons_and_fixed_production(self):
        c=config_for('M1_Ellip_Inv');direct,offsets=ellip_horizon_grid(c)
        self.assertEqual(direct,list(range(8,257,8)));self.assertEqual(max(offsets),256)
        self.assertTrue(set(range(17)).issubset(offsets))
        for changes in [dict(device='cpu'),dict(steps=100),dict(batch_size=64),dict(delta=2)]:
            with self.assertRaises(ValueError):replace(c,**changes)

    def test_invariance_exact_formula_and_gradients_without_metadata(self):
        c=config_for('M1_Ellip_Inv','cpu',True);model,_=make_model('M1_Ellip_Inv',42,'cpu',None)
        _,offsets=ellip_horizon_grid(c);torch.manual_seed(7)
        first=torch.randn(c.batch_size,len(offsets),16,4);second=torch.randn_like(first)
        result=elliptic_training_losses(model,first,second,c,offsets)
        z1=model.encoder(first[:,0]);z2=model.encoder(second[:,0])
        expected=(torch.log(model.predictor.angular_rates(z1)+1e-8)-torch.log(model.predictor.angular_rates(z2)+1e-8)).square().sum(-1).mean()
        torch.testing.assert_close(result['frequency_invariance'],expected)
        result['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.context_encoder.parameters() if p.grad is not None))
        same=elliptic_training_losses(model,first,first,c,offsets)
        self.assertEqual(same['frequency_invariance'].item(),0.)

    def test_phase_invalid_not_dropped(self):
        self.assertAlmostEqual(phase_gap([2,4],[1,2])['mean_phase_gap'],np.log(2))
        result=phase_gap([2,np.nan],[1,2]);self.assertEqual(result['invalid'],1)
        self.assertTrue(np.isnan(result['mean_phase_gap']))

    def test_probe_train_only_scaler_and_negative_oos_r2(self):
        x=np.arange(20,dtype=float)[:,None];y=x[:,0]
        records=oos_probes({'norm':x},{'test':{'norm':x+100}},{'log_a':y},{'test':{'log_a':-y}})
        for r in records:
            self.assertEqual(r['n_fit_orbits'],20);self.assertEqual(r['train_feature_mean'],[9.5])
            self.assertLess(r['r2'],0)

    def test_constant_norm_probe_intercept_only(self):
        x=np.ones((20,1));y=np.arange(20.)
        rows=oos_probes({'norm':x},{'test':{'norm':x*99}},{'e':y},{'test':{'e':y}})
        for r in rows:
            self.assertEqual(r['active_columns'],[])
            np.testing.assert_allclose(r['predictions'],y.mean())

    def test_acceptance_missing_seeds_cannot_pass(self):
        report={k:[] for k in ['training_runs','frequency_runs','phase_runs','probe_runs']}
        self.assertTrue(all(not c['passed'] for c in acceptance(report)))


if __name__=='__main__':unittest.main()
