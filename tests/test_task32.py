"""Task 3.2 mathematical objectives and controls, with no GPU substitution."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.exp2_circular_trained import (_absolute_frequency_statistics, _model,
                                              _weights_hash, acceptance_checks)
from src.evaluation import evaluate_oos_scale_probes
from src.losses import opf_orthogonality_loss
from src.train import (TrainConfig, factor_scale_loss, frequency_coherence_loss,
                       temporal_displacement_loss, training_losses, train_model)
from src.models import M2_JEPA_OPF
from tests.test_task31_experiment import _passing_report


class Task32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_opf_exact_population_correlation_offblock_element_mean(self):
        torch.manual_seed(12)
        z = torch.randn(13, 12, dtype=torch.float64, requires_grad=True)
        unit = z / (z.square().mean(-1, keepdim=True)+1e-6).sqrt()
        centered = unit - unit.mean(0)
        normalized = centered / centered.square().mean(0).sqrt()
        corr = normalized.T @ normalized / len(z)
        factor = torch.arange(12) // 4
        expected = corr.square()[factor[:, None] != factor[None, :]].mean()
        actual = opf_orthogonality_loss(z, 3)
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
        self.assertTrue(0 <= actual.item() <= 1)
        actual.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_opf_removes_per_orbit_positive_scaling_up_to_epsilon(self):
        torch.manual_seed(7)
        z = torch.randn(30, 16, dtype=torch.float64) * 20
        scales = torch.logspace(-.5, .5, 30, dtype=z.dtype)[:, None]
        torch.testing.assert_close(opf_orthogonality_loss(z), opf_orthogonality_loss(z*scales),
                                   atol=1e-8, rtol=1e-7)

    def test_legacy_reduction_preserves_original_factor_pair_scaling(self):
        z = torch.randn(20, 64, dtype=torch.float64)
        legacy = opf_orthogonality_loss(z, scale_invariant=False, reduction="legacy")
        mean = opf_orthogonality_loss(z, scale_invariant=False, reduction="mean")
        torch.testing.assert_close(legacy, mean*256, atol=1e-12, rtol=1e-12)

    def test_opf_zero_constant_and_invalid_options(self):
        for fill in (0., 1.):
            z = torch.full((8, 16), fill, requires_grad=True)
            loss = opf_orthogonality_loss(z)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):
            opf_orthogonality_loss(z, reduction="median")

    def test_log_temporal_and_factor_scale_known_values(self):
        config = TrainConfig(steps=3, batch_size=2, horizon=2, ar_steps=1,
                             temporal_lag=1, device="cpu", smoke_test=True, protocol="task3_2")
        x = torch.ones(2, 2, 4, dtype=torch.float64)
        z = torch.full((2, 8), 2., dtype=x.dtype, requires_grad=True)
        actual = temporal_displacement_loss(x, x, z, z, config)
        expected = .5*np.log((2+1e-6)/(1+1e-6))**2
        self.assertAlmostEqual(float(actual.detach()), expected, places=12)
        self.assertAlmostEqual(float(factor_scale_loss(z, x, K=4).detach()), 2*expected, places=12)
        (actual + factor_scale_loss(z, x)).backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_log_rate_coherence_population_variance_and_gradients(self):
        rates = torch.tensor([[1., 2., 4., 8.], [3., 3., 3., 3.]],
                             dtype=torch.float64, requires_grad=True)
        expected = np.var(np.log(rates.detach().numpy()+1e-8),axis=1).mean()
        actual = frequency_coherence_loss(rates)
        self.assertAlmostEqual(float(actual.detach()), float(expected), places=14)
        actual.backward()
        self.assertTrue(torch.isfinite(rates.grad).all())
        self.assertEqual(float(frequency_coherence_loss(torch.ones(3, 32))), 0)

    def test_m0_koopman_is_frozen_and_identical_to_m1_before_training(self):
        m0, m1 = _model("M0_Koopman",42,"cpu"), _model("M1",42,"cpu")
        self.assertFalse(any(p.requires_grad for p in m0.parameters()))
        for left,right in ((m0.encoder,m1.encoder),(m0.predictor,m1.predictor)):
            for key,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[key],atol=0,rtol=0)
        before = _weights_hash(m0)
        m0.double()
        self.assertEqual(before,_weights_hash(m0,original_dtype=torch.float32))

    def test_objective_components_and_gradients_on_all_trainable_models(self):
        config = TrainConfig(steps=3,batch_size=2,horizon=4,ar_steps=2,temporal_lag=2,
                             device="cpu",smoke_test=True,protocol="task3_2")
        windows = torch.randn(2,5,16,4)
        for name in ("M1","M2","M3"):
            model = _model(name,42,"cpu")
            losses = training_losses(model,windows,config)
            self.assertEqual(float(losses['factor_scale'].detach())>0,name=='M2')
            self.assertEqual(float(losses['coherence'].detach())>0,name!='M3')
            if name!='M3':
                z0=model.encoder(windows[:,0])
                predicted=model.rollout_autoregressive(z0,2)
                targets=model.target_encoder(windows[:,:3].flatten(0,1)).reshape(2,3,64).detach()
                state_error=(predicted-targets[:,1:]).square().sum(-1).mean(1)
                expected=(state_error/(targets[:,1:].square().sum(-1).mean(1)+1e-4)).mean()
                torch.testing.assert_close(losses['ar_state'],expected)
            losses['total'].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_absolute_frequency_diagnostics_do_not_filter_invalid_orbits(self):
        row={'semi_major_axes':[1.,4.],'omega_est':[1.,.125]}
        self.assertEqual(_absolute_frequency_statistics(row)['max_abs_q_error'],0)
        row['omega_est'][1]=None
        self.assertTrue(np.isnan(_absolute_frequency_statistics(row)['q_median']))

    def test_acceptance_keeps_16_original_and_two_separate_controls(self):
        report=_passing_report()
        report['protocol']['training']={'protocol':'task3_2'}
        for r in report['training_runs']: r['config']['protocol']='task3_2'
        for r in list(report['frequency_runs']):
            if r['model']=='M0' and r['regime']!='T1':
                report['frequency_runs'].append({**r,'model':'M0_Koopman'})
        checks=acceptance_checks(report)
        self.assertEqual(len(checks),18)
        self.assertTrue(all(c['passed'] for c in checks))
        target=next(r for r in report['frequency_runs'] if r['model']=='M0_Koopman'
                    and r['regime']=='T4_512' and r['seed']==46)
        target.update(slope_abs_error=.001,r2=.99999)
        self.assertFalse(acceptance_checks(report)[-1]['passed'])

    def test_oos_coefficients_are_frozen_train_to_test_and_negative_scores_retained(self):
        x=np.linspace(-1,1,12)
        def record(phase,r,v,a):
            return {'n_orbits':len(phase),'physical_phase_norm':list(phase),
                    'r_rms':list(r),'v_rms':list(v),'log_a':list(a)}
        train=record(x,2*x+1,x*x,.4*(2*x+1)+.8*x*x-.1)
        test=record(x,2*x+11,x*x,.4*(2*x+1)+.8*x*x-.1)
        result=evaluate_oos_scale_probes(train,test)
        self.assertLess(result['oos_phase_norm_r2'],0)
        self.assertLess(result['oos_axis_probe_r2'],0)
        self.assertAlmostEqual(result['fits']['phase_norm']['train_fit']['intercept'],1)
        exact=evaluate_oos_scale_probes(train,deepcopy(train))
        self.assertAlmostEqual(exact['oos_axis_probe_r2'],1)

    def test_oos_constant_phase_response_uses_explicit_convention(self):
        rec={'n_orbits':10,'physical_phase_norm':list(np.linspace(0,1,10)),
             'r_rms':[0.]*10,'v_rms':list(np.linspace(0,1,10)),
             'log_a':list(np.linspace(0,1,10))}
        result=evaluate_oos_scale_probes(rec,rec)
        self.assertEqual(result['oos_phase_norm_r2'],0)
        self.assertTrue(result['fits']['phase_norm']['constant_response_convention_applied'])

    def test_task32_checkpoint_resume_matches_uninterrupted_and_rejects_task31(self):
        config=TrainConfig(steps=4,batch_size=4,context_length=2,horizon=4,ar_steps=2,
                           temporal_lag=2,device='cpu',smoke_test=True,protocol='task3_2',
                           log_every=1,checkpoint_every=1)
        torch.manual_seed(42)
        full=M2_JEPA_OPF(L=2,C=2,d=8,H_max=4)
        initial=deepcopy(full.state_dict())
        data=np.random.default_rng(42).normal(size=(3,40,2)).astype(np.float32)
        train_model(full,data,config,42)
        partial=M2_JEPA_OPF(L=2,C=2,d=8,H_max=4)
        partial.load_state_dict(initial)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint=Path(directory)/'task32.pt'
            def interrupt(record):
                if record['step']==2: raise RuntimeError('interrupt')
            with self.assertRaisesRegex(RuntimeError,'interrupt'):
                train_model(partial,data,config,42,checkpoint,interrupt)
            resumed=M2_JEPA_OPF(L=2,C=2,d=8,H_max=4)
            train_model(resumed,data,config,42,checkpoint,resume_from=checkpoint)
            for name,value in full.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                train_model(resumed,data,replace(config,protocol='task3_1'),42,resume_from=checkpoint)


if __name__=='__main__':
    unittest.main()
