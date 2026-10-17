"""Task 4 analytical ellipse batches; legacy dataset builders stay unchanged."""
from __future__ import annotations

import numpy as np

from .dataset import build_dataset_circ
from .simulator import DT, MU, generate_kepler_analytical


def elliptic_id_axes():
    a=build_dataset_circ('test_id',42)['meta']['a']
    return a[(a>=.5)&(a<=2)].copy()


def _stratified_eccentricities(n, low, high):
    """Deterministic permutation with near-zero Pearson/Spearman association.

    Selection uses only grid labels, never trajectories, model outputs or test
    errors. Same permutation is used for all seeds. The residual association
    is disclosed in batch metadata rather than claiming exact independence.
    """
    rng=np.random.default_rng(49017)
    grid=np.linspace(low,high,n)
    log_axes=np.log(elliptic_id_axes())
    best=None;score=np.inf
    for _ in range(2048):
        candidate=grid[rng.permutation(n)]
        value=abs(float(np.corrcoef(log_axes,candidate)[0,1]))
        if value<score:best,score=candidate,value
    return best


def analytical_ellipse_batch(a,e,M0,omega_arg,*,split,raw_samples=2100):
    a,e,M0,omega_arg=np.broadcast_arrays(*[np.asarray(v,dtype=np.float64) for v in (a,e,M0,omega_arg)])
    if a.ndim!=1 or len(a)<1 or raw_samples<16 or np.any(a<=0) or np.any((e<0)|(e>=1)):
        raise ValueError('Expected one-dimensional bound ellipse parameters and enough samples')
    states=generate_kepler_analytical(a[:,None],e[:,None],np.arange(raw_samples)[None]*DT,
                                     M0=M0[:,None],omega_arg=omega_arg[:,None])
    return {'trajectories':states,'split':split,'dt':DT,
            'meta':{'a':a.copy(),'e':e.copy(),'M0':M0.copy(),'omega_arg':omega_arg.copy(),
                    'omega_true':np.sqrt(MU/a**3),'energy_true':-MU/(2*a),
                    'angular_momentum_true':np.sqrt(MU*a*(1-e**2))}}


def build_dataset_ellip_task4(split,seed=42,*,raw_samples=2100):
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    names={'train_ellip':0,'test_id_ellip':1,'test_ood_ecc':2,'test_iso_energy':3}
    if split not in names:raise ValueError('Unknown Task4 ellipse split')
    rng=np.random.default_rng(np.random.SeedSequence([seed,440,names[split]]))
    if split=='train_ellip':a=np.geomspace(.5,5,200);e=rng.uniform(.05,.70,200)
    elif split=='test_iso_energy':a=np.repeat([.7,1.2,1.8],15);e=np.tile(np.linspace(.05,.75,15),3)
    else:
        a=elliptic_id_axes();low,high=(.10,.65) if split=='test_id_ellip' else (.72,.85)
        e=_stratified_eccentricities(len(a),low,high)
    batch=analytical_ellipse_batch(a,e,rng.uniform(0,2*np.pi,len(a)),
                                 rng.uniform(0,2*np.pi,len(a)),split=split,raw_samples=raw_samples)
    batch['meta']['log_a_e_pearson']=float(np.corrcoef(np.log(a),e)[0,1])
    return batch


def build_peri_apo_task4(*,raw_samples=2100):
    """Apsis at the first sample t=0, as requested (not at context center)."""
    a=elliptic_id_axes();e=np.full(len(a),.5)
    return {name:analytical_ellipse_batch(a,e,np.full(len(a),phase),np.zeros(len(a)),
                split='test_ellip_'+name,raw_samples=raw_samples)
            for name,phase in [('peri',0.),('apo',np.pi)]}


def instantaneous_probe_targets(batch):
    states=batch['trajectories'][:,0]
    r=np.linalg.norm(states[:,:2],axis=1);v=np.linalg.norm(states[:,2:],axis=1)
    a,e=batch['meta']['a'],batch['meta']['e']
    return {'log_r0':np.log(r),'log_v0':np.log(v),'log_a':np.log(a),'e':e.copy(),
            'log_h':.5*np.log(MU*a*(1-e**2)),'log_abs_energy':np.log(MU/(2*a))}
