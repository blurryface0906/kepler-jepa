"""Supplementary H3 calibration; never changes the acceptance criteria.

Identity embedding of the same physical context windows and an oracle harmonic
reference isolate finite-window estimator bias from learned encoder geometry.
The oracle is explicitly a calibration signal, never training/probe input.
"""
from pathlib import Path
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
limits=threadpool_limits(limits=4)
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT.parents[1]))
from src.data import analytical_ellipse_batch
from src.evaluation import evaluate_frequencies
from experiments.exp1_encoder_tautology import _write_json

records=[]
report=json.loads((OUT/'exp3_elliptic_results.json').read_text())
for seed in [42,43,44,45,46]:
    meta=report['dataset_metadata'][str(seed)]['test_id_ellip']
    batch=analytical_ellipse_batch(meta['a'],meta['e'],meta['M0'],meta['omega_arg'],split='test_id_ellip');a=batch['meta']['a']
    starts=np.arange(1,513)*4
    context=batch['trajectories'][:,starts[:,None]+np.arange(16)].reshape(len(a),512,64)
    phase=batch['meta']['M0'][:,None]+batch['meta']['omega_true'][:,None]*starts[None]*.01
    harmonic=np.stack([np.cos(phase),np.sin(phase)],axis=-1)
    for label,values in [('identity_physical_context',context),('oracle_mean_anomaly_circle',harmonic)]:
        for method,result in evaluate_frequencies(values,a,.04).items():
            records.append(dict(seed=seed,reference=label,**result))
_write_json(OUT/'task4_h3_reference.json',dict(purpose='Postprocessing calibration only; no training, no change to criteria',records=records))
print('H3 physical/oracle references saved:',len(records))
