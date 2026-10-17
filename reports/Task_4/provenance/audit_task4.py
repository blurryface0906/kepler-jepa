"""Independent artifact checks, preserving negative scientific results."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.data import analytical_ellipse_batch,instantaneous_probe_targets
from src.spectral import evaluate_kepler_regression
from src.evaluation import evaluate_frequencies
from experiments.exp3_elliptic_kepler import acceptance,make_model,probe_features,t1_t3,SEEDS,TRAINED
from experiments.exp2_circular_trained import _weights_hash


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/Task_4');args=parser.parse_args()
    out=args.output;p=out/'provenance';r=json.loads((out/'exp3_elliptic_results.json').read_text())
    def replay(split,seed):
        meta=r['dataset_metadata'][str(seed)][split]
        return analytical_ellipse_batch(meta['a'],meta['e'],meta['M0'],meta['omega_arg'],split=split)
    assert r['execution_completed'] and not r['protocol']['smoke_test']
    assert r['record_counts']==dict(training_runs=20,frequency_runs=1120,phase_runs=40,iso_runs=120,probe_runs=2880,frozen_checks=20)
    previous=json.loads((p/'previous_reports_sha256.json').read_text())
    for name,h in previous.items():assert digest(ROOT/name)==h,name
    manifest=json.loads((p/'artifact_sha256.json').read_text())
    presentation=json.loads((p/'presentation_changes.json').read_text()) if (p/'presentation_changes.json').exists() else None
    for name,h in manifest.items():
        if name=='figures/exp3_elliptic_kepler.png' and presentation:
            assert digest(p/'exp3_elliptic_kepler_gpu_original.png')==h==presentation['original_figure_sha256']
            assert digest(out/name)==presentation['final_figure_sha256']
            assert digest(out/'exp3_elliptic_results.json')==presentation['numerical_json_sha256']
        else:assert digest(out/name)==h,name
    with zipfile.ZipFile(p/'task4_sources.zip') as z:
        for name,h in r['source_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==h,name
            assert digest(ROOT/name)==h,name
    for x in r['frequency_runs']:
        w=np.asarray(x['omega_est'],float);a=np.asarray(x['semi_major_axes']);invalid=int((~np.isfinite(w)|(w<=0)).sum())
        assert invalid==x['invalid']
        np.testing.assert_allclose(np.asarray(x['q'],float),w/a**-1.5,rtol=1e-12,equal_nan=True)
        if not invalid:
            reg=evaluate_kepler_regression(a,w)
            for key in ['slope','r2','slope_abs_error']:np.testing.assert_allclose(x[key],reg[key],rtol=1e-11,atol=1e-12)
    for x in r['phase_runs']:
        pair={phase:next(row for row in r['frequency_runs'] if row['seed']==x['seed'] and row['model']==x['model'] and row['split']=='test_ellip_'+phase and row['method']=='S3b_diff' and row['regime']=='T3') for phase in ['peri','apo']}
        wp=np.asarray(pair['peri']['omega_est'],float);wa=np.asarray(pair['apo']['omega_est'],float)
        np.testing.assert_allclose(np.asarray(x['rho'],float),wp/wa,rtol=1e-12,equal_nan=True)
        if x['invalid']==0:np.testing.assert_allclose(x['mean_phase_gap'],abs(np.log(wp)-np.log(wa)).mean(),rtol=1e-12)
    for x in r['iso_runs']:
        w=np.asarray(x['omega_est'],float)
        if not x['invalid']:np.testing.assert_allclose(x['cv_log_omega'],np.log(w).std(),rtol=1e-12,atol=1e-12)
    for seed in SEEDS:
        data=replay('train_ellip',seed)
        dh=hashlib.sha256(data['trajectories'].astype(np.float32).tobytes()).hexdigest()
        tests={name:replay(name,seed) for name in ['test_id_ellip','test_ood_ecc']}
        targets={key:instantaneous_probe_targets(batch) for key,batch in tests.items()}
        for name in r['protocol']['models']:
            model,_=make_model(name,seed,'cpu',ROOT/'reports/Task_3_2/checkpoints')
            if name not in TRAINED:
                frozen=next(x for x in r['frozen_checks'] if x['seed']==seed and x['model']==name)
                if name=='M0_Koopman':
                    # Seeded initialization is not bitwise stable across torch versions.
                    saved=torch.load(p/'frozen_controls'/f'M0_Koopman_seed{seed}.pt',map_location='cpu',weights_only=True)
                    model.load_state_dict(saved['model_state_dict'])
                    assert saved['weights_sha256']==frozen['before_sha256']
                assert frozen['unchanged'] and frozen['before_sha256']==frozen['after_sha256']==_weights_hash(model)
            if name in TRAINED:
                saved=torch.load(out/'checkpoints'/f'{name}_seed{seed}.pt',map_location='cpu',weights_only=True)
                assert saved['step']==2000 and saved['config']['batch_size']==256 and saved['config']['device'].startswith('cuda')
                assert saved['provenance']['data_sha256']==dh and saved['provenance']['gpu_name']
                assert saved['provenance']['caller']['source_sha256']==r['source_sha256']
                model.load_state_dict(saved['model_state_dict'])
            model.double().eval()
            if seed==42 and name in ['M0_Koopman','M1_Ellip_Inv']:
                for split,batch in {'test_id_ellip':tests['test_id_ellip'],'test_ellip_peri':replay('test_ellip_peri',seed),'test_ellip_apo':replay('test_ellip_apo',seed)}.items():
                    sequences=t1_t3(model,batch,512)
                    for method,result in evaluate_frequencies(sequences['T3'],batch['meta']['a'],.04).items():
                        original=next(x for x in r['frequency_runs'] if x['seed']==seed and x['model']==name and x['split']==split and x['regime']=='T3' and x['method']==method)
                        np.testing.assert_allclose(np.asarray(original['omega_est'],float),np.asarray(result['omega_est'],float),rtol=2e-5,atol=1e-7,equal_nan=True)
            fit_features=probe_features(model,data)
            features={split:probe_features(model,batch) for split,batch in tests.items()}
            for x in (x for x in r['probe_runs'] if x['seed']==seed and x['model']==name):
                truth=targets[x['split']][x['target']];np.testing.assert_allclose(x['targets'],truth,atol=1e-12)
                raw=features[x['split']][x['feature']];cols=x['active_columns']
                fit_raw=fit_features[x['feature']]
                np.testing.assert_array_equal(cols,np.flatnonzero(fit_raw.std(0)>1e-6))
                fit_raw=fit_raw[:,cols] if cols else np.zeros((len(fit_raw),1))
                np.testing.assert_allclose(x['train_feature_mean'],fit_raw.mean(0),rtol=1e-7,atol=1e-9)
                fit_scale=fit_raw.std(0);fit_scale[fit_scale==0]=1.
                np.testing.assert_allclose(x['train_feature_scale'],fit_scale,rtol=1e-7,atol=1e-9)
                raw=raw[:,cols] if cols else np.zeros((len(raw),1))
                scaled=(raw-np.asarray(x['train_feature_mean']))/np.asarray(x['train_feature_scale'])
                if x['method']=='poly2':scaled=PolynomialFeatures(2,include_bias=False).fit_transform(scaled)
                pred=scaled@np.asarray(x['coefficients'])+x['intercept']
                np.testing.assert_allclose(x['predictions'],pred,rtol=2e-6,atol=2e-5)
                recorded=np.asarray(x['predictions']);sse=((truth-recorded)**2).sum();sst=((truth-truth.mean())**2).sum()
                np.testing.assert_allclose(x['r2'],1-sse/sst,atol=1e-10,rtol=1e-10)
    assert acceptance(r)==r['checks']
    receipt=dict(status='verified',previous_reports_unchanged=len(previous),artifact_hashes=len(manifest),source_hashes=len(r['source_sha256']),checkpoint_count=20,frequency_records=1120,probe_records=2880,independent_CPU_rollout_frequency_records=24,scientific_passed=sum(x['passed'] for x in r['checks']),scientific_total=len(r['checks']))
    (p/'audit_task4_results.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':torch.set_num_threads(4);main()
