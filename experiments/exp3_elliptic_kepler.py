"""Task4: all four ellipse blocks, 20 fixed CUDA trainings, honest acceptance."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from threadpoolctl import threadpool_limits

from src.data import build_dataset_ellip_task4,build_peri_apo_task4,instantaneous_probe_targets
from src.evaluation import evaluate_frequencies
from src.models import M0_Koopman,M1_IJEPA1D,M2_JEPA_OPF,M3_IJEPA_SIGReg
from src.train import TrainConfig,train_model
from .exp1_encoder_tautology import _write_json,_Tee
from .exp2_circular_trained import gpu_preflight,_model as circular_model,_weights_hash

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,43,44,45,46)
TRAINED=('M1_Ellip_Short','M1_Ellip_Inv','M2_Ellip_Inv','M3_Ellip_SIGReg')
MODELS=('M0_Koopman','M1_circ','M2_circ','M3_circ',*TRAINED)
PRIMARY='S3b_diff'


def source_hashes():
    files=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'experiments').glob('*.py')),ROOT/'requirements-gpu.txt']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def config_for(name,device='cuda',smoke=False):
    variant='short' if name=='M1_Ellip_Short' else 'sigreg' if name=='M3_Ellip_SIGReg' else 'inv'
    values=dict(protocol='task4_ellip',ellip_variant=variant,device=device,
                horizon=32 if variant=='short' else 256,frequency_invariance_weight=0. if variant=='short' else 2.,
                coherence_weight=0.,factor_scale_weight=0.)
    if smoke:values.update(steps=3,batch_size=8,ar_steps=2,temporal_lag=2,
                          horizon=8 if variant=='short' else 16,smoke_test=True,log_every=1,checkpoint_every=1)
    return TrainConfig(**values)


def make_model(name,seed,device,checkpoints):
    if name.endswith('_circ'):
        base=name[:2];model=circular_model(base,seed,device)
        path=checkpoints/f'{base}_seed{seed}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['step']!=2000 or saved['config']['protocol']!='task3_2':raise ValueError('Wrong circular checkpoint')
        model.load_state_dict(saved['model_state_dict']);model.requires_grad_(False)
        return model,{'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    torch.manual_seed(seed)
    kwargs=dict(L=16,C=4,d=64,H_max=512 if name=='M0_Koopman' else 32 if name=='M1_Ellip_Short' else 256,
                arch_type='mlp',dtype=torch.float32)
    cls=M0_Koopman if name=='M0_Koopman' else M2_JEPA_OPF if name.startswith('M2') else M3_IJEPA_SIGReg if name.startswith('M3') else M1_IJEPA1D
    if cls is M3_IJEPA_SIGReg:kwargs['normalize_embeddings']=True
    return cls(**kwargs).to(device),{}


def t1_t3(model,batch,n=512):
    """Independent truth encoding and closed-loop rollout; no orbital metadata."""
    data=batch['trajectories'];parameter=next(model.parameters())
    starts=np.arange(n+1)*4
    windows=torch.as_tensor(data[:,starts[:,None]+np.arange(16)],device=parameter.device,dtype=parameter.dtype)
    with torch.inference_mode():
        z=model.encode_sequence(windows,batch_size=2048)
        state=z[:,0];ar=[];basis=model.predictor.orthogonal_basis()
        for _ in range(n):state=model.predictor(state,1,basis=basis);ar.append(state)
    return {'T1':z[:,1:].cpu().numpy(),'T3':torch.stack(ar,1).cpu().numpy()}


def probe_features(model,batch):
    parameter=next(model.parameters())
    with torch.inference_mode():
        z=model.encoder(torch.as_tensor(batch['trajectories'][:,:16].copy(),dtype=parameter.dtype,device=parameter.device))
        omega=model.predictor.angular_rates(z)
    values=z.cpu().numpy();rates=omega.cpu().numpy()
    return {'norm':np.log(np.linalg.norm(values,axis=1)/np.sqrt(values.shape[1])+1e-6)[:,None],
            'state':values,'log_omega':np.log(rates+1e-8)}


def oos_probes(train_x,test_x,train_y,test_y):
    """Train-only scaling/column selection and fixed Ridge/poly2 probes."""
    records=[]
    for feature,x in train_x.items():
        active=x.std(0)>1e-6
        fit_x=x[:,active] if active.any() else np.zeros((len(x),1))
        for method in ('ridge','poly2'):
            for target,y in train_y.items():
                steps=[StandardScaler()]
                if method=='poly2':steps.append(PolynomialFeatures(2,include_bias=False))
                steps.append(Ridge(alpha=1e-2))
                fit=make_pipeline(*steps);fit.fit(fit_x,y)
                for split,features in test_x.items():
                    values=features[feature][:,active] if active.any() else np.zeros((len(features[feature]),1))
                    prediction=fit.predict(values);truth=test_y[split][target]
                    sse=float(np.sum((truth-prediction)**2));sst=float(np.sum((truth-truth.mean())**2))
                    records.append({'feature':feature,'method':method,'target':target,'split':split,
                        'r2':1-sse/sst if sst else float('nan'),'sse':sse,'sst':sst,
                        'predictions':prediction.tolist(),'targets':truth.tolist(),
                        'active_columns':np.flatnonzero(active).tolist(),'fit_split':'train_ellip',
                        'n_fit_orbits':len(x),'alpha':.01,'train_feature_mean':fit[0].mean_.tolist(),
                        'train_feature_scale':fit[0].scale_.tolist(),
                        'coefficients':fit[-1].coef_.tolist(),'intercept':float(fit[-1].intercept_)})
    return records


def phase_gap(peri,apo):
    p=np.asarray(peri,dtype=float);a=np.asarray(apo,dtype=float)
    valid=np.isfinite(p)&np.isfinite(a)&(p>0)&(a>0)
    ratio=np.full(len(p),np.nan);errors=ratio.copy()
    ratio[valid]=p[valid]/a[valid];errors[valid]=np.abs(np.log(p[valid])-np.log(a[valid]))
    return {'rho':ratio.tolist(),'absolute_log_gaps':errors.tolist(),'invalid':int((~valid).sum()),
            'mean_phase_gap':float(errors.mean()) if valid.all() else float('nan')}


def _finite(v):
    return v is not None and bool(np.isfinite(v))


def acceptance(report):
    checks=[]
    def add(label,passed,details=None):checks.append({'criterion':label,'passed':bool(passed),'details':details})
    def rows(field,**select):return [r for r in report[field] if all(r.get(k)==v for k,v in select.items())]
    def five(records):return len(records)==5 and {r['seed'] for r in records}==set(SEEDS)
    training=report['training_runs']
    add('All20 CUDA trainings:2000steps B256',len(training)==20 and
        {(r['seed'],r['model']) for r in training}=={(s,m) for s in SEEDS for m in TRAINED} and all(
        r.get('completed_steps')==2000 and r.get('device','').startswith('cuda') and
        r.get('config',{}).get('batch_size')==256 and r.get('status')=='completed' for r in training))
    for model in ('M0_Koopman','M1_Ellip_Inv'):
        candidates={}
        for method in ('S3b_std','S3b_diff'):
            r=rows('frequency_runs',model=model,split='test_id_ellip',regime='T1',method=method)
            candidates[method]=five(r) and all(x['invalid']==0 and _finite(x['slope_abs_error']) and x['slope_abs_error']<.015 and _finite(x['r2']) and x['r2']>.999 for x in r)
        add(f'H3 {model}: one phase method passes all5 seeds',any(candidates.values()),candidates)
        dmd=rows('frequency_runs',model=model,split='test_id_ellip',regime='T1',method='S3a_diff')
        phase=rows('frequency_runs',model=model,split='test_id_ellip',regime='T1',method='S3b_std')
        complete=five(dmd) and five(phase) and all(_finite(x['r2']) and _finite(x['slope_abs_error']) for x in dmd+phase)
        add(f'H3 {model}: DMD degradation',complete and (np.mean([x['r2'] for x in dmd])<np.mean([x['r2'] for x in phase])-.02 or np.mean([x['slope_abs_error'] for x in dmd])>.05))
    circ=rows('phase_runs',model='M1_circ');inv=rows('phase_runs',model='M1_Ellip_Inv')
    good_c=five(circ) and all(_finite(r['mean_phase_gap']) and r['invalid']==0 for r in circ)
    good_i=five(inv) and all(_finite(r['mean_phase_gap']) and r['invalid']==0 for r in inv)
    add('Zero-shot M1_circ mean phase gap>.35',good_c and np.mean([r['mean_phase_gap'] for r in circ])>.35)
    add('M1_Ellip_Inv phase gap<.12 all5 and at least3x mean reduction',good_c and good_i and all(r['mean_phase_gap']<.12 for r in inv) and np.mean([r['mean_phase_gap'] for r in inv])*3<np.mean([r['mean_phase_gap'] for r in circ]))
    r=rows('frequency_runs',model='M0_Koopman',split='test_id_ellip',regime='T3',method=PRIMARY)
    add('M0_Koopman T3 fails recovery',five(r) and all(_finite(x['slope_abs_error']) and _finite(x['r2']) and (x['slope_abs_error']>.5 or x['r2']<.60) for x in r))
    for model in ('M1_Ellip_Inv','M2_Ellip_Inv'):
        r=rows('frequency_runs',model=model,split='test_id_ellip',regime='T3',method=PRIMARY)
        valid=five(r) and all(x['invalid']==0 and _finite(x['slope_abs_error']) and _finite(x['r2']) for x in r)
        add(f'{model} T3 mean/worst law thresholds',valid and np.mean([x['slope_abs_error'] for x in r])<.06 and np.mean([x['r2'] for x in r])>.95 and max(x['slope_abs_error'] for x in r)<.09 and min(x['r2'] for x in r)>.92)
    for feature,method,target,label,threshold,high in (
        ('norm','ridge','log_a','M1Inv norm OOS loga R2<.75',.75,False),
        ('log_omega','poly2','log_abs_energy','M1Inv omega OOS energy R2>.94',.94,True)):
        r=rows('probe_runs',model='M1_Ellip_Inv',split='test_id_ellip',feature=feature,method=method,target=target)
        add(label,five(r) and all(_finite(x['r2']) and (x['r2']>threshold if high else x['r2']<threshold) for x in r))
    return checks


def evaluate_model(model,name,seed,batches,train,report,n):
    model.double().eval();freqs={}
    for split,batch in batches.items():
        sequences=t1_t3(model,batch,n)
        for regime in ('T1','T3') if split in ('test_id_ellip','test_ood_ecc') else ('T3',):
            results=evaluate_frequencies(sequences[regime],batch['meta']['a'],.04)
            for method,r in results.items():
                q=np.asarray(r['omega_est'])/batch['meta']['omega_true']
                report['frequency_runs'].append({'seed':seed,'model':name,'split':split,'regime':regime,**r,'q':q.tolist()})
            if regime=='T3':freqs[split]=results[PRIMARY]['omega_est']
        if split=='test_iso_energy':
            w=np.asarray(freqs[split]);a=batch['meta']['a']
            for axis in (.7,1.2,1.8):
                mask=a==axis;values=w[mask];valid=np.isfinite(values)&(values>0)
                report['iso_runs'].append({'seed':seed,'model':name,'a':axis,'e':batch['meta']['e'][mask].tolist(),
                    'omega_est':values.tolist(),'q':(values/axis**-1.5).tolist(),'invalid':int((~valid).sum()),
                    'cv_log_omega':float(np.log(values).std(ddof=0)) if valid.all() else float('nan')})
        print(f'evaluate seed={seed} model={name} split={split}',flush=True)
    report['phase_runs'].append({'seed':seed,'model':name,**phase_gap(freqs['test_ellip_peri'],freqs['test_ellip_apo'])})
    train_x=probe_features(model,train);test_x={k:probe_features(model,batches[k]) for k in ('test_id_ellip','test_ood_ecc')}
    train_y=instantaneous_probe_targets(train);test_y={k:instantaneous_probe_targets(batches[k]) for k in test_x}
    report['probe_runs'].extend({'seed':seed,'model':name,**r} for r in oos_probes(train_x,test_x,train_y,test_y))


def run(args):
    seeds=(42,) if args.smoke_test else SEEDS;n=16 if args.smoke_test else 512
    env=gpu_preflight()
    if not args.smoke_test and not env['cuda_available']:raise RuntimeError('CUDA unavailable; no CPU substitution')
    report={'experiment':'exp3_elliptic_kepler','status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':{'seeds':list(seeds),'smoke_test':args.smoke_test,'N_eval':n,'raw_samples':2100,
                        'models':list(MODELS),'primary_method':PRIMARY,'training_configs':{m:asdict(config_for(m,args.device,args.smoke_test)) for m in TRAINED},
                        'probe':'train-only StandardScaler + Ridge(.01) or degree2 polynomial + Ridge(.01)',
                        'target_semantics':'first physical sample of context t0; energy target affine in loga',
                        'source_apsis':'first sample t=0; not context-centered',
                        'training_time_sampling':'two independent valid start indices in one2100sample record; not full-cycle coverage for all a',
                        'M3_objective':'paired sparse long horizons + relative AR + invariance2 + SIGReg1; no temporal scale',
                        'short_objective':'one start direct1..32 + relative AR16 + lag16 temporal; invariance0',
                        'harmonics':'no across-plane coherence or factor-scale in Task4; rates may differ by plane',
                        'H3':'either std or diff must pass all5 for each model; never per-seed method selection; DMD compared with std',
                        'norm_acceptance_model':'M1_Ellip_Inv','energy_acceptance_probe':'poly2 on all32 log-rates',
                        'evaluation_dtype':'float64 after saving float32 weights'},
            'environment':env,'source_sha256':source_hashes(),'training_runs':[],'frequency_runs':[],
            'phase_runs':[],'iso_runs':[],'probe_runs':[],'frozen_checks':[],'checks':[]}
    started=time.perf_counter();path=args.output/'exp3_elliptic_results.json'
    for seed in seeds:
        train=build_dataset_ellip_task4('train_ellip',seed)
        batches={k:build_dataset_ellip_task4(k,seed) for k in ('test_id_ellip','test_ood_ecc','test_iso_energy')}
        batches.update({'test_ellip_'+k:v for k,v in build_peri_apo_task4().items()})
        report.setdefault('dataset_metadata',{})[str(seed)]={k:{key:np.asarray(value).tolist() for key,value in b['meta'].items()} for k,b in {'train_ellip':train,**batches}.items()}
        for name in MODELS:
            model,origin=make_model(name,seed,args.device,args.circular_checkpoints)
            before=_weights_hash(model) if name not in TRAINED else None
            if name in TRAINED:
                config=config_for(name,args.device,args.smoke_test);checkpoint=args.output/'checkpoints'/f'{name}_seed{seed}.pt'
                def log(row):print(f"seed={seed} model={name} step={row['step']}/{config.steps} total={row['total']:.6g} inv={row['frequency_invariance']:.6g}",flush=True)
                result=train_model(model,train['trajectories'],config,seed,checkpoint,log,
                    resume_from=checkpoint if args.resume and checkpoint.exists() else None,
                    provenance={'experiment':'task4','source_sha256':report['source_sha256']})
                report['training_runs'].append({'seed':seed,'model':name,'status':'completed',**result})
            evaluate_model(model,name,seed,batches,train,report,n)
            if before is not None:
                after=_weights_hash(model,original_dtype=torch.float32)
                if before!=after:raise RuntimeError('Frozen checkpoint weights changed')
                report['frozen_checks'].append({'seed':seed,'model':name,'before_sha256':before,'after_sha256':after,'unchanged':True,**origin})
            _write_json(path,report);del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    if source_hashes()!=report['source_sha256']:raise RuntimeError('Source changed during experiment')
    expected={'training_runs':4*len(seeds),'frequency_runs':224*len(seeds),
              'phase_runs':8*len(seeds),'iso_runs':24*len(seeds),
              'probe_runs':576*len(seeds),'frozen_checks':4*len(seeds)}
    report['record_counts']={key:len(report[key]) for key in expected}
    if report['record_counts']!=expected:raise RuntimeError(f'Incomplete experiment records: {report["record_counts"]}, expected {expected}')
    report['execution_completed']=True;report['status']='smoke_completed' if args.smoke_test else 'completed'
    report['checks']=acceptance(report);report['acceptance_passed']=all(c['passed'] for c in report['checks'])
    report['elapsed_seconds']=time.perf_counter()-started;_write_json(path,report)
    from .exp3_reporting import write_report
    write_report(report,args.output)
    for c in report['checks']:print(('PASS' if c['passed'] else 'FAIL')+': '+c['criterion'],flush=True)
    return 0 if args.smoke_test or report['acceptance_passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/Task_4')
    parser.add_argument('--circular-checkpoints',type=Path,default=ROOT/'reports/Task_3_2/checkpoints')
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--smoke-test',action='store_true');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.smoke_test:args.output=args.output/'smoke';args.device='cpu'
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    if args.preflight:
        env=gpu_preflight();_write_json(args.output/'gpu_preflight.json',env);print(json.dumps(env,indent=2));return 0 if env['cuda_available'] else 2
    with (args.output/'exp3_elliptic_stdout.txt').open('w',encoding='utf-8') as stream:
        with redirect_stdout(_Tee(__import__('sys').stdout,stream)),threadpool_limits(limits=args.threads):return run(args)


if __name__=='__main__':raise SystemExit(main())
