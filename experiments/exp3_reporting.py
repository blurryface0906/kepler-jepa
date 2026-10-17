"""Complete Task4 tables and a predeclared four-panel scientific figure."""
from pathlib import Path
import numpy as np
from tabulate import tabulate


def table(rows, columns):
    return tabulate([[r.get(key) for key in columns] for r in rows],headers=columns,tablefmt='github',floatfmt='.6g')


def write_report(report, output):
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    lines=['# Task 4: elliptical Kepler experiment',
           f"Execution: {report['status']}. Scientific acceptance: {report['acceptance_passed']}.",
           'All seeds and invalid estimates are retained. No threshold or estimator is selected after observing outcomes.',
           '## Acceptance',table(report['checks'],['criterion','passed','details']),
           '## GPU training',table(report['training_runs'],['seed','model','status','completed_steps','device','elapsed_seconds']),
           '## Frequency estimates: all models, splits and estimators',
           table(report['frequency_runs'],['seed','model','split','regime','method','slope','slope_abs_error','r2','invalid']),
           '## Pericenter versus apocenter',table(report['phase_runs'],['seed','model','mean_phase_gap','invalid']),
           'Per-orbit ratios and absolute log gaps are in the JSON; invalid orbits make the aggregate undefined.',
           '## Isoenergetic dispersion',table(report['iso_runs'],['seed','model','a','cv_log_omega','invalid']),
           'cv_log_omega is the population standard deviation of log frequency, as specified; it is not std/mean.',
           '## Train-only physical probes',table(report['probe_runs'],['seed','model','split','feature','method','target','r2']),
           'Column selection, standardization and coefficients use only the 200 training orbits. Negative test R2 is retained.',
           '## Frozen controls',table(report['frozen_checks'],['seed','model','unchanged','before_sha256','after_sha256']),
           '## Interpretation limits',
           '- A 256-step horizon spans 10.24 time units; the period at a=5 is 70.25. This protocol does not cover a full cycle for every training orbit.',
           '- Paired starts share an orbit identity, which is additional supervision. The short and invariant variants also differ in horizon and number of anchors; this comparison does not isolate the invariance penalty alone.',
           '- Failure on ellipses does not identify a unique norm heuristic. Successful frequency probes do not prove causal use of physical energy.',
           '- log(abs(E)) is an affine transform of log(a). Their probe scores are not independent evidence.',
           '- Circular motion in latent action-angle coordinates can represent nonuniform motion on physical ellipses.',
           '- Finite-window phase regression need not recover the mean orbital frequency even when phase winding per full orbit is correct.',
           'All scalar aggregates below use all five seeds, without filtering failed seeds.']
    if not report['protocol']['smoke_test']:
        figure(report,output)
        lines.append('![Four-panel figure](figures/exp3_elliptic_kepler.png)')
    (output/'exp3_elliptic_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def figure(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def select(field,**kw):return [r for r in report[field] if all(r.get(k)==v for k,v in kw.items())]
    def numbers(rows,key):return np.asarray([np.nan if r[key] is None else r[key] for r in rows],float)
    fig,axes=plt.subplots(2,2,figsize=(17,11),layout='constrained')
    ax=axes[0,0];methods=['S3b_std','S3b_diff','S3a_diff','S2']
    # Explicit seed dots avoid hiding failed or unstable seeds behind means.
    for j,model in enumerate(['M0_Koopman','M1_Ellip_Inv']):
        for k,method in enumerate(methods):
            rows=select('frequency_runs',model=model,split='test_id_ellip',regime='T1',method=method)
            vals=numbers(rows,'slope_abs_error');x=k+(j-.5)*.23
            ax.scatter(np.full(len(vals),x),vals,label=model if k==0 else None,marker=['o','x'][j])
    ax.axhline(.015,color='black',ls=':',lw=1);ax.set_xticks(range(4),methods,rotation=15)
    ax.set_ylabel('Absolute slope error');ax.set_title('(a) T1 on ellipses: all five seeds');ax.legend(fontsize=8)
    ax=axes[0,1]
    for model,marker in [('M1_circ','o'),('M1_Ellip_Inv','x')]:
        for phase,color in [('peri','tab:red'),('apo','tab:blue')]:
            rows=select('frequency_runs',model=model,split='test_ellip_'+phase,regime='T3',method='S3b_diff')
            a=np.asarray(report['dataset_metadata']['42']['test_ellip_'+phase]['a'])
            # Geometric mean across ALL seeds; invalid estimates remain visible as missing points.
            w=np.asarray([[np.nan if v is None else v for v in r['omega_est']] for r in rows])
            logw=np.log(w);ax.scatter(np.log(a),logw.mean(0),color=color,marker=marker,s=20,label=model+' '+phase)
    ax.plot(np.log(a),-1.5*np.log(a),'k--',lw=1,label='True mean motion')
    ax.set(xlabel='log(a)',ylabel='log estimated frequency',title='(b) Peri/apo: five-seed mean log frequency');ax.legend(fontsize=7)
    ax=axes[1,0]
    for j,model in enumerate(['M1_Ellip_Inv','M2_Ellip_Inv']):
        for a,color in zip([.7,1.2,1.8],['tab:blue','tab:orange','tab:green']):
            rows=select('iso_runs',model=model,a=a);q=np.asarray([[np.nan if v is None else v for v in r['q']] for r in rows])
            e=rows[0]['e'];mean=q.mean(0);sd=q.std(0)
            ax.plot(e,mean,color=color,ls=['-','--'][j],label=f'{model}, a={a}')
            ax.fill_between(e,mean-sd,mean+sd,color=color,alpha=.09)
    ax.axhline(1,color='black',ls=':',lw=1);ax.set(xlabel='Eccentricity',ylabel='Estimated / true mean frequency',title='(c) Isoenergy: five-seed mean and SD');ax.legend(fontsize=7)
    ax=axes[1,1];targets=['log_r0','log_v0','log_a','e','log_h']
    for j,(feature,method) in enumerate([('norm','ridge'),('state','poly2'),('log_omega','poly2')]):
        means=[];stds=[]
        for target in targets:
            vals=numbers(select('probe_runs',model='M1_Ellip_Inv',split='test_id_ellip',feature=feature,method=method,target=target),'r2')
            means.append(vals.mean());stds.append(vals.std())
        ax.bar(np.arange(5)+(j-1)*.24,means,width=.24,yerr=stds,label=feature+' / '+method,capsize=2)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(5),targets,rotation=15)
    ax.set(ylabel='Out-of-sample R²',title='(d) M1 invariant: train → ID, five-seed mean and SD');ax.legend(fontsize=8)
    fig.suptitle('Elliptical Kepler tests — fixed protocol; negative results retained')
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'exp3_elliptic_kepler.png',dpi=300);plt.close(fig)
