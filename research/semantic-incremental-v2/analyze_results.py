"""Audit frozen runs, paired factorial contrasts, and publication-style plots.

Run after the configured complete matrix: python analyze_results.py --checkpoints
No held-out score is used to modify the mapper or select a seed.
"""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np

ROOT=Path.cwd()
METRICS=['psnr','ssim','object_sam_miou','fine_sam_miou','small_fine_sam_miou',
    'old_psnr','old_fine_sam_miou','old_psnr_change','old_fine_change',
    'gaussians','mapping_seconds','update_p95_ms','gpu_update_peak_mib',
    'evaluation_gpu_peak_mib','cpu_rss_peak_mib','process_seconds','controller_seconds_cumulative']
VARIANTS=[''.join(x) for x in itertools.product('01',repeat=3)]

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False))
def csv_write(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def mean_sd(values):
    a=np.asarray([v for v in values if v is not None],float)
    return dict(mean=float(a.mean()) if len(a) else None,
        sd=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a))

def factorial_contrasts(rows,seeds,dataset):
    """First average contexts within seed; seeds are the replicates."""
    lookup={(r['factors'],r['seed']):r for r in rows if r['dataset']==dataset}
    contrasts=[]
    for key in METRICS:
        for j,name in enumerate('SBR'):
            values=[]
            for seed in seeds:
                differences=[]
                for low in VARIANTS:
                    if low[j]=='1':continue
                    high=low[:j]+'1'+low[j+1:]
                    a,b=lookup[high,seed][key],lookup[low,seed][key]
                    if a is not None and b is not None:differences.append(a-b)
                values.append(float(np.mean(differences)))
            contrasts.append(dict(dataset=dataset,contrast='main_'+name,metric=key,
                **mean_sd(values),paired_values=values))
        for j,name in enumerate('SBR'):
            missing='111'[:j]+'0'+'111'[j+1:]
            values=[lookup['111',s][key]-lookup[missing,s][key] for s in seeds
                if lookup['111',s][key] is not None and lookup[missing,s][key] is not None]
            contrasts.append(dict(dataset=dataset,contrast='111_minus_'+missing+'_'+name,
                metric=key,**mean_sd(values),paired_values=values))
        values=[lookup['111',s][key]-lookup['000',s][key] for s in seeds
            if lookup['111',s][key] is not None and lookup['000',s][key] is not None]
        contrasts.append(dict(dataset=dataset,contrast='111_minus_000',metric=key,
            **mean_sd(values),paired_values=values))
        # Difference of effects, averaged across the remaining factor per seed.
        for a,b in itertools.combinations(range(3),2):
            other=next(i for i in range(3) if i not in (a,b));values=[]
            for seed in seeds:
                contexts=[]
                for c in '01':
                    v=[]
                    for x,y in [('1','1'),('1','0'),('0','1'),('0','0')]:
                        flag=['0']*3;flag[a]=x;flag[b]=y;flag[other]=c
                        v.append(lookup[''.join(flag),seed][key])
                    if all(t is not None for t in v):contexts.append(v[0]-v[1]-v[2]+v[3])
                if contexts:values.append(float(np.mean(contexts)))
            contrasts.append(dict(dataset=dataset,contrast='interaction_'+'SBR'[a]+'SBR'[b],
                metric=key,**mean_sd(values),paired_values=values))
    return contrasts

def audit_prefix(folder,plan):
    anchor=np.load(folder/'anchors.npz');release=anchor['release'];rows=[]
    for stage in range(1,plan['stages']+1):
        meta=read(folder/f'prefix-{stage}.json');a=np.load(folder/f'prefix-{stage}.npz')
        ids=a['ids'];nodes=meta['nodes'];end=meta['end']
        assert ids.shape==(2,len(release)) and not np.any(ids[:,release>=end])
        assert meta['max_teacher_frame']<end
        assert all(i<end for i in meta['consumed_teacher_frames'])
        assert len(nodes)<=1024
        assert all(n['born_frame']<end and all(i<end for i in n['observations']) for n in nodes.values())
        assert all(len(n['descriptors'])<=4 for n in nodes.values())
        known=ids>0
        assert set(np.unique(ids[known])).issubset(set(map(int,nodes)))
        for level in range(2):
            assert all(nodes[str(int(k))]['level']==level for k in np.unique(ids[level]) if k>0)
        assert all(not n['parent'] or nodes[str(n['parent'])]['level']==0 for n in nodes.values())
        # Measure candidate containment, without assuming it is a correct part tree.
        fine=ids[1];parent=np.zeros(max(map(int,nodes),default=0)+1,np.int32)
        for k,n in nodes.items():parent[int(k)]=n['parent']
        good=(fine>0)&(ids[0]>0)&(parent[fine]>0)
        agreement=float(np.mean(ids[0,good]==parent[fine[good]])) if good.any() else None
        rows.append(dict(stage=stage,end=end,nodes=len(nodes),available_anchors=int((release<end).sum()),
            object_anchor_coverage=float(np.mean(ids[0,release<end]>0)),
            fine_anchor_coverage=float(np.mean(ids[1,release<end]>0)),
            candidate_parent_anchor_agreement=agreement,parent_comparable_anchors=int(good.sum())))
    return rows

def audit_run(folder,plan,ds,seed,factors,checkpoints):
    run=read(folder/'run.json');args=run['args'];history=read(folder/'metrics.json')
    assert args['factors']==factors and args['seed']==seed
    assert args['cap']==plan['cap'] and args['steps']==plan['steps'] and args['width']==plan['width']
    assert run['optimizer_steps']==plan['steps']*plan['stages'] and len(history)==plan['stages']
    assert 'L4' in run['device']
    train_test_mod=6 if ds.startswith('tum-') else 8
    train_test_remainder=3 if ds.startswith('tum-') else 0
    for row in history:
        assert row['steps']==plan['steps'] and 0<row['gaussians']<=row['cap']<=plan['cap']
        assert row['teacher_max_frame']<row['end']
        assert len(row['replay_pool'])<=48
        assert all(i<round(history[-1]['end']*(row['stage']-1)/plan['stages']) for i in row['replay_pool'])
        assert all(i%train_test_mod!=train_test_remainder for i in row['replay_pool'])
        assert all(v is None or np.isfinite(v) for k,v in row.items() if k!='replay_pool')
        rgb=read(folder/f'rgb-quality-{row["stage"]}.json')
        assert all(r['index']<row['end'] and r['index']%train_test_mod==train_test_remainder for r in rgb)
    for sample in read(folder/'sampling-audit.json'):
        row=history[sample['stage']-1]
        assert sample['frame']<row['end'] and sample['frame']%train_test_mod!=train_test_remainder
    old_end=history[0]['end']
    first=read(folder/'rgb-quality-1.json');last=read(folder/f'rgb-quality-{plan["stages"]}.json')
    assert {r['index'] for r in first}=={r['index'] for r in last if r['index']<old_end}
    # Check identities and numeric validity, without requiring a GPU.
    checkpoint_audit=None
    if checkpoints:
        import torch
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        n=len(ck['anchor_ids']);assert n==history[-1]['gaussians']
        assert ck['semantic_ids'].shape==(2,n) and ck['step']==run['optimizer_steps']
        assert all(len(v)==n and bool(torch.isfinite(v).all()) for v in ck['params'].values())
        assert bool((ck['params']['scales'].exp()>0).all())
        assert bool((ck['params']['quats'].norm(dim=-1)>0).all())
        anchors=np.load(ROOT/'data'/ds/'prefixes'/'anchors.npz')
        assert int(ck['anchor_ids'].min())>=0 and int(ck['anchor_ids'].max())<len(anchors['release'])
        assert np.all(anchors['release'][ck['anchor_ids'].numpy()]<ck['end'])
        expected=np.load(ROOT/'data'/ds/'prefixes'/f'prefix-{plan["stages"]}.npz')['ids'][:,ck['anchor_ids'].numpy()]
        np.testing.assert_array_equal(expected,ck['semantic_ids'])
        checkpoint_audit=dict(gaussians=n,finite=True,semantic_identity=True)
    f=history[-1]
    row={k:f.get(k) for k in METRICS}
    row.update(dataset=ds,seed=seed,factors=factors,mapping_seconds=sum(r['train_seconds'] for r in history),
        gpu_update_peak_mib=max(r['gpu_update_peak_mib'] for r in history),
        evaluation_gpu_peak_mib=max(r['evaluation_gpu_peak_mib'] for r in history),
        old_psnr_change=f['old_psnr']-history[0]['old_psnr'],
        old_fine_change=(f['old_fine_sam_miou']-history[0]['old_fine_sam_miou']
            if f['old_fine_sam_miou'] is not None and history[0]['old_fine_sam_miou'] is not None else None),
        process_seconds=read(folder/'process-wall.json')['seconds'])
    events=read(folder/'density-events.json')
    row.update(growth_events=sum(e['event']=='growth' for e in events),
        pruned=sum(e.get('removed',0) for e in events),
        protected_max=max((e.get('protected',0) for e in events),default=0),
        optimized_pixels_total=sum(r['optimized_pixels'] for r in history))
    assert all(e.get('semantic',factors[1]=='1')==(factors[1]=='1') for e in events)
    if factors[1]=='0':assert row['protected_max']==0
    return row,history,checkpoint_audit

def plot(summary,stages,out,datasets,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    selected=['000','011','101','110','111']
    labels={'000':'Baseline','011':'Without S','101':'Without B','110':'Without R','111':'S+B+R'}
    for ds in datasets:
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for ax,key,title in zip(axes.flat,['psnr','fine_sam_miou','old_psnr_change','update_p95_ms'],
            ['Held-out PSNR (dB)','Fine SAM region agreement','Old-view PSNR change (dB)','Final-stage update p95 (ms)']):
            vals=[summary[ds][f][key]['mean'] for f in VARIANTS]
            err=[summary[ds][f][key]['sd'] or 0 for f in VARIANTS]
            ax.bar(VARIANTS,vals,yerr=err,color=['#74808b' if f=='000' else '#dc862c' if f=='111' else '#407fa5' for f in VARIANTS],capsize=3)
            ax.set_title(title);ax.set_xlabel('Switches S / B / R');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(ds+f' | mean +/- sample SD, {len(seeds)} paired seeds');fig.savefig(out/f'{ds}-factorial.png');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        for ax,key,title in zip(axes,['old_psnr','old_fine_sam_miou','gaussians'],
                ['Same early held-out views: PSNR','Same early views: fine SAM agreement','Actual Gaussian count']):
            for factors in selected:
                vals=np.asarray([[r[key] for r in stages[ds,factors,s]] for s in seeds])
                mean=vals.mean(0);sd=vals.std(0,ddof=1);x=np.arange(1,7)
                ax.plot(x,mean,label=labels[factors]);ax.fill_between(x,mean-sd,mean+sd,alpha=.1)
            ax.set_title(title);ax.set_xlabel('Arrived-data stage');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8);fig.suptitle(ds+' | stage curves');fig.savefig(out/f'{ds}-stages.png');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoints',action='store_true');args=parser.parse_args()
    plan=read(ROOT/'experiment-plan.json');status=read(ROOT/'status.json')
    assert status['state']=='complete' and len(status['completed'])==len(plan['datasets'])*len(plan['seeds'])*8,'Wait for all runs; partial data are not final comparisons.'
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    frozen=read(ROOT/'source-sha256.json')
    for name,digest in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name+' changed during matrix'
    datasets=[d['name'] for d in plan['datasets']];seeds=plan['seeds'];rows=[];stages={};checks=[];prefixes={}
    for ds in datasets:
        prefixes[ds]=audit_prefix(ROOT/'data'/ds/'prefixes',plan)
        teacher=read(ROOT/'data'/ds/'teachers'/'index.json')
        train_indices={r['index'] for r in teacher if not r['test']}
        for stage in range(1,7):
            meta=read(ROOT/'data'/ds/'prefixes'/f'prefix-{stage}.json')
            assert set(meta['consumed_teacher_frames']).issubset(train_indices)
        for seed in seeds:
            for factors in VARIANTS:
                folder=ROOT/'results'/ds/factors/f'seed-{seed}'
                row,h,ck=audit_run(folder,plan,ds,seed,factors,args.checkpoints)
                rows.append(row);stages[ds,factors,seed]=h;checks.append(dict(dataset=ds,seed=seed,factors=factors,checkpoint=ck))
            # Independent crop/replay Bernoulli stream must give equal optimized pixels.
            for stage in range(6):
                assert len({stages[ds,f,seed][stage]['optimized_pixels'] for f in VARIANTS})==1
            # Verify the controllers have a measurable effect on the sampled trace/pool.
            base=read(ROOT/'results'/ds/'000'/f'seed-{seed}'/'sampling-audit.json')
            assert base!=read(ROOT/'results'/ds/'100'/f'seed-{seed}'/'sampling-audit.json')
            assert any(stages[ds,'000',seed][i]['replay_pool']!=stages[ds,'001',seed][i]['replay_pool'] for i in range(6))
    if (ROOT/'gpu-telemetry.csv').exists():
        with (ROOT/'gpu-telemetry.csv').open() as f:telemetry=list(csv.DictReader(f))
        first_job=min((int(r['job']) for r in telemetry),default=0)
        for row in rows:
            t=[r for r in telemetry if (r['dataset'],r['factors'],int(r['seed']))==(row['dataset'],row['factors'],row['seed'])]
            row['whole_gpu_sampled_peak_mib']=max((float(r['gpu_used_mib']) for r in t),default=None)
            row['whole_gpu_samples']=len(t)
            row['whole_gpu_partial_coverage']=bool(t and int(t[0]['job'])==first_job)
    summary={}
    for ds in datasets:
        summary[ds]={f:{k:mean_sd([r.get(k) for r in rows if r['dataset']==ds and r['factors']==f]) for k in METRICS+['whole_gpu_sampled_peak_mib']} for f in VARIANTS}
    contrasts=[]
    for ds in datasets:contrasts+=factorial_contrasts(rows,seeds,ds)
    save(out/'summary.json',summary);save(out/'contrasts.json',contrasts);save(out/'prefix-audit.json',prefixes)
    save(out/'audit.json',dict(passed=True,runs=len(rows),frozen_sources_unchanged=True,
        paired_pixels_equal=True,all_checkpoint_numeric_checks=args.checkpoints,checks=checks,
        qualifications=[f'N={len(seeds)} seeds: descriptive uncertainty, no significance or SOTA claim',
            'TUM known GT poses; kitchen offline COLMAP inputs',
            'SAM pseudo-region agreement is not human-annotated semantic/part accuracy',
            'Whole-GPU nvidia-smi 1 Hz samples can miss brief peaks; first observed job is partially covered']))
    csv_write(out/'runs.csv',rows);csv_write(out/'contrasts.csv',contrasts)
    csv_write(out/'stages.csv',[dict(dataset=d,factors=f,seed=s,**{k:v for k,v in r.items() if k!='replay_pool'}) for (d,f,s),h in stages.items() for r in h])
    plot(summary,stages,out,datasets,seeds)
    print(f'ALL_{len(rows)}_RUNS_AUDITED: equal steps, caps and optimized pixel budgets; no future teacher input')
    for ds in datasets:
        print(ds)
        for f in VARIANTS:
            vals=summary[ds][f]
            print(f,'PSNR',round(vals['psnr']['mean'],3),'fine SAM',round(vals['fine_sam_miou']['mean'],4),
                'old delta',round(vals['old_psnr_change']['mean'],3),'p95 ms',round(vals['update_p95_ms']['mean'],2))

if __name__=='__main__':main()
