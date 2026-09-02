"""Reproducible controller execution with optional result persistence."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from threading import Lock
import csv
import json
import platform
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import ttest_rel
from algorithms.classical.rcga import RCGAConfig, RCGAOptimizer
from .protocol import make_environment, derive_seed
from .surrogate import enumerate_states, encode_plan, fit_qubo, energy, fidelity
from .sampling import sample_qaoa, select_candidates, balanced_expand, annealing_visits

ROOT=Path(__file__).resolve().parents[1]
SUPPORTED=('qaoa_rcga','rcga','pso_mpc','mpc_receding','tightened_mpc',
           'es_policy_search','sac_ppo','tube_rmpc',
           'random','exact_sim_top1','exact_sim_top5','exact_sim_top10','exact_sim_top50','sa_qubo_50','qaoa_50')


def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,np.ndarray): return safe(value.tolist())
    if isinstance(value,np.generic): return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def source_digest():
    h=sha256()
    for folder in ('paper_protocol','algorithms','environment'):
        for path in sorted((ROOT/folder).rglob('*.py')):
            h.update(path.relative_to(ROOT).as_posix().encode()); h.update(path.read_bytes())
    return h.hexdigest()


class CountedEnvironment:
    def __init__(self,env): self.env=env; self.calls=0; self.lock=Lock()
    def __getattr__(self,key): return getattr(self.env,key)
    def fitness(self,plan):
        with self.lock: self.calls+=1
        return self.env.fitness(plan)


def build_landscape(protocol,out=None):
    start=perf_counter(); env=make_environment(protocol,'baseline',42).base_env
    bits=enumerate_states(protocol.qubits); plans=encode_plan(bits,protocol)
    truth=np.array([env.fitness(plan)[0] for plan in plans]); enumeration=perf_counter()-start
    model=fit_qubo(bits,truth,protocol.alpha); energies=energy(bits,model)
    metrics=fidelity(truth,-energies)
    digest=sha256(bits.tobytes()+truth.tobytes()+model.Q.tobytes()+np.float64(model.offset).tobytes()).hexdigest()
    metadata={'landscape_hash':digest,'model_config':asdict(env.config),'protocol':asdict(protocol),
              'source_hash':source_digest(),'enumeration_seconds':enumeration,
              'total_seconds':perf_counter()-start,'simulator_evaluations':len(bits),'metrics':metrics}
    if out is not None:
        out=Path(out); out.mkdir(parents=True,exist_ok=True)
        path=out/'landscape.npz'
        if path.exists():
            previous=json.loads((out/'landscape.json').read_text(encoding='utf-8'))
            if previous['landscape_hash']!=digest or previous['source_hash']!=metadata['source_hash']:
                raise ValueError('landscape mismatch; use a new output directory')
        else:
            np.savez_compressed(path,states=bits,truth=truth,energies=energies,Q=model.Q,
                                offset=model.offset,coefficients=model.coefficients)
            write_json(out/'landscape.json',metadata)
    return {'states':bits,'plans':plans,'truth':truth,'model':model,'energies':energies,'metadata':metadata}


def polish(env,plan,iterations=500):
    """Accept only feasible profit improvement, or strict feasibility restoration."""
    low,high=env.bounds(); x=np.clip(np.asarray(plan).reshape(-1),low,high)
    _,before=env.fitness(x); started=perf_counter(); calls=env.calls
    if before['total_penalty']>1e-6:
        restore=minimize(lambda v:env.fitness(v)[1]['total_penalty'],x,method='L-BFGS-B',
                         bounds=list(zip(low,high)),options={'maxiter':iterations,'ftol':1e-12})
        if np.all(np.isfinite(restore.x)):
            _,d=env.fitness(restore.x)
            if d['total_penalty']<before['total_penalty']: x=restore.x
    def objective(v):
        _,d=env.fitness(v)
        return -d['profit_sgd']+100*d['total_penalty']
    result=minimize(objective,x,method='L-BFGS-B',bounds=list(zip(low,high)),
                    options={'maxiter':iterations,'ftol':1e-8})
    candidates=[np.asarray(plan).reshape(-1),x]
    if np.all(np.isfinite(result.x)): candidates.append(result.x)
    assessed=[(v,env.fitness(v)[1]) for v in candidates]
    qualified=[(v,d) for v,d in assessed if d['total_penalty']<=1e-6]
    chosen=max(qualified,key=lambda a:a[1]['profit_sgd']) if qualified else min(assessed,key=lambda a:(a[1]['total_penalty'],-a[1]['profit_sgd']))
    return chosen[0],{'seconds':perf_counter()-started,'evaluations':env.calls-calls,
                     'optimizer_success':bool(result.success),'optimizer_message':str(result.message)}


def initial_population(p,method,seed,env,landscape,out):
    low,high=env.bounds(); bits=landscape['states']; e=landscape['energies']; truth=landscape['truth']
    metadata={}; selected=np.array([],int)
    if method in ('rcga','random'):
        pop=np.random.default_rng(derive_seed(seed,'random')).uniform(low,high,(p.population,len(low)))
    else:
        if method.startswith('exact_sim_top'):
            k=int(method.removeprefix('exact_sim_top'))
            if k>len(bits) or k>p.population: raise ValueError(f'cannot retain {k} distinct centers')
            selected=np.argsort(-truth,kind='stable')[:k]
            metadata['selection']='simulator_score_descending'
        elif method=='sa_qubo_50':
            counts=annealing_visits(e,p.qubits,derive_seed(seed,'sa'),p.sa_replicas,p.sa_sweeps)
            selected=select_candidates(counts,e,p.candidate_k,'energy',require_k=True)
            metadata['selected_counts']=counts[selected]
        else:
            cache=(Path(out)/'samples'/f'qaoa_{seed}.json') if out is not None and p.profile!='principal' else None
            reused=cache is not None and cache.exists() and p.profile!='principal'
            if reused:
                payload=json.loads(cache.read_text(encoding='utf-8'))
                if payload['protocol_hash']!=p.digest() or payload['landscape_hash']!=landscape['metadata']['landscape_hash']:
                    raise ValueError('QAOA cache mismatch; use a new output directory')
                q=payload['sampling']
            else:
                q=sample_qaoa(landscape['model'],p.layers,p.shots,p.qaoa_maxiter,derive_seed(seed,'qaoa'),p.backend,p.restarts)
                if cache is not None:
                    write_json(cache,{'protocol_hash':p.digest(),'landscape_hash':landscape['metadata']['landscape_hash'],'sampling':q})
            counts=np.asarray(q['counts'])
            selected=select_candidates(counts,e,p.candidate_k,p.retention,require_k=method=='qaoa_50')
            metadata.update({'selected_counts':counts[selected],'unique_measured_states':int(np.count_nonzero(counts)),
                             'sampling_reused':reused,'sampling_seconds_recorded':q['total_seconds'],
                             'qaoa_final_expectation':q['final_expectation'],
                             'sampling_file':str(cache.relative_to(out)) if cache is not None else None})
        centers=landscape['plans'][selected]
        if p.profile=='principal':
            # Keep each center once; fill remaining slots with uniform samples.
            pop=np.random.default_rng(derive_seed(seed,'random')).uniform(low,high,(p.population,len(low)))
            pop[:min(len(centers),p.population)]=centers[:p.population]
            allocation=np.ones(len(centers),int)
        else:
            pop,allocation=balanced_expand(centers,p.population,low,high,derive_seed(seed,'jitter'),p.jitter)
        metadata.update({'allocation':allocation,'best_simulator_score':float(truth[selected].max()),
                         'mean_pairwise_hamming':float(pdist(bits[selected],'hamming').mean()*p.qubits) if len(selected)>1 else 0.})
    metadata['selected_state_indices']=selected
    metadata['actual_unique_centers']=len(selected)
    metadata['initial_population']=pop
    return pop,metadata


def run_one(p,method,scenario,seed,out,landscape,data_path=None):
    if method not in SUPPORTED:
        raise ValueError(f'{method} is not implemented as specified in the manuscript; see docs/ALIGNMENT.md')
    if method in ('qaoa_50','sa_qubo_50') and (p.candidate_k!=50 or p.population<50):
        raise ValueError('named K=50 methods require candidate_k=50 and population>=50')
    out=Path(out) if out is not None else None
    path=(out/'runs'/f'{method}__{scenario}__{seed}.json') if out is not None else None
    code_hash=source_digest()
    if path is not None and path.exists():
        prior=json.loads(path.read_text(encoding='utf-8'))
        if prior['provenance']['protocol_hash']!=p.digest() or prior['provenance']['source_hash']!=code_hash:
            raise ValueError('run mismatch; do not mix protocols in one output directory')
        return prior
    env=CountedEnvironment(make_environment(p,scenario,derive_seed(seed,'environment'),data_path=data_path))
    started=perf_counter(); meta={}; history={}; polish_info={}
    if method in ('pso_mpc','mpc_receding','tightened_mpc','es_policy_search','sac_ppo','tube_rmpc'):
        if method=='pso_mpc':
            from algorithms.classical.pso_mpc import run_pso_mpc
            result=run_pso_mpc(env,seed=seed,pred_horizon=6,swarm_size=6 if p.smoke else 40,pso_iters=2 if p.smoke else 80)
        elif method=='mpc_receding':
            from algorithms.classical.mpc_receding import run_mpc_receding
            result=run_mpc_receding(env,seed=seed,pred_horizon=6,max_iter=2 if p.smoke else 300)
        elif method=='tightened_mpc':
            from algorithms.classical.rmpc import run_tightened_mpc
            result=run_tightened_mpc(env,seed=seed,pred_horizon=4,max_iter=2 if p.smoke else 300)
        elif method=='es_policy_search':
            from algorithms.classical.es_policy_search import run_es_policy_search
            result=run_es_policy_search(env,seed=seed,budget=3 if p.smoke else 40000)
        elif method=='sac_ppo':
            from algorithms.classical.sac_ppo import run_sac_ppo
            kwargs={'budget':32,'hidden_dim':16,'warmup_steps':4,'batch_size':4,
                    'rollout_size':4,'ppo_epochs':1} if p.smoke else {'budget':40000}
            result=run_sac_ppo(env,seed=seed,**kwargs)
        else:
            from algorithms.classical.tube_rmpc import run_tube_rmpc
            result=run_tube_rmpc(env,seed=seed,prediction_horizon=2 if p.smoke else 6,
                                 max_iter=3 if p.smoke else 300)
        plan=result.best_x; meta={'identity':result.method,'information_set':'full known scenario trajectory; window optimization, not certified causal control'}
    else:
        population,meta=initial_population(p,method,seed,env,landscape,out)
        initial=[env.fitness(row) for row in population]
        meta['generation_zero_best']=max(f for f,d in initial)
        meta['generation_zero_threshold_qualified']=sum(d['is_feasible'] for f,d in initial)
        cfg=RCGAConfig(pop_size=p.population,n_generations=p.generations,n_elites=p.elites,pc=p.crossover,
              pm=p.mutation,eta_c=p.eta,eta_m=p.eta,tournament_size=p.tournament,patience=p.generations+1,
              relative_tolerance=p.relative_tolerance,relative_patience=p.consecutive_generations,
              fitness_workers=p.fitness_workers,random_seed=derive_seed(seed,'rcga'))
        optimizer=RCGAOptimizer(env,cfg); plan,_,_=optimizer.optimize(population,verbose=False)
        history=optimizer.history
        meta['rcga_fitness_evaluations']=optimizer.n_fitness_evaluations
        if p.polish: plan,polish_info=polish(env,plan,iterations=2 if p.smoke else 500)
    score,details=env.fitness(plan)
    result={'method':method,'scenario':scenario,'seed':seed,'fitness':float(score),'profit':details['profit_sgd'],
            'penalty':details['total_penalty'],'threshold_qualified':details['total_penalty']<=1e-6,
            'plan':np.asarray(plan).reshape(p.hours,4),'details':details,'history':history,'initializer':meta,
            'polishing':polish_info,'total_seconds':perf_counter()-started,'simulator_evaluations':env.calls,
            'provenance':{'protocol':asdict(p),'protocol_hash':p.digest(),'source_hash':code_hash,
              'landscape_hash':landscape['metadata']['landscape_hash'],'python':sys.version,'platform':platform.platform(),
              'result_kind':'corrected_model_smoke' if p.smoke else 'corrected_model_experiment',
              'economic_units':'inherited model accounting scale; physical 25 m2 conversion unverified',
              'timing_scope':'online pipeline; shared offline grid fit excluded; cached sampling may be reused'}}
    if path is not None:
        write_json(path,result)
    return safe(result)


def summarize(out):
    out=Path(out); runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'runs').glob('*.json'))]
    hashes={r['provenance']['protocol_hash'] for r in runs}
    sources={r['provenance']['source_hash'] for r in runs}
    landscapes={r['provenance']['landscape_hash'] for r in runs}
    if len(sources)>1 or len(landscapes)>1: raise ValueError('cannot aggregate different source or landscape versions')
    if len(hashes)>1: raise ValueError('cannot aggregate different protocols')
    identifiers=[(r['method'],r['scenario'],r['seed']) for r in runs]
    if len(identifiers)!=len(set(identifiers)): raise ValueError('duplicate method/scenario/seed runs')
    groups=[]
    for method,scenario in sorted({(r['method'],r['scenario']) for r in runs}):
        rows=[r for r in runs if r['method']==method and r['scenario']==scenario]; values=np.array([r['profit'] for r in rows])
        groups.append({'method':method,'scenario':scenario,'n':len(rows),'mean_profit':float(values.mean()),
                       'sd_profit':float(values.std(ddof=1)) if len(values)>1 else None,
                       'threshold_qualified':sum(r['threshold_qualified'] for r in rows),
                       'sampling_reused_runs':sum(r['initializer'].get('sampling_reused',False) for r in rows),
                       'mean_seconds':float(np.mean([r['total_seconds'] for r in rows]))})
    pairs=[]
    for qmethod in ('qaoa_rcga','qaoa_50'):
        q={(r['scenario'],r['seed']):r['profit'] for r in runs if r['method']==qmethod}
        for method in sorted({r['method'] for r in runs}-{qmethod}):
            b={(r['scenario'],r['seed']):r['profit'] for r in runs if r['method']==method}; keys=sorted(q.keys() & b.keys())
            if len(keys)<2: continue
            x=np.array([q[k] for k in keys]); y=np.array([b[k] for k in keys]); delta=x-y
            test=ttest_rel(x,y)
            pairs.append({'reference':qmethod,'comparator':method,'n_pairs':len(keys),
                          'paired_mean_difference':float(delta.mean()),'p':float(test.pvalue),
                          'scope':'fixed scenarios, paired seed index; no population-level disturbance inference'})
    for reference in ('qaoa_rcga','qaoa_50'):
        family=[r for r in pairs if r['reference']==reference and np.isfinite(r['p'])]
        ordered=sorted(family,key=lambda r:r['p']); previous=0.
        for i,row in enumerate(ordered):
            previous=max(previous,min(1.,(len(ordered)-i)*row['p'])); row['holm_p']=previous
    report={'groups':groups,'paired_comparisons':pairs,'protocol_hash':next(iter(hashes),None),
            'source_hash':next(iter(sources),None),'landscape_hash':next(iter(landscapes),None),
            'timing_definition':'actual online execution, excluding shared offline fit; sampling_reused_runs marks cache hits; cached and uncached timing must not be compared as full pipelines'}
    write_json(out/'summary.json',report)
    if groups:
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    return report
