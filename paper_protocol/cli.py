"""Command-line entry points; full runs are explicit, smoke runs labelled."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
from .protocol import Protocol
from .runner import build_landscape, run_one, summarize, write_json, SUPPORTED


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['landscape','principal','matched','seed-ablation','depth','shots','encoding','summarize'])
    parser.add_argument('--out',type=Path,help='optional directory for generated results')
    parser.add_argument('--data',type=Path,help='reader-supplied hourly Singapore greenhouse CSV')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--methods',help='comma-separated identifiers; unsupported paper labels fail explicitly')
    parser.add_argument('--scenarios',help='comma-separated scenarios')
    parser.add_argument('--seeds',help='comma-separated integers; default is all ten manuscript seeds')
    parser.add_argument('--smoke',action='store_true',help='reduced budgets; not a paper replication')
    parser.add_argument('--backend',choices=['aer','numpy'])
    parser.add_argument('--fitness-workers',type=int)
    args=parser.parse_args(argv)
    if args.study=='summarize' and args.out is None:
        parser.error('summarize requires --out')
    if args.study=='summarize':
        print(json.dumps(summarize(args.out),indent=2)); return
    p=Protocol()
    methods=['qaoa_rcga','rcga','pso_mpc','tube_rmpc','mpc_receding','sac_ppo']
    if args.study in ('matched','seed-ablation'):
        p=replace(p,profile=args.study,candidate_k=50,polish=False,relative_tolerance=None)
        if args.study=='matched':
            p=replace(p,scenarios=('baseline','rolling','stochastic'))
            methods=['random','exact_sim_top1','exact_sim_top5','exact_sim_top10','exact_sim_top50','sa_qubo_50','qaoa_50']
        else: methods=['random','exact_sim_top1','sa_qubo_50','qaoa_50']
    elif args.study in ('depth','shots','encoding'):
        p=replace(p,profile=args.study,population=100,candidate_k=100,eta=20.,retention='energy',restarts=4,
                  polish=False,relative_tolerance=None,scenarios=('baseline',))
        methods=['qaoa_rcga']
        if args.study=='encoding': p=replace(p,generations=150,seeds=(42,142,242))
    if args.config:
        raw=json.loads(args.config.read_text(encoding='utf-8'))
        p=replace(p,**raw)
    if args.backend: p=replace(p,backend=args.backend)
    if args.scenarios: p=replace(p,scenarios=tuple(args.scenarios.split(',')))
    # Base landscape remains nominal even when the selected study excludes baseline.
    if args.seeds: p=replace(p,seeds=tuple(map(int,args.seeds.split(','))))
    if args.fitness_workers is not None: p=replace(p,fitness_workers=args.fitness_workers)
    if args.methods: methods=args.methods.split(',')
    if args.smoke:
        p=replace(p,smoke=True,population=60,generations=3,candidate_k=min(p.candidate_k,50),qaoa_maxiter=12,
                  shots=1024,restarts=1,sa_replicas=64,sa_sweeps=5,fitness_workers=1,seeds=p.seeds if args.seeds else (42,))
    invalid=set(methods)-set(SUPPORTED)
    if invalid: parser.error(f'not implemented as specified: {sorted(invalid)}; see docs/ALIGNMENT.md')
    variants=[('main',p)]
    if args.study=='depth': variants=[(f'p{v}',replace(p,layers=v)) for v in (1,2,3,4,5)]
    if args.study=='shots': variants=[(f'shots{v}',replace(p,shots=v)) for v in (1024,2048,4096,8192,16384,32768)]
    if args.study=='encoding': variants=[(f'blocks{b}_bits{n}',replace(p,blocks=b,bits_per_variable=n)) for b,n in ((1,2),(1,3),(1,4),(2,2))]
    for name,protocol in variants:
        out=args.out if len(variants)==1 else (args.out/name if args.out is not None else None)
        if out is not None:
            out.mkdir(parents=True,exist_ok=True)
            manifest=out/'protocol.json'
            if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=json.loads(json.dumps(asdict(protocol))):
                parser.error(f'protocol mismatch in {out}; choose a fresh output directory')
            write_json(manifest,asdict(protocol))
        print(f'[{name}] enumerating {1<<protocol.qubits} states; profile={protocol.profile}; smoke={protocol.smoke}',flush=True)
        landscape=build_landscape(protocol,out)
        if args.study=='landscape':
            print(json.dumps(landscape['metadata']['metrics'],indent=2)); continue
        for scenario in protocol.scenarios:
            for seed in protocol.seeds:
                for method in methods:
                    result=run_one(protocol,method,scenario,seed,out,landscape,data_path=args.data)
                    print(f'{method}/{scenario}/{seed}: profit={result["profit"]:.6f}, penalty={result["penalty"]:.3g}, time={result["total_seconds"]:.2f}s',flush=True)
        if out is not None:
            summarize(out)


if __name__=='__main__': main()
