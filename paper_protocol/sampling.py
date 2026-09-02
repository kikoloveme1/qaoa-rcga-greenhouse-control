"""Cost/mixer circuits, finite-shot selection, and balanced population lifting."""
from time import perf_counter
import os
import numpy as np
from scipy.optimize import minimize
from .surrogate import energy, enumerate_states, qubo_to_ising


def probabilities(model,params,backend='aer'):
    n = len(model.Q); dim = 1<<n; params = np.asarray(params,float); p = len(params)//2
    if len(params) != 2*p or p < 1 or not np.all(np.isfinite(params)): raise ValueError('invalid angles')
    if backend == 'numpy':
        e = energy(enumerate_states(n),model)
        psi = np.full(dim,1/np.sqrt(dim),complex)
        for g,b in zip(params[:p],params[p:]):
            psi *= np.exp(-1j*g*e)
            for q in range(n):
                view = psi.reshape(-1,2,1<<q); a=view[:,0,:].copy(); c=view[:,1,:].copy()
                view[:,0,:]=np.cos(b)*a-1j*np.sin(b)*c
                view[:,1,:]=np.cos(b)*c-1j*np.sin(b)*a
        result = np.abs(psi)**2
    elif backend == 'aer':
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        _,h,j = qubo_to_ising(model)
        circuit = QuantumCircuit(n); circuit.h(range(n))
        for g,b in zip(params[:p],params[p:]):
            for i in range(n): circuit.rz(2*g*h[i],i)
            for i in range(n):
                for k in range(i+1,n):
                    if j[i,k] != 0: circuit.rzz(2*g*j[i,k],i,k)
            for i in range(n): circuit.rx(2*b,i)
        circuit.save_statevector()
        aer_threads=max(1,int(os.environ.get('QAOA_AER_THREADS','1')))
        sim = AerSimulator(method='statevector',max_parallel_threads=aer_threads)
        state = sim.run(circuit).result().get_statevector()
        result = np.abs(np.asarray(state))**2
    else: raise ValueError('backend must be aer or numpy')
    return result/result.sum()


def sample_qaoa(model,p=3,shots=32768,maxiter=100,seed=42,backend='aer',restarts=1):
    if min(p,shots,maxiter,restarts)<1: raise ValueError('positive sampling budgets required')
    start=perf_counter(); rng=np.random.default_rng(seed); e=energy(enumerate_states(len(model.Q)),model)
    records=[]; best=None
    for restart in range(restarts):
        x0=rng.uniform(0,2*np.pi,2*p); history=[]; best_value=np.inf; best_params=None
        def objective(x):
            nonlocal best_value,best_params
            value=float(probabilities(model,x,backend)@e); history.append(value)
            if value<best_value: best_value=value; best_params=np.array(x,copy=True)
            return value
        initial=objective(x0)
        result=minimize(objective,x0,method='COBYLA',options={'maxiter':maxiter,'rhobeg':.5})
        # Budget exhaustion is not grounds to discard all optimized parameters.
        record={'restart':restart,'initial_expectation':initial,'final_expectation':best_value,
                'parameters':best_params,'history':history,'nfev':len(history),
                'optimizer_success':bool(result.success),'optimizer_message':str(result.message)}
        records.append(record)
        if best is None or best_value<best['final_expectation']: best=record
    optimization=perf_counter()-start
    probs=probabilities(model,best['parameters'],backend)
    draws=rng.choice(len(e),size=shots,p=probs); counts=np.bincount(draws,minlength=len(e))
    return {**best,'counts':counts,'probabilities':probs,'restarts':records,'backend':backend,
            'optimization_seconds':optimization,'total_seconds':perf_counter()-start}


def select_candidates(counts,energies,k,rule='frequency',require_k=False):
    c=np.asarray(counts); e=np.asarray(energies); idx=np.arange(len(c))
    if c.shape != e.shape or c.ndim != 1 or k<1 or np.any(c<0): raise ValueError('invalid counts')
    if rule=='frequency': order=np.lexsort((idx,e,-c))
    elif rule=='energy': order=np.lexsort((idx,-c,e))
    else: raise ValueError('unknown selection rule')
    selected=order[c[order]>0][:k]
    if require_k and len(selected)!=k: raise ValueError(f'only {len(selected)} distinct states, require {k}')
    return selected


def balanced_expand(centers,size,low,high,seed,jitter=.01):
    centers=np.atleast_2d(np.asarray(centers,float))
    if not 1<=len(centers)<=size: raise ValueError('centers must fit in population')
    order=np.arange(size)%len(centers); population=centers[order].copy(); duplicate=np.arange(size)>=len(centers)
    rng=np.random.default_rng(seed)
    population[duplicate]+=rng.normal(0,jitter*(np.asarray(high)-low),(int(duplicate.sum()),centers.shape[1]))
    return np.clip(population,low,high),np.bincount(order,minlength=len(centers))


def annealing_visits(energies,n,seed,replicas=4096,sweeps=400):
    rng=np.random.default_rng(seed); indices=rng.integers(0,len(energies),replicas)
    visits=np.bincount(indices,minlength=len(energies)); e=np.asarray(energies)
    for temp in np.geomspace(2.,1e-3,sweeps):
        candidates=indices^(1<<rng.integers(0,n,replicas)); delta=e[candidates]-e[indices]
        accepted=(delta<=0)|(rng.random(replicas)<np.exp(-np.maximum(delta,0)/temp))
        indices[accepted]=candidates[accepted]; visits+=np.bincount(indices,minlength=len(e))
    return visits
