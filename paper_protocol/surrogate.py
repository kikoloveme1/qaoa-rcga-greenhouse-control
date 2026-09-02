"""Full-grid Ridge fit; upper triangular QUBO energy is negative simulator score."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


@dataclass
class QUBO:
    Q: np.ndarray
    offset: float
    coefficients: np.ndarray
    alpha: float


def enumerate_states(n):
    if not 1 <= n <= 16: raise ValueError('supported register: 1..16 bits')
    return ((np.arange(1 << n,dtype=np.int64)[:,None] >> np.arange(n)) & 1).astype(np.int8)


def features(states):
    z = np.asarray(states,float)
    if z.ndim != 2 or not np.all((z == 0) | (z == 1)): raise ValueError('binary row matrix required')
    pairs = [(i,j) for i in range(z.shape[1]) for j in range(i+1,z.shape[1])]
    return np.column_stack([np.ones(len(z)),z,*[z[:,i]*z[:,j] for i,j in pairs]]),pairs


def fit_qubo(states,targets,alpha=1.):
    X,pairs = features(states)
    y = np.asarray(targets,float)
    if y.shape != (len(X),) or not np.all(np.isfinite(y)) or alpha < 0:
        raise ValueError('finite targets and nonnegative ridge alpha required')
    R = np.eye(X.shape[1]); R[0,0] = 0
    beta = np.linalg.solve(X.T@X+alpha*R,X.T@y) if alpha else np.linalg.lstsq(X,y,rcond=None)[0]
    n = np.asarray(states).shape[1]; Q = np.zeros((n,n)); Q[np.diag_indices(n)] = -beta[1:1+n]
    for value,(i,j) in zip(beta[1+n:],pairs): Q[i,j] = -value
    return QUBO(Q,-float(beta[0]),beta,float(alpha))


def energy(states,model):
    z = np.asarray(states)
    return model.offset + np.einsum('...i,ij,...j->...',z,model.Q,z)


def qubo_to_ising(model):
    Q = model.Q
    j = np.triu(Q,1)/4
    h = -np.diag(Q)/2 - (np.sum(j,axis=0)+np.sum(j,axis=1))
    c = model.offset + np.trace(Q)/2 + np.sum(j)
    return float(c),h,j


def encode_plan(states,protocol):
    z = np.atleast_2d(np.asarray(states,dtype=int)); p = protocol
    if z.shape[1] != p.qubits or not np.all((z == 0) | (z == 1)):
        raise ValueError('invalid encoded controls')
    levels = z.reshape(-1,p.blocks,4,p.bits_per_variable) @ (1 << np.arange(p.bits_per_variable))
    controls = np.asarray(p.lower) + levels / ((1<<p.bits_per_variable)-1) * (np.asarray(p.upper)-p.lower)
    return np.repeat(controls,p.hours//p.blocks,axis=1).reshape(len(z),p.hours*4)


def fidelity(truth,prediction):
    y = np.asarray(truth,float); yhat = np.asarray(prediction,float)
    true_order = np.argsort(-y,kind='stable'); fitted_order = np.argsort(-yhat,kind='stable')
    ranks = np.empty(len(y),int); ranks[true_order] = np.arange(1,len(y)+1)
    report = {'pearson':float(pearsonr(y,yhat).statistic),'spearman':float(spearmanr(y,yhat).statistic),
              'rmse':float(np.sqrt(np.mean((y-yhat)**2))), 'feature_fit':'in-sample, all enumerated states', 'tail':{}}
    report['normalized_rmse'] = report['rmse']/max(float(np.ptp(y)),1e-12)
    for k in (1,5,10,25,50,100,185):
        if k > len(y): continue
        a,b = true_order[:k],fitted_order[:k]; union = np.union1d(a,b)
        tau = kendalltau(y[union],yhat[union]).statistic if len(union)>1 else None
        report['tail'][str(k)] = {'overlap':len(np.intersect1d(a,b)),
            'mean_simulator_rank':float(np.mean(ranks[b])), 'best_regret':float(y.max()-y[b].max()),
            'kendall_union':float(tau) if tau is not None and np.isfinite(tau) else None}
    return report
