import numpy as np
import pytest
from paper_protocol.surrogate import enumerate_states, fit_qubo, qubo_to_ising, energy, encode_plan
from paper_protocol.sampling import select_candidates, balanced_expand, probabilities, sample_qaoa
from paper_protocol.protocol import Protocol, make_environment


def test_ridge_unregularized_intercept_energy_and_ising_identity():
    bits = enumerate_states(3)
    y = 7 + 2*bits[:, 0] - 3*bits[:, 1] + 4*bits[:, 0]*bits[:, 2]
    model = fit_qubo(bits, y, alpha=1.)
    X = np.column_stack([np.ones(8), bits, bits[:,0]*bits[:,1], bits[:,0]*bits[:,2], bits[:,1]*bits[:,2]])
    R = np.diag([0, 1, 1, 1, 1, 1, 1])
    expected = np.linalg.solve(X.T@X+R, X.T@y)
    np.testing.assert_allclose(energy(bits, model), -(X@expected), atol=1e-12)
    c, h, j = qubo_to_ising(model)
    spin = 1 - 2*bits
    np.testing.assert_allclose(c + spin@h + np.einsum('bi,ij,bj->b',spin,j,spin), energy(bits,model))


def test_decode_extremes_and_zero_slew():
    p = Protocol()
    bits = enumerate_states(12)
    plans = encode_plan(bits, p)
    assert plans.shape == (4096, 96)
    np.testing.assert_equal(plans[0].reshape(24,4)[0], [15,0,300,30])
    np.testing.assert_equal(plans[-1].reshape(24,4)[0], [38,800,1800,95])
    assert np.all(np.diff(plans.reshape(-1,24,4),axis=1) == 0)


def test_frequency_selection_does_not_use_energy_as_primary_order():
    counts = np.array([5, 5, 9, 0, 5])
    e = np.array([1., -2., 100., -100., -2.])
    np.testing.assert_equal(select_candidates(counts,e,4,'frequency'),[2,1,4,0])
    np.testing.assert_equal(select_candidates(counts,e,2,'energy'),[1,4])
    with pytest.raises(ValueError): select_candidates(counts,e,5,'frequency',require_k=True)


def test_balanced_jitter_preserves_first_centers_and_allocation():
    centers = np.repeat(np.linspace(.2,.8,50)[:,None],96,axis=1)
    pop, allocation = balanced_expand(centers,185,np.zeros(96),np.ones(96),42,.01)
    np.testing.assert_equal(allocation,np.r_[np.full(35,4),np.full(15,3)])
    np.testing.assert_equal(pop[:50],centers)
    assert np.any(np.diff(pop[50:].reshape(-1,24,4),axis=1) != 0)
    np.testing.assert_equal(pop,balanced_expand(centers,185,np.zeros(96),np.ones(96),42,.01)[0])


def test_aer_matches_numpy_exact_statevector():
    model = fit_qubo(enumerate_states(3),np.arange(8)**2,1.)
    params = np.array([.19,.32,.17,.21])
    a = probabilities(model,params,'numpy')
    b = probabilities(model,params,'aer')
    np.testing.assert_allclose(a,b,atol=2e-12)
    assert a.sum() == pytest.approx(1.)


def test_finite_shots_only_affect_final_counts():
    model = fit_qubo(enumerate_states(2),np.array([0.,1.,2.,-1.]),1.)
    a = sample_qaoa(model,p=1,shots=32,maxiter=10,seed=42,backend='numpy')
    b = sample_qaoa(model,p=1,shots=64,maxiter=10,seed=42,backend='numpy')
    np.testing.assert_equal(a['parameters'],b['parameters'])
    assert sum(a['counts']) == 32
    assert sum(b['counts']) == 64
    assert a['final_expectation'] <= a['initial_expectation'] + 1e-10


def test_scenario_profit_and_score_have_single_penalty():
    p = Protocol()
    plan = np.tile([25.,100.,600.,70.],(24,1))
    for scenario in p.scenarios:
        env = make_environment(p,scenario,42)
        score,d = env.fitness(plan)
        assert score == pytest.approx(d['profit_sgd']-d['total_penalty'])
        assert d['profit_sgd'] == pytest.approx(d['revenue_sgd']-d['total_energy']
            -d['total_water']*env.base_env.config.water_cost_per_m3*.001
            -env.base_env.config.fixed_cost_per_day)
        assert np.sum(d['yield_hourly']) == pytest.approx(d['total_yield'])
        assert np.sum(d['energy_hourly']) == pytest.approx(d['total_energy'])
