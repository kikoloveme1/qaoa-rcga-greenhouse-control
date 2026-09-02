import numpy as np
from algorithms.classical.rcga import RCGAConfig, RCGAOptimizer
from environment.greenhouse_model import GreenhouseEnv


def test_relative_stopping_does_not_count_generation_zero():
    env=GreenhouseEnv()
    cfg=RCGAConfig(pop_size=8,n_elites=1,n_generations=20,pc=0,pm=0,
                   relative_tolerance=5e-4,relative_patience=5,patience=100)
    pop=np.tile(np.tile([25.,100.,600.,70.],24),(8,1))
    opt=RCGAOptimizer(env,cfg)
    opt.optimize(seed_population=pop,verbose=False)
    assert len(opt.history['best_fitness'])==6  # gen0 + five unchanged transitions
    assert opt.n_fitness_evaluations == 8*6+1


def test_uniform_initialization_has_independent_hourly_variables():
    env=GreenhouseEnv()
    opt=RCGAOptimizer(env,RCGAConfig(pop_size=20,random_seed=123))
    expected=np.random.default_rng(123).uniform(*env.bounds(),size=(20,96))
    np.testing.assert_equal(opt._initialize_population(),expected)


def test_generation_cap_includes_terminal_evaluation_in_history():
    opt=RCGAOptimizer(GreenhouseEnv(),RCGAConfig(pop_size=8,n_elites=1,n_generations=2,patience=99))
    _,best,_=opt.optimize(verbose=False)
    assert opt.history['generation']==[0,1,2]
    assert opt.history['best_fitness'][-1]==best
