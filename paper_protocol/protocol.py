"""Explicit manuscript settings and deterministic scenario construction."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import numpy as np
from environment.greenhouse_model import GreenhouseConfig
from environment.greenhouse_perturbation import PerturbationConfig, PerturbationEnv, SCENARIO_LIBRARY


@dataclass(frozen=True)
class Protocol:
    profile: str = 'principal'
    scenarios: tuple = ('baseline','heat_wave','cold_snap','co2_failure','compound_crisis','rolling','stochastic')
    seeds: tuple = (42,142,242,342,442,542,642,742,842,942)
    lower: tuple = (15.,0.,300.,30.)
    upper: tuple = (38.,800.,1800.,95.)
    slew: tuple = (6.,400.,500.,20.)
    hours: int = 24
    blocks: int = 1
    bits_per_variable: int = 3
    alpha: float = 1.
    layers: int = 3
    shots: int = 32768
    qaoa_maxiter: int = 100
    restarts: int = 1
    backend: str = 'aer'
    candidate_k: int = 185
    retention: str = 'frequency'
    population: int = 185
    generations: int = 200
    eta: float = 5.
    tournament: int = 5
    elites: int = 5
    crossover: float = .9
    mutation: float = 1.
    jitter: float = .01
    polish: bool = True
    relative_tolerance: float | None = 5e-4
    consecutive_generations: int = 5
    penalty_weight: float = 500.
    humidity_tracking: float = .5
    fitness_workers: int = 16
    sa_replicas: int = 4096
    sa_sweeps: int = 400
    smoke: bool = False

    def __post_init__(self):
        if self.blocks < 1 or self.hours % self.blocks or self.bits_per_variable < 1:
            raise ValueError('blocks must divide hours; bits must be positive')
        if self.qubits > 16:
            raise ValueError('exhaustive protocol limited to <=16 qubits')
        if min(self.population,self.generations,self.layers,self.shots,self.restarts,self.qaoa_maxiter,self.candidate_k,self.fitness_workers) < 1:
            raise ValueError('budgets must be positive')
        if self.elites >= self.population or not 0 <= self.humidity_tracking <= 1:
            raise ValueError('invalid elites or humidity_tracking')
        if self.backend not in ('aer','numpy') or self.retention not in ('frequency','energy'):
            raise ValueError('unknown backend or candidate retention')

    @property
    def qubits(self): return self.blocks * 4 * self.bits_per_variable

    def digest(self):
        return sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def derive_seed(seed, stream):
    return int.from_bytes(sha256(f'{seed}:{stream}'.encode()).digest()[:4],'little')


def make_environment(protocol, scenario, seed, data_path=None):
    if scenario not in Protocol().scenarios:
        raise ValueError(f'unknown scenario: {scenario}')
    p = protocol
    cfg = GreenhouseConfig(T_steps=p.hours,T_lower=p.lower[0],T_upper=p.upper[0],
        L_upper=p.upper[1],C_lower=p.lower[2],C_upper=p.upper[2],H_lower=p.lower[3],H_upper=p.upper[3],
        max_dT=p.slew[0],max_dL=p.slew[1],max_dC=p.slew[2],max_dH=p.slew[3],
        lambda_bound=p.penalty_weight,lambda_rate=p.penalty_weight,humidity_tracking=p.humidity_tracking)
    env = PerturbationEnv(PerturbationConfig(base_config=cfg,seed=seed),[SCENARIO_LIBRARY[scenario]()])
    if data_path is not None:
        from environment.singapore_data import apply_singapore_data, load_singapore_csv
        apply_singapore_data(env, load_singapore_csv(data_path, horizon=p.hours))
    return env
