# -*- coding: utf-8 -*-
"""Real-coded genetic optimization with SBX and polynomial mutation.

Supports supplied initialization populations, tournament selection and elitism.
References: Deb & Agrawal (1995); Deb & Goyal (1996)."""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
from time import perf_counter
import sys, os as _os

# Inject parent into path for greenhouse_env import
_sys_dir = _os.path.join(_os.path.dirname(__file__), "..")
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

try:
    from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig
except ImportError:
    # Fallback for direct execution
    from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig

# RCGA Configuration


@dataclass
class RCGAConfig:
    """Hyperparameters for the Real-Coded Genetic Algorithm."""

    # Population
    pop_size: int = 100          # Population size
    n_generations: int = 200     # Max generations
    seed_size: int = 0           # Number of externally-seeded individuals

    # SBX crossover
    pc: float = 0.9              # Crossover probability
    eta_c: float = 20.0          # SBX distribution index (higher = children closer to parents)

    # Polynomial mutation
    pm: float = 1.0              # Mutation probability (per-variable)
    eta_m: float = 20.0          # Mutation distribution index (higher = smaller mutations)

    # Selection & elitism
    tournament_size: int = 3     # Tournament selection group size
    n_elites: int = 5            # Number of elites preserved each generation

    # Termination
    patience: int = 30           # Early stop if no improvement for N generations
    tol: float = 1e-6            # Improvement threshold for patience
    relative_tolerance: Optional[float] = None
    relative_patience: int = 5
    fitness_workers: int = 1

    # Diversity
    min_diversity: float = 1e-8  # Minimum population std to avoid premature convergence

    # Seed
    random_seed: int = 42

# Genetic operators


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    eta_c: float = 20.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for real-valued vectors.

    Creates two children that lie between parents, with spread controlled
    by eta_c. Preserves the property: if parents are feasible, children
    are bounded.

    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Parent solution vectors.
    bounds_low, bounds_high : np.ndarray
        Variable bounds.
    eta_c : float
        Distribution index. Larger = children closer to parents.
    rng : np.random.Generator

    Returns
    -------
    child1, child2 : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(n):
        if rng.random() > 0.5:
            continue  # This variable stays with parent

        if np.abs(parent1[i] - parent2[i]) < 1e-14:
            continue  # Parents identical at this locus

        # Ensure parent1[i] <= parent2[i] for SBX formula
        y1 = min(parent1[i], parent2[i])
        y2 = max(parent1[i], parent2[i])

        xl = bounds_low[i]
        xu = bounds_high[i]

        # Calculate spread factor beta
        u = rng.random()
        if u <= 0.5:
            beta_q = (2.0 * u) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

        # Children
        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

        # Boundary repair: clip and reflect
        c1 = max(xl, min(xu, c1))
        c2 = max(xl, min(xu, c2))

        # Restore original parent ordering
        if parent1[i] <= parent2[i]:
            child1[i] = c1
            child2[i] = c2
        else:
            child2[i] = c1
            child1[i] = c2

    return child1, child2


def polynomial_mutation(
    individual: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    pm: float = 1.0,
    eta_m: float = 20.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Polynomial Mutation for real-valued vectors.

    Applies per-variable mutation with probability pm/n_vars.
    Mutation strength controlled by eta_m.

    Parameters
    ----------
    individual : np.ndarray
        Solution to mutate.
    bounds_low, bounds_high : np.ndarray
        Variable bounds.
    pm : float
        Mutation probability per variable = pm / n_vars.
    eta_m : float
        Distribution index. Larger = smaller mutations.
    rng : np.random.Generator

    Returns
    -------
    mutant : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(individual)
    mutant = individual.copy()
    per_var_prob = pm / n

    for i in range(n):
        if rng.random() > per_var_prob:
            continue

        xl = bounds_low[i]
        xu = bounds_high[i]
        y = individual[i]

        delta = rng.random()
        if delta < 0.5:
            delta_q = (2.0 * delta) ** (1.0 / (eta_m + 1.0)) - 1.0
            mutant[i] = y + delta_q * (y - xl)
        else:
            delta_q = 1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta_m + 1.0))
            mutant[i] = y + delta_q * (xu - y)

        # Clamp to bounds
        mutant[i] = max(xl, min(xu, mutant[i]))

    return mutant


def tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int = 3,
    rng: np.random.Generator = None,
) -> int:
    """Tournament selection: pick best of `tournament_size` random individuals.

    Returns index of winner.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size = len(population)
    candidates = rng.choice(pop_size, size=tournament_size, replace=False)
    best_idx = candidates[np.argmax(fitness[candidates])]
    return best_idx

# RCGA Optimiser


class RCGAOptimizer:
    """Real-Coded Genetic Algorithm for greenhouse environment control.

    Parameters
    ----------
    env : GreenhouseEnv
        The greenhouse environment simulator.
    config : RCGAConfig
        Algorithm hyperparameters.
    """

    def __init__(
        self,
        env: GreenhouseEnv,
        config: Optional[RCGAConfig] = None,
    ):
        self.env = env
        self.config = config or RCGAConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        # Bounds per variable (96-D flattened vector)
        self.bounds_low, self.bounds_high = env.bounds()
        self.n_vars = env.n_vars

        # History
        self.history: Dict[str, List[float]] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "best_yield": [],
            "best_energy": [],
            "best_penalty": [],
            "population_std": [],
        }
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.best_details: Dict = {}
        self.elapsed_time: float = 0.0
        self.n_fitness_evaluations: int = 0

    def _evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate fitness for entire population."""
        pop_size = len(pop)
        fitness = np.zeros(pop_size)
        yields = np.zeros(pop_size)
        energies = np.zeros(pop_size)
        penalties = np.zeros(pop_size)

        if self.config.fitness_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.fitness_workers) as pool:
                evaluations = list(pool.map(self.env.fitness, pop))
        else:
            evaluations = [self.env.fitness(row) for row in pop]
        self.n_fitness_evaluations += len(pop)
        for i, (f, d) in enumerate(evaluations):
            fitness[i] = f
            yields[i] = d["total_yield"]
            energies[i] = d["total_energy"]
            penalties[i] = d["total_penalty"]

        return fitness, yields, energies, penalties

    def _initialize_population(self, seed_population: Optional[np.ndarray] = None) -> np.ndarray:
        """Create initial population, optionally seeded with external solutions."""
        cfg = self.config
        pop = np.zeros((cfg.pop_size, self.n_vars))

        n_seeded = 0
        if seed_population is not None and len(seed_population) > 0:
            n_seeded = min(len(seed_population), cfg.pop_size)
            # Flatten and clip seeded solutions
            for i in range(n_seeded):
                seed = np.asarray(seed_population[i], dtype=float).flatten()
                if len(seed) == self.n_vars:
                    pop[i] = np.clip(seed, self.bounds_low, self.bounds_high)

        # Fill remaining with random solutions
        pop[n_seeded:] = self.rng.uniform(self.bounds_low, self.bounds_high,
                                         size=(cfg.pop_size-n_seeded, self.n_vars))

        return pop

    def optimize(
        self,
        seed_population: Optional[np.ndarray] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None,
        observer: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run RCGA optimisation.

        Parameters
        ----------
        seed_population : np.ndarray, optional
            Externally-seeded individuals (e.g., from QAOA decoding).
            Shape (k, 96) or (k, 24, 4).
        verbose : bool
            Print progress.
        callback : callable, optional
            Called each generation with (gen, best_fitness, best_solution).

        Returns
        -------
        best_solution : np.ndarray, shape (96,)
        best_fitness : float
        details : dict
        """
        cfg = self.config
        t0 = perf_counter()

        # Initialise
        pop = self._initialize_population(seed_population)
        fitness, yields, energies, penalties = self._evaluate_population(pop)
        evaluation_count = len(pop)
        evaluated_offspring = []

        # Track best
        best_idx = np.argmax(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = pop[best_idx].copy()

        # Elitism: track elite indices
        elite_indices = set()

        patience_counter = 0
        prev_best = self.best_fitness

        for gen in range(cfg.n_generations):
            if observer is not None:
                observer({
                    "generation": gen,
                    "evaluation_count": evaluation_count,
                    "best_solution": self.best_solution,
                    "population": pop,
                    "fitness": fitness,
                    "offspring": tuple(evaluated_offspring),
                })
            # --- Record history ---
            history_evaluation_count = self.n_fitness_evaluations
            self.history["generation"].append(gen)
            self.history["best_fitness"].append(self.best_fitness)
            self.history["mean_fitness"].append(float(np.mean(fitness)))
            self.history["best_yield"].append(float(yields[best_idx]))
            self.history["best_energy"].append(float(energies[best_idx]))
            self.history["best_penalty"].append(float(penalties[best_idx]))
            self.history["population_std"].append(float(np.std(fitness)))

            if verbose and gen % 20 == 0:
                print(
                    f"  Gen {gen:4d} | Best Fit={self.best_fitness:10.2f} | "
                    f"Yield={yields[best_idx]:6.3f} | Energy={energies[best_idx]:7.2f} | "
                    f"Penalty={penalties[best_idx]:10.2f} | PopStd={np.std(fitness):.2f}"
                )

            if callback is not None:
                callback(gen, self.best_fitness, self.best_solution)

            # --- Early stopping ---
            if cfg.relative_tolerance is not None:
                relative_change = abs(self.best_fitness-prev_best)/max(abs(prev_best),1e-12)
                patience_counter = patience_counter + 1 if gen > 0 and relative_change <= cfg.relative_tolerance else 0
            elif self.best_fitness - prev_best < cfg.tol:
                patience_counter += 1
            else:
                patience_counter = 0
            prev_best = self.best_fitness

            stopping_patience = cfg.relative_patience if cfg.relative_tolerance is not None else cfg.patience
            if patience_counter >= stopping_patience:
                if verbose:
                    print(f"  Early stop at generation {gen}: no improvement for {cfg.patience} gens")
                break

            # --- Create next generation ---
            new_pop = np.zeros_like(pop)
            offspring_provenance = []

            # Elitism: preserve top individuals
            elite_idx = np.argsort(fitness)[-cfg.n_elites:]
            for j, idx in enumerate(elite_idx):
                new_pop[j] = pop[idx].copy()

            # Fill rest via selection, crossover, mutation
            for j in range(cfg.n_elites, cfg.pop_size, 2):
                # Tournament selection
                p1_idx = tournament_select(pop, fitness, cfg.tournament_size, self.rng)
                p2_idx = tournament_select(pop, fitness, cfg.tournament_size, self.rng)

                p1 = pop[p1_idx].copy()
                p2 = pop[p2_idx].copy()

                # SBX crossover
                if self.rng.random() < cfg.pc:
                    c1, c2 = sbx_crossover(
                        p1, p2, self.bounds_low, self.bounds_high,
                        cfg.eta_c, self.rng,
                    )
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Polynomial mutation
                c1 = polynomial_mutation(
                    c1, self.bounds_low, self.bounds_high,
                    cfg.pm, cfg.eta_m, self.rng,
                )
                c2 = polynomial_mutation(
                    c2, self.bounds_low, self.bounds_high,
                    cfg.pm, cfg.eta_m, self.rng,
                )

                new_pop[j] = c1
                offspring_provenance.append((j, p1.copy(), float(fitness[p1_idx])))
                if j + 1 < cfg.pop_size:
                    new_pop[j + 1] = c2
                    offspring_provenance.append((j + 1, p2.copy(), float(fitness[p2_idx])))

            # Replace population
            pop = new_pop
            fitness, yields, energies, penalties = self._evaluate_population(pop)
            evaluation_count += len(pop)
            evaluated_offspring = [
                {
                    "parent": parent,
                    "child": pop[index],
                    "parent_fitness": parent_fitness,
                    "child_fitness": float(fitness[index]),
                    "successful": bool(fitness[index] > parent_fitness),
                }
                for index, parent, parent_fitness in offspring_provenance
            ]
            best_idx = np.argmax(fitness)

            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = pop[best_idx].copy()

        self.elapsed_time = perf_counter() - t0

        if observer is not None:
            observer({
                "generation": gen + 1,
                "evaluation_count": evaluation_count,
                "best_solution": self.best_solution,
                "population": pop,
                "fitness": fitness,
                "offspring": tuple(evaluated_offspring),
                "final_post_evaluation": True,
            })

        # At the budget cap the final offspring evaluation occurs after the
        # last loop-entry record. Preserve this terminal generation as well.
        if self.n_fitness_evaluations != history_evaluation_count:
            self.history['generation'].append(gen + 1)
            self.history['best_fitness'].append(float(self.best_fitness))
            self.history['mean_fitness'].append(float(np.mean(fitness)))
            self.history['best_yield'].append(float(yields[best_idx]))
            self.history['best_energy'].append(float(energies[best_idx]))
            self.history['best_penalty'].append(float(penalties[best_idx]))
            self.history['population_std'].append(float(np.std(fitness)))

        # Final evaluation
        _, self.best_details = self.env.fitness(self.best_solution)
        self.n_fitness_evaluations += 1

        if verbose:
            print(
                f"\n  RCGA Complete: {gen+1} generations in {self.elapsed_time:.1f}s"
            )
            print(
                f"  Best: Fitness={self.best_fitness:.4f}, "
                f"Yield={self.best_details['total_yield']:.4f}, "
                f"Energy={self.best_details['total_energy']:.2f}, "
                f"Penalty={self.best_details['total_penalty']:.2f}, "
                f"Feasible={self.best_details['is_feasible']}"
            )

        return self.best_solution, self.best_fitness, self.best_details

    def get_convergence_curve(self) -> np.ndarray:
        """Return best-fitness trajectory."""
        return np.array(self.history["best_fitness"])


# Ablation RCGA: controlled penalty-mode experiments


class AblationRCGA(RCGAOptimizer):
    """RCGA variant with selectable penalty mode for ablation studies.

    Parameters
    ----------
    env : GreenhouseEnv
    config : RCGAConfig
    penalty_mode : str
        - "smooth" : quadratic penalty bridge (default, same as RCGAOptimizer)
        - "hard"   : infinite penalty for infeasible candidates
        - "none"   : no penalty (optimises yield - energy only)
    """

    def __init__(
        self,
        env: GreenhouseEnv,
        config: Optional[RCGAConfig] = None,
        penalty_mode: str = "smooth",
    ):
        super().__init__(env, config)
        if penalty_mode not in ("smooth", "hard", "none"):
            raise ValueError(f"Unknown penalty_mode: {penalty_mode!r}. "
                             f"Use 'smooth', 'hard', or 'none'.")
        self.penalty_mode = penalty_mode

    def _evaluate_population(
        self, pop: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate with penalty-mode specific fitness computation."""
        n = len(pop)
        fitness = np.zeros(n)
        yields = np.zeros(n)
        energies = np.zeros(n)
        penalties = np.zeros(n)

        for i in range(n):
            f, d = self.env.fitness(pop[i])
            yields[i] = d["total_yield"]
            energies[i] = d["total_energy"]
            penalties[i] = d["total_penalty"]

            if self.penalty_mode == "smooth":
                # Default: fitness = alpha*yield - beta*energy - penalty
                fitness[i] = f
            elif self.penalty_mode == "hard":
                # VRPTW-style: any infeasibility => effectively dead
                if d["total_penalty"] > 0:
                    fitness[i] = -1e15  # effectively -inf without numeric issues
                else:
                    # Use only yield-energy component (penalty is zero anyway)
                    fitness[i] = d["total_yield"] - d["total_energy"]
            elif self.penalty_mode == "none":
                # Ignore penalty entirely
                fitness[i] = d["total_yield"] - d["total_energy"]

        return fitness, yields, energies, penalties

    def optimize(
        self,
        seed_population: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run RCGA optimisation with selected penalty mode."""
        if verbose:
            print(f"  AblationRCGA: penalty_mode={self.penalty_mode!r}")
        return super().optimize(seed_population=seed_population, verbose=verbose)

# Self-test


if __name__ == "__main__":
    env = GreenhouseEnv(seed=42)
    rcga_cfg = RCGAConfig(
        pop_size=60,
        n_generations=100,
        patience=20,
        random_seed=42,
    )
    optimizer = RCGAOptimizer(env, rcga_cfg)

    print("RCGA Self-test")
    print(f"  Config: pop={rcga_cfg.pop_size}, gens={rcga_cfg.n_generations}")
    print(f"  n_vars: {optimizer.n_vars}")

    # Quick run without seeding (Baseline 1: pure random init)
    best_x, best_f, details = optimizer.optimize(verbose=True)

    print("\n  Environment summary:")
    print(env.summary(best_x))
