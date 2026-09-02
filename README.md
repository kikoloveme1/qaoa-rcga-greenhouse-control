# QAOA–RCGA greenhouse control

Research code for a 24-hour greenhouse optimization problem with four hourly controls. QAOA supplies coarse initialization centers; RCGA refines the full continuous plan. The repository also contains classical comparison controllers and matched initializer experiments.

This is a **source-only release**: no measured records, experiment results, checkpoints, or manuscript figures are included. Readers obtain their own Singapore greenhouse data. Omitting `--data` uses synthetic outdoor profiles, not measured observations.

## Implemented controllers

| Identifier | Implementation |
|---|---|
| `qaoa_rcga` | Simulator-fitted QUBO, QAOA sampling, RCGA refinement and local polishing |
| `rcga` | Random-initialized RCGA with local polishing |
| `pso_mpc` | Six-hour rolling-window PSO, 40 particles and 80 iterations per step |
| `mpc_receding` | Six-hour rolling-window SLSQP MPC |
| `tube_rmpc` | Local linearization, LQR feedback, error-tube propagation and tightened SLSQP constraints |
| `sac_ppo` | Stochastic actor, twin soft-Q critics, replay training and PPO clipped actor updates; 40,000 interactions |
| `es_policy_search` | ESPolicySearch evolution-strategy policy search |
| `tightened_mpc` | Fixed-margin constraint-tightened MPC |

Final plans have shape `24 × 4`: temperature setpoint, supplemental light, CO₂ injection command, humidity setpoint. Definitions and source mapping: [docs/ALIGNMENT.md](docs/ALIGNMENT.md).

## Scientific scope

`threshold_qualified` means `total_penalty <= 1e-6` for **control bounds and inter-hour slew only**. It does not certify indoor CO₂, realized RH, SWC, or physical safety. State diagnostic ranges are CO₂ 300–1800 ppm, RH 30–95%, SWC 0.12–0.35; the audit utility does not add them to the optimization objective.

The model clips RH to 0–100% and supplies irrigation to compensate evapotranspiration. Constant SWC is therefore not evidence of optimizer robustness. Indoor CO₂ can leave its diagnostic range, including nonphysical values in some schedules. Inspect state trajectories alongside Profit.

Comparisons concern complete pipelines under known scenario trajectories. They do not establish quantum advantage or guarantee causal online control under unknown weather. Profit uses model economic accounting; physical area conversion and field calibration are not established by this code release. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Installation

Python **3.12** is tested. Commands use one line for compatibility with PowerShell and POSIX shells.

```text
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. Check `python --version`. On Windows with multiple versions, create the environment with `py -3.12 -m venv .venv` instead.

```text
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

The lock file records the tested Windows environment. Where pinned wheels are unavailable, `python -m pip install -e ".[dev]"` resolves the dependency ranges; record that environment separately. For CPU-only PyTorch on Windows/Linux, install it before the lock file using `python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`.

## Quick checks and experiments

Smoke checks use reduced budgets, are not paper-scale replications, and print summaries without saving results:

```text
python -m paper_protocol principal --smoke --backend numpy --methods qaoa_rcga,rcga,pso_mpc,mpc_receding,tube_rmpc,sac_ppo,es_policy_search --scenarios baseline --seeds 42
```

External data instructions: [data/README.md](data/README.md). The schema contains headers only, not measured examples.

```text
python -m paper_protocol principal --smoke --backend numpy --methods qaoa_rcga --scenarios baseline --seeds 42 --data ../private-data/singapore.csv
```

Full principal study: six methods × seven scenarios × ten paired seeds. Run sequentially without retaining results:

```text
python -m paper_protocol principal --config configs/principal.json
```

Or run across CPU processes, explicitly saving outside the source tree:

```text
python scripts/run_parallel_principal.py --workers 8 --fitness-workers 2 --backend aer --out ../private-results/principal
```

Append `--data ../private-data/singapore.csv` for external records. Reduce workers on smaller machines. This parallel runner requires `--out` for resume support. Use a new output directory whenever input data changes, even if its filename does not.

Matched initializer comparisons fix downstream RCGA and disable polishing:

```text
python -m paper_protocol matched --config configs/matched.json --out ../private-results/matched
```

## Final trajectories and state audit

With `--out`, run JSONs retain `plan`, `details.co2_realized`, `details.rh_realized`, and `details.soil_water_content_hourly`. Audit extrema, hourly violation rates and joint control/state qualification:

```text
python scripts/audit_states.py --runs ../private-results/principal/runs --out ../private-results/state-audit
```

Auditing does not repair trajectories or change constraints. Hours are repeated observations, not independent experimental replicates.

## Layout

```text
algorithms/       RCGA and classical controllers
environment/      greenhouse dynamics, perturbations, CSV loader
paper_protocol/   protocol, surrogate, sampling, dispatch, summaries
configs/          principal, matched and sensitivity settings
scripts/          CPU parallel runner, state audit, archive packaging
tests/            model contracts, algorithms, data and smoke tests
data/             acquisition instructions and header-only CSV schema
docs/             reproducibility, alignment, validation, publishing
.github/          automated source checks
```

Run `python scripts/package_repository.py` for a source-only ZIP outside the repository. Read [docs/PUBLISHING.md](docs/PUBLISHING.md) before release. No project reuse license has been selected in this preparation step; dependency licenses remain their own.
