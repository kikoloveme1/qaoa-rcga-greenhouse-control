# Reproducibility and interpretation

## Protocol and input

`configs/principal.json` specifies 24 hours, seven scenarios, ten paired seeds, QAOA depth 3 and 32,768 shots, RCGA population 185 and at most 200 generations. PSO-MPC uses six hours, 40 particles and 80 iterations per step; SAC-PPO uses 40,000 interactions. These are not equal-evaluation budgets.

The matched study fixes downstream RCGA and disables polishing. Principal results include polishing and stopping-rule effects and cannot isolate the causal contribution of QAOA.

Without `--data`, outdoor profiles are synthetic. With it, outdoor columns drive scenario evaluation. Optional measured indoor columns are loaded but not automatically used for fitting or validation. The QUBO landscape still uses the configured synthetic nominal baseline; external observations do not replace its fitting data.

Observations, measurement validation records and generated outputs are not distributed. This release does not certify field accuracy. Keep data provider, acquisition dates, permissions, preprocessing, timezone and input hashes with your own records.

## Separate metrics

- `profit`: modeled economic objective before control residual penalties.
- `fitness`: `profit - total_penalty` under the principal protocol.
- `threshold_qualified`: control-bound/slew penalty at most `1e-6`.

Diagnostic intervals: CO₂ [300, 1800] ppm, RH [30, 95]%, SWC [0.12, 0.35]. Strictly outside counts as a violation. State qualification requires all saved hours in all intervals; joint qualification also requires the control threshold to pass. These manuscript ranges are not agronomic or occupational safety declarations.

RH is projected to [0, 100]. SWC uses simulator-provided irrigation. CO₂ is propagated without clipping to input bounds. Inspect physical plausibility independently of optimizer success. Tube tightening in a local model does not prove that the final nonlinear trajectory is within range.

## Persistence and timing

The main CLI saves only with explicit `--out`; the parallel script requires it. Keep outputs outside Git. Final JSONs contain the 24×4 plan, 24 post-update samples of CO₂/RH/SWC and other states, objective values and protocol/source/landscape fingerprints. Initial states are not included in these 24 samples.

Resume checks reject incompatible protocols and source/landscape versions. CSV content is not currently part of resume identity: use a new directory whenever data changes. Generic summaries do not automatically enforce state qualification.

Task timing covers the online pipeline, excluding shared offline landscape fitting; sampling can be cached. Parallel tasks compete for CPU resources. These are not isolated timings or cold-start quantum resource estimates. Simulator counters count wrapper calls and may exclude internal controller state transitions; they are not a universal compute metric.

## Reporting

Report economic performance, control qualification and state diagnostics separately. Label any exclusion of failed runs. An observed best feasible schedule is a lower bound on an instance's attainable optimum, not an upper bound or a cross-scenario mean. Use paired scenario/seed comparisons and do not treat hourly points as independent experiments.
