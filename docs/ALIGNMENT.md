# Manuscript-to-code alignment

| Manuscript method or setting | Source implementation |
|---|---|
| 24 hourly decisions × 4 controls | `environment/greenhouse_model.py`, `paper_protocol/protocol.py` |
| SF-QUBO construction and Ridge fit | `paper_protocol/surrogate.py` |
| QAOA statevector sampling | `paper_protocol/sampling.py` |
| QAOA-seeded RCGA | `paper_protocol/runner.py`, `algorithms/classical/rcga.py` |
| Random-initialized RCGA | `algorithms/classical/rcga.py` |
| PSO-MPC, horizon 6 | `algorithms/classical/pso_mpc.py` |
| Receding MPC, horizon 6, SLSQP ≤300 | `algorithms/classical/mpc_receding.py` |
| Tube RMPC, constraint tightening, SLSQP ≤300 | `algorithms/classical/tube_rmpc.py` |
| SAC-PPO, 40,000 environment interactions | `algorithms/classical/sac_ppo.py` |
| Evolution-strategy policy search | `algorithms/classical/es_policy_search.py` |
| Seven scenario definitions | `environment/greenhouse_perturbation.py` |
| Ten paired random seeds | `paper_protocol/protocol.py` |

## SAC-PPO definition

The stochastic actor produces tanh-squashed Gaussian actions. Two soft-Q critics use clipped double-Q Bellman targets and target networks updated by Polyak averaging. Replay samples train the critics; fresh rollout samples train the actor through the PPO clipped probability-ratio objective with entropy regularization. The deterministic actor mean produces the reported daily plan.

## Tube RMPC definition

The controller obtains local `A` and `B` matrices by central finite differences of the greenhouse transition. A discrete Riccati solution supplies the ancillary feedback gain. Axis-aligned disturbance sets are propagated under the closed-loop matrix, converted into control-error radii, and subtracted from admissible bounds. The nominal horizon is optimized by SLSQP and the first feedback-corrected action is applied before re-linearization.

## Data boundary

The repository supplies a strict CSV schema and injection code. It does not include Singapore greenhouse observations. Weather records used by a reader are external inputs and their provenance, permission, quality control and calibration remain attached to the selected provider. Generated controller outputs are not part of the source repository.
