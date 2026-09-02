# Source verification

Use Python 3.12 and install `requirements-lock.txt`. Verification is source based and creates no retained experiment results.

```bash
python -m pytest -q
python -m compileall -q algorithms environment paper_protocol
python -m paper_protocol principal --smoke --backend numpy --methods es_policy_search,sac_ppo,tube_rmpc --scenarios baseline --seeds 42
```

The tests check greenhouse equations and bounds, QAOA backend equivalence, RCGA contracts, SAC-PPO actor and update behavior, tube propagation and tightened constraints, external CSV validation, controller dispatch, deterministic smoke behavior, opt-in output persistence, diagnostic state qualification, and source-archive exclusions. Passing these tests verifies implementation contracts; it does not certify field validity or full state feasibility.

For an external data check, copy the schema outside the repository, add 24 consecutive hourly observations, and pass the resulting path through `--data`. Downloaded observations and generated outputs must remain outside the public source tree.
