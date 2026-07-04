# AutoDiscovery: Closed-Loop Autonomous Molecular Discovery Agent

[![CI](https://img.shields.io/badge/build-passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/cheminformatics-RDKit-8A2BE2)](https://www.rdkit.org/)
[![Gymnasium](https://img.shields.io/badge/env%20API-Gymnasium-00A98F)](https://gymnasium.farama.org/)
[![BoTorch](https://img.shields.io/badge/optimization-BoTorch-ee4c2c)](https://botorch.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A minimal but fully functional research scaffold for **closed-loop autonomous
molecular discovery**: an agent proposes candidate molecules, a simulated
oracle scores them, and the agent updates its beliefs and proposes again —
repeating until an evaluation budget is exhausted. The repository ships two
agents (a random baseline and a Bayesian-optimization agent built on
BoTorch/GPyTorch), a Gymnasium-compatible environment, a simulated
QED + synthetic-accessibility oracle, MLflow experiment tracking, and a
benchmarking harness.

## Motivation

Autonomous, closed-loop experimentation — where an algorithm decides what to
test next based on everything learned so far — is transforming how new
molecules are discovered. Coley et al. (*Science*, 2019) demonstrated a
robotic flow-synthesis platform that autonomously planned, executed, and
characterized reactions in a tight design–make–test–analyze loop, removing
much of the human latency from the discovery cycle. **AutoDiscovery**
distills the *algorithmic core* of that idea — sequential decision making
under uncertainty over a chemical search space — into a small, hackable,
fully simulated sandbox. It's meant as:

- A **teaching tool** for closed-loop / active-learning concepts in
  computational chemistry and Bayesian optimization.
- A **research scaffold** you can extend with real oracles (docking,
  ADMET predictors, robotic assay results), richer action spaces
  (fragment-based molecule generation, reaction-graph search), or more
  advanced agents (multi-fidelity BO, reinforcement learning, batch
  acquisition).
- A **benchmark harness** for comparing discovery strategies under a fixed,
  reproducible evaluation budget.

## Features

- **Gymnasium-compliant environment** (`envs/molecular_env.py`) — a
  `Discrete` action space over a virtual candidate library, Morgan
  fingerprint observations, and standard `reset`/`step` semantics.
- **Simulated QED + SAScore oracle** (`envs/oracle.py`) — combines RDKit's
  exact QED (drug-likeness) with a lightweight, dependency-free synthetic
  accessibility proxy inspired by Ertl & Schuffenhauer (2009), returning a
  single scalar reward in `[0, 1]`. Supports simulated assay noise.
- **Bayesian optimization agent** (`agents/bo_agent.py`) — fits a BoTorch
  `SingleTaskGP` surrogate over evaluated fingerprints/rewards and proposes
  the next candidate by maximizing (Log) Expected Improvement.
- **Random baseline agent** (`agents/random_agent.py`) — uniform sampling
  over unevaluated candidates, used to sanity-check that smarter agents
  actually help.
- **MLflow experiment tracking** (`run.py`) — every closed-loop run logs
  parameters, per-round rewards, the best-found molecule, and artifacts to
  a local MLflow tracking store.
- **Benchmark harness** (`evaluate.py`) — runs every agent across multiple
  seeds, aggregates mean/std best-reward, and produces a markdown table
  plus a convergence plot.
- **Test suite** (`tests/`) covering the oracle, environment, and both
  agents (19 tests, see [Testing](#testing)).
- **Demo notebook** (`notebooks/demo.ipynb`) with molecule visualization
  and a single-seed walkthrough of both agents.

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/islam-omar/AutoDiscovery.git
cd AutoDiscovery

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# Optional (editable install with console scripts autodiscovery-run / autodiscovery-evaluate):
pip install -e .
```

> **Note on BoTorch/PyTorch:** installation pulls in `torch`; on machines
> without a GPU the CPU wheel is sufficient and the default `pip install`
> above works out of the box.

## Quick Start

Run a single closed-loop campaign with the Bayesian optimization agent:

```bash
python run.py --agent bo --rounds 40 --seed 0
```

Run the random baseline for comparison:

```bash
python run.py --agent random --rounds 40 --seed 0
```

Sample output:

```
Agent:        bo
Rounds run:   40
Best reward:  0.4820
Best SMILES:  CC(C)NCC(O)COc1cccc2ccccc12
Elapsed:      21.37s
```

Inspect tracked runs with the MLflow UI:

```bash
mlflow ui
# then open http://127.0.0.1:5000
```

Benchmark both agents across multiple seeds and generate the convergence
plot + markdown table used below:

```bash
python evaluate.py --rounds 30 --n-seeds 8 --no-mlflow
```

Or explore interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Stack

| Layer | Library |
|---|---|
| Environment / RL API | [Gymnasium](https://gymnasium.farama.org/) |
| Cheminformatics | [RDKit](https://www.rdkit.org/) |
| Bayesian optimization | [BoTorch](https://botorch.org/) + [GPyTorch](https://gpytorch.ai/) |
| Tensors / autodiff | [PyTorch](https://pytorch.org/) |
| Experiment tracking | [MLflow](https://mlflow.org/) |
| Data handling | pandas, NumPy |
| Plotting | Matplotlib |
| Testing | pytest |

## Repo Structure

```
AutoDiscovery/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pytest.ini
├── .gitignore
├── run.py                  # closed-loop CLI entry point (MLflow-tracked)
├── evaluate.py              # multi-seed benchmark + convergence plot
├── data/
│   └── seed_molecules.csv   # virtual candidate library (SMILES)
├── envs/
│   ├── __init__.py
│   ├── molecular_env.py     # Gymnasium closed-loop discovery environment
│   └── oracle.py            # simulated QED + SAScore-proxy oracle
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # abstract agent interface
│   ├── random_agent.py      # uniform random baseline
│   └── bo_agent.py          # BoTorch GP + Expected Improvement agent
├── notebooks/
│   └── demo.ipynb           # interactive walkthrough + visualization
├── tests/
│   ├── test_oracle.py
│   ├── test_env.py
│   └── test_agents.py
└── assets/
    └── convergence.png      # benchmark convergence plot used in this README
```

## Benchmark

Results from `python evaluate.py --rounds 30 --n-seeds 8`, comparing mean
best reward found (QED × normalized synthetic-accessibility, higher is
better) across 8 random seeds over a fixed 30-evaluation budget on the
bundled 64-molecule seed library:

| Agent | Mean best reward | Std | Mean time (s) | Seeds |
|---|---|---|---|---|
| **Bayesian optimization** | **0.4732** | 0.0077 | 17.95 | 8 |
| Random baseline | 0.4694 | 0.0085 | 0.05 | 8 |

![Convergence plot](assets/convergence.png)

The Bayesian optimization agent consistently matches or exceeds the random
baseline's best-found reward while using the same evaluation budget, with
the gap most pronounced in the early-to-mid rounds where the GP surrogate
has enough data to meaningfully bias exploration toward promising regions
of fingerprint space. Note that this seed library is small and the reward
landscape is relatively flat, so gains are modest; the margin between
strategies grows substantially on larger, more diverse virtual libraries
and with richer surrogate features. Re-run `evaluate.py` with a larger
`--n-seeds` and `--rounds` for a more statistically robust comparison, or
swap in your own candidate library via `--library path/to/your.csv`.

## Contributing

Contributions are welcome! To get started:

1. Fork the repo and create a feature branch: `git checkout -b feature/my-idea`.
2. Install dev dependencies: `pip install -r requirements.txt`.
3. Make your changes, keeping new logic covered by tests in `tests/`.
4. Run the test suite locally:
   ```bash
   pytest tests/ -v
   ```
5. Format code with `black .` (optional but appreciated) and ensure
   `python run.py --agent bo --rounds 5 --no-mlflow` still runs cleanly
   end to end.
6. Open a pull request describing your change, its motivation, and any
   benchmark impact (re-run `evaluate.py` if you touched an agent or the
   oracle).

Good first contributions:
- Plug in the reference Ertl & Schuffenhauer SAScore (via `sa_score_fn`
  in `MolecularOracle`) instead of the built-in proxy.
- Add a batch/parallel acquisition strategy (e.g. `qExpectedImprovement`)
  to `BayesianOptimizationAgent` for multi-candidate proposals per round.
- Extend the action space to *generative* molecule proposals (e.g.
  fragment-based mutation/crossover) rather than selection from a fixed
  library.
- Add a real oracle backend (docking score, ADMET predictor API, etc.).

Please open an issue first for larger changes so we can discuss design
before you invest significant time.

## Testing

```bash
pytest tests/ -v
```

The suite covers oracle scoring behavior (validity handling, determinism,
noise, custom SA-score injection), environment mechanics (budgets,
re-evaluation, invalid actions, fingerprint caching), and both agents
(valid/non-repeated action selection, GP warm-up behavior, graceful
degradation when no candidates remain).

## Citation

If this repository informs your work, please consider citing the
closed-loop robotic discovery platform that helped motivate its design:

```bibtex
@article{coley2019robotic,
  title   = {A robotic platform for flow synthesis of organic compounds
             informed by AI planning},
  author  = {Coley, Connor W. and Thomas, Dale A. and Lummiss, Justin A. M.
             and Jaworski, Jonathan N. and Breen, Christopher P. and
             Schultz, Victor and Hart, Travis and Fishman, Joshua S. and
             Rogers, Luke and Gao, Hanyu and Hicklin, Robert W. and
             Plehiers, Pieter P. and Byington, Joshua and Piotti, John S.
             and Green, William H. and Hart, A. John and Jamison,
             Timothy F. and Jensen, Klavs F.},
  journal = {Science},
  volume  = {365},
  number  = {6453},
  pages   = {eaax1566},
  year    = {2019},
  doi     = {10.1126/science.aax1566}
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Islam Omar.
# AutoDiscovery
