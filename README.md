# Agent-Based Modelling of Drinking Behaviour

Group project for the Agent-Based Modelling course (UvA, June 2026).

We replicate and extend the drinking-behaviour model of Gorman et al. (2006),
in which agents on a one-dimensional lattice move, interact locally, and
transition between three states: susceptible nondrinkers (S), current
drinkers (D), and former drinkers (R). In the extended version of the original
model, a bar is introduced as an environmental attractor for current drinkers,
altering their movement and the spatial clustering of drinking behaviour.
Our research question (from the project proposal):

> How do bounded rationality, risk/loss aversion, and alcohol-outlet
> attraction affect the emergence, clustering, and persistence of drinking
> behaviour?

The full submitted proposal (abstract + motivation for ABM) is in
[`docs/proposal.md`](docs/proposal.md).

## Status

Early setup. The proposal has been submitted; the model design and exact
extensions are still being decided by the group.

Planned outline (subject to change):

1. Replicate the baseline Gorman et al. (2006) model and key results.
2. Extend it (candidate directions: alternative topologies such as a 2D
   lattice or social network, heterogeneous agent attributes, rewiring of
   social ties).
3. Sensitivity analysis, which is mandatory for the course. We currently plan
   to use OFAT and, if computationally feasible, Morris or Sobol analysis via
   SALib.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the baseline model

Reproduces the dynamics of Figure 2 in Gorman et al. (2006), susceptibles
decline to zero around iteration 500–600, current and former drinkers
equilibrate around 0.5 each:

```bash
python experiments/run_baseline.py
```

With the bar at the left edge of the lattice (Figures 4–5 of the paper):

```bash
python experiments/run_baseline.py --p-move 0.3 --bar-site 0
```

Output (a figure and a CSV time series) is written to `results/`.

## Repository structure

```
docs/         project proposal and notes
src/          model code
experiments/  reproducible experiment scripts
results/      generated figures and data (raw/processed runs are gitignored)
```

## Attribution

All code in this repository is written by the project group unless explicitly
stated otherwise. Reused or adapted code will be clearly marked in the file
where it appears and listed here.

## References

- Gorman, D. M., Mezic, J., Mezic, I., & Gruenewald, P. J. (2006). Agent-based
  modeling of drinking behavior: a preliminary model and potential
  applications to theory and practice. *American Journal of Public Health*,
  96(11), 2055–2060.
- Müller, B., et al. (2013). Describing human decisions in agent-based models
  – ODD+D, an extension of the ODD protocol. *Environmental Modelling &
  Software*, 48, 37–48.
