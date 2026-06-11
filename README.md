# Agent-Based Modelling of Drinking Behaviour

Group project for the Agent-Based Modelling course (UvA, June 2026).

We replicate and extend the drinking-behaviour model of Gorman et al. (2006), in
which agents on a one-dimensional lattice move, interact locally, and transition
between three states: susceptible nondrinkers (S), current drinkers (D), and
former drinkers (R). In the second-generation model described by Gorman et al.,
a bar is introduced as an environmental attractor for current drinkers, altering
their movement and the spatial clustering of drinking behaviour. Our research
question (from the project proposal):

> How do bounded rationality, risk/loss aversion, and alcohol-outlet attraction
> affect the emergence, clustering, and persistence of drinking behaviour?

The full submitted proposal (abstract + motivation for ABM) is in
[`docs/proposal.md`](docs/proposal.md).

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

With the bar near the left edge of the lattice (Figures 4–5 of the paper;
see the replication notes for why site 7):

```bash
python experiments/run_baseline.py --p-move 0.3 --bar-site 7
```

Output (a figure and a CSV time series) is written to `results/`.

To reproduce all four main results of the paper at once (Figures 2–5: S/D/R
dynamics, the non-monotonic effect of agent motion, drinker clustering around
the bar, and the bar's buffering of susceptibles):

```bash
python experiments/run_replication.py
```

### Replication notes

Figures 2–4 reproduce the qualitative behaviour of the paper and approximately
match the reported time scales. Susceptible agents disappear in the no-bar
baseline, movement speed affects the rate of conversion non-monotonically, and
current drinkers cluster around the bar.

The exact implementation of the bar is not fully specified in the paper. The
paper states that the bar is at the left edge of the domain, while the plotted
peak in Figure 4 appears a few sites from the left boundary. In
`run_replication.py` we therefore use `BAR_SITE = 7`, which better matches the
visual location of the reported clustering peak than placing the bar exactly at
site 0.

Figure 5 is reproduced qualitatively: adding a bar buffers the conversion of
susceptibles, causing the susceptible fraction to plateau above zero rather than
disappear completely. The plateau level in our implementation is higher than in
the paper. Under our implementation assumptions, including synchronous
transitions, closed boundaries, and global attraction of current drinkers toward
the bar, we have not found a parameterization that simultaneously matches both
the rapid early conversion and the later low susceptible plateau shown in the
original figure. We therefore document this difference and use the replication
primarily as a baseline for the extension model.

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

## Status (temporary)

This section tracks work in progress during the course and will be removed
before submission.

The baseline Gorman et al. (2006) model is implemented. Figures 2–4 are
reproduced qualitatively and approximately in time scale, while Figure 5 is
reproduced qualitatively but differs quantitatively; see the replication notes
above. The extension model is still being decided by the group.

Planned outline (subject to change):

1. ~~Replicate the baseline Gorman et al. (2006) model and key results.~~ done
2. Extend it (candidate directions for now: alternative topologies such as a 2D
   lattice or social network, heterogeneous agent attributes, rewiring of social
   ties).
3. Sensitivity analysis, which is mandatory for the course. We currently plan to
   use OFAT and, if computationally feasible, Morris or Sobol analysis via
   SALib.

## References

- Gorman, D. M., Mezic, J., Mezic, I., & Gruenewald, P. J. (2006). Agent-based
  modeling of drinking behavior: a preliminary model and potential applications
  to theory and practice. *American Journal of Public Health*, 96(11),
  2055–2060.
- Müller, B., et al. (2013). Describing human decisions in agent-based models –
  ODD+D, an extension of the ODD protocol. *Environmental Modelling & Software*,
  48, 37–48.
