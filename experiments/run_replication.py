"""Replicate the main results of Gorman et al. (2006), Figures 2-5.

One figure per result in results/figures/:

1. replication_fig2_sdr.png        - S/D/R fractions over time
                                     (p=0.1, gamma=rho=0.3)
2. replication_fig3_motion.png     - susceptible fraction over time for
                                     p=0.5, 0.01, 0.2 (the paper's panel order;
                                     non-monotonic effect of motion)
3. replication_fig4_bar_space.png  - drinker distribution over sites after
                                     1000 iterations, bar vs no bar (p=0.3)
4. replication_fig5_bar_susc.png   - susceptible fraction over time,
                                     bar vs no bar (p=0.3)

Figures 2, 4 and 5 show single stochastic runs, as in the paper; Figure 3 shows
15 runs per panel with the median-depletion run highlighted, because single runs
fluctuate too much to compare time scales. Pass --seed to vary.

Usage:
    python experiments/run_replication.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import GormanModel

FIGURES = Path(__file__).resolve().parents[1] / "results" / "figures"

GAMMA = RHO = 0.3
BAR_P = 0.3  # motion parameter used in the paper's bar experiments
# The paper places the bar "at the left edge of the domain"; the spike in its
# Figure 4 is centred near site 7, so the bar sits a few sites in from the edge.
# Placing it exactly at site 0 piles drinkers onto one site against the boundary
# and overshoots the peak.
BAR_SITE = 7


def fig2_sdr(seed):
    model = GormanModel(p_move=0.1, gamma=GAMMA, rho=RHO, seed=seed)
    history = model.run(1000)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    series = [("susceptible", "% Susceptible"),
              ("current_drinkers", "% Current Drinkers"),
              ("former_drinkers", "% Former Drinkers")]
    for ax, (column, label) in zip(axes, series):
        ax.plot(history["iteration"], history[column], lw=0.8)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Iterations")
    fig.suptitle("Fig. 2 replication: S/D/R evolution (p=0.1, γ=ρ=0.3)")
    return fig, "replication_fig2_sdr.png"


def fig3_motion(seed, n_runs=15):
    """Single runs fluctuate a lot (the paper notes this), so each panel
    shows all runs faintly and highlights the run with the median
    susceptible-depletion time."""
    settings = [(0.5, 1000), (0.01, 10000), (0.2, 1000)]
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    for ax, (p_move, steps) in zip(axes, settings):
        histories, depletion = [], []
        for run in range(n_runs):
            model = GormanModel(p_move=p_move, gamma=GAMMA, rho=RHO,
                                seed=seed + run)
            history = model.run(steps)
            histories.append(history)
            depleted = np.nonzero(history["susceptible"] == 0)[0]
            depletion.append(depleted[0] if len(depleted) else steps)
        for history in histories:
            ax.plot(history["iteration"], history["susceptible"],
                    lw=0.5, color="0.85")
        median = int(np.argsort(depletion)[n_runs // 2])
        ax.plot(histories[median]["iteration"],
                histories[median]["susceptible"], lw=1.0, color="C0")
        ax.set_ylabel("% Susceptible")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"p = {p_move} (median depletion at {depletion[median]} "
            f"iterations over {n_runs} runs)", fontsize=10)
    axes[-1].set_xlabel("Iterations")
    fig.suptitle("Fig. 3 replication: effect of motion on susceptibility")
    return fig, "replication_fig3_motion.png"


def fig4_bar_space(seed):
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = [(None, "no bar", dict(lw=0.7, color="0.7", ls="--")),
              (BAR_SITE, f"bar at site {BAR_SITE}", dict(lw=1.0, color="C0"))]
    for bar_site, label, style in styles:
        model = GormanModel(p_move=BAR_P, gamma=GAMMA, rho=RHO,
                            bar_site=bar_site, seed=seed)
        model.run(1000)
        ax.plot(model.drinker_distribution(), label=label, **style)
    ax.set_xlabel("Index of Lattice Sites")
    ax.set_ylabel("Proportion of Drinkers at Site")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.suptitle("Fig. 4 replication: drinkers cluster around the bar "
                 f"(p={BAR_P}, 1000 iterations)")
    return fig, "replication_fig4_bar_space.png"


def fig5_bar_susc(seed):
    """Replicates the buffering effect qualitatively: with a bar the
    susceptible fraction plateaus at a nonzero level instead of reaching 0."""
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = [(None, "no bar", dict(lw=0.7, color="0.7", ls="--")),
              (BAR_SITE, f"bar at site {BAR_SITE}", dict(lw=1.0, color="C0"))]
    for bar_site, label, style in styles:
        model = GormanModel(p_move=BAR_P, gamma=GAMMA, rho=RHO,
                            bar_site=bar_site, seed=seed)
        history = model.run(1000)
        ax.plot(history["iteration"], history["susceptible"], label=label,
                **style)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("% Susceptible")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.suptitle("Fig. 5 replication: the bar buffers conversion of "
                 f"susceptibles (p={BAR_P})")
    return fig, "replication_fig5_bar_susc.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    for experiment in (fig2_sdr, fig3_motion, fig4_bar_space, fig5_bar_susc):
        fig, filename = experiment(args.seed)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=150)
        plt.close(fig)
        print(f"Wrote {FIGURES / filename}")


if __name__ == "__main__":
    main()
