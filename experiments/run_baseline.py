"""Run the baseline Gorman et al. (2006) model and plot S/D/R fractions.

Defaults reproduce Figure 2 of the paper (p = 0.1, gamma = rho = 0.3, 100 sites,
1000 iterations): the susceptible fraction drops to 0 around iteration 600 and
current/former drinkers settle around 0.5 each.

Usage:
    python experiments/run_baseline.py python experiments/run_baseline.py
    --p-move 0.3 --bar-site 7

Site 7 matches the visual location of the clustering peak in the paper's Figure
4 (see the replication notes in the README); pass --bar-site 0 for the literal
left-edge reading.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import GormanModel

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--p-move", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--bar-site", type=int, default=None,
                        help="lattice index of the bar (omit for no bar)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = GormanModel(
        n_sites=args.n_sites, p_move=args.p_move, gamma=args.gamma,
        rho=args.rho, bar_site=args.bar_site, seed=args.seed,
    )
    history = pd.DataFrame(model.run(args.steps))

    tag = "bar" if args.bar_site is not None else "baseline"
    (RESULTS / "processed").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "processed" / f"{tag}_timeseries.csv"
    history.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    series = [("susceptible", "% Susceptible"),
              ("current_drinkers", "% Current Drinkers"),
              ("former_drinkers", "% Former Drinkers")]
    for ax, (column, label) in zip(axes, series):
        ax.plot(history["iteration"], history[column], lw=0.8)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Iterations")
    fig.suptitle(
        f"Gorman et al. (2006) {tag} model: p={args.p_move}, "
        f"γ={args.gamma}, ρ={args.rho}, seed={args.seed}"
    )
    fig.tight_layout()
    fig_path = RESULTS / "figures" / f"{tag}_sdr_fractions.png"
    fig.savefig(fig_path, dpi=150)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
