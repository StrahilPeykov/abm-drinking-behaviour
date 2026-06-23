"""Run the social-network drinking model and plot S/D/R fractions over time.

Exposes every model parameter on the command line, including the behavioural
ones (bounded rationality, risk/loss aversion, age, habituation), so the full
extension can be driven without editing code.

Usage:
    python experiments/run_network.py
    python experiments/run_network.py --kappa-mean 0.5 --habituation-rate 0.2
    python experiments/run_network.py --tau 0.1 --age-mean 17
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import NetworkModel

RESULTS = Path(__file__).resolve().parents[1] /  "results"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Network / structural parameters
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p-rewire", type=float, default=0.1)
    # Inherited Gorman transition rates
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=0.3)
    # Behavioural parameters (the extension)
    parser.add_argument("--tau", type=float, default=0.5,
                        help="logit temperature (bounded rationality)")
    parser.add_argument("--lam-mean", type=float, default=2.0,
                        help="mean loss-aversion coefficient")
    parser.add_argument("--kappa-mean", type=float, default=0.0,
                        help="mean risk-aversion / utility curvature")
    parser.add_argument("--age-mean", type=float, default=20.0,
                        help="mean agent age (clipped to 16-24)")
    parser.add_argument("--habituation-rate", type=float, default=0.0,
                        help="decay of quit rate with drinking tenure")
    parser.add_argument("--initial-drinker-frac", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = NetworkModel(
        n_agents=args.n_agents, k=args.k, p_rewire=args.p_rewire,
        gamma=args.gamma, rho=args.rho, tau=args.tau, lam_mean=args.lam_mean,
        kappa_mean=args.kappa_mean, age_mean=args.age_mean,
        habituation_rate=args.habituation_rate,
        initial_drinker_frac=args.initial_drinker_frac, seed=args.seed,
    )
    history = pd.DataFrame(model.run(args.steps))

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

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
        f"Network model: n={args.n_agents}, k={args.k}, p_rewire={args.p_rewire}, "
        f"γ={args.gamma}, ρ={args.rho}, seed={args.seed}"
    )
    fig.tight_layout()

    fig_path = RESULTS / "figures" / "network_sdr_fractions.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()