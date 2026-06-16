import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import NetworkModel

RESULTS = Path(__file__).resolve().parents[1] /  "results"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p-rewire", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--initial-drinker-frac", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42,)
    args = parser.parse_args()

    model = NetworkModel(
        n_agents=args.n_agents, k=args.k, p_rewire=args.p_rewire, gamma=args.gamma,
        rho=args.rho, initial_drinker_frac=args.initial_drinker_frac, seed=args.seed
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