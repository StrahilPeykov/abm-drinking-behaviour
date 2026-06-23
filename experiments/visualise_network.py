"""Animated visualisation of the NetworkModel showing state changes over time.

Each frame shows:
  - Left: network graph with nodes coloured by state
      blue  = susceptible (S)
      red   = current drinker (D)
      grey  = former drinker (R)
  - Right: S/D/R fraction time series growing with each step

Usage:
    python experiments/visualise_network.py
    python experiments/visualise_network.py --n-agents 80 --steps 300 --save
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import NetworkModel, S, D, R

RESULTS = Path(__file__).resolve().parents[1] / "results"

STATE_COLOURS = {S: "#4C72B0", D: "#DD3333", R: "#888888"}
STATE_LABELS  = {S: "Susceptible", D: "Current drinker", R: "Former drinker"}


def node_colours(model: NetworkModel) -> list[str]:
    return [STATE_COLOURS[a.state] for a in model.agents]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents",  type=int,   default=60)
    parser.add_argument("--k",         type=int,   default=4)
    parser.add_argument("--p-rewire",  type=float, default=0.1)
    parser.add_argument("--gamma",     type=float, default=0.3)
    parser.add_argument("--rho",       type=float, default=0.3)
    parser.add_argument("--steps",     type=int,   default=200)
    parser.add_argument("--interval",  type=int,   default=120,
                        help="milliseconds between frames")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save",      action="store_true",
                        help="save animation to results/figures/ instead of showing")
    args = parser.parse_args()

    model = NetworkModel(
        n_agents=args.n_agents, k=args.k, p_rewire=args.p_rewire,
        gamma=args.gamma, rho=args.rho, seed=args.seed,
    )

    pos = nx.spring_layout(model.graph, seed=args.seed)

    fig, (ax_net, ax_ts) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax_net, ax_ts):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── network panel ────────────────────────────────────────────────────────
    ax_net.set_title("Network state", color="white", pad=10)
    ax_net.set_axis_off()

    nx.draw_networkx_edges(
        model.graph, pos, ax=ax_net,
        edge_color="#333355", alpha=0.5, width=0.6,
    )
    node_collection = nx.draw_networkx_nodes(
        model.graph, pos, ax=ax_net,
        node_color=node_colours(model),
        node_size=120, linewidths=0.4,
    )
    node_collection.set_edgecolor("#cccccc")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=STATE_COLOURS[s], markersize=9,
                   label=STATE_LABELS[s])
        for s in (S, D, R)
    ]
    ax_net.legend(handles=legend_handles, loc="lower left",
                  framealpha=0.3, labelcolor="white",
                  facecolor="#222244", edgecolor="#444466")

    step_text = ax_net.text(
        0.02, 0.97, "Step 0", transform=ax_net.transAxes,
        color="white", fontsize=11, va="top",
    )

    # ── time-series panel ────────────────────────────────────────────────────
    ax_ts.set_xlim(0, args.steps)
    ax_ts.set_ylim(0, 1)
    ax_ts.set_xlabel("Step", color="white")
    ax_ts.set_ylabel("Fraction", color="white")
    ax_ts.set_title("Population fractions", color="white", pad=10)

    xs: list[int]   = [0]
    ys_s: list[float] = [model.fractions()[S]]
    ys_d: list[float] = [model.fractions()[D]]
    ys_r: list[float] = [model.fractions()[R]]

    line_s, = ax_ts.plot(xs, ys_s, color=STATE_COLOURS[S], lw=1.5,
                         label=STATE_LABELS[S])
    line_d, = ax_ts.plot(xs, ys_d, color=STATE_COLOURS[D], lw=1.5,
                         label=STATE_LABELS[D])
    line_r, = ax_ts.plot(xs, ys_r, color=STATE_COLOURS[R], lw=1.5,
                         label=STATE_LABELS[R])
    ax_ts.legend(framealpha=0.3, labelcolor="white",
                 facecolor="#222244", edgecolor="#444466")

    fig.tight_layout(pad=2)

    # ── animation ────────────────────────────────────────────────────────────
    def update(frame: int):
        if frame == 0:
            return node_collection, step_text, line_s, line_d, line_r

        model.step()
        fracs = model.fractions()

        node_collection.set_facecolor(node_colours(model))
        step_text.set_text(f"Step {frame}")

        xs.append(frame)
        ys_s.append(fracs[S])
        ys_d.append(fracs[D])
        ys_r.append(fracs[R])

        line_s.set_data(xs, ys_s)
        line_d.set_data(xs, ys_d)
        line_r.set_data(xs, ys_r)

        return node_collection, step_text, line_s, line_d, line_r

    ani = animation.FuncAnimation(
        fig, update, frames=args.steps + 1,
        interval=args.interval, blit=True, repeat=False,
    )

    if args.save:
        (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        out = RESULTS / "figures" / "network_animation.gif"
        ani.save(out, writer="pillow", fps=1000 // args.interval)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()