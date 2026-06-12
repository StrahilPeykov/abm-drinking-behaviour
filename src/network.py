"""Small-world network extension of the Gorman et al. (2006) drinking-behaviour ABM.

Agents are nodes on a Watts-Strogatz small-world graph. Each iteration they
convert between states based on the composition of their neighbourhood
(``_transition``). There is no spatial movement; the network topology encodes
social structure.

Transition rules mirror the lattice model, replacing site co-occupants with
graph neighbours::

    P(S -> D) = d_neighbours / degree  if d_neighbours / degree >= 0.5, else 0
    P(D -> R) = r_neighbours / degree + gamma
    P(R -> D) = d_neighbours / degree + rho

Isolated nodes (degree 0) use only the baseline rates gamma / rho.
Transitions are synchronous: all agents convert based on neighbourhood
composition at the start of the iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

S, D, R = 0, 1, 2


@dataclass
class Agent:
    """A single agent node with a drinking state."""

    node_id: int
    state: int = S

    @property
    def is_susceptible(self) -> bool:
        return self.state == S

    @property
    def is_drinker(self) -> bool:
        return self.state == D

    @property
    def is_former_drinker(self) -> bool:
        return self.state == R


@dataclass
class NetworkModel:
    """Drinking-behaviour ABM on a Watts-Strogatz small-world network.

    Parameters
    ----------
    n_agents:  number of nodes / agents
    k:         each node connects to k nearest neighbours in the ring lattice
               before rewiring (must be even)
    p_rewire:  Watts-Strogatz rewiring probability; 0 = regular ring,
               1 = random graph, ~0.1 gives the small-world regime
    gamma:     baseline quit probability (no former drinkers in neighbourhood)
    rho:       baseline relapse probability (no drinkers in neighbourhood)
    seed:      random seed for reproducibility
    """

    n_agents: int = 100
    k: int = 4
    p_rewire: float = 0.1
    gamma: float = 0.3
    rho: float = 0.3
    seed: int | None = None

    graph: nx.Graph = field(init=False, repr=False)
    agents: list[Agent] = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    _adj: np.ndarray = field(init=False, repr=False)
    _degree: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if not 1 <= self.k < self.n_agents:
            raise ValueError("k must be in [1, n_agents)")
        if not 0.0 <= self.p_rewire <= 1.0:
            raise ValueError("p_rewire must be in [0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")

        self.rng = np.random.default_rng(self.seed)
        nx_seed = int(self.rng.integers(0, 2**31))
        self.graph = nx.watts_strogatz_graph(
            self.n_agents, self.k, self.p_rewire, seed=nx_seed
        )

        self._adj = nx.to_numpy_array(self.graph)
        self._degree = self._adj.sum(axis=1)

        self.agents = [Agent(node_id=i, state=S) for i in range(self.n_agents)]
        n_initial_drinkers = max(1, round(0.2 * self.n_agents))
        for i in self.rng.choice(self.n_agents, size=n_initial_drinkers, replace=False):
            self.agents[i].state = D

    def _states(self) -> np.ndarray:
        return np.array([a.state for a in self.agents], dtype=np.int64)

    def fractions(self) -> np.ndarray:
        """Population fractions [S, D, R]."""
        return np.bincount(self._states(), minlength=3) / self.n_agents

    def _transition(self):
        """Convert agents synchronously based on neighbourhood composition."""
        states = self._states()
        d_vec = (states == D).astype(float)
        r_vec = (states == R).astype(float)

        d_frac = np.where(self._degree > 0, self._adj @ d_vec / self._degree, 0.0)
        r_frac = np.where(self._degree > 0, self._adj @ r_vec / self._degree, 0.0)

        u = self.rng.random(self.n_agents)
        s_mask = states == S
        d_mask = states == D
        r_mask = states == R

        new_states = states.copy()
        new_states[s_mask & (d_frac >= 0.5) & (u < d_frac)] = D
        new_states[d_mask & (u < np.minimum(r_frac + self.gamma, 1.0))] = R
        new_states[r_mask & (u < np.minimum(d_frac + self.rho, 1.0))] = D

        for agent in self.agents:
            agent.state = new_states[agent.node_id]

    def step(self):
        self._transition()

    def run(self, n_steps: int) -> dict[str, np.ndarray]:
        """Run the model and return per-iteration population fractions."""
        history = np.empty((n_steps + 1, 3))
        history[0] = self.fractions()
        for step in range(1, n_steps + 1):
            self.step()
            history[step] = self.fractions()
        return {
            "iteration": np.arange(n_steps + 1),
            "susceptible": history[:, S],
            "current_drinkers": history[:, D],
            "former_drinkers": history[:, R],
        }