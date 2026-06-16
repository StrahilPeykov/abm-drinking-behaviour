"""Small-world network extension of the Gorman et al. (2006) drinking-behaviour ABM.

Agents are nodes on a Watts-Strogatz small-world graph. Each iteration they
convert between states based on the composition of their neighbourhood
(``_transition``). There is no spatial movement; the network topology encodes
social structure.

Transition rules mirror the lattice model, replacing site co-occupants with
graph neighbours::

    P(S -> D) = d_frac  if d_frac >= agent.risk_aversion_start, else 0
    P(D -> R) = r_frac + gamma
    P(R -> D) = d_frac + rho  if d_frac >= agent.risk_aversion_relapse, else 0

where d_frac and r_frac are the fractions of current drinkers and former
drinkers among the agent's neighbours. Each agent's risk_aversion thresholds
are drawn independently from Uniform(0, 1) at initialisation.

Isolated nodes (degree 0) face zero social influence; D agents still quit
at rate gamma and R agents are shielded from relapse (threshold never met).
Transitions are synchronous: all agents convert based on neighbourhood
composition at the start of the iteration.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

S, D, R = 0, 1, 2


@dataclass
class Agent:
    """A single agent node on the social network.

    Attributes
    ----------
    node_id:               index of the corresponding node in the graph
    risk_aversion_start:   personal threshold for d_frac required before a
                           susceptible agent can start drinking; drawn from
                           Uniform(0, 1)
    risk_aversion_relapse: personal threshold for d_frac required before a
                           former drinker can relapse; drawn from
                           Uniform(0, 1)
    state:                 current drinking state - S (susceptible),
                           D (current drinker), or R (former drinker)
    """
    node_id: int
    risk_aversion_start: float = 0.0
    risk_aversion_relapse: float = 0.0
    state: int = S

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
    gamma:                baseline quit probability (no former drinkers in neighbourhood)
    rho:                  baseline relapse probability (no drinkers in neighbourhood)
    initial_drinker_frac: fraction of agents initialised as current drinkers,
                          chosen randomly across the network
    seed:                 random seed for reproducibility
    """

    n_agents: int = 100
    k: int = 4
    p_rewire: float = 0.1
    gamma: float = 0.3
    rho: float = 0.3
    initial_drinker_frac: float = 0.1
    seed: int | None = None

    graph: nx.Graph = field(init=False, repr=False)
    agents: list[Agent] = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

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
        if not 0.0 <= self.initial_drinker_frac <= 1.0:
            raise ValueError("initial_drinker_frac must be in [0, 1]")

        self.rng = np.random.default_rng(self.seed)
        nx_seed = int(self.rng.integers(0, 2**31))
        self.graph = nx.watts_strogatz_graph(
            self.n_agents, self.k, self.p_rewire, seed=nx_seed
        )

        self.agents = [Agent(node_id=i, state=S) for i in range(self.n_agents)]
        n_initial_drinkers = max(1, round(self.initial_drinker_frac * self.n_agents))
        for i in self.rng.choice(self.n_agents, size=n_initial_drinkers, replace=False):
            self.agents[i].state = D

        start_draws = self.rng.uniform(0.0, 1.0, size=self.n_agents)
        relapse_draws = self.rng.uniform(0.0, 1.0, size=self.n_agents)

        for i, agent in enumerate(self.agents):
            agent.risk_aversion_start = start_draws[i]
            agent.risk_aversion_relapse = relapse_draws[i]

    def _states(self) -> np.ndarray:
        return np.array([a.state for a in self.agents], dtype=np.int64)

    def fractions(self) -> np.ndarray:
        """Population fractions [S, D, R]."""
        return np.bincount(self._states(), minlength=3) / self.n_agents

    def _transition(self):
        """Convert agents synchronously based on neighbourhood composition."""
        new_states = [None] * self.n_agents

        for agent in self.agents:
            neighbors = list(self.graph.neighbors(agent.node_id))
            neighbor_total = len(neighbors)

            if neighbor_total == 0:
                d_frac = 0.0
                r_frac = 0.0
            else:
                neighbor_states = [self.agents[i].state for i in neighbors]
                d_frac = neighbor_states.count(D) / neighbor_total
                r_frac = neighbor_states.count(R) / neighbor_total

            draw = self.rng.random()

            if agent.state == S:
                if d_frac >= agent.risk_aversion_start and draw < d_frac:
                    new_states[agent.node_id] = D
                else: 
                    new_states[agent.node_id] = S
            elif agent.state == D: 
                probability = min(r_frac + self.gamma, 1.0)
                if draw < probability:
                    new_states[agent.node_id] = R
                else: 
                    new_states[agent.node_id] = D
            elif agent.state == R:
                if d_frac >= agent.risk_aversion_relapse:
                    probability = min(d_frac + self.rho, 1.0)
                    if draw < probability:
                        new_states[agent.node_id] = D
                    else:
                        new_states[agent.node_id] = R
                else:
                    new_states[agent.node_id] = R

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