"""Small-world network extension of the Gorman et al. (2006) drinking-behaviour ABM.

Agents are nodes on a Watts-Strogatz small-world graph. Each iteration they
convert between states based on the composition of their neighbourhood
(``_transition``). There is no spatial movement; the network topology encodes
social structure.

Transition rules mirror the lattice model, replacing site co-occupants with
graph neighbours::

    P(S -> D) = d_frac  if d_frac >= agent.risk_aversion_start, else 0 -> start drinking if social pressure higher than own psychological barrier
    P(D -> R) = r_frac + gamma
    P(R -> D) = d_frac + rho  if d_frac >= agent.risk_aversion_relapse, else rho

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
    #risk_aversion_start: float = 0.0
    #risk_aversion_relapse: float = 0.0
    lam_start: float = 2.0
    lam_relapse: float = 2.0
    state: int = S

def _logit_probability(delta_u: float, tau: float) -> float:
    """Convert a utility difference into a choice probability via the
    logit / quantal-response rule (McKelvey & Palfrey, 1995).
 
    P(drink) = 1 / (1 + exp(-delta_u / tau))
 
    tau controls bounded rationality:
      - tau -> 0: approaches a deterministic best response (drink if delta_u > 0)
      - tau -> infinity: approaches a coin flip (0.5), no matter how favourable delta_u is"""
    tau = max(tau, 1e-9) #avoid division by zero
    z = np.clip(-delta_u / tau, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))
    


"""def _loss_averse_utility(d_frac: float, lam: float) -> float:
    Utility difference U(drink) - U(abstain) under a loss-averse value
    function (Kahneman & Tversky, 1979).
 
    d_frac -> gain term: reward for matching a drinking-heavy neighbourhood (social reinforcement)
    (1 - d_frac) -> loss term: downside of drinking when the neighbourhood does NOT support it,
                                   weighted lambda times more heavily than the equivalent gain
                                   
    Simplified from the traditional prospect theory function: 
    the full prospect-theory value is V = Σ w(p_i) · v(x_i) - value function v(x) and probability weighting function w(p). 
    As it would expand our scope even more.

    delta_u = d_frac - lam * (1.0 - d_frac)
    return delta_u"""

#loss and risk aversion

def _risk_averse_value(x: float, rho: float) -> float:
    """Concave value function capturing risk aversion.
    rho = 0   -> risk-neutral (linear, recovers the current behaviour)
    rho > 0   -> risk-averse (concave, diminishing marginal value)
    rho < 0   -> risk-seeking (convex)
    """
    eps = 1e-6 #epsilon to avoid division by 0
    x_safe = max(x, eps)
    if abs(rho) < 1e-9:
        return x_safe
    return (x_safe ** (1 - rho)) / (1 - rho)

def _loss_averse_utility(d_frac: float, lam: float, rho: float = 0.0) -> float:
    gain_term = _risk_averse_value(d_frac, rho)
    loss_term = _risk_averse_value(1.0 - d_frac, rho)
    return gain_term - lam * loss_term

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
    tau:                  rationality temperature for the logit choice rule
                          (bounded rationality, shared across the population).
                          lower tau = more rational/deterministic; higher tau = noisier/more random decisions. 
    lam_mean, lam_sd:     mean and standard deviation of the per-agent loss-aversion coefficients, 
                          drawn from a Normal and clipped at a small positive floor so lam never goes non-positive. 
                          lam_mean defaults to 2.0 (commonly-cited empirical ratio (losses feel ~2x as strong
                          as equivalent gains).
    initial_drinker_frac: fraction of agents initialised as current drinkers,
                          chosen randomly across the network
    seed:                 random seed for reproducibility
    """

    n_agents: int = 100
    k: int = 4
    p_rewire: float = 0.1
    gamma: float = 0.3
    rho: float = 0.3
    tau: float = 0.5
    lam_mean: float = 2.0
    lam_sd: float = 0.5
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
        
        #new from Game Theory:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive (use a small value, not 0, for near-deterministic behaviour)")
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

        # draw per-agent loss-aversion coefficients from a Normal but cclipped at a small positive floor,
        # instead of the old Uniform(0, 1) thresholds 
        # this preserves population heterogeneity in risk attitude and givs lambda interpetaiton as in prospect theory
        lam_start_draws = self.rng.normal(self.lam_mean, self.lam_sd, size=self.n_agents)
        lam_relapse_draws = self.rng.normal(self.lam_mean, self.lam_sd, size=self.n_agents)
        lam_floor = 1e-3
        lam_start_draws = np.clip(lam_start_draws, lam_floor, None)
        lam_relapse_draws = np.clip(lam_relapse_draws, lam_floor, None)

        for i, agent in enumerate(self.agents):
            agent.lam_start = float(lam_start_draws[i])
            agent.lam_relapse = float(lam_relapse_draws[i])

    def _states(self) -> np.ndarray:
        return np.array([a.state for a in self.agents], dtype=np.int64)

    def fractions(self) -> np.ndarray:
        """Population fractions [S, D, R]."""
        return np.bincount(self._states(), minlength=3) / self.n_agents

    def _transition(self):
        """Convert agents synchronously based on neighbourhood composition.
        S -> D and R -> D now go through the loss-averse-utility + logit
        pipeline. 
        D -> R is the same as from the Gorman rule. """
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
                # Loss-averse utility
                delta_u = _loss_averse_utility(d_frac, agent.lam_start)
                # Bounded rationality 
                p_drink = _logit_probability(delta_u, self.tau)
                new_states[agent.node_id] = D if draw < p_drink else S

            elif agent.state == D:
                # same as before: quitting depends only on social encouragement
                probability = min(r_frac + self.gamma, 1.0)
                new_states[agent.node_id] = R if draw < probability else D

            elif agent.state == R:
                # Loss-averse utility
                delta_u = _loss_averse_utility(d_frac, agent.lam_relapse)
                # Bounded rationality / logit choice
                p_relapse_drive = _logit_probability(delta_u, self.tau)
                probability = min(p_relapse_drive + self.rho, 1.0)
                new_states[agent.node_id] = D if draw < probability else R

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