"""Baseline replication of the drinking-behaviour ABM of Gorman et al. (2006).

Gorman, D. M., Mezic, J., Mezic, I., & Gruenewald, P. J. (2006). Agent-based
modeling of drinking behavior: a preliminary model and potential applications
to theory and practice. American Journal of Public Health, 96(11), 2055-2060.

Agents on a one-dimensional lattice are susceptible nondrinkers (S), current
drinkers (D), or former drinkers (R). Each iteration they convert between
states based on the composition of their own site (``_transition``) and then
perform a random walk (``_move``). A bar site, when present, biases the
movement of current drinkers toward it.

Assumptions where the paper is silent (to be reviewed by the group):

* Transitions are synchronous: all agents convert based on the site counts
  at the start of the iteration, then everyone moves.
* Lattice boundaries are closed: a step that would leave the lattice is not
  taken.
* Transition probabilities are capped at 1 (e.g. d/t + rho can exceed 1).
* A current drinker already at the bar treats "toward the bar" as staying,
  and "away from the bar" as a step in a uniformly random direction.
"""

from dataclasses import dataclass, field

import numpy as np

# Agent states
S, D, R = 0, 1, 2


@dataclass
class GormanModel:
    """Gorman et al. (2006) drinking-behaviour model on a 1D lattice."""

    n_sites: int = 100
    p_move: float = 0.1
    gamma: float = 0.3
    rho: float = 0.3
    bar_site: int | None = None
    bar_toward: float = 0.5
    bar_away: float = 0.1
    seed: int | None = None

    positions: np.ndarray = field(init=False, repr=False)
    states: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not 0.0 <= self.p_move <= 0.5:
            raise ValueError("p_move must be in [0, 0.5]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if not 0.0 <= self.bar_toward <= 1.0:
            raise ValueError("bar_toward must be in [0, 1]")
        if not 0.0 <= self.bar_away <= 1.0:
            raise ValueError("bar_away must be in [0, 1]")
        if self.bar_toward + self.bar_away > 1.0:
            raise ValueError("bar_toward + bar_away must not exceed 1")
        if self.bar_site is not None and not 0 <= self.bar_site < self.n_sites:
            raise ValueError("bar_site must be a site on the lattice")
        self.rng = np.random.default_rng(self.seed)
        # One susceptible per site plus one current drinker at the middle.
        self.positions = np.concatenate(
            [np.arange(self.n_sites), [self.n_sites // 2]]
        )
        self.states = np.concatenate(
            [np.full(self.n_sites, S), [D]]
        ).astype(np.int64)

    @property
    def n_agents(self) -> int:
        return len(self.states)

    def site_counts(self) -> np.ndarray:
        """Counts per site and state, shape (3, n_sites): rows are S, D, R."""
        counts = np.zeros((3, self.n_sites), dtype=np.int64)
        for state in (S, D, R):
            counts[state] = np.bincount(
                self.positions[self.states == state], minlength=self.n_sites
            )
        return counts

    def fractions(self) -> np.ndarray:
        """Population fractions [S, D, R]."""
        return np.bincount(self.states, minlength=3) / self.n_agents

    def drinker_distribution(self) -> np.ndarray:
        """Fraction of the total population located at each site as current
        drinkers (D at site i / total population, not the drinker share
        within a site).

        This is the quantity plotted in Figure 4 of the paper (clustering of
        drinkers around the bar).
        """
        return self.site_counts()[D] / self.n_agents

    def step(self):
        self._transition()
        self._move()

    def _transition(self):
        """Convert agents based on their site's composition (paper Figure 1).

        With s(i), d(i), r(i) the counts at site i and t(i) their sum::

            P(S -> D) = d(i) / t(i)
            P(D -> R) = r(i) / t(i) + gamma
            P(R -> D) = d(i) / t(i) + rho

        gamma: probability a drinker quits with no former drinkers present;
        rho: probability a former drinker relapses with no drinkers present.
        The paper uses gamma = rho = 0.3.
        """
        counts = self.site_counts()
        t = counts.sum(axis=0)
        # Per-site conversion probabilities, looked up per agent below.
        # Sites with t = 0 hold no agents, so the lookup never uses them;
        # np.divide keeps those entries at 0 instead of dividing by zero.
        d_frac = np.divide(counts[D], t, out=np.zeros(self.n_sites), where=t > 0)
        r_frac = np.divide(counts[R], t, out=np.zeros(self.n_sites), where=t > 0)
        p_convert = np.zeros(self.n_agents)
        s_mask, d_mask, r_mask = (self.states == S, self.states == D,
                                  self.states == R)
        p_convert[s_mask] = d_frac[self.positions[s_mask]]
        p_convert[d_mask] = r_frac[self.positions[d_mask]] + self.gamma
        p_convert[r_mask] = d_frac[self.positions[r_mask]] + self.rho
        converts = self.rng.random(self.n_agents) < np.minimum(p_convert, 1.0)
        new_states = self.states.copy()
        new_states[s_mask & converts] = D
        new_states[d_mask & converts] = R
        new_states[r_mask & converts] = D
        self.states = new_states

    def _move(self):
        """Random walk: left/right with probability p_move each, else stay.

        Current drinkers with a bar present instead move toward it with
        probability bar_toward and away with bar_away (paper example:
        0.5 / 0.1 / 0.4). Susceptibles and former drinkers ignore the bar.
        """
        u = self.rng.random(self.n_agents)
        # Unbiased random walk: left with p_move, right with p_move.
        steps = np.where(u < self.p_move, -1,
                         np.where(u < 2 * self.p_move, 1, 0))
        if self.bar_site is not None:
            drinkers = self.states == D
            toward = np.sign(self.bar_site - self.positions)
            bar_steps = np.where(
                u < self.bar_toward, toward,
                np.where(u < self.bar_toward + self.bar_away, -toward, 0),
            )
            # At the bar itself toward = 0: "toward" becomes staying and
            # "away" a step in a random direction.
            at_bar = self.positions == self.bar_site
            away_dir = self.rng.choice([-1, 1], size=self.n_agents)
            bar_steps = np.where(
                at_bar & (u >= self.bar_toward)
                & (u < self.bar_toward + self.bar_away),
                away_dir, bar_steps,
            )
            steps = np.where(drinkers, bar_steps, steps)
        # Closed boundaries: a step off the lattice is not taken.
        self.positions = np.clip(self.positions + steps, 0, self.n_sites - 1)

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
