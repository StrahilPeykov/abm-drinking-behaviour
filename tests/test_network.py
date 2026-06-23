"""Tests for the social-network extension model (src/network.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import (
    NetworkModel, Agent, S, D, R,
    _logit_probability, _risk_averse_value, _loss_averse_utility,
)


# --- decision-rule unit tests -------------------------------------------------

def test_logit_monotonic_and_bounded():
    """Higher utility difference -> higher drink probability, always in (0, 1)."""
    ps = [_logit_probability(du, tau=0.5) for du in [-5, -1, 0, 1, 5]]
    assert all(0.0 < p < 1.0 for p in ps)
    assert ps == sorted(ps)
    assert _logit_probability(0.0, 0.5) == pytest.approx(0.5)


def test_low_tau_approaches_best_response():
    """As tau -> 0 the logit becomes a deterministic best response."""
    assert _logit_probability(1.0, tau=1e-6) == pytest.approx(1.0, abs=1e-3)
    assert _logit_probability(-1.0, tau=1e-6) == pytest.approx(0.0, abs=1e-3)


def test_risk_neutral_value_is_linear():
    """kappa = 0 recovers the identity (linear) value function."""
    for x in [0.1, 0.5, 0.9]:
        assert _risk_averse_value(x, kappa=0.0) == pytest.approx(x)


def test_loss_aversion_penalises_unsupported_drinking():
    """With lambda > 1, drinking against the neighbourhood has negative utility."""
    assert _loss_averse_utility(d_frac=0.1, lam=2.0) < 0
    assert _loss_averse_utility(d_frac=0.9, lam=2.0) > 0


# --- model-level tests --------------------------------------------------------

def test_population_conserved_and_fractions_sum_to_one():
    model = NetworkModel(n_agents=80, seed=0)
    history = model.run(120)
    assert model.n_agents == 80
    total = (history["susceptible"] + history["current_drinkers"]
             + history["former_drinkers"])
    assert np.allclose(total, 1.0)


def test_reproducible_with_seed():
    a = NetworkModel(seed=7).run(100)["current_drinkers"]
    b = NetworkModel(seed=7).run(100)["current_drinkers"]
    assert np.array_equal(a, b)


def test_ages_within_window():
    """Ages are clipped to the modelled 16-24 young-adult window."""
    ages = np.array([a.age for a in NetworkModel(seed=1).agents])
    assert ages.min() >= 16.0 and ages.max() <= 24.0


def test_age_susceptibility_decreases_with_age():
    """Younger agents are more susceptible to peer influence."""
    assert Agent(0, age=16).age_susceptibility > Agent(0, age=24).age_susceptibility


def test_habituation_off_is_noop():
    """habituation_rate = 0 must reproduce the pre-habituation dynamics exactly."""
    base = NetworkModel(habituation_rate=0.0, seed=5).run(200)["current_drinkers"]
    # exp(-0 * tenure) == 1, so any tenure leaves the quit rate unchanged
    again = NetworkModel(habituation_rate=0.0, seed=5).run(200)["current_drinkers"]
    assert np.array_equal(base, again)


def test_habituation_increases_drinking_persistence():
    """Stronger habituation -> more drinkers persist at steady state."""
    def final_d(h):
        seeds = range(5)
        return np.mean([
            NetworkModel(habituation_rate=h, kappa_mean=0.5, seed=s)
            .run(300)["current_drinkers"][-1] for s in seeds])
    assert final_d(0.5) > final_d(0.0)


@pytest.mark.parametrize("kwargs", [
    {"n_agents": 0}, {"k": 0}, {"p_rewire": 1.5}, {"gamma": -0.1},
    {"rho": 2.0}, {"tau": 0.0}, {"habituation_rate": -1.0},
    {"initial_drinker_frac": 1.5},
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        NetworkModel(**kwargs)
