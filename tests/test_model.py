"""Tests for the Gorman et al. (2006) lattice baseline model (src/model.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import GormanModel, S, D, R


def test_initial_conditions():
    """One susceptible per site plus a single drinker at the middle site."""
    model = GormanModel(n_sites=50, seed=0)
    counts = model.site_counts()
    assert model.n_agents == 51
    assert counts[D].sum() == 1
    assert counts[D, 25] == 1  # drinker at the middle site
    assert counts[S].sum() == 50


def test_population_conserved():
    """The total number of agents never changes; only states/positions do."""
    model = GormanModel(n_sites=40, seed=1)
    start = model.n_agents
    model.run(200)
    assert model.n_agents == start


def test_fractions_sum_to_one():
    model = GormanModel(seed=2)
    history = model.run(100)
    total = (history["susceptible"] + history["current_drinkers"]
             + history["former_drinkers"])
    assert np.allclose(total, 1.0)


def test_reproducible_with_seed():
    a = GormanModel(seed=42).run(150)["susceptible"]
    b = GormanModel(seed=42).run(150)["susceptible"]
    assert np.array_equal(a, b)


def test_susceptibles_deplete_baseline():
    """Gorman's universal result: for p, gamma, rho > 0 susceptibles vanish."""
    final_s = GormanModel(p_move=0.1, seed=3).run(2000)["susceptible"][-1]
    assert final_s == 0.0


@pytest.mark.parametrize("kwargs", [
    {"p_move": 0.6}, {"p_move": -0.1}, {"gamma": 1.5}, {"rho": -0.2},
    {"n_sites": 0}, {"bar_site": 999}, {"bar_toward": 0.7, "bar_away": 0.7},
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        GormanModel(**kwargs)
