"""
Sanity tests. Run with `python -m pytest tests -q` from the project root.

These are the checks that catch the errors that would otherwise show up as a
suspiciously good backtest: a gradient that does not match the objective, a
probability grid that does not sum to one, a tau correction with the wrong sign.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dixon_coles as dc  # noqa: E402
from data_io import log_loss, ranked_probability_score  # noqa: E402


# --------------------------------------------------------------------------- #
def test_tau_is_identity_when_rho_zero():
    goals = np.arange(4)
    t = dc.tau(goals[:, None], goals[None, :], 1.5, 1.1, 0.0)
    assert np.allclose(t, 1.0)


def test_tau_signs_match_the_paper():
    """rho < 0 must push 0-0 and 1-1 up, and 1-0 / 0-1 down."""
    lam, mu, rho = 1.5, 1.1, -0.1
    assert dc.tau(0, 0, lam, mu, rho) > 1
    assert dc.tau(1, 1, lam, mu, rho) > 1
    assert dc.tau(1, 0, lam, mu, rho) < 1
    assert dc.tau(0, 1, lam, mu, rho) < 1
    assert dc.tau(2, 3, lam, mu, rho) == 1


def test_score_grid_sums_to_one_and_matches_outcomes():
    """The mechanical step from step 2: two lambdas in, three probabilities out."""
    model = dc.DixonColes(
        teams=["A", "B"],
        attack=np.array([0.0, 0.0]),
        defence=np.array([0.0, 0.0]),
        home_advantage=np.log(1.5 / 1.1) / 1,   # so that lam=1.5, mu=1.1 below
        level=np.log(1.1),
        rho=-0.1, xi=0.0, prior_sd=1.0,
    )
    lam, mu = model.expected_goals(0, 1)
    assert np.isclose(float(lam), 1.5) and np.isclose(float(mu), 1.1)

    grid = model.score_matrix(0, 1)
    assert np.isclose(grid.sum(), 1.0)

    probs = model.outcome_probs(0, 1)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    assert (probs > 0).all()
    # home side has the bigger lambda, so it must be the favourite
    assert probs[0] > probs[2]


def test_predict_matches_single_fixture_path():
    rng = np.random.default_rng(0)
    n = 6
    model = dc.DixonColes(
        teams=[f"T{i}" for i in range(n)],
        attack=rng.normal(0, 0.3, n),
        defence=rng.normal(0, 0.3, n),
        home_advantage=0.26, level=0.1, rho=-0.12, xi=0.0, prior_sd=1.0,
    )
    home = np.array([0, 1, 2, 3])
    away = np.array([4, 5, 0, 1])
    batch = model.predict(home, away)
    single = np.array([model.outcome_probs(h, a) for h, a in zip(home, away)])
    assert np.allclose(batch, single, atol=1e-12)
    assert np.allclose(batch.sum(axis=1), 1.0)


def test_analytic_gradient_matches_finite_differences():
    """If this fails, every parameter estimate downstream is suspect."""
    rng = np.random.default_rng(42)
    n_teams, n_matches = 5, 200
    home = rng.integers(0, n_teams, n_matches).astype(np.intp)
    away = (home + 1 + rng.integers(0, n_teams - 1, n_matches)) % n_teams
    hg = rng.poisson(1.5, n_matches).astype(float)
    ag = rng.poisson(1.2, n_matches).astype(float)
    w = rng.uniform(0.2, 1.0, n_matches)

    theta = rng.normal(0, 0.3, 2 * n_teams + 3)
    theta[-1] = -0.08
    args = (home, away.astype(np.intp), hg, ag, w, n_teams, 1.0 / 0.35 ** 2)

    _, grad = dc._objective(theta, *args)

    eps = 1e-6
    numeric = np.zeros_like(theta)
    for i in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[i] += eps
        down[i] -= eps
        numeric[i] = (dc._objective(up, *args)[0] - dc._objective(down, *args)[0]) / (2 * eps)

    assert np.allclose(grad, numeric, rtol=1e-5, atol=1e-5)


def test_time_weights_decay():
    dates = np.array(["2024-01-01", "2025-01-01"], dtype="datetime64[D]")
    w = dc.time_weights(dates, np.datetime64("2025-01-01"), xi=np.log(2) / 366)
    assert np.isclose(w[1], 1.0)
    assert np.isclose(w[0], 0.5, atol=1e-2)


def test_fit_recovers_known_parameters():
    """Simulate from the model, refit, and check we get the truth back."""
    rng = np.random.default_rng(7)
    n_teams = 20
    true_attack = rng.normal(0, 0.30, n_teams)
    true_attack -= true_attack.mean()
    true_defence = rng.normal(0, 0.25, n_teams)
    true_defence -= true_defence.mean()
    home_adv, level, rho = 0.26, 0.15, -0.10

    home, away = [], []
    for i in range(n_teams):
        for j in range(n_teams):
            if i != j:
                for _ in range(12):          # a big synthetic sample
                    home.append(i)
                    away.append(j)
    home = np.array(home, dtype=np.intp)
    away = np.array(away, dtype=np.intp)

    truth = dc.DixonColes(
        teams=[f"T{i}" for i in range(n_teams)],
        attack=true_attack, defence=true_defence,
        home_advantage=home_adv, level=level, rho=rho, xi=0.0, prior_sd=1.0,
    )
    hg, ag = truth.sample_scores(home, away, rng=rng)
    hg = hg.astype(float)
    ag = ag.astype(float)

    model = dc.fit(home, away, hg, ag, [f"T{i}" for i in range(n_teams)],
                   prior_sd=5.0)          # weak prior: let the data speak

    fitted_attack = model.attack - model.attack.mean()
    fitted_defence = model.defence - model.defence.mean()
    assert np.corrcoef(fitted_attack, true_attack)[0, 1] > 0.97
    assert np.corrcoef(fitted_defence, true_defence)[0, 1] > 0.97
    assert abs(model.home_advantage - home_adv) < 0.05
    assert abs(model.rho - rho) < 0.03


def test_log_loss_and_rps_behave():
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    assert log_loss(probs, ["H", "A"]) < log_loss(probs, ["A", "H"])
    # RPS knows H/D/A is ordered: a draw is a nearer miss than an away win
    p = np.array([[0.7, 0.2, 0.1]])
    assert ranked_probability_score(p, ["D"]) < ranked_probability_score(p, ["A"])

    uniform = np.full((100, 3), 1 / 3)
    assert np.isclose(log_loss(uniform, ["H"] * 100), -np.log(1 / 3))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
