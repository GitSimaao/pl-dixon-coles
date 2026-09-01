"""
Walk-forward backtesting.

The one rule
------------
To predict a match played on date D, the model may only see matches played
strictly before D. No exceptions. If you fit on everything and then "predict"
the past, the numbers come out beautiful and mean nothing -- that is lookahead
bias, and it is what makes most amateur backtests worthless.

Here that rule is enforced structurally: matches are sorted by date once, and
the training slice is taken with `searchsorted`, so it is impossible for a
future match to leak into a fit.

Cost
----
The model is refitted on every match date -- about 1,200 fits per pass through
the data. That is affordable because `dixon_coles._objective` supplies an
analytic gradient and each fit is warm-started from the previous one, so a
refit costs a few milliseconds rather than a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import dixon_coles as dc
from data_io import accuracy, log_loss, ranked_probability_score, result_index

PROB_COLS = ["p_H", "p_D", "p_A"]


# --------------------------------------------------------------------------- #
def walk_forward(
    matches: pd.DataFrame,
    teams: list[str],
    xi: float = 0.0018,
    prior_sd: float = 0.35,
    start_date=None,
    end_date=None,
    min_train_matches: int = 100,
    warm_start: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Produce a genuinely out-of-sample probability for every match in the window.

    Returns the input rows in the window plus p_H / p_D / p_A and the two
    expected-goal numbers the model used.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    dates = matches["Date"].to_numpy(dtype="datetime64[D]")

    start = dates[0] if start_date is None else np.datetime64(pd.Timestamp(start_date), "D")
    end = dates[-1] if end_date is None else np.datetime64(pd.Timestamp(end_date), "D")

    match_dates = np.unique(dates[(dates >= start) & (dates <= end)])

    home_id = matches["home_id"].to_numpy(np.intp)
    away_id = matches["away_id"].to_numpy(np.intp)
    hg = matches["FTHG"].to_numpy(float)
    ag = matches["FTAG"].to_numpy(float)

    rows, theta = [], None

    for day in match_dates:
        cut = int(np.searchsorted(dates, day, side="left"))   # matches strictly before `day`
        if cut < min_train_matches:
            continue

        model = dc.fit(
            home_id[:cut], away_id[:cut], hg[:cut], ag[:cut], teams,
            match_dates=dates[:cut], ref_date=day,
            xi=xi, prior_sd=prior_sd,
            init=theta if warm_start else None,
        )
        if warm_start:
            theta = model.theta

        today = np.flatnonzero(dates == day)
        probs = model.predict(home_id[today], away_id[today])
        lam, mu = model.expected_goals(home_id[today], away_id[today])

        block = matches.iloc[today].copy()
        block[PROB_COLS] = probs
        block["xg_home"] = lam
        block["xg_away"] = mu
        block["train_n"] = cut
        block["train_effective_n"] = model.effective_n
        block["gamma"] = model.gamma
        block["rho"] = model.rho
        rows.append(block)

        if verbose and len(rows) % 100 == 0:
            print(f"  ...{day} ({len(rows)} match days done)")

    if not rows:
        raise RuntimeError("Nothing was predicted -- check the date window.")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
def evaluate(preds: pd.DataFrame, prob_cols=PROB_COLS, odds_only: bool = True) -> dict:
    """Score a set of predictions. `odds_only` restricts to matches the market priced."""
    df = preds[preds["has_odds"]] if odds_only else preds
    p = df[prob_cols].to_numpy(float)
    return {
        "n": int(len(df)),
        "log_loss": log_loss(p, df["FTR"]),
        "rps": ranked_probability_score(p, df["FTR"]),
        "accuracy": accuracy(p, df["FTR"]),
    }


def tune(
    matches: pd.DataFrame,
    teams: list[str],
    xi_grid,
    prior_sd_grid,
    start_date,
    end_date,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid-search the two hyperparameters on a *validation* window.

    Time decay (`xi`) and prior strength (`prior_sd`) are the only two knobs.
    They are chosen on seasons that are then never used to report performance,
    because picking a hyperparameter on the same data you report is just a
    slower form of lookahead bias.
    """
    results = []
    for xi in xi_grid:
        for prior_sd in prior_sd_grid:
            preds = walk_forward(matches, teams, xi=xi, prior_sd=prior_sd,
                                 start_date=start_date, end_date=end_date)
            scores = evaluate(preds)
            half_life = np.inf if xi == 0 else np.log(2) / xi
            results.append({"xi": xi, "half_life_days": half_life,
                            "prior_sd": prior_sd, **scores})
            if verbose:
                print(f"  xi={xi:<8.5f} half-life={half_life:7.0f}d  "
                      f"prior_sd={prior_sd:<5.2f}  log_loss={scores['log_loss']:.5f}")
    return pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Combining the model with the market
# --------------------------------------------------------------------------- #
def blend(model_probs: np.ndarray, market_probs: np.ndarray, weight: float) -> np.ndarray:
    """
    Geometric (log-space) pool of the two forecasts, renormalised.

    `weight` is the share given to the model. weight=0 is the market alone.
    A blend that beats the market is the honest way to say "the model contains
    information the market does not", even when the model on its own is worse.
    """
    logs = weight * np.log(np.clip(model_probs, 1e-12, 1)) + \
        (1 - weight) * np.log(np.clip(market_probs, 1e-12, 1))
    out = np.exp(logs)
    return out / out.sum(axis=1, keepdims=True)


def best_blend_weight(preds: pd.DataFrame, grid=None) -> tuple[float, float]:
    """Pick the blend weight that minimises log loss on the given set."""
    grid = np.linspace(0, 1, 51) if grid is None else np.asarray(grid)
    df = preds[preds["has_odds"]]
    model_p = df[PROB_COLS].to_numpy(float)
    market_p = df[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(float)
    losses = [log_loss(blend(model_p, market_p, w), df["FTR"]) for w in grid]
    best = int(np.argmin(losses))
    return float(grid[best]), float(losses[best])


# --------------------------------------------------------------------------- #
#  Diagnostics
# --------------------------------------------------------------------------- #
def by_season(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-season log loss for the model, the market and the 1/3 baseline."""
    rows = []
    for season, block in preds.groupby("Season"):
        graded = block[block["has_odds"]]
        if graded.empty:        # seasons before the benchmark exists train only
            continue
        row = {
            "season": season,
            "n": int(len(graded)),
            "model": log_loss(graded[PROB_COLS].to_numpy(float), graded["FTR"]),
            "market": log_loss(graded[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(float),
                               graded["FTR"]),
        }
        row["gap"] = row["model"] - row["market"]
        rows.append(row)
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame, prob_cols=PROB_COLS, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability table: of the matches where we said ~30%, did ~30% happen?

    All three outcomes are pooled into one set of (predicted, observed) pairs,
    which is the usual way to read calibration for a 1X2 forecast.
    """
    df = preds[preds["has_odds"]]
    p = df[prob_cols].to_numpy(float).ravel()

    obs = np.zeros((len(df), 3))
    obs[np.arange(len(df)), result_index(df["FTR"])] = 1.0
    obs = obs.ravel()

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin_low": edges[b], "bin_high": edges[b + 1],
            "n": int(sel.sum()),
            "predicted": float(p[sel].mean()),
            "observed": float(obs[sel].mean()),
        })
    return pd.DataFrame(rows)


def betting_simulation(preds: pd.DataFrame, edge_threshold: float = 0.02,
                       kelly_fraction: float = 0.25, stake_cap: float = 0.05) -> dict:
    """
    A deliberately pessimistic staking check against the CLOSING price.

    Read this as a diagnostic, not a P&L forecast. Beating a closing line in a
    backtest is not the same as getting that price in a live market: closing
    odds are the sharpest number of the day, you would have had to bet earlier
    at worse prices, and a bettor who consistently beat the close would be
    limited long before the sample ended.
    """
    from data_io import ODDS_COLS
    df = preds[preds["has_odds"]].copy()
    model_p = df[PROB_COLS].to_numpy(float)
    odds = df[ODDS_COLS].to_numpy(float)

    ev = model_p * odds - 1.0                     # expected profit per unit staked
    outcome = result_index(df["FTR"])

    bankroll, n_bets, staked, pnl_curve = 1.0, 0, 0.0, []
    for i in range(len(df)):
        for k in range(3):
            if ev[i, k] <= edge_threshold:
                continue
            b = odds[i, k] - 1.0
            kelly = max(0.0, (model_p[i, k] * odds[i, k] - 1.0) / b)
            stake = bankroll * min(kelly_fraction * kelly, stake_cap)
            if stake <= 0:
                continue
            n_bets += 1
            staked += stake
            bankroll += stake * b if outcome[i] == k else -stake
        pnl_curve.append(bankroll)

    return {
        "n_bets": n_bets,
        "total_staked": staked,
        "final_bankroll": bankroll,
        "roi": (bankroll - 1.0) / staked if staked > 0 else 0.0,
        "curve": np.array(pnl_curve),
    }
