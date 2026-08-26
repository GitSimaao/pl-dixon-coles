"""
End-to-end pipeline: data -> market benchmark -> walk-forward backtest -> report.

    python run.py                 # full backtest with the stored hyperparameters
    python run.py --tune          # re-run the grid search first (slow, ~15 min)
    python run.py --no-figures    # skip matplotlib

Everything written to outputs/ and figures/ is reproducible from the CSVs in
data/ -- there is no hidden state.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import backtest as bt
import dixon_coles as dc
from data_io import (add_market_probabilities, load_matches, log_loss,
                     summarise_market)

# --------------------------------------------------------------------------- #
# Chosen by grid search on the VALIDATION window only (see outputs/tuning.csv).
# xi = 0.002/day is a half-life of about 347 days: a match counts half as much
# as a fresh one after roughly a year.
BEST_XI = 0.002
BEST_PRIOR_SD = 0.6

# Two full seasons of burn-in: the model needs history before it can say anything.
BURN_IN_END = "2017-07-31"
# Hyperparameters and the blend weight are picked here...
VALIDATION_END = "2021-06-30"
# ...and never touched again, so everything after this date is truly held out.

XI_GRID = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006]
PRIOR_GRID = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]


def split_periods(preds: pd.DataFrame):
    val = preds[preds["Date"] <= pd.Timestamp(VALIDATION_END)]
    test = preds[preds["Date"] > pd.Timestamp(VALIDATION_END)]
    return val, test


def market_only(preds: pd.DataFrame) -> dict:
    return bt.evaluate(preds, prob_cols=["mkt_H", "mkt_D", "mkt_A"])


def main(tune: bool = False, figures: bool = True) -> dict:
    matches = add_market_probabilities(load_matches())
    teams = matches.attrs["teams"]

    print("=" * 72)
    print("DATA")
    print("=" * 72)
    info = summarise_market(matches)
    print(f"  matches                 {info['n_matches']}")
    print(f"  seasons                 {matches['Season'].nunique()} "
          f"({matches['Season'].min()} - {matches['Season'].max()})")
    print(f"  teams                   {len(teams)}")
    print(f"  with Pinnacle closing   {info['n_with_odds']}")
    print(f"  without odds            {info['n_without_odds']}  (kept for training, "
          f"excluded from the comparison)")
    print(f"  mean overround          {info['mean_overround']:.3%}")
    print()
    print(f"  BENCHMARK  market log loss  {info['market_log_loss']:.4f}")
    print(f"             uniform 1/3      {info['uniform_log_loss']:.4f}")
    print(f"             everything known about football is worth "
          f"{info['uniform_log_loss'] - info['market_log_loss']:.4f}")

    xi, prior_sd = BEST_XI, BEST_PRIOR_SD
    if tune:
        print()
        print("=" * 72)
        print(f"TUNING  (validation window {BURN_IN_END} -> {VALIDATION_END})")
        print("=" * 72)
        grid = bt.tune(matches, teams, XI_GRID, PRIOR_GRID,
                       start_date=BURN_IN_END, end_date=VALIDATION_END)
        grid.to_csv("outputs/tuning.csv", index=False)
        xi = float(grid.iloc[0]["xi"])
        prior_sd = float(grid.iloc[0]["prior_sd"])
        print(f"\n  best: xi={xi}  prior_sd={prior_sd}")

    print()
    print("=" * 72)
    print("WALK-FORWARD BACKTEST")
    print("=" * 72)
    print(f"  xi = {xi} (half-life {np.log(2) / xi:.0f} days), prior_sd = {prior_sd}")
    print("  refitting on every match date; each fit sees only earlier matches")

    preds = bt.walk_forward(matches, teams, xi=xi, prior_sd=prior_sd,
                            start_date=BURN_IN_END)
    preds.to_csv("outputs/predictions.csv", index=False)

    val, test = split_periods(preds)

    # The blend weight is a hyperparameter too -- fit it on validation only.
    w_blend, _ = bt.best_blend_weight(val)
    for frame in (preds, val, test):
        graded = frame["has_odds"].to_numpy()
        blended = np.full((len(frame), 3), np.nan)
        blended[graded] = bt.blend(
            frame.loc[graded, bt.PROB_COLS].to_numpy(float),
            frame.loc[graded, ["mkt_H", "mkt_D", "mkt_A"]].to_numpy(float),
            w_blend,
        )
        frame[["b_H", "b_D", "b_A"]] = blended

    def block(name, frame):
        m = bt.evaluate(frame)
        k = market_only(frame)
        b = bt.evaluate(frame, prob_cols=["b_H", "b_D", "b_A"])
        print(f"\n  {name}  ({m['n']} matches with odds)")
        print(f"    {'':16s}{'log loss':>10s}{'RPS':>9s}{'accuracy':>10s}")
        print(f"    {'model':16s}{m['log_loss']:>10.4f}{m['rps']:>9.4f}{m['accuracy']:>10.1%}")
        print(f"    {'market':16s}{k['log_loss']:>10.4f}{k['rps']:>9.4f}{k['accuracy']:>10.1%}")
        print(f"    {'blend ' + f'({w_blend:.0%} model)':16s}"
              f"{b['log_loss']:>10.4f}{b['rps']:>9.4f}{b['accuracy']:>10.1%}")
        print(f"    gap to market   {m['log_loss'] - k['log_loss']:+.4f}"
              f"   (blend {b['log_loss'] - k['log_loss']:+.4f})")
        return {"model": m, "market": k, "blend": b}

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    scores = {
        "all": block("ALL (2017/18 - 2025/26)", preds),
        "validation": block("VALIDATION (used to pick xi, prior_sd, blend weight)", val),
        "test": block("TEST (never used for any choice)", test),
    }

    season_table = bt.by_season(preds)
    season_table.to_csv("outputs/log_loss_by_season.csv", index=False)
    print("\n  per season")
    print("    " + season_table.to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}").replace("\n", "\n    "))

    calib = bt.calibration(preds)
    calib.to_csv("outputs/calibration.csv", index=False)

    sim = bt.betting_simulation(test)
    print(f"\n  staking check on the TEST period, against the closing price")
    print(f"    bets {sim['n_bets']}   staked {sim['total_staked']:.1f}u   "
          f"ROI {sim['roi']:+.2%}   final bankroll {sim['final_bankroll']:.3f}")
    print("    (diagnostic only -- you cannot actually bet into a closing line)")

    # ---- a final model fitted on everything, for the ratings table ----------
    last_date = matches["Date"].max().to_numpy().astype("datetime64[D]")
    final = dc.fit_from_frame(matches, teams, ref_date=last_date,
                              xi=xi, prior_sd=prior_sd)
    current = sorted(set(matches[matches["Season"] == matches["Season"].max()]["HomeTeam"]))
    ratings = final.ratings_table()
    ratings = ratings[ratings["team"].isin(current)].reset_index(drop=True)
    ratings.to_csv("outputs/team_ratings.csv", index=False)

    print()
    print("=" * 72)
    print(f"FITTED PARAMETERS (all data, as of {last_date})")
    print("=" * 72)
    print(f"  home advantage gamma   {final.gamma:.3f}")
    print(f"  low-score rho          {final.rho:+.4f}")
    print(f"  league mean goals      {final.league_mean:.3f}")
    print(f"  effective sample       {final.effective_n:.0f} of {final.n_train} matches")
    print("\n  " + ratings.head(10).to_string(
        index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))

    summary = {
        "hyperparameters": {"xi": xi, "half_life_days": float(np.log(2) / xi),
                            "prior_sd": prior_sd, "blend_weight": w_blend},
        "data": info,
        "scores": {k: {kk: {m: v for m, v in vv.items()} for kk, vv in s.items()}
                   for k, s in scores.items()},
        "final_fit": {"gamma": final.gamma, "rho": final.rho,
                      "league_mean": final.league_mean},
        "betting_diagnostic_test": {k: v for k, v in sim.items() if k != "curve"},
    }
    with open("outputs/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    if figures:
        import figures as fig
        fig.make_all(preds, season_table, calib, final, current)
        print("\n  figures written to figures/")

    print("\nDone. outputs/ has predictions, per-season log loss, calibration, ratings.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-run the hyperparameter grid")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    main(tune=args.tune, figures=not args.no_figures)
