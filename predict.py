"""
Score a fixture with a model fitted on everything in data/.

    python predict.py "Arsenal" "Man City"
    python predict.py "Arsenal" "Man City" --scores 6

Note this is the *in-sample* model -- it has seen every match in the folder.
That is the right thing for forecasting the next match and the wrong thing for
measuring yourself; measurement is what backtest.py is for.
"""

from __future__ import annotations

import argparse

import numpy as np

import dixon_coles as dc
from data_io import load_matches
from run import BEST_PRIOR_SD, BEST_XI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--scores", type=int, default=5,
                    help="how many most-likely scorelines to print")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    matches = load_matches(args.data)
    teams = matches.attrs["teams"]
    for name in (args.home, args.away):
        if name not in teams:
            raise SystemExit(f"Unknown team {name!r}.\nKnown: {', '.join(teams)}")

    ref = matches["Date"].max().to_numpy().astype("datetime64[D]")
    model = dc.fit_from_frame(matches, teams, ref_date=ref,
                              xi=BEST_XI, prior_sd=BEST_PRIOR_SD)

    h, a = teams.index(args.home), teams.index(args.away)
    lam, mu = model.expected_goals(h, a)
    probs = model.outcome_probs(h, a)

    print(f"\n{args.home} vs {args.away}   (model as of {ref})")
    print(f"  expected goals   {float(lam):.2f} - {float(mu):.2f}")
    print(f"  home win {probs[0]:6.1%}   draw {probs[1]:6.1%}   away win {probs[2]:6.1%}")
    print(f"  fair odds        {1/probs[0]:6.2f}       {1/probs[1]:6.2f}        {1/probs[2]:6.2f}")

    grid = model.score_matrix(h, a)
    flat = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1], grid.shape))[0]
    print(f"\n  most likely scorelines")
    for hg, ag in flat[: args.scores]:
        print(f"    {hg}-{ag}   {grid[hg, ag]:5.1%}")

    over_25 = 1 - sum(grid[i, j] for i in range(6) for j in range(6) if i + j <= 2)
    btts = 1 - grid[0, :].sum() - grid[:, 0].sum() + grid[0, 0]
    print(f"\n  over 2.5 goals   {over_25:5.1%}")
    print(f"  both teams score {btts:5.1%}\n")


if __name__ == "__main__":
    main()
