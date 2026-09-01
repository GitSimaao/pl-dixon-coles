"""
Loading, cleaning and market-benchmark utilities.

Input: the raw season CSVs from football-data.co.uk (E0 = English Premier League).
Output: one tidy DataFrame, plus the *market* probabilities that the model has to beat.

The only rows we ever drop are rows without a final score. Matches without closing
odds are KEPT -- they carry information about how teams play, they just cannot be
part of the market comparison.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# Columns we actually need. AvgCH/AvgCD/AvgCA are the market-average *closing*
# prices for home / draw / away — the mean across the books football-data.co.uk
# surveys, taken at kickoff. Closing prices are the sharpest public number in
# football: they are the market after all the informed money has gone in.
#
# Until January 2026 the benchmark was Pinnacle's close (PSCH/PSCD/PSCA); then
# football-data stopped carrying those columns. On the 2,490 matches carrying
# both prices the two de-vigged benchmarks differ by 0.0001 nats of log loss —
# scripts/check_benchmark.py in the production repo reruns that measurement.
# AvgC* is published from 2019/20, so earlier seasons train the model but
# cannot be graded.
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLS = ["AvgCH", "AvgCD", "AvgCA"]

OUTCOMES = ("H", "D", "A")
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def load_matches(data_dir: str = "data") -> pd.DataFrame:
    """Read every E0_*.csv in `data_dir` into a single, sorted DataFrame."""
    paths = sorted(glob.glob(os.path.join(data_dir, "E0_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No E0_*.csv files found in {data_dir!r}")

    frames = []
    for path in paths:
        # "E0_1516.csv" -> season label "2015/16"
        code = re.search(r"E0_(\d{4})", os.path.basename(path)).group(1)
        season = f"20{code[:2]}/{code[2:]}"

        # utf-8-sig strips the byte-order mark that some of these files carry.
        raw = pd.read_csv(path, encoding="utf-8-sig")

        keep = [c for c in CORE_COLS + ODDS_COLS if c in raw.columns]
        df = raw[keep].copy()
        for col in ODDS_COLS:                      # older files may lack a column
            if col not in df.columns:
                df[col] = np.nan
        df["Season"] = season
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)

    # football-data mixes "08/08/2015" and "08/08/15"; dayfirst + format="mixed"
    # handles both without silently swapping day and month.
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, format="mixed")

    matches = matches.dropna(subset=["FTHG", "FTAG", "FTR"])
    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)

    matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    # A stable integer id per team, used everywhere downstream.
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    lookup = {name: i for i, name in enumerate(teams)}
    matches["home_id"] = matches["HomeTeam"].map(lookup)
    matches["away_id"] = matches["AwayTeam"].map(lookup)

    matches.attrs["teams"] = teams
    return matches


def add_market_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Turn closing decimal odds into probabilities that sum to one.

    A bookmaker's three prices imply more than 100% -- the excess is the margin
    (the "overround", ~4% for the market average). Dividing each implied
    probability by the total removes it proportionally. This is the simplest
    de-vigging method; it slightly over-weights longshots compared with Shin or
    power methods, but it is the standard baseline and it is what the model is
    measured against.
    """
    out = matches.copy()
    # `> 1` because a 0.0 in an odds column is a placeholder, not a price --
    # notna() lets it through, and 1/0 turns the row into probabilities of
    # [0, 0, nan] and a log loss of ~34 for one match.
    has_odds = (out[ODDS_COLS].notna().all(axis=1)
                & (out[ODDS_COLS] > 1).all(axis=1))

    inv = 1.0 / out.loc[has_odds, ODDS_COLS].to_numpy(dtype=float)
    overround = inv.sum(axis=1, keepdims=True)
    fair = inv / overround

    for col in ["mkt_H", "mkt_D", "mkt_A"]:
        out[col] = np.nan
    out.loc[has_odds, ["mkt_H", "mkt_D", "mkt_A"]] = fair
    out["has_odds"] = has_odds
    out["overround"] = np.nan
    out.loc[has_odds, "overround"] = overround.ravel() - 1.0
    return out


def result_index(results: pd.Series | np.ndarray) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2, the column of the probability that actually happened."""
    return pd.Series(results).map(OUTCOME_INDEX).to_numpy(dtype=int)


def log_loss(probs: np.ndarray, results) -> float:
    """
    Mean of -log(p) over the outcome that actually occurred.

    Lower is better. Guessing 1/3 each time gives -log(1/3) = 1.0986, which is the
    "I know nothing about football" line. Everything anyone knows about football is
    worth the gap between that and the closing market.
    """
    probs = np.asarray(probs, dtype=float)
    idx = result_index(results)
    picked = probs[np.arange(len(idx)), idx]
    return float(-np.log(np.clip(picked, 1e-15, 1.0)).mean())


def ranked_probability_score(probs: np.ndarray, results) -> float:
    """
    RPS -- the standard ordinal scoring rule for 1X2 forecasts.

    Unlike log loss it knows that H / D / A are ordered: calling a home win when
    the game was a draw is a smaller miss than calling it when the away side won.
    Lower is better.
    """
    probs = np.asarray(probs, dtype=float)
    idx = result_index(results)
    obs = np.zeros_like(probs)
    obs[np.arange(len(idx)), idx] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    return float((((cum_p - cum_o) ** 2).sum(axis=1) / (probs.shape[1] - 1)).mean())


def accuracy(probs: np.ndarray, results) -> float:
    """Share of matches where the highest-probability outcome was the right one."""
    return float((np.asarray(probs).argmax(axis=1) == result_index(results)).mean())


def summarise_market(matches: pd.DataFrame) -> dict:
    """The benchmark line: how good is the market-average close, in log loss?"""
    graded = matches[matches["has_odds"]]
    probs = graded[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(dtype=float)
    return {
        "n_matches": int(len(matches)),
        "n_with_odds": int(len(graded)),
        "n_without_odds": int(len(matches) - len(graded)),
        "mean_overround": float(graded["overround"].mean()),
        "market_log_loss": log_loss(probs, graded["FTR"]),
        "market_rps": ranked_probability_score(probs, graded["FTR"]),
        "market_accuracy": accuracy(probs, graded["FTR"]),
        "uniform_log_loss": float(-np.log(1 / 3)),
    }


if __name__ == "__main__":
    m = add_market_probabilities(load_matches())
    info = summarise_market(m)
    for key, value in info.items():
        print(f"{key:22s} {value}")
