"""
Figures for the README.

Four charts, each with one job:
  1. log loss by season      -- magnitude comparison, model vs market
  2. reliability diagram     -- is a stated 30% really a 30%?
  3. cumulative gap          -- where the model loses ground, over time
  4. attack vs defence       -- what the fitted parameters actually say

Colour follows the validated categorical palette (blue = model, orange =
market); the 1/3 baseline is grey because it is a reference, not a series.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_io import result_index

# --- palette (light mode, validated categorical slots 1 and 2) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
MODEL = "#2a78d6"      # slot 1, blue
MARKET = "#eb6834"     # slot 2, orange
NEUTRAL = "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def _clean(ax, y_grid=True):
    """Recessive axes: no box, a faint horizontal rule, nothing else."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if y_grid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)


# --------------------------------------------------------------------------- #
def log_loss_by_season(season_table, path="figures/log_loss_by_season.png"):
    """
    A dumbbell, not a grouped bar chart.

    Log loss lives in a narrow band around 0.9-1.0, so bars would have to start
    at a non-zero baseline -- which exaggerates a difference of 0.02 into a
    difference that looks like half the bar. Two dots joined by a segment show
    the same comparison without lying about the ratio.
    """
    df = season_table.copy()
    y = np.arange(len(df))[::-1]        # newest season at the top

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    baseline = -np.log(1 / 3)
    ax.axvline(baseline, color=NEUTRAL, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("no knowledge\n(1/3 each)", (baseline, y.max() + 0.55),
                xytext=(4, 0), textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=NEUTRAL)

    ax.hlines(y, df["market"], df["model"], color=GRID, linewidth=3, zorder=3)
    ax.scatter(df["market"], y, s=80, color=MARKET, zorder=5,
               edgecolor=SURFACE, linewidth=2, label="Pinnacle closing")
    ax.scatter(df["model"], y, s=80, color=MODEL, zorder=5,
               edgecolor=SURFACE, linewidth=2, label="Dixon-Coles model")

    for yi, row in zip(y, df.itertuples()):
        ax.annotate(f"{row.gap:+.3f}", (max(row.model, row.market), yi),
                    xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)

    labels = [f"{s}" + ("  (part)" if n < 380 else "")
              for s, n in zip(df["season"], df["n"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.86, 1.16)
    ax.set_ylim(-0.7, len(df) - 0.1)
    ax.set_xlabel("log loss  (lower is better)")
    ax.set_title("Out-of-sample log loss by season", loc="left", pad=26)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.10), ncol=2)
    _clean(ax, y_grid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability(calib, preds, path="figures/calibration.png"):
    from backtest import calibration
    market_calib = calibration(preds, prob_cols=["mkt_H", "mkt_D", "mkt_A"])

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("perfect calibration", (0.90, 0.86), rotation=45,
                rotation_mode="anchor", ha="right", va="bottom",
                fontsize=9, color=NEUTRAL)

    ax.plot(calib["predicted"], calib["observed"], color=MODEL, linewidth=2,
            solid_capstyle="round", zorder=4)
    ax.plot(calib["predicted"], calib["observed"], "o", color=MODEL, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="model", zorder=5)
    ax.plot(market_calib["predicted"], market_calib["observed"], color=MARKET,
            linewidth=2, solid_capstyle="round", zorder=3)
    ax.plot(market_calib["predicted"], market_calib["observed"], "o", color=MARKET,
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="market", zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated probability")
    ax.set_ylabel("share that actually happened")
    ax.set_title("Reliability: does 30% mean 30%?", loc="left", pad=26)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.10), ncol=2)
    _clean(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cumulative_gap(preds, path="figures/cumulative_gap.png"):
    """
    Running total of (model loss - market loss), match by match.

    A flat line means the model is keeping pace. A line drifting upward is the
    model steadily paying for what it does not know. Reading the slope is far
    more informative than one summary number.
    """
    df = preds[preds["has_odds"]].sort_values("Date").reset_index(drop=True)
    idx = result_index(df["FTR"])
    rows = np.arange(len(df))
    p_model = df[["p_H", "p_D", "p_A"]].to_numpy(float)[rows, idx]
    p_market = df[["mkt_H", "mkt_D", "mkt_A"]].to_numpy(float)[rows, idx]
    gap = np.cumsum(-np.log(p_model) + np.log(p_market))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color=NEUTRAL, linewidth=1.2, zorder=2)
    ax.fill_between(df["Date"], 0, gap, color=MODEL, alpha=0.12, zorder=3)
    ax.plot(df["Date"], gap, color=MODEL, linewidth=2, zorder=4)

    ax.annotate(f"{gap[-1]:+.0f} nats over {len(df)} matches\n"
                f"= {gap[-1] / len(df):+.4f} per match",
                (0.02, 0.95), xycoords="axes fraction", ha="left", va="top",
                fontsize=9, color=INK_2)

    ax.set_ylabel("cumulative log loss vs market\n(above 0 = model is worse)")
    ax.set_title("Where the model gives ground", loc="left", pad=12)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def attack_defence(model, current_teams, path="figures/team_ratings.png"):
    keep = [i for i, t in enumerate(model.teams) if t in set(current_teams)]
    a = model.alpha[keep]
    d = model.beta[keep]
    names = [model.teams[i] for i in keep]

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.axvline(1.0, color=NEUTRAL, linewidth=1.0, alpha=0.5, zorder=2)
    ax.axhline(1.0, color=NEUTRAL, linewidth=1.0, alpha=0.5, zorder=2)
    ax.scatter(a, d, s=70, color=MODEL, edgecolor=SURFACE, linewidth=2, zorder=4)

    ax.margins(0.12)
    ax.invert_yaxis()      # so "good defence" is at the top

    texts = [ax.text(x, y, name, fontsize=8.5, color=INK_2, zorder=6)
             for x, y, name in zip(a, d, names)]
    try:                                    # optional: nudges labels apart
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.15, 1.35),
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=0.6))
    except ImportError:
        pass

    ax.set_xlabel("attack   (1.0 = league average, higher scores more)")
    ax.set_ylabel("defence   (1.0 = league average, lower concedes less)")
    ax.set_title(f"Fitted ratings, {str(model.ref_date)[:10]}", loc="left", pad=12)
    ax.annotate("strong both ways ↗", (0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color=NEUTRAL)
    _clean(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all(preds, season_table, calib, final_model, current_teams):
    log_loss_by_season(season_table)
    reliability(calib, preds)
    cumulative_gap(preds)
    attack_defence(final_model, current_teams)
