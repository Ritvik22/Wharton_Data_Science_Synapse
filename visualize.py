"""
Create visualizations for WHL 2025 Phase 1a data:
- League standings (points)
- Power rankings (Elo)
- League vs Power rank comparison
- First-round win probabilities
- Goal differential overview
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style: clean, readable, distinct from generic "AI slop"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
COLORS = {
    "primary": "#1a365d",      # dark blue
    "accent": "#2b6cb0",       # medium blue
    "highlight": "#ed8936",    # orange
    "home": "#38a169",         # green
    "away": "#e53e3e",        # red
    "neutral": "#718096",     # gray
    "bg": "#f7fafc",
}


def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    season = pd.read_csv(os.path.join(base, "whl_2025 (1)_league_table.csv"))
    power = pd.read_csv(os.path.join(base, "power_rankings.csv"))
    probs = pd.read_csv(os.path.join(base, "first_round_win_probs.csv"))
    return season, power, probs


def fig_league_standings(season, out_dir):
    """Bar chart: season points by team (league table order)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    teams = season["team"].str.replace("_", " ").str.title()
    pts = season["pts"]
    colors = [COLORS["accent"] if r <= 8 else COLORS["neutral"] for r in season["rank"]]
    bars = ax.barh(range(len(teams)), pts, color=colors, height=0.75, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=9)
    ax.set_xlabel("Season points")
    ax.set_title("League standings (82 games)\nTop 8 in blue")
    ax.invert_yaxis()
    ax.set_xlim(0, pts.max() * 1.05)
    ax.axvline(pts.median(), color=COLORS["highlight"], linestyle="--", alpha=0.7, label="Median points")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_league_standings.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_league_standings.png")


def fig_power_rankings(power, out_dir):
    """Bar chart: Elo power rating by team."""
    fig, ax = plt.subplots(figsize=(12, 10))
    teams = power["team"].str.replace("_", " ").str.title()
    elo = power["elo"]
    colors = [COLORS["primary"] if r <= 8 else COLORS["neutral"] for r in power["power_rank"]]
    ax.barh(range(len(teams)), elo, color=colors, height=0.75, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=9)
    ax.set_xlabel("Elo rating")
    ax.set_title("Power rankings (Elo)\nTop 8 in dark blue")
    ax.invert_yaxis()
    ax.axvline(1500, color=COLORS["highlight"], linestyle="--", alpha=0.7, label="Initial Elo (1500)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_power_rankings.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_power_rankings.png")


def fig_league_vs_power(season, power, out_dir):
    """Scatter: league rank vs power rank; annotate big movers."""
    merged = season[["team", "rank"]].merge(power[["team", "power_rank"]], on="team")
    merged["team_label"] = merged["team"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(merged["rank"], merged["power_rank"], c=COLORS["accent"], s=80, alpha=0.8, edgecolors="white")
    # Diagonal = same rank
    ax.plot([1, 32], [1, 32], "k--", alpha=0.4, label="League rank = Power rank")

    # Annotate teams with large rank difference
    merged["diff"] = merged["rank"] - merged["power_rank"]
    big = merged[abs(merged["diff"]) >= 5].sort_values("diff")
    for _, row in big.iterrows():
        ax.annotate(
            row["team_label"],
            (row["rank"], row["power_rank"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
            alpha=0.9,
        )
    ax.set_xlabel("League standing (rank 1–32)")
    ax.set_ylabel("Power ranking (1–32)")
    ax.set_title("League rank vs Power rank\nPoints above line = stronger schedule; below = weaker schedule")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(32.5, 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_league_vs_power_rank.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_league_vs_power_rank.png")


def fig_first_round_probs(probs, out_dir):
    """Horizontal bar: home win probability for each of 16 first-round matchups."""
    probs = probs.copy()
    probs["matchup"] = (
        probs["home_team"].str.replace("_", " ").str.title()
        + " vs "
        + probs["away_team"].str.replace("_", " ").str.title()
    )
    probs["pct"] = probs["home_win_probability"] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(probs))
    bars = ax.barh(y_pos, probs["pct"], color=COLORS["home"], height=0.6, edgecolor="white", linewidth=0.5)
    ax.axvline(50, color=COLORS["neutral"], linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(probs["matchup"], fontsize=9)
    ax.set_xlabel("Home team win probability (%)")
    ax.set_title("First-round tournament: home win probability (16 matchups)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for i, (_, row) in enumerate(probs.iterrows()):
        ax.text(row["pct"] + 1, i, f"{row['pct']:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_first_round_win_probabilities.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_first_round_win_probabilities.png")


def fig_goal_differential(season, out_dir):
    """Bar chart: goal differential by team (league order)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    teams = season["team"].str.replace("_", " ").str.title()
    gd = season["gd"]
    colors = [COLORS["home"] if g >= 0 else COLORS["away"] for g in gd]
    ax.barh(range(len(teams)), gd, color=colors, height=0.75, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=9)
    ax.set_xlabel("Goal differential (GF − GA)")
    ax.set_title("Season goal differential by team")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(
        [mpatches.Patch(color=COLORS["home"]), mpatches.Patch(color=COLORS["away"])],
        ["Positive", "Negative"],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_goal_differential.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_goal_differential.png")


def fig_summary_dashboard(season, power, probs, out_dir):
    """2x2 summary: points distribution, Elo distribution, top matchups, rank comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Points distribution
    ax = axes[0, 0]
    ax.hist(season["pts"], bins=12, color=COLORS["accent"], edgecolor="white")
    ax.set_xlabel("Season points")
    ax.set_ylabel("Number of teams")
    ax.set_title("Distribution of season points")

    # Elo distribution
    ax = axes[0, 1]
    ax.hist(power["elo"], bins=12, color=COLORS["primary"], edgecolor="white")
    ax.axvline(1500, color=COLORS["highlight"], linestyle="--", label="Initial 1500")
    ax.set_xlabel("Elo rating")
    ax.set_ylabel("Number of teams")
    ax.set_title("Distribution of power ratings (Elo)")
    ax.legend()

    # Top 8 first-round home win probs
    ax = axes[1, 0]
    top8 = probs.head(8)
    labels = [f"#{int(r.home_seed)} {r.home_team}" for _, r in top8.iterrows()]
    ax.barh(range(len(labels)), top8["home_win_probability"] * 100, color=COLORS["home"], height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Home win %")
    ax.set_title("First round: top 8 seeds home win probability")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    # League vs Power (compact)
    ax = axes[1, 1]
    merged = season[["team", "rank"]].merge(power[["team", "power_rank"]], on="team")
    ax.scatter(merged["rank"], merged["power_rank"], c=COLORS["accent"], s=50, alpha=0.7)
    ax.plot([1, 32], [1, 32], "k--", alpha=0.4)
    ax.set_xlabel("League rank")
    ax.set_ylabel("Power rank")
    ax.set_title("League vs Power rank")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(32.5, 0.5)

    fig.suptitle("WHL 2025 Phase 1a — Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_summary_dashboard.png")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    season, power, probs = load_data()

    fig_league_standings(season, out_dir)
    fig_power_rankings(power, out_dir)
    fig_league_vs_power(season, power, out_dir)
    fig_first_round_probs(probs, out_dir)
    fig_goal_differential(season, out_dir)
    fig_summary_dashboard(season, power, probs, out_dir)

    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
