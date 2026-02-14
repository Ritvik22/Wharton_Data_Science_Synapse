"""
Phase 1a: Power rankings (Elo) and first-round win probability predictions.
Uses league table and game-level data from phase1a_preprocess().
"""
import os
import pandas as pd
import numpy as np

from ting import phase1a_preprocess


# Elo parameters
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 75  # used in expected score and in win-prob formula


def _game_order_key(game_id_series):
    """Extract numeric order from game_id (e.g. 'game_1' -> 1) for sorting."""
    return game_id_series.str.extract(r"game_(\d+)", expand=False).astype(int)


def compute_elo_rankings(game, initial_elo=INITIAL_ELO, k=K_FACTOR, home_advantage=HOME_ADVANTAGE_ELO):
    """
    Compute Elo ratings by processing games in chronological order.
    game: DataFrame with columns home_team, away_team, home_win; must be sortable by game_id.
    Returns: DataFrame with columns team, elo, power_rank (1 = strongest).
    """
    game = game.copy()
    game["_order"] = _game_order_key(game["game_id"])
    game = game.sort_values("_order").reset_index(drop=True)

    teams = pd.unique(
        np.concatenate([game["home_team"].values, game["away_team"].values])
    )
    elo = {t: float(initial_elo) for t in teams}

    for _, row in game.iterrows():
        h, a = row["home_team"], row["away_team"]
        home_win = int(row["home_win"])

        # Expected score for home team (with home-ice advantage)
        q_home = 10 ** ((elo[h] + home_advantage - elo[a]) / 400)
        e_home = q_home / (1 + q_home)
        e_away = 1 - e_home

        actual_home = home_win
        actual_away = 1 - home_win

        elo[h] = elo[h] + k * (actual_home - e_home)
        elo[a] = elo[a] + k * (actual_away - e_away)

    power = (
        pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    power["power_rank"] = np.arange(1, len(power) + 1)
    return power


def first_round_matchups(season):
    """
    Standard bracket: seed 1 vs 32, 2 vs 31, ..., 16 vs 17.
    Higher seed (lower rank number) is home team.
    season: DataFrame with columns team, rank (1..32).
    Returns: DataFrame with home_team, away_team, home_seed, away_seed.
    """
    seed_to_team = season.set_index("rank")["team"].to_dict()
    rows = []
    for home_seed in range(1, 17):
        away_seed = 33 - home_seed  # 32, 31, ..., 17
        rows.append({
            "home_team": seed_to_team[home_seed],
            "away_team": seed_to_team[away_seed],
            "home_seed": home_seed,
            "away_seed": away_seed,
        })
    return pd.DataFrame(rows)


def home_win_probability(home_elo, away_elo, home_advantage=HOME_ADVANTAGE_ELO):
    """P(home wins) = 1 / (1 + 10^((away_elo - home_elo - home_advantage)/400))."""
    exponent = (away_elo - home_elo - home_advantage) / 400
    return 1.0 / (1.0 + 10 ** exponent)


def predict_first_round(matchups, power_rankings, home_advantage=HOME_ADVANTAGE_ELO):
    """
    matchups: DataFrame with home_team, away_team.
    power_rankings: DataFrame with team, elo.
    Returns: matchups with home_win_probability and away_win_probability.
    """
    elo_map = power_rankings.set_index("team")["elo"].to_dict()
    out = matchups.copy()
    home_elos = out["home_team"].map(elo_map)
    away_elos = out["away_team"].map(elo_map)
    out["home_win_probability"] = [
        home_win_probability(h, a, home_advantage)
        for h, a in zip(home_elos, away_elos)
    ]
    out["away_win_probability"] = 1 - out["home_win_probability"]
    return out


def run_phase1a_rankings(path, output_dir=None):
    """
    Load data, compute Elo power rankings, define first-round matchups,
    compute home win probabilities, and save CSVs.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(path))

    raw, game, team_game, season = phase1a_preprocess(path)

    # Power rankings from Elo
    power_rankings = compute_elo_rankings(game)
    power_rankings.to_csv(os.path.join(output_dir, "power_rankings.csv"), index=False)

    # First-round matchups (1v32, 2v31, ..., 16v17)
    matchups = first_round_matchups(season)
    predictions = predict_first_round(matchups, power_rankings)
    predictions.to_csv(
        os.path.join(output_dir, "first_round_win_probs.csv"),
        index=False,
    )

    return {
        "season": season,
        "power_rankings": power_rankings,
        "matchups": matchups,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "whl_2025 (1).xlsx"
    result = run_phase1a_rankings(path)
    out_dir = os.path.dirname(os.path.abspath(path))
    print(f"Outputs written to: {out_dir}")
    print("power_rankings.csv — 32 teams with elo, power_rank")
    print("first_round_win_probs.csv — 16 matchups with home_win_probability")
    print("\nPower rankings (top 10):")
    print(result["power_rankings"].head(10).to_string(index=False))
    print("\nFirst-round win probabilities:")
    print(result["predictions"].to_string(index=False))
