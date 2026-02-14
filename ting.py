import pandas as pd
import numpy as np

def phase1a_preprocess(path, sheet_name=None):
    if path.lower().endswith(".csv"):
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path, sheet_name=sheet_name) if sheet_name else pd.read_excel(path)

    need = ["game_id", "home_team", "away_team", "home_goals", "away_goals", "went_ot"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = raw[need].copy()

    game = (
        df.groupby(["game_id", "home_team", "away_team", "went_ot"], as_index=False)
          .agg(total_home_goals=("home_goals", "sum"),
               total_away_goals=("away_goals", "sum"))
    )

    game["home_win"] = (game["total_home_goals"] > game["total_away_goals"]).astype(int)
    game["away_win"] = 1 - game["home_win"]

    game["home_reg_win"] = ((game["went_ot"] == 0) & (game["home_win"] == 1)).astype(int)
    game["home_ot_win"]  = ((game["went_ot"] == 1) & (game["home_win"] == 1)).astype(int)
    game["home_reg_loss"]= ((game["went_ot"] == 0) & (game["home_win"] == 0)).astype(int)
    game["home_ot_loss"] = ((game["went_ot"] == 1) & (game["home_win"] == 0)).astype(int)

    game["away_reg_win"] = ((game["went_ot"] == 0) & (game["away_win"] == 1)).astype(int)
    game["away_ot_win"]  = ((game["went_ot"] == 1) & (game["away_win"] == 1)).astype(int)
    game["away_reg_loss"]= ((game["went_ot"] == 0) & (game["away_win"] == 0)).astype(int)
    game["away_ot_loss"] = ((game["went_ot"] == 1) & (game["away_win"] == 0)).astype(int)

    game["home_points"] = 2 * (game["home_reg_win"] + game["home_ot_win"]) + 1 * game["home_ot_loss"]
    game["away_points"] = 2 * (game["away_reg_win"] + game["away_ot_win"]) + 1 * game["away_ot_loss"]

    home_rows = pd.DataFrame({
        "game_id": game["game_id"],
        "team": game["home_team"],
        "opponent": game["away_team"],
        "is_home": 1,
        "went_ot": game["went_ot"],
        "goals_for": game["total_home_goals"],
        "goals_against": game["total_away_goals"],
        "reg_win": game["home_reg_win"],
        "ot_win": game["home_ot_win"],
        "reg_loss": game["home_reg_loss"],
        "ot_loss": game["home_ot_loss"],
        "points": game["home_points"],
    })

    away_rows = pd.DataFrame({
        "game_id": game["game_id"],
        "team": game["away_team"],
        "opponent": game["home_team"],
        "is_home": 0,
        "went_ot": game["went_ot"],
        "goals_for": game["total_away_goals"],
        "goals_against": game["total_home_goals"],
        "reg_win": game["away_reg_win"],
        "ot_win": game["away_ot_win"],
        "reg_loss": game["away_reg_loss"],
        "ot_loss": game["away_ot_loss"],
        "points": game["away_points"],
    })

    team_game = pd.concat([home_rows, away_rows], ignore_index=True)
    team_game["win"] = team_game["reg_win"] + team_game["ot_win"]
    team_game["loss"] = team_game["reg_loss"] + team_game["ot_loss"]
    team_game["goal_diff"] = team_game["goals_for"] - team_game["goals_against"]

    season = (
        team_game.groupby("team", as_index=False)
                 .agg(gp=("game_id", "count"),
                      reg_w=("reg_win", "sum"),
                      ot_w=("ot_win", "sum"),
                      reg_l=("reg_loss", "sum"),
                      ot_l=("ot_loss", "sum"),
                      w=("win", "sum"),
                      l=("loss", "sum"),
                      gf=("goals_for", "sum"),
                      ga=("goals_against", "sum"),
                      gd=("goal_diff", "sum"),
                      pts=("points", "sum"))
    )

    season = season.sort_values(["pts", "gd", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    season["rank"] = np.arange(1, len(season) + 1)

    return raw, game, team_game, season


def clean_and_export(path, sheet_name=None, output_dir=None):
    """Run phase1a preprocessing and export cleaned datasets to CSV."""
    import os
    raw, game, team_game, season = phase1a_preprocess(path, sheet_name=sheet_name)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(path))

    base = os.path.splitext(os.path.basename(path))[0]
    game.to_csv(os.path.join(output_dir, f"{base}_game_level.csv"), index=False)
    team_game.to_csv(os.path.join(output_dir, f"{base}_team_game_level.csv"), index=False)
    season.to_csv(os.path.join(output_dir, f"{base}_league_table.csv"), index=False)

    return raw, game, team_game, season


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "whl_2025 (1).xlsx"
    clean_and_export(path)
    print(f"Cleaned data exported to directory containing: {path}")
