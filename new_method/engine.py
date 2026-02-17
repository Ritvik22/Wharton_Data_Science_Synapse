import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, brier_score_loss, ndcg_score
from scipy.stats import spearmanr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RAW_PATH = os.path.join(HERE, "whl_2025_raw.csv")
PARENT = os.path.dirname(HERE)
LEAGUE_TABLE = os.path.join(PARENT, "whl_2025 (1)_league_table.csv")

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADV = 75

os.makedirs(RESULTS, exist_ok=True)

def load_game_data():
    raw = pd.read_csv(RAW_PATH)
    game = raw.groupby("game_id").agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
    ).reset_index()
    game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
    game["goal_diff"] = game["home_goals"] - game["away_goals"]
    game["_order"] = game["game_id"].str.extract(r"game_(\d+)", expand=False).astype(int)
    game = game.sort_values("_order").reset_index(drop=True)
    print(f"[DATA] {len(game)} games, "
          f"{len(set(game['home_team']) | set(game['away_team']))} teams")
    return game

def get_seed_map():
    if os.path.exists(LEAGUE_TABLE):
        return pd.read_csv(LEAGUE_TABLE).set_index("rank")["team"].to_dict()
    return None

def load_ground_truth():
    if not os.path.exists(LEAGUE_TABLE):
        return None, None
    lt = pd.read_csv(LEAGUE_TABLE)
    true_pts = dict(zip(lt["team"], lt["pts"]))
    teams_ordered = list(lt["team"])
    return true_pts, teams_ordered

def _elo_expected(elo_h, elo_a):
    q = 10 ** ((elo_h + HOME_ADV - elo_a) / 400)
    return q / (1 + q)

def fit_sigmoid(game, rmap):
    x = np.array([rmap.get(r["home_team"], 0) - rmap.get(r["away_team"], 0)
                   for _, r in game.iterrows()], dtype=float)
    y = np.array([1 if r["home_goals"] > r["away_goals"] else 0
                   for _, r in game.iterrows()], dtype=float)
    try:
        (s,), _ = curve_fit(lambda t, s: 1 / (1 + np.exp(-t / s)), x, y, p0=[1.0], maxfev=10000)
        s = abs(s)
    except (RuntimeError, ValueError):
        s = 1.0
    return max(s, 1e-6)

def elo_game_probs(game, elo_map):
    return np.array([_elo_expected(elo_map.get(h, INITIAL_ELO), elo_map.get(a, INITIAL_ELO))
                     for h, a in zip(game["home_team"], game["away_team"])])

def sigmoid_game_probs(game, rmap, sigma):
    return np.array([1 / (1 + np.exp(-(rmap.get(h, 0) - rmap.get(a, 0)) / sigma))
                     for h, a in zip(game["home_team"], game["away_team"])])

def _run_elo(game, k_func):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    elo = {t: float(INITIAL_ELO) for t in teams}
    for _, row in game.iterrows():
        h, a = row["home_team"], row["away_team"]
        hw = int(row["home_win"])
        gd = abs(row["goal_diff"])
        e = _elo_expected(elo[h], elo[a])
        k = k_func(K_FACTOR, gd, abs(elo[h] - elo[a]))
        elo[h] += k * (hw - e)
        elo[a] += k * ((1 - hw) - (1 - e))
    return elo

def _run_joint_additive(game):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    elo = {t: float(INITIAL_ELO) for t in teams}
    K1, K2 = K_FACTOR, K_FACTOR * 0.4
    max_gd = max(abs(game["goal_diff"].max()), abs(game["goal_diff"].min()), 1)
    for _, row in game.iterrows():
        h, a = row["home_team"], row["away_team"]
        hw = int(row["home_win"])
        m = row["goal_diff"] / max_gd
        e_w = _elo_expected(elo[h], elo[a])
        e_m = (2 * e_w - 1) * 0.5
        elo[h] += K1 * (hw - e_w) + K2 * (m - e_m)
        elo[a] += K1 * ((1 - hw) - (1 - e_w)) + K2 * (-m + e_m)
    return elo

def compute_all_elo(game):
    results = {}
    results["elo_basic"] = _run_elo(game, lambda K, gd, diff: K)
    results["elo_mov"] = _run_elo(game, lambda K, gd, diff:
        K * max(np.log(gd + 1) * (2.2 / (diff * 0.001 + 2.2)), 0.5))
    results["lin_elo"] = _run_elo(game, lambda K, gd, diff: K * (1 + 0.5 * gd))
    results["mult_elo"] = _run_elo(game, lambda K, gd, diff: K * (1 + gd) ** 0.7)
    results["log_elo"] = _run_elo(game, lambda K, gd, diff:
        K * 2.0 / (1.0 + np.exp(-0.8 * gd)))
    results["ja_elo"] = _run_joint_additive(game)
    n = len(next(iter(results.values())))
    print(f"[ELO] 6 variants computed for {n} teams")
    return results

def compute_ip(game, epochs=10):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    tg = {}
    for t in teams:
        recs = []
        for _, r in game.iterrows():
            if r["home_team"] == t:
                recs.append({"opp": r["away_team"], "gd": r["home_goals"] - r["away_goals"]})
            elif r["away_team"] == t:
                recs.append({"opp": r["home_team"], "gd": r["away_goals"] - r["home_goals"]})
        tg[t] = recs
    agd = {t: np.mean([r["gd"] for r in tg[t]]) if tg[t] else 0 for t in teams}
    power = {t: 0.0 for t in teams}
    for _ in range(epochs):
        prev = dict(power)
        for t in teams:
            opp = [prev[r["opp"]] for r in tg[t]]
            power[t] = agd[t] + (np.mean(opp) if opp else 0)
    print(f"[IP] Converged ({epochs} epochs)")
    return power

def compute_bscore(game, alpha=300):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    T = len(game)
    W = np.zeros((n, n))
    for gi, (_, r) in enumerate(game.iterrows()):
        h, a = idx[r["home_team"]], idx[r["away_team"]]
        decay = 1.0 / (1.0 + (T - 1 - gi) / alpha)
        if r["home_goals"] > r["away_goals"]:
            W[a, h] += decay
        elif r["home_goals"] < r["away_goals"]:
            W[h, a] += decay
    Wt = W.T + 1e-8
    evals, evecs = np.linalg.eig(Wt)
    v = np.real(evecs[:, np.argmax(np.real(evals))])
    if v.sum() < 0:
        v = -v
    v /= np.linalg.norm(v)
    print(f"[BS] Eigenvector centrality computed (alpha={alpha})")
    return {teams[i]: v[i] for i in range(n)}

def _poisson_win_prob(lam_h, lam_a, max_goals=12):
    p = 0.0
    for hg in range(1, max_goals + 1):
        ph = poisson.pmf(hg, lam_h)
        for ag in range(0, hg):
            p += ph * poisson.pmf(ag, lam_a)
    return np.clip(p, 0.01, 0.99)

def compute_parx(game, decay=0.94, home_boost=1.08):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    total_goals = game["home_goals"].sum() + game["away_goals"].sum()
    avg_gpg = total_goals / (2 * len(game))
    att = {t: avg_gpg for t in teams}
    dfn = {t: avg_gpg for t in teams}
    for _, row in game.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        att[h] = decay * att[h] + (1 - decay) * hg
        dfn[h] = decay * dfn[h] + (1 - decay) * ag
        att[a] = decay * att[a] + (1 - decay) * ag
        dfn[a] = decay * dfn[a] + (1 - decay) * hg
    probs = []
    for _, row in game.iterrows():
        h, a = row["home_team"], row["away_team"]
        lam_h = att[h] * dfn[a] / avg_gpg * home_boost
        lam_a = att[a] * dfn[h] / avg_gpg
        lam_h = max(lam_h, 0.1)
        lam_a = max(lam_a, 0.1)
        probs.append(_poisson_win_prob(lam_h, lam_a))
    strength = {}
    for t in teams:
        strength[t] = np.log(att[t] / dfn[t]) if dfn[t] > 0.01 else 0.0
    print(f"[PARX] Poisson autoregressive computed (decay={decay})")
    return strength, att, dfn, avg_gpg, home_boost, np.array(probs)

def parx_matchup_prob(h, a, att, dfn, avg_gpg, home_boost):
    lam_h = max(att.get(h, avg_gpg) * dfn.get(a, avg_gpg) / avg_gpg * home_boost, 0.1)
    lam_a = max(att.get(a, avg_gpg) * dfn.get(h, avg_gpg) / avg_gpg, 0.1)
    return _poisson_win_prob(lam_h, lam_a)

def compute_ndcg(true_pts, pred_scores, teams):
    y_true = np.array([[true_pts.get(t, 0) for t in teams]])
    y_score = np.array([[pred_scores.get(t, 0) for t in teams]])
    return ndcg_score(y_true, y_score)

def compute_map_at_k(true_pts, pred_scores, teams, k=16):
    sorted_true = sorted(teams, key=lambda t: true_pts.get(t, 0), reverse=True)
    relevant = set(sorted_true[:k])
    sorted_pred = sorted(teams, key=lambda t: pred_scores.get(t, 0), reverse=True)
    hits = 0
    sum_precision = 0.0
    for i, t in enumerate(sorted_pred):
        if t in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(k, len(relevant))

def normalize_ratings(rating_map):
    vals = np.array(list(rating_map.values()))
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return {k: 0.5 for k in rating_map}
    return {k: (v - mn) / (mx - mn) for k, v in rating_map.items()}

def blend_ratings(rating_maps, weights):
    normed = [normalize_ratings(rm) for rm in rating_maps]
    teams = list(normed[0].keys())
    return {t: sum(w * nm.get(t, 0) for w, nm in zip(weights, normed))
            for t in teams}

def pythagorean_rankings(league_csv):
    lt = pd.read_csv(league_csv)
    exp = 2.0
    lt["pyth_win_pct"] = lt["gf"] ** exp / (lt["gf"] ** exp + lt["ga"] ** exp)
    lt = lt.sort_values("pyth_win_pct", ascending=False).reset_index(drop=True)
    lt["pyth_rank"] = range(1, len(lt) + 1)
    return dict(zip(lt["team"], lt["pyth_win_pct"])), dict(zip(lt["team"], lt["pyth_rank"]))

def compute_rank_correlations(pred_scores, true_pts, pyth_scores, teams):
    pred_arr = [pred_scores.get(t, 0) for t in teams]
    pts_arr = [true_pts.get(t, 0) for t in teams]
    pyth_arr = [pyth_scores.get(t, 0) for t in teams]
    sp_pts, _ = spearmanr(pred_arr, pts_arr)
    kt_pts, _ = kendalltau(pred_arr, pts_arr)
    sp_pyth, _ = spearmanr(pred_arr, pyth_arr)
    kt_pyth, _ = kendalltau(pred_arr, pyth_arr)
    return sp_pts, kt_pts, sp_pyth, kt_pyth

def optimal_blend(y, probs_list, step=0.01):
    n = len(probs_list)
    if n == 1:
        return (1.0,), brier_score_loss(y, probs_list[0])
    best_b, best_w = 1.0, tuple([1 / n] * n)
    if n == 2:
        for w in np.arange(0, 1.001, step):
            b = brier_score_loss(y, w * probs_list[0] + (1 - w) * probs_list[1])
            if b < best_b:
                best_b, best_w = b, (w, 1 - w)
    elif n == 3:
        for w1 in np.arange(0, 1.01, 0.05):
            for w2 in np.arange(0, 1.01 - w1, 0.05):
                w3 = 1.0 - w1 - w2
                b = brier_score_loss(y, w1 * probs_list[0] + w2 * probs_list[1] + w3 * probs_list[2])
                if b < best_b:
                    best_b, best_w = b, (w1, w2, w3)
    return best_w, best_b

def build_rank_df(name, rating_map):
    df = pd.DataFrame({"team": list(rating_map.keys()),
                        name: list(rating_map.values())})
    df = df.sort_values(name, ascending=False).reset_index(drop=True)
    df[f"{name}_rank"] = range(1, len(df) + 1)
    return df

def make_rankings(rank_dfs, rank_cols):
    merged = rank_dfs[0].copy()
    for df in rank_dfs[1:]:
        merged = merged.merge(df, on="team")
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged = merged.sort_values("avg_rank").reset_index(drop=True)
    merged["combined_rank"] = range(1, len(merged) + 1)
    return merged

def make_predictions(seed_map, prob_funcs, weights):
    rows = []
    for s in range(1, 17):
        h, a = seed_map[s], seed_map[33 - s]
        indiv = [f(h, a) for f in prob_funcs]
        blend = sum(w * p for w, p in zip(weights, indiv))
        row = {"home_team": h, "away_team": a, "home_seed": s, "away_seed": 33 - s,
               "blend_home_prob": round(blend, 6), "blend_away_prob": round(1 - blend, 6)}
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    print("\n" + "=" * 70)
    print("  ENGINE — All 9 Rating Methods + All Combinations")
    print("=" * 70 + "\n")
    game = load_game_data()
    y = game["home_win"].values
    seed_map = get_seed_map()
    true_pts, gt_teams = load_ground_truth()
    has_gt = true_pts is not None
    pyth_scores, pyth_ranks = None, None
    if has_gt:
        print(f"[GT] League table loaded — {len(gt_teams)} teams, top: {gt_teams[0]} ({true_pts[gt_teams[0]]} pts)")
        pyth_scores, pyth_ranks = pythagorean_rankings(LEAGUE_TABLE)
        pyth_top3 = sorted(pyth_scores, key=pyth_scores.get, reverse=True)[:3]
        print(f"[PYTH] Pythagorean top-3: {pyth_top3[0]} ({pyth_scores[pyth_top3[0]]:.3f}), "
              f"{pyth_top3[1]} ({pyth_scores[pyth_top3[1]]:.3f}), "
              f"{pyth_top3[2]} ({pyth_scores[pyth_top3[2]]:.3f})")
    else:
        print("[GT] No league table found — NDCG/MAP will be skipped")
    elo_maps = compute_all_elo(game)
    ip_map = compute_ip(game, epochs=10)
    bs_map = compute_bscore(game, alpha=300)
    parx_str, parx_att, parx_def, parx_avg, parx_hb, parx_probs = compute_parx(game)
    ip_sigma = fit_sigmoid(game, ip_map)
    bs_sigma = fit_sigmoid(game, bs_map)
    print(f"[SIGMOID] IP σ={ip_sigma:.4f}, BS σ={bs_sigma:.6f}\n")
    base = {}
    elo_labels = {
        "elo_basic": "Elo(basic)",
        "elo_mov":   "Elo(538-MOV)",
        "lin_elo":   "Lin-Elo",
        "ja_elo":    "J-A-Elo",
        "mult_elo":  "Mult-Elo",
        "log_elo":   "Log-Elo",
    }
    for key, label in elo_labels.items():
        emap = elo_maps[key]
        base[key] = {
            "label": label,
            "probs": elo_game_probs(game, emap),
            "rank_df": build_rank_df(key, emap),
            "rank_col": f"{key}_rank",
            "rating_map": emap,
            "prob_func": lambda h, a, em=emap: _elo_expected(em.get(h, INITIAL_ELO), em.get(a, INITIAL_ELO)),
        }
    base["ip"] = {
        "label": "IP",
        "probs": sigmoid_game_probs(game, ip_map, ip_sigma),
        "rank_df": build_rank_df("ip", ip_map),
        "rank_col": "ip_rank",
        "rating_map": ip_map,
        "prob_func": lambda h, a: 1 / (1 + np.exp(-(ip_map.get(h, 0) - ip_map.get(a, 0)) / ip_sigma)),
    }
    base["bs"] = {
        "label": "BS",
        "probs": sigmoid_game_probs(game, bs_map, bs_sigma),
        "rank_df": build_rank_df("bs", bs_map),
        "rank_col": "bs_rank",
        "rating_map": bs_map,
        "prob_func": lambda h, a: 1 / (1 + np.exp(-(bs_map.get(h, 0) - bs_map.get(a, 0)) / bs_sigma)),
    }
    base["parx"] = {
        "label": "PARX",
        "probs": parx_probs,
        "rank_df": build_rank_df("parx", parx_str),
        "rank_col": "parx_rank",
        "rating_map": parx_str,
        "prob_func": lambda h, a: parx_matchup_prob(h, a, parx_att, parx_def, parx_avg, parx_hb),
    }
    keys = list(base.keys())
    print(f"Base methods: {len(keys)}  ({', '.join(base[k]['label'] for k in keys)})")
    comparison_rows = []
    MAX_SIZE = 3
    for size in range(1, MAX_SIZE + 1):
        count = 0
        for combo in combinations(keys, size):
            labels = [base[k]["label"] for k in combo]
            probs_list = [base[k]["probs"] for k in combo]
            weights, best_brier = optimal_blend(y, probs_list)
            p_blend = sum(w * p for w, p in zip(weights, probs_list))
            auc = roc_auc_score(y, p_blend)
            w_str = "—" if size == 1 else " / ".join(f"{l}={w:.0%}" for l, w in zip(labels, weights))
            row_dict = {
                "method": "+".join(combo),
                "size": size,
                "components": " + ".join(labels),
                "AUC": auc,
                "Brier": best_brier,
            }
            if has_gt:
                if size == 1:
                    r_map = base[combo[0]]["rating_map"]
                else:
                    r_maps = [base[k]["rating_map"] for k in combo]
                    r_map = blend_ratings(r_maps, weights)
                row_dict["NDCG"] = compute_ndcg(true_pts, r_map, gt_teams)
                row_dict["MAP@16"] = compute_map_at_k(true_pts, r_map, gt_teams, k=16)
                sp_pts, kt_pts, sp_pyth, kt_pyth = compute_rank_correlations(
                    r_map, true_pts, pyth_scores, gt_teams)
                row_dict["Sp_Standings"] = sp_pts
                row_dict["Tau_Standings"] = kt_pts
                row_dict["Sp_Pyth"] = sp_pyth
                row_dict["Tau_Pyth"] = kt_pyth
            row_dict["weights"] = w_str
            comparison_rows.append(row_dict)
            count += 1
        print(f"  Size {size}: {count} combos evaluated")
    comp = pd.DataFrame(comparison_rows).sort_values("Brier").reset_index(drop=True)
    for key in keys:
        info = base[key]
        info["rank_df"].to_csv(os.path.join(RESULTS, f"{key}_rankings.csv"), index=False)
        if seed_map:
            preds = make_predictions(seed_map, [info["prob_func"]], (1.0,))
            preds.to_csv(os.path.join(RESULTS, f"{key}_predictions.csv"), index=False)
    top_blends = comp[comp["size"] > 1].head(5)
    for _, row in top_blends.iterrows():
        combo_keys = _parse_combo(row["method"], keys)
        if not combo_keys:
            continue
        probs_list = [base[k]["probs"] for k in combo_keys]
        weights, _ = optimal_blend(y, probs_list)
        rank_dfs = [base[k]["rank_df"] for k in combo_keys]
        rank_cols = [base[k]["rank_col"] for k in combo_keys]
        rankings = make_rankings(rank_dfs, rank_cols)
        name = "_".join(combo_keys)
        rankings.to_csv(os.path.join(RESULTS, f"{name}_rankings.csv"), index=False)
        if seed_map:
            pfuncs = [base[k]["prob_func"] for k in combo_keys]
            preds = make_predictions(seed_map, pfuncs, weights)
            preds.to_csv(os.path.join(RESULTS, f"{name}_predictions.csv"), index=False)
    if has_gt:
        def _norm(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn) if mx > mn else pd.Series([0.5]*len(series))
        comp["rank_q"] = (
            _norm(comp["NDCG"]) +
            _norm(comp["MAP@16"]) +
            _norm(comp["Tau_Pyth"])
        ) / 3.0
        comp["prob_q"] = _norm(1 - comp["Brier"])
        comp["SubScore"] = 0.50 * comp["rank_q"] + 0.50 * comp["prob_q"]
        comp = comp.sort_values("SubScore", ascending=False).reset_index(drop=True)
    comp_out = comp.copy()
    comp_out["AUC"] = comp_out["AUC"].map("{:.4f}".format)
    comp_out["Brier"] = comp_out["Brier"].map("{:.4f}".format)
    metric_cols = ["components", "AUC", "Brier"]
    if has_gt:
        for c in ["NDCG", "MAP@16", "Sp_Standings", "Tau_Standings", "Sp_Pyth", "Tau_Pyth"]:
            comp_out[c] = comp_out[c].map("{:.4f}".format)
        comp_out["SubScore"] = comp_out["SubScore"].map("{:.4f}".format)
        metric_cols += ["NDCG", "MAP@16"]
    comp_out.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)
    if has_gt:
        print(f"\n{'='*100}")
        print("  PYTHAGOREAN EXPECTATION vs LEAGUE TABLE")
        print("  (teams where underlying quality differs most from standings)")
        print(f"{'='*100}")
        lt = pd.read_csv(LEAGUE_TABLE)
        lt["pyth_wpct"] = lt["gf"]**2 / (lt["gf"]**2 + lt["ga"]**2)
        lt = lt.sort_values("pyth_wpct", ascending=False).reset_index(drop=True)
        lt["pyth_rank"] = range(1, len(lt) + 1)
        lt["rank_diff"] = lt["rank"] - lt["pyth_rank"]
        lt_show = lt[["team", "rank", "pyth_rank", "rank_diff", "gf", "ga", "gd", "pts"]].copy()
        lt_show.columns = ["Team", "StdRank", "PythRank", "Δ", "GF", "GA", "GD", "Pts"]
        print(lt_show.to_string(index=False))
    print(f"\n{'='*100}")
    print("  SINGLES — Full Metric Dashboard")
    print(f"{'='*100}")
    s_cols = ["components", "AUC", "Brier"]
    if has_gt:
        s_cols += ["NDCG", "MAP@16", "Sp_Pyth", "Tau_Pyth", "SubScore"]
    s = comp_out[comp_out["size"] == 1][s_cols]
    print(s.to_string(index=False))
    print(f"\n{'='*100}")
    print("  TOP 15 BY SUBMISSION SCORE (composite: 50% ranking + 50% probability)")
    print(f"{'='*100}")
    top_cols = ["components", "Brier", "NDCG", "MAP@16", "Tau_Pyth", "SubScore", "weights"] if has_gt \
        else ["components", "AUC", "Brier", "weights"]
    top = comp_out[top_cols].head(15)
    print(top.to_string(index=False))
    print(f"\n{'='*100}")
    print("  ANALYSIS: BEST METHOD PER METRIC")
    print(f"{'='*100}")
    best_brier_idx = comp["Brier"].idxmin()
    best_auc_idx = comp["AUC"].idxmax()
    print(f"  Best Brier (probability):  {comp.loc[best_brier_idx, 'components']:<40}  Brier={comp.loc[best_brier_idx, 'Brier']:.4f}")
    print(f"  Best AUC (discrimination): {comp.loc[best_auc_idx, 'components']:<40}  AUC={comp.loc[best_auc_idx, 'AUC']:.4f}")
    if has_gt:
        best_ndcg_idx = comp["NDCG"].idxmax()
        best_map_idx = comp["MAP@16"].idxmax()
        best_pyth_idx = comp["Tau_Pyth"].idxmax()
        best_sub_idx = comp["SubScore"].idxmax()
        print(f"  Best NDCG (full ranking):  {comp.loc[best_ndcg_idx, 'components']:<40}  NDCG={comp.loc[best_ndcg_idx, 'NDCG']:.4f}")
        print(f"  Best MAP@16 (playoff ID): {comp.loc[best_map_idx, 'components']:<40}  MAP@16={comp.loc[best_map_idx, 'MAP@16']:.4f}")
        print(f"  Best Tau(Pyth) (strength): {comp.loc[best_pyth_idx, 'components']:<40}  Tau={comp.loc[best_pyth_idx, 'Tau_Pyth']:.4f}")
        print(f"  ★ Best SubScore (overall): {comp.loc[best_sub_idx, 'components']:<40}  SubScore={comp.loc[best_sub_idx, 'SubScore']:.4f}")
    print(f"{'='*100}")
    total = len(comp)
    print(f"\nTotal combos: {total}")
    print(f"Results: {RESULTS}/")

def _parse_combo(method_str, valid_keys):
    parts = method_str.split("+")
    result = []
    for p in parts:
        if p in valid_keys:
            result.append(p)
        else:
            return []
    return result

if __name__ == "__main__":
    main()
