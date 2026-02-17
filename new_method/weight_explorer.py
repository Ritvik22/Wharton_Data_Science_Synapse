import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

game = pd.read_csv("new_method/whl_2025_raw.csv").groupby("game_id").agg(
    home_team=("home_team", "first"),
    away_team=("away_team", "first"),
    home_goals=("home_goals", "sum"),
    away_goals=("away_goals", "sum"),
).reset_index()
game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
game["_order"] = game["game_id"].str.extract(r"game_(\d+)", expand=False).astype(int)
game = game.sort_values("_order").reset_index(drop=True)
y = game["home_win"].values

print(f"Loaded {len(game)} games\n")

def elo_expected(elo_h, elo_a, home_adv=75):
    q = 10 ** ((elo_h + home_adv - elo_a) / 400)
    return q / (1 + q)

def run_elo_basic(game):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    elo = {t: 1500.0 for t in teams}
    probs = []
    for _, r in game.iterrows():
        h, a = r["home_team"], r["away_team"]
        p = elo_expected(elo[h], elo[a])
        probs.append(p)
        hw = int(r["home_win"])
        elo[h] += 20 * (hw - p)
        elo[a] += 20 * ((1 - hw) - (1 - p))
    return np.array(probs), elo

def run_elo_mov(game):
    teams = sorted(set(game["home_team"]) | set(game["away_team"]))
    elo = {t: 1500.0 for t in teams}
    probs = []
    for _, r in game.iterrows():
        h, a = r["home_team"], r["away_team"]
        p = elo_expected(elo[h], elo[a])
        probs.append(p)
        hw = int(r["home_win"])
        gd = abs(r["home_goals"] - r["away_goals"])
        diff = abs(elo[h] - elo[a])
        k = 20 * max(np.log(gd + 1) * (2.2 / (diff * 0.001 + 2.2)), 0.5)
        elo[h] += k * (hw - p)
        elo[a] += k * ((1 - hw) - (1 - p))
    return np.array(probs), elo

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
    from scipy.optimize import curve_fit
    x = np.array([power.get(r["home_team"], 0) - power.get(r["away_team"], 0)
                   for _, r in game.iterrows()], dtype=float)
    y_fit = np.array([1 if r["home_goals"] > r["away_goals"] else 0
                       for _, r in game.iterrows()], dtype=float)
    (sigma,), _ = curve_fit(lambda t, s: 1 / (1 + np.exp(-t / s)), x, y_fit, p0=[1.0], maxfev=10000)
    sigma = max(abs(sigma), 1e-6)
    probs = np.array([1 / (1 + np.exp(-(power.get(h, 0) - power.get(a, 0)) / sigma))
                      for h, a in zip(game["home_team"], game["away_team"])])
    return probs, power

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
    bscore = {teams[i]: v[i] for i in range(n)}
    from scipy.optimize import curve_fit
    x = np.array([bscore.get(r["home_team"], 0) - bscore.get(r["away_team"], 0)
                   for _, r in game.iterrows()], dtype=float)
    y_fit = np.array([1 if r["home_goals"] > r["away_goals"] else 0
                       for _, r in game.iterrows()], dtype=float)
    (sigma,), _ = curve_fit(lambda t, s: 1 / (1 + np.exp(-t / s)), x, y_fit, p0=[1.0], maxfev=10000)
    sigma = max(abs(sigma), 1e-6)
    probs = np.array([1 / (1 + np.exp(-(bscore.get(h, 0) - bscore.get(a, 0)) / sigma))
                      for h, a in zip(game["home_team"], game["away_team"])])
    return probs, bscore

print("Computing base method probabilities...")
elo_basic_probs, _ = run_elo_basic(game)
elo_mov_probs, _ = run_elo_mov(game)
ip_probs, _ = compute_ip(game)
bs_probs, _ = compute_bscore(game)

print(f"  Elo(basic): AUC={roc_auc_score(y, elo_basic_probs):.4f}, Brier={brier_score_loss(y, elo_basic_probs):.4f}")
print(f"  Elo(538-MOV): AUC={roc_auc_score(y, elo_mov_probs):.4f}, Brier={brier_score_loss(y, elo_mov_probs):.4f}")
print(f"  IP: AUC={roc_auc_score(y, ip_probs):.4f}, Brier={brier_score_loss(y, ip_probs):.4f}")
print(f"  BS: AUC={roc_auc_score(y, bs_probs):.4f}, Brier={brier_score_loss(y, bs_probs):.4f}")
print()

def explore_2way(name1, name2, probs1, probs2, step=0.001):
    results = []
    for w in np.arange(0, 1.0001, step):
        blend = w * probs1 + (1 - w) * probs2
        auc = roc_auc_score(y, blend)
        brier = brier_score_loss(y, blend)
        results.append({
            f"{name1}_weight": w,
            f"{name2}_weight": 1 - w,
            "AUC": auc,
            "Brier": brier,
        })
    df = pd.DataFrame(results)
    df = df.sort_values("Brier").reset_index(drop=True)
    return df

print("="*80)
print("EXHAUSTIVE 2-WAY ANALYSIS: Elo(538-MOV) + IP")
print("="*80)
print("Testing 1001 weight combinations (0.1% increments)...\n")

df_elo_ip = explore_2way("Elo(538-MOV)", "IP", elo_mov_probs, ip_probs, step=0.001)

print("TOP 20 WEIGHT COMBINATIONS:")
print(df_elo_ip.head(20).to_string(index=False))
print()

print("BOTTOM 5 (worst combinations):")
print(df_elo_ip.tail(5).to_string(index=False))
print()

df_elo_ip.to_csv("new_method/results/weight_analysis_elo_mov_ip.csv", index=False)
print(f"✓ Saved all 1001 combinations to: new_method/results/weight_analysis_elo_mov_ip.csv\n")

def explore_3way(name1, name2, name3, probs1, probs2, probs3, step=0.01):
    results = []
    count = 0
    for w1 in np.arange(0, 1.0001, step):
        for w2 in np.arange(0, 1.0001 - w1, step):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6:
                continue
            w3 = max(0, w3)
            blend = w1 * probs1 + w2 * probs2 + w3 * probs3
            auc = roc_auc_score(y, blend)
            brier = brier_score_loss(y, blend)
            results.append({
                f"{name1}_weight": w1,
                f"{name2}_weight": w2,
                f"{name3}_weight": w3,
                "AUC": auc,
                "Brier": brier,
            })
            count += 1
    df = pd.DataFrame(results)
    df = df.sort_values("Brier").reset_index(drop=True)
    print(f"  Tested {count} valid combinations\n")
    return df

print("="*80)
print("FINE-GRAINED 3-WAY ANALYSIS: Elo(538-MOV) + IP + BS")
print("="*80)
print("Testing all valid weight combinations (1% increments)...")

df_3way = explore_3way("Elo(538-MOV)", "IP", "BS", elo_mov_probs, ip_probs, bs_probs, step=0.01)

print("TOP 20 WEIGHT COMBINATIONS:")
print(df_3way.head(20).to_string(index=False))
print()

df_3way.to_csv("new_method/results/weight_analysis_elo_mov_ip_bs.csv", index=False)
print(f"✓ Saved all combinations to: new_method/results/weight_analysis_elo_mov_ip_bs.csv\n")

print("="*80)
print("EXHAUSTIVE 2-WAY ANALYSIS: Elo(basic) + IP")
print("="*80)
print("Testing 1001 weight combinations (0.1% increments)...\n")

df_elo_basic_ip = explore_2way("Elo(basic)", "IP", elo_basic_probs, ip_probs, step=0.001)

print("TOP 20 WEIGHT COMBINATIONS:")
print(df_elo_basic_ip.head(20).to_string(index=False))
print()

df_elo_basic_ip.to_csv("new_method/results/weight_analysis_elo_basic_ip.csv", index=False)
print(f"✓ Saved all 1001 combinations to: new_method/results/weight_analysis_elo_basic_ip.csv\n")

print("="*80)
print("SUMMARY: OPTIMAL WEIGHTS")
print("="*80)

best_elo_ip = df_elo_ip.iloc[0]
print(f"Elo(538-MOV) + IP:")
print(f"  Weights: {best_elo_ip['Elo(538-MOV)_weight']:.1%} / {best_elo_ip['IP_weight']:.1%}")
print(f"  Brier: {best_elo_ip['Brier']:.6f}")
print(f"  AUC: {best_elo_ip['AUC']:.4f}")
print()

best_elo_basic_ip = df_elo_basic_ip.iloc[0]
print(f"Elo(basic) + IP:")
print(f"  Weights: {best_elo_basic_ip['Elo(basic)_weight']:.1%} / {best_elo_basic_ip['IP_weight']:.1%}")
print(f"  Brier: {best_elo_basic_ip['Brier']:.6f}")
print(f"  AUC: {best_elo_basic_ip['AUC']:.4f}")
print()

best_3way = df_3way.iloc[0]
print(f"Elo(538-MOV) + IP + BS:")
print(f"  Weights: {best_3way['Elo(538-MOV)_weight']:.1%} / {best_3way['IP_weight']:.1%} / {best_3way['BS_weight']:.1%}")
print(f"  Brier: {best_3way['Brier']:.6f}")
print(f"  AUC: {best_3way['AUC']:.4f}")
print()

print("="*80)
print("ALTERNATIVE GOOD SOLUTIONS (within 0.0001 of optimal Brier)")
print("="*80)

threshold = best_elo_ip['Brier'] + 0.0001
alternatives = df_elo_ip[df_elo_ip['Brier'] <= threshold]
print(f"\nElo(538-MOV) + IP: {len(alternatives)} near-optimal combinations")
print(alternatives.head(10).to_string(index=False))

print("\n" + "="*80)
print("✓ All weight analysis files saved to new_method/results/")
print("="*80)
