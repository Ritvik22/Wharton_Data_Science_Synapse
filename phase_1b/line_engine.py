import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
DATA_PATH = os.path.join(PARENT, "whl_2025 (1).xlsx")
RESULTS = os.path.join(HERE, "results")

INITIAL_ELO = 1500
K_FACTOR = 20

os.makedirs(RESULTS, exist_ok=True)

print("="*80)
print("PHASE 1B: LINE ENGINE - Comprehensive Rating-Based Analysis")
print("="*80)

def load_line_data():
    print("\n[LOAD] Reading Excel data...")
    df = pd.read_excel(DATA_PATH)
    
    even_strength_lines = ['first_off', 'second_off']
    even_strength_def = ['first_def', 'second_def']
    
    df_es = df[
        df['home_off_line'].isin(even_strength_lines) & 
        df['away_off_line'].isin(even_strength_lines) &
        df['home_def_pairing'].isin(even_strength_def) &
        df['away_def_pairing'].isin(even_strength_def)
    ].copy()
    
    print(f"[LOAD] Total records: {len(df):,}")
    print(f"[LOAD] Even-strength records: {len(df_es):,} ({len(df_es)/len(df)*100:.1f}%)")
    
    home = df_es[['game_id', 'home_team', 'home_off_line', 'away_team', 'away_def_pairing',
                   'toi', 'home_xg', 'home_goals', 'away_xg']].copy()
    home.columns = ['game_id', 'team', 'off_line', 'opp_team', 'opp_def',
                    'toi', 'xg_for', 'goals_for', 'xg_against']
    
    away = df_es[['game_id', 'away_team', 'away_off_line', 'home_team', 'home_def_pairing',
                   'toi', 'away_xg', 'away_goals', 'home_xg']].copy()
    away.columns = ['game_id', 'team', 'off_line', 'opp_team', 'opp_def',
                    'toi', 'xg_for', 'goals_for', 'xg_against']
    
    matchups = pd.concat([home, away], ignore_index=True)
    matchups['line_id'] = matchups['team'] + '_' + matchups['off_line']
    matchups['xg_diff'] = matchups['xg_for'] - matchups['xg_against']
    matchups['dominance'] = (matchups['xg_for'] > matchups['xg_against']).astype(int)
    
    matchups = matchups.sort_values('game_id').reset_index(drop=True)
    
    teams = sorted(matchups['team'].unique())
    lines = sorted(matchups['line_id'].unique())
    
    print(f"[LOAD] {len(teams)} teams, {len(lines)} unique lines")
    print(f"[LOAD] {len(matchups)} line matchups")
    
    return matchups, teams, lines

matchups, teams, lines = load_line_data()
y = matchups['dominance'].values

print("\n" + "="*80)
print("METHOD 1: ELO RATINGS")
print("="*80)

def run_elo_basic(matchups):
    elo = {line: float(INITIAL_ELO) for line in lines}
    ratings_over_time = []
    
    for idx, row in matchups.iterrows():
        line = row['line_id']
        xg_for = row['xg_for']
        xg_against = row['xg_against']
        
        expected = 1 / (1 + 10 ** ((INITIAL_ELO - elo[line]) / 400))
        actual = 1 if xg_for > xg_against else 0
        
        elo[line] += K_FACTOR * (actual - expected)
        
        ratings_over_time.append({
            'idx': idx,
            'line': line,
            'rating': elo[line]
        })
    
    print(f"[ELO-BASIC] Computed ratings for {len(elo)} lines")
    return elo, ratings_over_time

def run_elo_xg_weighted(matchups):
    elo = {line: float(INITIAL_ELO) for line in lines}
    
    for idx, row in matchups.iterrows():
        line = row['line_id']
        xg_diff = row['xg_diff']
        
        expected = 1 / (1 + 10 ** ((INITIAL_ELO - elo[line]) / 400))
        actual = 1 if xg_diff > 0 else 0
        
        k_mult = max(np.log(abs(xg_diff) + 1) * 1.5, 0.5)
        k = K_FACTOR * k_mult
        
        elo[line] += k * (actual - expected)
    
    print(f"[ELO-XG] Computed xG-weighted Elo for {len(elo)} lines")
    return elo

elo_basic, elo_history = run_elo_basic(matchups)
elo_xg = run_elo_xg_weighted(matchups)

print("\n" + "="*80)
print("METHOD 2: ITERATIVE POWER")
print("="*80)

def compute_iterative_power(matchups, epochs=15):
    line_matchups = {}
    for line in lines:
        line_matchups[line] = []
    
    for _, row in matchups.iterrows():
        line = row['line_id']
        opp_team = row['opp_team']
        opp_def = row['opp_def']
        xg_diff = row['xg_diff']
        
        line_matchups[line].append({
            'opp_team': opp_team,
            'opp_def': opp_def,
            'xg_diff': xg_diff
        })
    
    avg_xg_diff = {}
    for line in lines:
        if line_matchups[line]:
            avg_xg_diff[line] = np.mean([m['xg_diff'] for m in line_matchups[line]])
        else:
            avg_xg_diff[line] = 0.0
    
    power = {line: 0.0 for line in lines}
    
    for epoch in range(epochs):
        prev_power = dict(power)
        
        for line in lines:
            team = '_'.join(line.split('_')[:-1])
            
            opp_strengths = []
            for match in line_matchups[line]:
                opp_team = match['opp_team']
                opp_def = match['opp_def']
                opp_line_id = f"{opp_team}_{opp_def}"
                if opp_line_id in prev_power:
                    opp_strengths.append(prev_power[opp_line_id])
            
            sos = np.mean(opp_strengths) if opp_strengths else 0
            power[line] = avg_xg_diff[line] + 0.3 * sos
    
    print(f"[IP] Converged after {epochs} epochs")
    return power

ip_ratings = compute_iterative_power(matchups, epochs=15)

print("\n" + "="*80)
print("METHOD 3: B-SCORE (Eigenvector Centrality)")
print("="*80)

def compute_bscore(matchups, alpha=100):
    n = len(lines)
    line_idx = {line: i for i, line in enumerate(lines)}
    
    W = np.zeros((n, n))
    
    total_matchups = len(matchups)
    for mi, (_, row) in enumerate(matchups.iterrows()):
        line = row['line_id']
        opp_team = row['opp_team']
        opp_def = row['opp_def']
        opp_line = f"{opp_team}_{opp_def}"
        
        if opp_line in line_idx:
            i = line_idx[line]
            j = line_idx[opp_line]
            
            decay = 1.0 / (1.0 + (total_matchups - 1 - mi) / alpha)
            
            if row['xg_for'] > row['xg_against']:
                W[j, i] += decay * abs(row['xg_diff'])
            elif row['xg_for'] < row['xg_against']:
                W[i, j] += decay * abs(row['xg_diff'])
    
    Wt = W.T + 1e-8
    evals, evecs = np.linalg.eig(Wt)
    v = np.real(evecs[:, np.argmax(np.real(evals))])
    
    if v.sum() < 0:
        v = -v
    v /= np.linalg.norm(v)
    
    bscore = {lines[i]: v[i] for i in range(n)}
    
    print(f"[BS] Eigenvector centrality computed (alpha={alpha})")
    return bscore

bs_ratings = compute_bscore(matchups, alpha=100)

print("\n" + "="*80)
print("METHOD 4: XG PER 60 (Baseline)")
print("="*80)

line_stats = matchups.groupby('line_id').agg({
    'xg_for': 'sum',
    'xg_against': 'sum',
    'toi': 'sum',
    'goals_for': 'sum'
}).reset_index()

line_stats['xg_per_60'] = (line_stats['xg_for'] / line_stats['toi']) * 3600
line_stats['xg_diff_per_60'] = ((line_stats['xg_for'] - line_stats['xg_against']) / line_stats['toi']) * 3600

xg_per_60_ratings = dict(zip(line_stats['line_id'], line_stats['xg_per_60']))
xg_diff_per_60_ratings = dict(zip(line_stats['line_id'], line_stats['xg_diff_per_60']))

print(f"[XG/60] Baseline metrics computed for {len(xg_per_60_ratings)} lines")

print("\n" + "="*80)
print("CALCULATE DISPARITY RATIOS")
print("="*80)

def calculate_disparity(ratings, method_name):
    team_disparities = []
    
    for team in teams:
        first_line = f"{team}_first_off"
        second_line = f"{team}_second_off"
        
        if first_line in ratings and second_line in ratings:
            first_rating = ratings[first_line]
            second_rating = ratings[second_line]
            
            if second_rating > 0:
                ratio = first_rating / second_rating
            elif second_rating < 0:
                ratio = first_rating / second_rating if first_rating < 0 else -first_rating / second_rating
            else:
                ratio = 1.0
            
            diff = first_rating - second_rating
            
            team_disparities.append({
                'team': team,
                'method': method_name,
                'first_rating': first_rating,
                'second_rating': second_rating,
                'difference': diff,
                'ratio': ratio,
                'abs_ratio': abs(ratio)
            })
    
    df = pd.DataFrame(team_disparities)
    df = df.sort_values('difference', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df

disparity_elo_basic = calculate_disparity(elo_basic, "Elo-Basic")
disparity_elo_xg = calculate_disparity(elo_xg, "Elo-XG")
disparity_ip = calculate_disparity(ip_ratings, "IP")
disparity_bs = calculate_disparity(bs_ratings, "B-Score")
disparity_xg60 = calculate_disparity(xg_per_60_ratings, "XG/60")
disparity_xg_diff = calculate_disparity(xg_diff_per_60_ratings, "XG-Diff/60")

all_disparities = pd.concat([
    disparity_elo_basic,
    disparity_elo_xg,
    disparity_ip,
    disparity_bs,
    disparity_xg60,
    disparity_xg_diff
], ignore_index=True)

all_disparities.to_csv(os.path.join(RESULTS, "all_method_disparities.csv"), index=False)

print(f"[DISPARITY] Calculated for 6 methods across {len(teams)} teams")

print("\n" + "="*80)
print("ENSEMBLE RANKING")
print("="*80)

rank_matrix = pd.pivot_table(
    all_disparities,
    values='rank',
    index='team',
    columns='method',
    aggfunc='first'
).reset_index()

rank_matrix['avg_rank'] = rank_matrix[['Elo-Basic', 'Elo-XG', 'IP', 'B-Score', 'XG/60', 'XG-Diff/60']].mean(axis=1)
rank_matrix['median_rank'] = rank_matrix[['Elo-Basic', 'Elo-XG', 'IP', 'B-Score', 'XG/60', 'XG-Diff/60']].median(axis=1)

rank_matrix = rank_matrix.sort_values('avg_rank').reset_index(drop=True)
rank_matrix['ensemble_rank'] = range(1, len(rank_matrix) + 1)

rank_matrix.to_csv(os.path.join(RESULTS, "ensemble_rankings.csv"), index=False)

print(f"[ENSEMBLE] Combined rankings for {len(rank_matrix)} teams")

print("\n" + "="*80)
print("TOP 10 TEAMS BY METHOD")
print("="*80)

methods = ["Elo-Basic", "Elo-XG", "IP", "B-Score", "XG/60", "XG-Diff/60"]
for method in methods:
    print(f"\n{method}:")
    method_data = all_disparities[all_disparities['method'] == method].head(10)
    print(method_data[['rank', 'team', 'difference', 'ratio']].to_string(index=False))

print("\n" + "="*80)
print("ENSEMBLE TOP 10")
print("="*80)

ensemble_top10 = rank_matrix.head(10)
print(ensemble_top10[['ensemble_rank', 'team', 'avg_rank', 'median_rank'] + methods].to_string(index=False))

submission = ensemble_top10[['ensemble_rank', 'team']].copy()
submission.columns = ['rank', 'team']
submission.to_csv(os.path.join(RESULTS, "submission_ensemble.csv"), index=False)

print("\n" + "="*80)
print("METHOD COMPARISON")
print("="*80)

method_corr = rank_matrix[methods].corr(method='spearman')
print("\nSpearman Rank Correlations:")
print(method_corr.round(3))

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print(f"✓ All method disparities: {RESULTS}/all_method_disparities.csv")
print(f"✓ Ensemble rankings: {RESULTS}/ensemble_rankings.csv")
print(f"✓ Submission (ensemble top 10): {RESULTS}/submission_ensemble.csv")
print("="*80)
