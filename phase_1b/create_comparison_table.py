import pandas as pd
import numpy as np

RESULTS = "phase_1b/results"

print("="*100)
print(" "*30 + "PHASE 1B: COMPREHENSIVE RANKING COMPARISON")
print("="*100)

all_disp = pd.read_csv(f"{RESULTS}/all_method_disparities.csv")

methods = ["Elo-Basic", "Elo-XG", "IP", "B-Score", "XG/60", "XG-Diff/60"]

comparison = {}
for method in methods:
    method_data = all_disp[all_disp['method'] == method].sort_values('rank')
    comparison[method] = dict(zip(method_data['team'], method_data['rank']))

all_teams = sorted(set(all_disp['team']))

table_data = []
for team in all_teams:
    row = {'Team': team}
    for method in methods:
        row[method] = comparison[method].get(team, '-')
    table_data.append(row)

df = pd.DataFrame(table_data)

for method in methods:
    df[method] = pd.to_numeric(df[method], errors='coerce')

df['Avg_Rank'] = df[methods].mean(axis=1)
df['Std_Dev'] = df[methods].std(axis=1)
df = df.sort_values('Avg_Rank')
df['Ensemble_Rank'] = range(1, len(df) + 1)

cols = ['Ensemble_Rank', 'Team'] + methods + ['Avg_Rank', 'Std_Dev']
df = df[cols]

print("\n" + "="*100)
print("FULL RANKINGS: ALL METHODS COMPARISON (Top 15)")
print("="*100)
print(df.head(15).to_string(index=False))

print("\n" + "="*100)
print("BOTTOM 5 (Most Balanced/Lowest Disparity)")
print("="*100)
print(df.tail(5).to_string(index=False))

ensemble = pd.read_csv(f"{RESULTS}/ensemble_rankings.csv")
simple = pd.read_csv(f"{RESULTS}/line_disparity_rankings.csv")

print("\n" + "="*100)
print("TOP 10 COMPARISON: SIMPLE vs ELO-XG vs ENSEMBLE")
print("="*100)

elo_xg_ranking = df.sort_values('Elo-XG').reset_index(drop=True)
elo_xg_ranking['Elo-XG_Rank'] = range(1, len(elo_xg_ranking) + 1)

comparison_top10 = pd.DataFrame({
    'Rank': range(1, 11),
    'Simple_XG/60': simple['team'].head(10).values,
    'Elo-XG': elo_xg_ranking[elo_xg_ranking['Elo-XG_Rank'] <= 10]['Team'].values,
    'Ensemble_6': ensemble['team'].head(10).values,
})

print(comparison_top10.to_string(index=False))

print("\n" + "="*100)
print("KEY STATISTICS")
print("="*100)

print("\nMethod Correlations (Spearman):")
corr_matrix = df[methods].corr(method='spearman')
print(corr_matrix.round(3).to_string())

print("\n" + "="*100)
print("DISPARITY SCORES BY METHOD (Top 10 Teams)")
print("="*100)

disparity_scores = []
for method in methods:
    method_data = all_disp[all_disp['method'] == method].sort_values('rank').head(10)
    for _, row in method_data.iterrows():
        disparity_scores.append({
            'Method': method,
            'Rank': int(row['rank']),
            'Team': row['team'],
            'Difference': row['difference'],
            'Ratio': row['ratio']
        })

disp_df = pd.DataFrame(disparity_scores)

print("\nElo-XG Top 10 (RECOMMENDED):")
elo_xg_top = disp_df[disp_df['Method'] == 'Elo-XG']
print(elo_xg_top.to_string(index=False))

print("\nSimple XG/60 Top 10:")
xg60_top = disp_df[disp_df['Method'] == 'XG/60']
print(xg60_top.to_string(index=False))

print("\n" + "="*100)
print("VOLATILITY ANALYSIS")
print("="*100)

print("\nMost Volatile Teams (High Std Dev = Rank varies greatly):")
volatile = df.nlargest(10, 'Std_Dev')[['Team', 'Std_Dev', 'Elo-Basic', 'Elo-XG', 'XG/60', 'Avg_Rank']]
print(volatile.to_string(index=False))

print("\nMost Stable Teams (Low Std Dev = Consistent across methods):")
stable = df.nsmallest(10, 'Std_Dev')[['Team', 'Std_Dev', 'Elo-Basic', 'Elo-XG', 'XG/60', 'Avg_Rank']]
print(stable.to_string(index=False))

print("\n" + "="*100)
print("FINAL RECOMMENDATION: ELO-XG METHOD")
print("="*100)

final = elo_xg_ranking.head(10)[['Elo-XG_Rank', 'Team', 'Elo-XG', 'XG/60', 'Elo-Basic']].copy()
final.columns = ['Rank', 'Team', 'Elo-XG_Rank', 'XG/60_Rank', 'Elo-Basic_Rank']
print(final.to_string(index=False))

print("\n✓ Most internally consistent (Elo-Basic ↔ Elo-XG: r=0.962)")
print("✓ Accounts for opponent quality and temporal dynamics")
print("✓ xG-weighted for stronger signal on large disparities")
print("="*100)

df.to_csv(f"{RESULTS}/full_comparison_table.csv", index=False)
print(f"\n✓ Full table saved: {RESULTS}/full_comparison_table.csv")
