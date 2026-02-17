import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RESULTS = "phase_1b/results"

print("="*80)
print("GENERATING PHASE 1B: comparison.csv")
print("="*80)

all_disp = pd.read_csv(f"{RESULTS}/all_method_disparities.csv")

methods = ["Elo-Basic", "Elo-XG", "IP", "B-Score", "XG/60", "XG-Diff/60"]

comparison_rows = []

for method in methods:
    method_data = all_disp[all_disp['method'] == method].copy()
    
    mean_diff = method_data['difference'].mean()
    median_diff = method_data['difference'].median()
    std_diff = method_data['difference'].std()
    
    mean_ratio = method_data['ratio'].mean()
    median_ratio = method_data['ratio'].median()
    
    top3_teams = method_data.nsmallest(3, 'rank')['team'].tolist()
    top3_str = " > ".join(top3_teams)
    
    rank_corrs = {}
    for other_method in methods:
        if other_method != method:
            other_data = all_disp[all_disp['method'] == other_method].copy()
            
            merged = method_data[['team', 'rank']].merge(
                other_data[['team', 'rank']], 
                on='team', 
                suffixes=('_1', '_2')
            )
            
            if len(merged) > 0:
                rho, _ = spearmanr(merged['rank_1'], merged['rank_2'])
                rank_corrs[other_method] = rho
    
    avg_corr = np.mean(list(rank_corrs.values()))
    max_corr = max(rank_corrs.values())
    min_corr = min(rank_corrs.values())
    
    most_similar = max(rank_corrs, key=rank_corrs.get)
    least_similar = min(rank_corrs, key=rank_corrs.get)
    
    rank_stability = method_data.groupby('team')['rank'].std().mean()
    
    row = {
        'method': method,
        'mean_difference': mean_diff,
        'median_difference': median_diff,
        'std_difference': std_diff,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'avg_correlation': avg_corr,
        'max_correlation': max_corr,
        'min_correlation': min_corr,
        'most_similar_method': most_similar,
        'least_similar_method': least_similar,
        'top_3_teams': top3_str,
        'reliability_score': avg_corr
    }
    
    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)

comparison = comparison.sort_values('reliability_score', ascending=False)
comparison['reliability_rank'] = range(1, len(comparison) + 1)

cols_order = [
    'reliability_rank', 'method', 'mean_difference', 'median_difference', 
    'std_difference', 'mean_ratio', 'median_ratio', 'avg_correlation',
    'max_correlation', 'min_correlation', 'most_similar_method',
    'least_similar_method', 'top_3_teams', 'reliability_score'
]

comparison = comparison[cols_order]

comparison.to_csv(f"{RESULTS}/comparison.csv", index=False)

print("\n" + "="*80)
print("METHOD COMPARISON SUMMARY")
print("="*80)
print(comparison[['reliability_rank', 'method', 'avg_correlation', 'top_3_teams']].to_string(index=False))

print("\n" + "="*80)
print("DETAILED METRICS")
print("="*80)

for _, row in comparison.iterrows():
    print(f"\n{row['method']} (Rank #{int(row['reliability_rank'])}):")
    print(f"  Mean Difference: {row['mean_difference']:.4f}")
    print(f"  Mean Ratio: {row['mean_ratio']:.4f}")
    print(f"  Avg Correlation: {row['avg_correlation']:.4f}")
    print(f"  Most Similar: {row['most_similar_method']} (r={row['max_correlation']:.3f})")
    print(f"  Least Similar: {row['least_similar_method']} (r={row['min_correlation']:.3f})")
    print(f"  Top 3 Teams: {row['top_3_teams']}")

print("\n" + "="*80)
print("RELIABILITY RANKING")
print("="*80)
print("\nMethods ordered by average correlation with other methods:")
print("(Higher correlation = more consensus with other approaches)")
print()

for i, (_, row) in enumerate(comparison.iterrows(), 1):
    status = "✓ RELIABLE" if row['avg_correlation'] > 0.6 else "⚠ MODERATE" if row['avg_correlation'] > 0.3 else "✗ UNRELIABLE"
    print(f"{i}. {row['method']:<15} - Avg Corr: {row['avg_correlation']:.3f}  {status}")

print("\n" + "="*80)
print("✓ Saved: phase_1b/results/comparison.csv")
print("="*80)

print("\nRECOMMENDATION:")
top_method = comparison.iloc[0]
print(f"  Use {top_method['method']} (highest reliability score: {top_method['reliability_score']:.3f})")
print(f"  Most similar to: {top_method['most_similar_method']}")
print(f"  Top 3 teams: {top_method['top_3_teams']}")
print("="*80)
