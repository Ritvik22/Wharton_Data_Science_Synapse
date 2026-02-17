import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RESULTS = "phase_1b/results"

print("="*80)
print("PHASE 1B: REFINED ENSEMBLE (Reliable Methods Only)")
print("="*80)

ensemble = pd.read_csv(f"{RESULTS}/ensemble_rankings.csv")
simple = pd.read_csv(f"{RESULTS}/line_disparity_rankings.csv")

reliable_methods = ['Elo-Basic', 'Elo-XG', 'XG/60']

print("\n[ANALYSIS] Using only reliable methods:")
for method in reliable_methods:
    print(f"  - {method}")

ensemble['refined_avg_rank'] = ensemble[reliable_methods].mean(axis=1)
ensemble['refined_median_rank'] = ensemble[reliable_methods].median(axis=1)

ensemble_refined = ensemble.sort_values('refined_avg_rank').reset_index(drop=True)
ensemble_refined['refined_rank'] = range(1, len(ensemble_refined) + 1)

print("\n" + "="*80)
print("REFINED ENSEMBLE TOP 10")
print("="*80)

top10_refined = ensemble_refined.head(10)[['refined_rank', 'team', 'refined_avg_rank'] + reliable_methods]
print(top10_refined.to_string(index=False))

print("\n" + "="*80)
print("COMPARISON: Simple vs Refined Ensemble")
print("="*80)

simple_dict = dict(zip(simple['team'], simple['rank']))
refined_dict = dict(zip(ensemble_refined['team'], ensemble_refined['refined_rank']))

comparison = []
for team in ensemble_refined['team']:
    simple_rank = simple_dict.get(team, None)
    refined_rank = refined_dict.get(team, None)
    
    if simple_rank and refined_rank:
        diff = simple_rank - refined_rank
        comparison.append({
            'team': team,
            'simple_rank': simple_rank,
            'refined_rank': refined_rank,
            'difference': diff
        })

comp_df = pd.DataFrame(comparison)

simple_ranks = [simple_dict.get(team, 99) for team in ensemble_refined['team']]
refined_ranks = ensemble_refined['refined_rank'].tolist()

rho, pval = spearmanr(simple_ranks, refined_ranks)
print(f"\nSpearman correlation: {rho:.4f} (p={pval:.4f})")

print("\nTop 10 Comparison:")
comparison_table = pd.DataFrame({
    'Refined_Rank': ensemble_refined.head(10)['refined_rank'].values,
    'Team': ensemble_refined.head(10)['team'].values,
    'Simple_Rank': [simple_dict.get(team, '-') for team in ensemble_refined.head(10)['team']],
    'Elo-Basic': ensemble_refined.head(10)['Elo-Basic'].values,
    'Elo-XG': ensemble_refined.head(10)['Elo-XG'].values,
    'XG/60': ensemble_refined.head(10)['XG/60'].values,
})
print(comparison_table.to_string(index=False))

print("\n" + "="*80)
print("METHOD AGREEMENT MATRIX")
print("="*80)

print("\nPairwise Spearman Correlations:")
for i, m1 in enumerate(reliable_methods):
    for j, m2 in enumerate(reliable_methods):
        if i < j:
            rho_pair, _ = spearmanr(ensemble[m1], ensemble[m2])
            print(f"  {m1} vs {m2}: {rho_pair:.4f}")

print("\n" + "="*80)
print("STABILITY CHECK")
print("="*80)

ensemble_refined['refined_std'] = ensemble_refined[reliable_methods].std(axis=1)
ensemble_refined['refined_range'] = (ensemble_refined[reliable_methods].max(axis=1) - 
                                      ensemble_refined[reliable_methods].min(axis=1))

print("\nMost Stable Rankings (low variance):")
stable = ensemble_refined.nsmallest(10, 'refined_std')[['refined_rank', 'team', 'refined_std'] + reliable_methods]
print(stable.to_string(index=False))

print("\nMost Volatile Rankings (high variance):")
volatile = ensemble_refined.nlargest(10, 'refined_std')[['refined_rank', 'team', 'refined_std'] + reliable_methods]
print(volatile.to_string(index=False))

print("\n" + "="*80)
print("FINAL DECISION")
print("="*80)

if rho > 0.7:
    print(f"\n✓ HIGH AGREEMENT (r={rho:.3f}) between Simple and Refined methods")
    print("\nRECOMMENDATION: Use Refined Ensemble")
    print("  - Strong validation from simple method")
    print("  - More sophisticated analysis")
    print("  - Accounts for multiple perspectives")
    final = ensemble_refined.head(10)[['refined_rank', 'team']].copy()
    final.columns = ['rank', 'team']
    decision = "refined"
elif rho > 0.4:
    print(f"\n⚠ MODERATE AGREEMENT (r={rho:.3f}) between methods")
    print("\nRECOMMENDATION: Use method with strongest internal consistency")
    
    elo_basic_xg_corr, _ = spearmanr(ensemble['Elo-Basic'], ensemble['Elo-XG'])
    print(f"  - Elo-Basic vs Elo-XG: {elo_basic_xg_corr:.3f}")
    
    if elo_basic_xg_corr > 0.9:
        print("  → Elo methods highly consistent, use Elo-XG ranking")
        elo_xg_ranking = ensemble.sort_values('Elo-XG').reset_index(drop=True)
        elo_xg_ranking['rank'] = range(1, len(elo_xg_ranking) + 1)
        final = elo_xg_ranking.head(10)[['rank', 'team']]
        decision = "elo_xg"
    else:
        print("  → Use Refined Ensemble as compromise")
        final = ensemble_refined.head(10)[['refined_rank', 'team']].copy()
        final.columns = ['rank', 'team']
        decision = "refined"
else:
    print(f"\n✗ LOW AGREEMENT (r={rho:.3f}) between methods")
    print("\nRECOMMENDATION: Use Simple method (most conservative)")
    print("  - Direct interpretation")
    print("  - No assumptions about matchup dynamics")
    print("  - Transparent calculation")
    final = simple.head(10)[['rank', 'team']]
    decision = "simple"

print("\n" + "="*80)
print(f"SELECTED METHOD: {decision.upper()}")
print("="*80)
print(final.to_string(index=False))

final.to_csv(f"{RESULTS}/submission_final.csv", index=False)
print(f"\n✓ Submission saved: {RESULTS}/submission_final.csv")

ensemble_refined.to_csv(f"{RESULTS}/refined_ensemble_full.csv", index=False)
print(f"✓ Full refined rankings: {RESULTS}/refined_ensemble_full.csv")
print("="*80)
