import pandas as pd
import numpy as np

DATA_PATH = "whl_2025 (1).xlsx"
OUTPUT_PATH = "phase_1b/results/line_disparity_rankings.csv"

print("="*80)
print("PHASE 1B: OFFENSIVE LINE QUALITY DISPARITY ANALYSIS")
print("="*80)

df = pd.read_excel(DATA_PATH)
print(f"\nLoaded {len(df):,} records from {df['game_id'].nunique()} games")
print(f"Teams: {df['home_team'].nunique()}")

print("\n" + "="*80)
print("STEP 1: Filter to Even-Strength Situations")
print("="*80)

even_strength_lines = ['first_off', 'second_off']
df_es = df[df['home_off_line'].isin(even_strength_lines) & 
           df['away_off_line'].isin(even_strength_lines)].copy()

print(f"Even-strength records: {len(df_es):,} ({len(df_es)/len(df)*100:.1f}% of total)")

home_records = df_es[['game_id', 'home_team', 'home_off_line', 'away_def_pairing', 
                       'toi', 'home_xg', 'home_goals']].copy()
home_records.columns = ['game_id', 'team', 'off_line', 'opp_def_pairing', 'toi', 'xg', 'goals']

away_records = df_es[['game_id', 'away_team', 'away_off_line', 'home_def_pairing', 
                       'toi', 'away_xg', 'away_goals']].copy()
away_records.columns = ['game_id', 'team', 'off_line', 'opp_def_pairing', 'toi', 'xg', 'goals']

all_records = pd.concat([home_records, away_records], ignore_index=True)

print(f"\nTotal team-line records: {len(all_records):,}")
print(f"First line records: {(all_records['off_line'] == 'first_off').sum():,}")
print(f"Second line records: {(all_records['off_line'] == 'second_off').sum():,}")

print("\n" + "="*80)
print("STEP 2: Calculate Defensive Pairing Strength")
print("="*80)

def_strength = all_records.groupby('opp_def_pairing').agg({
    'xg': 'sum',
    'goals': 'sum',
    'toi': 'sum'
}).reset_index()
def_strength['xg_against_per_min'] = def_strength['xg'] / (def_strength['toi'] / 60)
def_strength['goals_against_per_min'] = def_strength['goals'] / (def_strength['toi'] / 60)

league_avg_xg_per_min = all_records['xg'].sum() / (all_records['toi'].sum() / 60)
def_strength['def_quality'] = def_strength['xg_against_per_min'] / league_avg_xg_per_min

def_quality_map = dict(zip(def_strength['opp_def_pairing'], def_strength['def_quality']))
all_records['opp_def_quality'] = all_records['opp_def_pairing'].map(def_quality_map)

print(f"League average xG per minute: {league_avg_xg_per_min:.4f}")
print(f"\nTop 5 toughest defensive pairings (high xG allowed = weak defense):")
print(def_strength.nlargest(5, 'xg_against_per_min')[['opp_def_pairing', 'xg_against_per_min', 'def_quality']])
print(f"\nTop 5 strongest defensive pairings (low xG allowed = strong defense):")
print(def_strength.nsmallest(5, 'xg_against_per_min')[['opp_def_pairing', 'xg_against_per_min', 'def_quality']])

print("\n" + "="*80)
print("STEP 3: Calculate Line Performance Metrics")
print("="*80)

line_stats = all_records.groupby(['team', 'off_line']).agg({
    'xg': 'sum',
    'goals': 'sum',
    'toi': 'sum',
    'opp_def_quality': 'mean'
}).reset_index()

line_stats['xg_per_60'] = (line_stats['xg'] / line_stats['toi']) * 3600
line_stats['goals_per_60'] = (line_stats['goals'] / line_stats['toi']) * 3600

line_stats['adjusted_xg_per_60'] = line_stats['xg_per_60'] / line_stats['opp_def_quality']

print(f"\nLine statistics computed for {len(line_stats)} team-line combinations")
print(f"Teams: {line_stats['team'].nunique()}")

first_line = line_stats[line_stats['off_line'] == 'first_off'].copy()
second_line = line_stats[line_stats['off_line'] == 'second_off'].copy()

print(f"\nFirst line records: {len(first_line)}")
print(f"Second line records: {len(second_line)}")

print("\n" + "="*80)
print("STEP 4: Calculate Line Disparity Ratios")
print("="*80)

first_line = first_line.rename(columns={
    'xg': 'first_xg',
    'goals': 'first_goals',
    'toi': 'first_toi',
    'xg_per_60': 'first_xg_per_60',
    'goals_per_60': 'first_goals_per_60',
    'adjusted_xg_per_60': 'first_adj_xg_per_60',
    'opp_def_quality': 'first_opp_def_quality'
})

second_line = second_line.rename(columns={
    'xg': 'second_xg',
    'goals': 'second_goals',
    'toi': 'second_toi',
    'xg_per_60': 'second_xg_per_60',
    'goals_per_60': 'second_goals_per_60',
    'adjusted_xg_per_60': 'second_adj_xg_per_60',
    'opp_def_quality': 'second_opp_def_quality'
})

disparity = first_line[['team', 'first_xg', 'first_toi', 'first_xg_per_60', 
                        'first_goals_per_60', 'first_adj_xg_per_60', 
                        'first_opp_def_quality']].merge(
    second_line[['team', 'second_xg', 'second_toi', 'second_xg_per_60', 
                 'second_goals_per_60', 'second_adj_xg_per_60',
                 'second_opp_def_quality']],
    on='team',
    how='inner'
)

disparity['disparity_ratio'] = disparity['first_adj_xg_per_60'] / disparity['second_adj_xg_per_60']
disparity['raw_xg_ratio'] = disparity['first_xg_per_60'] / disparity['second_xg_per_60']

disparity = disparity.sort_values('disparity_ratio', ascending=False).reset_index(drop=True)
disparity['rank'] = range(1, len(disparity) + 1)

print(f"\nDisparity ratios calculated for {len(disparity)} teams")
print(f"Mean disparity ratio: {disparity['disparity_ratio'].mean():.3f}")
print(f"Median disparity ratio: {disparity['disparity_ratio'].median():.3f}")

print("\n" + "="*80)
print("TOP 10 TEAMS WITH LARGEST OFFENSIVE LINE QUALITY DISPARITY")
print("="*80)

top10 = disparity.head(10)[['rank', 'team', 'disparity_ratio', 'raw_xg_ratio',
                             'first_adj_xg_per_60', 'second_adj_xg_per_60',
                             'first_toi', 'second_toi']]

print(top10.to_string(index=False))

print("\n" + "="*80)
print("BOTTOM 10 TEAMS (Smallest Disparity - Most Balanced Lines)")
print("="*80)

bottom10 = disparity.tail(10)[['rank', 'team', 'disparity_ratio', 'raw_xg_ratio',
                                'first_adj_xg_per_60', 'second_adj_xg_per_60',
                                'first_toi', 'second_toi']]

print(bottom10.to_string(index=False))

print("\n" + "="*80)
print("SAVE RESULTS")
print("="*80)

submission = disparity.head(10)[['rank', 'team', 'disparity_ratio']].copy()
submission.to_csv(OUTPUT_PATH, index=False)

full_results = disparity.copy()
full_results.to_csv("phase_1b/results/full_line_analysis.csv", index=False)

print(f"✓ Top 10 submission saved to: {OUTPUT_PATH}")
print(f"✓ Full analysis saved to: phase_1b/results/full_line_analysis.csv")

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print(f"\nMetric: Adjusted xG per 60 minutes")
print(f"  - Accounts for time on ice (per 60 min rate)")
print(f"  - Adjusts for defensive matchup strength")
print(f"  - Disparity Ratio = (1st line adjusted xG/60) / (2nd line adjusted xG/60)")
print(f"\nInterpretation:")
print(f"  - Higher ratio = Greater disparity (1st line much better than 2nd)")
print(f"  - Lower ratio = More balanced (lines perform similarly)")
print("="*80)
