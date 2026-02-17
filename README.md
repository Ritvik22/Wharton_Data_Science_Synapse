# Wharton Data Science Synapse - Competition Submissions

## Project Structure

### Phase 1a: Team Performance Analysis
**Directory:** `phase_1a/`

**Task:** Create power rankings and predict tournament game outcomes

**Approach:**
- Implemented 9 rating methods: 6 Elo variants, Iterative Power, B-score, PARX
- Evaluated all 129 possible combinations (singles, pairs, triples)
- Optimized blend weights using Brier score minimization
- Selected IP + BS (64%/36%) based on composite SubScore metric

**Key Files:**
- `engine.py` - Core evaluation framework for all methods
- `generate_final_submission.py` - Creates final submission file
- `weight_explorer.py` - Exhaustive weight optimization
- `submission.csv` - Final rankings and win probabilities
- `results/comparison.csv` - Full method comparison (129 combinations)

**Submission:**
- 32 team power rankings
- 16 first-round matchup win probabilities
- Method: IP + BS blend (NDCG=0.999, AUC=0.631, Brier=0.237)

---

### Phase 1b: Line Performance Analysis
**Directory:** `phase_1b/`

**Task:** Quantify offensive line quality disparity for each team

**Methodology:**
Applied Phase 1a's comprehensive rating framework to line-level analysis:

1. **Treated each line as independent entity** (64 total: 32 teams × 2 lines)
2. **Implemented 6 rating methods:**
   - Elo-Basic: Standard Elo for line matchups
   - Elo-XG: xG-weighted Elo with MOV adjustment
   - Iterative Power: AGD + opponent strength
   - B-Score: Eigenvector centrality
   - XG/60: Raw performance baseline
   - XG-Diff/60: Net xG differential
3. **Calculated disparity ratios** for each method
4. **Validated and ensembled** using correlation analysis
5. **Selected most reliable method** (Elo-XG) based on internal consistency

**Key Files:**
- `line_engine.py` - Comprehensive multi-method analysis
- `line_disparity_analysis.py` - Simple baseline (XG/60 ratio)
- `refined_ensemble.py` - Method validation and selection
- `compare_methods.py` - Detailed method comparison
- `ANALYSIS_SUMMARY.md` - Complete methodology documentation
- `results/submission_final.csv` - **RECOMMENDED SUBMISSION** (Elo-XG)

**Final Top 10 (Elo-XG Method):**
1. Panama
2. Guatemala
3. Netherlands
4. UK
5. Peru
6. USA
7. New Zealand
8. Iceland
9. South Korea
10. Indonesia

**Method Validation:**
- Elo-Basic vs Elo-XG correlation: **0.962** (high consistency)
- Simple baseline vs Elo-XG: **0.654** (moderate agreement)
- Captures matchup-adjusted disparity, not just raw performance

**Alternative Rankings Available:**
- Simple XG/60 ratio (baseline): Guatemala #1, USA #2
- Refined ensemble (3 methods): Guatemala #1, Panama #2
- Full ensemble (6 methods): USA #1, UK #2

**Interpretation:**
- Elo-XG accounts for opponent quality and temporal dynamics
- Mirrors Phase 1a's rigorous methodology
- Higher disparity = greater reliance on first line

---

## Data Files

**Main Dataset:** `whl_2025 (1).xlsx`
- 25,827 records from 1,312 games
- Line-level granularity with xG, TOI, matchup data

**Aggregated Data:**
- `whl_2025 (1)_team_game_level.csv` - Team game results
- `whl_2025 (1)_game_level.csv` - Game summaries
- `whl_2025 (1)_league_table.csv` - Final standings

---

## Running the Analysis

**Phase 1a:**
```bash
cd phase_1a
python3 engine.py                      # Full method evaluation
python3 generate_final_submission.py   # Create submission
python3 weight_explorer.py             # Weight optimization
```

**Phase 1b:**
```bash
python3 phase_1b/line_engine.py              # Full multi-method analysis
python3 phase_1b/refined_ensemble.py         # Method validation + final selection
python3 phase_1b/line_disparity_analysis.py  # Simple baseline
python3 phase_1b/compare_methods.py          # Detailed comparison
```

---

## Dependencies

- pandas
- numpy
- scipy
- scikit-learn
- openpyxl (for Excel reading)

Install: `pip3 install pandas numpy scipy scikit-learn openpyxl`
