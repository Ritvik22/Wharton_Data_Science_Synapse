# Phase 1b: Line Disparity Analysis - Summary

## Objective
Quantify offensive line quality disparity for 32 teams by comparing first line vs second line performance.

---

## Methodology

### Approach 1: Simple XG/60 Ratio (Baseline)
**File:** `line_disparity_analysis.py`

- **Metric:** Adjusted xG per 60 minutes
- **Adjustments:**
  - Normalized for time on ice (per 60 min rate)
  - Adjusted for defensive matchup strength
- **Disparity:** Ratio of first line to second line adjusted xG/60

**Top 3:**
1. Guatemala (1.362)
2. USA (1.360)
3. Saudi Arabia (1.356)

---

### Approach 2: Rating-Based Ensemble (Phase 1a Methodology)
**File:** `line_engine.py`

Treated each line as independent entity (64 total: 32 teams × 2 lines) and applied multiple rating systems:

#### Methods Implemented:
1. **Elo-Basic:** Standard Elo rating for each line based on xG dominance
2. **Elo-XG:** xG-weighted Elo with margin-of-victory adjustment
3. **Iterative Power:** AGD + opponent strength adjustment
4. **B-Score:** Eigenvector centrality on matchup network
5. **XG/60:** Raw expected goals per 60 minutes
6. **XG-Diff/60:** Net xG differential per 60 minutes

#### Ensemble Strategy:
- Calculate disparity ratio for each method separately
- Rank teams by each method
- Average rankings across methods
- Validate internal consistency

---

## Method Validation

### Reliability Analysis
**File:** `refined_ensemble.py`

**Findings:**
- **Elo-Basic vs Elo-XG:** r = 0.962 (highly consistent)
- **Elo methods vs XG/60:** r = 0.51-0.59 (moderate agreement)
- **Simple vs Refined Ensemble:** r = 0.654 (moderate agreement)

**Issues Identified:**
- B-Score: Degenerate results (all teams ≈ 0 difference) - network structure unsuitable for disparity
- IP: Unstable ratios - needs refinement for line-level analysis

**Reliable Methods:**
- ✓ Elo-Basic
- ✓ Elo-XG  
- ✓ XG/60

---

## Final Recommendation

### Selected Method: **Elo-XG**

**Rationale:**
1. **High internal consistency:** r = 0.962 with Elo-Basic (independent validation)
2. **Accounts for matchup dynamics:** Not just raw performance
3. **xG-weighted:** Incorporates margin of victory (larger disparities → stronger signal)
4. **Temporal dynamics:** Ratings evolve game-by-game
5. **Robust to outliers:** Elo framework dampens extreme results

**Comparison to Simple Method:**
- Moderate correlation (r = 0.654) indicates capturing different aspects
- Simple method: "What is the performance gap?"
- Elo-XG: "How consistently does first line outperform given matchup context?"

---

## Final Top 10 (Elo-XG Method)

| Rank | Team         | Interpretation |
|------|--------------|----------------|
| 1    | Panama       | Highest line disparity when accounting for matchup strength |
| 2    | Guatemala    | Strong agreement with simple method (#1 there) |
| 3    | Netherlands  | Consistent first-line dominance |
| 4    | UK           | Significant drop-off to second line |
| 5    | Peru         | Top-heavy offense |
| 6    | USA          | High disparity (#2 in simple method) |
| 7    | New Zealand  | Reliant on first line |
| 8    | Iceland      | Clear quality gap |
| 9    | South Korea  | First line significantly stronger |
| 10   | Indonesia    | Volatile but high disparity |

---

## Key Insights

### 1. **Most Volatile Teams** (rank varies greatly across methods):
- Indonesia, Saudi Arabia, India, UAE, Serbia
- **Implication:** Disparity depends heavily on matchup context for these teams

### 2. **Most Stable Teams** (consistent across methods):
- Vietnam, Ethiopia, Guatemala, New Zealand
- **Implication:** Clear, unambiguous disparity regardless of analysis approach

### 3. **Disagreement Zones:**
Teams that rank very differently in Simple vs Elo-XG:
- **Panama:** #10 simple → #1 Elo-XG (first line dominates against tough opponents)
- **UAE:** #4 simple → #15 Elo-XG (performance inflated by weak matchups)
- **Singapore:** #7 simple → #18 Elo-XG (similar issue)

---

## Files Generated

### Primary Outputs:
- `results/submission_final.csv` - **RECOMMENDED SUBMISSION** (Elo-XG top 10)
- `results/line_disparity_rankings.csv` - Simple method top 10
- `results/FINAL_SUBMISSION.csv` - Original ensemble (all methods)

### Analysis Files:
- `results/all_method_disparities.csv` - All 6 methods, all 32 teams
- `results/ensemble_rankings.csv` - Full 6-method ensemble
- `results/refined_ensemble_full.csv` - Reliable methods only (Elo-Basic, Elo-XG, XG/60)

### Scripts:
- `line_engine.py` - Comprehensive multi-method analysis
- `line_disparity_analysis.py` - Simple baseline method
- `refined_ensemble.py` - Validation and method selection
- `compare_methods.py` - Method comparison report

---

## Conclusion

**Recommended Submission:** Elo-XG ranking (`submission_final.csv`)

**Strengths:**
- Mirrors Phase 1a's rigorous methodology
- Accounts for opponent quality and temporal dynamics
- Internally validated (high consistency with Elo-Basic)
- Moderately correlated with simple baseline (r=0.654)
- Captures matchup-adjusted disparity, not just raw performance

**Top 3 Teams with Largest Disparity:** Panama, Guatemala, Netherlands

---

## Running the Analysis

```bash
# Full multi-method analysis
python3 phase_1b/line_engine.py

# Simple baseline
python3 phase_1b/line_disparity_analysis.py

# Method comparison and final selection
python3 phase_1b/refined_ensemble.py

# Detailed comparison report
python3 phase_1b/compare_methods.py
```
