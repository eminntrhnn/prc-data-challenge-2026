# PRC Data Challenge 2026 — Taxi-Out Time Prediction (Team *bright-llama*)

Gradient-boosted ensemble for predicting the **taxi-out time** (`TAXITIME_SEC_mvt`, seconds from off-block to take-off) of departures at ten major European airports, with causal airport-surface congestion features and a leakage-aware, time-ordered validation scheme.

> **Status of this document.** All items were verified against the final four-month run (see sections 3 and 6).

---

## 1. Project Summary and Challenge Objective

| Item | Description |
|---|---|
| Task | Predict taxi-out time per departure (`TAXITIME_SEC_mvt`, seconds). Lower leaderboard score is better. |
| Airports (`ADEP_mvt`) | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM |
| Training data | 2025 departures, 2,085,047 rows after de-duplication (10 airports, all 12 months) |
| Initial-phase test | January and July 2026, 344,841 departures |
| Final-phase test | January, February, June, July 2026, 670,790 departures (single submission)** |
| Submission format | Parquet with columns `MVT_ID_mvt`, `TAXITIME_SEC_mvt` (float64), one row per template ID |

The final phase evaluates generalisation across months (Jan/Jul, Feb/Jun and all four combined), so the pipeline avoids tuning against the initial-phase leaderboard. Note that `AOBT_3_flt` (actual off-block time) is an *input* used to derive congestion features; the model does not predict EOBT/AOBT.

---

## 2. Data Pipeline Architecture

### 2.1 Pipeline overview

```
final_ranking.parquet (DEP + ARR rows)
        │
        ├─ de-duplicate on MVT_ID_mvt
        ├─ airport-level queue features  (dep_back_30, arr_back_30, dep_fwd_30)   ── all rows (DEP+ARR)
        ├─ keep DEP rows
        ├─ calendar features (hour, dow, month, is_weekend, hour_sin, hour_cos), wake constants
        ├─ runway-level queue features   (rwyq_dep_15, rwyq_dep_30)              ── DEP rows
        └─ align schema to the reference ranking_v8r  ─►  final_v8r.parquet
                                                           │
                       ┌───────────────────────────────────┴──────────────────┐
                       ▼                                                      ▼
        full-2025 models (LightGBM ×2 configs, XGBoost) + target encoding   fold model "last1"
                       └───────────────────────  50/50 blend  ───────────────┘
                                                  │
                                      ID-keyed merge into the template
```

### 2.2 Feature engineering

All model inputs are available at off-block time or earlier.

| Group | Features | Definition |
|---|---|---|
| Calendar | `hour`, `dow`, `month`, `is_weekend`, `hour_sin`, `hour_cos` | Derived from `MVT_TIME_UTC_mvt`; `hour_sin/cos = sin/cos(2π·hour/24)` encode the daily cycle |
| Airport queue | `dep_back_30` | Other departures with actual off-block in (t−30 min, t] at the airport |
| | `arr_back_30` | Arrivals with actual arrival in (t−30 min, t] at the airport |
| | `dep_fwd_30` | Scheduled departures in (t, t+30 min] (excluding the flight itself) |
| Runway queue | `rwyq_dep_15`, `rwyq_dep_30` | Other departures with actual off-block in the last 15/30 min on the same `ADEP|RUNWAY` |
| Wake | `wake_category`, `wake_rank` | **Constant in train and test** (`MISSING` / `-1`): the aircraft-type mapping did not resolve, so these columns carry no information. They are retained only for schema consistency with the trained models; `WK_TBL_CAT_flt` carries wake information. |
| Categorical | `FLIGHT_RULE_mvt`, `ADEP_mvt`, `PHASE_mvt`, `AIRCRAFT_TYPE_mvt`, `RUNWAY_mvt`, `ADEP_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt` | Operator identifiers are hashed in the source data |
| Target encoding | `stand_te` (`te_0` in the full-data scripts) | Smoothed mean of the target on key `ADEP|RUNWAY|STAND` (Section 3) |

Weather features were explored during development but are **not part of the final feature set**; no weather data is needed for inference.

### 2.3 Four-month inference and parity validation

`71_build_final_v8r.py` rebuilds the feature table for the four-month ranking file with the same functions used for training, then compares its January/July rows against the reference table produced for the initial phase.

| Check | Rule |
|---|---|
| Row count / uniqueness | 670,790 DEP rows, unique `MVT_ID_mvt`, ID set equals the template |
| Schema | Every column of the reference `ranking_v8r` must be reproduced |
| Parity (Jan/Jul rows) | Queue and hour features ≥ 99 % match; all other columns ≥ 99.9 % (timestamps within 30 s) |

Observed results:

| Run | Result |
|---|---|
| Builder reproduction on the initial-phase input (344,841 rows) | `dep_back_30`, `arr_back_30`, `dep_fwd_30`, `rwyq_dep_15`, `rwyq_dep_30`: exact match on all rows |
| Raw columns, final vs. initial ranking (Jan/Jul IDs) | Identical except `MVT_TIME_UTC_mvt`, which differs by at most ±12 s |
| Four-month run (Feb/Jun rows) | 670,790 rows (Jan 152,719 / Feb 144,158 / Jun 181,791 / Jul 192,122 DEP); all 26 features reproduced with parity ≥ 0.995 (queue features 1.0000); v9 re-prediction on Jan/Jul: corr 0.99989, mean abs diff 0.20 s |

The parity check validates reproducibility of the feature code; it does not guarantee predictive accuracy on unseen months.

---

## 3. Models and Training Strategy

### 3.1 Validation

Expanding-window monthly validation over 2025: for each month *m* from February to December, models are trained on all earlier months and evaluated on *m* (11 folds, no future data). Training targets are winsorised at the 99.5th percentile; evaluation uses the raw target.

| Configuration | OOF RMSE (s) |
|---|---|
| Baseline LightGBM (no stand TE, no runway queue) | 319.2 |
| + `stand_te` | ≈ 308.1 |
| + `stand_te` + runway queue (`rwyq_*`) | 306.9 |

Baseline per-month RMSE: Feb 391.9, Mar 291.4, Apr 280.9, May 285.6, Jun 300.1, Jul 408.7, Aug 299.2, Sep 292.3, Oct 302.4, Nov 319.4, Dec 322.4. February and July are the hardest months; the largest 1 % of errors account for 42.6 % of the total squared error.

### 3.2 Models

| Model | Main settings |
|---|---|
| LightGBM (base) | L2 objective, lr 0.03, 63 leaves, depth 8, subsample 0.8, colsample 0.7, α=0.1, λ=1.0, min child 30, up to 3000 rounds, early stopping on a time-ordered 10 % hold-out, several seeds |
| LightGBM (wide) | 127 leaves, depth 10, lr 0.02, other settings as above |
| XGBoost | `hist`, native categorical support, depth 8, lr 0.03, several seeds |

Categorical levels are stored at training time (`cats.json`) and re-applied at inference so that category codes match exactly. In-training target encoding uses 5-fold out-of-fold means with smoothing k = 10; unseen keys fall back to the global mean (key coverage on the initial test: ≈ 0.99).

### 3.3 Ensemble

* **v9**: equal-weight mean of the three full-2025 model families above.
* **last1**: the 2025-12 fold model (trained on January–November 2025).
* **Final prediction** = 0.5 · v9 + 0.5 · last1, clipped to [0, 3600]. The blend was chosen on rationale (variance reduction, recency) and **has no out-of-fold estimate**.
* The submitted file uses v9 only (equal mean of LightGBM, wide LightGBM and XGBoost) for February/June, because no `last1` run was available for those months; January/July use the 50/50 v9 + `last1` blend. This inconsistency is accepted.

### 3.4 Initial-phase leaderboard (for transparency)

| Submission | Score (lower is better) |
|---|---|
| Constant (train mean) | 677.8 |
| Airport mean | 661.2 |
| Blend submitted in the initial phase | 612.6 |

The leaderboard score scale was not reconciled with local RMSE, so these numbers are not directly comparable with Section 3.1.

---

## 4. Repository Structure

```
.
├── 12_production_lgb_stand.py   # expanding-window monthly folds, OOF + per-fold test predictions (--stand-te)
├── 27_rebag_test.py             # average/select per-fold test predictions (e.g. --bag-last 1 -> "last1")
├── 40_final_te.py               # full-data training/prediction with target encoding (LightGBM/XGBoost)
├── 41_runway_queue.py           # runway-level queue features (rwyq_dep_15/30)
├── 26_merge_queue_into.py       # merge extra feature columns by MVT_ID_mvt
├── 43_final_ensemble.py         # trains and blends the "v9" model family
├── 30_blend_preds.py            # equal-weight prediction blending
├── 16_make_submission.py        # writes a submission in template format
├── 62_final_blend.py            # 0.5·v9 + 0.5·last1 via ID-keyed merge
├── 71_build_final_v8r.py        # four-month feature builder + parity check
├── 70_final_predict_merge.py    # Feb/Jun inference + ID-keyed merge -> final candidate
├── 63_upload_final.py           # schema check and S3/MinIO upload
├── data/                        # train_v8r.parquet, ranking_v8r.parquet, final_ranking.parquet, final_v8r.parquet
├── run_v9_lgb/ run_v9_lgbw/ run_v9_xgb/   # trained models + cats.json, te_maps.json, meta.json
├── run_stand_rwyq/              # OOF predictions and per-fold test predictions
└── submissions/                 # candidate files
```

Note: the 2025 training-side airport queue features (`dep_back_30`, `arr_back_30`, `dep_fwd_30`) were originally produced in notebook cells; `71_build_final_v8r.py` reimplements the same logic as a script and is verified against the original outputs on the initial-phase test data.

---

## 5. Setup and Execution

### 5.1 Requirements

Python 3.10+ with `pandas`, `numpy`, `pyarrow`, `lightgbm`, `xgboost` (and `boto3`/`minio` for upload). Pin exact versions from the environment used for the run:

```bash
pip freeze | grep -i -E "pandas|numpy|pyarrow|lightgbm|xgboost|boto3|minio" > requirements.txt
```

### 5.2 Inference on the four-month ranking file

```bash
export WORK=/content/drive/MyDrive/prc2026_work && cd $WORK

# 1) Features for Jan/Feb/Jun/Jul + parity table against the initial-phase reference
python 71_build_final_v8r.py --final-ranking data/final_ranking.parquet \
                             --template data/final_submitting.parquet

# 2) Predictions: v9 for Feb/Jun (parity check on Jan/Jul included), merged by ID into the template
python 70_final_predict_merge.py --allow-v9-only --template data/final_submitting.parquet

# 3) "last1" fold model for Feb/Jun (verify flags with --help), then rerun step 2 without --allow-v9-only
python 12_production_lgb_stand.py --train data/train_v8r.parquet --test data/final_v8r_FebJun.parquet \
       --out-dir run_stand_final --single --winsor 99.5 --stand-te --num-threads 4 --start-month 2025-12
python 70_final_predict_merge.py --template data/final_submitting.parquet
```

Output: `submissions/final_candidate.parquet` (670,790 rows, no missing values, values in [0, 3600]).

### 5.3 Final file checks

```python
import pandas as pd
c = pd.read_parquet("submissions/final_candidate.parquet")
t = pd.read_parquet("data/final_submitting.parquet")
assert len(c) == 670_790 and c.MVT_ID_mvt.is_unique and set(c.MVT_ID_mvt) == set(t.MVT_ID_mvt)
assert c.TAXITIME_SEC_mvt.notna().all() and str(c.TAXITIME_SEC_mvt.dtype) == "float64"
```

### 5.4 Reproducing training

```bash
python 12_production_lgb_stand.py --train data/train_v8r.parquet --test data/ranking_v8r.parquet \
       --out-dir run_stand_rwyq --single --winsor 99.5 --stand-te --num-threads 4
python 43_final_ensemble.py --run --ver v9 --train data/train_v8r.parquet --test data/ranking_v8r.parquet \
       --te-keys "ADEP_mvt|RUNWAY_mvt|STAND_mvt" --algos lgb lgbw xgb
```

---

## 6. Assumptions and Limitations

* `RUNWAY_mvt` and `STAND_mvt` are assumed known at prediction time; `dep_fwd_30` uses scheduled departure times, assumed available in advance.
* No out-of-fold estimate exists for the final blend; January 2026 has no 2025 analogue before the first fold and is unvalidated.
* `wake_category` / `wake_rank` are constant and therefore inactive.
* Only the `ADEP|RUNWAY|STAND` target-encoding key was validated; wider key sweeps were not completed.
* Feb/Jun features are validated indirectly (Jan/Jul parity), not against ground truth.Jan/Jul parity of the final pipeline: corr 0.99989, mean abs diff 0.20 s.
* The relation between local RMSE and the leaderboard score was not established.
