# Revision Report: student_dropout_project_submission.ipynb

**Date:** 2026-11-17 · **Original file:** `student_dropout_project_submission.ipynb` · **Revised file:** `student_dropout_project_submission_revised.ipynb`
**Note:** No notebook code was executed during this revision. Stored outputs come from the author's last run of the original code and are preserved unchanged (49 original code cells, outputs and execution counts byte-identical). Narrative numbers not visible in stored outputs were verified with small, separate scratch calculations on the public raw dataset and are labelled as such in the notebook.

## Summary

- Total findings: **9 Critical · 13 Major · 8 Minor**
- Most important changes:
  1. **Train/test leakage fixed (Critical).** The original undersampled with `resample()` (default `replace=True`) *before* splitting. Reproducing the sampling and split on the public data put identical copies of **48 of 239 test students (20.1%)** in the training set. The stored test accuracy (79.5%) and Dropout recall (0.75) are therefore optimistic. The split now comes first (stratified), and undersampling runs on the training split only, without replacement.
  2. **Status chart mislabelled (Critical).** Tick labels were overwritten by position, so the Enrolled bar (794) was labelled "Dropout". Real dropout = **1,421 students (32.1%)**, not 794 (~18%). The narrative is corrected.
  3. **Two engineered-feature bugs fixed (Critical).** `Grade_vs_prev_qual` subtracted a 0–200 grade from a 0–20 grade, so it was essentially −previous grade (r = −0.94), and the "transition challenge" insight was an artefact. `Sem2_approved_to_credited_ratio` copied the semester-1 value.
  4. **Three EDA conclusions reversed.** Reading dropout *rates* instead of counts shows strong risk differences that the original denied: tuition arrears 86.6% vs. 24.7%, debtors 62.0% vs. 28.3%, scholarship holders 12.2% vs. 38.7%, male 45.1% vs. female 25.1%, and large differences by application mode and course. First-semester metrics also clearly separate dropouts, contrary to the original text.
  5. **Full CRISP-DM 0–9 restructure in English**, with Business Understanding, B#/D# success criteria and benchmarks, a leakage-safe Phase A–D preparation, baselines, an assessment table, deployment and conclusions. Preprocessing and model are now one scikit-learn `Pipeline`, so the Streamlit app can reproduce the transformation.

## ⚠️ Action Required

| # | Location | Action |
|---|---|---|
| 1 ✅ | Whole notebook | **Kernel → Restart & Run All** (done 2026-11-18, `USE_PCA = False`; see the run addendum below). Sampling and split order, two features, the log-transform rule, the statistical tests and the model pipeline changed, so every result from cell [27] onward will change. The original also had a non-sequential run (cell 109 had `execution_count=None`). |
| 2 ✅ | Cells [6], [51], [101], [117], [119], [121], [123], [131], [141] | New cells, no stored output: configuration, dropout-rate table, training-split chi-square, CV folds, training utilities, baselines, Optuna tuning utilities, benchmarks, permutation importance. |
| 3 ✅ | Section 7.5 | After re-running, fill in the Evidence/Status column of the B#/D# assessment table and apply the decision table in 7.7. |
| 4 ✅ | Section 5.11 | Re-check `PCA_N_COMPONENTS_1/2` on the new explained-variance plots (done: 3 and 2 components; PCA disabled in this run). |
| 5 | Streamlit app (outside the notebook) | Build one multipage app: a dashboard page on `student.csv` and a prediction page on the new artifacts (`model/random_forest_pipeline.joblib`, `model/encoder_target.joblib`, `model/model_metadata.joblib`). The old `random_forest.joblib` expected PCA inputs that an app could not reproduce. |
| 6 | Section 2.3 | `[TODO]`: confirm real stakeholders, whether MIT data with this schema exists, a dropout-reason field (voluntary vs. expelled), and the fairness policy for gender and marital status. |
| 7 | Section 2.4 | `[TODO]`: confirm the proposed targets (Dropout recall ≥ 0.75, macro F1 ≥ 0.70, CV–test gap ≤ 5 points) against advisor capacity. |
| 8 | Section 3.2 `DROP_COLS`, Section 5.6 | `[TODO]`: document why `Application_mode` (strongly associated with dropout) and the parental variables are excluded. |
| 9 | Section 9.6 / Insight 5 | Confirm with the registrar when `Tuition_fees_up_to_date` and `Debtor` are recorded; if after disengagement, they leak the outcome. |
| 10 | Sections 8.1–8.2 | `[TODO]`: point the Streamlit app to the new artifact names, and add a code-to-label mapping file so the dashboard shows names (e.g. "Nursing") instead of codes (9500). |

## Findings by Stage

### 1. Code Quality

| Severity | Location (original cell) | Issue | Fix |
|---|---|---|---|
| Critical | 102, 109 | `resample()` with replacement before `train_test_split`: duplicated students in train and test; test set artificially balanced | Stratified split first; `undersample()` on the training split only, without replacement; `UNDERSAMPLE_TRAIN` toggle |
| Critical | 32 | `set_xticklabels` by position mislabelled the bars | `plot_status_distribution` re-indexes counts to `STATUS_ORDER` |
| Critical | 68 | `Grade_vs_prev_qual` mixed 0–200 and 0–20 scales | Divide previous grade by `PREV_QUAL_GRADE_SCALE = 10` |
| Critical | 70 | `Sem2_approved_to_credited_ratio` assigned from the Sem1 column | Loop over `SEMESTER_PREFIXES` |
| Critical | 87 | log1p applied to left-skewed features (increases skew) and to a feature with negatives (−1 → −∞ → 0) | Transform only right-skewed, non-negative features |
| Critical | 92–93 | "Reject H0" based on non-zero coefficients and training accuracy; regularized model | Unpenalized multinomial model (`C=np.inf`); global and per-feature likelihood-ratio tests with p-values |
| Critical | 95, 106 | Chi-square feature selection on the full dataset (label leakage into the test design) | Descriptive test on full data (5.5.2); selection repeated on the training split (5.8) |
| Critical | 112 | `LabelEncoder` on features crashes on categories unseen in training | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` |
| Critical | 141 | Scalers and PCA fitted but never saved, so the app could not reproduce the model inputs | Single `Pipeline` (preprocessing + forest) saved with target encoder and metadata |
| Major | 112, 117–124, 131 | Scaler/PCA fitted once on all training rows, then used inside CV (fold leakage); manual, duplicated PCA code | `ColumnTransformer` inside the pipeline, refitted per fold; original cell 122 merged and removed |
| Major | 92 | `LogisticRegression(multi_class=...)` deprecated in scikit-learn 1.5, removed in 1.8; `penalty` itself deprecated in 1.8 | Removed `multi_class`; `C=np.inf` instead of `penalty=None` |
| Major | 112 | `select_dtypes(include=["int", "float"])` misses int64 on Windows | `include="number"` |
| Major | 131 | Accuracy as tuning metric, misaligned with dropout-focused goals | `SEARCH_SCORING = "f1_macro"` (criterion D2); search over the full pipeline with `StratifiedKFold` |
| Major | 133 | Best model refitted a second time (redundant; `refit=True` already did it) | Uses `best_estimator_` directly |
| Major | 137, 148 | No naive or rule benchmark; only weighted metrics | Majority-class and business-rule benchmarks; Dropout recall/precision and macro F1 |
| Major | 144 | Feature importance on PCA components, not actionable | Kept, plus permutation importance on raw features (test split) |
| Major | 152 | Hard-coded `target_names` | Labels from `target_encoder.classes_` |
| Major | 5 | Global `warnings.filterwarnings("ignore")` hid the deprecation warnings above | Removed; minimum-version check added |
| Major | 8, 80, 141 | `try/except` printing the error and continuing (later `NameError`); `model/` directory assumed to exist | Let load errors raise; `Path.mkdir(parents=True, exist_ok=True)` |
| Major | 29–53 | Same config-building loop copy-pasted in 8 cells; unused `plot_countplot` | `plot_boxplots_by_target`, `plot_stacked_bars_by_target`, `dropout_rate_table` |
| Major | 73, 83, 95, 112 | `cat_cols` redefined four times with different meanings | Distinct names (`categorical_cols`, `categorical_like_cols`, `categorical_features`, …) |
| Major | 25 | `select_dtypes(include=object)` misses text columns in pandas ≥ 3.0 | `include=["object", "string"]` |
| Minor | 5 | Unused imports (`os`, `math`, `matplotlib`, `researchpy`, `stats`, `uniform`, `reciprocal`), unsorted | One sorted import cell |
| Minor | all | Magic numbers and paths scattered (0.66, 0.6, 794, 126, 42, `"model/..."`) | Single configuration cell (3.2) |
| Minor | 60 | `replace(0, nan).fillna(0)` no-op | Removed |
| Minor | 17, 109, 119–124 | `inplace=True` (PD002) | Reassignment |
| Minor | 58–70 | Repeated ratio-with-zero-guard code | `safe_ratio()` helper |
| Minor | 73 | `linewidths=-.05` (negative width) | Valid value |
| Minor | 137 | Comment said "with XGBoost" | Corrected |
| Minor | 109 | `execution_count=None` among executed cells | Flagged; Restart & Run All |

### 2. Documentation

- All 15 functions have Google-style docstrings and type hints (helpers, `safe_ratio`, `undersample`, `build_preprocessor`, `build_model_pipeline`, `chi_square_tests`, `multinomial_log_likelihood`, `classification_metrics`, …).
- Comments rewritten to explain *why* (e.g. why split first, why OrdinalEncoder, why `C=np.inf`).
- All markdown translated to professional English, first-person voice removed. Technical terms explained at first use (box plot, skewness, log1p, stratification, undersampling, PCA, cross-validation, chi-square, Cramér's V, likelihood-ratio test, recall/precision/F1, permutation importance).

### 3. Insights & Visualizations

All 16 insights now follow **Finding · Fact check · Caveats · Follow-up check · Implication**. Main corrections (before → after):

| Insight | Before | After |
|---|---|---|
| 1 Target | 794 dropouts (~18%, "almost a quarter", "under a fifth") | 1,421 dropouts (32.1%); chart mislabelled |
| 2 Academic | Semester-1 metrics show no clear difference | Both semesters separate dropouts (median approved units 2 vs. 6 in S1; 0 vs. 6 in S2); low grades may be a symptom of leaving |
| 3 Demographics | Dropouts "typically single and Portuguese"; no gender effect | Base-rate artefact; rates higher for male (45.1%), married (47.2%) and evening (42.9%) students |
| 4 Age | Dropouts mostly > 25 | Median 23 (IQR 19–30) vs. 19 for Graduates |
| 5 Financial | Counts follow group size; no higher risk | Arrears 86.6%, debtor 62.0%, scholarship 12.2% vs. 38.7%; timing/leakage caveat |
| 6 Application/course | Bigger category → more dropouts | Mode 39 (Over 23) 55.4%, mode 7 61.2% vs. 20.2%; courses 15.4%–55.3% |
| 7 Macro | Spread of unemployment differs significantly | About 10 distinct values = cohort proxies |
| 8 Displaced/special needs | Count-based reading | Displaced 27.6% vs. 37.6%; special needs n = 51, no difference |
| 11–12, 15 | Flat boxes = "no correlation" | Flat boxes = mostly zeros; share-of-non-zero table added |
| 14, 16 | Strong negative r = transition challenge | Scale-mismatch artefact |
| 16 | Multicollinearity = "values vary from negative to positive" | Multicollinearity = predictors strongly correlated with each other |
| LR test | Non-zero coefficients + 72% accuracy → reject H0 | Likelihood-ratio test required; coefficients give direction only |
| Model | 0.73 CV score in text | Stored output says 0.75; test > CV is consistent with leakage |

### 4. CRISP-DM Narrative

- **Before:** Persiapan → Data Understanding → Data Preparation (incl. hypothesis testing, undersampling before split, PCA) → Modelling → Evaluation (model export in the middle). No Business Understanding, success criteria, deployment section or conclusion.
- **After:** 0 Intro & roadmap (with 5 declared deviations) → 1 Project Overview + 1.1 References → 2 Business Understanding (2.1–2.5, B1–B3/D1–D4) → 3 Setup → 4 Data Understanding (4.1–4.4, EDA questions Q1–Q8, EDA summary, EDA→Preparation table) → 5 Data Preparation (Phases A–D) → 6 Modelling (baseline, tuning, selection) → 7 Evaluation (benchmarks, per-class report, diagnostics, assessment table, process review, decision table) → 8 Deployment → 9 Conclusion & Recommendations.
- Common thread checked: every B#/D# defined in 2.4, referenced in 2.2, assessed in 7.5 and summarized in 9.2–9.3; all five stakeholders in 2.3 own a recommendation in 9.5; every preparation decision traces to an insight (4.4.12); the test split is used only once (Section 7); all `Section X.Y` references resolve.
- Criteria with status **Pending re-run:** B1, B2, D1, D2, D3, D4.

### 5. Sources

The original notebook cited no sources. Added references:

| Source | Status | Verified via | Action |
|---|---|---|---|
| OECD (2022), *Education at a Glance 2022*, completion-rate indicator | Official report | Web (oecd.org chapter text: 39%, 68%, 12%; Portugal figures) | Added (context, Section 1) |
| Realinho et al. (2022), *Data* 7(11) 146 | Peer-reviewed | Crossref; MDPI page (coding, period, label timing) | Added |
| Realinho et al. (2021), UCI dataset, DOI 10.24432/C5MC89 | Official dataset record | UCI page (license, codes) | Added |
| Martins et al. (2021), Springer AISC, pp. 166–175 | Peer-reviewed proceedings | Crossref | Added |
| Breiman (2001); He & Garcia (2009); Kaufman et al. (2012); Tinto (1975) | Peer-reviewed | Crossref | Added |
| Pedregosa et al. (2011), *JMLR* 12 | Peer-reviewed | jmlr.org | Added |
| scikit-learn `LogisticRegression` docs (1.7) | Official documentation | Web | Used for deprecation facts |

*Validation fallback:* Scite `verify_papers` was not available (monthly account limit reached; resets 2026-11-23), so DOIs were verified through Crossref and the publishers' pages. No retraction notices appeared in the Crossref records. The OECD *Education at a Glance 2025* chapter pages could not be read (script-rendered), so the 2022 edition, whose figures could be verified, is cited.

## Linter Results

- Before: **369** ruff violations (`E,W,F,I,B,UP,N,PD,NPY`) → After: **1**.
- Intentional remaining: `PD101` in the zero-variance check. `nunique(dropna=False) <= 1` is kept for clarity on a 4,424-row table.
- `ruff format` applied to all code cells. `nb_tool.py validate`: OK (147 cells after the Optuna addendum, nbformat 4.5).

## Further Recommendations (out of scope)

- Move the feature engineering and log transform into the pipeline (`FunctionTransformer`) so the app needs no duplicated logic.
- Compare `USE_PCA = False` and `class_weight="balanced"` (no undersampling) after the re-run; both improve interpretability or data use.
- Evaluate a semester-1-only model to quantify how early a useful warning is possible.
- Report per-group recall (gender, age band, course) before any pilot.
- Add `requirements.txt` with pinned versions (Python 3.11; pandas ≥ 2.1; scikit-learn ≥ 1.4; seaborn ≥ 0.13; optuna ≥ 4.0, latest 5.1.0).
- Move plotting helpers to `src/` if they are reused in the app or other notebooks.

## Addendum (2026-11-17): Hyperparameter Tuning Migrated to Optuna

**Request:** replace `RandomizedSearchCV` with Optuna and save the tuning results to a JSON flat file. No code was executed; the stored output of the tuning cell still comes from the original `RandomizedSearchCV` run.

| Location | Change |
|---|---|
| 3.1 Imports (cell [4]) | Removed `RandomizedSearchCV` and `scipy.stats.randint`; added `optuna`, `json`, `datetime.UTC`, `sklearn.base.clone`, `sklearn.metrics.get_scorer`; added `optuna>=4.0` and Python ≥ 3.11 to the version checks |
| 3.2 Configuration (cell [6]) | `RF_PARAM_DISTRIBUTIONS` → `RF_SEARCH_SPACE` (same ranges; `randint` upper bounds were exclusive, Optuna bounds are inclusive, e.g. 100–999); new `RUN_TUNING`, `N_TRIALS = 50`, `OPTUNA_STUDY_NAME`, pruner settings, `TUNING_RESULTS_PATH = model/optuna_tuning_results.json` |
| 6.4 (new cell [123]) | Tuning utilities: `suggest_params`, `make_cv_objective` (fold-by-fold stratified CV of the full pipeline with pruning), `trial_to_record`, `save_tuning_results`, `load_tuning_results` |
| 6.4 (cell [125]) | Optuna study (TPE sampler seeded with `RANDOM_STATE`, `MedianPruner`), JSON export, then the best params are always read back from the JSON file (one code path for `RUN_TUNING = True/False`); top-5 trials displayed |
| 6.5 (cell [127]) | Explicit refit of the best pipeline on the training split (Optuna has no `refit=True`) |
| 8.1 (cell [144]) | Model metadata records `tuning_results_path` |
| Markdown | Sections 1.1 (Akiba et al., 2019 added; verified via Crossref), 2.5, 6.4, 6.5, 8.1, 8.2 updated |

**JSON file structure (`model/optuna_tuning_results.json`).** Study-level fields (`study_name`, `direction`, `scoring`, `cv_folds`, `n_trials`, `sampler`, `pruner`, `seed`, `optuna_version`, `saved_at_utc`, `best_value`, `best_params`), followed by `trials`. That is a list of flat, single-level records with `number`, `state`, `value`, `folds_evaluated`, `param_<name>`, `attr_cv_score_std`, `datetime_start` and `duration_seconds`. `pd.DataFrame(payload["trials"])` loads it as a table.

**Expected differences after re-running.** TPE and pruning explore the space differently from random search, so the selected hyperparameters and CV score will differ from the stored run. Optuna 5.0 changed TPE defaults (multivariate TPE on), so record `optuna_version` (already in the JSON) when comparing studies. The API calls were checked against Optuna 5.1.0 on a toy dataset. All arguments are passed by keyword because Optuna 5.0 removed the positional-argument compatibility.

| ⚠️ Action | |
|---|---|
| `pip install "optuna>=4.0"` before running | Required |
| Re-run Section 6.4 onward | Tuning output, best params, the refit model and all of Section 7 change |
| Commit `model/optuna_tuning_results.json` with the model | Reproducibility |

## Addendum (2026-11-17): Dashboard Moved from Tableau Public to Streamlit

**Request:** build the dashboard in Streamlit, together with the machine-learning prototype, instead of Tableau Public. Only narrative and comments changed; no code logic changed, so no additional re-run is needed beyond the existing actions.

| Location | Change |
|---|---|
| Intro, 2.2 Goal 3, 2.3 (stakeholder outputs, resources), 2.4 B3, 2.5 | Tableau Public replaced by one Streamlit application with a dashboard page and a prediction page |
| 3.2 Configuration (`DASHBOARD_CSV_PATH`), 5.2 | `student.csv` is now the data source of the Streamlit dashboard page (file name unchanged) |
| 7.7, 9.2 | Pilot and operational-tool wording refer to the Streamlit pages |
| 8.2 (cell [145]) | New *Streamlit application* table: Dashboard page (KPI cards, dropout rate and lift by category, academic-progress distributions, filters) and Prediction page (single-student form, batch upload, ranked Dropout probabilities); caching with `st.cache_data` / `st.cache_resource`; `[TODO]` code-to-label mapping file |

Your local edit `USE_PCA = False` (cell [6]) was kept. The narrative in Sections 5.11 and 7.4 still describes the stored PCA run and says PCA can be switched off. No "Tableau" references remain in the notebook. The B3 criterion now reads "Streamlit dashboard dataset is available in original units".

## Addendum (2026-11-18): Revision of the Executed Notebook (`..._revised_ran.ipynb`, `USE_PCA = False`)

**Scope:** narrative only. All 59 code cells and their outputs are unchanged (verified), so no re-run is needed. Your edits were kept: the author block removed from the title, and the `%pip install optuna==5.0` cell. The run itself was clean: sequential execution counts 1–59, no errors and no stderr output.

**What changed in the markdown**

- Every *Pending re-run* note and every reference to the pre-revision outputs was replaced with the actual results of this run. "Correction" bullets that referred to the original notebook were removed (the history stays in this report). The leakage story is kept once, as data storytelling in Sections 5 and 7.2: 79.5% inflated accuracy before, 74.5% honest accuracy now.
- PCA narrative aligned with `USE_PCA = False`. Section 5.11 is now a *redundancy diagnostic*: in group 1, PC1 explains about 60% and 3 components exceed 80%; in group 2, PC1 explains about 76% and 2 components about 85%.
- Six result cells were added after code outputs (split, training-split chi-square, undersampling, baseline, rate table, permutation importance), plus **Insight 17** (permutation importance).
- Sections 6.5, 7.2–7.7 and 9 were rewritten from the outputs; Insights 11, 14, 15 and 16 were updated to the new charts and correlations.

**Key results of the run**

| Metric (test, n = 443) | Random Forest | Business rule | Majority class |
|---|---|---|---|
| Accuracy | 74.49% | 69.30% | 49.89% |
| Macro F1 | 69.21% | 50.35% | 22.19% |
| Dropout recall | 70.42% | 66.20% | 0.00% |
| Dropout precision | 87.72% | 83.19% | 0.00% |

- Optuna: 50 trials (36 complete, 14 pruned). Best trial 47 reached a CV macro F1 of 0.717 (default RF 0.712, so tuning added only +0.005). Train accuracy 95.89%, so there is a 21-point overfitting gap; the CV–test gap is only 2.5 points.
- Per class: Enrolled F1 0.45 (weak); Dropout F1 0.78; Graduate F1 0.84. Of 142 dropouts, 42 were missed (32 predicted Enrolled).
- Permutation importance (test): `Tuition_fees_up_to_date` (≈0.040 drop in macro F1) > `Overall_approval_rate` (≈0.029) > `Age_at_enrollment` (≈0.013); the rest is near zero.
- LR test: LR = 2,917.6 (df 12), McFadden R² 0.323. The credited ratio (p = 0.19) and change in units without evaluation (p = 0.76) are not significant.

**Success criteria:** B1 ✅ (app pending) · B2 ✅ · B3 ✅ · D1 ❌ (0.704 < 0.75, but > rule) · D2 ❌ (0.692 < 0.70, marginal) · D3 ✅ · D4 ✅ → **Decision: iterate** (Section 7.7).

**Recommended next iteration (in priority order)**

1. Tune the Dropout decision threshold on out-of-fold CV probabilities (only 14 false alarms, so there is room to trade precision for recall) to reach D1.
2. Compare `class_weight="balanced"` on the full training split (undersampling discards 1,839 students).
3. Widen the Optuna search space toward regularization (e.g. `min_samples_leaf` 3–20, `max_depth` 5–30).
4. Evaluate a binary "Dropout vs. not" model; drop `Change_units_without_evaluations` and the credited ratios.
5. Confirm when tuition status is recorded: it is the top feature, so late recording would mean leakage.

**Code observations (not changed, so the outputs stay valid)**

| Item | Recommendation |
|---|---|
| `PCA_N_COMPONENTS_1 = 4` in the configuration | Set to 3 if PCA is re-enabled (the new plot shows 3 components reach 80%) |
| `%pip install optuna==5.0` cell | Move to `requirements.txt` (latest is 5.1.0); `%pip` in a notebook is environment-specific |
| `student.csv` written to the working directory, then placed in `data/raw/` | It is an engineered dataset, so `DASHBOARD_CSV_PATH = Path("data/processed/student.csv")` would be clearer; the Streamlit app should read from the same path |
| Global LR p-value printed as `0` | Floating-point underflow; the narrative states "below floating-point precision" |
| Dummy baseline Dropout recall 0.600 | Tie-breaking artefact of `most_frequent` on balanced folds; explained in Section 6.3 |
| Kernel metadata says Python 3.11.11; packages were loaded from a Python 3.13 environment | Harmless (the version check passed); re-save the notebook from the intended kernel |

## Addendum (2026-11-18): Iteration 2 Added (`..._revised_iter2.ipynb`)

**Request:** implement the next modelling iteration. A new file was created; the executed `..._revised_ran.ipynb` is left unchanged as the iteration-1 record. No code was executed: the new cells have no output, and all existing outputs are preserved.

**New Section 7.8 (cells [150]–[162])**

| Sub-section | Content |
|---|---|
| 7.8.1 Setup and helpers | Full (non-undersampled) training split; Candidate B feature set; `out_of_fold_proba`, `predict_with_dropout_threshold`, `scan_thresholds`, `choose_threshold` |
| 7.8.2 Candidate A | Iteration-1 hyperparameters, undersampling repeated inside each training fold; out-of-fold (OOF) probabilities on all 3,981 training students |
| 7.8.3 Candidate B | New Optuna study (`RF_SEARCH_SPACE_V2`: `min_samples_leaf` 3–20, `max_depth` 5–30, `max_features` sqrt/log2/0.5, `class_weight` balanced/balanced_subsample/None; four low-value features dropped); log in `model/optuna_tuning_results_v2.json` |
| 7.8.4 Threshold tuning | Dropout thresholds 0.20–0.60 scanned on OOF predictions; picks the best macro F1 among thresholds with recall ≥ 0.75; candidate selection prefers meeting D1, then macro F1 |
| 7.8.5 Final test | Refit the selected candidate, evaluate once on the test split with its threshold vs. iteration 1 and both benchmarks; classification report, confusion matrix, permutation importance scored with the threshold rule |
| 7.8.6 Assessment | D1–D3 assessed automatically in a table; decision paragraph left as `[Pending run]` |

**Changes to existing cells.** Seven existing cells changed without changing their behavior, so their stored outputs remain valid:

- Configuration: new targets and iteration-2 constants.
- 5.9: keeps `X_train_full` / `y_train_full` before undersampling.
- 5.10: `build_preprocessor(feature_columns=None)`.
- 6.2: `build_model_pipeline(feature_columns=None, **params)`.
- 6.4: `make_cv_objective(..., search_space=RF_SEARCH_SPACE)` and `save_tuning_results(..., extra=None)`.
- 7.3: `plot_confusion_matrix()` helper.
- 7.4: `plot_permutation_importance()` helper.

One cell changed its behavior: **8.1 export** now saves the *final* model, and the metadata adds `selected_candidate` and a `decision_rule` (`dropout_threshold`).

Markdown updated: intro (deviation 6, status note), 7.7 (link to 7.8), 8.1, 8.2 (the app must apply the threshold; `predict()` alone ignores it), 9 (note that 9.1–9.3 still reflect iteration 1).

**Methodological note.** The test split is used a second time for the final model. Every choice in iteration 2 (candidate, hyperparameters, threshold) is made on cross-validation only, to keep that second look fair. This is stated in the notebook.

| ⚠️ Action | |
|---|---|
| Run the new notebook from the top; set `RUN_TUNING = False` to reuse the iteration-1 JSON (identical model, skips about 10 minutes) | Required |
| Candidate B tunes on 3,981 students (vs. 2,142), so expect a longer Optuna run; `N_TRIALS_V2 = 50` can be lowered | Note |
| Complete the `[Pending run]` paragraph in 7.8.6, then update Sections 0 and 9 from the outputs (a narrative pass can do this) | After run |
| Streamlit prediction page: apply `model_metadata["decision_rule"]` | App |

## Addendum (2026-11-18): Final Revision of the Executed Iteration-2 Notebook (`..._revised_iter2_ran.ipynb`)

**Run check.** Clean sequential run (execution counts 1–65), no errors, no stderr. Iteration 1 reproduced exactly from the JSON log (`RUN_TUNING = False`); every iteration-1 text output matches the earlier run.

**Iteration-2 results**

| Test split (n = 443) | Business rule | Iteration 1 | **Final (B, threshold 0.37)** |
|---|---|---|---|
| Accuracy | 69.30% | 74.49% | **75.62%** |
| Macro F1 | 50.35% | 69.21% | **67.99%** |
| Dropout recall | 66.20% | 70.42% | **76.06%** |
| Dropout precision | 83.19% | 87.72% | **83.72%** |

- **Candidate B** (Optuna v2: 32 complete, 18 pruned) won. Best trial 28: 394 trees, `max_depth = 14`, `max_features = 0.5`, `min_samples_split = 15`, `min_samples_leaf = 5`, gini, `class_weight = "balanced"`; CV macro F1 0.718.
- **Out-of-fold, default rule:** B 0.719 macro F1 / 0.721 recall vs. A 0.705 / 0.684.
- **Out-of-fold, tuned threshold:** B at 0.37 gives recall 0.754 and macro F1 0.706; A at 0.34 gives 0.758 and 0.700.
- **Criteria:** D1 ✅ (0.761 vs. 0.75; rule 0.662) · D2 ❌ (0.680 vs. 0.70; caused by Enrolled recall falling from 0.51 to 0.36) · D3 ✅ (OOF 0.706 vs. test 0.680, 2.6 points).
- **Decision recorded in 7.8.6:** formally "iterate" (D2 unmet). Recommended: Academic Affairs decides on a supervised pilot that uses only the Dropout flag (15 extra flags per 443 students, 8 more dropouts caught).
- **Permutation importance (final model, scored with the threshold rule):** `Overall_approval_rate` ≈ 0.084 > `Tuition_fees_up_to_date` ≈ 0.037 > `Sem2_approval_rate` ≈ 0.010 > `Debtor` ≈ 0.004; age now ≈ 0.

**Removed (no longer used)**

| Item | Reason |
|---|---|
| Section 5.11 "Feature Redundancy and PCA": header, two explained-variance plot cells (with their outputs) and two notes | PCA was disabled and is not part of the final model |
| PCA configuration (`USE_PCA`, `PCA_GROUP_1/2`, `PCA_N_COMPONENTS_1/2`, `PCA_VARIANCE_TARGET`), `from sklearn.decomposition import PCA`, the PCA branch of `build_preprocessor` and the `pca_*` / `scaled_only_features` variables | Same; `build_preprocessor` now scales `numeric_features` directly (same columns as before) |
| "Running this section" note (7.8), `[Pending run]` paragraph (7.8.6), "not yet run" notes (intro, Section 9) | Iteration 2 has been run |
| Code comments and docstrings that referred to the original notebook or `RandomizedSearchCV` | Revision history lives in this report |

`RUN_TUNING_V2` was set to `False`, since the study is complete and is reloaded from `optuna_tuning_results_v2.json`. Section 5.12 was renumbered to 5.11, and all references were updated.

**Narrative updated:** intro key results; Section 5 tables; Insight 16; 7.5–7.7 marked "(Iteration 1)"; new interpretation cells after Candidates A and B and the threshold scan; **Insight 18** (final model); 7.8.6 decision; Section 9 rewritten for the final model (including a new limitation: the test split was used twice).

**Outputs.** All outputs of kept code cells are unchanged. Eight code cells changed only in comments, imports, configuration or the removed PCA branch, so their behavior is identical. The only stale text is cell [110]'s summary line, which still prints "… 0 compressed by PCA"; the new code prints "10 categorical | 26 numeric". An optional final *Restart & Run All* (both tuning flags `False`) refreshes it in a few minutes.

**Still open (`[TODO]`, user input):** stakeholder names, MIT data availability, dropout-reason field, fairness policy, confirmation of the targets (especially D2), the rationale for `DROP_COLS`, and a code-to-label mapping file for the Streamlit app.

## Sources

- [OECD (2022). Education at a Glance 2022, completion indicator chapter](https://www.oecd.org/en/publications/education-at-a-glance-2022_3197152b-en/full-report/component-19.html)
- [Realinho et al. (2022). Predicting Student Dropout and Academic Success, *Data*](https://www.mdpi.com/2306-5729/7/11/146)
- [UCI ML Repository: Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [Crossref: Martins et al. (2021)](https://doi.org/10.1007/978-3-030-72657-7_16) · [Breiman (2001)](https://doi.org/10.1023/A:1010933404324) · [He & Garcia (2009)](https://doi.org/10.1109/TKDE.2008.239) · [Kaufman et al. (2012)](https://doi.org/10.1145/2382577.2382579) · [Tinto (1975)](https://doi.org/10.3102/00346543045001089)
- [Pedregosa et al. (2011), JMLR](https://jmlr.org/papers/v12/pedregosa11a.html)
- [scikit-learn 1.7 LogisticRegression documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Akiba et al. (2019), Optuna, KDD '19](https://doi.org/10.1145/3292500.3330701)
- [Optuna 5.0.0 release notes](https://newreleases.io/project/pypi/optuna/release/5.0.0)
