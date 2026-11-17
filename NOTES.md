# Open design choices (P2)
- **Sparsity budget.** "Sparse" = <= K nonzero terms / leaves (K = 10 default). The result depends on K; report curves, not one K.
- **Tuning parity.** XGBoost is untuned in the baseline. A fair test needs matched search budgets on both sides.
- **Binarization.** Quartile thresholds for all sparse methods; GOSDT/FasterRisk accuracy is sensitive to this. Ablate.
- **Label choice.** `load()` maps the second most frequent class to 1; fine for accuracy, check for AUC.
- **Rashomon proxy** is crude (share of 200 random shallow trees within 1 pt of the best). Semenova et al. estimate the ratio differently; decide which to use.
- **Optional solvers.** Originally written from memory; now checked against the READMEs and source (see "Resolved unknowns" below). The FasterRisk call in `src/` matches the current API. The GOSDT call is valid but skips the recommended guessing steps.
- **Benchmark suite.** Resolved: Grinsztajn et al. classification suites are OpenML 337 and 334 (below).
- **OpenML IDs.** Resolved: all 9 IDs in `DEFAULT` match the names the code claims (below).

## Resolved unknowns (checked 2026-11-17 from primary sources)

### GOSDT — VERIFIED (README of https://github.com/ubc-systopia/gosdt-guesses, PyPI `gosdt` 1.0.4, Python >= 3.9)
- Install: `pip install gosdt`. One package covers GOSDT (Lin et al. 2020) and the guessing extension (McTavish et al. 2022).
- Import: `from gosdt import GOSDTClassifier, ThresholdGuessBinarizer`.
- Constructor (all keyword, with defaults): `GOSDTClassifier(regularization=0.05, allow_small_reg=False, depth_budget=None, time_limit=None, balance=False, cancellation=True, look_ahead=True, similar_support=True, rule_list=False, non_binary=False, diagnostics=False, model_limit=1, debug=False)`; the README example also passes `verbose=True`.
  - `regularization` is the per-leaf penalty. It is clipped below at 1/n unless `allow_small_reg=True`.
  - `depth_budget` counts a root-only tree as depth 0.
  - `time_limit` is in seconds. **If no solution is found by then, it raises an error** (so the try/except in `src/` matters; log these as timeouts).
- `fit(X, y, input_features=None, y_ref=None, cost_matrix=None)`. **X must be boolean (binarized).** `y_ref` = predictions of a black-box reference model (lower-bound guessing).
- Guessing is optional but recommended for speed. README recipe:
  1. Threshold guessing: `enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=...)`; `enc.set_output(transform="pandas")`; `Xtr_b = enc.fit_transform(Xtr, ytr)`; `Xte_b = enc.transform(Xte)`.
  2. Lower-bound guessing: fit `GradientBoostingClassifier(n_estimators=40, max_depth=1)` on `Xtr_b`, take `warm = gbdt.predict(Xtr_b)`.
  3. `GOSDTClassifier(regularization=..., depth_budget=..., time_limit=..., similar_support=False).fit(Xtr_b, ytr, y_ref=warm)`.
- Implication for `src/`: the current call `GOSDTClassifier(regularization=0.01, depth_budget=5, time_limit=60).fit(Btr, ytr)` is valid for 1.x if `Btr` is 0/1, but it uses our quartile binarization and no `y_ref`. Using `ThresholdGuessBinarizer` would give GOSDT different features from the other sparse methods; decide whether to hold binarization fixed (fairer) or use each method's preferred pipeline (closer to "best search"). Not run yet.

### FasterRisk — VERIFIED (README and `src/fasterrisk/fasterrisk.py` of https://github.com/interpretml/FasterRisk, which `jiachangliu/FasterRisk` now redirects to; PyPI `fasterrisk` 0.1.10, Python >= 3.8)
- Install: `pip install fasterrisk`. Import: `from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier`.
- `RiskScoreOptimizer(X, y, k, select_top_m=50, lb=-5, ub=5, gap_tolerance=0.05, parent_size=10, child_size=None, maxAttempts=50, num_ray_search=20, lineSearch_early_stop_tolerance=0.001, group_sparsity=None, featureIndex_to_groupIndex=None)`. `lb`/`ub` bound the integer points.
- Then `opt.optimize()`; `mults, icpts, coefs = opt.get_models()` returns the whole pool (sorted by increasing logistic loss); `opt.get_models(model_index=0)` returns just the best one.
- `RiskScoreClassifier(multiplier, intercept, coefficients, featureNames=None, X_train=None)`; methods `predict(X)`, `predict_prob(X)`, `compute_logisticLoss(X, y)`, `get_acc_and_auc(X, y)`, `print_model_card()`.
- **Labels must be exactly +1/-1** (asserted in the constructor); `predict` returns +1.0/-1.0. X is documented as a 2-D float ndarray (cast with `.astype(float)` to be safe).
- Implication for `src/`: the call there (`y=np.where(ytr == 1, 1, -1)`, `get_models()` then index `[0]`, map predictions `== 1`) matches this API. Not run yet.

### Grinsztajn et al. 2022 benchmark — VERIFIED (README of https://github.com/LeoGrin/tabular-benchmark and the OpenML API)
- OpenML suites: **337** = classification, numerical features (16 tasks); **334** = classification, numerical + categorical (7 tasks); 336 / 335 = the regression versions.
- Load with `openml.study.get_suite(337).tasks` and `openml.tasks.get_task(t).get_dataset()`. The datasets are the paper's processed versions (the README says "with the same transformations that are used in the paper"), not the raw OpenML originals.
- Suite 337 data_ids: 44089 credit, 44120 electricity, 44121 covertype, 44122 pol, 44123 house_16H, 44125 MagicTelescope, 44126 bank-marketing, 44128 MiniBooNE, 44129 Higgs, 44130 eye_movements, 45022 Diabetes130US, 45021 jannis, 45020 default-of-credit-card-clients, 45019 Bioresponse, 45028 california, 45026 heloc.
- Suite 334 data_ids: 44156 electricity, 44157 eye_movements, 44159 covertype, 45035 albert, 45036 default-of-credit-card-clients, 45038 road-safety, 45039 compas-two-years.
- Several datasets appear in both suites (numerical-only vs with categoricals); dedupe before counting "datasets" for correlations. COMPAS (45039) is in suite 334.
- Also on Hugging Face: https://huggingface.co/datasets/inria-soda/tabular-benchmark.

### OpenML data_ids in `src/` — VERIFIED (OpenML API `/api/v1/json/data/<id>`, all status "active")
| id | OpenML name | version | target |
|---|---|---|---|
| 31 | credit-g | 1 | class |
| 37 | diabetes | 1 | class |
| 1464 | blood-transfusion-service-center | 1 | Class |
| 1462 | banknote-authentication | 1 | Class |
| 1489 | phoneme | 1 | Class |
| 1480 | ilpd | 1 | Class |
| 1510 | wdbc | 1 | Class |
| 1067 | kc1 | 1 | defects |
| 1590 | adult | 2 | class |

All nine are what the code claims.

## Verification status
- Every related-work entry was checked to exist (title + authors) via the arXiv API or Crossref on 2026-11-10.
- **Venues and years: VERIFIED 2026-11-17** for all 16 references (Crossref DOIs, PMLR, NeurIPS proceedings pages, Semantic Scholar, arXiv journal_ref). Corrections made in README.md: Shwartz-Ziv & Armon is Information Fusion 81, **2022** (README said 2021, no venue); Lipton's published version is ACM Queue 16(3), 2018 (README had no venue); Rudin 2019 is Nature Machine Intelligence 1(5) (README had no venue); Grinsztajn's proceedings title adds "typical"; Rudin et al. 2024 is titled "Position: ..." and has 10 authors (bib listed 7). All other venues in the README were already correct. references.bib now carries venue/year/volume/DOI.
- Not verified: whether Lipton (2016) also appeared at an ICML 2016 workshop (not claimed anywhere now).
- Code **ran end to end on 2026-11-17** on all 9 datasets (`pip install -r requirements.txt`, Python 3.12) after one fix (XGBoost rejects column names with `[`, `]` or `<`). GOSDT 1.0.4 ran in the same environment (slow, see README). FasterRisk 0.1.10 ran in a separate Python 3.11 environment because it pins scikit-learn 1.2.0.
