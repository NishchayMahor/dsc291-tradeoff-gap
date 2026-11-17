# Research brief: when does the accuracy-interpretability trade-off close?

Everything here was checked on 2026-11-17 against primary sources (arXiv, Crossref, PMLR, NeurIPS proceedings, GitHub, PyPI, the OpenML API). Anything not checked is marked. Full citations are in `references.bib`; the exact solver APIs are in `NOTES.md`.

## The problem
Rudin (2019) argues that on tabular data with meaningful features, a well-searched sparse model is usually about as accurate as a black box. When people see a gap, she says the cause is weak search, not a real trade-off. Nobody has tested this claim systematically. We want to measure the gap between optimally searched sparse models (GOSDT trees, FasterRisk risk scores, sparse L1 logistic regression) and tuned gradient boosting across 25+ datasets. Then we ask which dataset properties (size, dimension, label noise, estimated Rashomon ratio) predict when the gap is near zero and when it is not.

## What is known
- **Rudin 2019, Nature Machine Intelligence.** A position paper. It argues that for high-stakes tabular decisions we should build interpretable models, not explain black boxes, and that the trade-off is often a myth. It gives examples, not a large benchmark.
- **Lipton 2016 (ACM Queue 2018).** Argues that "interpretability" covers several different goals, and treats the accuracy trade-off as a real tension. Conceptual, no experiments.
- **Semenova, Rudin, Parr, FAccT 2022.** Defines the Rashomon ratio (share of a model class that is nearly as good as the best model). It shows that when the ratio is large, a simple model with good accuracy is likely to exist. This is the theory behind our H3.
- **Semenova, Chen, Parr, Rudin, NeurIPS 2023.** Shows that label noise tends to enlarge the Rashomon set, so noisy problems should favor simple models. This predicts that our gap shrinks on noisy datasets.
- **Lin et al., ICML 2020 (GOSDT).** Dynamic programming plus bounds that find provably optimal sparse decision trees under a per-leaf penalty.
- **McTavish et al., AAAI 2022.** Makes GOSDT much faster by "guessing" thresholds, depth and lower bounds from a boosted-tree reference model. This is what makes GOSDT usable on medium-sized data.
- **Ustun and Rudin, JMLR 2019 (RiskSLIM)** and **Liu et al., NeurIPS 2022 (FasterRisk).** Risk scores (small integer-point linear models). RiskSLIM solves them with integer programming; FasterRisk finds high-quality ones in minutes with beam search and rounding.
- **Xin et al., NeurIPS 2022 (TreeFARMS).** Enumerates the whole Rashomon set of sparse decision trees. Gives a direct way to estimate the Rashomon ratio, better than our random-tree proxy.
- **Rudin et al., ICML 2024 (position paper).** Argues that large Rashomon sets have practical benefits, including that simpler accurate models then exist.
- **Grinsztajn et al., NeurIPS 2022 D&B.** On a curated benchmark of medium-sized tabular datasets (~10K samples), tree ensembles beat deep nets, partly because nets struggle with uninformative features and irregular functions. Its OpenML suites are our main dataset source.
- **McElfresh et al., NeurIPS 2023 D&B.** Compares 19 algorithms on 176 datasets and uses dataset meta-features to predict which family wins. This is the closest design to ours, but it compares nets with boosted trees, not sparse models with boosted trees.
- **Shwartz-Ziv and Armon, Information Fusion 2022.** Finds XGBoost generally beats the deep tabular models proposed at the time.
- **Bischl et al., NeurIPS 2021 D&B.** OpenML benchmarking suites (e.g. OpenML-CC18) and tools to load them.

### Recent work worth knowing (2023-2026)
- **Kruschel et al., Business & Information Systems Engineering, 2025** (arXiv 2409.14429). Compares 7 GAMs with 7 common ML models on 20 tabular datasets, with matched hyperparameter search (68,500 runs). Concludes there is "no strict trade-off" for tabular data. **Why it matters:** the most direct recent test of Rudin's claim, but for GAMs (additive, many shape functions), not sparse models with a few terms. Our project tests the stricter version, and their tuning protocol is a template for ours.
- **Erickson et al., TabArena, NeurIPS 2025** (arXiv 2506.16791). A maintained tabular benchmark with curated datasets and a public leaderboard. It reports that gradient-boosted trees are still strong but deep learning and foundation models have caught up in some settings. **Why it matters:** a second, curated dataset pool, and its findings on validation and tuning protocols bear on our "matched tuning budget" design.

## The gap
Benchmarks of boosting vs. neural nets (Grinsztajn, McElfresh, Shwartz-Ziv, TabArena) never include sparse optimal models. Sparse-model papers (GOSDT, FasterRisk, RiskSLIM) each report accuracy on a handful of datasets. The one recent large comparison of interpretable models (Kruschel et al.) uses GAMs, not sparse models, and does not relate the gap to data properties. No one has measured "sparse optimal vs. tuned boosting" across many datasets and tested whether noise or the Rashomon ratio predicts the gap. That is the test Rudin's claim needs.

## Open-source implementations (links checked)
| Tool | What it does | Install | Link |
|---|---|---|---|
| GOSDT (+ guessing) | Optimal sparse decision trees | `pip install gosdt` (1.0.4) | https://github.com/ubc-systopia/gosdt-guesses |
| FasterRisk | Sparse integer risk scores | `pip install fasterrisk` (0.1.10) | https://github.com/interpretml/FasterRisk |
| RiskSLIM | Optimal risk scores via MIP (needs a MIP solver; install not checked) | see repo | https://github.com/ustunb/riskslim |
| TreeFARMS | Enumerate the Rashomon set of sparse trees | `pip install treefarms` (0.2.4 on PyPI) | https://github.com/ubc-systopia/treeFarms |
| InterpretML (EBM) | GAM baseline, if you want a "glass box but not sparse" middle point | see repo | https://github.com/interpretml/interpret |
| Grinsztajn benchmark code | Dataset processing and random-search results | — | https://github.com/LeoGrin/tabular-benchmark |
| TabZilla | McElfresh et al. code, meta-features | — | https://github.com/naszilla/tabzilla |
| TabArena | Living benchmark | — | https://github.com/autogluon/tabarena |

Two API facts that will bite you (details in `NOTES.md`): FasterRisk needs labels in {-1, +1}; GOSDT needs boolean features and raises an error if `time_limit` is hit before any solution is found.

## Datasets and benchmarks
- **Starter set (in `src/`)**: 9 OpenML datasets, IDs verified: credit-g (31), diabetes (37), blood-transfusion (1464), banknote (1462), phoneme (1489), ilpd (1480), wdbc (1510), kc1 (1067), adult (1590). https://www.openml.org
- **Grinsztajn suites on OpenML**: 337 (classification, numerical, 16 tasks) and 334 (classification, numerical + categorical, 7 tasks, includes COMPAS). Some datasets appear in both; dedupe. Also on Hugging Face: https://huggingface.co/datasets/inria-soda/tabular-benchmark
- **COMPAS (original)**: https://github.com/propublica/compas-analysis
- **Optional extra pools**: OpenML-CC18 (Bischl et al.), TabZilla hard suite, TabArena datasets. Exact suite IDs for these were not checked.

## Week 1-2 experiments
1. **Get the solvers running and record failures (week 1).** Install `gosdt` and `fasterrisk`, run `src/run_baseline.py --max-terms 10` on the 9 datasets, and log for every (dataset, method, fold): accuracy, AUC, model size, runtime, and whether it timed out. Success = a full table with no silent skips. This also tells you which datasets are too big for GOSDT at a 60 s limit.
2. **Sparsity curves on 3 anchor datasets (week 1-2).** On COMPAS (45039 or ProPublica), Adult (1590) and credit-g (31), sweep K in {3, 5, 7, 10, 15, 20} for FasterRisk and L1, and GOSDT regularization in {0.1, 0.05, 0.02, 0.01, 0.005}. Plot accuracy vs. number of terms/leaves against tuned XGBoost. Check that it reproduces our talk's numbers (gaps of ~0.4 / 2.8 / 5.7 points) and whether optimal search moves the German and Adult curves (first look at H2).
3. **Search vs. binarization ablation (week 2).** On the same 3 datasets, run GOSDT (a) on our quartile binarization and (b) with `ThresholdGuessBinarizer` + `y_ref` guessing, and FasterRisk on both feature sets. If the gap changes more with binarization than with the solver, then "better search" includes feature construction, and the main experiment must fix one protocol. Also start the tuned-XGBoost random search (50 trials) with the same CV folds, so both sides get matched budgets.

## Key risks
- **Solver scale.** GOSDT and FasterRisk can be slow on wide or large data (several Grinsztajn datasets have tens of thousands of rows). Cap time, subsample consistently, and report timeouts as results, not as missing data.
- **Binarization drives results.** Every sparse method sees binarized features; different thresholds can change accuracy by more than the gap you are measuring. Fix one protocol and ablate.
- **Unfair tuning.** If XGBoost gets more tuning than the sparse models (or the reverse), the gap is an artifact. Match budgets and CV folds.
- **Small sample for correlations.** About 25-30 datasets (after deduping the suites) gives wide confidence intervals for Spearman rho. Treat property analysis as exploratory and report intervals.
- **Rashomon ratio is hard to estimate.** Our random-tree proxy is crude; TreeFARMS gives an exact count for sparse trees but may not scale. Pick one estimator early and check it on small data.
- **Processed vs. raw data.** The Grinsztajn suites contain their processed versions of each dataset, so numbers will not match papers that used the raw OpenML originals.
