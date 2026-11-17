# When Does the Trade-off Close? Optimal Sparse Models on a Tabular Benchmark

> **Starter code for a DSC 291 course project** (Interpretable Machine Learning, UC San Diego, Fall 2026).
> We built this after presenting Lipton, *The Mythos of Model Interpretability* (2016), and Rudin, *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions* (2019), in class on Nov 19, 2026.
> It gives you a research question, a consolidated and checked reading list ([RESEARCH.md](RESEARCH.md)), and a baseline that runs end to end in about 2 minutes, so a project can start from experiments instead of setup. Fork it, change the question, make it yours.

## Quickstart
Python 3.10 to 3.12.
```bash
git clone https://github.com/NishchayMahor/dsc291-tradeoff-gap.git
cd dsc291-tradeoff-gap
pip install -r requirements.txt
python src/run_baseline.py
```
Data is downloaded on the first run (no account needed).

| File | What it is |
|---|---|
| `README.md` | Question, motivation, hypotheses, method, plan (this page) |
| `RESEARCH.md` | Consolidated literature: what is known, the gap, code to reuse, first experiments |
| `NOTES.md` | Open design choices and what has and has not been verified |
| `references.bib` | BibTeX for every paper cited |
| `src/run_baseline.py` | The runnable baseline |

## Research question
Across many tabular datasets, how large is the accuracy gap between **sparse, optimally searched** models (GOSDT trees, FasterRisk scores, sparsity-constrained L1) and gradient boosting, and which dataset properties (size, dimension, noise, estimated Rashomon ratio) predict it?

## Motivation
- **Rudin (2019)** claims there is often no accuracy-interpretability trade-off on structured data with meaningful features, and blames the belief in one on weak search. **Lipton (2016)** treats the trade-off as a real tension worth naming. Neither tests it at scale.
- **Semenova et al. (2022, 2023)** give a mechanism: when the Rashomon set is large (often because the data are noisy), simple models are likely to be in it.
- Our results make the claim dataset-dependent:
  - Gap (best black box minus best off-the-shelf sparse model): **0.4 pts on COMPAS, 2.8 on Adult, 5.7 on German credit.**
  - On COMPAS the accuracy-vs-size curve flattens early: picked on validation data, a 6-term model matches the best black box (68.0% vs 67.7%, inside the margin of error).
  - Tuning closes the German gap to 0.8 pts only with **49 terms**; Adult needs **85 terms** to get within 2.5. So with sparsity enforced, the gap is real on those two.
- Open question: is this a search failure (Rudin's reply) or a property of the data? Optimal methods plus a benchmark answer that.

## Related work (titles/arXiv IDs checked 2026-11-10; venues and years checked 2026-11-17 against Crossref, PMLR, NeurIPS proceedings and Semantic Scholar)
1. Lipton, *The Mythos of Model Interpretability*. arXiv 2016; journal version ACM Queue 16(3), 2018 (reprinted in CACM 61(10), 2018). https://arxiv.org/abs/1606.03490
2. Rudin, *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*, Nature Machine Intelligence 1(5), 2019. https://arxiv.org/abs/1811.10154
3. Semenova, Rudin, Parr, *On the Existence of Simpler ML Models*, FAccT 2022. https://arxiv.org/abs/1908.01755
4. Semenova, Chen, Parr, Rudin, *A Path to Simpler Models Starts With Noise*, NeurIPS 2023. https://arxiv.org/abs/2310.19726
5. Lin et al., *Generalized and Scalable Optimal Sparse Decision Trees* (GOSDT), ICML 2020 (PMLR 119). https://arxiv.org/abs/2006.08690
6. McTavish et al., *Fast Sparse Decision Tree Optimization via Reference Ensembles*, AAAI 2022. https://arxiv.org/abs/2112.00798
7. Liu et al., *FasterRisk: Fast and Accurate Interpretable Risk Scores*, NeurIPS 2022. https://arxiv.org/abs/2210.05846
8. Ustun, Rudin, *Learning Optimized Risk Scores* (RiskSLIM), JMLR 20(150), 2019. https://arxiv.org/abs/1610.00168
9. Grinsztajn, Oyallon, Varoquaux, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 Datasets & Benchmarks track (arXiv title omits "typical"). https://arxiv.org/abs/2207.08815
10. McElfresh et al., *When Do Neural Nets Outperform Boosted Trees on Tabular Data?*, NeurIPS 2023 Datasets & Benchmarks track. https://arxiv.org/abs/2305.02997 (closest design: meta-features predicting which family wins)
11. Shwartz-Ziv, Armon, *Tabular Data: Deep Learning is Not All You Need*, Information Fusion 81, 2022 (arXiv 2021). https://arxiv.org/abs/2106.03253
12. Bischl et al., *OpenML Benchmarking Suites*, NeurIPS 2021 Datasets & Benchmarks track. https://arxiv.org/abs/1708.03731
13. Rudin et al., *Position: Amazing Things Come From Having Many Good Models*, ICML 2024. https://arxiv.org/abs/2407.04846
14. Xin et al., *Exploring the Whole Rashomon Set of Sparse Decision Trees* (TreeFARMS), NeurIPS 2022. https://arxiv.org/abs/2209.08040 (enumerates the Rashomon set of sparse trees; a direct way to estimate the Rashomon ratio for H3)
15. Kruschel et al., *Challenging the Performance-Interpretability Trade-Off: An Evaluation of Interpretable Machine Learning Models*, Business & Information Systems Engineering, 2025. https://arxiv.org/abs/2409.14429 (GAMs vs black boxes on 20 tabular datasets; additive, not sparse, models)
16. Erickson et al., *TabArena: A Living Benchmark for Machine Learning on Tabular Data*, NeurIPS 2025. https://arxiv.org/abs/2506.16791 (curated, maintained dataset collection and tuning protocol)

See RESEARCH.md for what each paper shows.

Gap: refs 9-11 compare boosting with neural nets, not with sparse models; refs 5-8 report sparse-model accuracy on a handful of datasets each. Ref 15 compares interpretable GAMs with black boxes on 20 datasets, but GAMs are additive, not sparse, and it does not model the gap against data properties. Nobody has run the meta-analysis "sparse optimal vs boosting, gap vs data properties" that Rudin's claim needs.

## Hypotheses (falsifiable)
- **H1.** With sparsity fixed (<= 10 terms or leaves), the median gap to tuned XGBoost across >= 25 datasets is under 2 points. *False if* the median is >= 2.
- **H2.** Optimal search (GOSDT / FasterRisk) shrinks the gap vs greedy sparse models (CART / L1) by >= 30% on average. *False if* < 10% (then Rudin's "it's the search" reply is weak).
- **H3.** The gap correlates negatively with the estimated Rashomon ratio (Spearman rho <= -0.4). *False if* |rho| < 0.2.
- **H4.** The gap grows with n and with the share of numeric features (more room for interactions). *False if* no or opposite correlation.

## Method
1. Datasets: start with the 9 OpenML tasks in `src/` (incl. Adult and German), then the Grinsztajn et al. 2022 classification suites on OpenML: suite 337 (numerical features, 16 tasks) and suite 334 (numerical + categorical, 7 tasks), 23 tasks in all (IDs verified 2026-11-17 from their GitHub README and the OpenML API).
2. Same binarization for all sparse methods (quantile thresholds + one-hots); sparsity budget K in {5, 10, 20}.
3. Models: CART depth 3, sparsity-constrained L1, FasterRisk (k = K), GOSDT (depth 5, regularization tuned), tuned XGBoost (random search, 50 trials), tuned logistic.
4. Properties: n, d, binarized d, base rate, share categorical, label-noise estimate (e.g., cross-validated kNN disagreement), Rashomon proxy (share of random sparse models within 1 pt of best; Semenova's ratio estimate).
5. Analysis: gap vs properties via rank correlations and a small regression; accuracy-vs-size Pareto curves per dataset.

## Datasets (open)
OpenML (https://www.openml.org) via `sklearn.datasets.fetch_openml`: credit-g (31), diabetes (37), blood-transfusion (1464), banknote (1462), phoneme (1489), ilpd (1480), wdbc (1510), kc1 (1067), adult (1590). Grinsztajn benchmark code and dataset list: https://github.com/LeoGrin/tabular-benchmark (OpenML suites 337 and 334 for classification; also on Hugging Face: https://huggingface.co/datasets/inria-soda/tabular-benchmark). COMPAS: https://github.com/propublica/compas-analysis.

## Evaluation metrics
5-fold CV accuracy (and AUC, balanced accuracy) with SD; gap in points; model size; Spearman rho of gap with each property; Pareto curve area.

## Baselines
Untuned and tuned XGBoost; full logistic; CART; our talk's numbers (slide 10) (COMPAS 0.4, Adult 2.8, German 5.7 pts; tuned L1 49/85 terms).

## Current baseline (`src/run_baseline.py`)
Loops over 9 OpenML datasets, fits a sparse L1 model (at most K terms), a depth-3 tree and XGBoost, records dataset properties and a Rashomon proxy, prints the gap and exploratory correlations, and writes `p2_results.csv`. Datasets are cached in `~/scikit_learn_data`.
```bash
python src/run_baseline.py --max-terms 10
```
**First run (2026-11-17, 5-fold CV, sparse = at most 10 terms):** gap = XGBoost minus best sparse model, in accuracy points.

| Dataset | n | Gap |
|---|---|---|
| phoneme | 5,404 | 10.6 |
| credit-g | 1,000 | 5.7 |
| adult (20k sample) | 20,000 | 5.2 |
| wdbc | 569 | 4.0 |
| banknote | 1,372 | 3.7 |
| kc1 | 2,109 | 1.5 |
| ilpd | 583 | -0.2 |
| diabetes | 768 | -0.4 |
| blood-transfusion | 748 | -2.3 |

The median gap is 3.7 points with greedy sparse models and an untuned XGBoost. Correlations over 9 datasets are not meaningful yet (none significant); scaling to the Grinsztajn suites is the point of the project.

**Optional solvers (both tested):**
- **GOSDT** (`pip install gosdt`, 1.0.4): works in the main environment but is slow without its threshold-guessing step: each fold hits the 60 s time limit (about 8 minutes for 3 datasets). Adding `ThresholdGuessBinarizer` and a reference model is the first speed-up to try (see NOTES.md).
- **FasterRisk** (`fasterrisk` 0.1.10) pins `scikit-learn==1.2.0`, which conflicts with `requirements.txt`. Run it in its own Python 3.11 environment, e.g. `uv run --python 3.11 --with fasterrisk --with xgboost --with scipy python src/run_baseline.py`.

## Expected results and what would falsify them
We expect a bimodal picture: near-zero gaps on small, noisy, low-dimensional sets (COMPAS-like, consistent with Semenova's noise result) and 2-6 point gaps on larger sets with numeric interactions (Adult-like). H1 failing would mean Rudin's "no trade-off" is a property of a subset of problems, which is how we framed T1 in the talk; H2 failing would mean "better search" is not the main lever.

## Risks and limitations
- Optimal solvers time out on wide or large data; report timeouts, cap with time limits, use McTavish's guessing.
- Binarization choice affects every sparse method; hold it fixed and ablate.
- Tuning budgets must be matched, or the comparison favors whichever side gets more search.
- 25-30 datasets is a small sample for correlation; treat property analysis as exploratory.

## 6-week plan
| Week | Work |
|---|---|
| 1 | Run baseline on 9 datasets; add AUC; install FasterRisk/GOSDT and pin versions |
| 2 | Tuned XGBoost; matched tuning budget; K sweep |
| 3 | Extend to the Grinsztajn suite |
| 4 | Properties: noise estimate, Rashomon ratio estimate |
| 5 | Analysis, Pareto curves, regression |
| 6 | Write-up, figures |

## Path to a workshop paper
A benchmark paper ("Is there a trade-off? 30 datasets, 5 sparse methods") fits a NeurIPS Datasets & Benchmarks-adjacent or tabular-learning workshop (e.g., Table Representation Learning; check the current call). The contribution is the measured gap distribution plus the property analysis, released as a reproducible script.
