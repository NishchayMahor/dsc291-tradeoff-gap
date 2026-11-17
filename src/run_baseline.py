"""P2 baseline: sparse vs boosting across OpenML datasets, with dataset properties recorded.

For each dataset: 5-fold CV accuracy of
  - sparse_l1   : L1 logistic on a binarized view, strongest C with <= K nonzero terms (sparsity enforced)
  - tree_d3     : CART depth 3
  - fasterrisk  : FasterRisk k-term risk score (optional, if installed)
  - gosdt       : GOSDT optimal tree (optional, if installed)
  - xgboost     : fixed reasonable config (not tuned; see NOTES.md)
plus properties: n, d, binarized d, base rate, share categorical, and an empirical Rashomon proxy
(share of 200 random sparse models within 1 pt of the best of them).
Prints a table of gap = xgboost - best sparse, and Spearman correlations of the gap with each property.

Usage: python src/run_baseline.py [--max-terms 10] [--datasets 31 37 1464 ...] [--max-rows 20000]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Small, binary, open OpenML tasks (data_id: name). credit-g = German credit, 1590 = Adult (the extra datasets in our talk).
DEFAULT = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion", 1462: "banknote", 1489: "phoneme",
           1480: "ilpd", 1510: "wdbc", 1067: "kc1", 1590: "adult"}

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", type=int, nargs="*", default=list(DEFAULT))
ap.add_argument("--max-terms", type=int, default=10)
ap.add_argument("--max-rows", type=int, default=20000, help="subsample large datasets for speed")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

try:
    from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
    HAVE_FR = True
except ImportError:
    HAVE_FR = False
    print("[optional] fasterrisk not installed: skipping FasterRisk (pip install fasterrisk).")
try:
    from gosdt import GOSDTClassifier
    HAVE_GOSDT = True
except ImportError:
    HAVE_GOSDT = False
    print("[optional] gosdt not installed: skipping GOSDT (pip install gosdt).")


def load(did):
    ds = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df, t = ds.data.copy(), ds.target
    classes = t.value_counts().index
    y = (t == classes[1]).astype(int).values  # minority-ish class = 1; binary tasks only
    if len(df) > args.max_rows:
        keep = np.random.default_rng(0).choice(len(df), args.max_rows, replace=False)
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    cat = df.select_dtypes(exclude="number").columns
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median())
    X = pd.get_dummies(df, columns=list(cat), dummy_na=False).astype(float)
    B = X.drop(columns=num).copy()  # binarized view: one-hots + quartile thresholds
    for c in num:
        for q in (0.25, 0.5, 0.75):
            thr = df[c].quantile(q)
            B[f"{c}_gt_{thr:g}"] = (df[c] > thr).astype(float)
    B = B.loc[:, B.std() > 0]
    B = B.loc[:, ~B.T.duplicated()]
    clean = lambda cols: cols.astype(str).str.replace(r"[\[\]<]", "_", regex=True)  # XGBoost rejects [ ] <
    X.columns, B.columns = clean(X.columns), clean(B.columns)
    props = {"n": len(y), "d": df.shape[1], "d_bin": B.shape[1], "base_rate": float(y.mean()),
             "share_cat": len(cat) / max(df.shape[1], 1)}
    return X, B, y, props


def sparse_l1(Btr, ytr, Bte, K):
    """Largest C whose L1 fit keeps <= K nonzero terms (sparsity is the constraint, not a tuned penalty)."""
    best = None
    for C in np.logspace(-3, 1, 15):
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear").fit(Btr, ytr)
        if (np.abs(m.coef_) > 1e-6).sum() <= K:
            best = m
    best = best or LogisticRegression(penalty="l1", C=1e-3, solver="liblinear").fit(Btr, ytr)
    return best.predict(Bte)


def fasterrisk(Btr, ytr, Bte, K):
    opt = RiskScoreOptimizer(X=Btr.values, y=np.where(ytr == 1, 1, -1), k=K)
    opt.optimize()
    mult, icpt, coefs = opt.get_models()
    clf = RiskScoreClassifier(multiplier=mult[0], intercept=icpt[0], coefficients=coefs[0])
    return (np.asarray(clf.predict(X=Bte.values)) == 1).astype(int)


def gosdt(Btr, ytr, Bte):
    m = GOSDTClassifier(regularization=0.01, depth_budget=5, time_limit=60)  # API differs across gosdt versions
    m.fit(Btr, ytr)
    return np.asarray(m.predict(Bte)).astype(int)


def rashomon_proxy(Btr, ytr, Bva, yva, n=200, eps=0.01):
    """Share of random sparse models within eps of the best one: a crude empirical Rashomon ratio."""
    rng, accs = np.random.default_rng(0), []
    for i in range(n):
        cols = list(rng.choice(Btr.columns, size=int(rng.integers(2, min(len(Btr.columns), 15) + 1)), replace=False))
        m = DecisionTreeClassifier(max_depth=int(rng.integers(1, 4)), min_samples_leaf=20, random_state=i)
        m.fit(Btr[cols], ytr)
        accs.append((m.predict(Bva[cols]) == yva).mean())
    accs = np.array(accs)
    return float((accs >= accs.max() - eps).mean())


rows = []
for did in args.datasets:
    name = DEFAULT.get(did, str(did))
    try:
        X, B, y, props = load(did)
    except Exception as e:
        print(f"[skip] {name} ({did}): {e!r}")
        continue
    acc = {k: [] for k in ["sparse_l1", "tree_d3", "xgboost"] + (["fasterrisk"] if HAVE_FR else []) + (["gosdt"] if HAVE_GOSDT else [])}
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y):
        Btr, Bte = B.iloc[tr], B.iloc[te]
        acc["sparse_l1"].append((sparse_l1(Btr, y[tr], Bte, args.max_terms) == y[te]).mean())
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(Btr, y[tr])
        acc["tree_d3"].append((t.predict(Bte) == y[te]).mean())
        xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=0)
        acc["xgboost"].append((xg.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]) == y[te]).mean())
        for k, fn in [("fasterrisk", lambda: fasterrisk(Btr, y[tr], Bte, args.max_terms)), ("gosdt", lambda: gosdt(Btr, y[tr], Bte))]:
            if k in acc:
                try:
                    acc[k].append((fn() == y[te]).mean())
                except Exception as e:
                    print(f"[{k}] failed on {name}: {e!r}")
                    acc[k].append(np.nan)
    tr, va = next(StratifiedKFold(3, shuffle=True, random_state=1).split(B, y))
    props["rashomon_proxy"] = rashomon_proxy(B.iloc[tr], y[tr], B.iloc[va], y[va])
    mean = {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in acc.items()}
    best_sparse = np.nanmax([v for k, v in mean.items() if k != "xgboost"])
    rows.append({"dataset": name, **props, **mean, "gap_pts": 100 * (mean["xgboost"] - best_sparse)})
    print(f"[done] {name}: gap {rows[-1]['gap_pts']:.1f} pts")

T = pd.DataFrame(rows).set_index("dataset")
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
print(f"\nSparse (<= {args.max_terms} terms) vs XGBoost, {args.folds}-fold CV accuracy:")
print(T.to_string())
if len(T) >= 4:
    print("\nSpearman correlation of gap with dataset properties (exploratory, n = %d datasets):" % len(T))
    for p in ["n", "d", "d_bin", "base_rate", "share_cat", "rashomon_proxy"]:
        r, pv = spearmanr(T[p], T["gap_pts"])
        print(f"  {p:15s} rho = {r:+.2f}  (p = {pv:.2f})")
T.to_csv("p2_results.csv")
print("\nWrote p2_results.csv")
