#!/usr/bin/env python3
"""Check that the Week 9 animation page's JavaScript engine agrees with scikit-learn.

Usage:  /opt/anaconda3/bin/python tools/check_week9_page.py      (needs node on the PATH)

The page (assignments/practice_activities/week_9/algorithms.html) computes everything in the browser
with a small engine between the `ENGINE START` / `ENGINE END` markers. This script runs that block under
node on the eight-house data and compares, for every new-house size on the slider (8 to 34 by 0.5):
  * LinearRegression: intercept, slope, sum of squared residuals, prediction
  * KNeighborsRegressor for k in 1, 2, 3, 5, 8: prediction (ties are skipped, scikit-learn breaks them arbitrarily)
  * DecisionTreeRegressor for max_depth 1, 2, 3: thresholds, leaf means, node samples / value / squared_error, prediction
and, for the cross-validation section, for K in 2, 3, 4, 8 with and without shuffling:
  * KFold: the row order the page hard-codes for random_state=5544 and every fold's train / validation indices
  * per-fold training and validation RMSE and their means for linear regression, k-NN (k up to the smallest
    training set) and trees (max_depth 1-4), against a hand loop and against cross_val_score
Exit status 1 and a message on the first disagreement.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "assignments" / "practice_activities" / "week_9" / "algorithms.html"

html = PAGE.read_text()
engine = re.search(r"/\* ENGINE START.*?/\* ENGINE END \*/", html, re.S).group(0)
node = shutil.which("node") or "/opt/homebrew/bin/node"
JS = engine + r"""
const E = module.exports; const xs = E.DATA.size, ys = E.DATA.price; const N = xs.length;
const out = {line: null, knn: {}, tree: {}, perm: E.PERM5544, cv: {}};
const L = E.fitLine(xs, ys); out.line = {intercept: L.intercept, slope: L.slope, sse: L.sse};
const grid = []; for (let x = 8; x <= 34.001; x += 0.5) grid.push(Math.round(x * 2) / 2);
out.grid = grid; out.linePred = grid.map(x => L.predict(x));
for (const k of [1, 2, 3, 5, 8]) out.knn[k] = grid.map(x => { const r = E.knn(xs, ys, x, k); return r.tie ? null : r.pred; });
for (const d of [1, 2, 3]) { const root = E.growTree(xs, ys, d); const leaves = E.treeLeaves(root).map(l => l.node.mean);
  const thr = [], nodes = []; (function w(n) { nodes.push({n: n.n, value: n.mean, mse: n.mse}); if (n.leaf) return; thr.push(n.split.t); w(n.left); w(n.right); })(root);
  out.tree[d] = {thresholds: thr, leaves, nodes, pred: grid.map(x => E.treePredict(root, x).pred)}; }
for (const K of [2, 3, 4, 8]) for (const sh of [false, true]) {
  const maxK = N - Math.ceil(N / K); const specs = [{type: 'linear'}];
  for (let k = 1; k <= maxK; k++) specs.push({type: 'knn', k});
  for (let d = 1; d <= 4; d++) specs.push({type: 'tree', depth: d});
  for (const s of specs) { const cv = E.crossValidate(s, xs, ys, K, sh);
    const key = `${K}|${sh ? 1 : 0}|${s.type}|${s.k || s.depth || 0}`;
    out.cv[key] = {folds: cv.folds.map(f => ({train: f.train, val: f.val, tr: f.trainRmse, va: f.valRmse, tieVal: f.tieVal, tieTrain: f.tieTrain})), meanVal: cv.meanVal, meanTrain: cv.meanTrain, anyTie: cv.anyTie}; } }
console.log(JSON.stringify(out));
"""
res = subprocess.run([node, "-e", JS], capture_output=True, text=True, check=True)
js = json.loads(res.stdout)

houses = pd.DataFrame({"size": [10, 12, 15, 18, 20, 24, 28, 32], "price": [150, 180, 200, 260, 250, 320, 360, 410]})
X, y = houses[["size"]], houses["price"]
N = len(houses)
grid = pd.DataFrame({"size": js["grid"]})
problems: list[str] = []

# ---------------------------------------------------------------- the three algorithms on all eight houses
lm = LinearRegression().fit(X, y)
sse = float(((y - lm.predict(X)) ** 2).sum())
for name, a, b in [("intercept", js["line"]["intercept"], lm.intercept_), ("slope", js["line"]["slope"], lm.coef_[0]), ("sse", js["line"]["sse"], sse)]:
    if abs(a - b) > 1e-6:
        problems.append(f"line {name}: page {a} vs sklearn {b}")
if not np.allclose(js["linePred"], lm.predict(grid), atol=1e-6):
    problems.append("line predictions differ")

for k in (1, 2, 3, 5, 8):
    sk = KNeighborsRegressor(n_neighbors=k).fit(X, y).predict(grid)
    for x, a, b in zip(js["grid"], js["knn"][str(k)], sk):
        if a is not None and abs(a - b) > 1e-6:
            problems.append(f"knn k={k} at size {x}: page {a} vs sklearn {b}")


def sk_tree(d):
    t = DecisionTreeRegressor(max_depth=d).fit(X, y).tree_
    thr, leaves, nodes = [], [], []
    def walk(n):
        nodes.append((int(t.n_node_samples[n]), float(t.value[n][0][0]), float(t.impurity[n])))
        if t.children_left[n] == -1:
            leaves.append(float(t.value[n][0][0])); return
        thr.append(float(t.threshold[n])); walk(t.children_left[n]); walk(t.children_right[n])
    walk(0)
    return thr, leaves, nodes


for d in (1, 2, 3):
    thr, leaves, nodes = sk_tree(d)
    if not np.allclose(js["tree"][str(d)]["thresholds"], thr):
        problems.append(f"tree depth {d} thresholds: page {js['tree'][str(d)]['thresholds']} vs sklearn {thr}")
    if not np.allclose(js["tree"][str(d)]["leaves"], leaves):
        problems.append(f"tree depth {d} leaves: page {js['tree'][str(d)]['leaves']} vs sklearn {leaves}")
    page_nodes = [(n["n"], n["value"], n["mse"]) for n in js["tree"][str(d)]["nodes"]]
    if len(page_nodes) != len(nodes) or not np.allclose(page_nodes, nodes, atol=1e-6):
        problems.append(f"tree depth {d} node samples/value/squared_error: page {page_nodes} vs sklearn {nodes}")
    sk = DecisionTreeRegressor(max_depth=d).fit(X, y).predict(grid)
    if not np.allclose(js["tree"][str(d)]["pred"], sk):
        problems.append(f"tree depth {d} predictions differ")

# ---------------------------------------------------------------- cross-validation
perm = np.arange(N)
np.random.RandomState(5544).shuffle(perm)
if list(js["perm"]) != perm.tolist():
    problems.append(f"PERM5544: page {js['perm']} vs numpy RandomState(5544) {perm.tolist()}")


def make_model(spec_type, h):
    if spec_type == "linear":
        return LinearRegression()
    if spec_type == "knn":
        return KNeighborsRegressor(n_neighbors=h)
    return DecisionTreeRegressor(max_depth=h)


n_cv = 0
n_tie_skipped = 0
for key, res_js in js["cv"].items():
    K_s, sh_s, spec_type, h_s = key.split("|")
    K, shuffle, h = int(K_s), sh_s == "1", int(h_s)
    kf = KFold(n_splits=K, shuffle=shuffle, random_state=5544 if shuffle else None)
    splits = list(kf.split(X))
    tr_rmse, va_rmse = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        pf = res_js["folds"][i]
        if pf["train"] != train_idx.tolist() or pf["val"] != val_idx.tolist():
            problems.append(f"cv {key} fold {i + 1} indices: page train {pf['train']} val {pf['val']} vs sklearn {train_idx.tolist()} / {val_idx.tolist()}")
            continue
        m = make_model(spec_type, h).fit(X.iloc[train_idx], y.iloc[train_idx])
        tr = root_mean_squared_error(y.iloc[train_idx], m.predict(X.iloc[train_idx]))
        va = root_mean_squared_error(y.iloc[val_idx], m.predict(X.iloc[val_idx]))
        tr_rmse.append(tr); va_rmse.append(va)
        # ties (equal distances, or two splits with equal SSE) are broken arbitrarily by scikit-learn: the page
        # flags them and the numbers that depend on them are not compared
        if pf["tieTrain"] or pf["tieVal"]:
            n_tie_skipped += 1
        if not pf["tieTrain"] and abs(pf["tr"] - tr) > 1e-6:
            problems.append(f"cv {key} fold {i + 1} training RMSE: page {pf['tr']} vs sklearn {tr}")
        if not pf["tieVal"] and abs(pf["va"] - va) > 1e-6:
            problems.append(f"cv {key} fold {i + 1} validation RMSE: page {pf['va']} vs sklearn {va}")
    if res_js["anyTie"]:
        n_cv += 1
        continue
    if tr_rmse and (abs(res_js["meanTrain"] - np.mean(tr_rmse)) > 1e-6 or abs(res_js["meanVal"] - np.mean(va_rmse)) > 1e-6):
        problems.append(f"cv {key} means: page {res_js['meanTrain']:.4f}/{res_js['meanVal']:.4f} vs sklearn {np.mean(tr_rmse):.4f}/{np.mean(va_rmse):.4f}")
    scores = cross_val_score(make_model(spec_type, h), X, y, cv=kf, scoring="neg_root_mean_squared_error")
    if abs(res_js["meanVal"] - (-scores.mean())) > 1e-6:
        problems.append(f"cv {key}: page mean validation RMSE {res_js['meanVal']:.4f} vs cross_val_score {-scores.mean():.4f}")
    n_cv += 1

if problems:
    print("\n".join("MISMATCH: " + p for p in problems)); sys.exit(1)
print(f"OK: page engine matches scikit-learn on {len(js['grid'])} slider positions "
      f"(line, knn k=1/2/3/5/8, tree depth 1/2/3 incl. node stats) and on {n_cv} cross-validation set-ups "
      f"(K = 2/3/4/8, shuffled and not, linear / k-NN / trees; {n_tie_skipped} tie-dependent fold scores flagged on the page and "
      f"not compared). Line: {lm.intercept_:.2f} + {lm.coef_[0]:.3f} x size, SSE {sse:.1f}.")
