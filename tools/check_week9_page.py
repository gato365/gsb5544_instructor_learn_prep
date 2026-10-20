#!/usr/bin/env python3
"""Check that the Week 9 animation page's JavaScript engine agrees with scikit-learn.

Usage:  /opt/anaconda3/bin/python tools/check_week9_page.py      (needs node on the PATH)

The page (assignments/practice_activities/week_9/algorithms.html) computes everything in the browser
with a small engine between the `ENGINE START` / `ENGINE END` markers. This script runs that block under
node on the eight-house data and compares, for every new-house size on the slider (8 to 34 by 0.5):
  * LinearRegression: intercept, slope, sum of squared residuals, prediction
  * KNeighborsRegressor for k in 1, 2, 3, 5, 8: prediction (ties are skipped, scikit-learn breaks them arbitrarily)
  * DecisionTreeRegressor for max_depth 1, 2, 3: thresholds, leaf means and prediction
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "assignments" / "practice_activities" / "week_9" / "algorithms.html"

html = PAGE.read_text()
engine = re.search(r"/\* ENGINE START.*?/\* ENGINE END \*/", html, re.S).group(0)
node = shutil.which("node") or "/opt/homebrew/bin/node"
JS = engine + r"""
const E = module.exports; const xs = E.DATA.size, ys = E.DATA.price;
const out = {line: null, knn: {}, tree: {}};
const L = E.fitLine(xs, ys); out.line = {intercept: L.intercept, slope: L.slope, sse: L.sse};
const grid = []; for (let x = 8; x <= 34.001; x += 0.5) grid.push(Math.round(x * 2) / 2);
out.grid = grid; out.linePred = grid.map(x => L.predict(x));
for (const k of [1, 2, 3, 5, 8]) out.knn[k] = grid.map(x => { const r = E.knn(xs, ys, x, k); return r.tie ? null : r.pred; });
for (const d of [1, 2, 3]) { const root = E.growTree(xs, ys, d); const leaves = E.treeLeaves(root).map(l => l.node.mean);
  const thr = []; (function w(n) { if (n.leaf) return; thr.push(n.split.t); w(n.left); w(n.right); })(root);
  out.tree[d] = {thresholds: thr, leaves, pred: grid.map(x => E.treePredict(root, x).pred)}; }
console.log(JSON.stringify(out));
"""
res = subprocess.run([node, "-e", JS], capture_output=True, text=True, check=True)
js = json.loads(res.stdout)

houses = pd.DataFrame({"size": [10, 12, 15, 18, 20, 24, 28, 32], "price": [150, 180, 200, 260, 250, 320, 360, 410]})
X, y = houses[["size"]], houses["price"]
grid = pd.DataFrame({"size": js["grid"]})
problems: list[str] = []

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
    thr, leaves = [], []
    def walk(n):
        if t.children_left[n] == -1:
            leaves.append(float(t.value[n][0][0])); return
        thr.append(float(t.threshold[n])); walk(t.children_left[n]); walk(t.children_right[n])
    walk(0)
    return thr, leaves

for d in (1, 2, 3):
    thr, leaves = sk_tree(d)
    if not np.allclose(js["tree"][str(d)]["thresholds"], thr):
        problems.append(f"tree depth {d} thresholds: page {js['tree'][str(d)]['thresholds']} vs sklearn {thr}")
    if not np.allclose(js["tree"][str(d)]["leaves"], leaves):
        problems.append(f"tree depth {d} leaves: page {js['tree'][str(d)]['leaves']} vs sklearn {leaves}")
    sk = DecisionTreeRegressor(max_depth=d).fit(X, y).predict(grid)
    if not np.allclose(js["tree"][str(d)]["pred"], sk):
        problems.append(f"tree depth {d} predictions differ")

if problems:
    print("\n".join("MISMATCH: " + p for p in problems)); sys.exit(1)
print(f"OK: page engine matches scikit-learn on {len(js['grid'])} slider positions "
      f"(line, knn k=1/2/3/5/8, tree depth 1/2/3). Line: {lm.intercept_:.2f} + {lm.coef_[0]:.3f} x size, SSE {sse:.1f}.")
