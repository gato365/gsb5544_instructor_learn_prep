#!/usr/bin/env python3
"""Build the Week 9 practice-activity INSTRUCTOR SOLUTION notebooks.

  PA 9.1 — K-Nearest Neighbors and Decision Trees for Regression (wine + Ames)
  PA 9.2 — Training and Test Errors, Cross-Validation, Pipelines (Ames)

Usage:  python3 tools/build_week9_pas.py                 # build + execute both -solution files
        python3 tools/build_week9_pas.py --pa 9.2        # just one of them (9.1 or 9.2)
        python3 tools/build_week9_pas.py --no-exec       # build without executing

PA 9.2: the student notebook `GSB_5544_PA_9_2_Cross_Validation.ipynb` is left untouched. Every numbered
question cell is followed by a written answer plus the code that checks it (the answer to the first part 11
goes after the `cross_val_score` cell it refers to; that cell is tagged `raises-exception` so execution runs
past its intended failure and keeps the output). The empty code cell under "Practice Activity" is replaced by
the worked competition: exploration, a shared-folds `judge` helper, linear / polynomial / KNN / tree candidates,
the leaderboard, and the five-beat discussion.

The student notebook `GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression.ipynb` is the source of truth for the
questions and is left untouched. This script writes a `-solution.ipynb` sibling:
  * every "**Sketch on paper and discuss with your partner.**" / "**Discuss with your partner.**" cell (in order)
    is replaced by a written answer plus the code that draws the sketch;
  * every "# YOUR CODE HERE. ADD CELLS AS NEEDED" cell (in order) is replaced by an answer group:
    approach, complete runnable code, what the code does, expected output, common mistakes;
  * a discussion cell is added after the K = 1, 2, 10, 27 loop (question 4).
Data (read live, network needed to execute): the Bordeaux wine data (38 vintages) and the Ames housing data
(2,930 homes) from dlsun.github.io/pods. Needs scikit-learn, pandas and matplotlib in the Anaconda kernel.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_9"
JUPYTER = "/opt/anaconda3/bin/jupyter"
PA91_SRC = WEEK / "GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


PA91_TITLE = "# K-Nearest Neighbors and Decision Trees for Regression — INSTRUCTOR SOLUTION"

PA91_INTRO = md("""
**How to use this notebook.** The sketch-and-discuss questions get a written answer *and* the code that draws
the sketch, so the expected picture is on screen. The coding questions (11–14) are answered in five beats:
**Approach** → **code** (complete and executed) → **What the code does** → **Expected output** → **Common
mistakes**. Everything else is the student notebook, unchanged.

**Links to the topic.** Topic 9.1 and its animation page (`week_9/algorithms.html` on the course site) show
the same three mechanisms on eight houses: the line, the k nearest neighbors and their average, the SSE-scored
splits of a tree. The wine questions here are the same ideas on 27 vintages.

Model names used below: `knn1`, `knn2`, `knn27`, `knn_model` (the notebook's own name for the neighborhood
model), `knn_pipe`, `tree_pipe`. Random splits use `random_state=5544`.

*Note on the K = 1 cell further down:* the comment `# ADD YOUR CODE HERE,` inside `pd.Series(...)` is a
leftover in the student notebook; the prediction line below it is already complete and the cell runs as is.
""")

# ---- Sketch / discuss answers, in document order ---------------------------------------------------------

S1 = [
    md("""
**Answer 1 (1-nearest neighbor).** The regression function is a **staircase**. For any age, the prediction is
the log price of the single closest training vintage, so between two training ages nothing changes and the
function is flat; it jumps at the **midpoints** between consecutive training ages (12.5, 13.5, 14.5, …). The
function passes through every training point.

| age | nearest training age | prediction |
|---|---|---|
| 12 | 12 (the 1980 vintage) | 2.64 |
| 13 | 13 (1979) | 3.04 |
| 11.5 | 12 (nothing is closer on the left) | 2.64 |
| 12.49 | 12 | 2.64 |
| 12.51 | 13 | 3.04 |
| 13.49 | 13 | 3.04 |
| 13.51 | 14 (1978) | 3.30 |

The code below checks the table and draws the staircase.
"""),
    code("""
from sklearn.neighbors import KNeighborsRegressor

knn1 = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
ages = pd.DataFrame({"age": [12, 13, 11.5, 12.49, 12.51, 13.49, 13.51]})
ages["predicted"] = knn1.predict(ages).round(3)
ages
"""),
    code("""
grid = pd.DataFrame({"age": np.linspace(10, 45, 1000)})
ax = bordeaux_train.plot.scatter(x="age", y="log(price)")
ax.plot(grid["age"], knn1.predict(grid), color="orange", label="1-NN regression function")
ax.legend();
"""),
]

S2 = [
    md("""
**Answer 2 (2-nearest neighbors).** Still a staircase, but each step is the **average of two** vintages, so
the function no longer passes through the points, and the jumps move: the pair of nearest ages changes at the
**training ages themselves**, not at the midpoints. Between 12 and 13 the two nearest are {12, 13}; between 13
and 14 they are {13, 14}.

| age | two nearest ages | prediction |
|---|---|---|
| 12 | 12 and 13 | (2.639 + 3.045) / 2 = 2.84 |
| 12.99 | 12 and 13 | 2.84 |
| 13.01 | 13 and 14 | (3.045 + 3.296) / 2 = 3.17 |
| 13.99 | 13 and 14 | 3.17 |
| 14.01 | 14 and 15 | (3.296 + 2.398) / 2 = 2.85 |

At *exactly* age 13 the second neighbor is a tie (12 and 14 are both one year away); scikit-learn breaks the
tie arbitrarily (here it picks 14, giving 3.17). Ties only matter at the jump points.
"""),
    code("""
knn2 = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
ages = pd.DataFrame({"age": [12, 12.99, 13, 13.01, 13.99, 14, 14.01]})
ages["predicted"] = knn2.predict(ages).round(3)
ages
"""),
    code("""
ax = bordeaux_train.plot.scatter(x="age", y="log(price)")
ax.plot(grid["age"], knn1.predict(grid), color="orange", alpha=0.5, label="1-NN")
ax.plot(grid["age"], knn2.predict(grid), color="blue", label="2-NN")
ax.legend();
"""),
]

S3 = [
    md("""
**Answer 3 (27-nearest neighbors).** 27 is the **number of training vintages** (1952–1980, minus the two
missing years). Whatever the age, the 27 nearest neighbors are *all* the training wines, so every prediction is
the same number: the mean log price, **3.158**. The regression function is a **horizontal line** at the mean,
and it ignores age completely.
"""),
    code("""
knn27 = KNeighborsRegressor(n_neighbors=27).fit(X_train, y_train)
print("number of training wines:", len(X_train))
print("mean log(price):", round(y_train.mean(), 3))
print("27-NN predictions at ages 12, 25, 40:", knn27.predict(pd.DataFrame({"age": [12, 25, 40]})).round(3))
"""),
]

S4 = md("""
**Discussion 4 (K = 1, 2, 10, 27).** The functions should look like the sketches: K = 1 is the jagged orange
staircase through every point; K = 2 (blue) averages pairs and is slightly smoother; K = 10 (green) is a slow
wave that follows the general rise of price with age; K = 27 (magenta) is flat at the mean. **As K increases
the function gets smoother and less flexible.** Small K fits the training data closely, noise included (K = 1
has zero training error); large K averages over so many wines that it stops responding to age at all. Neither
extreme predicts *new* vintages well; a middle K is chosen on test data, which is what questions 12 and 14
practise.
""")

S5 = [
    md("""
**Answer 5 (neighborhoods).** The only neighborhood whose predicted price **equals its mean** is **Blueste**,
which has **exactly 10 houses**.

Why: after one-hot encoding, every house in the same neighborhood has *identical* features, so all of them are
at distance 0 from the query, and every house in any other neighborhood is at distance √2. The 10-nearest
neighbors of "Blueste" are therefore precisely its 10 houses, and the prediction is their mean.

- A neighborhood with **more than 10 houses** (NAmes has 443): 443 houses tie at distance 0, and scikit-learn
  keeps *some* 10 of them; which 10 is an arbitrary tie-break, not the first ten rows. The prediction is the mean
  of those 10 houses, not of all 443 (142,771 versus 145,097 for NAmes).
- A neighborhood with **fewer than 10 houses** (Greens 8, GrnHill 2, Landmrk 1): its own houses are used, then
  the remaining neighbors are filled from *other* neighborhoods, all tied at distance √2, so the prediction is
  pulled toward those neighborhoods' prices (GrnHill: 150,790 predicted versus a mean of 280,000).

A categorical feature with KNN therefore behaves like a group mean only when the group size happens to equal
k. The code shows the neighbor counts and which houses the model actually used.
"""),
    code("""
counts = df_ames["Neighborhood"].value_counts()
print("neighborhoods with 10 or fewer houses:")
print(counts[counts <= 10])

def neighbors_used(neighborhood):
    query = knn_model["preprocessing"].transform(pd.DataFrame({"Neighborhood": [neighborhood]}))
    dist, idx = knn_model["knn"].kneighbors(query)
    used = df_ames.iloc[idx[0]][["Neighborhood", "SalePrice"]].assign(distance=dist[0].round(2))
    print(f"\\n{neighborhood}: {counts[neighborhood]} houses in the data; predicted = mean of these 10 = {used['SalePrice'].mean():,.0f}")
    return used

neighbors_used("Blueste")
"""),
    code("""
neighbors_used("GrnHill")
"""),
]

SURFACE_HELPER = code("""
def knn_surface(points, targets, k, title):
    \"\"\"Draw the KNN regression surface for a tiny two-feature training set (features already scaled).\"\"\"
    pts = pd.DataFrame(points, columns=["x1", "x2"])
    model = KNeighborsRegressor(n_neighbors=k).fit(pts, targets)
    g1, g2 = np.meshgrid(np.linspace(0, 5, 251), np.linspace(0, 5, 251))
    z = model.predict(pd.DataFrame({"x1": g1.ravel(), "x2": g2.ravel()})).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cs = ax.contourf(g1, g2, z, levels=sorted(set(np.round(z.ravel(), 6))) + [z.max() + 1], cmap="viridis")
    fig.colorbar(cs, ax=ax, label="predicted y")
    ax.scatter(pts["x1"], pts["x2"], c="white", edgecolor="black", s=90, zorder=3)
    for (a, b), t in zip(points, targets):
        ax.annotate(f"y = {t}", (a, b), xytext=(7, 7), textcoords="offset points", color="white", fontweight="bold")
    ax.set_xlabel("feature 1 (scaled)"); ax.set_ylabel("feature 2 (scaled)"); ax.set_title(title)
""")

S6 = [
    md("""
**Answer 6 (1-NN, two training points).** The feature plane is cut in two by the **perpendicular bisector** of
the segment joining the two points. Every location on A's side is closer to A, so the surface is flat at A's
target there; on B's side it is flat at B's target. Two flat plateaus with a straight cliff between them.
"""),
    SURFACE_HELPER,
    code("""
knn_surface(points=[(1, 1), (4, 3)], targets=[10, 30], k=1, title="1-NN, 2 training points")
"""),
]

S7 = [
    md("""
**Answer 7 (1-NN, three training points).** Three flat regions, one per training point: each region is the
set of locations closer to that point than to the other two (its *Voronoi cell*). The boundaries are the three
perpendicular bisectors, which meet at a single point equidistant from all three observations.
"""),
    code("""
knn_surface(points=[(1, 1), (4, 1), (2, 4)], targets=[10, 20, 30], k=1, title="1-NN, 3 training points")
"""),
]

S8 = [
    md("""
**Answer 8 (2-NN, three training points).** Again three flat regions with the *same* bisector boundaries, but
the values and the geometry flip: with k = 2 each prediction is the mean of the **two closest** points, which is
"everything except the farthest". So the region where C is farthest predicts (y_A + y_B)/2 = 15, the region where
B is farthest predicts (y_A + y_C)/2 = 20, and the region where A is farthest predicts (y_B + y_C)/2 = 25. Each
region now sits *opposite* its excluded point rather than around its own point.
"""),
    code("""
knn_surface(points=[(1, 1), (4, 1), (2, 4)], targets=[10, 20, 30], k=2, title="2-NN, 3 training points")
"""),
]

S9 = [
    md("""
**Answer 9 (one split at 20.5).** A tree with one split is a **two-step staircase**: for age ≤ 20.5 it predicts
the mean log price of those vintages (9 wines, **2.830**); for age > 20.5 the mean of the rest (18 wines,
**3.322**). One vertical jump at 20.5, flat on either side.
"""),
    code("""
def tree_steps(splits):
    \"\"\"Group means for a tree whose thresholds on age are `splits`, as a step function on the grid.\"\"\"
    bins = [-np.inf] + list(splits) + [np.inf]
    means = bordeaux_train.groupby(pd.cut(bordeaux_train["age"], bins), observed=True)["log(price)"].agg(["mean", "count"])
    grid_bins = pd.cut(grid["age"], bins)
    return means, grid_bins.map(means["mean"]).astype(float)

means1, step1 = tree_steps([20.5])
print(means1.round(3))
ax = bordeaux_train.plot.scatter(x="age", y="log(price)")
ax.plot(grid["age"], step1, color="firebrick", label="tree: split at 20.5")
ax.legend();
"""),
]

S10 = [
    md("""
**Answer 10 (splits at 20.5 and 30.5).** Three steps: **2.830** for age ≤ 20.5 (9 wines), **3.082** for 20.5 <
age ≤ 30.5 (10 wines), **3.622** for age > 30.5 (8 wines). Each extra split adds one more flat segment; the
second split is applied inside the right-hand group of question 9, which is how a tree grows — it never redraws
earlier splits.
"""),
    code("""
means2, step2 = tree_steps([20.5, 30.5])
print(means2.round(3))
ax = bordeaux_train.plot.scatter(x="age", y="log(price)")
ax.plot(grid["age"], step1, color="firebrick", alpha=0.4, label="one split")
ax.plot(grid["age"], step2, color="black", label="two splits")
ax.legend();
"""),
]

PA91_DISCUSS = [S1, S2, S3, S5, S6, S7, S8, S9, S10]

# ---- Coding questions 11–14 -------------------------------------------------------------------------------

Q11 = [
    md("""
### Solution 11 — 10-NN with five features, in a pipeline

**Approach.** Two kinds of columns, two preprocessing steps in one `ColumnTransformer`: `OneHotEncoder` for
`Neighborhood` and `StandardScaler` for the four numeric features (KNN measures distances, and square feet
would otherwise swamp bedrooms and bathrooms). Chain it with `KNeighborsRegressor(n_neighbors=10)` in a
`Pipeline`, fit on all 2,930 houses as the question asks, predict, and plot predicted against actual with a
45° reference line.
"""),
    code("""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

features = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath", "Neighborhood"]
numeric  = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath"]
X = df_ames[features]
y = df_ames["SalePrice"]

preprocessing = ColumnTransformer([
    ("dummify", OneHotEncoder(handle_unknown="ignore"), ["Neighborhood"]),
    ("scale",   StandardScaler(), numeric),
])
knn_pipe = Pipeline([
    ("preprocessing", preprocessing),
    ("knn", KNeighborsRegressor(n_neighbors=10)),
])
knn_pipe.fit(X, y)

df_ames["predicted_knn10"] = knn_pipe.predict(X)
print("RMSE on the fitted data:", round(root_mean_squared_error(y, df_ames["predicted_knn10"])))
"""),
    code("""
def predicted_vs_actual(predicted, title):
    ax = df_ames.plot.scatter(x="SalePrice", y=predicted, alpha=0.3, figsize=(5.5, 5))
    lim = [0, df_ames["SalePrice"].max()]
    ax.plot(lim, lim, color="firebrick", label="predicted = actual")
    ax.set_xlabel("actual SalePrice"); ax.set_ylabel("predicted SalePrice"); ax.set_title(title); ax.legend()

predicted_vs_actual("predicted_knn10", "10-nearest neighbors")
"""),
    md("""
**What the code does.** `ColumnTransformer` routes each column to its own transformer and concatenates the
results (28 dummy columns + 4 scaled columns); `Pipeline` makes the whole thing one model with `fit` and
`predict`. `handle_unknown="ignore"` lets `predict` cope with a neighborhood it has not seen (it matters once we
split the data in question 12). The helper plots predictions against the truth.

**Expected output.** RMSE about **34,700** dollars on the houses the model was fitted on. The cloud follows the
45° line with a fan that widens for expensive houses, and a few very large homes on the right are badly
under-predicted (their ten nearest neighbors are cheaper).

**Common mistakes.**
- Leaving the numeric features **unscaled**: `Gr Liv Area` (hundreds to thousands of square feet) then
  dominates every distance and the bedroom/bath columns are ignored; test RMSE in question 12 is about 56,500
  instead of 40,100 with `random_state=5544`.
- Putting `Neighborhood` through the scaler (it is text) or forgetting it in the `ColumnTransformer` (columns
  not named are **dropped** by default; `remainder="passthrough"` keeps them).
- Plotting the log of price: the question asks for `SalePrice` itself.
"""),
]

Q12 = [
    md("""
### Solution 12 — choosing k

**Approach.** "Best" has to mean *best at predicting houses the model has not seen*, so hold out a test set
(`train_test_split`, 25 %, `random_state=5544`) and compare **test RMSE** across a handful of k. Training RMSE is
shown alongside to make the point that it cannot be used: k = 1 reproduces the training prices almost exactly.
"""),
    code("""
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=5544)

rows = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    pipe = Pipeline([("preprocessing", preprocessing), ("knn", KNeighborsRegressor(n_neighbors=k))])
    pipe.fit(X_tr, y_tr)
    rows.append({"k": k,
                 "train RMSE": root_mean_squared_error(y_tr, pipe.predict(X_tr)),
                 "test RMSE":  root_mean_squared_error(y_te, pipe.predict(X_te))})
k_results = pd.DataFrame(rows).set_index("k").round(0)
k_results
"""),
    code("""
ax = k_results.plot(marker="o", logx=True)
ax.set_ylabel("RMSE (dollars)"); ax.set_xlabel("k (log scale)");
"""),
    md("""
**What the code does.** One split, then the same pipeline refitted for each k on the training rows only.
Both errors are computed from the same fitted model: on the rows it saw (train) and on the rows it did not
(test).

**Expected output (random_state 5544).**

| k | train RMSE | test RMSE |
|---|---|---|
| 1 | ≈ 6,700 | ≈ 46,900 |
| 3 | ≈ 28,100 | **≈ 38,000** |
| 5 | ≈ 31,900 | ≈ 38,300 |
| 10 | ≈ 34,900 | ≈ 40,100 |
| 20 | ≈ 38,700 | ≈ 41,800 |
| 50 | ≈ 43,400 | ≈ 45,000 |
| 100 | ≈ 47,100 | ≈ 46,700 |

Training RMSE rises steadily with k (k = 1 is nearly perfect on the training rows, exactly the wine staircase
through every point). Test RMSE is U-shaped: worst at k = 1, best around **k = 3 to 5**, then slowly worse as the
average pulls in houses that are not really similar. **How to decide:** by test error, and since the two best
values are close, prefer the one that is less jumpy (k = 5) or, better, cross-validate — one split is one random
draw, and a different `random_state` can swap 3 and 5.

**Common mistakes.**
- Judging by training RMSE and concluding k = 1 is best.
- Fitting on all the data and then "testing" on part of it (the test rows were seen in training).
- Splitting *after* fitting the preprocessing on everything: the pipeline avoids this because `fit` on `X_tr`
  learns the scaler on training rows only.
"""),
]

Q13 = [
    md("""
### Solution 13 — a regression tree of depth 4

**Approach.** Same five features and the same pipeline shape, with two changes. A tree does not need scaling
(each question is about one feature), so the numeric columns pass through untouched and the thresholds in the
plot stay in square feet and bedrooms; `remainder="passthrough"` does that. Then `DecisionTreeRegressor(
max_depth=4)`, `plot_tree` with the feature names the transformer produced, and the same predicted-versus-actual
plot.
"""),
    code("""
from sklearn.tree import DecisionTreeRegressor, plot_tree

tree_preprocessing = ColumnTransformer(
    [("dummify", OneHotEncoder(), ["Neighborhood"])],
    remainder="passthrough",          # the four numeric columns, unscaled
)
tree_pipe = Pipeline([
    ("preprocessing", tree_preprocessing),
    ("tree", DecisionTreeRegressor(max_depth=4, random_state=5544)),
])
tree_pipe.fit(X, y)

feature_names = [n.replace("dummify__", "").replace("remainder__", "")
                 for n in tree_pipe["preprocessing"].get_feature_names_out()]
plt.figure(figsize=(24, 10))
plot_tree(tree_pipe["tree"], feature_names=feature_names, filled=True, rounded=True, fontsize=8, precision=0);
"""),
    code("""
df_ames["predicted_tree4"] = tree_pipe.predict(X)
print("RMSE on the fitted data:", round(root_mean_squared_error(y, df_ames["predicted_tree4"])))
print("number of leaves:", tree_pipe["tree"].get_n_leaves())
predicted_vs_actual("predicted_tree4", "regression tree, max_depth = 4")
"""),
    md("""
**What the code does.** `get_feature_names_out()` returns the transformer's column names
(`dummify__Neighborhood_Blueste`, …, `remainder__Gr Liv Area`); stripping the prefixes makes the plot readable.
`plot_tree` draws every node with its question, its `squared_error` (the mean squared error of prices around
the node mean), `samples` and `value` (the mean price, which is the prediction for that leaf).

**Expected output and interpretation.** The root asks **Gr Liv Area ≤ 1610.5 sq ft**: size is the single most
useful question. Below it, the small-house branch keeps splitting on size (1198.5, then 847 / 989 sq ft) with a
bathroom and bedroom question at the end; the large-house branch asks whether the house is in **NridgHt**
(Northridge Heights, the priciest neighborhood), then about size again (2447.5, 3225 sq ft) and, inside
NridgHt, about bedrooms. The 16 leaves predict from about **74,000** (under 750 sq ft) to **574,000** (NridgHt,
over 2,270 sq ft, two or fewer bedrooms). Reading a prediction: follow the questions from the top; e.g. a 1,400
sq ft house with 2 full baths and 3 bedrooms goes left (≤ 1610.5), right (> 1198.5), right (Full Bath > 1.5),
left (Bedroom ≤ 3.5): about 186,000.

The predicted-versus-actual plot has exactly **16 horizontal bands**, one per leaf; RMSE on the fitted data is
about **47,900**, worse than 10-NN's 34,700, because 16 flat values cannot track prices as finely as a local
average.

**Common mistakes.**
- Scaling the numeric features: harmless for the tree's accuracy (the splits are the same) but the thresholds
  come out in standard-deviation units (`Gr Liv Area ≤ 0.22`) and the tree becomes hard to read.
- Calling `plot_tree(tree_pipe)` on the pipeline instead of the tree step inside it.
- Forgetting the figure size: a depth-4 tree is 31 nodes wide and unreadable at the default size.
""")
]

Q14 = [
    md("""
### Solution 14 — choosing max_depth

**Approach.** Exactly question 12 with `max_depth` in place of k: refit on the training rows for a range of
depths (including `None`, which grows the tree until every leaf is pure), record training and test RMSE, and
pick by test error.
"""),
    code("""
rows = []
for depth in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None]:
    pipe = Pipeline([("preprocessing", tree_preprocessing),
                     ("tree", DecisionTreeRegressor(max_depth=depth, random_state=5544))])
    pipe.fit(X_tr, y_tr)
    rows.append({"max_depth": "None" if depth is None else depth,
                 "leaves": pipe["tree"].get_n_leaves(),
                 "train RMSE": root_mean_squared_error(y_tr, pipe.predict(X_tr)),
                 "test RMSE":  root_mean_squared_error(y_te, pipe.predict(X_te))})
depth_results = pd.DataFrame(rows).set_index("max_depth").round(0)
depth_results
"""),
    code("""
ax = depth_results[["train RMSE", "test RMSE"]].plot(marker="o")
ax.set_ylabel("RMSE (dollars)");
"""),
    md("""
**What the code does.** The loop is the generic "tune one hyperparameter" pattern from Topic 8.2: create, fit
on train, score on train and test, collect.

**Expected output (random_state 5544).** Training RMSE falls at every step, from about 66,500 at depth 1 to
about 4,700 with no limit (thousands of leaves, essentially one per house). Test RMSE falls quickly — 63,000
(depth 1), 49,100 (depth 4), 43,200 (depth 6) — bottoms out around **depth 10 (≈ 39,600)**, then creeps back up
(≈ 43,100 at depth 15, ≈ 42,100 unlimited). **Decision:** the depth with the lowest test RMSE, about 10; depth 4
from question 13 was too shallow for prediction (though far easier to read), and an unlimited tree overfits.
Compared with KNN, the best tree (≈ 39,600) is a little worse than the best k (≈ 38,000) on this split.

**Common mistakes.**
- Choosing by training error, which always prefers the deepest tree.
- Reading `max_depth=None` as "no tree": it means no depth limit.
- Forgetting that a different random split moves these numbers by a few thousand dollars; the *shape* of the
  curves is the lesson, and cross-validation is the fix.
""")
]

PA91_CODE_ANSWERS = [Q11, Q12, Q13, Q14]

# ============================================================================ #
#  PA 9.2 — Training and Test Errors, Cross-Validation, Pipelines
# ============================================================================ #

PA92_SRC = WEEK / "GSB_5544_PA_9_2_Cross_Validation.ipynb"
PA92_TITLE = "# Training and Test Errors, Cross-Validation, Pipelines — INSTRUCTOR SOLUTION"

PA92_INTRO = md("""
**How to use this notebook.** Each numbered question is followed by a written answer *and* the code that checks
it, so every claim has a number next to it. The code answers (part 10, the fix in part 11, and the competition
at the end) are in five beats: **Approach** → **code** (complete and executed) → **What the code does** →
**Expected output** → **Common mistakes**. Everything else is the student notebook, unchanged — including the
`cross_val_score` cell that is meant to fail; it is left as is, so its warning / error output is on screen.

**Links to the topic.** Topic 9.2 and the *Cross-validation in motion* section of the animation page
(`week_9/algorithms.html#cv` on the course site) show the same mechanics on eight houses: training vs
validation RMSE, K folds, a fresh model per fold, the average. This notebook does it on 2,930 houses.

Conventions below: `root_mean_squared_error` for RMSE in dollars; `scoring="neg_root_mean_squared_error"` in
`cross_val_score` (negate the scores); `cv=10` matches the student cell (10 consecutive folds); the competition
uses `KFold(10, shuffle=True, random_state=5544)` so every candidate is judged on the *same* folds.
""")

# ---- Answers to the numbered questions, in document order ------------------------------------------------
# Each entry: (after_code, [cells]).  after_code=True means the answer goes after the next code cell
# (question 11's `cross_val_score` cell), not directly after the question.

A1 = (False, [
    md("""
**Answer 1.** `fit` is the **learning step**. For a linear model it is least squares: scikit-learn searches over
every combination of intercept and four slopes and keeps the one that makes the **sum of squared residuals**
(actual − predicted price, squared, added over the 2,930 houses) as small as possible. Nothing else is tried;
"fit" is this optimization.

The **output** is the fitted model object itself, `model1`, which now carries the learned **parameters**:
`model1.intercept_` and `model1.coef_` (one coefficient per feature, in dollars per unit of that feature). The
trailing underscores mark "learned from data". Those five numbers *are* the model; `predict` just evaluates
intercept + Σ coefficient × feature.
"""),
    code("""
print("intercept:", round(model1.intercept_))
pd.Series(model1.coef_, index=X_train.columns, name="dollars per unit").round(1)
"""),
])

A2 = (False, [
    md("""
**Answer 2.** The training RMSE is the typical size of the model's error *on the houses it was fitted to*:

1. predict a price for every training house: `model1.predict(X_train)`;
2. subtract from the actual prices to get 2,930 residuals (actual − predicted);
3. square them (so over- and under-predictions do not cancel), average, take the square root — back to dollars.

`root_mean_squared_error(y_train, model1.predict(X_train))` does steps 2–3 in one call. "Training" because the
same rows were used to fit *and* to score.
"""),
    code("""
from sklearn.metrics import root_mean_squared_error

predicted = model1.predict(X_train)
residuals = y_train - predicted
by_hand = np.sqrt((residuals ** 2).mean())
print("training RMSE by hand:", round(by_hand), "   with scikit-learn:", round(root_mean_squared_error(y_train, predicted)))
"""),
])

A3 = (False, [
    md("""
**Answer 3.** About **$51,350**: on a typical house this model misses the sale price by roughly fifty thousand
dollars. That is big — the median house sells for $160,000, so the typical error is about a third of a typical
price.

Benchmarks for "big or small":

- **Predict the mean for every house.** The dumbest model's RMSE is the standard deviation of `SalePrice`,
  about **$79,900**. Four features bring that down to 51,350 — an improvement, but only to 64 % of the
  no-information error. (Equivalently R² ≈ 0.59: the model explains 59 % of the variance.)
- **The price scale itself**: median $160,000, so a 51,000 miss is roughly ±32 %. A realtor would not accept it.
- **A better model** (the competition at the end gets below $27,000) shows how much room there was.
"""),
    code("""
sd_price = y_train.std()
rmse_mean_model = root_mean_squared_error(y_train, np.full(len(y_train), y_train.mean()))
print(f"sd of SalePrice: {sd_price:,.0f}   RMSE of 'predict the mean': {rmse_mean_model:,.0f}")
print(f"median SalePrice: {y_train.median():,.0f}   training RMSE as a share of it: {by_hand / y_train.median():.0%}")
print(f"R^2 of model1: {model1.score(X_train, y_train):.3f}")
"""),
])

A4 = (False, [
    md("""
**Answer 4.** The training RMSE can only **decrease (or stay the same), never increase.** Least squares with
the extra column can always set that column's coefficient to 0 and recover the old model exactly, so the best
fit with five features is at least as good as the best fit with four. Here the drop is almost nothing —
`TotRms AbvGrd` is nearly redundant with bedrooms, baths and living area — but the sign is guaranteed.

That is exactly why comparing *training* RMSEs is **not useful**: the more complex model wins (or ties) by
construction, whether or not it predicts new houses any better. The comparison is rigged in favour of
complexity. We need an error measured on houses the model has *not* seen — parts 7–10.
"""),
    code("""
features5 = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath", "TotRms AbvGrd"]
model_5 = LinearRegression().fit(df_ames[features5], y_train)
rmse4 = root_mean_squared_error(y_train, model1.predict(X_train))
rmse5 = root_mean_squared_error(y_train, model_5.predict(df_ames[features5]))
print(f"training RMSE, 4 features: {rmse4:,.2f}")
print(f"training RMSE, 5 features: {rmse5:,.2f}   (decrease of {rmse4 - rmse5:,.2f} dollars)")
"""),
])

A5 = (False, [
    md("""
**Answer 5.** It *should* be **0**. Each training house's nearest neighbor is itself (distance 0), so 1-NN
"predicts" its own price exactly. Training error zero is the signature of a model that has memorized the data
— and says nothing about how it would do on a new house.

The fine print, which this data set happens to exhibit: 1,197 houses share an **identical feature row** (same
living area, bedrooms, full and half baths) with at least one other house that sold for a *different* price.
For those, several houses tie at distance 0, scikit-learn picks one of them, and the "prediction" can be the
twin's price. So 1-NN's training RMSE on these four features is not 0 but about **$21,700** (21,900 with the
features scaled, since scaling changes which twin wins a tie) — still far below any other model's, and exactly
0 as soon as the tied rows are removed (second cell).
"""),
    code("""
from sklearn.neighbors import KNeighborsRegressor

knn1 = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
print("1-NN training RMSE on all 2,930 houses:", round(root_mean_squared_error(y_train, knn1.predict(X_train))))
print("houses whose four features are identical to another house's:", X_train.duplicated(keep=False).sum())
"""),
    code("""
unique_rows = ~X_train.duplicated(keep=False)                 # houses with a one-of-a-kind feature row
knn1u = KNeighborsRegressor(n_neighbors=1).fit(X_train[unique_rows], y_train[unique_rows])
print("1-NN training RMSE with the tied rows removed:",
      round(root_mean_squared_error(y_train[unique_rows], knn1u.predict(X_train[unique_rows]))))
"""),
])

A6 = (False, [
    md("""
**Answer 6.** It **increases** as k grows. With k = 1 each training house is predicted by itself; with larger k
the prediction is the average of k houses, which pulls it away from the house's own price toward its
neighbors'. The regression function gets smoother and less flexible, so it matches the training prices less
well. At k = 2,930 (every house is a neighbor of every other) the prediction is the overall mean for everyone
and the training RMSE equals the standard deviation of price, about $79,900 — the "predict the mean" benchmark
from part 3.
""")
    ,
    code("""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rows = []
for k in [1, 2, 5, 20, 100, 500, len(X_train)]:
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)).fit(X_train, y_train)
    rows.append({"k": k, "training RMSE": root_mean_squared_error(y_train, knn.predict(X_train))})
pd.DataFrame(rows).set_index("k").round(0)
"""),
])

A7 = (False, [
    md("""
**Answer 7.** The test RMSE is the error on the houses the model will actually be used for: houses that have
**not sold yet**, or whose prices we do not know. For those, there is no `SalePrice` to compare the prediction
with, so the error cannot be computed. And every house whose price we *do* know is already in the training
data — scoring the model on them gives the training RMSE, which part 4 showed is rigged. The test error is the
number we want and the one number we can never calculate directly; the best we can do is **estimate** it,
which is what cross-validation is for.
"""),
])

A8 = (False, [
    md("""
**Answer 8.** Cross-validation manufactures "unseen" houses out of the training data:

1. **Cut** the training rows into K folds (K = 10 below: ten groups of 293 houses).
2. For fold 1: **hold it out**, fit a *fresh* copy of the model on the other nine folds, predict the 293 held-out
   houses, and compute the RMSE of those predictions. The model never saw their prices, so this is a genuine
   out-of-sample error.
3. **Repeat** for folds 2 … 10. Every house is held out exactly once and is predicted by a model that was not
   fitted on it.
4. **Average** the ten RMSEs. The average estimates the test RMSE; the spread of the ten shows how uncertain the
   estimate is.

Why it estimates the *test* error: each fold's model is fitted on about 90 % of the data, almost as much as the
final model, and is scored only on houses it did not train on — the same situation the final model will face
with new houses. (K is the number of folds; it has nothing to do with the k of k-nearest neighbors.)
"""),
])

A9 = (False, [
    md("""
**Answer 9.** **Greater than**, almost always: a model is scored on houses it did not fit, so it does a little
worse than on the ones it did. For *this* model the gap is tiny — training RMSE 51,350, 10-fold estimate of
test RMSE about **51,450** — because a line with five parameters cannot memorize 2,930 houses; it is too simple
to overfit. (A 1-NN model would show a huge gap: 21,900 vs about 60,000.)

`cross_val_score` returns one score per fold, **negative** because scikit-learn maximizes scores; negate and
average. With `scoring="neg_mean_squared_error"` as in the pre-class reading, take `np.sqrt(-scores)` fold by
fold before averaging — it gives the same number.
"""),
    code("""
from sklearn.model_selection import cross_val_score

def cv_rmse(model, X, cv=10):
    \"\"\"Cross-validation estimate of the test RMSE: mean of the per-fold RMSEs.\"\"\"
    scores = cross_val_score(model, X, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()

scores = cross_val_score(LinearRegression(), X_train, y_train, cv=10, scoring="neg_root_mean_squared_error")
print("per-fold validation RMSE:", (-scores).round(0))
print(f"estimated test RMSE (mean of the 10): {-scores.mean():,.0f}    training RMSE: {rmse4:,.0f}")
"""),
    code("""
# the reading's route: negative MSE per fold, square-root each, then average -> the same estimate
mse_scores = cross_val_score(LinearRegression(), X_train, y_train, cv=10, scoring="neg_mean_squared_error")
print("via neg_mean_squared_error:", round(np.sqrt(-mse_scores).mean()))
"""),
])

A10 = (False, [
    md("""
**Answer 10.** **Not necessarily.** Adding a feature is guaranteed to lower the *training* RMSE (part 4) but
can leave the *test* RMSE unchanged or make it worse: a column that carries no new information adds a
coefficient that is fitted to noise, and the noise does not repeat on new houses.

### The comparison

**Approach.** Same `cv_rmse` helper, same `cv=10`, two feature lists.
"""),
    code("""
cv4 = cv_rmse(LinearRegression(), X_train)
cv5 = cv_rmse(LinearRegression(), df_ames[features5])
print(f"estimated test RMSE, 4 features: {cv4:,.0f}")
print(f"estimated test RMSE, 5 features: {cv5:,.0f}   (change: {cv5 - cv4:+,.0f})")
"""),
    md("""
**What the code does.** `cross_val_score` refits `LinearRegression()` ten times for each feature set and scores
each fit on its held-out fold; `cv_rmse` averages the ten RMSEs.

**Expected output.** 4 features ≈ **51,450**; 5 features ≈ **51,580**: the model *with* `TotRms AbvGrd` is
slightly **worse** on held-out houses even though its training RMSE was (microscopically) better. By this
criterion the four-feature model is the better one — and the honest conclusion is that total rooms adds nothing
once bedrooms, baths and living area are known. (The difference is small relative to the fold-to-fold spread,
so "no better" is the safest wording.)

**Common mistakes.** Comparing training RMSEs and declaring the bigger model the winner; using different folds
for the two models (set `cv` the same way for both); reporting the negative scores without flipping the sign.
"""),
])

A11_error = (True, [
    md("""
**Answer 11 (why the cell fails).** Two layers.

1. **The immediate error**: `cross_val_score` was never imported in the student notebook, so Python raises
   `NameError`. (In this solution notebook it *was* imported in part 9, so the cell runs — and exposes the
   second, more interesting problem.)
2. **The real problem.** Cross-validation refits the *whole pipeline* — `OneHotEncoder` included — on each
   fold's nine training folds, then transforms the held-out fold. `Neighborhood` has rare values: **Landmrk**
   has one house, **GrnHill** two, **Greens** eight. In fold 10 the single Landmrk house is in the held-out
   fold, so the encoder fitted on the other nine folds has never seen "Landmrk" and refuses to transform it:
   `ValueError: Found unknown categories ['Landmrk']`. Current scikit-learn catches that during scoring,
   issues a warning, and records **`nan`** for the fold (the last score above); older versions raised the error
   outright — hence "will (probably) error".

Why categorical variables are special here: a numeric column always *can* be transformed; a one-hot encoder
can only transform categories it was fitted on, and with few houses in a category there is a real chance that
all of them land in the validation fold.

### The fix

**Approach.** Tell the encoder what to do with a category it has not seen: `handle_unknown="ignore"` produces
an all-zero dummy row for it (the house is treated as "none of the known neighborhoods"). Shuffling the folds
also makes the problem *less* likely but does not rule it out, so the encoder setting is the real fix.
"""),
    code("""
ct_fixed = ColumnTransformer(
    [("standardize", StandardScaler(), ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath"]),
     ("dummify", OneHotEncoder(handle_unknown="ignore"), ["Neighborhood"])],
    remainder="drop",
)
neighborhood_model_fixed = Pipeline([("preprocessing", ct_fixed), ("lm", LinearRegression())])

scores = cross_val_score(neighborhood_model_fixed, X=X_train, y=y_train,
                         scoring="neg_root_mean_squared_error", cv=10)
print("per-fold validation RMSE:", (-scores).round(0))
print(f"estimated test RMSE with Neighborhood: {-scores.mean():,.0f}   (without it, part 9: {cv4:,.0f})")
"""),
    md("""
**What the code does.** Same pipeline, one argument added to the encoder; `cross_val_score` now completes all
ten folds and the Landmrk house in fold 10 is encoded as all zeros.

**Expected output.** Ten numbers, no `nan`; estimated test RMSE about **39,400**, down from 51,450 — the
neighborhood is worth about twelve thousand dollars of RMSE. (The fold with the one Landmrk house predicts it
as a "no-neighborhood" house, which is a little wrong for one house and irrelevant for the average.)

**Common mistakes.** Dropping the rare neighborhoods from the data instead of handling them (the final model
must cope with them too); fitting the encoder on *all* the data first and then cross-validating — it hides
the error but leaks the validation rows' categories into training (part 12); forgetting that `X_train` was
redefined above to include `Neighborhood`.
"""),
])

A11_standardize = (False, [
    md("""
**Answer 11 (standardizing first).** The **coefficients change, the training RMSE does not.**

Standardizing replaces each feature x by (x − mean) / sd. A linear model absorbs that perfectly: the new
coefficient for a feature is the old coefficient × that feature's standard deviation (so it now reads "dollars
per *standard deviation* of living area" instead of "dollars per square foot"), and the intercept becomes the
mean price, because every standardized feature is 0 at the average house. The fitted *predictions* are
identical to the dollar, so the residuals and the training RMSE are unchanged.

Standardizing matters for KNN (distances), not for least squares (coefficients adjust) — Topic 9.1's table.
It can still be useful with linear regression for comparing the sizes of coefficients across features.
"""),
    code("""
X4 = df_ames[["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath"]]
std_model = make_pipeline(StandardScaler(), LinearRegression()).fit(X4, y_train)

compare = pd.DataFrame({
    "coefficient (raw)":          model1.coef_,
    "coefficient (standardized)": std_model[-1].coef_,
    "raw coef x feature sd":      model1.coef_ * X4.std(ddof=0),
}, index=X4.columns).round(0)
print("intercept raw:", round(model1.intercept_), "  standardized:", round(std_model[-1].intercept_), "  mean price:", round(y_train.mean()))
print("largest difference between the two models' predictions:", np.abs(std_model.predict(X4) - model1.predict(X4)).max().round(6))
print("training RMSE standardized:", round(root_mean_squared_error(y_train, std_model.predict(X4))))
compare
"""),
])

A12 = (False, [
    md("""
**Answer 12.** Standardization is a **fitted step**: `StandardScaler` learns a mean and a standard deviation
from whatever rows it is given. Inside cross-validation it must therefore be treated like the model:

- in each fold, the scaler is **fitted on the nine training folds only**, and the held-out fold is transformed
  with *those* means and standard deviations — exactly what will happen later when a new house arrives and is
  scaled with the training statistics;
- it is refitted for every fold, because every fold has a different training set.

Putting the scaler **in the pipeline** makes `cross_val_score` do this automatically: it clones and fits the
whole pipeline on each training fold. Scaling the full data set *before* cross-validating breaks the rule in
two ways: the validation rows' values leak into the means and standard deviations the training rows are scaled
with (a small data leak that makes the estimate optimistic), and the step is no longer part of the model, so it
is easy to forget — or to apply with the wrong statistics — when predicting new houses. The pipeline also
guarantees that `predict` on new data uses the *training* scaling.

For linear regression the leak cannot move the number at all (part 11: predictions do not depend on the
scaling); for KNN, where distances depend on the scale, it is real but tiny here — a few tens of dollars with
2,930 houses, because each fold's mean and standard deviation barely differ from the full data's. The check
below does both versions for a 10-nearest-neighbor model. The principle, not the size, is the lesson.
"""),
    code("""
from sklearn.model_selection import KFold

knn10 = KNeighborsRegressor(n_neighbors=10)
cv = KFold(n_splits=10, shuffle=True, random_state=5544)

# WRONG: scale everything first, then cross-validate (validation rows shaped the scaler)
X4_scaled = pd.DataFrame(StandardScaler().fit_transform(X4), columns=X4.columns, index=X4.index)
leaky = cv_rmse(knn10, X4_scaled, cv=cv)

# RIGHT: the scaler inside the pipeline is refitted on the training folds of every split
clean = cv_rmse(make_pipeline(StandardScaler(), knn10), X4, cv=cv)

print(f"scale first, then cross-validate: {leaky:,.0f}")
print(f"scaler inside the pipeline:       {clean:,.0f}")
"""),
])

PA92_ANSWERS = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11_error, A11_standardize, A12]

# ---- The competition --------------------------------------------------------------------------------------

PA92_COMPETITION = [
    md("""
### Solution — the competition

**Approach.** Judge every candidate the same way: 10-fold cross-validation with **the same shuffled folds**
(`KFold(10, shuffle=True, random_state=5544)`) and RMSE in dollars, through one helper that takes a pipeline
and a list of columns. Preprocessing lives in a `ColumnTransformer`: numeric columns go through a median
imputer (two columns have a single missing value), optional `PolynomialFeatures`, and a scaler when the model
needs one; categorical columns go through `OneHotEncoder(handle_unknown="ignore")` (part 11). Then explore,
try families — linear with more features, with categoricals, with squares and interactions; KNN over k; trees
over depth — and keep a leaderboard. Nothing is tuned automatically; that is next week.
"""),
    code("""
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

folds = KFold(n_splits=10, shuffle=True, random_state=5544)
leaderboard = {}

def judge(name, model, columns):
    \"\"\"Cross-validated RMSE (same 10 folds for every candidate), recorded on the leaderboard.\"\"\"
    rmse = -cross_val_score(model, df_ames[columns], y_train, cv=folds, scoring="neg_root_mean_squared_error").mean()
    leaderboard[name] = rmse
    print(f"{name:48s} {rmse:>9,.0f}")
    return rmse

def preprocessing(numeric, categorical=(), scale=True, degree=None):
    \"\"\"Impute (median) -> optional polynomial terms -> optional scaling for numerics; one-hot for categoricals.\"\"\"
    steps = [("impute", SimpleImputer(strategy="median"))]
    if degree:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    if scale:
        steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(steps), list(numeric))]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)))
    return ColumnTransformer(transformers)
"""),
    md("""
**Explore first.** Which numeric columns move with price, and which categorical columns separate it?
"""),
    code("""
numeric_candidates = ["Overall Qual", "Gr Liv Area", "Garage Cars", "Garage Area", "Total Bsmt SF", "1st Flr SF",
                      "Year Built", "Full Bath", "Year Remod/Add", "TotRms AbvGrd", "Fireplaces", "Half Bath",
                      "Lot Area", "Bedroom AbvGr", "Overall Cond", "Yr Sold"]
corr = df_ames[numeric_candidates + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
print("missing values:", df_ames[numeric_candidates].isna().sum()[lambda s: s > 0].to_dict())
corr.round(2).to_frame("correlation with SalePrice")
"""),
    code("""
(ggplot(df_ames, aes(x="Gr Liv Area", y="SalePrice", color="Overall Qual"))
 + geom_point(alpha=0.4, size=1)
 + labs(title="Price rises with living area, and faster for high-quality houses", color="Overall\\nQual"))
"""),
    code("""
order = df_ames.groupby("Neighborhood")["SalePrice"].median().sort_values().index
(ggplot(df_ames, aes(x="Neighborhood", y="SalePrice"))
 + geom_boxplot(outlier_size=0.5)
 + scale_x_discrete(limits=list(order))
 + theme(axis_text_x=element_text(rotation=90))
 + labs(title="Neighborhood medians span $90,000 to $300,000+"))
"""),
    code("""
df_ames.groupby("Kitchen Qual")["SalePrice"].agg(["median", "count"]).round(0)
"""),
    md("""
Three things stand out: **Overall Qual** is the single strongest numeric predictor (r ≈ 0.80, stronger than
living area); the price-vs-area cloud fans out, with the slope steepening for high-quality houses (an
**interaction** or a **square** will help a linear model); and the categorical columns **Neighborhood** and
**Kitchen Qual** / **Exter Qual** separate prices sharply. Also, `Overall Qual` and `Kitchen Qual` are quality
grades — the first is already a 1–10 number, the second is text and needs dummies.

**Linear models**, from the baseline up.
"""),
    code("""
base4   = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath"]
twelve  = ["Gr Liv Area", "Overall Qual", "Year Built", "Year Remod/Add", "Total Bsmt SF", "Garage Cars",
           "Lot Area", "Fireplaces", "Bedroom AbvGr", "Full Bath", "Half Bath", "TotRms AbvGrd"]
cats4   = ["Neighborhood", "Kitchen Qual", "Exter Qual", "Bldg Type"]
cats6   = cats4 + ["Sale Condition", "Central Air"]

lm = LinearRegression
judge("linear: base 4 numeric",                       lm(), base4)
judge("linear: base 4 + Neighborhood",                Pipeline([("prep", preprocessing(base4, ["Neighborhood"])), ("model", lm())]), base4 + ["Neighborhood"])
judge("linear: 12 numeric",                           Pipeline([("prep", preprocessing(twelve)), ("model", lm())]), twelve)
judge("linear: 12 numeric + Neighborhood",            Pipeline([("prep", preprocessing(twelve, ["Neighborhood"])), ("model", lm())]), twelve + ["Neighborhood"])
judge("linear: 12 numeric + 4 categoricals",          Pipeline([("prep", preprocessing(twelve, cats4)), ("model", lm())]), twelve + cats4)
judge("linear: 12 numeric + 6 categoricals",          Pipeline([("prep", preprocessing(twelve, cats6)), ("model", lm())]), twelve + cats6);
"""),
    md("""
**Polynomial features and an interaction.** `PolynomialFeatures(degree=2)` on the twelve numeric columns adds
every square and every pairwise product (90 columns), including the area × quality interaction the plot asked
for. Degree 3 is included on purpose: watch it collapse.
"""),
    code("""
df_ames["Area x Qual"] = df_ames["Gr Liv Area"] * df_ames["Overall Qual"]

judge("linear: 12 numeric + Area x Qual + 4 categoricals", Pipeline([("prep", preprocessing(twelve + ["Area x Qual"], cats4)), ("model", lm())]), twelve + ["Area x Qual"] + cats4)
judge("linear: degree-2 poly (12 numeric) + 4 categoricals", Pipeline([("prep", preprocessing(twelve, cats4, degree=2)), ("model", lm())]), twelve + cats4)
judge("linear: degree-2 poly (12 numeric) + 6 categoricals", Pipeline([("prep", preprocessing(twelve, cats6, degree=2)), ("model", lm())]), twelve + cats6)
judge("linear: degree-3 poly (12 numeric) + 4 categoricals", Pipeline([("prep", preprocessing(twelve, cats4, degree=3)), ("model", lm())]), twelve + cats4);
"""),
    md("""
**K-nearest neighbors**, scaled numerics plus the neighborhood dummies, over a range of k.
"""),
    code("""
for k in [3, 5, 8, 10, 15, 25]:
    judge(f"KNN: 12 numeric (scaled) + Neighborhood, k = {k}",
          Pipeline([("prep", preprocessing(twelve, ["Neighborhood"])), ("model", KNeighborsRegressor(n_neighbors=k))]),
          twelve + ["Neighborhood"])
judge("KNN: base 4 numeric (scaled), k = 10", Pipeline([("prep", preprocessing(base4)), ("model", KNeighborsRegressor(n_neighbors=10))]), base4);
"""),
    md("""
**Decision trees** over depth. No scaling (thresholds are on one feature at a time); `random_state` fixes the
tie-breaking so the numbers repeat.
"""),
    code("""
for depth in [3, 5, 7, 9, 12, None]:
    judge(f"tree: 12 numeric + 4 categoricals, max_depth = {depth}",
          Pipeline([("prep", preprocessing(twelve, cats4, scale=False)),
                    ("model", DecisionTreeRegressor(max_depth=depth, random_state=5544))]),
          twelve + cats4)
"""),
    md("""
**The leaderboard.**
"""),
    code("""
board = pd.Series(leaderboard, name="estimated test RMSE").sort_values().round(0).to_frame()
board["vs. baseline"] = (board["estimated test RMSE"] / board.loc["linear: base 4 numeric", "estimated test RMSE"]).round(2)
board
"""),
    code("""
best_name = board.index[0]
best_model = Pipeline([("prep", preprocessing(twelve, cats6, degree=2)), ("model", LinearRegression())])
best_scores = -cross_val_score(best_model, df_ames[twelve + cats6], y_train, cv=folds, scoring="neg_root_mean_squared_error")
print(best_name)
print("per-fold RMSE:", best_scores.round(0), "  mean:", round(best_scores.mean()), "  sd:", round(best_scores.std()))
"""),
    md("""
**What the code does.** `judge` cross-validates one candidate on the shared folds and stores the mean RMSE;
`preprocessing` builds the column transformer each pipeline needs; the loops are plain trial and error over
feature sets, k and depth; the leaderboard sorts the results.

**Expected output (same folds, `random_state=5544`).**

| family | best setting tried | estimated test RMSE |
|---|---|---|
| linear, 4 features (part 1) | — | ≈ 51,400 |
| linear, 4 features + Neighborhood | — | ≈ 39,600 |
| linear, 12 numeric + 6 categoricals | — | ≈ 29,600 |
| **linear, degree-2 polynomial on 12 numeric + 6 categoricals** | | **≈ 26,200** |
| linear, degree-3 polynomial | | ≈ 52,700 (!) |
| KNN, 12 numeric scaled + Neighborhood | k = 5 | ≈ 28,900 (k = 3 … 8 all within 100) |
| tree, 12 numeric + 4 categoricals | max_depth = 9 | ≈ 32,600 |

Reading it: **features mattered more than the model family.** Going from four columns to twelve took the
linear model from 51,400 to 35,300; the neighborhood and quality categoricals took it to 29,600; the squares and
products of the numeric columns (the steepening in the area plot) to about **26,200**, the winner and roughly a
third of the baseline's error. The degree-3 model has about 450 numeric columns and overfits spectacularly: a
textbook case of training error going down while test error explodes. KNN plateaus near 29,000 whatever k;
trees bottom out around depth 9 (32,600) and get worse both shallower and deeper, the U from Topic 9.2. The
winner's per-fold RMSEs run from about 20,000 to 34,000: the estimate is ±4,000 or so, so candidates within a
couple of thousand of each other are not really distinguishable on this evidence.

What students will typically find: adding `Overall Qual` is the single biggest win; adding `Neighborhood` the
second; polynomial terms help the linear model but only on the numeric columns (one-hot columns squared are
themselves); KNN needs scaling to be competitive at all (base-4 unscaled or wrong-scaled models sit near 47,000);
trees need depth 7–10.

**Common mistakes.**
- Comparing on **training** RMSE, which hands the win to the deepest tree or k = 1.
- Using **different folds** for different candidates (`cv=10` for one, a shuffled `KFold` for another): the
  comparison then mixes fold luck with model quality. One `folds` object, passed everywhere.
- **Scaling outside the pipeline**, or not at all for KNN.
- Leaving `handle_unknown="ignore"` off the encoder, then hitting `nan` scores or an error on a fold with a
  rare neighborhood (part 11).
- Columns with **missing values** (`Total Bsmt SF`, `Garage Cars`, `Garage Area`, `Mas Vnr Area`, many in
  `Lot Frontage`) crash `LinearRegression`; the imputer handles them, or choose complete columns.
- Applying `PolynomialFeatures` to the one-hot columns as well (hundreds of useless products), or going to
  degree 3 and reporting the lowest *training* RMSE ever seen.
- Modelling `log(SalePrice)` and reporting the RMSE in log units: the task fixes `SalePrice` in dollars.
"""),
]

PA92_DISCUSS_RE = re.compile(r"^\s*\d+\.\s")


def build_solution_pa92(src_path: Path) -> dict:
    nb = json.loads(src_path.read_text())
    out: list[dict] = []
    n_q = 0
    pending: list[dict] | None = None      # an answer waiting for the next code cell
    for cell in nb["cells"]:
        src = source(cell)
        cell = copy.deepcopy(cell)
        if cell["cell_type"] == "code" and "cross_val_score(neighborhood_model" in src:
            # the cell the activity expects to fail: let nbconvert run past it and keep its output
            cell.setdefault("metadata", {})["tags"] = ["raises-exception"]
        if cell["cell_type"] == "code" and src.strip() == "" and n_q == len(PA92_ANSWERS):
            out.extend(copy.deepcopy(PA92_COMPETITION))          # the empty competition cell
            continue
        out.append(cell)
        if cell["cell_type"] == "code" and pending is not None:
            out.extend(pending); pending = None
        if cell["cell_type"] == "markdown" and PA92_DISCUSS_RE.match(src):
            if n_q >= len(PA92_ANSWERS):
                raise SystemExit(f"{src_path.name}: more numbered questions than answers (at #{n_q})")
            after_code, cells = PA92_ANSWERS[n_q]
            if after_code:
                pending = copy.deepcopy(cells)
            else:
                out.extend(copy.deepcopy(cells))
            n_q += 1
    if n_q != len(PA92_ANSWERS) or pending is not None:
        raise SystemExit(f"{src_path.name}: matched {n_q} questions, expected {len(PA92_ANSWERS)}; pending={pending is not None}")
    first_md = next(i for i, c in enumerate(out) if c["cell_type"] == "markdown")
    out[first_md]["source"] = re.sub(r"^#+[^\n]*", lambda m: PA92_TITLE, source(out[first_md]), count=1)
    out.insert(first_md + 1, copy.deepcopy(PA92_INTRO))
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:03d}"
        c["source"] = source(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = out
    nb["metadata"] = copy.deepcopy(METADATA)
    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    return nb


# ============================================================================ #
#  Build
# ============================================================================ #

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
DISCUSS_MARKERS = {"**Sketch on paper and discuss with your partner.**", "**Discuss with your partner.**"}


def source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def build_solution(src_path: Path) -> dict:
    nb = json.loads(src_path.read_text())
    out: list[dict] = []
    n_code = n_md = 0
    for cell in nb["cells"]:
        src = source(cell)
        if cell["cell_type"] == "code" and src.strip().startswith("# YOUR CODE HERE"):
            if n_code >= len(PA91_CODE_ANSWERS):
                raise SystemExit(f"{src_path.name}: more '# YOUR CODE HERE' cells than answer groups (at #{n_code})")
            out.extend(copy.deepcopy(PA91_CODE_ANSWERS[n_code]))
            n_code += 1
        elif cell["cell_type"] == "markdown" and src.strip() in DISCUSS_MARKERS:
            if n_md >= len(PA91_DISCUSS):
                raise SystemExit(f"{src_path.name}: more discuss cells than written answers (at #{n_md})")
            out.extend(copy.deepcopy(PA91_DISCUSS[n_md]))
            n_md += 1
        else:
            out.append(copy.deepcopy(cell))
            if cell["cell_type"] == "code" and "ks = [1, 2, 10, 27]" in src:
                out.append(copy.deepcopy(S4))
    if (n_code, n_md) != (len(PA91_CODE_ANSWERS), len(PA91_DISCUSS)):
        raise SystemExit(f"{src_path.name}: found {n_code} code / {n_md} discuss cells, "
                         f"expected {len(PA91_CODE_ANSWERS)} / {len(PA91_DISCUSS)}")
    first_md = next(i for i, c in enumerate(out) if c["cell_type"] == "markdown")
    out[first_md]["source"] = re.sub(r"^#+[^\n]*", lambda m: PA91_TITLE, source(out[first_md]), count=1)
    out.insert(first_md + 1, copy.deepcopy(PA91_INTRO))
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:03d}"
        c["source"] = source(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = out
    nb["metadata"] = copy.deepcopy(METADATA)
    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    return nb


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=900", str(path)], check=True)
    nb = json.loads(path.read_text())
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              and "raises-exception" not in c.get("metadata", {}).get("tags", [])
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"unexpected errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    which = sys.argv[sys.argv.index("--pa") + 1] if "--pa" in sys.argv else "both"
    jobs = []
    if which in ("9.1", "both"):
        jobs.append((PA91_SRC, build_solution))
    if which in ("9.2", "both"):
        jobs.append((PA92_SRC, build_solution_pa92))
    for src, builder in jobs:
        dest = src.with_name(src.stem + "-solution.ipynb")
        save(dest, builder(src))
        print(f"wrote {dest.relative_to(ROOT)}")
        if "--no-exec" not in sys.argv:
            execute(dest)


if __name__ == "__main__":
    main()
