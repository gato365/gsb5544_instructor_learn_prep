#!/usr/bin/env python3
"""Build the Week 9 practice-activity INSTRUCTOR SOLUTION notebook (PA 9.1 — K-Nearest Neighbors and Decision Trees for Regression).

Usage:  python3 tools/build_week9_pas.py            # build + execute the -solution file
        python3 tools/build_week9_pas.py --no-exec  # build without executing

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
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"unexpected errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    dest = PA91_SRC.with_name(PA91_SRC.stem + "-solution.ipynb")
    save(dest, build_solution(PA91_SRC))
    print(f"wrote {dest.relative_to(ROOT)}")
    if "--no-exec" not in sys.argv:
        execute(dest)


if __name__ == "__main__":
    main()
