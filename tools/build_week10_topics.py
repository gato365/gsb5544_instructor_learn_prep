#!/usr/bin/env python3
"""Generate the Week 10 Topic notebook (student + executed solution).

Usage:  python3 tools/build_week10_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week10_topics.py --no-exec  # write both notebooks without executing

Topic 10.1 — Model Selection and Hyperparameter Tuning with GridSearchCV
  A deliberately straightforward walk-through, built to scaffold PA 10.1 (same Ames data, same tools) without
  doing it: one rule (compare candidates on cross-validated test RMSE, same folds for everyone) and one tool
  (GridSearchCV). Order: the rule and two helpers; model selection as a loop over feature sets; tuning k by hand
  (last week's loop); GridSearchCV — param_grid, the step__parameter naming convention, best_params_,
  best_score_ (negative), cv_results_ as a DataFrame and a plot; two and three parameters at once (k, metric,
  weights) and how many fits that is; comparing model families on the same folds; using best_estimator_ to
  predict; review questions.
  Needs scikit-learn (>= 1.4), pandas and matplotlib in the kernel; one cell reads the Ames data from the web.
Same markers as the earlier builders:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_10"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"
AMES_URL = "https://raw.githubusercontent.com/kevindavisross/data301/main/data/AmesHousing.txt"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 10.1 — Model selection and hyperparameter tuning
# ============================================================================ #

t: list[dict] = []

md(t, """
# GSB 5544 — Topic 10.1: Model Selection and Hyperparameter Tuning — SOLUTION
*Which features? Which k? Which model? One rule — compare candidates on cross-validated test RMSE, with the same
folds for everyone — and one tool that applies it for you: `GridSearchCV`.*
""")

md(t, """
## The plan

Last week: training error flatters, so we estimate the **test error** with K-fold cross-validation. This week
we *use* that estimate to make every choice a model needs, and we let scikit-learn run the loops.

| | | tools |
|---|---|---|
| **1. The rule, and two helpers** | one scorer, one preprocessing builder, used for every candidate | `KFold`, `cross_val_score`, `ColumnTransformer` |
| **2. Model selection** | which *features*? a loop over candidate feature sets | a dictionary of lists |
| **3. Tuning k by hand** | which *k*? last week's loop | `KNeighborsRegressor(n_neighbors=k)` |
| **4. `GridSearchCV`** | the loop, packaged: grid → fit → best | `param_grid`, `best_params_`, `best_score_`, `cv_results_` |
| **5. Several parameters at once** | k × metric × weights; how many fits is that? | a grid with several keys |
| **6. Comparing model families** | KNN vs tree vs line, same folds | one leaderboard |
| **7. Use the winner** | `best_estimator_` is already refitted | `.predict()` |
| **8. Review questions** | | |

**Data:** the Ames houses again (2,930 sales), predicting `SalePrice` in dollars. PA 10.1 uses the same data,
the same tools and the same conventions, with different feature sets — so everything below is a template you
can reuse directly.
""")

code(t, """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
""")

# --------------------------------------------------------------------------- 1
md(t, """
---
## 1. The rule, and two helpers

**The rule.** Every decision — which features, which k, which distance, which model — is made by comparing the
candidates' **cross-validated test RMSE**, computed on the **same folds**. Training RMSE is never consulted: it
always votes for the most flexible candidate (Topic 9.2).

Two helpers keep every comparison identical:

- `folds`: one `KFold` object, created once and passed everywhere. Shuffled, with a `random_state`, so the
  folds are random but repeatable.
- `preprocessing(numeric, categorical)`: builds the `ColumnTransformer` a pipeline needs — a median imputer
  (two Ames columns have one missing value), a scaler for the numeric columns (KNN measures distance), and a
  one-hot encoder with `handle_unknown="ignore"` for the categorical ones (so a rare neighborhood in a
  validation fold does not crash the fold; Topic 9.2 / PA 9.2).
""")

code(t, """
df_ames = pd.read_csv("%s", sep="\\t")
y = df_ames["SalePrice"]

folds = KFold(n_splits=«5», shuffle=True, random_state=5544)     # the same 5 folds for every candidate

def preprocessing(numeric, categorical=(), scale=True):
    \"\"\"Impute + (optionally) scale the numeric columns; one-hot encode the categorical ones.\"\"\"
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", «StandardScaler»()))
    transformers = [("num", Pipeline(numeric_steps), list(numeric))]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown=«"ignore"»), list(categorical)))
    return ColumnTransformer(transformers)

def cv_rmse(model, columns):
    \"\"\"Cross-validated test RMSE (dollars) of a model on the given columns, on the shared folds.\"\"\"
    scores = cross_val_score(model, df_ames[columns], y, cv=folds, scoring=«"neg_root_mean_squared_error"»)
    return -scores.mean()

print(df_ames.shape)
""" % AMES_URL)

md(t, """
✅ **Checkpoint 1.** (a) Why create `folds` once instead of writing `cv=5` in every call? (b) Why does the
scorer's name start with `neg_`, and what does `cv_rmse` do about it?
""")

answer(t, """
(a) `cv=5` means `KFold(5)` *without* shuffling: consecutive rows, which is a trap when rows are ordered, and
no `random_state`, so there is nothing to make two runs comparable. One shared `folds` object guarantees that
every candidate is scored on exactly the same five validation sets, so differences between candidates are
differences between models, not between folds. (b) scikit-learn treats every score as "higher is better", so an
error must be returned as its negative. `cv_rmse` negates the mean, giving an RMSE in dollars.
""")

# --------------------------------------------------------------------------- 2
md(t, """
---
## 2. Model selection: a loop over feature sets

"Model selection" here means choosing **which columns** the model gets. Put the candidate feature sets in a
dictionary (name → list of columns), loop, score each with the *same* model and the *same* folds, and read off
the smallest RMSE. Note the model is a 10-nearest-neighbor pipeline every time; only the columns change.
""")

code(t, """
feature_sets = {
    "size only":                             (["Gr Liv Area"], []),
    "size + quality":                        (["Gr Liv Area", "Overall Qual"], []),
    "size + quality + year":                 (["Gr Liv Area", "Overall Qual", "Year Built"], []),
    "size + quality + year + Neighborhood":  (["Gr Liv Area", "Overall Qual", "Year Built"], ["Neighborhood"]),
}

results = {}
for name, (numeric, categorical) in feature_sets.items():
    pipe = Pipeline([("prep", preprocessing(numeric, categorical)),
                     ("knn", KNeighborsRegressor(n_neighbors=«10»))])
    results[name] = cv_rmse(pipe, numeric + categorical)

pd.Series(results, name="CV test RMSE (10-NN)").round(0).to_frame()
""")

md(t, """
Each added feature helps, and quality helps the most: knowing `Overall Qual` cuts the error from about 57,900
to 36,500. The last set, **size + quality + year + Neighborhood**, wins at about **33,000** and is the feature
set used for the rest of this notebook.

✅ **Checkpoint 2.** (a) What is the only thing that changes from one loop iteration to the next? (b) Suppose
"size only" had the lowest *training* RMSE. Would that change the decision?
""")

answer(t, """
(a) The columns (and therefore the preprocessing built for them). The model (`KNeighborsRegressor(10)`), the
folds and the scorer are identical, which is what makes the four numbers comparable. (b) No. Training RMSE is
not part of the rule; only the cross-validated test RMSE decides. (It would not happen anyway — more features
can only lower a KNN model's training error on these rows — but the point stands.)
""")

# --------------------------------------------------------------------------- 3
md(t, """
---
## 3. Tuning k by hand: last week's loop

A **hyperparameter** is a setting you choose before fitting (k, `max_depth`, the distance metric). Choosing
it is the same game as choosing features: try candidates, score each on the shared folds, take the smallest.
Topic 9.2 did this with a loop. Here it is once more, so that `GridSearchCV` in the next section is recognisably
the same thing.
""")

code(t, """
numeric, categorical = feature_sets["size + quality + year + Neighborhood"]
columns = numeric + categorical

by_hand = {}
for k in [1, 3, 5, 10, 20, 50]:
    pipe = Pipeline([("prep", preprocessing(numeric, categorical)),
                     ("knn", KNeighborsRegressor(n_neighbors=«k»))])
    by_hand[k] = cv_rmse(pipe, columns)

pd.Series(by_hand, name="CV test RMSE").round(0).to_frame()
""")

md(t, """
The usual U: k = 1 is noisy (≈ 36,700), the error bottoms out around k = 10 (≈ 33,000), and k = 50 averages
over too many unlike houses (≈ 34,300). The loop works, but it means writing the loop every time, and with two
or three hyperparameters it becomes nested loops. Hence the tool.
""")

# --------------------------------------------------------------------------- 4
md(t, """
---
## 4. `GridSearchCV`: the loop, packaged

`GridSearchCV(model, param_grid, cv, scoring)` does exactly what the loop above did, for every combination of
settings in `param_grid`, and keeps the results:

| argument / attribute | what it is |
|---|---|
| `model` | the pipeline (or model) to tune, **unfitted** |
| `param_grid` | a dictionary: parameter name → list of values to try |
| `cv`, `scoring` | the folds and the scorer, same as `cross_val_score` |
| `.fit(X, y)` | runs the whole search: for every combination, fit on K−1 folds and score on the held-out fold, K times; then **refits the best combination on all of `X`** |
| `.best_params_` | the winning combination |
| `.best_score_` | its mean cross-validated score — **negative**, because of `neg_` |
| `.cv_results_` | a dictionary with every combination's mean and standard deviation of the fold scores, ranks, timings; `pd.DataFrame(...)` makes it readable |
| `.best_estimator_` | the winning pipeline, already refitted on all the rows; use it to predict |

**The naming convention.** The parameter lives inside a pipeline step, so its name is
`<step name>__<parameter>` with a **double underscore**: our KNN step is called `"knn"`, so k is
`"knn__n_neighbors"`. (With `make_pipeline`, steps are named after their class in lower case:
`"kneighborsregressor__n_neighbors"`.) A typo here is the most common `GridSearchCV` error; the message lists
the valid names.
""")

code(t, """
pipe = Pipeline([("prep", preprocessing(numeric, categorical)),
                 ("knn", KNeighborsRegressor())])            # no k yet: the grid supplies it

param_grid = {«"knn__n_neighbors"»: range(1, 26)}

grid = GridSearchCV(pipe, param_grid, cv=«folds», scoring="neg_root_mean_squared_error")
grid.«fit»(df_ames[columns], y)

print("best parameters:", grid.best_params_)
print("best CV test RMSE:", round(-grid.«best_score_»))
""")

md(t, """
`cv_results_` holds the whole search. The columns to know: `param_knn__n_neighbors` (the setting),
`mean_test_score` (mean over folds, negative RMSE), `std_test_score` (spread over folds), `rank_test_score`.
A plot of RMSE against k is the picture that justifies the choice.
""")

code(t, """
cv_table = pd.DataFrame(grid.cv_results_)
cv_table["CV test RMSE"] = -cv_table[«"mean_test_score"»]
cv_table[["param_knn__n_neighbors", "CV test RMSE", "std_test_score", "rank_test_score"]].round(0).head(12)
""")

code(t, """
ax = cv_table.plot(x="param_knn__n_neighbors", y="CV test RMSE", marker="o", legend=False)
ax.axvline(grid.best_params_["knn__n_neighbors"], color="firebrick", linestyle="--")
ax.set_xlabel("k (n_neighbors)"); ax.set_ylabel("cross-validated test RMSE ($)")
ax.set_title("GridSearchCV over k");
""")

md(t, """
✅ **Checkpoint 3.** (a) The grid has 25 values of k and `folds` has 5 folds. How many models did `grid.fit`
fit, and why is it one more than 25 × 5? (b) `grid.best_score_` is about −32,900. What is the test RMSE
estimate, and in what units? (c) k = 7, 8, 9 and 10 are all within about 100 dollars of each other while the
fold-to-fold standard deviation is about 2,700. How confident should you be that k = 8 is really best?
""")

answer(t, """
(a) 25 × 5 = 125 fits for the search, plus **one more**: after choosing the best k, `GridSearchCV` refits that
pipeline on all 2,930 rows (that is `best_estimator_`). 126 in total. (b) About **32,900 dollars**: negate the
score. (c) Not very. The differences between k = 7 … 10 are tiny compared with how much the RMSE varies from
fold to fold, so any of them is a reasonable choice; the search picks the smallest *mean*, and that is fine,
but "k somewhere around 8" is the honest conclusion. The by-hand loop in section 3 chose 10 on the same folds
for the same reason: it had not tried 8.
""")

# --------------------------------------------------------------------------- 5
md(t, """
---
## 5. Several parameters at once

Add keys to `param_grid` and `GridSearchCV` tries **every combination** (the "grid"). `KNeighborsRegressor` has
more than k to tune:

- `metric`: how distance is measured — `"euclidean"` (straight line, the default) or `"manhattan"` (sum of
  absolute differences along each feature).
- `weights`: `"uniform"` averages the k neighbors equally; `"distance"` gives closer neighbors more weight.

The cost: the number of fits is (grid size) × (folds). 25 values of k × 2 metrics = 50 combinations × 5 folds
= 250 fits; add 2 weightings and it is 500. Still seconds here, but grids grow fast.
""")

code(t, """
param_grid_2 = {
    "knn__n_neighbors": range(1, 26),
    «"knn__metric"»: ["euclidean", "manhattan"],
}
grid2 = GridSearchCV(pipe, param_grid_2, cv=folds, scoring="neg_root_mean_squared_error")
grid2.fit(df_ames[columns], y)

print("combinations tried:", len(grid2.cv_results_["params"]), "   fits:", len(grid2.cv_results_["params"]) * folds.get_n_splits())
print("best:", grid2.best_params_, "  CV test RMSE:", round(-grid2.best_score_))
""")

md(t, """
With two parameters the natural picture is a table or two curves: `pivot` the results so that rows are k and
columns are the metric.
""")

code(t, """
table2 = pd.DataFrame(grid2.cv_results_)
table2["CV test RMSE"] = -table2["mean_test_score"]
wide = table2.pivot(index="param_knn__n_neighbors", columns="param_knn__metric", values="CV test RMSE")
ax = wide.plot(marker="o"); ax.set_xlabel("k"); ax.set_ylabel("cross-validated test RMSE ($)")
wide.round(0).head(12)
""")

code(t, """
param_grid_3 = {
    "knn__n_neighbors": range(1, 26),
    "knn__metric": ["euclidean", "manhattan"],
    «"knn__weights"»: ["uniform", "distance"],
}
grid3 = GridSearchCV(pipe, param_grid_3, cv=folds, scoring="neg_root_mean_squared_error")
grid3.fit(df_ames[columns], y)
print("fits:", len(grid3.cv_results_["params"]) * folds.get_n_splits())
print("best:", grid3.best_params_, "  CV test RMSE:", round(-grid3.best_score_))
""")

md(t, """
✅ **Checkpoint 4.** (a) Manhattan distance with k = 6 beat Euclidean with k = 8 by about 70 dollars. Is the
metric worth changing? (b) `weights="distance"` brought the RMSE down to about 31,000. In one sentence, why
might weighting by distance help KNN on house prices? (c) You want to also try `p` in {1, 2, 3} with
`metric="minkowski"`. How many fits would a grid over k (25), weights (2) and p (3) need with 5 folds?
""")

answer(t, """
(a) Not on this evidence: 70 dollars is noise next to a fold-to-fold spread of thousands. Either metric is
fine; report that they tie. (b) With k = 6 the sixth-nearest house may be noticeably less similar than the
first; weighting by inverse distance lets the near-identical houses dominate the average instead of counting
equally with the stragglers. (c) 25 × 2 × 3 = 150 combinations × 5 folds = **750 fits** (plus the final
refit). Grids multiply; tune the parameters that matter.
""")

# --------------------------------------------------------------------------- 6
md(t, """
---
## 6. Comparing model families on the same folds

Hyperparameter tuning picks the best *version* of a model. Model selection in the wide sense also compares
*kinds* of model. Same rule: every family gets its best settings by grid search (or needs none), and the
families are compared by cross-validated test RMSE on the same folds. The tree's grid is over `max_depth`; the
tree needs no scaling, so `preprocessing(..., scale=False)` keeps the thresholds in real units.
""")

code(t, """
tree_pipe = Pipeline([("prep", preprocessing(numeric, categorical, scale=False)),
                      ("tree", DecisionTreeRegressor(random_state=5544))])
grid_tree = GridSearchCV(tree_pipe, {«"tree__max_depth"»: [2, 4, 6, 8, 10, 12, None]},
                         cv=folds, scoring="neg_root_mean_squared_error")
grid_tree.fit(df_ames[columns], y)

linear_pipe = Pipeline([("prep", preprocessing(numeric, categorical)), ("lm", LinearRegression())])

leaderboard = pd.Series({
    f"KNN {grid3.best_params_}":            -grid3.best_score_,
    f"tree {grid_tree.best_params_}":       -grid_tree.best_score_,
    "linear regression (nothing to tune)":  cv_rmse(linear_pipe, columns),
}, name="CV test RMSE").sort_values().round(0).to_frame()
leaderboard
""")

md(t, """
✅ **Checkpoint 5.** (a) Why is it fair to compare `grid3.best_score_` with `cv_rmse(linear_pipe, columns)`?
(b) The tuned tree and the line are within a few hundred dollars of each other at about 35,500. Which would
you choose, and on what grounds other than RMSE?
""")

answer(t, """
(a) Both numbers are mean validation RMSEs over the *same* five folds, so each candidate was scored on the
same held-out houses. (Strictly, the KNN number is a little optimistic because it is the best of 100
combinations and some of that is luck; with a sealed test set you would confirm it there.) (b) A tie on error
is decided by other things: the linear model is simpler, faster and gives interpretable coefficients; the tree
is easier to explain to a non-technical reader and handles interactions without being told. Either is
defensible; what is not defensible is picking by training error.
""")

# --------------------------------------------------------------------------- 7
md(t, """
---
## 7. Use the winner

`best_estimator_` is the winning pipeline **already refitted on all 2,930 houses** — exactly the "refit the
chosen model on all the data" step from Topic 9.2, done for you. Predict with it directly.
""")

code(t, """
new_house = pd.DataFrame({"Gr Liv Area": [1500], "Overall Qual": [7], "Year Built": [2005], "Neighborhood": ["CollgCr"]})
best_knn = grid3.«best_estimator_»
print("predicted SalePrice:", best_knn.predict(new_house).round(0))
best_knn
""")

md(t, """
✅ **Checkpoint 6.** `grid3.predict(new_house)` gives the same number as `best_knn.predict(new_house)`. Why,
and which rows was that model fitted on?
""")

answer(t, """
A fitted `GridSearchCV` forwards `predict` to its `best_estimator_`, so they are the same model. It was fitted
on **all** the rows passed to `grid3.fit` — the five fold-models used during the search were only for scoring
and are discarded.
""")

# --------------------------------------------------------------------------- 8
md(t, """
---
## 8. Review questions

Short answers; a sentence or a number each.
""")

md(t, """
**Q1.** State the one rule behind every choice in this notebook, and name the quantity that is never used.
""")
answer(t, """
Compare candidates by cross-validated **test** RMSE on the same folds and pick the smallest. **Training** RMSE
is never used — it always favours the most flexible candidate.
""")

md(t, """
**Q2.** A pipeline has steps `("prep", ...)` and `("model", KNeighborsRegressor())`. Write the `param_grid`
that tries k = 5, 10, 15 and both weightings. How many fits with `KFold(10)`?
""")
answer(t, """
`{"model__n_neighbors": [5, 10, 15], "model__weights": ["uniform", "distance"]}` — 6 combinations × 10 folds =
60 fits, plus one refit of the best.
""")

md(t, """
**Q3.** `grid.best_score_` prints `-41237.5`. What does it mean?
""")
answer(t, """
The best combination's mean validation RMSE over the folds is about **41,238 dollars**; the sign is scikit-learn's
"higher is better" convention for `neg_root_mean_squared_error`.
""")

md(t, """
**Q4.** What is the difference between `grid.cv_results_` and `grid.best_estimator_`?
""")
answer(t, """
`cv_results_` is the record of the search: every combination with its mean and standard deviation of fold
scores and its rank (make it a DataFrame to read or plot it). `best_estimator_` is one fitted pipeline: the
winning combination refitted on all the training rows, ready to `predict`.
""")

md(t, """
**Q5.** Two values of k differ by 50 dollars in mean CV RMSE; the standard deviation across folds is 3,000.
What do you report?
""")
answer(t, """
That they are indistinguishable on this evidence: pick either (the search picks the smaller mean), and say so.
Differences far smaller than the fold-to-fold spread are not real.
""")

md(t, """
---
## The lines to keep

| | |
|---|---|
| **Shared folds** | `folds = KFold(n_splits=5, shuffle=True, random_state=5544)` — pass the same object to every `cross_val_score` and `GridSearchCV` |
| **Feature selection** | a dict `name → columns`; loop; `cv_rmse(pipeline, columns)`; smallest wins |
| **Grid search** | `GridSearchCV(pipe, {"knn__n_neighbors": range(1, 26)}, cv=folds, scoring="neg_root_mean_squared_error").fit(X, y)` |
| **Names** | `"<step>__<parameter>"` — `knn__n_neighbors`, `knn__metric`, `knn__weights`, `tree__max_depth` |
| **Results** | `best_params_`; `-best_score_` (RMSE); `pd.DataFrame(cv_results_)` → `mean_test_score`, `std_test_score`, `rank_test_score`; plot RMSE vs the parameter |
| **Cost** | fits = (number of combinations) × (folds) + 1 |
| **Use it** | `best_estimator_` is refitted on all rows: `grid.predict(new)` |

Course site: [%s](%s).
""" % (SITE, SITE))


# ============================================================================ #
#  Build
# ============================================================================ #

BLANK = re.compile(r"«(.*?)»", re.S)
METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def notebook(cell_list: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cell_list):
        c = copy.deepcopy(c)
        c["id"] = f"cell-{i:03d}"
        out.append(c)
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def solution_cells(cells: list[dict]) -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub(lambda m: m.group(1), c["source"])
        out.append(c)
    return out


def student_cells(cells: list[dict], student_title: str) -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub("____", c["source"])
        elif c["source"].startswith("**Answer:**"):
            c["source"] = "**Your answer:** *(write it here — replace this line)*"
        out.append(c)
    out[0]["source"] = student_title
    return out


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


TOPICS = [
    (t, WEEK / "GSB5544_Topic_10_1_Model_Selection_and_Hyperparameter_Tuning",
     "# GSB 5544 — Topic 10.1: Model Selection and Hyperparameter Tuning  \n"
     "*Which features? Which k? Which model? One rule — compare candidates on cross-validated test RMSE, with the "
     "same folds for everyone — and one tool that applies it for you: `GridSearchCV`. Fill each `____` blank as you "
     "work; the ✅ checks and the review questions ask for a sentence or a number.*"),
]


def main() -> None:
    WEEK.mkdir(parents=True, exist_ok=True)
    for cells, stem, student_title in TOPICS:
        student = stem.with_name(stem.name + "-empty.ipynb")
        solution = stem.with_name(stem.name + "-solution.ipynb")
        save(student, notebook(student_cells(cells, student_title)))
        save(solution, notebook(solution_cells(cells)))
        print(f"wrote {student.relative_to(ROOT)} and {solution.relative_to(ROOT)} ({len(cells)} cells)")
        if "--no-exec" not in sys.argv:
            execute(solution)


if __name__ == "__main__":
    main()
