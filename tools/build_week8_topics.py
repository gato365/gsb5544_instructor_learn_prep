#!/usr/bin/env python3
"""Generate the Week 8 Topic notebook (student + executed solution).

Usage:  python3 tools/build_week8_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week8_topics.py --no-exec  # write both notebooks without executing

Topic 8.2 — Regression in Python, and the first ideas of machine learning  (scikit-learn + plotnine)
  Students already know regression (assumptions, coefficients, polynomials, interactions) from other courses, so
  nothing is re-taught. The notebook (1) shows the scikit-learn recipe: X as a DataFrame, fit, coef_, predict,
  score; more features / interactions / polynomial terms as more columns; (2) uses the linear model to introduce
  ML vocabulary: features and target, parameters vs hyperparameters, train/test split, RMSE, overfitting via a
  polynomial-degree loop in a scaled pipeline.
  Data: the Advertising data set (200 markets; TV, radio and newspaper budgets in $1,000s; sales in 1,000s of
  units), read from its home at statlearning.com. A copy sits in assignments/Data/advertising.csv.
  Needs scikit-learn and plotnine in the kernel.
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
WEEK = ROOT / "assignments" / "practice_activities" / "week_8"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"
DATA_URL = "https://www.statlearning.com/s/Advertising.csv"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str, raises: bool = False) -> None:
    meta = {"tags": ["raises-exception"]} if raises else {}
    cells.append({"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 8.2 — Regression in Python, and the first ideas of machine learning
# ============================================================================ #

t82: list[dict] = []

md(t82, """
# GSB 5544 — Topic 8.2: Regression in Python — SOLUTION
*You already know regression. This is how to run it in Python, and how it introduces machine learning.*
""")

md(t82, """
## The next 20 minutes

You have fit linear models before: assumptions, reading coefficients, polynomial terms, interactions.
None of that is re-taught here. Two things are new:

1. **How to do it in Python** with scikit-learn: one recipe that every model in this course will follow.
2. **Machine-learning vocabulary**, introduced on the familiar ground of a linear model: features and
   target, fit and predict, training and test data, overfitting.

| | | tools |
|---|---|---|
| **1. Data** | 200 markets, ad budgets and sales | `read_csv`, `corr`, `geom_smooth(method="lm")` |
| **2. The scikit-learn recipe** | fit a line, read it, predict with it | `LinearRegression`, `.fit`, `.coef_`, `.predict`, `.score` |
| **3. More terms** | more features, interactions, polynomials are all just more columns | `PolynomialFeatures` |
| **4. The machine-learning turn** | how well does the model predict data it has not seen? | `train_test_split`, `root_mean_squared_error`, `make_pipeline` |
""")

code(t82, """
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth, geom_line, labs, theme_minimal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
""")

md(t82, """
---
## 1. Data

**Question.** A product is advertised in 200 markets with different **TV**, **radio** and **newspaper**
budgets (in \\$1,000s). Which budgets move **sales** (1,000s of units), and what sales should a new budget
produce?

One row per market. (The Advertising data from *An Introduction to Statistical Learning*; a copy is in the
course repo as `assignments/Data/advertising.csv`.)
""")

code(t82, """
ads = pd.read_csv("%s", index_col=0)
print(ads.shape)
ads.head()
""" % DATA_URL)

md(t82, """
The usual quick look: correlations, and the outcome against the strongest feature with a least-squares
line (`method="lm"`).
""")

code(t82, """
ads.«corr»().round(2)
""")

code(t82, """
(ggplot(ads, aes(x=«"TV"», y=«"sales"»))
 + geom_point(alpha=0.6)
 + geom_smooth(method="lm", se=False, color="firebrick")
 + labs(x="TV budget ($1,000s)", y="Sales (1,000s of units)")
 + theme_minimal())
""")

md(t82, """
---
## 2. The scikit-learn recipe

Every scikit-learn model works the same way. Learn it once:

| step | code | |
|---|---|---|
| choose | `model = LinearRegression()` | an empty model |
| fit | `model.fit(X, y)` | learn the coefficients from data |
| read | `model.intercept_`, `model.coef_` | the trailing `_` means "learned from data" |
| use | `model.predict(X_new)`, `model.score(X, y)` | predictions; $R^2$ |

**Vocabulary.** `X` is the table of **features** (explanatory variables), `y` is the **target** (response).
`X` must be **2-D** — a DataFrame, even with one column — because features are always a table with one
column per feature. `y` is a Series.
""")

code(t82, """
X = ads[[«"TV"»]]          # double brackets -> DataFrame (2-D)
y = ads[«"sales"»]         # single brackets -> Series (1-D)
print(type(X).__name__, X.shape, "|", type(y).__name__, y.shape)
""")

code(t82, """
model_tv = «LinearRegression»()
model_tv.«fit»(X, y)
print("intercept:", round(model_tv.intercept_, 4))
print("coef:     ", model_tv.coef_.round(4))
print("R²:       ", round(model_tv.score(X, y), 3))
""")

md(t82, """
The error everyone hits once — single brackets give a 1-D Series. The key words in the message are
*2-dimensional* and *DataFrame*.
""")

code(t82, """
LinearRegression().fit(ads["TV"], ads["sales"])     # 1-D -> error
""", raises=True)

md(t82, """
**Predict** takes a DataFrame with the **same column names** as `X`, one row per new case.
""")

code(t82, """
new = pd.DataFrame({"TV": [0, 100, 200, 300]})
new["pred_sales"] = model_tv.«predict»(new[["TV"]])
new
""")

md(t82, """
✅ **Checkpoint 1.** (a) Write the fitted equation and check the prediction at TV = 100 by hand.
(b) In one sentence, read the TV coefficient in the data's units.
""")

answer(t82, """
(a) sales ≈ 7.03 + 0.0475 · TV; at TV = 100: 7.03 + 4.75 = **11.79** thousand units (matches the table).
(b) Each extra \\$1,000 of TV budget is associated with about 47.5 more units sold.
""")

md(t82, """
---
## 3. More terms are just more columns

A second feature, an interaction, a squared term: in scikit-learn these are all the same move — **put
another column in `X`**. Nothing else in the recipe changes.
""")

code(t82, """
X2 = ads[[«"TV", "radio"»]]
model_2 = LinearRegression().fit(X2, y)
print(pd.Series(model_2.coef_, index=X2.columns).round(4))
print("R²:", round(model_2.score(X2, y), 3))
""")

md(t82, """
An **interaction** is a column you make yourself:
""")

code(t82, """
ads["TV_x_radio"] = ads[«"TV"»] * ads[«"radio"»]
X3 = ads[["TV", "radio", "TV_x_radio"]]
model_3 = LinearRegression().fit(X3, y)
print(pd.Series(model_3.coef_, index=X3.columns).round(5))
print("R²:", round(model_3.score(X3, y), 3))
""")

md(t82, """
**Polynomial terms** can be made the same way (`ads["TV"] ** 2`), or generated: `PolynomialFeatures(degree=2)`
builds every term up to that degree — squares *and* cross-products — from the columns you give it.
`fit_transform` returns a NumPy array; `get_feature_names_out()` tells you which column is which.
""")

code(t82, """
poly = «PolynomialFeatures»(degree=2, include_bias=False)
X_poly = poly.fit_transform(ads[["TV", "radio"]])
print(X_poly.shape)
print(poly.get_feature_names_out())
""")

code(t82, """
model_poly = LinearRegression().fit(X_poly, y)
print(pd.Series(model_poly.coef_, index=poly.get_feature_names_out()).round(5))
print("R²:", round(model_poly.score(X_poly, y), 3))
""")

md(t82, """
✅ **Checkpoint 2.** $R^2$ went 0.61 → 0.90 → 0.97 → 0.99 as terms were added. Can $R^2$ on the fitted data
ever go *down* when you add a column? What does that imply about using it to choose between models?
""")

answer(t82, """
No — a new column can be given a coefficient of zero, so the fit can only stay the same or improve. Fitted
$R^2$ therefore always favours the bigger model and cannot tell you when you have gone too far. That is the
problem section 4 solves.
""")

md(t82, """
---
## 4. The machine-learning turn

Up to now the model was judged on the **same rows it was fitted on**. Machine learning asks a different
question: *how well does the model predict rows it has never seen?* The vocabulary:

| statistics says | machine learning says |
|---|---|
| explanatory variables, response | **features** `X`, **target** `y` |
| estimate the coefficients | **fit** / **train** the model |
| coefficients | **parameters** (learned by `.fit`) |
| model specification (which terms, which degree) | **hyperparameters** (chosen by *you*, before fitting) |
| goodness of fit | **training error** vs **test error** |

### Train / test split

Hold some rows back. Fit on the **training** rows; measure error on the **test** rows the model never saw.
`random_state` fixes the shuffle so everyone gets the same split.
""")

code(t82, """
train, test = «train_test_split»(ads, test_size=0.25, random_state=5544)
print(train.shape, test.shape)
""")

md(t82, """
The error measure is **RMSE** (root mean squared error): the typical size of a miss, in the units of `y`.
""")

code(t82, """
features = ["TV", "radio"]
model = LinearRegression().fit(train[features], train[«"sales"»])

rmse_train = root_mean_squared_error(train["sales"], model.predict(train[features]))
rmse_test = root_mean_squared_error(«test»["sales"], model.predict(«test»[features]))
print(f"train RMSE = {rmse_train:.3f}   test RMSE = {rmse_test:.3f}   (thousand units)")
""")

md(t82, """
### Overfitting: turn the degree up and watch both errors

`make_pipeline` chains steps into one model object that still has `.fit` and `.predict`: scale the
features (polynomials of raw budgets in the hundreds are numerically unstable), build the polynomial
terms, fit the line. The **degree is a hyperparameter**: we loop over it and record training and test
RMSE for each.
""")

code(t82, """
rows = []
for degree in range(1, 11):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression())
    pipe.«fit»(train[features], train["sales"])
    rows.append({"degree": degree,
                 "n_terms": pipe[-1].coef_.size,
                 "train": root_mean_squared_error(train["sales"], pipe.predict(train[features])),
                 "test": root_mean_squared_error(test["sales"], pipe.predict(«test»[features]))})
errors = pd.DataFrame(rows)
errors.round(3)
""")

code(t82, """
errors_long = errors.melt(id_vars="degree", value_vars=["train", "test"], var_name="data", value_name="rmse")
(ggplot(errors_long, aes(x="degree", y="rmse", color="data"))
 + geom_line() + geom_point()
 + labs(x="Polynomial degree (hyperparameter)", y="RMSE (thousand units)", color="")
 + theme_minimal())
""")

md(t82, """
✅ **Checkpoint 3.** (a) Describe the two curves. (b) Which degree would you choose, and why is that not
the degree with the lowest *training* error? (c) Name the phenomenon on the right-hand side of the plot.
""")

answer(t82, """
(a) Training RMSE falls at every step (0.25 by degree 10): more terms always fit the fitted rows better.
Test RMSE drops sharply from degree 1 to 2, bottoms out around **degree 3** (≈ 0.37), then climbs back to
≈ 0.71 by degree 10. (b) Degree 2 or 3 — the lowest **test** error, with the simpler model preferred when
they are close. Training error keeps rewarding complexity, so it cannot be used to choose; that was
Checkpoint 2. (c) **Overfitting**: past degree 3 the model is fitting the noise in the 150 training rows
(66 terms for 150 rows by degree 10), and that does not carry over to new markets.
""")

md(t82, """
*One test split is one random draw.* The standard fix is **cross-validation** — several splits, averaged —
which is where the next topic starts.

---
## The lines to keep

| | |
|---|---|
| **Recipe** | `model = LinearRegression().fit(X, y)` → `model.intercept_`, `model.coef_` → `model.predict(new_df)`, `model.score(X, y)` |
| **`X` is a DataFrame** | double brackets, even for one feature; `predict` needs the same column names |
| **More terms** | more columns in `X`: a second feature, `ads["a"] * ads["b"]`, or `PolynomialFeatures(degree=d)` |
| **Train / test** | `train_test_split(df, test_size=0.25, random_state=…)`; fit on train, measure on test |
| **Choose a hyperparameter** | loop over it, compare **test** error; the fitted-data error only ever goes down |
| **Pipeline** | `make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())` is one model with `.fit` / `.predict` |

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
                    "--ExecutePreprocessor.timeout=600", str(path)], check=True)
    nb = json.loads(path.read_text())
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              and "raises-exception" not in c["metadata"].get("tags", [])
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    missing = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
               and "raises-exception" in c["metadata"].get("tags", [])
               and not any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if errors or missing:
        raise SystemExit(f"unexpected errors: {errors}; demo cells that did not raise: {missing}")
    print(f"executed {path.relative_to(ROOT)} (no unexpected errors)")


TOPICS = [
    (t82, WEEK / "GSB5544_Topic_8_2_Regression",
     "# GSB 5544 — Topic 8.2: Regression in Python  \n"
     "*How to run regression in Python, and how it introduces machine learning. Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
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
