#!/usr/bin/env python3
"""Generate the Topic 3.2 (Distances Between Observations) student and solution notebooks.

Usage:  python3 tools/build_week3_distances.py            # write -empty and -solution, execute the solution
        python3 tools/build_week3_distances.py --no-exec  # write both notebooks without executing

A 15-minute opener for PA 3.2, pitched at the PA's level (pandas only, no functions or if-statements):
  a. what a distance measures and that options exist   b. why use them   c. the two rules (scale, one-hot) + which variables
Emphasis follows the course lead's guidance: Euclidean is the default, other options exist and can change
results but need not be memorised; always standardize quantitative variables and one-hot encode categoricals.
The same markers as build_week3_joins.py control the student copy:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
The Ames data is read from the same public URL the PA uses, so executing needs a network connection.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_3" / "distances_between_observations"
STUDENT = WEEK / "GSB5544_Topic_3_2_Distances_Between_Observations-empty.ipynb"
SOLUTION = WEEK / "GSB5544_Topic_3_2_Distances_Between_Observations-solution.ipynb"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(text: str) -> None:
    md("**Answer:** " + text.strip())


# ============================================================================ #
#  Title and agenda
# ============================================================================ #

md("""
# GSB 5544 — Topic 3.2: Distances Between Observations — SOLUTION
*How to make "similar" precise: what a distance is, why we use one, and the two rules to follow before computing it*
""")

md("""
## The next 15 minutes

You have done the reading, so this is a quick synthesis before PA 3.2 — not a replacement for it.

| | Question | Where it lands in PA 3.2 |
|---|---|---|
| **a. Types** | What does a distance measure, and what options exist? | every part |
| **b. Why** | What question does a distance answer that a filter, a `groupby`, or a join cannot? | Ames 1, College 1 |
| **c. When** | Which variables go in, what to do to them first, and what to leave out | Ames 2–3, College 2–3 |

Last topic combined **two tables** through a key. This topic stays inside **one table** and asks a different question: *which rows are most like this row?* No key, no matching — a **number** that says how far apart two rows are.
""")

code("""
import pandas as pd
import numpy as np
""")

# ============================================================================ #
#  b. Why
# ============================================================================ #

md("""
---
## 1. Why distances?  — *"Find me something like this one."*

Questions that sound like this come up constantly, and none of them can be answered with the tools we have so far:

| The question | Filtering / grouping can't, because… |
|---|---|
| *I like house 0 but it is too expensive — find cheaper houses like it.* | "like it" is not a single condition; it is *close on several variables at once* |
| *Which colleges are most similar to Cal Poly?* | there is no category called "Cal-Poly-like" to group by |
| *This customer just signed up — which existing customers do they resemble?* (recommendations; nearest-neighbour prediction later in the course) | we need a **ranking** of all rows by resemblance, not a yes/no |
| *Which transaction looks like none of the others?* (outliers, fraud) | "unlike everything" is *far from every row* |

A **distance** turns "similar" into a number: pick some variables, compute how far each row is from the target row on those variables, and sort. Small distance = similar. The rest of this notebook is about computing that number so that the ranking reflects what *we* mean by "similar" — and not an accident of units.

**The one line of pandas that does it** (you will write it many times in the PA):

```python
dist = np.sqrt(((X - X.loc[target]) ** 2).sum(axis=1))   # Euclidean distance from `target` to every row
```

`X - X.loc[target]` subtracts the target's row from every row (broadcasting); square, sum across the columns (`axis=1`), square-root.
""")

# ============================================================================ #
#  a. Types — tiny example
# ============================================================================ #

md("""
---
## 2. What a distance measures — on a table small enough to check by hand

Three houses, four variables. House **A** is the one we like. Before computing anything, decide for yourself: is **B** (44 more square feet, otherwise identical) or **C** (4 more square feet, but an extra bedroom *and* an extra bathroom) more like A?
""")

code("""
houses = pd.DataFrame({
    "sqft":     [1656, 1700, 1660],
    "bedrooms": [3,    3,    4],
    "baths":    [1,    1,    2],
    "style":    ["1Story", "2Story", "1Story"],
}, index=["A", "B", "C"])
houses
""")

md("""
### 2a. Euclidean distance — the default

$$d(i,t) = \\sqrt{\\sum_j (x_{ij} - x_{tj})^2}$$

Square each difference, add them up, take the square root: the straight-line distance you would measure with a ruler if the variables were axes on a map. This is the distance we use unless there is a reason not to (it is also scikit-learn's default later in the course). Start with the quantitative columns.
""")

code("""
X = houses[["sqft", "bedrooms", "baths"]]

diff = X - X.loc["A"]              # each row minus house A's row
diff
""")

code("""
euclid = np.sqrt((diff «** 2»).sum(axis=«1»))
euclid
""")

md("""
Check by hand: A→B differs only in square feet, so d = √(44²) = **44**. A→C: √(4² + 1² + 1²) = √18 ≈ **4.2**. The formula says C — the house with an extra bedroom *and* bathroom — is ten times closer to A than B, which merely has 44 more square feet. Does that match what you decided above? Probably not. Hold that thought for Section 3.

### 2b. Other options exist — and give different numbers

Euclidean is not the only way to add up differences. **Manhattan distance** adds the absolute differences instead of squaring them:

$$d(i,t) = \\sum_j |x_{ij} - x_{tj}|$$

There are others (the quiz asks you to compute a couple). The point is not to memorise them: it is to know that **the choice exists, that different choices give different numbers, and that they can occasionally change which rows come out "closest"**. When in doubt, use Euclidean.
""")

code("""
manhattan = diff.«abs»().sum(axis=1)
pd.DataFrame({"euclidean": euclid, "manhattan": manhattan})
""")

md("""
### 2c. A categorical variable — one-hot encode it first

"1Story" minus "2Story" is not a number, so a categorical column cannot go into either formula as it is. **One-hot encode** it: one 0/1 column per category. Two houses with the same style differ by 0 on every style column; two with different styles differ by 1 in two columns — a mismatch adds √2 ≈ 1.41 to a Euclidean distance.
""")

code("""
style = pd.«get_dummies»(houses["style"], dtype=float)
style
""")

code("""
style_dist = np.sqrt(((style - style.loc["A"]) ** 2).sum(axis=1))
style_dist                                   # 0 = same style as A, 1.41 = different style
""")

md("""
### 2d. One more option, for *profiles*: cosine similarity

Some rows are **profiles** — the number of students in each field of study, the counts of each word in a document, the mix of products in a basket. For those, we often care about the *mix* and not the *size*. Cosine similarity compares the direction of two rows and ignores their length; it is 1 for an identical mix and 0 for nothing in common (a **similarity**, so *large* means close).

Below, **P, Q, and R are three schools** (the rows); the columns are fields of study, and each cell is the number of students in that field.
""")

code("""
schools = pd.DataFrame({"engineering": [600, 60, 100],
                        "business":    [300, 30, 600],
                        "agriculture": [100, 10, 300]},
                       index=["P", "Q", "R"])            # rows = schools, columns = fields, cells = students
schools
""")

code("""
target = schools.loc["P"]
euclid_schools = np.sqrt(((schools - target) ** 2).sum(axis=1))

norms  = np.sqrt((schools ** 2).sum(axis=1))
cosine = (schools «@» target) / (norms * norms["P"])     # @ = dot product of every row with P

pd.DataFrame({"euclidean": euclid_schools.round(1), "cosine": cosine.round(3)})
""")

answer("""
Euclidean says Q and R are both far from P (≈ 610 and ≈ 616) and cannot tell them apart. Cosine says Q is **identical** to P (1.000) — the same 6 : 3 : 1 mix, one-tenth the size — while R (business-heavy) is different (0.59). Same data, different option, different answer: that is the lesson. (In PA 3.2 College 3 the field columns are already proportions, so the two agree there.)
""")

# ============================================================================ #
#  3. The two rules
# ============================================================================ #

md("""
---
## 3. The two rules before computing any distance

Back to the houses. Square feet are in the thousands; bedrooms and baths are in ones. The raw Euclidean distance added `4²` to `1² + 1²` and decided that an extra bedroom and bathroom matter about as much as four square feet. The **units** made that decision, not us.

> **Rule 1 — always scale the quantitative variables.**
> **Rule 2 — always one-hot encode the categorical variables.**

For Rule 1 we **standardize** (z-scores): subtract each column's mean and divide by its standard deviation, so every column has mean 0 and SD 1 and "one unit" means "one standard deviation" in every column. Other scalings exist — min-max, `(X - X.min()) / (X.max() - X.min())`, squeezes each column into 0–1 — and the PA asks you to try one so you see that the option exists. From then on we always standardize.
""")

code("""
X_z = (X - X.«mean»()) / X.«std»()
X_z.round(2)
""")

code("""
euclid_z = np.sqrt(((X_z - X_z.loc["A"]) ** 2).sum(axis=1))
pd.DataFrame({"raw euclidean": euclid.round(2), "standardized euclidean": euclid_z.round(2)})
""")

md("""
✅ **Check:** which house is nearer to A on the raw scale, and which after standardizing? Which answer matches what you decided at the start of Section 2, and in one sentence, why did it change?
""")

answer("""
Raw: **C** is nearer (4.2 vs 44), purely because square feet dominate. Standardized: **B** is nearer (1.8 vs 2.5) — which matches intuition: 44 square feet is nothing, while an extra bedroom *and* bathroom make C a different kind of house. After standardizing, the 44 sq ft is about 1.8 SDs of `sqft`, and each extra room is about 1.7 SDs of its column, so the two room differences together outweigh the one size difference. Min-max scaling gives the same ranking (B 1.0, C 1.4). Scaling changed the answer, which is why Rule 1 is a rule.
""")

md("""
### 3b. Quantitative and categorical together

You do not have to choose between them. Standardize the quantitative columns (Rule 1), one-hot encode the categorical one (Rule 2), put the columns side by side in **one** table, and compute **one** distance. PA 3.2 Ames 2 and 3 do exactly this.
""")

code("""
X_all = pd.«concat»([X_z, style], axis=1)     # 3 standardized columns + 3 style indicator columns
X_all.round(2)
""")

code("""
euclid_all = np.sqrt(((X_all - X_all.loc["A"]) ** 2).sum(axis=1))
pd.DataFrame({"standardized (quant only)": euclid_z.round(2), "quant + style": euclid_all.round(2)})
""")

md("""
B is a different style from A, so it picks up the √2 mismatch penalty and its distance rises from 1.8 to 2.3; C keeps its 2.5. B is still nearer — but only just. If style should count *less* than a full mismatch, multiply the indicator columns by a weight below 1 (e.g. `0.5 * style`) before concatenating; if it should count *more*, weight it above 1.
""")

# ============================================================================ #
#  c. When: the questions to ask before computing
# ============================================================================ #

md("""
---
## 4. Before you compute: three questions

**Question 1 — which variables define "similar" *for this question*?** Distance treats every included column as an equal vote. Choose a few variables deliberately; do not throw in all 80 columns, or ten near-duplicate basement columns will out-vote living area.

**Question 2 — for "cheaper houses like house 0", should `SalePrice` be one of the distance variables?** Think about it before reading on, and be ready to say why.
""")

md("""
✅ **Check:** write your answer to Question 2 and your reason in one or two sentences *before* running anything.
""")

answer("""
No. There are two different roles a variable can play:

- a **similarity variable** is something the two houses should *match on* — it goes *into* the distance (living area, bedrooms, …);
- a **constraint** is a rule about which houses are *allowed at all* — it is a *filter*, applied to the rows, not a distance ingredient.

Here the whole point is a house that is like house 0 **but cheaper**, so price is the constraint: compute the distance *without* it, then keep only rows with `SalePrice < house0["SalePrice"]`. If price went into the distance, the nearest houses would be the ones priced *like* house 0 — the opposite of a good deal. Ask this question about every variable: does it describe *what I want to match*, or *which rows are eligible*?
""")

md("""
**Question 3 — which distance, and which scaling?** Rule 1 and Rule 2 are not optional. Beyond that, the table below is a **suggestion, not a rule** — it says what people usually reach for. Different options exist, they are all available to you, and they can influence which rows come out nearest. Standardized Euclidean is the default; the way to find out whether the choice matters for *your* question is to try another one and compare.

| Situation | What people usually do |
|---|---|
| several quantitative variables | standardize, Euclidean |
| a categorical variable in the mix | one-hot encode it, then the same distance |
| rows are profiles (proportions, counts) where only the mix matters | cosine similarity |
| you want one extreme variable to dominate less | Manhattan |

**And always look at the neighbours it picks.** If the "most similar" houses look wrong to a human, the distance is measuring the wrong thing — usually the variable list, not the formula.
""")

# ============================================================================ #
#  5. On the real data (PA readiness)
# ============================================================================ #

md("""
---
## 5. The same steps on the real data — PA 3.2 readiness check

The 2,930-house Ames data set the PA uses. The recipe is five steps of ordinary pandas: **select → scale → distance → constrain → sort and look**. Write it once; when the PA asks you to try another option, copy the cell, paste it, and change one line.
""")

code("""
df_ames = pd.read_csv("https://dlsun.github.io/pods/data/AmesHousing.txt", sep="\\t")
df_ames["Bathrooms"] = df_ames["Full Bath"] + 0.5 * df_ames["Half Bath"]

house0 = df_ames.loc[0]
house0[["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "SalePrice"]]
""")

code("""
# 1. select the similarity variables (price is the constraint, so it is NOT here)
X = df_ames[["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]]

# 2. scale (Rule 1)
X_z = (X - X.mean()) / X.std()

# 3. distance from house 0 to every house
df_ames["dist"] = np.sqrt(((X_z - X_z.«loc[0]») ** 2).sum(axis=1))

# 4. constrain: only houses cheaper than house 0
cheaper = df_ames[df_ames["SalePrice"] «<» house0["SalePrice"]]

# 5. sort and look
show = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "SalePrice", "dist"]
cheaper.sort_values("dist")[show].head(5)
""")

md("""
✅ **Check (copy, paste, edit):** copy the cell above into the empty cell below and change **one** line so it uses Manhattan distance (absolute differences, no square root). Then copy it again and *remove* the scaling step (use `X` instead of `X_z`). Which change alters the five houses?
""")

code("""
# Manhattan: only step 3 changes
X = df_ames[["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]]
X_z = (X - X.mean()) / X.std()
df_ames["dist"] = (X_z - X_z.loc[0]).«abs»().sum(axis=1)
cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
cheaper.sort_values("dist")[show].head(5)
""")

code("""
# No scaling: step 2 removed, step 3 uses X instead of X_z
X = df_ames[["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]]
df_ames["dist"] = np.sqrt(((«X» - «X».loc[0]) ** 2).sum(axis=1))
cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
cheaper.sort_values("dist")[show].head(5)
""")

answer("""
Manhattan returns the same five houses as Euclidean (in a slightly different order). Removing the scaling replaces the list with houses matched on square feet alone — some with a different number of bedrooms or bathrooms. Same lesson as Section 3: Rule 1 changes the answer; the choice of formula, here, barely does. In the PA you add a categorical variable and then many more variables, and the *variable list* becomes the decision that matters most.
""")

# ============================================================================ #
#  Pair up + summary
# ============================================================================ #

md("""
---
## 6. Before you start the PA — pair up

With the person next to you, agree on answers to these three questions for PA 3.2 Ames part 3, where *you* choose the variables. Be ready to share with the class.

1. Which **four to six** variables would you use to decide that two houses are "similar"? Name at least one categorical variable.
2. Which variable(s) are **constraints** rather than similarity variables?
3. Which *one* option would you change (scaling, formula, or the variable list) to test whether your answer is robust — and what would convince you it matters?
""")

md(f"""
## Summary

| | |
|---|---|
| **What a distance is** | one number per row saying how far it is from a target row on chosen variables; small = similar |
| **Why** | "find rows like this one": recommendations, comparable cases, nearest-neighbour prediction, outliers — questions with no category to filter or group on |
| **Options** | Euclidean is the default; Manhattan, cosine similarity, and others exist and can give different answers — try one to see whether it matters |
| **Rule 1** | always scale the quantitative variables (we standardize) |
| **Rule 2** | always one-hot encode the categorical variables, then put everything in one table |
| **Before computing** | pick the similarity variables deliberately; keep constraints (like price) out of the distance and filter on them afterwards |
| **After computing** | sort, **look** at the neighbours, and change one option to check sensitivity |
| **The code** | `X_z = (X - X.mean()) / X.std()` → `np.sqrt(((X_z - X_z.loc[t]) ** 2).sum(axis=1))` → filter → `.sort_values("dist").head(k)` |

PA 3.2 is on the course site: [{SITE}]({SITE}).
""")


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


def solution_cells() -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub(lambda m: m.group(1), c["source"])
        out.append(c)
    return out


def student_cells() -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub("____", c["source"])
        elif c["source"].startswith("**Answer:**"):
            c["source"] = "**Your answer:** *(write it here — replace this line)*"
        out.append(c)
    out[0]["source"] = (
        "# GSB 5544 — Topic 3.2: Distances Between Observations  \n"
        "*Fill each `____` blank as you work; the ✅ checks ask for a sentence or two, or for you to copy, paste, and edit a cell.*"
    )
    return out


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=600", str(path)], check=True)
    nb = json.loads(path.read_text())
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"execution errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    save(STUDENT, notebook(student_cells()))
    save(SOLUTION, notebook(solution_cells()))
    print(f"wrote {STUDENT.relative_to(ROOT)} and {SOLUTION.relative_to(ROOT)} ({len(cells)} cells)")
    if "--no-exec" not in sys.argv:
        execute(SOLUTION)


if __name__ == "__main__":
    main()
