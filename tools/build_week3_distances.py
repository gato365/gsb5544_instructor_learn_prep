#!/usr/bin/env python3
"""Generate the Topic 3.2 (Distances Between Observations) student and solution notebooks.

Usage:  python3 tools/build_week3_distances.py            # write -empty and -solution, execute the solution
        python3 tools/build_week3_distances.py --no-exec  # write both notebooks without executing

A 15-minute opener for PA 3.2, pitched at the PA's level (pandas only; z-scores and min-max;
Euclidean and Manhattan; one-hot encoding for categoricals; cosine similarity for profiles):
  a. types of distances   b. why use them   c. when to use which
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
*How to make "similar" precise: the types of distance, why we use them, and when to use which*
""")

md("""
## The next 15 minutes

| | Question | Where it lands in PA 3.2 |
|---|---|---|
| **a. Types** | What are the distance formulas, and what does each one measure? | every part |
| **b. Why** | What question does a distance answer that a filter, a `groupby`, or a join cannot? | Ames 1, College 1 |
| **c. When** | Which distance, which scaling, which variables — and what to leave out? | Ames 2–3, College 2–3 |

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
| *This customer just signed up — which existing customers do they resemble?* (recommendations, next week's nearest-neighbour prediction) | we need a **ranking** of all rows by resemblance, not a yes/no |
| *Which transaction looks like none of the others?* (outliers, fraud) | "unlike everything" is *far from every row* |

A **distance** turns "similar" into a number: pick some variables, compute how far each row is from the target row on those variables, and sort. Small distance = similar. Everything else in this topic is about computing that number *honestly* — so that the ranking reflects what we mean by "similar" and not an accident of units.

**The one line of pandas that does it** (you will write it a dozen times in the PA):

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
## 2. Types of distance — on a table small enough to check by hand

Three houses, four variables. House **A** is the one we like.
""")

code("""
houses = pd.DataFrame({
    "sqft":     [1656, 1680, 1700],
    "bedrooms": [3,    3,    4],
    "baths":    [1,    1,    2],
    "style":    ["1Story", "2Story", "1Story"],
}, index=["A", "B", "C"])
houses
""")

md("""
### 2a. Euclidean distance — straight-line distance

$$d(i,t) = \\sqrt{\\sum_j (x_{ij} - x_{tj})^2}$$

Square each difference, add them up, take the square root. It is the distance you would measure with a ruler if the variables were axes on a map. Start with the quantitative columns only.
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
Check A→C by hand: differences are 44, 1, 1 → √(44² + 1² + 1²) = √1938 ≈ 44.0. And A→B = √(24²) = 24. So B is "closer" — but look at *why*: 24 vs 44 is entirely about square feet. The extra bedroom and bathroom of house C contributed 1 + 1 = 2 to a sum of 1938. **The units decided the answer.** Hold that thought for Section 3.

### 2b. Manhattan distance — city-block distance

$$d(i,t) = \\sum_j |x_{ij} - x_{tj}|$$

Add up the absolute differences (walk along the streets, not through the buildings). Same idea, but one big difference is not squared, so a single extreme variable dominates less.
""")

code("""
manhattan = diff.«abs»().sum(axis=1)
pd.DataFrame({"euclidean": euclid, "manhattan": manhattan})
""")

md("""
### 2c. Distance on a categorical variable — one-hot encoding

"1Story" minus "2Story" is not a number. **One-hot encode**: one 0/1 column per category. Two rows with the same style differ by 0 on every style column; two rows with different styles differ by 1 in two columns, so a mismatch adds √2 ≈ 1.41 to a Euclidean distance (or 2 to Manhattan). Counting mismatches like this is sometimes called the *Hamming* distance.
""")

code("""
style = pd.«get_dummies»(houses["style"], dtype=float)
style
""")

code("""
style_dist = np.sqrt(((style - style.loc["A"]) ** 2).sum(axis=1))
style_dist                                   # 0 = same style, 1.41 = different style
""")

md("""
### 2d. Cosine similarity — same *mix*, regardless of size

Some rows are **profiles**: the share of students in each major, the counts of words in a document, the mix of products in a basket. Two profiles can have the same shape at different scales. Cosine similarity compares the *direction* of two rows and ignores their length:

$$\\text{cos}(i,t) = \\frac{\\sum_j x_{ij}\\,x_{tj}}{\\sqrt{\\sum_j x_{ij}^2}\\;\\sqrt{\\sum_j x_{tj}^2}}$$

It is 1 for an identical mix and 0 for nothing in common — a **similarity**, so *large* is close (the opposite of a distance).
""")

code("""
majors = pd.DataFrame({"engineering": [600, 60, 100],
                       "business":    [300, 30, 600],
                       "agriculture": [100, 10, 300]},
                      index=["P", "Q", "R"])            # number of students in each field
majors
""")

code("""
target = majors.loc["P"]
euclid_majors = np.sqrt(((majors - target) ** 2).sum(axis=1))

norms  = np.sqrt((majors ** 2).sum(axis=1))
cosine = (majors «@» target) / (norms * norms["P"])      # @ = dot product of every row with P

pd.DataFrame({"euclidean": euclid_majors.round(1), "cosine": cosine.round(3)})
""")

answer("""
Euclidean says Q and R are both *far* from P (≈ 610 and ≈ 616) and cannot tell them apart. Cosine says Q is **identical** to P (1.000) — it has the same 6 : 3 : 1 mix, just one-tenth the size — while R (business-heavy) is clearly different (0.59). When the question is "same kind of school?", cosine is the right lens; when it is "same size and mix?", Euclidean is. In PA 3.2 College 3 the PCIP columns are already proportions, so both agree; with raw counts they would not.
""")

# ============================================================================ #
#  3. Scaling
# ============================================================================ #

md("""
---
## 3. The trap every distance falls into: units

Back to the houses. Square feet are in the thousands; bedrooms and baths are in ones. Euclidean distance added `44²` to `1²` and `1²` and called house C "far". Nobody thinks a house with the same square footage but *an extra bedroom and bathroom* is far from house A — the **units** made that decision, not us.

**Fix: put every variable on the same scale before computing distances.** Two standard ways:

| Method | Formula | Afterwards each column has… |
|---|---|---|
| **z-score** (standardize) | `(X - X.mean()) / X.std()` | mean 0, SD 1 — "one unit" = one standard deviation |
| **min-max** | `(X - X.min()) / (X.max() - X.min())` | minimum 0, maximum 1 |

Either removes the units. z-scores are the default in this course; min-max is common when a bounded 0–1 range matters.
""")

code("""
X_z = (X - X.«mean»()) / X.«std»()
X_z.round(3)
""")

code("""
euclid_z = np.sqrt(((X_z - X_z.loc["A"]) ** 2).sum(axis=1))
pd.DataFrame({"raw euclidean": euclid, "z-scored euclidean": euclid_z.round(3)})
""")

md("""
✅ **Check:** on the raw scale B was nearer to A than C was. What happened after standardizing, and in one sentence, why?
""")

answer("""
The order flips: C is now *farther* from A (≈ 3.2) than B is (≈ 1.1). Standardizing made the 24-square-foot difference (small relative to the spread of `sqft`) count for little, while the extra bedroom and bathroom (each a full standard deviation or more in this tiny table) now count fully. Scaling changed the answer — which is exactly why you must decide on it deliberately.
""")

# ============================================================================ #
#  c. When
# ============================================================================ #

md("""
---
## 4. When to use which — the decisions, in the order you make them

**Decision 1 — which variables?** Distance treats every included column as a vote. Choose the variables that define "similar" *for the question*, and nothing else. Do **not** throw in all 80 columns: ten near-duplicate columns about the basement would out-vote living area.

**Decision 2 — is a variable the *constraint* or part of the *similarity*?** "Cheaper houses like house 0": price is the **constraint** — filter on it *after* computing distances. Putting price *into* the distance would pull the matches toward houses priced like house 0, the opposite of a good deal.

**Decision 3 — scale?** Yes, unless every variable is already in the same unit (e.g. proportions that all lie in 0–1, where standardizing would over-weight rare categories).

**Decision 4 — which formula?**

| Situation | Use | Because |
|---|---|---|
| Several quantitative variables (the default) | **Euclidean** on z-scores | straight-line distance; the standard choice |
| Same, but you want one extreme variable to dominate less | **Manhattan** on z-scores | differences are not squared |
| Categorical variables in the mix | **one-hot encode**, then Euclidean / Manhattan | a mismatch costs √2 (or 2); scale the dummies down (× 0.5) if that is too strict |
| Rows are profiles / proportions / counts where only the mix matters | **cosine similarity** | ignores the size of the row |
| A single categorical variable | count mismatches (Hamming) | that is all one-hot + Manhattan does |

**Decision 5 — sensitivity.** Change one decision at a time (metric, scaling, variable list) and see whether the nearest neighbours change. If they don't, the result is robust; if they do, the decision matters and you should be able to defend it. PA 3.2 asks for exactly this.

**Always look at the neighbours it picks.** A ranking is only as good as the variables behind it; if the "most similar" houses look wrong to a human, the distance is measuring the wrong thing.
""")

# ============================================================================ #
#  5. The recipe on real data (PA readiness)
# ============================================================================ #

md("""
---
## 5. The recipe on the real data — PA 3.2 readiness check

The same steps on the 2,930-house Ames data set the PA uses: **select → scale → distance → constrain → sort → look**. This is Ames part 1 of the PA in one function; you will extend it (more variables, categoricals, other data) there.
""")

code("""
df_ames = pd.read_csv("https://dlsun.github.io/pods/data/AmesHousing.txt", sep="\\t")
df_ames["Bathrooms"] = df_ames["Full Bath"] + 0.5 * df_ames["Half Bath"]

house0 = df_ames.loc[0]
house0[["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "SalePrice"]]
""")

code("""
def nearest(df, features, target=0, metric="euclidean", scaling="z", k=5, max_price=None):
    \"\"\"The k rows of `df` nearest to row `target` on `features` (optionally only rows cheaper than max_price).\"\"\"
    X = df[features].astype(float)                                   # 1. select
    if scaling == "z":                                               # 2. scale
        X = (X - X.mean()) / X.std()
    elif scaling == "minmax":
        X = (X - X.min()) / (X.max() - X.min())
    diff = X - X.loc[target]                                         # 3. distance
    dist = np.sqrt((diff ** 2).sum(axis=1)) if metric == "euclidean" else diff.abs().sum(axis=1)
    if max_price is not None:                                        # 4. constrain (price is NOT in the distance)
        dist = dist[df["SalePrice"] < max_price]
    dist = dist.drop(target, errors="ignore")
    return dist.sort_values().head(k)                                # 5. sort

features = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]
best = nearest(df_ames, features, max_price=house0["SalePrice"])
best
""")

code("""
# 6. look at what it picked
df_ames.loc[best.index, ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "SalePrice"]]
""")

md("""
✅ **Check:** run the recipe again with `scaling="none"` and with `metric="manhattan"`. Which of the two changes the five houses, and does that match what Section 3 predicted?
""")

code("""
print("z / euclidean :", list(nearest(df_ames, features, max_price=house0["SalePrice"]).index))
print("z / manhattan :", list(nearest(df_ames, features, metric=«"manhattan"», max_price=house0["SalePrice"]).index))
print("none/euclidean:", list(nearest(df_ames, features, scaling=«"none"», max_price=house0["SalePrice"]).index))
""")

answer("""
Manhattan returns the same five houses as Euclidean (in a slightly different order); removing the scaling replaces the list with houses matched on square feet alone. As in Section 3, the **scaling** decision changes the answer and the **metric** decision barely does — for these three variables. In the PA you will add a categorical variable and then many variables, and see the *variable list* become the decision that matters most.
""")

# ============================================================================ #
#  Summary
# ============================================================================ #

md(f"""
---
## Summary

| | |
|---|---|
| **What a distance is** | one number per row saying how far it is from a target row on chosen variables; small = similar |
| **Why** | "find rows like this one": recommendations, comparable cases, nearest-neighbour prediction, outliers — questions with no category to filter or group on |
| **Types** | Euclidean (straight line) · Manhattan (city block) · one-hot + either for categoricals · cosine similarity for profiles |
| **Before computing** | choose the variables deliberately; keep the constraint (price) *out* of the distance; scale (z-score) unless the variables share a unit |
| **After computing** | filter on the constraint, sort, **look** at the neighbours, and test sensitivity to each decision |
| **The code** | `X = (X - X.mean()) / X.std()` → `np.sqrt(((X - X.loc[t]) ** 2).sum(axis=1))` → `.sort_values().head(k)` |

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
        "*Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"
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
