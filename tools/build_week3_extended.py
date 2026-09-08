#!/usr/bin/env python3
"""Build the Week 3 *extended* notebooks: fully worked, step-by-step walkthroughs.

Usage:  python3 tools/build_week3_extended.py            # write + execute all three
        python3 tools/build_week3_extended.py --no-exec  # write only

Produces (all executed, no student version — these are reference walkthroughs):
  joining_and_merging/GSB5544_Topic_3_1_Joining_and_Merging-extended.ipynb
  joining_and_merging/GSB5544_PA_3_1_Concatenating_Joining_Pivoting-extended.ipynb
  distances_between_observations/GSB5544_PA_3_2_Distances_Between_Observations-extended.ipynb

What "extended" adds over the -solution notebooks:
  * a "Why are we doing this?" block before every operation (the question, why one table
    cannot answer it, what the key is, what to expect),
  * every input table displayed BEFORE the operation and the result AFTER it (plain tables;
    the colour-coded, animated version of each operation lives on the interactive page),
  * every practice-activity answer broken into numbered steps with the logic for each step
    and a plain-language explanation of each pandas function and argument.

The Topic notebook reads course data from GitHub raw URLs (redirected to assignments/Data
while executing locally); the PA notebooks read public data sets over the network.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_3"
DATA_DIR = ROOT / "assignments" / "Data"
JUPYTER = "/opt/anaconda3/bin/jupyter"

TOPIC_OUT = WEEK / "joining_and_merging" / "GSB5544_Topic_3_1_Joining_and_Merging-extended.ipynb"
PA31_OUT = WEEK / "joining_and_merging" / "GSB5544_PA_3_1_Concatenating_Joining_Pivoting-extended.ipynb"
PA32_OUT = WEEK / "distances_between_observations" / "GSB5544_PA_3_2_Distances_Between_Observations-extended.ipynb"

PAGE = "https://gato365.github.io/gsb5544_instructor_learn_prep/week_3/pandas.html"
DATA = "https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


import ast


def strip_show_args(src: str) -> str:
    """Rewrite show(df, title, <colour>, sources=..., key=..., rows=n) as show(df, title, rows=n)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    # ast column offsets are UTF-8 *byte* offsets, so splice on the encoded bytes
    raw = src.encode("utf-8")
    offsets = [0]
    for ln in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(ln))
    pos = lambda n, end=False: offsets[(n.end_lineno if end else n.lineno) - 1] + (n.end_col_offset if end else n.col_offset)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "show":
            args = [ast.get_source_segment(src, a) for a in node.args[:2]]
            rows = [ast.get_source_segment(src, k.value) for k in node.keywords if k.arg == "rows"]
            edits.append((node, "show(" + ", ".join(args) + (f", rows={rows[0]}" if rows else "") + ")"))
    for node, new in sorted(edits, key=lambda e: pos(e[0]), reverse=True):
        raw = raw[:pos(node)] + new.encode("utf-8") + raw[pos(node, True):]
    return raw.decode("utf-8")


class Book:
    def __init__(self) -> None:
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})

    def code(self, text: str) -> None:
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": strip_show_args(text.strip("\n"))})

    def notebook(self) -> dict:
        out = []
        for i, c in enumerate(self.cells):
            c = copy.deepcopy(c)
            c["id"] = f"cell-{i:03d}"
            out.append(c)
        return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


# ============================================================================ #
#  Shared helper cell: captioned plain tables
# ============================================================================ #

HELPER_MD = """
### How the tables are shown

Every table is printed with a caption giving its name and its full size, followed by its first rows. The **interactive page** shows the same operations with columns coloured by the table they came from (blue = left, orange = right, green = key, yellow = created) and animates each result row: [PAGE](PAGE). Run the next cell once; it defines the `show()` helper used throughout.
""".replace("PAGE", PAGE)

HELPER_CODE = r'''
import pandas as pd
import numpy as np
from IPython.display import display, Markdown

def show(df, title, rows=8):
    """Print a caption (name and full size), then the first `rows` rows of `df`."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    more = "" if len(df) <= rows else f"  (first {rows} shown)"
    display(Markdown(f"**{title}** — {df.shape[0]:,} rows × {df.shape[1]} columns{more}"))
    display(df.head(rows))

print("show() is ready")
'''


def add_helper(b: Book) -> None:
    b.md(HELPER_MD)
    b.code(HELPER_CODE)


# ============================================================================ #
#  1. Topic 3.1 — extended
# ============================================================================ #

def build_topic() -> Book:
    b = Book()
    b.md("""
# GSB 5544 — Topic 3.1 (Extended): Joining and Merging Data, step by step
*Every operation with the question that motivates it, the tables before, and the table after.*
""")
    b.md(f"""
## What this extended version adds

The standard Topic 3.1 notebook shows each technique once. This version slows down and, for **every** operation, answers four questions *before* running anything:

1. **What question are we trying to answer?** — the reason a single table is not enough.
2. **Which tables hold the pieces, and what is the key** that links them?
3. **What do we expect** the result to look like (how many rows, which columns, where the `NaN`s will be)?
4. **What actually happened** — read off the result and check it against the expectation.

Each operation shows the input table(s) first, then the result. The interactive page shows the same operation with columns coloured by the table they came from and the key highlighted, and animates the rows: [{PAGE}]({PAGE}) (every section links to its technique). The standard notebook and its student version are on the course site: [{SITE}]({SITE}).
""")
    add_helper(b)

    # ---- Setup ----
    b.md("""
---
## 1. The situation: two tables, one question

A department keeps a **roster** of students and, separately, a **score sheet**. Neither table alone can answer *"what did each named student score?"* — the roster has names but no scores, the score sheet has scores but no names. They are linked only by `student_id`. That link is the **key**, and using it to combine tables is a **join**.
""")
    b.code("""
students = pd.DataFrame({"student_id": [1, 2, 3], "name": ["Ava", "Ben", "Cam"]})
grades   = pd.DataFrame({"student_id": [2, 3, 4], "score": [80, 90, 70]})

show(students, "students  (LEFT table: the roster)", "blue", key="student_id")
show(grades,   "grades  (RIGHT table: the score sheet)", "orange", key="student_id")
""")
    b.md("""
**Look at the key column before doing anything.** In `students` the IDs are 1, 2, 3; in `grades` they are 2, 3, 4.

- IDs **2 and 3** are in *both* tables — these rows will match.
- ID **1** (Ava) is only in the roster — no score exists.
- ID **4** is only in the score sheet — a score with no roster entry (a typo? a dropped student?).

Every join below is a different policy for rows 1 and 4. Write down what you expect *before* each result: it is the fastest way to learn what a join does.
""")

    # ---- Four joins ----
    b.md(f"""
---
## 2. The four joins — same inputs, four policies for unmatched rows

The inputs are the two tables above; they do not change in this section, so they are not re-printed. The key is `student_id` in both.

### 2a. Inner join  🎬 [{PAGE}#inner-join]({PAGE}#inner-join)

**Question:** *Among students on the roster, who has a score, and what is it?*
**Why a join:** names are in `students`, scores in `grades`.
**Expect:** only IDs found in both tables → 2 rows (Ben, Cam); columns `student_id`, `name`, `score`; no `NaN`.
""")
    b.code("""
inner = students.merge(grades, on="student_id", how="inner")
show(inner, 'students.merge(grades, on="student_id", how="inner")', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
""")
    b.md("""
**What happened:** pandas walked the roster row by row and looked each `student_id` up in `grades`. Rows 2 and 3 found a partner and were glued together — the roster's columns first, then the score sheet's; the key appears once. Row 1 (no partner) and row 4 (not in the roster at all) were dropped. `how="inner"` is the default, which is why an unadorned `.merge()` quietly loses rows.
""")
    b.md(f"""
### 2b. Left join  🎬 [{PAGE}#left-join]({PAGE}#left-join)

**Question:** *Give me the complete roster, with scores where they exist — which students are still missing a score?*
**Why a join:** same two tables, but now the roster is the thing we must not lose.
**Expect:** all 3 roster rows; Ava's `score` is `NaN`; ID 4 still absent because it is not on the roster.
""")
    b.code("""
left = students.merge(grades, on="student_id", how="left")
show(left, 'students.merge(grades, on="student_id", how="left")', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
left.dtypes
""")
    b.md("""
**What happened:** every left row survives. Ava had no partner, so her right-side columns were filled with `NaN`. Notice the side effect in `dtypes`: `score` was an integer column but a column containing `NaN` must be stored as **float** — hence `80.0`. The missing score is the *answer* to the question: it points at the student without a score.
""")
    b.md(f"""
### 2c. Right join  🎬 [{PAGE}#right-join]({PAGE}#right-join)

**Question:** *Keep every score record — which scores belong to an ID that is not on the roster?*
**Why a join:** now the score sheet is the thing we must not lose (auditing the scores, not the roster).
**Expect:** all 3 score rows; ID 4 has `NaN` for `name`; Ava absent.
""")
    b.code("""
right = students.merge(grades, on="student_id", how="right")
show(right, 'students.merge(grades, on="student_id", how="right")', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
""")
    b.md("""
**What happened:** the mirror image of the left join. The `NaN` under `name` is a score recorded for someone who is not a student — exactly the kind of data-quality problem a right join exposes. (`students.merge(grades, how="right")` is the same as `grades.merge(students, how="left")` apart from column order.)
""")
    b.md(f"""
### 2d. Outer (full) join  🎬 [{PAGE}#full-outer-join]({PAGE}#full-outer-join)

**Question:** *Across both files, which IDs appear only in the roster, only in the scores, and in both?*
**Why a join:** an audit of both sources at once.
**Expect:** 4 rows (IDs 1–4), `NaN` on whichever side is missing, rows sorted by key.
""")
    b.code("""
outer = students.merge(grades, on="student_id", how="outer")
show(outer, 'students.merge(grades, on="student_id", how="outer")', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
""")
    b.md("""
**What happened:** nothing was dropped. Each `NaN` marks a one-sided row. A useful identity when keys are unique on both sides: `rows(outer) = rows(left) + rows(right) − rows(inner)` = 3 + 3 − 2 = 4.

**Check yourself:** the four results have 2, 3, 3, 4 rows. Say, for each, which of the two "odd" IDs (1 and 4) survived and why.
""")

    # ---- Anti joins ----
    b.md(f"""
---
## 3. Anti joins — when the *unmatched* rows are the answer

### 3a. Left anti join  🎬 [{PAGE}#left-anti-join]({PAGE}#left-anti-join)

**Question:** *Which students have not received a score yet?* (Someone has to chase them.)
**Why not a plain join:** a join returns the matches; we want the leftovers. pandas has no `how="anti"`, so we filter: keep roster rows whose ID is **not in** the score sheet.
**Expect:** just Ava, with only the roster's columns.
""")
    b.code("""
mask = students["student_id"].isin(grades["student_id"])
show(pd.DataFrame({"student_id": students["student_id"], "in_grades": mask}),
     'Step 1: students["student_id"].isin(grades["student_id"])', "grey", key="student_id")

left_anti = students[~mask]
show(left_anti, "Step 2: students[~mask]  (rows with NO partner in grades)", "purple",
     sources={"student_id": "green", "name": "blue"}, key="student_id")
""")
    b.md("""
**What happened:** `.isin()` produced one True/False per roster row; `~` flips it; the boolean filter keeps the False rows. Only left-table columns exist because no join took place.

### 3b. Right anti join  🎬 [{PAGE}#right-anti-join]({PAGE}#right-anti-join)

**Question:** *Which scores were recorded under an ID that no student has?* **Expect:** only ID 4.
""")
    b.code("""
right_anti = grades[~grades["student_id"].isin(students["student_id"])]
show(right_anti, 'grades[~grades["student_id"].isin(students["student_id"])]', "purple",
     sources={"student_id": "green", "score": "orange"}, key="student_id")
""")
    b.md("""
**Same answer via the outer join:** add `indicator=True` and keep the `left_only` (or `right_only`) rows. This scales to multi-column keys, where `.isin()` on a single column is not enough.
""")
    b.code("""
audit = students.merge(grades, on="student_id", how="outer", indicator=True)
show(audit, "outer join with indicator=True", "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
show(audit[audit["_merge"] == "left_only"], 'audit[audit["_merge"] == "left_only"]  (= left anti join)', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
""")

    # ---- Cross join ----
    b.md(f"""
---
## 4. Cross join — every combination  🎬 [{PAGE}#cross-join]({PAGE}#cross-join)

**Question:** *If every student could take every course, what would the full enrollment grid look like?* (Useful for building a template you then fill in, or for pricing every product × every region.)
**Why not a key:** there is nothing to match on — we *want* all pairs.
**Expect:** 3 names × 2 courses = 6 rows.
""")
    b.code("""
names   = pd.DataFrame({"name": ["Ava", "Ben", "Cam"]})
courses = pd.DataFrame({"course": ["Stats", "Python"]})
show(names, "names (LEFT)", "blue")
show(courses, "courses (RIGHT)", "orange")

grid = names.merge(courses, how="cross")
show(grid, 'names.merge(courses, how="cross")', "purple", sources={"name": "blue", "course": "orange"})
""")
    b.md("""
**What happened:** each left row was paired with each right row; no key, no `NaN`, and the row count is the *product* of the two sizes — which is why a cross join on two large tables is dangerous.
""")

    # ---- Join details ----
    b.md(f"""
---
## 5. Details that break real joins

### 5a. The key has a different name in each table  🎬 [{PAGE}#different-key-names]({PAGE}#different-key-names)

**Question:** *Attach scores to the roster when the score file calls the ID column `id`.*
**Why it matters:** `on="student_id"` would raise `KeyError` because the right table has no such column.
**Expect:** the same left join as 2b, but **both** key columns survive (`student_id` and `id`).
""")
    b.code("""
grades_id = pd.DataFrame({"id": [2, 3, 4], "score": [80, 90, 70]})
show(students, "students (LEFT) — key is student_id", "blue", key="student_id")
show(grades_id, "grades_id (RIGHT) — key is id", "orange", key="id")

diff_names = students.merge(grades_id, how="left", left_on="student_id", right_on="id")
show(diff_names, 'students.merge(grades_id, how="left", left_on="student_id", right_on="id")', "purple",
     sources={"student_id": "green", "name": "blue", "id": "green", "score": "orange"}, key=["student_id", "id"])
""")
    b.md("""
**What happened:** pandas matched `student_id` against `id`, but because the names differ it cannot collapse them into one column, so both stay. Ava's `id` is `NaN` because that value came from the right side. Drop one afterwards with `.drop(columns="id")`.
""")
    b.md(f"""
### 5b. The key is a combination of columns  🎬 [{PAGE}#multiple-keys]({PAGE}#multiple-keys)

**Question:** *What score did each student earn in each course they are enrolled in?*
**Why one key is not enough:** Ava is enrolled in two courses and has two grades. Matching on `student_id` alone cannot tell which grade belongs to which enrollment.
**Expect (correct):** 3 rows, one per enrollment, each with the matching course's score. **Expect (wrong, single key):** Ava's two enrollments × Ava's two grades = 4 Ava rows.
""")
    b.code("""
enrollment    = pd.DataFrame({"student_id": [1, 1, 2], "course": ["Stats", "Python", "Stats"]})
course_grades = pd.DataFrame({"student_id": [1, 1, 2], "course": ["Python", "Stats", "Stats"], "score": [95, 80, 90]})
show(enrollment, "enrollment (LEFT)", "blue", key=["student_id", "course"])
show(course_grades, "course_grades (RIGHT)", "orange", key=["student_id", "course"])

two_keys = enrollment.merge(course_grades, how="left", on=["student_id", "course"])
show(two_keys, 'on=["student_id", "course"]  (correct)', "purple",
     sources=sources_from(enrollment, course_grades, ["student_id", "course"]), key=["student_id", "course"])

one_key = enrollment.merge(course_grades, how="left", on="student_id")
show(one_key, 'on="student_id"  (WRONG: courses mixed, rows multiplied)', "purple",
     sources={"student_id": "green", "course_x": "yellow", "course_y": "yellow", "score": "orange"}, key="student_id")
""")
    b.md("""
**What happened:** with both columns in the key, each enrollment found exactly one grade. With one key, every Ava enrollment matched every Ava grade, producing 5 rows, and the two `course` columns collided and were renamed `course_x` / `course_y` — a sure sign the key was incomplete.
""")
    b.md(f"""
### 5c. Where did each row come from, and is the key really unique?  🎬 [{PAGE}#audit-validate]({PAGE}#audit-validate)

**Question:** *After the join, can I prove nothing was lost or duplicated?*
**Why:** joins fail silently. `indicator=True` labels every row's origin; `validate=` asserts the key relationship you expect (`"one_to_one"`, `"one_to_many"`, `"many_to_one"`) and raises `MergeError` if the data disagrees.
""")
    b.code("""
audit = students.merge(grades, on="student_id", how="outer", indicator=True, validate="one_to_many")
show(audit, 'outer join, indicator=True, validate="one_to_many"', "purple",
     sources=sources_from(students, grades, "student_id"), key="student_id")
audit["_merge"].value_counts()
""")
    b.md(f"""
### 5d. Duplicate keys multiply rows  🎬 [{PAGE}#duplicate-matches]({PAGE}#duplicate-matches)

**Question:** *Why did my row count jump after a join?*
**Setup:** Ben (ID 2) is in two clubs and has two exam scores.
**Expect:** joining on `student_id` gives 2 × 2 = **4** rows for one student — every club paired with every exam.
""")
    b.code("""
clubs       = pd.DataFrame({"student_id": [2, 2], "club": ["Art", "Music"]})
exam_grades = pd.DataFrame({"student_id": [2, 2], "exam": ["E1", "E2"], "score": [80, 90]})
show(clubs, "clubs (LEFT) — key repeated!", "blue", key="student_id")
show(exam_grades, "exam_grades (RIGHT) — key repeated!", "orange", key="student_id")

blowup = clubs.merge(exam_grades, on="student_id")
show(blowup, 'clubs.merge(exam_grades, on="student_id")  — 2 × 2 = 4 rows', "purple",
     sources=sources_from(clubs, exam_grades, "student_id"), key="student_id")

try:
    clubs.merge(exam_grades, on="student_id", validate="one_to_many")
except pd.errors.MergeError as e:
    print("validate caught it →", str(e).splitlines()[0])
""")
    b.md("""
**What happened:** a join produces one output row per *matching pair*. Two left rows × two right rows with the same key = four pairs. Nothing is wrong syntactically, which is why `validate=` exists: it turns a silent multiplication into an error.
""")
    b.md(f"""
### 5e. Both tables have a column with the same name  🎬 [{PAGE}#overlapping-columns-suffixes]({PAGE}#overlapping-columns-suffixes)

**Question:** *How did each student's score change from the midterm to the final?*
**Why it matters:** both tables call their column `score`; pandas must rename one or both. The default suffixes `_x` / `_y` tell you nothing — choose your own.
""")
    b.code("""
midterm = pd.DataFrame({"student_id": [1, 2, 3], "score": [72, 85, 90]})
final   = pd.DataFrame({"student_id": [1, 2, 3], "score": [80, 79, 95]})
show(midterm, "midterm (LEFT)", "blue", key="student_id")
show(final, "final (RIGHT)", "orange", key="student_id")

both = midterm.merge(final, on="student_id", suffixes=("_mid", "_final"))
both["change"] = both["score_final"] - both["score_mid"]
show(both, 'suffixes=("_mid", "_final"), then change = score_final − score_mid', "purple",
     sources={"student_id": "green", "score_mid": "blue", "score_final": "orange", "change": "yellow"}, key="student_id")
""")

    # ---- Concat ----
    b.md(f"""
---
## 6. Concatenation — stacking, not matching

### 6a. Add rows  🎬 [{PAGE}#concat-add-rows]({PAGE}#concat-add-rows)

**Question:** *What is the combined list of scores across fall and spring?*
**Why not a join:** the two tables hold the *same kind* of rows for different periods. There is no key to match; we want them one after another.
**Expect:** 2 + 1 = 3 rows, same columns, a fresh 0-1-2 index.
""")
    b.code("""
fall   = pd.DataFrame({"student": ["Ava", "Ben"], "score": [80, 90]})
spring = pd.DataFrame({"student": ["Cam"], "score": [70]})
show(fall, "fall", "blue")
show(spring, "spring", "orange")

stacked = pd.concat([fall, spring], ignore_index=True)
stacked["term"] = ["fall", "fall", "spring"]
show(stacked, "pd.concat([fall, spring], ignore_index=True)  (+ a term column added afterwards)", "purple",
     sources={"student": "purple", "score": "purple", "term": "yellow"})
""")
    b.md(f"""
**What happened:** rows were appended; columns aligned by *name*. Without `ignore_index=True` the index would read 0, 1, 0. Good practice: add a column saying which piece each row came from (done here as `term`) *before* or right after stacking, so the origin is not lost.

### 6b. Add columns  🎬 [{PAGE}#concat-add-columns]({PAGE}#concat-add-columns)

**Question:** *Two measurements share index labels but not row positions — how do we put them side by side?*
**Why it is a trap:** `axis=1` aligns on the **index labels**, not on row order. Set the index to the identifier first.
""")
    b.code("""
names_idx  = pd.DataFrame({"index": [0, 1], "name": ["Ava", "Ben"]})
scores_idx = pd.DataFrame({"index": [1, 2], "score": [90, 70]})
show(names_idx, "names_idx", "blue", key="index")
show(scores_idx, "scores_idx", "orange", key="index")

side = pd.concat([names_idx.set_index("index"), scores_idx.set_index("index")], axis=1).reset_index()
show(side, "pd.concat([... set_index('index') ...], axis=1)", "purple",
     sources={"index": "green", "name": "blue", "score": "orange"}, key="index")
""")

    # ---- Reshape ----
    b.md(f"""
---
## 7. Reshaping — same data, different shape

### 7a. Pivot: long → wide  🎬 [{PAGE}#pivot-long-wide]({PAGE}#pivot-long-wide)

**Question:** *Did each student improve from exam 1 to exam 2?* — easiest to see with E1 and E2 side by side.
**Expect:** one row per student, one column per exam; `pivot` does **not** aggregate, so (student, exam) pairs must be unique.
""")
    b.code("""
exam_long = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Ben"], "exam": ["E1", "E2", "E1", "E2"],
                          "score": [80, 90, 70, 100]})
show(exam_long, "exam_long (one row per student × exam)", "blue", key=["student", "exam"])

wide = exam_long.pivot(index="student", columns="exam", values="score").reset_index()
wide.columns.name = None
show(wide, 'exam_long.pivot(index="student", columns="exam", values="score")', "purple",
     sources={"student": "green", "E1": "yellow", "E2": "yellow"}, key="student")
""")
    b.md(f"""
### 7b. Pivot table: aggregate while reshaping  🎬 [{PAGE}#pivot-table-aggregate]({PAGE}#pivot-table-aggregate)

**Question:** *Ava retook E1 — what is each student's average per exam?* Duplicated pairs make `pivot` raise; `pivot_table` summarizes them.
""")
    b.code("""
exam_dup = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Ben", "Ava"], "exam": ["E1", "E2", "E1", "E2", "E1"],
                         "score": [80, 90, 70, 100, 100]})
show(exam_dup, "exam_dup — (Ava, E1) appears twice", "blue", key=["student", "exam"])

try:
    exam_dup.pivot(index="student", columns="exam", values="score")
except ValueError as e:
    print("pivot →", e)

pt = exam_dup.pivot_table(index="student", columns="exam", values="score", aggfunc="mean").reset_index()
pt.columns.name = None
show(pt, 'pivot_table(..., aggfunc="mean")  — Ava E1 = mean(80, 100) = 90', "purple",
     sources={"student": "green", "E1": "yellow", "E2": "yellow"}, key="student")
""")
    b.md(f"""
### 7c. Melt: wide → long  🎬 [{PAGE}#melt-wide-long]({PAGE}#melt-wide-long)

**Question:** *How do we plot every exam score as one series, coloured by exam?* Plotting libraries want *long* data: one row per observation.
""")
    b.code("""
show(wide, "wide (input)", "blue", key="student")
long_again = wide.melt(id_vars="student", var_name="exam", value_name="score")
show(long_again, 'wide.melt(id_vars="student", var_name="exam", value_name="score")', "purple",
     sources={"student": "green", "exam": "yellow", "score": "yellow"}, key="student")
""")
    b.md(f"""
### 7d. Drop duplicates — making a key unique before a join  🎬 [{PAGE}#drop-duplicates]({PAGE}#drop-duplicates)

**Question:** *Is this roster safe to use as a lookup table?* A lookup must have each key **once**, otherwise Section 5d happens.
""")
    b.code("""
roster = pd.DataFrame({"student_id": [1, 2, 2, 3, 1], "name": ["Ava", "Ben", "Ben", "Cam", "Ava"]})
show(roster, "roster — IDs 1 and 2 repeated", "blue", key="student_id")
show(roster.drop_duplicates(subset="student_id"), 'roster.drop_duplicates(subset="student_id")  (first occurrence kept)',
     "purple", sources={"student_id": "green", "name": "blue"}, key="student_id")
""")
    b.md(f"""
### 7e. Crosstab — counting pairs of categories  🎬 [{PAGE}#crosstab-counts]({PAGE}#crosstab-counts)

**Question:** *How many times is each student enrolled in each course?* A crosstab is a pivot table whose cells are counts.
""")
    b.code("""
enroll2 = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Cam", "Cam"],
                        "course": ["Stats", "Python", "Stats", "Python", "Stats"]})
show(enroll2, "enroll2", "blue", key=["student", "course"])
ct = pd.crosstab(enroll2["student"], enroll2["course"]).reset_index()
ct.columns.name = None
show(ct, 'pd.crosstab(enroll2["student"], enroll2["course"])', "purple",
     sources={"student": "green", "Python": "yellow", "Stats": "yellow"}, key="student")
""")

    # ---- Real data ----
    b.md(f"""
---
## 8. Real data: why we join Kobe Bryant's game log to a team table

**The question:** *Which division did Kobe score the most against, and which did he beat most often?*

**Why one table cannot answer it:** the game log has one row per game with the opponent as a three-letter abbreviation — no division anywhere. The team table has divisions but no games. The abbreviation is the only bridge, so it is the key.

**Expect:** a left join keeps all 1,103 games (we must not lose games); each game gains `team`, `conference`, `division`. Any `NaN` in those columns means an abbreviation the team table does not know.
""")
    b.code(f"""
DATA = "{DATA}"
games = pd.read_csv(DATA + "kobe_bryant_games.csv")
teams = pd.read_csv(DATA + "nba_teams.csv")
moves = pd.read_csv(DATA + "franchise_moves.csv")

game_cols = ["season_label", "date", "location", "opponent", "outcome", "points"]
show(games[game_cols], "games (LEFT) — selected columns; key is opponent", "blue", key="opponent")
show(teams, "teams (RIGHT) — key is abbr", "orange", key="abbr")
""")
    b.md("""
### 8a. The naive join — and the two numbers to check immediately

After **any** join: (1) did the row count change? (2) how many right-side values are missing?
""")
    b.code("""
joined = games.merge(teams, left_on="opponent", right_on="abbr", how="left")

print("rows before:", len(games), "  rows after:", len(joined), "  ← unchanged, so no duplicate keys in teams")
print("games with no team match:", joined["team"].isna().sum())

show(joined[game_cols + ["abbr", "team", "conference", "division"]].sort_values("team", na_position="first"),
     "games ⟕ teams (left join) — unmatched games first", "purple",
     sources={**{c: "blue" for c in game_cols}, "opponent": "green", "abbr": "green",
              "team": "orange", "conference": "orange", "division": "orange"}, key=["opponent", "abbr"])
""")
    b.md("""
**What happened:** 138 games came back with `NaN` in every team column. **Question: which opponents failed to match?** That is a left anti join on the key.
""")
    b.code("""
unmatched = games[~games["opponent"].isin(teams["abbr"])]["opponent"].value_counts().reset_index()
unmatched.columns = ["opponent", "games"]
show(unmatched, "opponent abbreviations with NO row in teams (left anti join)", "purple",
     sources={"opponent": "green", "games": "yellow"}, key="opponent")

never_faced = teams[~teams["abbr"].isin(games["opponent"])]
show(never_faced, "teams Kobe never faced under these keys (right anti join)", "purple",
     sources={"abbr": "green", "team": "orange", "city": "orange", "conference": "orange", "division": "orange"}, key="abbr")
""")
    b.md("""
**Interpretation:** every unmatched abbreviation is a franchise that moved or was renamed after Kobe played it (Seattle → Oklahoma City, New Jersey → Brooklyn, Vancouver → Memphis, the Hornets/Bobcats/Pelicans saga). The right anti join shows the *new* names of those same franchises — plus LAL, Kobe's own team, which he never played against. The keys are not wrong; they are from a different era than the lookup. **This is the everyday reality of joins: keys drift.**

### 8b. Fix the keys with a second join, then validate

**Question:** *How do we translate old abbreviations into current ones without touching games that already match?*
**Plan:** left-join the `moves` table (old → current) onto games; where a current abbreviation exists use it, otherwise keep the original. `fillna` does the "otherwise".
""")
    b.code("""
show(moves[["old_abbr", "old_name", "current_abbr"]], "moves — the translation table (key: old_abbr)", "orange", key="old_abbr", rows=8)

step = games.merge(moves[["old_abbr", "current_abbr"]], left_on="opponent", right_on="old_abbr", how="left")
step["opp_current"] = step["current_abbr"].fillna(step["opponent"])
show(step[["season_label", "opponent", "old_abbr", "current_abbr", "opp_current"]].drop_duplicates("opponent").sort_values("opponent"),
     "one row per opponent: how opp_current was built", "purple",
     sources={"season_label": "blue", "opponent": "green", "old_abbr": "green", "current_abbr": "orange", "opp_current": "yellow"},
     key=["opponent", "old_abbr"], rows=34)

games = step.drop(columns=["old_abbr", "current_abbr"])
""")
    b.code("""
joined = games.merge(teams, left_on="opp_current", right_on="abbr", how="left",
                     validate="many_to_one", indicator=True)
print(joined["_merge"].value_counts().to_string())
""")
    b.md("""
**What happened:** `validate="many_to_one"` asserted that `abbr` is unique in `teams` (it is) and `indicator=True` proved every game is now `both`. The question we started with is now one `groupby` away:
""")
    b.code("""
by_division = (joined.groupby("division")
                     .agg(games=("points", "size"), avg_points=("points", "mean"),
                          win_share=("outcome", lambda s: (s == "W").mean()))
                     .round(3).sort_values("avg_points", ascending=False).reset_index())
show(by_division, "points and win share by opponent division — the answer", "purple",
     sources={"division": "green", "games": "yellow", "avg_points": "yellow", "win_share": "yellow"}, key="division")
""")
    b.md("""
**Answer:** Kobe scored most against the Pacific division (his own, 26.2 per game, 73.5% wins) and least against the Central (24.1). The Southeast was his toughest by win share (60.9%). Neither number existed in either table alone — the join created the possibility of asking.

### 8c. The duplicate trap on real data

**Question:** *What if someone joins franchise history onto games by the current abbreviation?* `moves` has **three** rows for NOP (CHH, NOH, NOK). **Expect:** games against that franchise tripled.
""")
    b.code("""
history = games.merge(moves, left_on="opp_current", right_on="current_abbr", how="inner")
print("games vs. a relocated franchise:", games["opp_current"].isin(moves["current_abbr"]).sum())
print("rows after the join:            ", len(history))

nop = history[history["opp_current"] == "NOP"][["season_label", "date", "opponent", "opp_current", "old_abbr", "old_name"]]
show(nop, "one game vs. NOP became three rows", "purple",
     sources={"season_label": "blue", "date": "blue", "opponent": "blue", "opp_current": "green",
              "old_abbr": "orange", "old_name": "orange"}, key="opp_current", rows=6)

try:
    games.merge(moves, left_on="opp_current", right_on="current_abbr", how="inner", validate="many_to_one")
except pd.errors.MergeError:
    print('validate="many_to_one" raised MergeError — exactly what we want')
""")
    b.md("""
### 8d. Suffixes and pivots on real data: home vs. away

**Question:** *Did Kobe score more at home than on the road, season by season?*
**Why a join (and then why not):** two grouped summaries — home points and away points per season — share the column name `points`, so joining them needs suffixes. Then notice the same table is really a *pivot* of one summary.
""")
    b.code("""
home = games[games["location"] == "Home"].groupby("season_label")["points"].mean().reset_index()
away = games[games["location"] == "Away"].groupby("season_label")["points"].mean().reset_index()
show(home, "home — mean points per season", "blue", key="season_label", rows=5)
show(away, "away — mean points per season", "orange", key="season_label", rows=5)

home_away = home.merge(away, on="season_label", suffixes=("_home", "_away"))
home_away["home_edge"] = (home_away["points_home"] - home_away["points_away"]).round(2)
show(home_away, 'home.merge(away, on="season_label", suffixes=("_home", "_away")) + home_edge', "purple",
     sources={"season_label": "green", "points_home": "blue", "points_away": "orange", "home_edge": "yellow"},
     key="season_label", rows=15)
""")
    b.code("""
pt = games.pivot_table(index="season_label", columns="location", values="points", aggfunc="mean").round(1).reset_index()
pt.columns.name = None
show(pt, 'games.pivot_table(index="season_label", columns="location", values="points")  — same table, no join', "purple",
     sources={"season_label": "green", "Away": "yellow", "Home": "yellow"}, key="season_label", rows=15)
""")

    b.md(f"""
---
## 9. Summary: the question decides the join

| The question sounds like… | You need | Because |
|---|---|---|
| "…for the ones that appear in both" | inner join | unmatched rows are noise |
| "…keep all of *these*, add info where it exists" | left join | the left table is the population; `NaN` = "no info" |
| "…which of these have no counterpart?" | left anti join | the *absence* of a match is the answer |
| "…audit both sources" | outer join + `indicator` | you need to see every row and where it came from |
| "…every combination of" | cross join | there is no key; you want all pairs |
| "same kind of rows, more of them" | `concat` (rows) | stacking, not matching |
| "side by side per period / category" | `pivot` / `pivot_table` | reshape one table instead of joining two |
| "my row count changed" | `validate=`, `drop_duplicates` | a duplicated key multiplied rows |

Watch any of these on the interactive page: [{PAGE}]({PAGE}).
""")
    return b


# ============================================================================ #
#  2. PA 3.1 — extended (MovieLens)
# ============================================================================ #

def build_pa31() -> Book:
    b = Book()
    b.md("""
# GSB 5544 — PA 3.1 (Extended): Concatenating, Joining, and Pivoting — every step explained
*The MovieLens 1M answers broken into steps, with the logic for each step and what each function does.*
""")
    b.md(f"""
## How to use this notebook

Each question from PA 3.1 is worked in **numbered steps**. Every step has three parts: **the logic** (why this step, in words), **the code**, and **the table it produces**. (The interactive page shows the join operations with columns coloured by the table they came from.) Functions are explained the first time they appear in a *📘 Function* note. The compact answer key is on the course site: [{SITE}]({SITE}); the interactive join page is [{PAGE}]({PAGE}).
""")
    add_helper(b)

    # ---- Q1 ----
    b.md("""
---
## Question 1 — Read in each of the data files

**Logic.** Three files describe one thing each: movies, ratings, users. They are linked by ID columns (`MovieID`, `UserID`) — the keys we will join on later. Before reading, look at the README: it tells us the separator is `::`, there is **no header row**, and what the columns are.

**Step 1.1 — read the README's format lines.** `UserID::MovieID::Rating::Timestamp`, `UserID::Gender::Age::Occupation::Zip-code`, `MovieID::Title::Genres`.

📘 **Function: `pd.read_csv(path, sep, engine, header, names, encoding_errors)`**
- `sep="::"` — the field separator. It is two characters, and the fast C parser only handles one, so we also pass `engine="python"`.
- `header=None` — the first line is data, not column names.
- `names=[...]` — the column names to use (from the README).
- `encoding_errors="ignore"` — `movies.dat` has a few bytes that are not valid UTF-8 (accented titles); ignore them instead of crashing.
""")
    b.code("""
base = "https://dlsun.github.io/pods/data/ml-1m/"

movies = pd.read_csv(base + "movies.dat", sep="::", engine="python", header=None,
                     names=["MovieID", "Title", "Genres"], encoding_errors="ignore")
show(movies, "movies — one row per movie (key: MovieID)", "blue", key="MovieID", rows=5)
""")
    b.code("""
ratings = pd.read_csv(base + "ratings.dat", sep="::", engine="python", header=None,
                      names=["UserID", "MovieID", "Rating", "Timestamp"])
show(ratings, "ratings — one row per (user, movie) rating (keys: UserID, MovieID)", "orange", key=["UserID", "MovieID"], rows=5)
""")
    b.code("""
users = pd.read_csv(base + "users.dat", sep="::", engine="python", header=None,
                    names=["UserID", "Gender", "Age", "Occupation", "Zip"])
show(users, "users — one row per user (key: UserID)", "green", key="UserID", rows=5)
""")
    b.md("""
**Step 1.2 — sanity-check what we read.** Shapes should match the README (≈3,900 movies, 1,000,209 ratings, 6,040 users), and the ID columns should be integers, otherwise the joins will silently fail to match (`"1"` ≠ `1`).
""")
    b.code("""
print("movies:", movies.shape, " ratings:", ratings.shape, " users:", users.shape)
print()
print(ratings.dtypes.to_string())
""")
    b.md("""
**Step 1.3 — the data model.** `ratings` is the *fact* table (one row per event); `movies` and `users` are *lookup* tables (one row per entity). Almost every question below is: summarize `ratings`, then join a lookup on to label the result.

```
users (UserID) ──< ratings (UserID, MovieID) >── movies (MovieID)
```
""")

    # ---- Q2 ----
    b.md("""
---
## Question 2 — Which age group tends to give the highest ratings?

**Logic.** The rating is in `ratings`; the rater's age is in `users`. Different tables → join on `UserID`. Then group by age and average. Two details: (a) every rating belongs to exactly one user, so the join is many-to-one and must not change the number of rows; (b) `Age` is a *code* (1, 18, 25, …) that stands for a range — translate it before presenting.

**Step 2.1 — join ratings to users.**

📘 **Function: `left.merge(right, on, how, validate)`**
- `on="UserID"` — the column to match on (same name in both tables).
- `how="left"` — keep every rating even if a user were missing from `users` (none are, but the habit is good).
- `validate="many_to_one"` — assert that `UserID` is unique in `users`; pandas raises `MergeError` if not. This is the guard against accidental row multiplication.
""")
    b.code("""
ratings_users = ratings.merge(users, on="UserID", how="left", validate="many_to_one")
print("rows before:", len(ratings), " rows after:", len(ratings_users), " (must be equal)")
show(ratings_users, "ratings ⟕ users", "purple", sources=sources_from(ratings, users, "UserID"), key="UserID", rows=5)
""")
    b.md("""
**Step 2.2 — translate the age codes.**

📘 **Function: `Series.map(dict)`** — replaces each value by looking it up in the dictionary; values not in the dictionary become `NaN` (a quick way to spot an unexpected code).
""")
    b.code("""
age_labels = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
ratings_users["AgeGroup"] = ratings_users["Age"].map(age_labels)

show(ratings_users[["UserID", "Age", "AgeGroup", "Rating"]].drop_duplicates("Age").sort_values("Age"),
     "each Age code and its label (one example row per code)", "purple",
     sources={"UserID": "blue", "Age": "orange", "AgeGroup": "yellow", "Rating": "blue"}, key="Age")
print("unmapped codes:", ratings_users["AgeGroup"].isna().sum())
""")
    b.md("""
**Step 2.3 — group and summarize.**

📘 **Function: `df.groupby(col)[value].agg(name=stat, ...)`** — splits the rows into one group per distinct value of `col`, computes each statistic on the `value` column within the group, and returns one row per group with your chosen column names ("named aggregation"). `size` counts rows (including missing values); `mean` averages.
""")
    b.code("""
by_age = (ratings_users.groupby("AgeGroup")["Rating"]
                       .agg(n_ratings="size", mean_rating="mean")
                       .round(3)
                       .sort_values("mean_rating", ascending=False)
                       .reset_index())
show(by_age, "mean rating by age group (sorted)", "purple",
     sources={"AgeGroup": "green", "n_ratings": "yellow", "mean_rating": "yellow"}, key="AgeGroup")
""")
    b.md("""
**Answer.** Ratings rise steadily with age: **56+** is the most generous group (≈3.77) and **18–24** the harshest (≈3.51). The gaps are small, but each group has tens of thousands of ratings, so they are not noise. Note the join did no computing at all — it only made `Age` *available* next to `Rating`.
""")

    # ---- Q3 ----
    b.md("""
---
## Question 3 — Among movies with ≥ 100 ratings, which have the highest and lowest average rating?

**Logic.** "Per movie" means group `ratings` by `MovieID`. We need two statistics per movie — how many ratings (to apply the 100 floor) and the mean (to rank). Titles live in `movies`, so join them on afterwards. Order matters: **count first, then filter, then rank**; filtering the raw ratings would throw away the counts.

**Step 3.1 — one row per movie with count and mean.**

📘 **`reset_index()`** — after `groupby`, the group labels become the *index*. `reset_index()` moves them back into an ordinary column so we can `merge` on it.
""")
    b.code("""
movie_stats = (ratings.groupby("MovieID")["Rating"]
                      .agg(n_ratings="size", avg_rating="mean")
                      .reset_index())
show(movie_stats, "one row per rated movie", "orange", key="MovieID", rows=5)
""")
    b.md("""
**Step 3.2 — attach titles.** `movie_stats` has one row per movie and so does `movies`: `validate="one_to_one"` documents that and checks it.
""")
    b.code("""
movie_stats = movie_stats.merge(movies, on="MovieID", how="left", validate="one_to_one")
show(movie_stats, "movie_stats ⟕ movies", "purple",
     sources={"MovieID": "green", "n_ratings": "orange", "avg_rating": "orange", "Title": "blue", "Genres": "blue"},
     key="MovieID", rows=5)
""")
    b.md("""
**Step 3.3 — filter to popular movies, then sort both ways.**

📘 **Boolean filtering `df[condition]`** keeps rows where the condition is True. 📘 **`sort_values(col, ascending=)`** orders rows; `head(10)` shows the top of that order.
""")
    b.code("""
popular = movie_stats[movie_stats["n_ratings"] >= 100]
print(f"{len(popular):,} of {len(movie_stats):,} rated movies have at least 100 ratings")

show(popular.sort_values("avg_rating", ascending=False)[["Title", "n_ratings", "avg_rating"]].round(3),
     "HIGHEST average rating (≥ 100 ratings)", "purple",
     sources={"Title": "blue", "n_ratings": "orange", "avg_rating": "orange"}, rows=10)
show(popular.sort_values("avg_rating")[["Title", "n_ratings", "avg_rating"]].round(3),
     "LOWEST average rating (≥ 100 ratings)", "purple",
     sources={"Title": "blue", "n_ratings": "orange", "avg_rating": "orange"}, rows=10)
""")
    b.md("""
**Answer.** Highest: *Seven Samurai* (4.56), *The Shawshank Redemption* (4.55), *The Godfather* (4.52), *A Close Shave*, *The Usual Suspects*. Lowest: *Kazaam* (1.47), *Battlefield Earth* (1.61), *Pokémon the Movie 2000*, *Aces: Iron Eagle III*, *Police Academy 6*. **Why the floor matters:** remove it and the extremes become movies rated by one or two people — a 5.0 from a single fan is not "the best movie".
""")

    # ---- Q4 ----
    b.md("""
---
## Question 4 — Average rating vs. share of 18–24 raters, per movie

**Logic.** Both quantities are per-movie summaries of the *joined* ratings–users table from Question 2. The "share of ratings from 18–24 users" is a proportion, and a proportion is the **mean of a 0/1 indicator** — so create the indicator, then one `groupby().agg()` computes the mean rating, the share, and the count together.

**Step 4.1 — the indicator column.** `(Age == 18)` is True/False; `.astype(int)` makes it 1/0 so its mean is a share.
""")
    b.code("""
ratings_users["is_18_24"] = (ratings_users["Age"] == 18).astype(int)
show(ratings_users[["UserID", "MovieID", "Rating", "Age", "is_18_24"]], "indicator added", "purple",
     sources={"UserID": "blue", "MovieID": "blue", "Rating": "blue", "Age": "orange", "is_18_24": "yellow"}, rows=5)
print("overall share of ratings from 18-24 users:", ratings_users["is_18_24"].mean().round(3))
""")
    b.md("""
**Step 4.2 — three summaries per movie, then titles.** Named aggregation with tuples: `avg_rating=("Rating", "mean")` means *apply `mean` to column `Rating` and call the result `avg_rating`*.
""")
    b.code("""
by_movie = (ratings_users.groupby("MovieID")
                         .agg(avg_rating=("Rating", "mean"),
                              share_18_24=("is_18_24", "mean"),
                              n_ratings=("Rating", "size"))
                         .reset_index()
                         .merge(movies, on="MovieID", how="left", validate="one_to_one"))
show(by_movie, "per-movie summary ⟕ titles", "purple",
     sources={"MovieID": "green", "avg_rating": "yellow", "share_18_24": "yellow", "n_ratings": "yellow",
              "Title": "blue", "Genres": "blue"}, key="MovieID", rows=5)
""")
    b.md("""
**Step 4.3 — plot.** In plotnine, `aes(x=, y=, size=)` maps columns to position and point size; `geom_point(alpha=)` draws translucent points so dense regions show through.
""")
    b.code("""
from plotnine import *

(ggplot(by_movie, aes(x="share_18_24", y="avg_rating", size="n_ratings"))
 + geom_point(alpha=0.25)
 + labs(x="Share of ratings from users aged 18-24", y="Average rating",
        size="Number of ratings", title="MovieLens: each point is one movie")
)
""")
    b.code("""
by_movie[["avg_rating", "share_18_24"]].corr().round(3)
""")
    b.md("""
**Answer.** A moderate **negative** relationship (r ≈ −0.35): movies whose audience skews young tend to be rated lower — consistent with Question 2, where 18–24-year-olds were the harshest raters, and with the fact that they watch different films. The big points (heavily rated movies) sit at a 10–30% share with ratings 3–4.5; the points at 0% or 100% are movies with a handful of ratings.
""")

    # ---- Q5 ----
    b.md("""
---
## Question 5 — Number of ratings by movie; how many movies had zero ratings?

**Logic.** Counting ratings per movie only produces rows for movies that *have* ratings. A movie with zero ratings does not appear in `ratings` at all, so it cannot appear in the count. To find it we must start from `movies` (the complete list) and **left-join** the counts: unrated movies survive with `NaN`, which we then read as 0. An **inner** join would drop exactly the rows we are looking for.

**Step 5.1 — count per movie.**

📘 **`groupby(col).size()`** returns a Series of row counts per group; **`.rename("n_ratings")`** names the Series so it becomes a sensibly named column after `reset_index()`.
""")
    b.code("""
counts = ratings.groupby("MovieID").size().rename("n_ratings").reset_index()
show(counts, "counts — one row per RATED movie", "orange", key="MovieID", rows=5)
print("rated movies:", len(counts), "   movies in the catalogue:", len(movies))
""")
    b.md("""
**Step 5.2 — inner vs. left join, side by side.**
""")
    b.code("""
inner = movies.merge(counts, on="MovieID", how="inner")
left  = movies.merge(counts, on="MovieID", how="left", validate="one_to_one")
print("inner join rows:", len(inner), "   left join rows:", len(left))

show(left.sort_values("n_ratings", na_position="first"), "movies ⟕ counts — unrated movies first (pink = no count)", "purple",
     sources=sources_from(movies, counts, "MovieID"), key="MovieID", rows=6)
""")
    b.md("""
**Step 5.3 — turn the missing counts into zeros and count them.**

📘 **`fillna(0)`** replaces `NaN` with 0; **`astype(int)`** converts the float column (float only because of the `NaN`s) back to integers.
""")
    b.code("""
left["n_ratings"] = left["n_ratings"].fillna(0).astype(int)
n_zero = (left["n_ratings"] == 0).sum()
print("movies with zero ratings:", n_zero, "  (= catalogue − rated =", len(movies), "−", len(counts), ")")
""")
    b.md("""
**Step 5.4 — cross-check with a left anti join.** The unrated movies are the catalogue rows whose `MovieID` is *not in* `ratings`.
""")
    b.code("""
unrated = movies[~movies["MovieID"].isin(ratings["MovieID"])]
print("left anti join count:", len(unrated))
show(unrated, "movies with no ratings at all", "purple", sources={"MovieID": "green", "Title": "blue", "Genres": "blue"},
     key="MovieID", rows=8)
""")
    b.md("""
**Answer.** **177 movies** have zero ratings (3,883 listed − 3,706 rated). The inner join returns 3,706 rows and can never show them; the left join keeps every movie and flags the 177 with `NaN`. That is the whole point of the hint: *which* join you pick decides whether the answer is even visible.
""")

    # ---- Q6 ----
    b.md("""
---
## Question 6 — Movies that received both a 1 and a 5

**Logic.** "Received a 1" and "received a 5" are two separate facts about a movie. Build a small table for each fact — one row per movie that has at least one 1-star rating (with the count), and likewise for 5-star — then **inner-join** them: a movie survives only if it is in *both* tables, which is exactly "received both". The two count columns answer "how many of each type".

**Step 6.1 — the 1-star table.** Filter, count per movie, name the count.
""")
    b.code("""
ones = ratings[ratings["Rating"] == 1].groupby("MovieID").size().rename("n_1").reset_index()
show(ones, "ones — movies with ≥ 1 one-star rating", "blue", key="MovieID", rows=5)
""")
    b.md("**Step 6.2 — the 5-star table, the same way.**")
    b.code("""
fives = ratings[ratings["Rating"] == 5].groupby("MovieID").size().rename("n_5").reset_index()
show(fives, "fives — movies with ≥ 1 five-star rating", "orange", key="MovieID", rows=5)
print("movies with a 1:", len(ones), "  movies with a 5:", len(fives))
""")
    b.md("""
**Step 6.3 — inner join = intersection.** `validate="one_to_one"`: each movie appears once in each table, so the join cannot multiply rows.
""")
    b.code("""
both = ones.merge(fives, on="MovieID", how="inner", validate="one_to_one")
print("movies with BOTH a 1 and a 5:", len(both))
show(both, "ones ⋈ fives (inner)", "purple", sources=sources_from(ones, fives, "MovieID"), key="MovieID", rows=5)
""")
    b.md("**Step 6.4 — attach titles and look at the extremes.**")
    b.code("""
both = both.merge(movies, on="MovieID", how="left", validate="one_to_one")
cols = {"MovieID": "green", "n_1": "blue", "n_5": "orange", "Title": "yellow", "Genres": "yellow"}

show(both.sort_values("n_1", ascending=False)[["Title", "n_1", "n_5"]], "most one-star ratings", "purple", sources=cols, rows=8)

both["min_count"] = both[["n_1", "n_5"]].min(axis=1)
show(both.sort_values("min_count", ascending=False)[["Title", "n_1", "n_5"]],
     "most POLARIZING — many 1s AND many 5s (sorted by the smaller of the two)", "purple", sources=cols, rows=8)
""")
    b.md("""
**Step 6.5 — the same answer by pivoting instead of joining.**

📘 **`pd.crosstab(rows, cols)`** builds a table with one row per distinct value of the first argument, one column per distinct value of the second, and counts in the cells. It is a pivot table whose values are counts. A movie has "both" when its `1` column and its `5` column are both positive.
""")
    b.code("""
star_counts = pd.crosstab(ratings["MovieID"], ratings["Rating"])
star_counts.columns.name = None
show(star_counts.reset_index(), "crosstab: MovieID × Rating (counts)", "purple",
     sources={"MovieID": "green", 1: "yellow", 2: "yellow", 3: "yellow", 4: "yellow", 5: "yellow"}, key="MovieID", rows=5)
print("movies with both a 1 and a 5 (via crosstab):", ((star_counts[1] > 0) & (star_counts[5] > 0)).sum())
""")
    b.md("""
**Answer.** **2,986** of the 3,706 rated movies received at least one 1 *and* at least one 5. The joined `n_1` / `n_5` columns give the count of each: *Wild Wild West* has 314 ones and 16 fives, while *The Blair Witch Project* (219 vs 180) and *Mars Attacks!* (163 vs 84) are the genuinely polarizing ones. The crosstab reaches the same 2,986 — a reminder that a join between two summaries and a pivot of one summary are often the same idea in different shapes.
""")
    return b


# ============================================================================ #
#  3. PA 3.2 — extended (distances)
# ============================================================================ #

def build_pa32() -> Book:
    b = Book()
    b.md("""
# GSB 5544 — PA 3.2 (Extended): Distances Between Observations — every step explained
*Finding "similar" houses and colleges: each answer broken into steps, with the reasoning and the functions.*
""")
    b.md(f"""
## How to use this notebook

"Similar" has to be made precise before a computer can find it: **which variables**, **on what scale**, and **which distance formula**. Every question below is worked in numbered steps that make each of those decisions explicit, show the table before and after each transformation, and end with an interpretation. Functions are explained in *📘 Function* notes the first time they appear. The compact answer key is on the course site: [{SITE}]({SITE}).

**The one idea behind everything here.** For a target row *t* and any other row *i*, put both on the same scale and compute
- Euclidean distance: $d(i,t)=\\sqrt{{\\sum_j (x_{{ij}}-x_{{tj}})^2}}$ — straight-line distance;
- Manhattan distance: $d(i,t)=\\sum_j |x_{{ij}}-x_{{tj}}|$ — sum of absolute differences.

Small distance = similar. Everything else is about choosing and preparing the $x$'s.
""")
    add_helper(b)

    # ---- Ames setup ----
    b.md("""
---
## Ames — recommending similar (but cheaper) homes

**Step 0 — load and look.** The Ames file is tab-separated (`sep="\\t"`). House 0 is the first row.
""")
    b.code("""
df_ames = pd.read_csv("https://dlsun.github.io/pods/data/AmesHousing.txt", sep="\\t")
print(df_ames.shape)

look = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath", "House Style", "Neighborhood", "Year Built", "SalePrice"]
show(df_ames[look], "df_ames — selected columns (house 0 is the first row)", "blue", rows=5)
""")

    # ---- Ames Q1 ----
    b.md("""
---
## Ames 1 — Cheaper homes similar to house 0 on living area, bedrooms, bathrooms

**Logic.** Four decisions: (1) *which variables* — the question names them; bathrooms come in two columns, so combine them; (2) *scale* — square feet are in the thousands, bedrooms in ones, so without scaling the distance is just "difference in square feet"; (3) *distance formula* — try two and see if it matters; (4) *the price constraint* — "cheaper" is a filter applied after computing distances, not a variable inside them.

**Step 1.1 — build the bathrooms variable and look at house 0.** A half bath counts as half.
""")
    b.code("""
df_ames["Bathrooms"] = df_ames["Full Bath"] + 0.5 * df_ames["Half Bath"]
house0 = df_ames.loc[0]

show_cols = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "Year Built", "SalePrice"]
show(df_ames.loc[[0], show_cols], "house 0 — the target", "green")
""")
    b.md("""
**Step 1.2 — select the features and see why scaling is needed.** Compare the standard deviations: a "typical" difference in living area is ~500 sq ft, in bedrooms < 1.
""")
    b.code("""
features = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]
X = df_ames[features].astype(float)

show(X.describe().loc[["mean", "std", "min", "max"]].round(2).reset_index().rename(columns={"index": "stat"}),
     "the three features on their ORIGINAL scales", "blue")
""")
    b.md("""
**Step 1.3 — standardize.** For each column, subtract its mean and divide by its standard deviation (a **z-score**). Afterwards every column has mean 0 and SD 1, so "one unit" means "one standard deviation" in every column.

📘 **Broadcasting:** `X - X.mean()` subtracts each column's mean from every value in that column; `/ X.std()` divides likewise. No loop needed.
""")
    b.code("""
X_z = (X - X.mean()) / X.std()
show(X_z, "X_z — standardized features (mean 0, SD 1 per column)", "purple",
     sources={c: "yellow" for c in features}, rows=5)
print(X_z.mean().round(3).to_dict(), X_z.std().round(3).to_dict())
""")
    b.md("""
**Step 1.4 — distance from house 0.** `X_z - X_z.loc[0]` subtracts house 0's row from *every* row (broadcasting again). Then Euclidean = square, sum across columns (`axis=1`), square-root; Manhattan = absolute value, sum across columns.
""")
    b.code("""
diff = X_z - X_z.loc[0]
df_ames["dist_euclid"]    = np.sqrt((diff ** 2).sum(axis=1))
df_ames["dist_manhattan"] = diff.abs().sum(axis=1)

show(pd.concat([X_z, diff.add_prefix("Δ "), df_ames[["dist_euclid", "dist_manhattan"]]], axis=1),
     "standardized values → differences from house 0 → distances", "purple",
     sources={**{c: "blue" for c in features}, **{"Δ " + c: "orange" for c in features},
              "dist_euclid": "yellow", "dist_manhattan": "yellow"}, rows=5)
""")
    b.md("""
**Step 1.5 — apply the price constraint, then sort.** Keep houses cheaper than house 0; the smallest distances are the recommendations.
""")
    b.code("""
cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
print(f"{len(cheaper):,} of {len(df_ames):,} houses are cheaper than house 0 (${house0['SalePrice']:,})")

show(cheaper.sort_values("dist_euclid")[show_cols + ["dist_euclid"]].round(3),
     "10 nearest CHEAPER houses — Euclidean, standardized", "purple",
     sources={**{c: "blue" for c in show_cols}, "dist_euclid": "yellow"}, rows=10)
show(cheaper.sort_values("dist_manhattan")[show_cols + ["dist_manhattan"]].round(3),
     "10 nearest CHEAPER houses — Manhattan, standardized", "purple",
     sources={**{c: "blue" for c in show_cols}, "dist_manhattan": "yellow"}, rows=10)
""")
    b.md("""
**Step 1.6 — sensitivity: wrap the recipe in a function and vary the choices.** A function lets us change one decision at a time (scaling: z-score / min-max / none; metric: Euclidean / Manhattan) and compare the top-5 lists.

📘 **Min-max scaling** maps each column to 0–1: `(X - X.min()) / (X.max() - X.min())`. It is an alternative to z-scores; both remove the units problem.
""")
    b.code("""
def nearest_cheaper(features, i=0, metric="euclidean", scaling="z", k=5):
    \"\"\"Indices of the k houses nearest house i on `features`, among houses cheaper than house i.\"\"\"
    X = df_ames[features].astype(float)
    if scaling == "z":
        X = (X - X.mean()) / X.std()
    elif scaling == "minmax":
        X = (X - X.min()) / (X.max() - X.min())
    diff = X - X.loc[i]
    dist = np.sqrt((diff ** 2).sum(axis=1)) if metric == "euclidean" else diff.abs().sum(axis=1)
    dist = dist[df_ames["SalePrice"] < df_ames.loc[i, "SalePrice"]]
    return dist.sort_values().head(k).index.tolist()

rows = []
for scaling in ["z", "minmax", "none"]:
    for metric in ["euclidean", "manhattan"]:
        rows.append({"scaling": scaling, "metric": metric, "nearest 5 (row numbers)": nearest_cheaper(features, scaling=scaling, metric=metric)})
show(pd.DataFrame(rows), "the five nearest cheaper houses under six settings", "purple",
     sources={"scaling": "blue", "metric": "orange", "nearest 5 (row numbers)": "yellow"})
""")
    b.code("""
show(df_ames.loc[nearest_cheaper(features, scaling="none"), show_cols],
     "what the UNSCALED distance picks — matched on square feet only", "purple",
     sources={c: "blue" for c in show_cols})
""")
    b.md("""
**Answer / interpretation.** House 0 is a 1,656 sq ft, 3-bed, 1-bath 1960 ranch in North Ames sold for $215,000. With standardized features the nearest cheaper houses (rows 1226, 1940, 1357, 758, 291 …) are 1,640–1,670 sq ft, 3 bed, 1 bath — several also in North Ames — priced $100k–$165k: sensible "same house, lower price" matches.

- **Metric:** Euclidean and Manhattan return the same five houses in slightly different order → insensitive.
- **z-score vs. min-max:** same five → insensitive.
- **No scaling:** a different list, matched on square footage alone, with 2–4 bedrooms and 1–2 baths → very sensitive. Scaling is the decision that matters.
- **Sale price in the distance?** No. The goal is "like house 0 *but cheaper*". Price is the constraint we filter on; putting it in the distance would pull matches toward houses priced *like* house 0 — the opposite of a good deal.
""")

    # ---- Ames Q2 ----
    b.md("""
---
## Ames 2 — Add House Style (a categorical variable)

**Logic.** You cannot subtract "1Story" from "2Story". The standard fix is **one-hot encoding**: one 0/1 column per category. Two houses with the same style then differ by 0 on all style columns; two with different styles differ by 1 in two columns, which adds √2 ≈ 1.41 to a Euclidean distance — large next to typical z-score differences, so a style mismatch is heavily penalized.

**Step 2.1 — one-hot encode.**

📘 **Function: `pd.get_dummies(series, dtype=float)`** — returns a DataFrame with one column per distinct value, 1.0 where the row has that value and 0.0 elsewhere.
""")
    b.code("""
style_dummies = pd.get_dummies(df_ames["House Style"], dtype=float)
show(pd.concat([df_ames[["House Style"]], style_dummies], axis=1), "House Style → one-hot columns", "purple",
     sources={"House Style": "blue", **{c: "yellow" for c in style_dummies.columns}}, rows=5)
""")
    b.md("""
**Step 2.2 — glue the encoded style onto the standardized numeric features.** `pd.concat(..., axis=1)` puts the columns side by side, aligned on the row index (both frames share `df_ames`'s index, so this is safe).
""")
    b.code("""
X2 = pd.concat([X_z, style_dummies], axis=1)
show(X2, "X2 = 3 standardized numeric columns + 8 style indicators", "purple",
     sources={**{c: "blue" for c in features}, **{c: "yellow" for c in style_dummies.columns}}, rows=5)
""")
    b.md("**Step 2.3 — distance, filter, sort — exactly as before.**")
    b.code("""
diff2 = X2 - X2.loc[0]
df_ames["dist_style"] = np.sqrt((diff2 ** 2).sum(axis=1))

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
show(cheaper.sort_values("dist_style")[show_cols + ["dist_style"]].round(3),
     "10 nearest CHEAPER houses — with House Style", "purple",
     sources={**{c: "blue" for c in show_cols}, "dist_style": "yellow"}, rows=10)
""")
    b.md("**Step 2.4 — what changed? Compare the styles of the top 10 with and without the style variable.**")
    b.code("""
top1 = cheaper.sort_values("dist_euclid").head(10)["House Style"].value_counts()
top2 = cheaper.sort_values("dist_style").head(10)["House Style"].value_counts()
cmp = pd.DataFrame({"part 1 (no style)": top1, "part 2 (with style)": top2}).fillna(0).astype(int).reset_index()
cmp.columns = ["House Style", "part 1 (no style)", "part 2 (with style)"]
show(cmp, "House Style of the ten nearest houses", "purple",
     sources={"House Style": "green", "part 1 (no style)": "blue", "part 2 (with style)": "orange"})
""")
    b.md("""
**Answer / interpretation.** All ten nearest houses are now `1Story` like house 0; in part 1, seven of the ten were 1.5- or 2-story houses that merely matched on size and rooms. Because a mismatch costs √2, the style dummies behave almost like a filter: same style first, then size and rooms. If that is too strict, multiply the dummy columns by a weight below 1 (e.g. 0.5) so style becomes a preference. Metric and scaling still barely matter; **how the categorical variable is encoded** is the decision here.
""")

    # ---- Ames Q3 ----
    b.md("""
---
## Ames 3 — Your own mix of quantitative and categorical variables

**Logic.** With 80 columns the temptation is to use them all; do not. Every added variable gets an equal vote after standardization, so ten near-duplicate basement columns would out-vote living area. Pick a small set that captures what a buyer reacts to — size, age, quality, land, location — with both kinds of variable.

**Chosen:** quantitative `Gr Liv Area`, `Bedroom AbvGr`, `Bathrooms`, `Year Built`, `Overall Qual`, `Lot Area`, `Garage Cars`; categorical `House Style`, `Neighborhood`, `Bldg Type`.

**Step 3.1 — encode categoricals and standardize numerics in one frame.**

📘 **`pd.get_dummies(df, columns=[...])`** — encodes only the listed columns and leaves the others as they are, so no concatenation is needed. 📘 **`fillna(0)`** — `Garage Cars` has one missing value; a `NaN` anywhere makes that house's distance `NaN`, so fill it.
""")
    b.code("""
quant = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "Year Built", "Overall Qual", "Lot Area", "Garage Cars"]
categ = ["House Style", "Neighborhood", "Bldg Type"]

X3 = pd.get_dummies(df_ames[quant + categ], columns=categ, dtype=float)
X3[quant] = (X3[quant] - X3[quant].mean()) / X3[quant].std()
X3 = X3.fillna(0)
print("columns in X3:", X3.shape[1], "(7 numeric +", X3.shape[1] - 7, "indicator columns)")
show(X3.iloc[:, :12], "X3 — first 12 columns (numeric standardized, then indicators)", "purple",
     sources={**{c: "blue" for c in quant}, **{c: "yellow" for c in X3.columns if c not in quant}}, rows=5)
""")
    b.md("**Step 3.2 — distance, filter, sort; then look at house 0 on the same columns.**")
    b.code("""
diff3 = X3 - X3.loc[0]
df_ames["dist_full"] = np.sqrt((diff3 ** 2).sum(axis=1))

show3 = show_cols + ["Overall Qual", "Lot Area", "Garage Cars", "Bldg Type"]
show(df_ames.loc[[0], show3], "house 0 on the chosen variables", "green")

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
show(cheaper.sort_values("dist_full")[show3 + ["dist_full"]].round(3),
     "10 nearest CHEAPER houses — full variable set", "purple",
     sources={**{c: "blue" for c in show3}, "dist_full": "yellow"}, rows=10)
""")
    b.md("""
**Step 3.3 — sensitivity to the variable list.** House 0 sits on an unusually large lot (31,770 sq ft, 99th percentile). Drop `Lot Area` and measure how much of the top 10 survives.
""")
    b.code("""
quant_no_lot = [q for q in quant if q != "Lot Area"]
X4 = pd.get_dummies(df_ames[quant_no_lot + categ], columns=categ, dtype=float)
X4[quant_no_lot] = (X4[quant_no_lot] - X4[quant_no_lot].mean()) / X4[quant_no_lot].std()
X4 = X4.fillna(0)
df_ames["dist_no_lot"] = np.sqrt(((X4 - X4.loc[0]) ** 2).sum(axis=1))

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
a = set(cheaper.sort_values("dist_full").head(10).index)
b_ = set(cheaper.sort_values("dist_no_lot").head(10).index)
print("top-10 overlap with / without Lot Area:", len(a & b_), "of 10")
show(cheaper.sort_values("dist_no_lot")[show3 + ["dist_no_lot"]].round(3),
     "10 nearest CHEAPER houses — without Lot Area", "purple",
     sources={**{c: "blue" for c in show3}, "dist_no_lot": "yellow"}, rows=10)
""")
    b.md("""
**Answer / interpretation.** With the full set, the nearest cheaper houses are one-story single-family homes of 1,470–1,770 sq ft, 3 bedrooms, 1–1.5 baths, quality 5–6, on similarly huge lots, mostly 1950s–70s — the same *kind* of property, which is what the extra variables buy. Sensitivity is now high: removing `Lot Area` alone replaces most of the top ten, because the lot is house 0's most unusual feature and standardization makes "unusual" expensive to match. Lesson: with many variables, *which* variables you include (and their scaling) matters far more than Euclidean vs. Manhattan. Sale price stays out, for the reason in part 1.
""")

    # ---- Colleges ----
    b.md("""
---
## Colleges similar to Cal Poly

**Step 0 — load, index by institution, single out Cal Poly.** Setting the index to the name makes `.loc[school_name]` pull out Cal Poly's row.
""")
    b.code("""
df_college = pd.read_csv("https://datasci112.stanford.edu/data/college_attributes.csv")
df_college.set_index("Institution", inplace=True)

school_name = "California Polytechnic State University-San Luis Obispo"
cp = df_college.loc[school_name]

print(df_college.shape)
show(df_college.loc[[school_name], ["City", "State", "AdmissionRate", "Undergraduates", "CarnegieClassification", "Ownership"]],
     "Cal Poly's row (target)", "green")
""")

    # ---- College Q1 ----
    b.md("""
---
## College 1 — Admission rate and number of undergraduates

**Logic.** Two quantitative variables on wildly different scales: admission rate runs 0–1, undergraduates 5–119,000. Without scaling, a 0.4 difference in admission rate is invisible next to a difference of a few hundred students. Standardize, then Euclidean distance.

**Step 1.1 — see the scales.**
""")
    b.code("""
num = ["AdmissionRate", "Undergraduates"]
A = df_college[num]
show(A.describe().loc[["mean", "std", "min", "max"]].round(3).reset_index().rename(columns={"index": "stat"}),
     "two variables, two scales", "blue")
""")
    b.md("**Step 1.2 — standardize and compute distance to Cal Poly.**")
    b.code("""
A_z = (A - A.mean()) / A.std()
dist1 = np.sqrt(((A_z - A_z.loc[school_name]) ** 2).sum(axis=1))

result1 = (df_college.assign(dist=dist1.round(3))
                     .sort_values("dist")[num + ["CarnegieClassification", "Ownership", "dist"]]
                     .reset_index())
show(result1, "nearest to Cal Poly — standardized Euclidean", "purple",
     sources={"Institution": "green", "AdmissionRate": "blue", "Undergraduates": "blue",
              "CarnegieClassification": "grey", "Ownership": "grey", "dist": "yellow"}, rows=11)
""")
    b.md("**Step 1.3 — the same distance without scaling, for contrast.**")
    b.code("""
dist1_raw = np.sqrt(((A - A.loc[school_name]) ** 2).sum(axis=1))
show(df_college.assign(dist=dist1_raw.round(1)).sort_values("dist")[num + ["dist"]].reset_index(),
     "nearest to Cal Poly — UNSCALED (Undergraduates dominates)", "purple",
     sources={"Institution": "green", "AdmissionRate": "blue", "Undergraduates": "blue", "dist": "yellow"}, rows=6)
""")
    b.md("""
**Answer / interpretation.** Standardized: UC Santa Barbara, DeVry–Illinois, UNC Chapel Hill, Clemson, Virginia, CUNY Hunter, Stony Brook, Boston University — selective-ish schools with 15,000–23,000 undergraduates. Unscaled: Iowa, East Carolina, VCU, Buffalo, Kentucky — schools with ~21,000 undergraduates and *any* admission rate (many admit 70–80%). "How we decide": standardized Euclidean distance on the two variables, because that is the only version in which both variables actually count.
""")

    # ---- College Q2 ----
    b.md("""
---
## College 2 — Add Carnegie classification and ownership

**Logic.** Same recipe as Ames 2: one-hot encode the two categoricals and append them to the standardized numerics. One wrinkle: `Institution` is **not unique** (15 duplicated names), and `pd.concat(axis=1)` aligns on the index, so it raises on duplicates. Avoid the alignment entirely by calling `pd.get_dummies` on a single frame with `columns=`.

**Step 2.1 — check the duplicate-index problem.**
""")
    b.code("""
dups = df_college.index[df_college.index.duplicated()].unique()
print("duplicated institution names:", len(dups), "→ e.g.", list(dups[:3]))
""")
    b.md("**Step 2.2 — encode + standardize in one frame, then distance.**")
    b.code("""
B = pd.get_dummies(df_college[num + ["CarnegieClassification", "Ownership"]],
                   columns=["CarnegieClassification", "Ownership"], dtype=float)
B[num] = (B[num] - B[num].mean()) / B[num].std()
print("columns in B:", B.shape[1])

dist2 = np.sqrt(((B - B.loc[school_name]) ** 2).sum(axis=1))
result2 = (df_college.assign(dist=dist2.round(3))
                     .sort_values("dist")[num + ["CarnegieClassification", "Ownership", "dist"]]
                     .reset_index())
show(result2, "nearest to Cal Poly — numerics + Carnegie + ownership", "purple",
     sources={"Institution": "green", "AdmissionRate": "blue", "Undergraduates": "blue",
              "CarnegieClassification": "orange", "Ownership": "orange", "dist": "yellow"}, rows=11)
""")
    b.md("""
**Answer / interpretation.** Cal Poly is a *public* "Master's Colleges & Universities: Larger Programs" school, and each of those two facts costs √2 to mismatch, so the neighbours are now CUNY Hunter, Baruch, John Jay and Brooklyn College, then Cal Poly Pomona and CUNY Queens — public master's-level institutions with 12,000–27,000 undergraduates. UCSB, UNC and UVA fall away (research doctoral universities); DeVry falls away (private for-profit). Which list is "more similar" depends on the question: *how selective and how big* → part 1; *what kind of institution* → this one.
""")

    # ---- College Q3 ----
    b.md("""
---
## College 3 — Only the mix of fields of study (PCIP columns)

**Logic.** All 38 `PCIP` columns are proportions on the same 0–1 scale, so this time **do not standardize**: z-scoring would inflate rare fields (a 1-point difference in library science would count as much as a 20-point difference in engineering). Euclidean distance on the raw proportions; as a check, **cosine similarity**, which compares the *shape* of the mix.

**Step 3.1 — select the columns and see Cal Poly's profile.**

📘 **`df.filter(like="PCIP")`** keeps columns whose name contains the text.
""")
    b.code("""
P = df_college.filter(like="PCIP")
cp_fields = P.loc[school_name]
print("PCIP columns:", P.shape[1], "  any missing:", P.isna().any().any())

top = cp_fields.sort_values(ascending=False).head(6).reset_index()
top.columns = ["CIP field", "share of Cal Poly students"]
show(top, "Cal Poly's largest fields (14 = engineering, 52 = business, 01 = agriculture, 45 = social sciences)", "green")
""")
    b.md("**Step 3.2 — Euclidean distance on the raw proportions.**")
    b.code("""
dist3 = np.sqrt(((P - cp_fields) ** 2).sum(axis=1))
result3 = (df_college.assign(dist=dist3.round(3))
                     .sort_values("dist")[["State", "Undergraduates", "CarnegieClassification", "dist"]]
                     .reset_index())
show(result3, "nearest to Cal Poly — field-of-study mix", "purple",
     sources={"Institution": "green", "State": "grey", "Undergraduates": "grey", "CarnegieClassification": "grey", "dist": "yellow"}, rows=11)
""")
    b.md("""
**Step 3.3 — cosine similarity as a cross-check.** Cosine similarity = (dot product of the two profiles) ÷ (product of their lengths); it is 1 when two schools have the same *proportions* of fields regardless of scale.

📘 **`P @ v`** — the `@` operator computes the dot product of every row of `P` with the vector `v` (matrix–vector multiplication).
""")
    b.code("""
norms = np.sqrt((P ** 2).sum(axis=1))
cosine = (P @ cp_fields) / (norms * norms.loc[school_name])
sim = cosine.sort_values(ascending=False).head(8).round(3).reset_index()
sim.columns = ["Institution", "cosine similarity"]
show(sim, "highest cosine similarity to Cal Poly's field mix", "purple",
     sources={"Institution": "green", "cosine similarity": "yellow"})
""")
    b.md("""
**Answer / interpretation.** Judged by *what students study*, Cal Poly's neighbours are the big public land-grant universities: NC State, Iowa State, Illinois, Mississippi State, Texas A&M, Clemson, Purdue, Virginia Tech, Auburn, West Virginia — the engineering + business + agriculture mix is rare outside that system. Cosine similarity agrees. None of these schools appeared in parts 1–2 (they are larger, doctoral, mostly less selective). The three lists are all "correct"; they answer three different questions, and choosing the variables *is* choosing the question.
""")
    return b


# ============================================================================ #
#  Build + execute
# ============================================================================ #

def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path, redirect_data: bool) -> None:
    nb = json.loads(path.read_text())
    local = copy.deepcopy(nb)
    if redirect_data:
        for c in local["cells"]:
            if c["cell_type"] == "code":
                c["source"] = c["source"].replace(DATA, str(DATA_DIR) + "/")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        save(tmp_path, local)
        subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=900", str(tmp_path)], check=True)
        ran = json.loads(tmp_path.read_text())
    for orig, r in zip(nb["cells"], ran["cells"]):
        if orig["cell_type"] == "code":
            orig["outputs"] = r.get("outputs", [])
            orig["execution_count"] = r.get("execution_count")
    nb["metadata"]["language_info"] = ran["metadata"].get("language_info", nb["metadata"]["language_info"])
    save(path, nb)
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c["outputs"] if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"{path.name}: execution errors {errors}")
    print(f"executed {path.relative_to(ROOT)} ({len(nb['cells'])} cells, no errors)")


def main() -> None:
    targets = [(TOPIC_OUT, build_topic(), True), (PA31_OUT, build_pa31(), False), (PA32_OUT, build_pa32(), False)]
    for path, book, redirect in targets:
        save(path, book.notebook())
        print(f"wrote {path.relative_to(ROOT)} ({len(book.cells)} cells)")
    if "--no-exec" not in sys.argv:
        for path, _, redirect in targets:
            execute(path, redirect)


if __name__ == "__main__":
    main()
