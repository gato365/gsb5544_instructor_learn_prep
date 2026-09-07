#!/usr/bin/env python3
"""Generate the Topic 3.1 (Joining and Merging Data) student and solution notebooks.

Usage:  python3 tools/build_week3_joins.py            # write -empty and -solution, execute the solution
        python3 tools/build_week3_joins.py --no-exec  # write both notebooks without executing

The notebook is written once here as the *solution*.  Two markers control the student copy:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt

The solution is executed with the course's Anaconda kernel.  The data cells point at the
GitHub raw URLs (so Colab works); during execution they are temporarily redirected to the
local assignments/Data folder so the notebook can be rendered before a push.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_3"
DATA_DIR = ROOT / "assignments" / "Data"
STUDENT = WEEK / "GSB5544_Topic_3_1_Joining_and_Merging-empty.ipynb"
SOLUTION = WEEK / "GSB5544_Topic_3_1_Joining_and_Merging-solution.ipynb"

PAGE = "https://gato365.github.io/gsb5544_instructor_learn_prep/week_3/pandas.html"
DATA = "https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/"
JUPYTER = "/opt/anaconda3/bin/jupyter"


def link(slug: str) -> str:
    return f"[{PAGE}#{slug}]({PAGE}#{slug})"


def page(slug: str, label: str | None = None) -> str:
    return f"🎬 *Watch it on the page:* [{label or slug.replace('-', ' ')}]({PAGE}#{slug})"


cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(text: str) -> None:
    md("**Answer:** " + text.strip())


# ============================================================================ #
#  Title and how the two tools work together
# ============================================================================ #

md("""
# GSB 5544 — Topic 3.1: Joining and Merging Data — SOLUTION
*Combining tables with `merge`, `concat`, and reshaping — paired with the interactive page **Pandas in motion***
""")

md(f"""
## 0. Two tools, one topic

Almost every real analysis needs data that lives in **more than one table**: a game log and a team roster, orders and customers, a survey and a code book. This topic is about putting those tables together correctly, and about the ways that goes wrong (rows silently disappearing, rows silently multiplying).

You have two tools for it, and they are built to be used side by side:

| | Interactive page: **Pandas in motion** | This notebook |
|---|---|---|
| Link | [{PAGE}]({PAGE}) | you are here |
| What it shows | every row of the result being built one step at a time, with the input rows that produced it highlighted | the same techniques run for real in pandas, then applied to a real data set |
| Data | tiny sample tables (3–5 rows) that you can edit | the *same* sample tables, then 1,103 Kobe Bryant games |
| Best for | *seeing* which rows survive and why | writing the code yourself and checking outputs |

**How the page works**

- **Learn tab.** Pick a technique from the *Technique* dropdown. **Play** animates the result one row at a time; **Step →** and **← Back** move a single row; **Restart** clears it. Hover any result row to trace it back to the input rows it came from. The status line under the result explains each row in words (for example *"student_id 1: no match in grades → keep the row, fill grades fields with NA."*).
- **Python syntax panel.** Shows the exact pandas call. The *Sample data setup* block underneath is copy-paste-ready Python that creates the input tables, so anything on the page can be reproduced in this notebook.
- **Edit or generate input data.** Change the input tables (one row per line, comma-separated, `NA` or an empty cell for missing) and click **Apply changes**. The result, the step-by-step notes, and the setup code all recompute. **Generate 3 rows** adds random rows for a quick what-if; **Reset to sample data** puts the originals back.
- **Quiz tab.** Three modes: *Write the call* (type the pandas code that produced a shown result), *Pick the technique* (given a question and the tables), and *What question is answered?* (given the code). Your running score is at the top right; every answer links back to the matching Learn demo.
- **Links.** Each technique has its own address, e.g. `#left-join`. The 🎬 links in this notebook open the page at exactly that technique.
- Keyboard: `←` `→` step, `Space` play/pause, `R` restart.

**Rhythm for each section below:** open the 🎬 link and watch the technique → *predict* which rows the result will contain → run the notebook cell → compare. If the page and pandas ever disagree, see Section 10.
""")

# ============================================================================ #
#  1. Setup
# ============================================================================ #

md("""
---
## 1. Setup: the same two tables as the page

The page's sample data is a three-row roster and a three-row score sheet. We build the identical tables here, so every result below can be checked against the animation row for row.
""")

code("""
import pandas as pd

students = pd.DataFrame({"student_id": [1, 2, 3], "name": ["Ava", "Ben", "Cam"]})
grades   = pd.DataFrame({"student_id": [2, 3, 4], "score": [80, 90, 70]})
""")

code("""
students
""")

code("""
grades
""")

md("""
Look at the **key** column, `student_id`, in both tables:

- IDs **2** and **3** appear in *both* tables.
- ID **1** (Ava) is only in `students` — no score yet.
- ID **4** is only in `grades` — a score with no student on the roster.

Every join below is a different answer to one question: **what should happen to rows 1 and 4?**
""")

# ============================================================================ #
#  2. The four joins
# ============================================================================ #

md(f"""
---
## 2. *Which rows survive?*  → the four joins

`left.merge(right, on="key", how=...)` matches rows whose key values are equal. `how=` decides what happens to rows that have **no** match.

| `how=` | Rows kept | Unmatched side is filled with |
|---|---|---|
| `"inner"` | keys present in **both** tables (the default) | — nothing is unmatched |
| `"left"` | **every** left row, plus matches | `NaN` in the right table's columns |
| `"right"` | **every** right row, plus matches | `NaN` in the left table's columns |
| `"outer"` | every row from **both** tables | `NaN` on whichever side is missing |

**Q: Among students on the roster, who has a score?**  {page("inner-join")}
""")

code("""
students.merge(grades, on="student_id", how=«"inner"»)
""")

md(f"""
Rows 1 and 4 are both gone: an inner join keeps only keys present on both sides. (`how="inner"` is the default, so `students.merge(grades, on="student_id")` gives the same thing — but write it out while you are learning.)

**Q: Keep the complete roster, even when a score is missing.**  {page("left-join")}
""")

code("""
students.merge(grades, on="student_id", how=«"left"»)
""")

md("""
Ava is back, with `NaN` for her score. Notice the dtype of `score`: it was an integer column, but it now holds a missing value, and pandas stores a column with `NaN` as **float** — hence `80.0` instead of `80`. The page shows the same thing (`80.0`).
""")

code("""
students.merge(grades, on="student_id", how="left").dtypes
""")

md(f"""
**Q: Keep all score records, including an ID that is not on the roster.**  {page("right-join")}
""")

code("""
students.merge(grades, on="student_id", how=«"right"»)
""")

md(f"""
**Q: Audit the complete set of IDs across both sources.**  {page("full-outer-join", "full / outer join")}
""")

code("""
students.merge(grades, on="student_id", how=«"outer"»)
""")

md("""
Nothing is dropped. One detail worth knowing: an **outer** join returns its rows sorted by key (1, 2, 3, 4); the other joins keep the row order of the table whose rows they preserve.

✅ **Check:** before running the next cell, predict the number of rows each join returns for `students` and `grades`. Then verify.
""")

code("""
for how in ["inner", "left", "right", "outer"]:
    n_rows = students.merge(grades, on="student_id", how=how).«shape[0]»
    print(f"{how:>6}: {n_rows} rows")
""")

answer("""
inner 2, left 3, right 3, outer 4. In general `rows(outer) = rows(left) + rows(right) − rows(inner)` when keys are unique on both sides.
""")

# ============================================================================ #
#  3. Anti joins
# ============================================================================ #

md(f"""
---
## 3. *Which rows have NO match?*  → anti joins

Often the unmatched rows are the whole point: students who never got a score, orders with no customer record, keys that failed to line up. pandas has no `how="anti"`, so we filter with `.isin()` and `~` (not).

**Q: Which students have not received a score yet?**  {page("left-anti-join")}
""")

code("""
students[«~»students["student_id"].isin(grades["student_id"])]
""")

md(f"""
**Q: Which scores were recorded under an ID that no student has?**  {page("right-anti-join")}
""")

code("""
grades[~grades["student_id"].«isin»(students["student_id"])]
""")

md("""
An alternative that scales to several key columns: do an outer join with `indicator=True` and keep the `left_only` rows (Section 5 explains the indicator).
""")

code("""
audit = students.merge(grades, on="student_id", how="outer", indicator=True)
audit[audit["_merge"] == "left_only"]
""")

# ============================================================================ #
#  4. Cross join
# ============================================================================ #

md(f"""
---
## 4. *Every combination?*  → cross join

A cross join has **no key at all**: every left row is paired with every right row, so 3 × 2 = 6 rows. Use it to build a grid of possibilities (every student × every course) that you then fill in.  {page("cross-join")}
""")

code("""
names   = pd.DataFrame({"name": ["Ava", "Ben", "Cam"]})
courses = pd.DataFrame({"course": ["Stats", "Python"]})

names.merge(courses, how=«"cross"»)
""")

# ============================================================================ #
#  5. Join details
# ============================================================================ #

md(f"""
---
## 5. The details that break real joins

### 5a. The key has a different name in each table  {page("different-key-names")}

Two files rarely agree on column names. `left_on=` and `right_on=` say which column is the key on each side. **Both** key columns survive in the result; drop one afterwards if you like.
""")

code("""
grades_id = pd.DataFrame({"id": [2, 3, 4], "score": [80, 90, 70]})

students.merge(grades_id, how="left", «left_on»="student_id", «right_on»="id")
""")

md(f"""
### 5b. The key is a *combination* of columns  {page("multiple-keys")}

A student can take several courses, so `student_id` alone does not identify a row of the grade sheet — the pair (`student_id`, `course`) does. Pass a **list** to `on=`. Joining on `student_id` alone would attach every course's grade to every enrollment — try it and count the rows.
""")

code("""
enrollment    = pd.DataFrame({"student_id": [1, 1, 2], "course": ["Stats", "Python", "Stats"]})
course_grades = pd.DataFrame({"student_id": [1, 1, 2], "course": ["Python", "Stats", "Stats"], "score": [95, 80, 90]})

enrollment.merge(course_grades, how="left", on=«["student_id", "course"]»)
""")

code("""
# Q: what goes wrong with a single key?  -> each Ava enrollment matches BOTH Ava grades
enrollment.merge(course_grades, how="left", on="student_id")
""")

md(f"""
### 5c. Where did each row come from? Is the key unique?  {page("audit-validate", "audit + validate")}

- `indicator=True` adds a `_merge` column: `both`, `left_only`, or `right_only`. Run `value_counts()` on it and you have a one-line audit of the join.
- `validate=` states the key relationship you *expect* and raises an error if the data disagrees: `"one_to_one"`, `"one_to_many"` (left keys unique), `"many_to_one"` (right keys unique), or `"many_to_many"` (checks nothing).
""")

code("""
audit = students.merge(grades, on="student_id", how="outer", «indicator»=True, «validate»="one_to_many")
audit
""")

code("""
audit["_merge"].value_counts()
""")

md(f"""
### 5d. Duplicate matches multiply rows  {page("duplicate-matches")}

Ben (ID 2) is in two clubs and has two exam scores. Joining clubs to exams on `student_id` produces **every combination**: 2 × 2 = 4 rows for one student. This is the most common way a join quietly corrupts an analysis — sums and counts double without any error message.
""")

code("""
clubs       = pd.DataFrame({"student_id": [2, 2], "club": ["Art", "Music"]})
exam_grades = pd.DataFrame({"student_id": [2, 2], "exam": ["E1", "E2"], "score": [80, 90]})

clubs.merge(exam_grades, on="student_id")
""")

md("""
`validate=` is the guard rail. Asking for `"one_to_many"` here raises `MergeError`, because the left keys are not unique:
""")

code("""
try:
    clubs.merge(exam_grades, on="student_id", validate=«"one_to_many"»)
except pd.errors.MergeError as e:
    print("MergeError:", e)
""")

md(f"""
### 5e. Both tables have a column with the same name  {page("overlapping-columns-suffixes", "overlapping columns (suffixes)")}

Non-key columns that share a name get suffixes so neither is lost; the default is `_x` / `_y`, which tells you nothing. Name them.

**Q: How did each student's score change from midterm to final?**
""")

code("""
midterm = pd.DataFrame({"student_id": [1, 2, 3], "score": [72, 85, 90]})
final   = pd.DataFrame({"student_id": [1, 2, 3], "score": [80, 79, 95]})

both = midterm.merge(final, on="student_id", «suffixes»=("_mid", "_final"))
both["change"] = both["score_final"] - both["score_mid"]
both
""")

md("""
✅ **Check:** a colleague joins a 500-row order table to a 40-row product table and gets 620 rows. In one sentence, what happened, and which single argument would have caught it?
""")

answer("""
The product table has repeated product keys, so some orders matched more than one product row and were duplicated; `validate="many_to_one"` would have raised a `MergeError` instead of silently returning 620 rows.
""")

# ============================================================================ #
#  6. Concat
# ============================================================================ #

md(f"""
---
## 6. *Same columns, more rows?*  → `pd.concat`

A join matches on a key. **Concatenation** does no matching at all — it just stacks tables.

**Q: What is the combined list of scores across fall and spring?**  {page("concat-add-rows", "concat: add rows")}
""")

code("""
fall   = pd.DataFrame({"student": ["Ava", "Ben"], "score": [80, 90]})
spring = pd.DataFrame({"student": ["Cam"], "score": [70]})

pd.concat([fall, spring], «ignore_index»=True)
""")

md("""
Columns are aligned by **name** (a column missing from one table is filled with `NaN`). Without `ignore_index=True` the row labels would be `0, 1, 0` — the originals stacked — which is rarely what you want.
""")

md(f"""
**Q: Place two measurements side by side when they share index labels but not row positions.**  {page("concat-add-columns", "concat: add columns")}

`axis=1` concatenates *columns*, aligning on the **index labels** — not on row position, and not on any column. Set the index first, or you will glue together rows that have nothing to do with each other.
""")

code("""
names_idx  = pd.DataFrame({"index": [0, 1], "name": ["Ava", "Ben"]})
scores_idx = pd.DataFrame({"index": [1, 2], "score": [90, 70]})

pd.concat([names_idx.set_index("index"), scores_idx.set_index("index")], «axis=1»).reset_index()
""")

# ============================================================================ #
#  7. Reshape
# ============================================================================ #

md(f"""
---
## 7. *Wide or long?*  → reshaping

Topic 2.2 introduced `melt` (wide → long) and grouped summaries. The page adds the opposite direction and a few relatives; here is each in one cell.

**Pivot: long → wide.**  Each unique (student, exam) pair becomes a cell. `pivot` does **not** aggregate, so duplicates raise an error.  {page("pivot-long-wide", "pivot: long → wide")}
""")

code("""
exam_long = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Ben"],
                          "exam": ["E1", "E2", "E1", "E2"],
                          "score": [80, 90, 70, 100]})

exam_long.pivot(index="student", columns=«"exam"», values="score").reset_index()
""")

md(f"""
**Pivot table: aggregate while reshaping.**  Ava retook E1, so (Ava, E1) has two rows. `pivot()` raises `ValueError`; `pivot_table()` summarizes them (`aggfunc="mean"` by default).  {page("pivot-table-aggregate", "pivot table: aggregate")}
""")

code("""
exam_long_dup = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Ben", "Ava"],
                              "exam": ["E1", "E2", "E1", "E2", "E1"],
                              "score": [80, 90, 70, 100, 100]})

exam_long_dup.«pivot_table»(index="student", columns="exam", values="score", aggfunc="mean").reset_index()
""")

md(f"""
**Melt: wide → long.**  The reverse of pivot — column names become values of `exam`.  {page("melt-wide-long", "melt: wide → long")}
""")

code("""
wide = pd.DataFrame({"student": ["Ava", "Ben"], "E1": [80, 70], "E2": [90, 100]})

wide.melt(id_vars=«"student"», var_name="exam", value_name="score")
""")

md(f"""
**GroupBy, drop duplicates, crosstab.**  Familiar from Topic 2.2. `drop_duplicates(subset=...)` matters *for joins*: it is how you make a key unique so a lookup table is safe to join `many_to_one`.  {page("groupby-aggregate", "groupby")} · {page("drop-duplicates")} · {page("crosstab-counts", "crosstab")}
""")

code("""
exam_long_dup.groupby("student")["score"].mean().reset_index()
""")

code("""
roster = pd.DataFrame({"student_id": [1, 2, 2, 3, 1], "name": ["Ava", "Ben", "Ben", "Cam", "Ava"]})

roster.«drop_duplicates»(subset="student_id")
""")

code("""
enrollment_by_name = pd.DataFrame({"student": ["Ava", "Ava", "Ben", "Cam", "Cam"],
                                   "course": ["Stats", "Python", "Stats", "Python", "Stats"]})

pd.crosstab(enrollment_by_name["student"], enrollment_by_name["course"])
""")

# ============================================================================ #
#  8. Real data
# ============================================================================ #

md(f"""
---
## 8. Real data: Kobe Bryant's game log meets a team lookup

Toy tables make the rules clear; real tables have the problems. Three files:

- `kobe_bryant_games.csv` — one row per game for Kobe's first 15 seasons (1,103 games). Each row names the `opponent` only by a three-letter abbreviation.
- `nba_teams.csv` — one row per **current** NBA team: abbreviation, name, city, conference, division.
- `franchise_moves.csv` — abbreviations that no longer exist, and which current franchise they became.

**Q: Which division did Kobe score the most against? Which did he beat most often?** The game log cannot answer this — it has no division column. The team table has divisions but no games. We need a join.
""")

code(f"""
DATA = "{DATA}"

games = pd.read_csv(DATA + "kobe_bryant_games.csv")
teams = pd.read_csv(DATA + "nba_teams.csv")
moves = pd.read_csv(DATA + "franchise_moves.csv")

print(games.shape, teams.shape, moves.shape)
games[["season_label", "date", "location", "opponent", "outcome", "points"]].head()
""")

code("""
teams.head()
""")

md("""
### 8a. The naive join, and the audit that saves it

The key is `opponent` in `games` and `abbr` in `teams` (different names → `left_on` / `right_on`). We want to keep **every game**, so this is a left join. Then the first thing to do after any join: **count the rows and count the missing values.**
""")

code("""
joined = games.merge(teams, «left_on»="opponent", «right_on»="abbr", how="left")

print("games:", len(games), "  joined:", len(joined))
print("games with no team match:", joined["team"].«isna»().sum())
""")

md("""
Row count unchanged — good, no duplicate keys in `teams`. But 138 games have no team information. Which opponents failed to match? That is a **left anti join**:
""")

code("""
games[~games["opponent"].isin(teams["abbr"])]["opponent"].«value_counts»()
""")

md("""
Eight abbreviations, all of them franchises that moved or were renamed since Kobe played them: Seattle became Oklahoma City, New Jersey became Brooklyn, Vancouver became Memphis, the Hornets/Bobcats/Pelicans story needs a diagram. The keys are *not wrong*; they are **from a different era** than the lookup table. This is the everyday reality of joins: keys drift.

And in the other direction — a **right anti join** — which current teams never appear as an opponent?
""")

code("""
teams[~teams["abbr"].isin(games["opponent"])]
""")

md("""
Three of the four are the *new* names of the franchises above. The fourth, LAL, is Kobe's own team — he never played against the Lakers, which is a nice sanity check that the anti join is telling the truth.
""")

md("""
### 8b. Fix the keys with a second join, then validate

`franchise_moves.csv` maps each old abbreviation to its current one. Left-join it onto `games` (every game must survive), then build a clean key `opp_current`: the current abbreviation where one exists, otherwise the original.
""")

code("""
moves[["old_abbr", "old_name", "current_abbr"]]
""")

code("""
games = games.merge(moves[["old_abbr", "current_abbr"]], left_on="opponent", right_on="old_abbr", how="left")
games["opp_current"] = games["current_abbr"].«fillna»(games["opponent"])
games = games.drop(columns=["old_abbr", "current_abbr"])

games[["opponent", "opp_current"]].drop_duplicates().sort_values("opponent").head(12)
""")

md("""
Now join to `teams` again, this time stating what we expect: many games per team, but each team **once** in the lookup (`validate="many_to_one"`), and an indicator so the audit is a single `value_counts`.
""")

code("""
joined = games.merge(teams, left_on="opp_current", right_on="abbr", how="left",
                     validate=«"many_to_one"», indicator=True)

joined["_merge"].value_counts()
""")

md("""
Every one of the 1,103 games now has a team, conference, and division. The question we started with takes one `groupby`:
""")

code("""
joined = joined.drop(columns="_merge")

by_division = (joined.groupby("division")
                     .agg(games=("points", "size"),
                          avg_points=("points", "mean"),
                          win_share=("outcome", lambda s: (s == "W").mean()))
                     .round(3)
                     .sort_values("avg_points", ascending=False))
by_division
""")

answer("""
Kobe averaged the most points against the Pacific division (his own — 26.2 per game, and a 73.5% win rate) and the fewest against the Central (24.1). The Southeast was his toughest division by win share (60.9%). Both facts needed a column that only existed in the *other* table.
""")

md("""
### 8c. The duplicate trap on real data

`moves` has **three** rows for the Pelicans franchise (CHH, NOH, NOK all became NOP). If someone joins the franchise history onto games by `current_abbr`, every game against that franchise is tripled:
""")

code("""
history = games.merge(moves, left_on="opp_current", right_on="current_abbr", how="inner")

print("games against a relocated franchise:", games["opp_current"].isin(moves["current_abbr"]).sum())
print("rows after the join:               ", len(history))
""")

code("""
try:
    games.merge(moves, left_on="opp_current", right_on="current_abbr", how="inner", validate="many_to_one")
except pd.errors.MergeError as e:
    print("MergeError:", str(e)[:90], "...")
""")

md("""
209 games became 289 rows, with no warning. `validate=` turns that into an error you cannot miss. Whenever a table is meant to be a *lookup*, make its key unique first (`drop_duplicates`) and say so (`validate="many_to_one"`).
""")

md("""
### 8d. Suffixes and pivots: home vs. away, season by season

**Q: Did Kobe score more at home than on the road?** Two grouped summaries share the column name `points`; joining them on `season_label` needs suffixes.
""")

code("""
home = games[games["location"] == "Home"].groupby("season_label")["points"].mean().reset_index()
away = games[games["location"] == "Away"].groupby("season_label")["points"].mean().reset_index()

home_away = home.merge(away, on="season_label", suffixes=(«"_home"», «"_away"»))
home_away["home_edge"] = (home_away["points_home"] - home_away["points_away"]).round(2)
home_away
""")

md("""
The same table in one line, with `pivot_table`: rows = season, columns = location, cells = mean points. Recognizing that a "join two summaries" problem is really a "reshape one summary" problem is a big time-saver.
""")

code("""
games.«pivot_table»(index="season_label", columns="location", values="points", aggfunc="mean").round(1)
""")

md("""
✅ **Check:** stack the early and late parts of the career with `pd.concat` — seasons 1–5 labelled `"early"` and seasons 11–15 labelled `"late"` — and compare average points per game between the two eras. (Hint: filter, add an `era` column to each piece, concatenate, `groupby`.)
""")

code("""
early = games[games["season"] <= 5].copy()
late  = games[games["season"] >= 11].copy()
early["era"] = "early"
late["era"]  = "late"

eras = pd.concat([early, late], ignore_index=True)
eras.groupby("era")["points"].agg(["size", "mean"]).round(1)
""")

answer("""
About 18.5 points per game in seasons 1–5 (he came off the bench as an 18-year-old) versus about 28 in seasons 11–15. Concatenation added rows without matching anything; the `era` column we created *before* stacking is what made the comparison possible afterwards.
""")

# ============================================================================ #
#  9. Test yourself on the page
# ============================================================================ #

md(f"""
---
## 9. Test yourself on the interactive page

Do these four exercises on the page, then come back and finish the cells. Each one checks that what you *see* on the page and what pandas *computes* agree.

**Exercise 1 — change the data, predict the result.**
1. Open {link("left-join")} and expand **Edit or generate input data**.
2. In the `grades` box add the line `1,65` (Ava now has a score) and click **Apply changes**.
3. *Predict* before you press Play: how many rows, and which cell was `NA` before but is not now?
4. Reproduce it here and compare:
""")

code("""
grades_v2 = pd.concat([grades, pd.DataFrame({"student_id": [1], "score": [65]})], ignore_index=True)
students.merge(grades_v2, on="student_id", how="left")
""")

md(f"""
**Exercise 2 — random data, same answer.**
1. Open {link("multiple-keys")}, expand the editor, press **Generate 3 rows** under *enrollment* and again under *grades*, then **Apply changes**.
2. Open **Sample data setup — copy before running the code**, copy the whole block, and paste it into the cell below (replacing the example).
3. Copy the call from the **Python syntax** panel into the same cell and run it. Compare your output with the page's result table row by row — including the `NA` cells.

The example below is what such a paste looks like; overwrite it with your own.
""")

code("""
import pandas as pd

enrollment = pd.DataFrame({'student_id': [1, 1, 2, 3, 4, 5], 'course': ['Stats', 'Python', 'Stats', 'Calc', 'Bio', 'Econ']})
grades = pd.DataFrame({'student_id': [1, 1, 2, 4, 6, 5], 'course': ['Python', 'Stats', 'Stats', 'Bio', 'Calc', 'Stats'], 'score': [95, 80, 90, 75, 100, 65]})

result = enrollment.merge(grades, how="left", on=["student_id", "course"])
result
""")

md(f"""
**Exercise 3 — the quiz.** Switch to the **Quiz** tab and answer at least five questions in *each* of the three modes. In *Write the call*, choose the DataFrame(s) from the dropdowns and type the code exactly as it appears on the page (spaces and quote style do not matter; argument names and values do). Record your scores in the cell below, and for any question you missed, follow the *Watch this technique in the Learn tab* link.

**Exercise 4 — break it on purpose.** On {link("pivot-long-wide")}, add the row `Ava,E1,100` to the input and apply. The page should report the same error pandas raises. Confirm below.
""")

code("""
try:
    exam_long_dup.pivot(index="student", columns="exam", values="score")
except ValueError as e:
    print("ValueError:", e)
""")

md("""
**Scores (Exercise 3):**

| Mode | Correct / attempted |
|---|---|
| Write the call | \\_\\_ / \\_\\_ |
| Pick the technique | \\_\\_ / \\_\\_ |
| What question is answered? | \\_\\_ / \\_\\_ |
""")

# ============================================================================ #
#  10. If the page and pandas disagree
# ============================================================================ #

md(f"""
---
## 10. If the page and pandas ever disagree

The page does **not** run Python. It has its own small engine that imitates pandas on the displayed tables (checked against pandas 2.2.3 on the sample data). pandas is the authority. If you find a case where the two differ:

1. In the page, copy the **Sample data setup** block and the code from the **Python syntax** panel.
2. Paste both into a cell here, run it, and take a screenshot of the page's result next to the pandas output.
3. Send the technique name, the pasted code, and the two results to the instructor (or post them in the course forum). Include the address bar link, e.g. `{PAGE}#full-outer-join`, so the exact technique can be opened.

Things that are *expected* and are not bugs: pandas prints the row index on the left, while the page hides it (except in the concat-columns example); both print `80.0` instead of `80` once a `NaN` enters an integer column; and both sort an outer join's rows by key.
""")

# ============================================================================ #
#  11. Practice activities
# ============================================================================ #

md("""
---
## 11. Practice Activities — you pick the technique

For each question, first say **which technique** (inner / left / right / outer / anti / cross join, concat, pivot, pivot_table, melt, drop_duplicates) answers it, then write the code. Use `games` (with the `opp_current` key), `teams`, and `moves` from Section 8.

**PA 1.** How many games did Kobe play against each current franchise? Show the top 5 **with the team's full name**, not the abbreviation.
""")

code("""
# left join -> value_counts
games.merge(teams, left_on="opp_current", right_on="abbr", how="left")["team"].«value_counts»().head(5)
""")

md("""
**PA 2.** Which current teams did Kobe *never* face in these 15 seasons? Answer it twice: once using the original `opponent` column and once using `opp_current`. Explain the difference in one sentence.
""")

code("""
# right anti join, twice
print("original keys:", sorted(teams[~teams["abbr"].isin(games[«"opponent"»])]["abbr"]))
print("fixed keys:   ", sorted(teams[~teams["abbr"].isin(games[«"opp_current"»])]["abbr"]))
""")

answer("""
With the original keys four teams look unplayed (BRK, CHO, LAL, NOP); with the fixed keys only LAL remains. The other three were played dozens of times under their old abbreviations — an anti join is only as good as the keys it compares.
""")

md("""
**PA 3.** Build a wide table with one row per `season_label` and one column per `outcome` (W / L) holding Kobe's mean points. In which seasons did he average *more* in losses than in wins?
""")

code("""
# pivot_table (aggregate + reshape)
wl = games.pivot_table(index="season_label", columns=«"outcome"», values="points", aggfunc="mean").round(1)
wl[wl["L"] > wl["W"]]
""")

answer("""
Only 2007-08, 2008-09, and marginally 2001-02 and 2010-11 — in most seasons he scored more in wins. (Interpretation: in losses the Lakers often needed him to shoot more, but it was rarely enough.)
""")

md("""
**PA 4.** Take `home_away` from Section 8d, melt it back to long form (one row per season × location), and draw a line chart of points by season with one line per location. Which technique undoes a pivot?
""")

code("""
# melt (wide -> long), then plotnine
from plotnine import *

ha_long = home_away.«melt»(id_vars="season_label", value_vars=["points_home", "points_away"],
                          var_name="location", value_name="points")

(ggplot(ha_long, aes(x="season_label", y="points", color="location", group="location"))
 + geom_line()
 + geom_point()
 + labs(x="Season", y="Mean points per game", title="Kobe Bryant: home vs. away scoring")
 + theme(axis_text_x=element_text(rotation=90))
)
""")

md("""
**PA 5.** A colleague wants each game's opponent conference and builds a lookup from the game log itself: `conf_lookup = games[["opponent", "opp_conference"]]`. They then merge it onto `games` on `opponent`. How many rows come back? Fix the lookup so the merge returns exactly 1,103 rows, and prove it with `validate=`.
""")

code("""
conf_lookup = games[["opponent", "opp_conference"]]
print("naive merge rows:", len(games.merge(conf_lookup, on="opponent")))

conf_lookup = conf_lookup.«drop_duplicates»(subset="opponent")
fixed = games.merge(conf_lookup, on="opponent", validate=«"many_to_one"», suffixes=("", "_lookup"))
print("fixed merge rows:", len(fixed))
""")

answer("""
44,269 rows: every game against an opponent matched every *other* game against the same opponent (a many-to-many join, roughly the sum of squared game counts). Making the lookup key unique with `drop_duplicates` brings it back to 1,103, and `validate="many_to_one"` guarantees it stays that way.
""")

md("""
**PA 6 (write, don't code).** A left join is supposed to "keep every left row." Explain how a left join can nevertheless return *more* rows than the left table, and how it can end up with `NaN` in columns the left table never had. Name the argument that guards against the first problem and the technique from Section 3 that diagnoses the second.
""")

answer("""
More rows: a left row matches *several* right rows when the right key is not unique, and each match becomes its own row (`validate="many_to_one"` catches it). `NaN`s: left rows with *no* match are kept and their right-side columns are filled with missing values; a left anti join (`left[~left[key].isin(right[key])]`) lists exactly those rows so you can see whether the keys are wrong, stale, or genuinely absent.
""")

# ============================================================================ #
#  12. Summary
# ============================================================================ #

md(f"""
---
## Summary

| Question | Technique | Code |
|---|---|---|
| Only rows with a match on both sides | inner join | `L.merge(R, on="k")` |
| Keep every left row | left join | `L.merge(R, on="k", how="left")` |
| Keep every right row | right join | `how="right"` |
| Keep everything, audit both sides | outer join | `how="outer", indicator=True` |
| Left rows with **no** match | left anti join | `L[~L["k"].isin(R["k"])]` |
| Every combination | cross join | `how="cross"` |
| Keys named differently | — | `left_on=, right_on=` |
| Key is several columns | — | `on=["k1", "k2"]` |
| Guard against row multiplication | validate | `validate="many_to_one"` |
| Same column name on both sides | suffixes | `suffixes=("_a", "_b")` |
| More rows, same columns | concat | `pd.concat([A, B], ignore_index=True)` |
| More columns, aligned by index | concat | `pd.concat([A, B], axis=1)` |
| Long → wide (unique pairs) | pivot | `.pivot(index=, columns=, values=)` |
| Long → wide with aggregation | pivot table | `.pivot_table(..., aggfunc="mean")` |
| Wide → long | melt | `.melt(id_vars=, var_name=, value_name=)` |
| Make a lookup key unique | drop duplicates | `.drop_duplicates(subset="k")` |

**After every join, check two numbers:** the row count (did it change?) and the missing count in a right-side column (how many left rows found no partner?). The page: [{PAGE}]({PAGE}).
""")


# ============================================================================ #
#  Build the two notebooks
# ============================================================================ #

BLANK = re.compile(r"«(.*?)»", re.S)

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3",
    },
}


def notebook(cell_list: list[dict]) -> dict:
    out = []
    for index, cell in enumerate(cell_list):
        cell = copy.deepcopy(cell)
        cell["id"] = f"cell-{index:03d}"
        out.append(cell)
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def solution_cells() -> list[dict]:
    out = []
    for cell in cells:
        cell = copy.deepcopy(cell)
        if cell["cell_type"] == "code":
            cell["source"] = BLANK.sub(lambda m: m.group(1), cell["source"])
        out.append(cell)
    return out


def student_cells() -> list[dict]:
    out = []
    for cell in cells:
        cell = copy.deepcopy(cell)
        if cell["cell_type"] == "code":
            cell["source"] = BLANK.sub("____", cell["source"])
        elif cell["source"].startswith("**Answer:**"):
            cell["source"] = "**Your answer:** *(write it here — replace this line)*"
        out.append(cell)
    out[0]["source"] = (
        "# GSB 5544 — Topic 3.1: Joining and Merging Data  \n"
        "*Fill each `____` blank as you work; the ✅ checks and practice activities ask for complete expressions.*\n\n"
        f"Paired with the interactive page **Pandas in motion**: [{PAGE}]({PAGE})"
    )
    return out


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    """Execute the solution in place, reading data locally, then restore the URL sources."""
    nb = json.loads(path.read_text())
    local = copy.deepcopy(nb)
    for cell in local["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace(DATA, str(DATA_DIR) + "/")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        save(tmp_path, local)
        subprocess.run(
            [JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=300", str(tmp_path)],
            check=True,
        )
        executed = json.loads(tmp_path.read_text())
    for original, ran in zip(nb["cells"], executed["cells"]):
        if original["cell_type"] == "code":
            original["outputs"] = ran.get("outputs", [])
            original["execution_count"] = ran.get("execution_count")
    nb["metadata"]["language_info"] = executed["metadata"].get("language_info", nb["metadata"]["language_info"])
    save(path, nb)
    errors = [
        (i, o.get("ename"), o.get("evalue"))
        for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
        for o in c["outputs"] if o.get("output_type") == "error"
    ]
    if errors:
        raise SystemExit(f"Execution errors: {errors}")


def main() -> None:
    WEEK.mkdir(parents=True, exist_ok=True)
    save(STUDENT, notebook(student_cells()))
    save(SOLUTION, notebook(solution_cells()))
    print(f"wrote {STUDENT.relative_to(ROOT)} and {SOLUTION.relative_to(ROOT)} ({len(cells)} cells)")
    if "--no-exec" not in sys.argv:
        if not shutil.which(JUPYTER) and not Path(JUPYTER).exists():
            raise SystemExit(f"{JUPYTER} not found; run with --no-exec or edit JUPYTER")
        execute(SOLUTION)
        print("executed the solution notebook (no errors)")


if __name__ == "__main__":
    main()
