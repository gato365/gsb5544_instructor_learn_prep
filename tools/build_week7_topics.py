#!/usr/bin/env python3
"""Generate the Week 7 Topic notebook (student + executed solution).

Usage:  python3 tools/build_week7_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week7_topics.py --no-exec  # write both notebooks without executing

Topic 7.1 — Association Rules: "Who gets played together?"  -> pairs with PA 7.1 (Association Rules, groceries)
  Playlists are the baskets and artists (Latin and hip-hop) are the items. Transactions -> one-hot table ->
  support -> confidence -> lift (by hand, for two rules) -> apriori + association_rules -> why confidence alone
  misleads when one item is very popular.
The playlists are SIMULATED inside the notebook (seeded) — real artist names used as labels, made-up listening
data, said so in the notebook. No download, no credentials.  Needs mlxtend in the kernel (pip install mlxtend).
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
WEEK = ROOT / "assignments" / "practice_activities" / "week_7"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str, raises: bool = False) -> None:
    meta = {"tags": ["raises-exception"]} if raises else {}
    cells.append({"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 7.1 — Association Rules (pairs with PA 7.1)
# ============================================================================ #

t71: list[dict] = []

md(t71, """
# GSB 5544 — Topic 7.1: Association Rules — SOLUTION
*Who gets played together? Finding "if this, then that" patterns in playlists*
""")

md(t71, """
## The next 20 minutes

| | Question | New tools | Where it lands in PA 7.1 |
|---|---|---|---|
| **1. Baskets** | What is a transaction, and how does a list of lists become a table of `True`/`False`? | `explode`, `crosstab` | parts 1 – 3 |
| **2. Support** | How common is an item — or a pair of items? | `.mean()` on booleans, `.all(axis=1)` | parts 4 – 5, 9 |
| **3. Confidence & lift** | If a playlist has A, how likely is B — and is that more than chance? | two formulas | parts 6 – 11 |
| **4. Mining** | How do we find *all* the good rules at once? | `apriori`, `association_rules` | parts 12 – 19 |

A grocery store asks "what sells with what?" A streaming service asks "who gets played with whom?" Same
question, same method: **association rules**. A rule looks like **{J. Cole} → {Kendrick Lamar}**: *playlists
that contain J. Cole tend to contain Kendrick Lamar too.* Our job is to measure how strong such a rule is.
""")

code(t71, """
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_col, coord_flip, labs
from mlxtend.frequent_patterns import apriori, association_rules
""")

md(t71, """
---
## 1. Transactions: playlists as baskets

**About the data.** The artists below are real — eight Latin artists and eight hip-hop artists — but the
**playlists are simulated** by the next cell, so that we know what patterns were planted and can check whether
the method finds them. Nothing here is a statement about real listening habits. Each simulated listener
has a taste (Latin, hip-hop, or mixed), picks a few artists accordingly, and Bad Bunny is added to most
playlists regardless of taste, and a handful of pairings are deliberately made more likely — for example,
listeners with J. Cole usually add Kendrick Lamar. *Run the cell;
you do not need to study it.*
""")

code(t71, """
LATIN = ["Bad Bunny", "J Balvin", "Karol G", "Shakira", "Rauw Alejandro", "Peso Pluma", "Daddy Yankee", "Rosalía"]
HIPHOP = ["Kendrick Lamar", "Drake", "J. Cole", "Travis Scott", "Cardi B", "Megan Thee Stallion", "Future", "Tyler, the Creator"]

rng = np.random.default_rng(5544)

def simulate_playlist():
    taste = rng.choice(["latin", "hiphop", "mixed"], p=[0.4, 0.4, 0.2])
    p_latin = {"latin": 0.45, "hiphop": 0.05, "mixed": 0.25}[taste]      # chance of including each Latin artist
    p_hiphop = {"latin": 0.05, "hiphop": 0.45, "mixed": 0.25}[taste]     # ... and each hip-hop artist
    artists = set(a for a in LATIN if rng.random() < p_latin) | set(a for a in HIPHOP if rng.random() < p_hiphop)
    if rng.random() < 0.75:                         artists.add("Bad Bunny")          # Bad Bunny is on most playlists, whatever the taste
    if "J. Cole" in artists and rng.random() < 0.7: artists.add("Kendrick Lamar")    # planted pairings
    if "Karol G" in artists and rng.random() < 0.6: artists.add("Shakira")
    if "Future" in artists and rng.random() < 0.6:  artists.add("Travis Scott")
    if "Cardi B" in artists and rng.random() < 0.6: artists.add("J Balvin")          # a planted cross-genre pairing
    if len(artists) < 2:                                                             # every playlist has at least 2 artists
        artists |= set(rng.choice(LATIN + HIPHOP, size=2, replace=False))
    return sorted(artists)

playlists = [simulate_playlist() for _ in range(400)]
len(playlists), playlists[:3]
""")

md(t71, """
`playlists` is a **list of lists**: one inner list per playlist (a *transaction* or *basket*), holding the
artists (*items*) in it. Playlists have different lengths — that is why this is not yet a table.

### From list of lists to one row per (playlist, artist)

The same two moves as PA 7.1: give each playlist an id, then `explode` the list so each artist gets its own row.
""")

code(t71, """
long = (pd.DataFrame({"artist": playlists})
          .assign(playlist_id=lambda d: d.index)
          .«explode»("artist")                     # one row per (playlist, artist)
          .reset_index(drop=True))
print(long.shape)
long.head(8)
""")

md(t71, """
### The one-hot table

Association-rule tools want **one row per transaction, one column per item**, with `True`/`False`.
`pd.crosstab` counts (playlist, artist) pairs; `> 0` turns the counts into `True`/`False`.
""")

code(t71, """
basket = pd.«crosstab»(long["playlist_id"], long["artist"]) > 0
print(basket.shape)
basket.iloc[:5, :6]
""")

md(t71, """
✅ **Checkpoint 1.** (a) What does one **row** of `basket` represent? One **column**? (b) `basket.shape` is
(400, 16) — where do the two numbers come from? (c) Why `> 0` rather than keeping the counts?
""")

answer(t71, """
(a) A row is one playlist; a column is one artist; the cell says whether that artist is on that playlist.
(b) 400 simulated playlists and 16 distinct artists (8 Latin + 8 hip-hop). (c) Association rules ask *whether*
an item is in a basket, not how many times; `apriori` expects a boolean table, and booleans also make the
support calculation a plain `.mean()` (next section). In PA 7.1 the counts are all 0/1 anyway, but `> 0` is the
safe habit (a basket that lists "whole milk" twice is still one basket containing whole milk).
""")

md(t71, """
---
## 2. Support: how common is an item?

**Support** of an item = the share of baskets that contain it. Because the column is `True`/`False`, its
**mean** is exactly that share (`True` counts as 1).
""")

code(t71, """
artist_support = basket.«mean»().sort_values(ascending=False)
artist_support.round(3)
""")

code(t71, """
support_df = artist_support.reset_index()
support_df.columns = ["artist", "support"]
support_df["artist"] = pd.Categorical(support_df["artist"], categories=support_df["artist"][::-1])   # keep the order in the plot

(ggplot(support_df, aes(x="artist", y="support"))
 + geom_col()
 + coord_flip()
 + labs(title="Share of playlists containing each artist", x="", y="support"))
""")

md(t71, """
Support of a **set** of items = the share of baskets that contain **all** of them. `.all(axis=1)` asks, row by
row, "are all of these columns `True`?"
""")

code(t71, """
both = basket[["J. Cole", "Kendrick Lamar"]].«all»(axis=1)       # True only when BOTH are on the playlist
both.mean()
""")

md(t71, """
✅ **Checkpoint 2.** (a) Which artist has the highest support, and roughly what share of playlists is that?
(b) Why is `basket[["J. Cole", "Kendrick Lamar"]].mean()` (no `.all`) *not* the support of the pair?
""")

answer(t71, """
(a) **Bad Bunny**, at about **79 %** of all playlists — four in five. He was planted that way (added to 75 % of
playlists whatever the listener's taste), which makes him this data's "whole milk". (b) Without `.all`,
`.mean()` runs per column and returns *two* numbers — the support of each artist on its own. The support of
the pair needs one `True`/`False` per row ("both present?") first; that is what `.all(axis=1)` builds.
""")

md(t71, """
---
## 3. Confidence and lift: from "together" to "if … then …"

A rule **{A} → {B}** ("A is the *antecedent*, B the *consequent*") is scored with two numbers:

| Measure | Formula | Question it answers |
|---|---|---|
| **confidence** | support(A and B) ÷ support(A) | *Of the baskets with A, what share also have B?* |
| **lift** | support(A and B) ÷ [ support(A) × support(B) ] | *How many times more often than chance do A and B appear together?* |

Lift compares the observed co-occurrence with what independence would predict. **Lift ≈ 1** → no
relationship; **lift > 1** → A makes B more likely; **lift < 1** → A makes B *less* likely.

### Rule 1: {J. Cole} → {Kendrick Lamar}
""")

code(t71, """
s_cole = basket["J. Cole"].mean()
s_kendrick = basket["Kendrick Lamar"].mean()
s_both = basket[["J. Cole", "Kendrick Lamar"]].all(axis=1).mean()

confidence = s_both / «s_cole»
lift = s_both / («s_cole» * «s_kendrick»)
print(f"support(J. Cole) = {s_cole:.3f}   support(Kendrick) = {s_kendrick:.3f}   support(both) = {s_both:.3f}")
print(f"confidence = {confidence:.3f}   lift = {lift:.2f}")
""")

md(t71, """
### Rule 2: {Drake} → {Bad Bunny}

Same formulas, a very popular consequent:
""")

code(t71, """
s_drake = basket["Drake"].mean()
s_bunny = basket["Bad Bunny"].mean()
s_both2 = basket[["Drake", "Bad Bunny"]].all(axis=1).mean()

print(f"confidence = {s_both2 / s_drake:.3f}   lift = {s_both2 / («s_drake» * «s_bunny»):.2f}")
""")

md(t71, """
✅ **Checkpoint 3.** (a) In one sentence each, interpret the confidence and the lift of {J. Cole} → {Kendrick
Lamar} with the numbers. (b) Rule 2 has a respectable confidence but a lift close to 1. What does that
combination tell a streaming service? (c) Does the rule {Kendrick Lamar} → {J. Cole} have the same confidence?
The same lift? Why?
""")

answer(t71, """
(a) *Confidence 0.747*: of the playlists that have J. Cole, about **three quarters** also have Kendrick Lamar.
*Lift 2.49*: that is about **2.5 times** the share of **all** playlists that have Kendrick (30 %) — the pair
occurs 2.5× more often than chance, exactly the planted pairing. (b) The two rules have the **same confidence,
0.747** — yet Rule 2's lift is **0.94**. Drake listeners include Bad Bunny about as often as *everyone* does
(79 %), slightly less in fact; the high confidence is only Bad Bunny's popularity. Recommending Bad Bunny
*because of* Drake adds nothing: lift ≈ 1 says "no information in this rule". Confidence alone cannot tell
these two rules apart; lift can. (c) The **lift is the same** (the formula is symmetric in A and B), but the
**confidence differs**: it divides by the support of the antecedent, and Kendrick is on more playlists (30 %)
than J. Cole (22 %), so {Kendrick} → {J. Cole} has a lower confidence (0.542). Direction matters for confidence;
it does not for lift.
""")

md(t71, """
---
## 4. Mining every rule at once: `apriori` and `association_rules`

With 16 artists there are already thousands of possible rules. `apriori` finds every **frequent itemset**
(every set of items whose support is at least `min_support`); `association_rules` then turns those itemsets
into rules and scores them. Two knobs:

- `min_support` — ignore combinations that are too rare to trust;
- a `metric` and `min_threshold` — keep only rules that score well enough (here: confidence ≥ 0.30).
""")

code(t71, """
frequent_itemsets = «apriori»(basket, min_support=0.05, use_colnames=True)
frequent_itemsets["n_items"] = frequent_itemsets["itemsets"].apply(len)
print(frequent_itemsets.shape)
frequent_itemsets.sort_values("support", ascending=False).head(8)
""")

code(t71, """
def make_rules(frequent_itemsets, min_confidence):
    \"\"\"Association rules from frequent itemsets, keeping those with confidence >= min_confidence.\"\"\"
    return association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

rules = make_rules(frequent_itemsets, «0.30»)
print(len(rules), "rules")
rules.columns.tolist()
""")

md(t71, """
`antecedents` and `consequents` are frozensets — fine for the computer, hard to read. Keep the **simple rules** —
one artist on each side, the easiest to read and to act on — turn the sets into text, and keep the columns we
understand:
""")

code(t71, """
one_each_side = (rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)
rules = rules[one_each_side]
print(len(rules), "simple rules")
rules["antecedent"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequent"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
show = ["antecedent", "consequent", "support", "confidence", "lift"]

rules.sort_values(«"confidence"», ascending=False)[show].head(8).round(3)
""")

code(t71, """
rules.sort_values(«"lift"», ascending=False)[show].head(8).round(3)
""")

md(t71, """
✅ **Checkpoint 4.** (a) What do most of the highest-**confidence** rules have in common, and why is that a
trap? (b) What kinds of pairings rise to the top when you sort by **lift** instead? Do they match what was
planted in the simulation? (c) Find the cross-genre rule involving **Cardi B**. What are its confidence and
lift, and how would you explain it to a non-technical manager?
""")

answer(t71, """
(a) Seven of the eight shown have the consequent **Bad Bunny**, with confidence 0.79 – 0.84 and **lift 0.99 – 1.05**.
Any antecedent "predicts" an artist who is on four of five playlists, so confidence looks great while lift sits
at 1 — these rules are popularity, not association. (The one exception in the list, Karol G → Shakira, is a
planted pair: confidence 0.83 *and* lift 2.16.) Sorting by confidence alone is the "popular items co-occur by
chance" trap that PA 7.1 asks about. (b) The planted pairs, each in both directions with the same lift:
**J. Cole ↔ Kendrick Lamar 2.49, Future ↔ Travis Scott 2.43, Karol G ↔ Shakira 2.16, Cardi B ↔ J Balvin 1.77**,
followed by mild within-hip-hop pairings (lift ≈ 1.4 – 1.7) created by shared taste. Yes — the method recovered
exactly what was planted, which is the point of simulating. (c) **{Cardi B} → {J Balvin}**: confidence **0.70**,
lift **1.77** — compare {Cardi B} → {Bad Bunny}, confidence 0.77 but lift 0.97. To a manager: *"Seven in ten
listeners who play Cardi B also play J Balvin — about 1.8 times the rate for listeners overall. That is a genuine
cross-genre link worth a joint recommendation; the Bad Bunny number looks higher but is just his popularity."*
(Say that the data is simulated; in the real world the two did record *I Like It* together, with Bad Bunny.)
""")

md(t71, """
---
## The four lines to keep

| | |
|---|---|
| **Baskets** | list of lists → `explode` → `pd.crosstab(id, item) > 0`: one row per basket, one `True`/`False` column per item |
| **Support** | `basket["A"].mean()` for one item; `basket[["A", "B"]].all(axis=1).mean()` for a set |
| **Confidence & lift** | confidence = supp(A,B) / supp(A); lift = supp(A,B) / (supp(A) · supp(B)); lift ≈ 1 means "no relationship" |
| **Mining** | `apriori(basket, min_support=…, use_colnames=True)` → `association_rules(…, metric="confidence", min_threshold=…)`; **sort by lift, check support** |

PA 7.1 (the groceries data) is on the course site: [%s](%s).
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
    (t71, WEEK / "GSB5544_Topic_7_1_Association_Rules",
     "# GSB 5544 — Topic 7.1: Association Rules  \n"
     "*Who gets played together? Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
]


def main() -> None:
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
