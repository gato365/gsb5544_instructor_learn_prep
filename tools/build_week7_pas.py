#!/usr/bin/env python3
"""Build the Week 7 practice-activity INSTRUCTOR SOLUTION notebook (PA 7.1 — Association Rules Mining).

Usage:  python3 tools/build_week7_pas.py            # build + execute the -solution file
        python3 tools/build_week7_pas.py --no-exec  # build without executing

The student notebook `GSB5544_PA_7_1_Association_Rules.ipynb` is the source of truth for the questions and is left
untouched.  This script writes a `-solution.ipynb` sibling: every empty code cell (in order) is replaced by an answer
group — approach, complete runnable code, what each piece does, expected output, common mistakes — and the two empty
markdown cells (parts 5 and 18) by written interpretations.  The stray empty code cell after the references is dropped.

The data is the Groceries set (9,835 baskets, 169 items) read live from GitHub, so executing needs a network
connection.  Needs mlxtend in the Anaconda kernel (`pip install mlxtend`).  `make_rules` (used in part 13) is the
helper from the reading; it is defined in the solution as a thin wrapper around mlxtend's `association_rules`.
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

PA71_SRC = WEEK / "GSB5544_PA_7_1_Association_Rules.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str, raises: bool = False) -> dict:
    meta = {"tags": ["raises-exception"]} if raises else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n")}


# ============================================================================ #
#  PA 7.1 — Association Rules Mining
# ============================================================================ #

PA71_TITLE = "## Practice Activity 7-1: Association Rules Mining — INSTRUCTOR SOLUTION"

PA71_INTRO = md("""
**How to use this notebook.** Each part is answered in the same five beats: **Approach** (the idea, in a sentence
or two) → **code** (complete and executed) → **What the code does** (piece by piece) → **Expected output** →
**Common mistakes** (what students actually do, and the symptom you will see on their screen).

**The one idea to keep returning to:** *support says how much evidence a rule rests on, confidence says how
reliable the "if … then" is, and lift says whether it beats chance.* Whole milk is in a quarter of all baskets,
so almost anything → whole milk has a decent confidence; only lift tells you whether the antecedent actually
matters. Sort by lift, then check support.

`make_rules` (part 13) is the reading's helper; it is defined below as a wrapper around mlxtend's
`association_rules`, so the notebook runs on its own.
""")

# ---- answer groups, in the order of the empty code cells of the student notebook -------------------------------

Q1 = [
    md("""
### Solution 1 — the transaction-by-item table

**Approach.** `pd.crosstab(basket_id, item)` counts how many times each item appears in each basket (0 or 1 here);
`> 0` turns the counts into `True`/`False`, which is the form `apriori` expects.
"""),
    code("""
basket = pd.crosstab(long["basket_id"], long["item"]) > 0

print(basket.shape)
print(basket.dtypes.iloc[0])
basket.iloc[:5, :6]
"""),
    md("""
**What the code does.**
- `pd.crosstab(rows, columns)` builds a table with one row per distinct `basket_id` and one column per distinct
  `item`, each cell holding the count of (basket, item) rows in `long`.
- `> 0` compares every cell with 0 and gives a boolean table of the same shape.
- The checks: 9,835 rows (baskets) × 169 columns (items), dtype `bool`.

**Expected output.** `(9835, 169)`, `bool`, and a corner of the table that is mostly `False`.

**Common mistakes.**
- Stopping at `pd.crosstab(...)` → an integer table. It still works for `.sum()`, but `apriori` warns about
  non-boolean input, and `.mean()` would be wrong if any basket listed an item twice (none do here, but the habit
  matters).
- `long.pivot(index="basket_id", columns="item", values=...)` → `ValueError: Index contains duplicate entries`
  or a `NaN`-filled table; `pivot_table(..., aggfunc="size", fill_value=0) > 0` works but is more typing.
- Building the table from `raw` (one column of text) instead of `long`.
"""),
]

Q2 = [
    md("""
### Solution 2 — basket sizes

**Approach.** A basket's size is the number of `True`s in its row; summing a boolean row counts them.
"""),
    code("""
basket_sizes = basket.sum(axis=1)
basket_sizes.head()
"""),
    md("""
**What the code does.** `axis=1` sums **across the columns** of each row — one number per basket. (`axis=0`, the
default, would count how many baskets contain each item: the numerator of support, part 4.)

**Expected output.** A Series indexed by `basket_id` — `0 → 4, 1 → 3, 2 → 1, 3 → 4, 4 → 4` (basket 2 is just
"whole milk").

**Common mistakes.** Forgetting `axis=1` → 169 numbers (per item) instead of 9,835 (per basket).
`len(basket)` → 9835, the number of baskets, not their sizes.
"""),
]

Q3 = [
    md("""
### Solution 3 — the distribution of basket sizes

**Approach.** Numbers first (`describe`, and the count of each size), then a picture: a bar chart of sizes.
"""),
    code("""
basket_sizes.describe().round(2)
"""),
    code("""
size_counts = basket_sizes.value_counts().sort_index()
size_counts.head(10)
"""),
    code("""
size_df = size_counts.reset_index()
size_df.columns = ["n_items", "baskets"]

(ggplot(size_df, aes(x="n_items", y="baskets"))
 + geom_col()
 + labs(title="How many items are in a basket?", x="items in the basket", y="number of baskets"))
"""),
    md("""
**What the code does.** `describe()` gives the five-number summary and the mean; `value_counts().sort_index()`
counts baskets of each size in size order; `geom_col` draws one bar per size.

**Expected output.** Mean **4.41** items, median **3**, quartiles 2 and 6, minimum 1, maximum **32**. The
distribution is strongly right-skewed: **2,159 baskets (22 %) contain a single item**, 1,643 have two, 1,299
three — more than half of all baskets have three items or fewer — and only a handful exceed 25.

**Points to make.** A single-item basket can never support a rule, and a two-item basket supports exactly one
pair. So the 43,367 item-rows carry far less *co-occurrence* evidence than the raw count suggests, which is why
support thresholds in this data end up around 1 %.

**Common mistakes.** `geom_histogram` on an integer variable with default bins — bars straddle the integers;
`geom_bar(stat="count")` on `basket_sizes` directly also works. Using `basket.describe()` (169 columns) instead of
`basket_sizes.describe()`.
"""),
]

Q4 = [
    md("""
### Solution 4 — item support and the 15 most popular items

**Approach.** Support of an item = share of baskets containing it = the mean of its boolean column. Sort, take
the top 15, and draw a horizontal bar chart.
"""),
    code("""
item_support = basket.mean().sort_values(ascending=False)
item_support.head(15).round(3)
"""),
    code("""
top15 = item_support.head(15).reset_index()
top15.columns = ["item", "support"]
top15["item"] = pd.Categorical(top15["item"], categories=top15["item"][::-1])   # most popular at the top after coord_flip

(ggplot(top15, aes(x="item", y="support"))
 + geom_col()
 + coord_flip()
 + labs(title="The 15 most popular items", x="", y="support (share of baskets)"))
"""),
    md("""
**What the code does.**
- `basket.mean()` averages each column; `True` counts as 1, so the mean is the share of baskets with the item.
- `pd.Categorical(..., categories=top15["item"][::-1])` fixes the plotting order (plotnine otherwise sorts the
  axis alphabetically); reversing it puts the most popular item at the **top** once `coord_flip()` turns the bars
  sideways.

**Expected output.** whole milk **0.256**, other vegetables 0.193, rolls/buns 0.184, soda 0.174, yogurt 0.140,
bottled water 0.111, root vegetables 0.109, tropical fruit 0.105, shopping bags 0.099, sausage 0.094, pastry
0.089, citrus fruit 0.083, bottled beer 0.081, newspapers 0.080, canned beer 0.078.

**Common mistakes.**
- `basket.sum()` — counts, not shares; divide by `len(basket)` or use `.mean()`.
- `long["item"].value_counts(normalize=True)` — the share of **item-rows**, not of baskets (whole milk would be
  5.8 %, not 25.6 %). Support is always "out of baskets".
- Bars in alphabetical order because the categorical order was not set.
"""),
]

Q5_MD = md("""
**Answer (part 5).** The most popular item is **whole milk**, in **25.6 %** of all baskets (2,513 of 9,835). The
next most common are other vegetables (19.3 %), rolls/buns (18.4 %), soda (17.4 %), and yogurt (14.0 %); after
that support drops quickly — the median item is in about 1 % of baskets and 81 of the 169 items are below 1 %.

Put that next to the basket sizes (median 3 items, 22 % single-item baskets): most baskets are small, and a
quarter of them contain whole milk regardless of what else is in them. So **any** item will show up with whole
milk fairly often *purely because whole milk is everywhere* — the rule {X} → {whole milk} will have a confidence
around 25 % even if X has nothing to do with milk. That is exactly the "popular items co-occur just by chance"
problem: confidence inherits the consequent's popularity. Lift divides it back out, which is why the rest of this
activity keeps asking for lift alongside confidence.
""")

Q6 = [
    md("""
### Solution 6 — confidence of {yogurt} → {whole milk}

**Approach.** confidence = support(yogurt **and** whole milk) ÷ support(yogurt): of the baskets with yogurt, what
share also contain whole milk?
"""),
    code("""
has_yogurt = basket["yogurt"]
has_milk = basket["whole milk"]

support_yogurt = has_yogurt.mean()
support_milk = has_milk.mean()
support_both = (has_yogurt & has_milk).mean()

confidence_y_m = support_both / support_yogurt
print(f"support(yogurt) = {support_yogurt:.4f}   support(whole milk) = {support_milk:.4f}   support(both) = {support_both:.4f}")
print(f"confidence(yogurt -> whole milk) = {confidence_y_m:.4f}")
"""),
    md("""
**What the code does.** `has_yogurt & has_milk` is `True` only for baskets with both items (`&` is the
element-wise *and* for boolean Series); its mean is the support of the pair. Dividing by the support of the
antecedent gives the confidence — equivalently, `has_milk[has_yogurt].mean()`.

**Expected output.** support(yogurt) 0.1395, support(whole milk) 0.2555, support(both) **0.0560**;
**confidence = 0.4016**.

**Interpretation.** About **40 % of the baskets that contain yogurt also contain whole milk.** On its own that
sounds strong — but compare it with the 25.6 % of *all* baskets that contain whole milk (part 7).

**Common mistakes.** Using `and` instead of `&` → the ambiguous-truth-value `ValueError`. Dividing by
support(whole milk) instead of support(yogurt) — that is the confidence of the reverse rule (part 8).
"""),
]

Q7 = [
    md("""
### Solution 7 — lift of {yogurt} → {whole milk}

**Approach.** lift = support(both) ÷ [support(yogurt) × support(whole milk)]: how many times more often do the two
appear together than if they were bought independently? Equivalently, confidence ÷ support(whole milk).
"""),
    code("""
lift_y_m = support_both / (support_yogurt * support_milk)
print(f"lift(yogurt -> whole milk) = {lift_y_m:.4f}")
print(f"check: confidence / support(whole milk) = {confidence_y_m / support_milk:.4f}")
"""),
    md("""
**What the code does.** If yogurt and milk were independent, the share of baskets with both would be
0.1395 × 0.2555 = 0.0356. The observed share is 0.0560; the ratio is the lift.

**Expected output.** **lift = 1.5717** (both ways).

**Interpretation.** Baskets with yogurt contain whole milk about **1.57 times as often as baskets in general**
(40.2 % versus 25.6 %). Lift above 1 means the two items are positively associated — yogurt does carry
information about milk, over and above milk's popularity.

**Common mistakes.** Reporting confidence as if it were lift ("yogurt buyers are 40 % more likely…"): 40 % is the
share, 57 % *more* likely is the lift − 1. Computing support(both) ÷ support(yogurt) ÷ support(milk) with
parentheses in the wrong place.
"""),
]

Q8 = [
    md("""
### Solution 8 — the reverse rule {whole milk} → {yogurt}

**Approach.** Same support of the pair; the confidence now divides by support(whole milk). The lift formula is
symmetric, so it does not change.
"""),
    code("""
confidence_m_y = support_both / support_milk
lift_m_y = support_both / (support_milk * support_yogurt)
print(f"confidence(whole milk -> yogurt) = {confidence_m_y:.4f}")
print(f"lift(whole milk -> yogurt)       = {lift_m_y:.4f}")
"""),
    md("""
**Expected output.** confidence **0.2193**, lift **1.5717**.

**Interpretation.** Only **22 % of whole-milk baskets contain yogurt** (because many milk baskets are small and
yogurt is less common than milk), yet the **lift is identical**: 22 % is still 1.57 times yogurt's overall
support of 14 %. Confidence depends on the direction of the rule; lift does not. For "what should we put next
to yogurt?" the direction matters (part 19).

**Common mistakes.** Expecting the lift to change; thinking the lower confidence means the association is weaker
in this direction — it is the same association, described from the other side.
"""),
]

Q9 = [
    md("""
### Solution 9 — `support()` and `rule_metrics()`

**Approach.** Package parts 6–8 into functions so they can be reused (Week 6). `support` handles a **list** of
items with `.all(axis=1)` ("every one of these columns is `True`"). `rule_metrics` calls `support` three times and
returns the three numbers as a Series so `apply` can stack them into a data frame (part 11).
"""),
    code("""
def support(items, onehot):
    \"\"\"
    Share of baskets that contain ALL of the items.

    Parameters
    ----------
    items : list of str
      Item names (columns of onehot).
    onehot : DataFrame of bool
      One row per basket, one column per item.

    Returns
    -------
    float
    \"\"\"
    return onehot[items].all(axis=1).mean()


def rule_metrics(antecedent, consequent, onehot):
    \"\"\"
    Support, confidence, and lift of the rule antecedent -> consequent.

    Parameters
    ----------
    antecedent, consequent : list of str
    onehot : DataFrame of bool

    Returns
    -------
    Series with index ["support", "confidence", "lift"]
    \"\"\"
    s_both = support(antecedent + consequent, onehot)
    s_ante = support(antecedent, onehot)
    s_cons = support(consequent, onehot)
    return pd.Series({"support": s_both,
                      "confidence": s_both / s_ante,
                      "lift": s_both / (s_ante * s_cons)})
"""),
    code("""
print(support(["yogurt"], basket), support(["whole milk"], basket), support(["yogurt", "whole milk"], basket))
rule_metrics(["yogurt"], ["whole milk"], basket)
"""),
    md("""
**What the code does.**
- `onehot[items]` selects the listed columns (a list inside the brackets); `.all(axis=1)` is one `True`/`False`
  per basket; `.mean()` is the share. For a one-item list it reduces to the column mean.
- `antecedent + consequent` concatenates the two lists — the itemset of the whole rule.
- Returning a **Series** (not a tuple or dict) is what lets `Series.apply` in part 11 produce a tidy table.

**Expected output.** `0.1395 0.2555 0.0560` and the Series support **0.0560**, confidence **0.4016**, lift
**1.5717** — identical to parts 6–7.

**Common mistakes.**
- `onehot[items].mean()` without `.all(axis=1)` → one support **per item**, not of the set.
- Passing strings instead of lists (`support("yogurt", basket)`) → `onehot["yogurt"]` is a Series and `.all(axis=1)`
  raises `ValueError: No axis named 1 for object type Series`. The docstring says *list*; `isinstance` checks
  from PA 6.1 would make the message friendlier.
- `antecedent & consequent` or `antecedent.append(consequent)` to combine the lists — the first is a `TypeError`,
  the second nests a list inside a list.
"""),
]

Q10_NOTE = md("""
**Instructor note (part 10, predictions).** Whole milk is in 25.6 % of baskets, so every rule {X} → {whole milk}
starts from a confidence "floor" of about 0.26 if X carries no information. Reasonable predictions: **yogurt** and
**other vegetables** are fresh-food staples bought on the same trips as milk → lift clearly above 1; **rolls/buns**
and **bottled water** are everyday items → lift modestly above 1; **soda** is a drink bought *instead of* milk or on
snack runs → confidence may look fine (~0.23) but lift near or **below 1**. Part 11 checks these.
""")

Q11 = [
    md("""
### Solution 11 — {X} → {whole milk} for five antecedents

**Approach.** `rule_metrics` handles one rule; `apply` with a lambda runs it for every X in the Series and stacks the
returned Series into a data frame. Label the rows with the antecedents and sort by lift.
"""),
    code("""
milk_rules = antecedents.apply(lambda x: rule_metrics([x], ["whole milk"], basket))
milk_rules.index = antecedents
milk_rules.sort_values("lift", ascending=False).round(4)
"""),
    md("""
**What the code does.** `Series.apply(f)` calls `f` once per element; because `f` returns a Series, the results
line up as rows of a DataFrame with the Series' index as columns. `[x]` wraps the string in a list, as the
function expects. Setting `.index = antecedents` replaces the 0–4 labels with the item names.

**Expected output (sorted by lift).**

| X | support | confidence | lift |
|---|---|---|---|
| yogurt | 0.0560 | 0.4016 | **1.5717** |
| other vegetables | 0.0748 | 0.3868 | **1.5136** |
| bottled water | 0.0344 | 0.3109 | 1.2169 |
| rolls/buns | 0.0566 | 0.3079 | 1.2050 |
| soda | 0.0401 | 0.2297 | **0.8991** |

**Compare with the predictions.** Yogurt and other vegetables lead; rolls/buns and bottled water are mildly
positive; **soda's lift is below 1**: baskets with soda contain whole milk *less* often (23 %) than baskets in
general (25.6 %). Its confidence of 0.23 would look acceptable on its own — the clearest illustration of why
confidence alone misleads.

**Common mistakes.**
- `rule_metrics(antecedents, ["whole milk"], basket)` with the whole Series → the support of all five items
  *together* (0), then a division by zero or `nan`.
- `antecedents.apply(rule_metrics, ...)` without the lambda → `TypeError: rule_metrics() missing 2 required
  positional arguments` (the Topic 6.1 / PA 6.2 pattern: the lambda adapts the arguments).
- Forgetting `[x]` → the string error from part 9.
"""),
]

Q12 = [
    md("""
### Solution 12 — frequent itemsets with `apriori`

**Approach.** `apriori` finds every set of items whose support is at least `min_support`. `use_colnames=True`
labels the itemsets with item names instead of column numbers. The itemsets are frozensets, so `len` gives the
number of items.
"""),
    code("""
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
frequent_itemsets["n_items"] = frequent_itemsets["itemsets"].apply(len)

print(frequent_itemsets.shape)
print(frequent_itemsets["n_items"].value_counts().sort_index())
frequent_itemsets.sort_values("support", ascending=False).head(8)
"""),
    md("""
**What the code does.** With `min_support=0.01` an itemset must appear in at least 1 % of baskets (≈ 99 baskets).
Apriori works level by level — single items, then pairs built from frequent single items, then triples — which is
why it is fast even with 169 items. `.apply(len)` counts the members of each frozenset.

**Expected output.** **333 frequent itemsets: 88 single items, 213 pairs, 32 triples**; no itemset of four or more
reaches 1 %. The most frequent itemsets are the popular single items (whole milk 0.256, …); the most frequent pair
is {other vegetables, whole milk} at 0.075.

**Common mistakes.**
- Omitting `use_colnames=True` → itemsets shown as column positions like `{18, 108}`.
- A `min_support` typed as `1` or `10` (thinking "percent") → an empty table; it is a share between 0 and 1.
- Passing the integer crosstab (part 1 without `> 0`) → a `DeprecationWarning` about non-bool input, and with
  counts above 1 wrong supports.
- `len(frequent_itemsets["itemsets"])` → 333 (number of itemsets), not the size of each.
"""),
]

Q13 = [
    md("""
### Solution 13 — rules with `make_rules`

**Approach.** `make_rules` is the reading's helper around mlxtend's `association_rules`: from the frequent
itemsets, form every rule *antecedent → consequent* and keep those whose confidence is at least the threshold.
It is defined here so the notebook is self-contained.
"""),
    code("""
def make_rules(frequent_itemsets, min_confidence):
    \"\"\"Association rules from frequent itemsets, keeping those with confidence >= min_confidence.\"\"\"
    return association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)


rules = make_rules(frequent_itemsets, 0.20)
print(len(rules), "rules")
rules.columns.tolist()
"""),
    md("""
**What the code does.** For each frequent itemset with at least two items, `association_rules` tries every split
into antecedent and consequent, computes the metrics from the supports already in `frequent_itemsets`, and keeps
the rules passing `metric="confidence", min_threshold=0.20`. The result has the frozenset columns `antecedents`
and `consequents`, then `antecedent support`, `consequent support`, `support`, `confidence`, `lift`, and several
further metrics (leverage, conviction, …) we do not use.

**Expected output.** **234 rules.**

**Common mistakes.**
- Calling `make_rules` on `basket` instead of `frequent_itemsets` → a `KeyError`/`ValueError` about missing
  `support`/`itemsets` columns.
- Reading `min_threshold` as a support threshold — it applies to whichever `metric` is named.
- A threshold of `20` → zero rules.
"""),
]

Q14 = [
    md("""
### Solution 14 — readable rules

**Approach.** Copy, then add three columns: the frozensets joined into text, and the number of baskets behind
each rule (support × number of baskets).
"""),
    code("""
rules_display = rules.copy()
rules_display["antecedent"] = rules_display["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules_display["consequent"] = rules_display["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules_display["transactions"] = (rules_display["support"] * len(basket)).round().astype(int)

show = ["antecedent", "consequent", "support", "confidence", "lift", "transactions"]
rules_display[show].head(8).round(3)
"""),
    md("""
**What the code does.** `", ".join(sorted(s))` turns a frozenset like `frozenset({'yogurt', 'tropical fruit'})`
into `"tropical fruit, yogurt"` (sorting makes the text deterministic). `support × 9835` is the number of baskets
containing every item of the rule; `.round().astype(int)` makes it a whole number.

**Expected output.** The first eight rules all have the antecedent **beef** or **berries**: beef → other
vegetables (support 0.020, confidence 0.376, lift 1.94, 194 baskets), beef → rolls/buns, beef → root vegetables
(lift **3.04**, 171 baskets), beef → whole milk, beef → yogurt, berries → other vegetables, berries → whole milk,
berries → yogurt (lift 2.28, 104 baskets).

**Common mistakes.**
- `str(s)` instead of joining → text like `frozenset({'beef'})`.
- `", ".join(s)` without `sorted` works but orders items arbitrarily.
- Forgetting `.copy()` and then being surprised that `rules` gained columns (harmless here, but the question asks
  for a copy).
- `rules["support"] * 9835` typed as a literal — use `len(basket)` so the code survives a different data set.
"""),
]

Q15 = [
    md("""
### Solution 15 — the 10 highest-confidence rules
"""),
    code("""
rules_display.sort_values("confidence", ascending=False)[show].head(10).round(3)
"""),
    md("""
**What they have in common.** Eight of the ten have the consequent **whole milk**, the other two **other
vegetables** — the two most popular items in the store. Every antecedent is a *pair* of fresh/dairy items
(root vegetables + tropical fruit, curd + yogurt, butter + other vegetables, …), and the top confidence is only
**0.586**: even the best rule is right a little over half the time. Each rule rests on roughly 100–140 baskets
(support 1.0–1.5 %).

**Why confidence alone is a poor guide.** Confidence is bounded below by the consequent's own support, so
popular consequents float to the top whatever the antecedent is. These rules say "people who buy two fresh
items also buy milk", which is close to "people buy milk". The lift column (2.0–3.0 here) is what shows the
antecedent adds information — and part 16 shows that sorting by lift brings up different consequents entirely.

**Common mistakes.** Sorting `rules` (frozensets) rather than `rules_display`; `ascending=True` by default →
the ten *lowest*; `nlargest(10, "confidence")` is a fine alternative.
"""),
]

Q16 = [
    md("""
### Solution 16 — the 10 highest-lift rules
"""),
    code("""
rules_display.sort_values("lift", ascending=False)[show].head(10).round(3)
"""),
    md("""
**Compared with the confidence list.** The consequents change: **root vegetables**, **whipped/sour cream**,
**yogurt**, and the pair {other vegetables, whole milk} replace plain whole milk. The top rules are produce
pairings — {citrus fruit, other vegetables} → root vegetables (lift **3.30**), {other vegetables, tropical fruit}
→ root vegetables (3.15), **beef → root vegetables (3.04)** — plus dairy pairings such as {curd, whole milk} →
yogurt (2.76). These are "cooking a meal" and "dairy aisle" baskets: items that genuinely travel together, not
items that are simply popular.

**How much evidence?** Look at `support` and `transactions`: most of these rules sit at **1.0–1.2 % support, about
100–120 baskets**; the strongest-evidence rules on the list are {other vegetables, whole milk} ↔ root vegetables
(**228 baskets**) and beef → root vegetables (171). A lift of 3.3 built on 102 baskets out of 9,835 is a real but
modest pattern — and some rules have confidence as low as 0.21–0.23 (e.g. → whipped/sour cream), so a high lift
can describe a rule that is still usually wrong. That is why part 17 filters on **both** confidence and lift.

**Common mistakes.** Treating lift as a probability; ignoring that the two rules with 228 transactions are the same
itemset read in two directions (identical lift, different confidence — part 8 again).
"""),
]

Q17 = [
    md("""
### Solution 17 — `interesting_rules`

**Approach.** Combine two boolean filters with `&`, then sort by lift.
"""),
    code("""
interesting_rules = (rules_display[(rules_display["confidence"] >= 0.30) & (rules_display["lift"] >= 1.5)]
                     .sort_values("lift", ascending=False))

print(len(interesting_rules), "rules")
interesting_rules[show].head(12).round(3)
"""),
    md("""
**What the code does.** Each comparison produces a boolean Series; `&` combines them element-wise (parentheses are
required around each comparison); the result indexes the rows to keep.

**Expected output.** **108 rules.** The top of the list: {citrus fruit, other vegetables} → root vegetables
(0.359, 3.30), {other vegetables, tropical fruit} → root vegetables (0.343, 3.15), beef → root vegetables
(0.331, 3.04), {citrus fruit, root vegetables} → other vegetables (0.586, 3.03), {root vegetables, tropical fruit}
→ other vegetables (0.585, 3.02), {other vegetables, whole milk} → root vegetables (0.310, 2.84, 228 baskets), {curd,
whole milk} → yogurt (0.385, 2.76), … Consequents are dominated by other vegetables (44 rules) and whole milk (49),
with yogurt (9), root vegetables (4), and rolls/buns (2).

**Common mistakes.** `and` instead of `&`; missing parentheses → `TypeError: unsupported operand type(s) for &`
because `0.30 & rules["lift"]` is evaluated first; filtering `rules` and then losing the text columns.
"""),
]

Q18_MD = md("""
**Answer (part 18) — one rule interpreted: {beef} → {root vegetables}.**

- **Support 0.017** — 1.7 % of all baskets, **171 baskets**, contain both beef and root vegetables. That is the
  evidence the rule rests on: not huge, but well above the 1 % floor.
- **Confidence 0.331** — of the baskets that contain beef, **33 % also contain root vegetables**. A shopper buying
  beef has about a one-in-three chance of also buying root vegetables (carrots, onions, potatoes — the makings of
  a stew or roast).
- **Lift 3.04** — root vegetables are in 10.9 % of baskets overall, so beef buyers pick them up **about three times
  as often as shoppers in general** (33 % vs 11 %). This is a strong, positive association, not a popularity effect:
  root vegetables are not an especially common item, so the confidence is *earned*.

Taken together: a genuine "cooking a meal" pattern with a moderate amount of evidence — a sensible candidate for
placing a root-vegetable display near the meat counter.
""")

Q19 = [
    md("""
### Solution 19 — increasing yogurt sales

**Approach.** To sell more yogurt, find the products whose buyers are unusually likely to *also* buy yogurt:
rules with **yogurt as the consequent**, ranked by lift, with enough support to matter. (Rules with yogurt as
the antecedent say what yogurt buyers add — useful for a different question.)
"""),
    code("""
to_yogurt = (rules_display[rules_display["consequent"] == "yogurt"]
             .sort_values("lift", ascending=False))

print(len(to_yogurt), "rules point to yogurt")
to_yogurt[show].head(12).round(3)
"""),
    code("""
# Single-item antecedents only — the easiest to act on (one product to pair with yogurt)
single = to_yogurt[~to_yogurt["antecedent"].str.contains(",")]
single[show].round(3)
"""),
    md("""
**What the code does.** `rules_display["consequent"] == "yogurt"` keeps rules whose *whole* consequent is yogurt;
`~ ... .str.contains(",")` keeps antecedents without a comma — single items.

**Expected output.** 32 rules point to yogurt. Single-item antecedents, by lift: **curd** (confidence 0.324, lift
2.33, 170 baskets), **berries** (0.318, 2.28, 104), **cream cheese** (0.313, 2.24, 122), **whipped/sour cream**
(0.289, 2.07, 204), **tropical fruit** (0.279, 2.00, **288 baskets**), butter (1.89), citrus fruit (1.88, 213),
fruit/vegetable juice (1.86), … down to whole milk (0.219, lift 1.57, but **551 baskets**) and bottled water
(1.49). Two-item antecedents go higher — {curd, whole milk} → yogurt lift 2.76, {tropical fruit, whole milk} →
yogurt 2.57 (149 baskets) — but are harder to act on.

**Recommendation.**
1. **The dairy case: curd, cream cheese, whipped/sour cream, butter.** Lift 1.9–2.3: their buyers are about twice as
   likely as average to buy yogurt. Shelve yogurt beside them (or cross-promote) — the natural "dairy basket".
2. **Fresh fruit: tropical fruit, berries, citrus fruit.** Lift 1.9–2.3 with the largest evidence among the
   high-lift items (tropical fruit 288 baskets, citrus 213). A "fruit + yogurt" breakfast promotion or a yogurt
   cooler at the fruit display.
3. **Whole milk** is the biggest *absolute* opportunity (551 shared baskets) but the weakest lift (1.57); a
   recommendation there mostly reaches people already likely to buy yogurt. Fine for a coupon printed with milk,
   weak as a "discovery" link.

Avoid items with lift near 1 (soda, shopping bags): they share baskets with yogurt only because they are common.

**Common mistakes.** Using rules with yogurt as the **antecedent** ("what do yogurt buyers also buy?") — those tell
you what to sell *to* yogurt buyers, not how to bring new buyers to yogurt. Ranking by confidence (whole milk →
yogurt would win on transactions but barely beats chance). Ignoring support and recommending a 99-basket rule as
the headline.
"""),
]

PA71_ANSWERS = [Q1, Q2, Q3, Q4, Q6, Q7, Q8, Q9, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q19, []]   # [] drops the stray last cell
PA71_MD_ANSWERS = [Q5_MD, Q18_MD]


# ============================================================================ #
#  Build
# ============================================================================ #

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def build_solution(src_path: Path) -> dict:
    nb = json.loads(src_path.read_text())
    out: list[dict] = []
    n_code = n_md = 0
    for cell in nb["cells"]:
        src = source(cell)
        if cell["cell_type"] == "code" and src.strip() == "":
            if n_code >= len(PA71_ANSWERS):
                raise SystemExit(f"{src_path.name}: more empty code cells than answer groups (at #{n_code})")
            out.extend(copy.deepcopy(PA71_ANSWERS[n_code]))
            n_code += 1
        elif cell["cell_type"] == "markdown" and src.strip() == "":
            if n_md >= len(PA71_MD_ANSWERS):
                raise SystemExit(f"{src_path.name}: more empty markdown cells than written answers (at #{n_md})")
            out.append(copy.deepcopy(PA71_MD_ANSWERS[n_md]))
            n_md += 1
        else:
            out.append(copy.deepcopy(cell))
            if cell["cell_type"] == "markdown" and src.lstrip().startswith("10."):
                out.append(copy.deepcopy(Q10_NOTE))                      # instructor note after the prediction prompt
    if (n_code, n_md) != (len(PA71_ANSWERS), len(PA71_MD_ANSWERS)):
        raise SystemExit(f"{src_path.name}: found {n_code} empty code / {n_md} empty markdown cells, "
                         f"expected {len(PA71_ANSWERS)} / {len(PA71_MD_ANSWERS)}")
    first_md = next(i for i, c in enumerate(out) if c["cell_type"] == "markdown")
    out[first_md]["source"] = re.sub(r"^#+[^\n]*", lambda m: PA71_TITLE, source(out[first_md]), count=1)
    out.insert(first_md + 1, copy.deepcopy(PA71_INTRO))
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
              and "raises-exception" not in c["metadata"].get("tags", [])
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"unexpected errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no unexpected errors)")


def main() -> None:
    dest = PA71_SRC.with_name(PA71_SRC.stem + "-solution.ipynb")
    save(dest, build_solution(PA71_SRC))
    print(f"wrote {dest.relative_to(ROOT)}")
    if "--no-exec" not in sys.argv:
        execute(dest)


if __name__ == "__main__":
    main()
