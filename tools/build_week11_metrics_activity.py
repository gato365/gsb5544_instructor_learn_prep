#!/usr/bin/env python3
r"""Generate the Week 11 Topic 11.2 activity (student notebook + separate instructor answer key).

Usage:  python3 tools/build_week11_metrics_activity.py

Topic 11.2 — Classification Metrics in Context: Cancer Screening and Fire Detection
  A guided, discussion-led classroom activity. Calculator only: the notebooks hold no code, so nothing is executed.
  Part 1 defines the confusion matrix and ten statistics (equation, plain-language definition, the question it
  answers, numerator and denominator, range and preferred direction). Parts 2 and 3 apply them to two hypothetical
  scenarios, each scored at two thresholds on the same 1,000 cases. Part 4 is the closing discussion (costs of the
  two kinds of error, thresholds and predicted probabilities, what else one would need to know).
  Every calculation question follows one pattern: equation -> meaning of the values -> calculation -> interpretation.
  It is not a fill-in-the-blank sheet: the student copy has labelled work space, not blanks.

Every number in the answer key is computed here from the four counts of each threshold (COUNTS below), so the
key cannot drift from the tables. In the text, ⟦name⟧ is replaced by the computed value of that name.

Cell kinds:
  md(...)       shown in both notebooks
  calc(...)     answer to a calculation question -> kept in the key, replaced by labelled work space for students
  discuss(...)  answer to a discussion question  -> kept in the key, replaced by a response prompt for students
  note(...)     instructor note                  -> kept in the key, dropped from the student notebook
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_11"
STEM = WEEK / "GSB5544_Topic_11_2_Classification_Metrics_in_Context"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"

# ============================================================================ #
#  The numbers
# ============================================================================ #
# prefix -> (TP, FP, FN, TN).  c = cancer screening, f = fire detection; A = more sensitive, B = less sensitive.
COUNTS = {
    "cA": (90, 180, 10, 720),
    "cB": (80, 45, 20, 855),
    "fA": (45, 95, 5, 855),
    "fB": (35, 19, 15, 931),
}


def d3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    s = f"{100 * x:.1f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


def num(x: float) -> str:
    return f"{x:g}"


V: dict[str, str] = {}
RAW: dict[str, dict[str, float]] = {}
for p, (TP, FP, FN, TN) in COUNTS.items():
    n = TP + FP + FN + TN
    assert n == 1000, p
    prec, rec, spec = TP / (TP + FP), TP / (TP + FN), TN / (TN + FP)
    m = {
        "acc": (TP + TN) / n, "err": (FP + FN) / n, "prec": prec, "rec": rec,
        "f1": 2 * TP / (2 * TP + FP + FN), "spec": spec, "fpr": FP / (FP + TN), "fnr": FN / (FN + TP),
        "npv": TN / (TN + FN), "bacc": (rec + spec) / 2,
        "base": (FP + TN) / n,                      # accuracy of "predict negative for everyone"
        "prev": (TP + FN) / n,
        "f1r": 2 * round(prec, 3) * round(rec, 3) / (round(prec, 3) + round(rec, 3)),   # F1 from rounded P and R
        "mean": (prec + rec) / 2,
        "fdr": FP / (TP + FP), "miss_neg": FN / (TN + FN),
    }
    RAW[p] = m
    counts = {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "N": n, "pos": TP + FN, "neg": FP + TN, "ppos": TP + FP,
              "pneg": TN + FN, "right": TP + TN, "wrong": FP + FN, "twoTP": 2 * TP, "f1den": 2 * TP + FP + FN}
    for k, v in counts.items():
        V[f"{p}_{k}"] = str(v)
    for k, v in m.items():
        V[f"{p}_{k}"] = d3(v)
        V[f"{p}_{k}_pct"] = pct(v)

for s in "cf":   # the same cases at both thresholds: the actual classes must not change
    assert V[f"{s}A_pos"] == V[f"{s}B_pos"] and V[f"{s}A_neg"] == V[f"{s}B_neg"]
    dFP = COUNTS[f"{s}A"][1] - COUNTS[f"{s}B"][1]
    dFN = COUNTS[f"{s}B"][2] - COUNTS[f"{s}A"][2]
    V[f"{s}_dFP"], V[f"{s}_dFN"], V[f"{s}_ratio"] = str(dFP), str(dFN), num(dFP / dFN)
    V[f"{s}_fn_times"] = num(COUNTS[f"{s}B"][2] / COUNTS[f"{s}A"][2])
    V[f"{s}_acc_gap"] = d3(RAW[f"{s}B"]["acc"] - RAW[f"{s}A"]["acc"])
    V[f"{s}_bacc_gap"] = d3(abs(RAW[f"{s}B"]["bacc"] - RAW[f"{s}A"]["bacc"]))
    V[f"{s}A_fp_per_fn"] = num(COUNTS[f"{s}A"][1] / COUNTS[f"{s}A"][2])

# facts the prose relies on; if the counts are ever edited, these say which sentences to revisit
assert RAW["cA"]["acc"] < RAW["cA"]["base"] and RAW["fA"]["acc"] < RAW["fA"]["base"]      # A loses to "flag no one"
assert RAW["cB"]["f1"] > RAW["cA"]["f1"] and RAW["fB"]["f1"] > RAW["fA"]["f1"]            # F1 favors B twice
assert RAW["cB"]["bacc"] > RAW["cA"]["bacc"] and RAW["fB"]["bacc"] < RAW["fA"]["bacc"]    # balanced accuracy splits
assert d3(RAW["cA"]["f1r"]) == d3(RAW["cA"]["f1"]) and d3(RAW["fA"]["f1r"]) != d3(RAW["fA"]["f1"])  # rounding note


def T(text: str) -> str:
    """Replace every ⟦name⟧ by its computed value; an unknown name is an error."""
    return re.sub(r"⟦(\w+)⟧", lambda mo: V[mo.group(1)], text)


# ============================================================================ #
#  Cell helpers
# ============================================================================ #
cells: list[dict] = []


def _add(kind: str, text: str) -> None:
    src = T(text.strip("\n"))
    src = re.sub(r"\n*(\$\$.+?\$\$)\n*", r"\n\n\1\n\n", src).strip("\n")   # display equations get their own paragraph
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src, "_kind": kind})


def md(text: str) -> None:
    _add("md", text)


def calc(text: str) -> None:
    _add("calc", "**Answer.** " + text.strip())


def discuss(text: str) -> None:
    _add("discuss", "**Suggested response.** " + text.strip())


def note(text: str) -> None:
    _add("note", "> **Instructor note.** " + text.strip().replace("\n", "\n> "))


CALC_SPACE = (
    "**Your work**\n\n"
    "*What the values represent in this scenario:*\n\n<br><br>\n\n"
    "*Substitution and calculation:*\n\n<br><br>\n\n"
    "*Interpretation in a complete sentence:*\n\n<br><br>"
)
DISCUSS_SPACE = "**Your response** *(two to four sentences)*\n\n<br><br><br>"

# The equations, written once and reused in Part 1 and in every calculation question.
EQ = {
    "acc": r"$$\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}$$",
    "err": r"$$\text{Error rate}=\frac{FP+FN}{TP+FP+FN+TN}$$",
    "prec": r"$$\text{Precision}=\frac{TP}{TP+FP}$$",
    "rec": r"$$\text{Recall}=\frac{TP}{TP+FN}$$",
    "f1": r"$$F_1=2\cdot\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}=\frac{2\,TP}{2\,TP+FP+FN}$$",
    "spec": r"$$\text{Specificity}=\frac{TN}{TN+FP}$$",
    "fpr": r"$$\text{False positive rate}=\frac{FP}{FP+TN}$$",
    "fnr": r"$$\text{False negative rate}=\frac{FN}{FN+TP}$$",
    "npv": r"$$\text{Negative predictive value}=\frac{TN}{TN+FN}$$",
    "bacc": r"$$\text{Balanced accuracy}=\frac{\text{Recall}+\text{Specificity}}{2}$$",
}
for _k, _v in EQ.items():
    V[f"EQ_{_k}"] = _v


def glance_table() -> str:
    """All ten statistics for the four threshold rows (answer key only)."""
    rows = [("Cancer A", "cA"), ("Cancer B", "cB"), ("Fire A", "fA"), ("Fire B", "fB")]
    keys = [("Accuracy", "acc"), ("Error rate", "err"), ("Precision", "prec"), ("Recall", "rec"), ("F1", "f1"),
            ("Specificity", "spec"), ("FPR", "fpr"), ("FNR", "fnr"), ("NPV", "npv"), ("Balanced accuracy", "bacc")]
    out = ["| Statistic | " + " | ".join(r for r, _ in rows) + " |", "|---|" + "---:|" * len(rows)]
    out.append("| TP / FP / FN / TN | " + " | ".join(" / ".join(str(c) for c in COUNTS[p]) for _, p in rows) + " |")
    for label, k in keys:
        out.append(f"| {label} | " + " | ".join(V[f"{p}_{k}"] for _, p in rows) + " |")
    out.append("| Accuracy of \"predict negative for everyone\" | " + " | ".join(V[f"{p}_base"] for _, p in rows) + " |")
    return "\n".join(out)


V["GLANCE"] = glance_table()

# ============================================================================ #
#  Front matter
# ============================================================================ #
md(r"""
# GSB 5544 — Topic 11.2: Classification Metrics in Context — Cancer Screening and Fire Detection — INSTRUCTOR ANSWER KEY
*The student activity with every calculation worked, an interpretation for each, and suggested discussion responses.*
""")

note(r"""
**Running the session.** About 75 minutes as written: Part 1 ≈ 15, Part 2 ≈ 25, Part 3 ≈ 20, Part 4 ≈ 15. For a
50-minute class, do Part 1 quickly from the cards, Part 2 through Q2.10, and in Part 3 only Q3.1, Q3.2, Q3.4,
Q3.5, Q3.7 and Q3.8 before Part 4; assign the "going further" questions as follow-up. Students work each question alone or in pairs for two or three minutes, then you take
answers. Insist on the full pattern every time (what the values mean → substitution → number → sentence); the
sentence is where the misunderstandings show.
**Where the discussion is headed.** In both scenarios threshold B has the higher accuracy and the higher F1, and
threshold A has the higher recall. Neither is "the answer": the activity is built so that the summary numbers do
not settle the choice, and the costs of the two errors and what happens after a positive prediction do.
""")

md(r"""
**All four thresholds at a glance** (three decimals):

⟦GLANCE⟧
""")
cells[-1]["_kind"] = "note"   # the table of results is for the key only

md(r"""
## How this activity works

- **Calculator only.** There is no code in this activity.
- **Every calculation question has the same four steps.** The equation is given → say what its values mean in the
  scenario → substitute and calculate → interpret the result in one complete sentence. A number without the
  sentence is half an answer.
- **Round to three decimals** (0.333, not 0.3).
- **All counts are hypothetical.** They were made up for teaching and are not real medical or fire-safety statistics.
- Work each question alone or with a neighbor first; we then discuss it as a class.

| Part | What we do |
|---|---|
| **1. Understanding the statistics** | the confusion matrix and ten statistics built from it |
| **2. Cancer screening** | one screening test, two thresholds, the same 1,000 people |
| **3. Fire detection** | one detection system, two thresholds, the same 1,000 observation periods |
| **4. Final discussion** | which mistakes matter, where thresholds come from, what else we would need to know |
""")

# ============================================================================ #
#  Part 1
# ============================================================================ #
md(r"""
---
# Part 1 — Understanding the statistics

## 1.1 The confusion matrix

A classifier puts each case into one of two classes. We choose one class to call **positive** — the thing we are
trying to detect — and the other is **negative**. Comparing what the model **predicted** with what was **actually**
true puts every case into one of four cells:

| | predicted **positive** | predicted **negative** | row total |
|---|---|---|---|
| actually **positive** | **TP** — true positive | **FN** — false negative | actual positives = TP + FN |
| actually **negative** | **FP** — false positive | **TN** — true negative | actual negatives = FP + TN |
| **column total** | predicted positives = TP + FP | predicted negatives = FN + TN | all cases = TP + FP + FN + TN |

- **True positive (TP):** the case is positive and the model predicted positive. A correct detection.
- **False positive (FP):** the case is negative but the model predicted positive. A **false alarm**.
- **True negative (TN):** the case is negative and the model predicted negative. Correctly left alone.
- **False negative (FN):** the case is positive but the model predicted negative. A **miss**.

How to read the names: the second word (**positive / negative**) is what the model *predicted*; the first word
(**true / false**) says whether that prediction was *correct*.

This is the layout from Topic 11.1. (scikit-learn's `confusion_matrix` prints the negative class first; the four
counts are the same.)
""")

md(r"""
**Q1.1.** Without using any formula: (a) which two cells are the model's mistakes, and how do the two mistakes
differ? (b) Which two cells together are "every case that really is positive"? (c) Which two cells together are
"every case the model called positive"? (d) Why are the groups in (b) and (c) not the same group?
""")
discuss(r"""
(a) FP and FN. A false positive is a false alarm (the model flagged a case that is negative); a false negative is
a miss (the model passed over a case that is positive). (b) TP + FN, the top row. (c) TP + FP, the first column.
(d) They share only the TP cell. The actual positives also include the cases the model missed (FN); the predicted
positives also include the false alarms (FP). *This is the idea the rest of the activity rests on: recall divides
by (b), precision divides by (c).*
""")

md(r"""
## 1.2 Ten statistics

Each statistic is a fraction built from the four counts. For every one, ask **"out of which group?"** — that is
the denominator, and it is what separates one statistic from another.

### 1. Accuracy
⟦EQ_acc⟧
- **In plain language:** the share of all cases the model classified correctly.
- **The question it answers:** *Of everything the model classified, how much did it get right?*
- **Numerator:** the correct predictions of both kinds (TP + TN). **Denominator:** every case.
- **Range:** 0 to 1. Higher is generally preferred.

### 2. Error rate
⟦EQ_err⟧
- **In plain language:** the share of all cases the model classified incorrectly. Error rate = 1 − accuracy.
- **The question it answers:** *Of everything the model classified, how much did it get wrong?*
- **Numerator:** the mistakes of both kinds (FP + FN), counted as equally bad. **Denominator:** every case.
- **Range:** 0 to 1. Lower is generally preferred.

### 3. Precision
⟦EQ_prec⟧
- **In plain language:** of the cases the model called positive, the share that really are positive.
- **The question it answers:** *When the model says "positive", how often is it right?*
- **Numerator:** correct positive predictions (TP). **Denominator:** all positive predictions (TP + FP), correct or not.
- **Range:** 0 to 1. Higher is generally preferred. (Undefined if the model never predicts positive.)

### 4. Recall (also called sensitivity)
⟦EQ_rec⟧
- **In plain language:** of the cases that really are positive, the share the model found.
- **The question it answers:** *Of all the real positives, how many did the model catch?*
- **Numerator:** positives the model found (TP). **Denominator:** all actual positives (TP + FN), found or missed.
- **Range:** 0 to 1. Higher is generally preferred.

### 5. F1 score
⟦EQ_f1⟧
- **In plain language:** one number that combines precision and recall (their harmonic mean). It is high only
  when **both** are high, and it sits closer to the smaller of the two.
- **The question it answers:** *Is the model doing well on precision and recall at the same time?*
- **Numerator and denominator (count form):** the numerator counts the true positives twice; the denominator adds
  the two kinds of mistakes (FP + FN) to that. True negatives appear nowhere.
- **Range:** 0 to 1. Higher is generally preferred — when precision and recall matter equally, which is an
  assumption, not a fact.

### 6. Specificity
⟦EQ_spec⟧
- **In plain language:** of the cases that really are negative, the share the model correctly called negative.
- **The question it answers:** *Of all the real negatives, how many did the model correctly leave alone?*
- **Numerator:** negatives correctly called negative (TN). **Denominator:** all actual negatives (TN + FP).
- **Range:** 0 to 1. Higher is generally preferred.

### 7. False positive rate (FPR)
⟦EQ_fpr⟧
- **In plain language:** of the cases that really are negative, the share the model wrongly flagged. FPR = 1 − specificity.
- **The question it answers:** *How often does a real negative set off a false alarm?*
- **Numerator:** false alarms (FP). **Denominator:** all actual negatives (FP + TN).
- **Range:** 0 to 1. Lower is generally preferred.

### 8. False negative rate (FNR)
⟦EQ_fnr⟧
- **In plain language:** of the cases that really are positive, the share the model missed. FNR = 1 − recall.
- **The question it answers:** *How often does a real positive slip through?*
- **Numerator:** misses (FN). **Denominator:** all actual positives (FN + TP).
- **Range:** 0 to 1. Lower is generally preferred.

### 9. Negative predictive value (NPV)
⟦EQ_npv⟧
- **In plain language:** of the cases the model called negative, the share that really are negative.
- **The question it answers:** *When the model says "negative", how often is it right?*
- **Numerator:** correct negative predictions (TN). **Denominator:** all negative predictions (TN + FN), correct or not.
- **Range:** 0 to 1. Higher is generally preferred.

### 10. Balanced accuracy
⟦EQ_bacc⟧
- **In plain language:** the average of the model's success rate on the positive class (recall) and on the
  negative class (specificity). Each class counts equally, however many cases it has.
- **The question it answers:** *How well does the model do on each class, when a small class matters as much as a large one?*
- **Numerator and denominator:** it is an average of two fractions, not one fraction. Recall is measured out of the
  actual positives, specificity out of the actual negatives; dividing by 2 averages them.
- **Range:** 0 to 1. Higher is generally preferred. A model that predicts the same class for every case scores 0.5.
""")

md(r"""
## 1.3 Three things to notice

**1. Precision and recall have the same numerator and different denominators.** Both start from TP. Precision
divides by the cases the model *called* positive (TP + FP); recall divides by the cases that *are* positive
(TP + FN). False alarms lower precision and leave recall untouched; misses lower recall and leave precision
untouched. So a model can be strong on one and weak on the other.

| Denominator — "out of which group?" | Statistics that use it |
|---|---|
| all cases | accuracy, error rate |
| actual positives (TP + FN) | **recall**, false negative rate |
| actual negatives (TN + FP) | specificity, false positive rate |
| predicted positives (TP + FP) | **precision** |
| predicted negatives (TN + FN) | negative predictive value |

**2. F1 combines precision and recall.** It gives the two equal weight and is pulled toward the smaller one, so
a high F1 needs both. It does not say which of the two is low, and it ignores true negatives.

**3. Accuracy can mislead when one class is much more common.** When nearly every case is negative, a model can
be right almost all the time by predicting "negative" for everything — and never detect a single positive.
""")

md(r"""
**Q1.2.** Precision and recall both have TP on top. Describe in words the group each one divides by. Then
describe a model that would have high recall and low precision, and one that would have high precision and low
recall.
""")
discuss(r"""
Precision divides by everything the model *flagged* (true positives plus false alarms); recall divides by
everything that *really is* positive (true positives plus misses). High recall, low precision: a model that flags
very readily — it catches almost every positive but buries them in false alarms (flag everyone and recall is 1).
High precision, low recall: a model that flags only when it is nearly certain — its few flags are almost always
right, but it misses many real positives.
""")

md(r"""
**Q1.3.** Three pairs of statistics in the list always add up to exactly 1. Find the three pairs, and explain why
each pair must add to 1 by pointing to what the two fractions share.
""")
discuss(r"""
Accuracy + error rate (both out of all cases: every case is either right or wrong). Recall + false negative rate
(both out of the actual positives: each is either found or missed). Specificity + false positive rate (both out of
the actual negatives: each is either left alone or falsely flagged). In every pair the denominators are the same
group and the numerators split that group in two. *Precision and NPV have no partner in this list; their
denominators are predicted groups, which no other statistic here shares.*
""")

md(r"""
**Q1.4.** A data set has 1,000 cases, of which 20 are positive. A "model" predicts **negative** for every case.
(a) Fill in the four cells of its confusion matrix in words or numbers. (b) Use the equations to find its accuracy
and its recall. (c) What does the comparison tell you about accuracy? (d) What happens when you try to compute
its precision, and what is its balanced accuracy?
""")
calc(r"""
(a) TP = 0, FP = 0, FN = 20, TN = 980. (b) Accuracy $=\frac{0+980}{1000}=0.980$; recall $=\frac{0}{0+20}=0.000$.
(c) The model is "98% accurate" and detects nothing: accuracy is high only because negatives are common. Always
compare accuracy with what predicting the common class for everyone would score. (d) Precision is $\frac{0}{0+0}$,
which is undefined — the model made no positive predictions to be right or wrong about. Specificity is
$\frac{980}{980}=1$, so balanced accuracy $=\frac{0+1}{2}=0.5$, which exposes the model that accuracy flattered.
""")

# ============================================================================ #
#  Part 2 — cancer screening
# ============================================================================ #
md(r"""
---
# Part 2 — Cancer screening

A clinic uses a screening test for a particular cancer. For each person the test produces a risk score, and the
clinic chooses a **threshold**: a score above it is a positive screen.

- **Positive** means the person **has cancer**. Negative means the person does not.
- A **positive screen** means the person is referred for **follow-up diagnostic testing** (for example further
  imaging or a biopsy). It is not a diagnosis. **A positive screening result does not mean the person receives
  chemotherapy**; treatment is decided only after a diagnosis has been confirmed.

The same 1,000 people were screened and their scores were classified at two thresholds. Their true status was
established later. These results are **hypothetical**.

| Threshold | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| **A: More sensitive** | 90 | 180 | 10 | 720 |
| **B: Less sensitive** | 80 | 45 | 20 | 855 |
""")

md(r"""
### What the numbers mean

**Q2.1.** For threshold A, say in a sentence each what the 90, the 180, the 10 and the 720 represent — describe
the people, not the letters. Then: how many of the 1,000 people have cancer according to row A? According to row B?
Why must those two answers agree?
""")
discuss(r"""
**TP = 90:** people who have cancer and screened positive (correctly referred). **FP = 180:** people who do not
have cancer but screened positive (referred for follow-up they did not need). **FN = 10:** people who have cancer
but screened negative (missed). **TN = 720:** people who do not have cancer and screened negative. People with
cancer: 90 + 10 = ⟦cA_pos⟧ in row A and 80 + 20 = ⟦cB_pos⟧ in row B; without: 180 + 720 = 45 + 855 = ⟦cA_neg⟧. They agree
because these are the same people — changing the threshold changes the *predictions*, never who actually has
cancer. It moves people between TP and FN, and between FP and TN. (⟦cA_prev_pct⟧ of this group has cancer.)
""")

md(r"""
**Q2.2.** Describe what happens to a person who is a **false positive** in this scenario, and what happens to a
person who is a **false negative**. Which error seems more serious for the individual, and why?
""")
discuss(r"""
**False positive:** a person without cancer is told the screen was abnormal and is sent for follow-up testing —
worry while waiting, time and cost, and a small risk from the follow-up procedure itself. The follow-up test will
most likely clear them. They are not given chemotherapy. **False negative:** a person with cancer is reassured and
sent home. The cancer goes undetected until symptoms appear or the next screening, when it may be more advanced
and harder to treat. Most students will judge the false negative more serious for the individual, since nothing
follows a negative screen to catch the mistake. Hold that thought, and note that at threshold A there are
⟦cA_fp_per_fn⟧ false positives for every false negative.
""")

md(r"""
### Threshold A

**Q2.3 — Accuracy.**

⟦EQ_acc⟧

In this scenario, what do TP + TN represent, and what does the denominator represent? Substitute the threshold A
values, calculate accuracy, and interpret your answer in a complete sentence. Then: a "screen" that tells all
1,000 people they do not have cancer would be correct for how many of them? What does that comparison tell you?
""")
calc(r"""
*Values:* TP + TN = ⟦cA_TP⟧ + ⟦cA_TN⟧ = ⟦cA_right⟧ people the screen classified correctly (cancer found, or correctly cleared);
the denominator is all ⟦cA_N⟧ people screened. *Calculation:* $\frac{90+720}{1000}=\frac{810}{1000}=⟦cA_acc⟧$.
*Interpretation:* At threshold A the screen gives the correct result for ⟦cA_acc_pct⟧ of the 1,000 people.
*Comparison:* telling everyone "no cancer" is correct for the ⟦cA_neg⟧ people without cancer — accuracy ⟦cA_base⟧,
*higher* than threshold A's ⟦cA_acc⟧, while finding none of the 100 cancers. With only ⟦cA_prev_pct⟧ of people positive,
accuracy mostly rewards calling people negative, so it cannot be the statistic that judges this screen.
""")

md(r"""
**Q2.4 — Error rate.**

⟦EQ_err⟧

In this scenario, what do FP and FN represent? Substitute the threshold A values, calculate the error rate, and
interpret your answer in a complete sentence. How many of the errors are of each kind, and does the error rate
show that?
""")
calc(r"""
*Values:* FP = ⟦cA_FP⟧ people without cancer who screened positive; FN = ⟦cA_FN⟧ people with cancer who screened negative.
*Calculation:* $\frac{180+10}{1000}=\frac{190}{1000}=⟦cA_err⟧$ (which is 1 − ⟦cA_acc⟧). *Interpretation:* At threshold A
the screen gives the wrong result for ⟦cA_err_pct⟧ of the 1,000 people. Of the ⟦cA_wrong⟧ errors, 180 are false positives and
10 are false negatives; the error rate adds them together as if a needless referral and a missed cancer were the
same mistake, so it hides the difference that matters most here.
""")

md(r"""
**Q2.5 — Precision.**

⟦EQ_prec⟧

In this scenario, what do TP and FP represent, and which group of people is TP + FP? Substitute the threshold A
values, calculate precision, and interpret your answer in a complete sentence.
""")
calc(r"""
*Values:* TP = ⟦cA_TP⟧ people with cancer who screened positive; FP = ⟦cA_FP⟧ people without cancer who screened positive;
TP + FP = ⟦cA_ppos⟧ is everyone who screened positive. *Calculation:* $\frac{90}{90+180}=\frac{90}{270}=⟦cA_prec⟧$.
*Interpretation:* Of the 270 people who screen positive at threshold A, about one in three (⟦cA_prec_pct⟧) actually has
cancer; the other two in three are false alarms. *Worth saying aloud:* this is the number a person holding a
positive result cares about — a positive screen here means "a one-in-three chance", not "you have cancer".
""")

md(r"""
**Q2.6 — Recall.**

⟦EQ_rec⟧

In this scenario, what do TP and FN represent, and which group of people is TP + FN? Substitute the threshold A
values, calculate recall, and interpret your answer in a complete sentence. Then compare the denominators you used
in Q2.5 and Q2.6.
""")
calc(r"""
*Values:* TP = ⟦cA_TP⟧ people with cancer who screened positive; FN = ⟦cA_FN⟧ people with cancer who screened negative;
TP + FN = ⟦cA_pos⟧ is everyone who has cancer. *Calculation:* $\frac{90}{90+10}=\frac{90}{100}=⟦cA_rec⟧$.
*Interpretation:* At threshold A the screen detects ⟦cA_rec_pct⟧ of the people who have cancer and misses the other ⟦cA_fnr_pct⟧.
*Denominators:* precision was out of the ⟦cA_ppos⟧ people who screened positive; recall is out of the ⟦cA_pos⟧ people who
have cancer. Same numerator (90), different groups, very different answers (⟦cA_prec⟧ and ⟦cA_rec⟧).
""")

md(r"""
**Q2.7 — F1 score.**

⟦EQ_f1⟧

Which two statistics does F1 combine, and which of the four counts does it leave out? Substitute your threshold A
values, calculate F1, and interpret your answer in a complete sentence. Is F1 closer to your precision or to your
recall?
""")
calc(r"""
*Values:* F1 combines precision (⟦cA_prec⟧) and recall (⟦cA_rec⟧); the true negatives — the ⟦cA_TN⟧ people correctly cleared —
are left out. *Calculation:* $2\cdot\frac{0.333\times 0.900}{0.333+0.900}=⟦cA_f1⟧$, or from the counts
$\frac{2(90)}{2(90)+180+10}=\frac{180}{370}=⟦cA_f1⟧$. *Interpretation:* Treating precision and recall as equally
important, threshold A scores ⟦cA_f1⟧ — low, because the low precision pulls it down despite the high recall. It is
closer to precision (⟦cA_prec⟧) than to recall (⟦cA_rec⟧); the ordinary average would have been ⟦cA_mean⟧. *Note:* unlike the
other statistics, F1 is not "a share of some group of people", so the sentence has to describe it as a combined
score.
""")

md(r"""
### Threshold B

**Q2.8 — Precision and recall.**

⟦EQ_prec⟧

⟦EQ_rec⟧

The meaning of TP, FP and FN has not changed; their values have. Substitute the threshold B values, calculate
precision and recall, and interpret each in a complete sentence.
""")
calc(r"""
*Values:* at threshold B, TP = ⟦cB_TP⟧, FP = ⟦cB_FP⟧, FN = ⟦cB_FN⟧; ⟦cB_ppos⟧ people screen positive and ⟦cB_pos⟧ people have cancer.
*Calculation:* precision $=\frac{80}{80+45}=\frac{80}{125}=⟦cB_prec⟧$; recall $=\frac{80}{80+20}=\frac{80}{100}=⟦cB_rec⟧$.
*Interpretation:* At threshold B, ⟦cB_prec_pct⟧ of the people who screen positive actually have cancer, and the screen
detects ⟦cB_rec_pct⟧ of the people who have cancer. (For reference: accuracy ⟦cB_acc⟧, F1 ⟦cB_f1⟧.)
""")

md(r"""
### Comparing the thresholds

**Q2.9.** Going from threshold A to threshold B, what happened to the number of false positives, the number of
false negatives, precision and recall? Put the trade in one sentence ("B avoids … at the price of …"). In what
circumstances would recall matter more than precision for a screening test?
""")
discuss(r"""
A → B: false positives fall from 180 to 45 (⟦c_dFP⟧ fewer), false negatives rise from 10 to 20 (⟦c_dFN⟧ more); precision
rises from ⟦cA_prec⟧ to ⟦cB_prec⟧ and recall falls from ⟦cA_rec⟧ to ⟦cB_rec⟧. *B avoids 135 unnecessary referrals at the price
of 10 more missed cancers* — ⟦c_ratio⟧ referrals avoided per additional miss. Recall matters more when a miss is far
costlier than a false alarm and the false alarms can be sorted out cheaply and safely afterwards: the cancer is
much more treatable when caught early, a negative screen ends the process (nobody re-checks the negatives), and a
reliable follow-up test stands behind every positive. *Do not let the class crown A.* B has the higher accuracy
(⟦cB_acc⟧ vs ⟦cA_acc⟧) and the higher F1 (⟦cB_f1⟧ vs ⟦cA_f1⟧); A has the higher recall. Whether ⟦c_ratio⟧ avoided referrals are
worth one missed cancer depends on facts the table does not contain.
""")

md(r"""
**Q2.10.** Suppose the clinic decides that missing a cancer is the worse error. Why do the false positives still
matter? In your answer, say what follows a positive screen, and explain how the follow-up test — its accuracy,
cost, risk and waiting time — changes how costly a false positive is.
""")
discuss(r"""
At threshold A, 180 people without cancer are referred. Each faces worry, cost, lost time and whatever risk the
follow-up procedure carries; together they use appointments and specialists that the 90 true positives also need,
and people who experience false alarms may trust the program less and skip the next screening. A screen that
ignored false positives entirely would refer everyone (recall 1, precision ⟦cA_prev⟧) and would not be a screen. What
follows a positive screen is **diagnostic testing, not treatment** — nobody receives chemotherapy on the strength
of a screening result. That is what makes low precision tolerable: if the follow-up is accurate, quick, cheap and
low-risk, a false positive is corrected at modest cost, and a sensitive threshold like A is easier to defend. If
the follow-up is invasive, expensive or has a long queue, each false positive costs more and may delay care for
true positives, which pushes toward B. The threshold cannot be chosen without knowing the step that comes after it.
""")

md(r"""
### Going further: the other five statistics

**Q2.11 — Specificity and false positive rate (threshold A).**

⟦EQ_spec⟧

⟦EQ_fpr⟧

In this scenario, what do TN and FP represent, and which group of people do the two statistics share as their
denominator? Substitute the threshold A values, calculate both, and interpret each in a complete sentence. How are
the two related?
""")
calc(r"""
*Values:* TN = ⟦cA_TN⟧ people without cancer who screened negative; FP = ⟦cA_FP⟧ people without cancer who screened positive;
the shared denominator, ⟦cA_neg⟧, is everyone who does not have cancer. *Calculation:* specificity
$=\frac{720}{720+180}=\frac{720}{900}=⟦cA_spec⟧$; FPR $=\frac{180}{180+720}=\frac{180}{900}=⟦cA_fpr⟧$.
*Interpretation:* At threshold A, ⟦cA_spec_pct⟧ of the people without cancer are correctly cleared, and ⟦cA_fpr_pct⟧ of them are
referred unnecessarily. They add to 1. *Contrast to draw out:* "20% of people without cancer screen positive"
(FPR, out of 900) and "⟦cA_fdr_pct⟧ of positive screens are people without cancer" (1 − precision, out of 270) describe
the same 180 people against different groups.
""")

md(r"""
**Q2.12 — False negative rate (thresholds A and B).**

⟦EQ_fnr⟧

In this scenario, what does FN represent, and which group is FN + TP? Substitute, calculate the false negative
rate at each threshold, and interpret the change from A to B in a complete sentence. How does this statistic relate
to recall?
""")
calc(r"""
*Values:* FN is the number of people with cancer who screened negative (10 at A, 20 at B); FN + TP = ⟦cA_pos⟧ is everyone
who has cancer. *Calculation:* A: $\frac{10}{10+90}=⟦cA_fnr⟧$; B: $\frac{20}{20+80}=⟦cB_fnr⟧$. *Interpretation:* Moving
from threshold A to threshold B doubles the share of cancers the screen misses, from ⟦cA_fnr_pct⟧ to ⟦cB_fnr_pct⟧. It is
1 − recall: the same information as recall, stated as the harm instead of the success.
""")

md(r"""
**Q2.13 — Negative predictive value (threshold A).**

⟦EQ_npv⟧

In this scenario, what do TN and FN represent, and which group of people is TN + FN? Substitute the threshold A
values, calculate NPV, and interpret your answer in a complete sentence. Then: before any screening, what share
of these 1,000 people do not have cancer? What does that tell you about how to read a high NPV?
""")
calc(r"""
*Values:* TN = ⟦cA_TN⟧ people without cancer who screened negative; FN = ⟦cA_FN⟧ people with cancer who screened negative;
TN + FN = ⟦cA_pneg⟧ is everyone who screened negative. *Calculation:* $\frac{720}{720+10}=\frac{720}{730}=⟦cA_npv⟧$.
*Interpretation:* Of the people who screen negative at threshold A, ⟦cA_npv_pct⟧ really do not have cancer; about
⟦cA_miss_neg_pct⟧ (10 of 730) have a cancer the screen missed. *Reading it:* ⟦cA_base_pct⟧ of the group is cancer-free before
anyone is screened, so NPV starts from 0.900, not from 0. The screen raises a negative person's reassurance from
0.900 to ⟦cA_npv⟧ (threshold B: ⟦cB_npv⟧). When the condition is rare, NPV is high almost automatically; judge it
against the share of negatives, not against 1.
""")

md(r"""
**Q2.14 — Balanced accuracy (thresholds A and B).**

⟦EQ_bacc⟧

Which group of people does recall describe, and which group does specificity describe? Use your recall and
specificity for threshold A to calculate balanced accuracy. For threshold B, first calculate specificity, then
balanced accuracy. Interpret the two results in a complete sentence, and compare them with accuracy at the two
thresholds (accuracy at B is ⟦cB_acc⟧).
""")
calc(r"""
*Values:* recall is the success rate among the ⟦cA_pos⟧ people with cancer; specificity is the success rate among the
⟦cA_neg⟧ people without. *Calculation:* A: $\frac{0.900+0.800}{2}=⟦cA_bacc⟧$. B: specificity
$=\frac{855}{855+45}=\frac{855}{900}=⟦cB_spec⟧$, so $\frac{0.800+0.950}{2}=⟦cB_bacc⟧$. *Interpretation:* Giving people
with and without cancer equal weight, the screen is right for an average of ⟦cA_bacc_pct⟧ of each group at threshold A
and ⟦cB_bacc_pct⟧ at threshold B. *Comparison:* accuracy separates the thresholds by ⟦c_acc_gap⟧ (⟦cA_acc⟧ vs ⟦cB_acc⟧) because
the 900 people without cancer dominate it; balanced accuracy separates them by only ⟦c_bacc_gap⟧. And "tell everyone
no cancer", which had accuracy ⟦cA_base⟧, has balanced accuracy 0.5.
""")

md(r"""
**Q2.15.** Precision, recall and F1 never use the true negatives. What did specificity, the false positive rate,
the false negative rate, NPV and balanced accuracy each tell you about this screening test that those three did
not? Whose point of view does each one take?
""")
discuss(r"""
**Specificity / FPR** describe the experience of the 900 people who do not have cancer — the large majority —
which precision only hints at. **FNR** states the misses directly: "one cancer in ten" (A) or "one in five" (B) is
the same fact as recall, in the form a patient-safety discussion uses. **NPV** takes the point of view of a
person holding a negative result: how much reassurance is it worth? **Balanced accuracy** gives a one-number
summary that the common class cannot dominate, repairing the specific flaw of accuracy. None of them says what an
error *costs*; they describe the errors more completely, and the judgment still has to be made.
""")

# ============================================================================ #
#  Part 3 — fire detection
# ============================================================================ #
md(r"""
---
# Part 3 — Fire detection

A facilities company monitors its buildings with an automated fire-detection system. For each observation period
(a fixed window of time in one monitored area) the system produces a fire-risk score, and the company chooses a
**threshold**: a score above it raises an alarm.

- **Positive** means **a fire occurred** during the observation period. Negative means no fire occurred.
- A **positive prediction** means the system **raised an alarm** for that period.

The same 1,000 observation periods were scored and classified at two thresholds. Whether a fire really occurred
was established afterwards. These results are **hypothetical**.

| Threshold | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| **A: More sensitive** | 45 | 95 | 5 | 855 |
| **B: Less sensitive** | 35 | 19 | 15 | 931 |
""")

md(r"""
### What the numbers mean

**Q3.1.** For threshold A, say in a sentence each what the 45, the 95, the 5 and the 855 represent — describe
what happened in those observation periods. How many of the 1,000 periods had a fire? Check that row B agrees.
""")
discuss(r"""
**TP = 45:** periods in which a fire occurred and the alarm was raised. **FP = 95:** periods with no fire in which
the alarm was raised anyway (false alarms). **FN = 5:** periods in which a fire occurred and no alarm was raised
(undetected fires). **TN = 855:** periods with no fire and no alarm. Fires: 45 + 5 = ⟦fA_pos⟧ at A and 35 + 15 = ⟦fB_pos⟧ at
B; no fire: 95 + 855 = 19 + 931 = ⟦fA_neg⟧. Same periods, same fires; only the alarms differ. (⟦fA_prev_pct⟧ of periods had
a fire — an even rarer positive class than in Part 2.)
""")

md(r"""
**Q3.2.** Describe what happens after a **false positive** in this scenario, and after a **false negative**.
Who bears the cost of each? How is the comparison similar to cancer screening, and how is it different?
""")
discuss(r"""
**False positive:** an alarm with no fire — people stop work or evacuate, someone is sent to check, perhaps the
fire service is called out. The cost is disruption, money and responders who are unavailable for a real emergency;
it is borne by occupants, the company and the fire service. **False negative:** a fire burns with no alarm —
response is delayed, the fire grows, and property damage, injury or death become more likely; it is borne by
whoever is in the building. *Similar:* the miss is the graver error per case, and the false alarms are far more
numerous. *Different:* a missed fire does its damage in minutes rather than months, and repeated false alarms
change behavior — people stop responding to alarms, or switch them off — so today's false positives can cause
tomorrow's missed fires.
""")

md(r"""
### Threshold A

**Q3.3 — Accuracy and error rate.**

⟦EQ_acc⟧

⟦EQ_err⟧

In this scenario, what do TP + TN and FP + FN represent? Substitute the threshold A values, calculate both
statistics, and interpret each in a complete sentence. What accuracy would a system that **never** raises an alarm
have?
""")
calc(r"""
*Values:* TP + TN = ⟦fA_TP⟧ + ⟦fA_TN⟧ = ⟦fA_right⟧ periods the system classified correctly (fire with alarm, or no fire and no
alarm); FP + FN = ⟦fA_FP⟧ + ⟦fA_FN⟧ = ⟦fA_wrong⟧ periods it got wrong (false alarm or undetected fire). *Calculation:* accuracy
$=\frac{45+855}{1000}=⟦fA_acc⟧$; error rate $=\frac{95+5}{1000}=⟦fA_err⟧$. *Interpretation:* At threshold A the system
is correct for ⟦fA_acc_pct⟧ of the observation periods and wrong for ⟦fA_err_pct⟧ of them. A system that never raises an alarm
is correct for the ⟦fA_neg⟧ periods without a fire: accuracy ⟦fA_base⟧, higher than ⟦fA_acc⟧, with every fire missed. And 95
of the 100 errors are false alarms, which the error rate does not reveal.
""")

md(r"""
**Q3.4 — Precision.**

⟦EQ_prec⟧

In this scenario, what do TP and FP represent, and what is TP + FP? Substitute the threshold A values, calculate
precision, and interpret your answer in a complete sentence.
""")
calc(r"""
*Values:* TP = ⟦fA_TP⟧ alarms raised when there was a fire; FP = ⟦fA_FP⟧ alarms raised when there was not; TP + FP = ⟦fA_ppos⟧ is
every alarm raised. *Calculation:* $\frac{45}{45+95}=\frac{45}{140}=⟦fA_prec⟧$. *Interpretation:* When the alarm
sounds at threshold A, there is a real fire about ⟦fA_prec_pct⟧ of the time — roughly one alarm in three; the other two
are false alarms.
""")

md(r"""
**Q3.5 — Recall.**

⟦EQ_rec⟧

In this scenario, what do TP and FN represent, and what is TP + FN? Substitute the threshold A values, calculate
recall, and interpret your answer in a complete sentence.
""")
calc(r"""
*Values:* TP = ⟦fA_TP⟧ fires that triggered an alarm; FN = ⟦fA_FN⟧ fires that did not; TP + FN = ⟦fA_pos⟧ is every period in which a
fire occurred. *Calculation:* $\frac{45}{45+5}=\frac{45}{50}=⟦fA_rec⟧$. *Interpretation:* At threshold A the system
raises an alarm for ⟦fA_rec_pct⟧ of the fires and stays silent for ⟦fA_fnr_pct⟧ of them (5 fires).
""")

md(r"""
**Q3.6 — F1 score.**

⟦EQ_f1⟧

Substitute your threshold A values, calculate F1, and interpret your answer in a complete sentence. Which of the
two statistics it combines is holding it down?
""")
calc(r"""
*Values:* precision ⟦fA_prec⟧ and recall ⟦fA_rec⟧; the ⟦fA_TN⟧ quiet, fire-free periods play no part. *Calculation:* from the
counts, $\frac{2(45)}{2(45)+95+5}=\frac{90}{190}=⟦fA_f1⟧$. *Interpretation:* Weighting precision and recall equally,
threshold A scores ⟦fA_f1⟧; recall is high, and the low precision (many false alarms) holds F1 down. *Rounding:*
students who use the rounded 0.321 and 0.900 in the first form get
$2\cdot\frac{0.321\times 0.900}{0.321+0.900}=⟦fA_f1r⟧$. Accept both, and use it to recommend the count form (or
carrying more decimals).
""")

md(r"""
### Threshold B

**Q3.7 — Precision and recall.**

⟦EQ_prec⟧

⟦EQ_rec⟧

Substitute the threshold B values, calculate precision and recall, and interpret each in a complete sentence.
""")
calc(r"""
*Values:* at threshold B, TP = ⟦fB_TP⟧, FP = ⟦fB_FP⟧, FN = ⟦fB_FN⟧; ⟦fB_ppos⟧ alarms are raised and ⟦fB_pos⟧ periods had a fire.
*Calculation:* precision $=\frac{35}{35+19}=\frac{35}{54}=⟦fB_prec⟧$; recall $=\frac{35}{35+15}=\frac{35}{50}=⟦fB_rec⟧$.
*Interpretation:* At threshold B about ⟦fB_prec_pct⟧ of alarms are real fires — roughly two in three — and the system
raises an alarm for ⟦fB_rec_pct⟧ of the fires, staying silent for 15 of the 50. (For reference: accuracy ⟦fB_acc⟧,
F1 ⟦fB_f1⟧.)
""")

md(r"""
### Comparing the thresholds

**Q3.8.** Going from threshold A to threshold B, what happened to the false alarms, the undetected fires,
precision and recall? Put the trade in one sentence. What would you need to know about what an alarm sets in
motion before saying which threshold you prefer?
""")
discuss(r"""
A → B: false alarms fall from 95 to 19 (⟦f_dFP⟧ fewer), undetected fires rise from 5 to 15 (⟦f_dFN⟧ more — three times as
many); precision rises from ⟦fA_prec⟧ to ⟦fB_prec⟧ and recall falls from ⟦fA_rec⟧ to ⟦fB_rec⟧. *B avoids 76 false alarms at
the price of 10 more undetected fires* — ⟦f_ratio⟧ false alarms avoided per additional miss. What matters is what an
alarm triggers. If it prompts an operator to look at a camera for thirty seconds, 95 false alarms are cheap and A
is easy to defend. If it evacuates a hospital or dispatches fire engines every time, 95 false alarms are very
costly and may be unsustainable. It also matters what stands behind a miss: sprinklers, ordinary smoke detectors
and people on site make an undetected fire less likely to become a disaster.
""")

md(r"""
**Q3.9.** A building manager says: "A missed fire is so much worse than a false alarm that we should simply use
the most sensitive setting available." Give two reasons the false alarms still deserve weight, and say how a
verification step after the alarm would change your view.
""")
discuss(r"""
(1) **Direct cost:** every false alarm interrupts work, may empty a building, and can tie up responders who are
then unavailable for a real emergency; in some places repeated false alarms also bring fees. (2) **Alarm
fatigue:** at threshold A two alarms in three are false, and people learn it — they respond slowly, ignore the
alarm, or disable the system, and the real fire is then effectively missed even though the system detected it.
The most sensitive setting of all raises an alarm every period, which detects every fire and tells nobody
anything. A **verification step** (an operator checking a camera, a second sensor that must agree) plays the role
that follow-up diagnostic testing played in Part 2: it filters the false alarms cheaply before the expensive
response, which makes a sensitive threshold affordable. Without such a step, each false alarm goes straight to
the costly response, and B looks better.
""")

md(r"""
### Going further: the other five statistics

**Q3.10 — Specificity and false positive rate (threshold B).**

⟦EQ_spec⟧

⟦EQ_fpr⟧

In this scenario, what do TN and FP represent, and what is TN + FP? Substitute the threshold B values, calculate
both, and interpret each in a complete sentence.
""")
calc(r"""
*Values:* TN = ⟦fB_TN⟧ fire-free periods with no alarm; FP = ⟦fB_FP⟧ fire-free periods with an alarm; TN + FP = ⟦fB_neg⟧ is every
period without a fire. *Calculation:* specificity $=\frac{931}{931+19}=\frac{931}{950}=⟦fB_spec⟧$; FPR
$=\frac{19}{19+931}=\frac{19}{950}=⟦fB_fpr⟧$. *Interpretation:* At threshold B the system stays quiet in ⟦fB_spec_pct⟧ of the
fire-free periods and raises a false alarm in ⟦fB_fpr_pct⟧ of them. *Worth noticing:* an FPR of only 0.020 still leaves
⟦fB_fdr_pct⟧ of alarms false (1 − precision), because fire-free periods outnumber fires 19 to 1. A small rate applied to
a large group is still a lot of alarms.
""")

md(r"""
**Q3.11 — False negative rate (thresholds A and B).**

⟦EQ_fnr⟧

In this scenario, what does FN represent, and what is FN + TP? Substitute, calculate the false negative rate at
each threshold, and interpret the change from A to B in a complete sentence.
""")
calc(r"""
*Values:* FN is the number of periods in which a fire occurred and no alarm was raised (5 at A, 15 at B); FN + TP
= ⟦fA_pos⟧ is every period with a fire. *Calculation:* A: $\frac{5}{5+45}=⟦fA_fnr⟧$; B: $\frac{15}{15+35}=⟦fB_fnr⟧$.
*Interpretation:* Moving from threshold A to threshold B triples the share of fires that go undetected, from
⟦fA_fnr_pct⟧ to ⟦fB_fnr_pct⟧ — nearly one fire in three.
""")

md(r"""
**Q3.12 — Negative predictive value (threshold B).**

⟦EQ_npv⟧

In this scenario, what do TN and FN represent, and what is TN + FN? Substitute the threshold B values, calculate
NPV, and interpret your answer in a complete sentence. Is a value this high reassuring? Compare it with the share
of all periods that had no fire.
""")
calc(r"""
*Values:* TN = ⟦fB_TN⟧ quiet periods with no fire; FN = ⟦fB_FN⟧ quiet periods in which there was a fire; TN + FN = ⟦fB_pneg⟧ is
every period in which the system stayed quiet. *Calculation:* $\frac{931}{931+15}=\frac{931}{946}=⟦fB_npv⟧$.
*Interpretation:* When the system stays quiet at threshold B, there really is no fire ⟦fB_npv_pct⟧ of the time; in
about ⟦fB_miss_neg_pct⟧ of quiet periods (15 of 946) a fire is burning undetected. *Reassuring?* Less than it sounds:
⟦fB_base_pct⟧ of all periods are fire-free, so silence was already "⟦fB_base⟧ reliable" before the system did anything.
(Threshold A: ⟦fA_npv⟧.) The same rate that looks excellent as a percentage is 15 undetected fires.
""")

md(r"""
**Q3.13 — Balanced accuracy (thresholds A and B).**

⟦EQ_bacc⟧

Calculate specificity for threshold A, then balanced accuracy for both thresholds, and interpret the results in a
complete sentence. Now line up three summaries for each threshold: accuracy (A from Q3.3; B is ⟦fB_acc⟧), F1 (A
from Q3.6; B is ⟦fB_f1⟧) and balanced accuracy. Which threshold does each summary favor? What do you conclude?
""")
calc(r"""
*Values:* recall is the success rate among the ⟦fA_pos⟧ periods with a fire; specificity is the success rate among the
⟦fA_neg⟧ periods without. *Calculation:* specificity at A $=\frac{855}{855+95}=\frac{855}{950}=⟦fA_spec⟧$. Balanced
accuracy: A $=\frac{0.900+0.900}{2}=⟦fA_bacc⟧$; B $=\frac{0.700+0.980}{2}=⟦fB_bacc⟧$. *Interpretation:* Giving fire
and fire-free periods equal weight, the system is right for an average of ⟦fA_bacc_pct⟧ of each kind at threshold A and
⟦fB_bacc_pct⟧ at threshold B.

| Summary | Threshold A | Threshold B | Favors |
|---|---:|---:|---|
| Accuracy | ⟦fA_acc⟧ | ⟦fB_acc⟧ | B |
| F1 | ⟦fA_f1⟧ | ⟦fB_f1⟧ | B |
| Balanced accuracy | ⟦fA_bacc⟧ | ⟦fB_bacc⟧ | A |

*Conclusion:* the summaries disagree, because each one builds in its own weighting of the two errors — accuracy
weights every period equally (so the 950 fire-free periods dominate), F1 weights precision and recall equally and
ignores true negatives, balanced accuracy weights the two classes equally. None of those weightings was chosen by
anyone who knows what a missed fire or a false alarm costs. (In Part 2 all three favored B — ⟦cA_bacc⟧ vs ⟦cB_bacc⟧ for
balanced accuracy — which does not make B right there either.)
""")

md(r"""
**Q3.14.** Looking back over Q3.10–Q3.13: which of these five statistics would you put in front of the building
manager, and which in front of the people who work in the building? Explain your choices.
""")
discuss(r"""
There is more than one good answer; look for a reason tied to a point of view. The **manager** pays for false
alarms and answers for missed fires, so precision or the count of false alarms (how many responses are wasted),
the false negative rate, and the trade between them are natural. **Occupants** experience two things: how often
they are disturbed for nothing (false positive rate, or better, false alarms per month) and whether silence can
be trusted (NPV, read against the ⟦fB_base_pct⟧ baseline, or the false negative rate stated as "3 fires in 10 would not
trigger the alarm"). Counts are often clearer than rates for either audience.
""")

# ============================================================================ #
#  Part 4 — final discussion
# ============================================================================ #
md(r"""
---
# Part 4 — Final discussion

### Where a threshold comes from

In neither scenario did we have two different models. A classifier such as **logistic regression** does not
output "positive" or "negative"; it outputs a **predicted probability** for each case (this person's probability
of cancer, this period's probability of fire). The class comes from a second step that we choose:

$$\text{predict positive if}\quad \hat p \ge t$$

where $t$ is the **classification threshold**. Threshold A in each scenario is a lower $t$ (more cases clear the
bar, so the model is "more sensitive"); threshold B is a higher $t$. The familiar $t = 0.5$ is a default, not a
recommendation. The fitted model and its probabilities stay the same; the threshold is a decision made on top of them.
""")

note(r"""
This connects to Topic 11.1, section 7, where the same idea was shown with a KNN classifier's vote shares and a
table of thresholds. If there is time, sketch a row of predicted probabilities on the board and slide a cutoff
along it. Q4.1–Q4.5 work well as small-group questions with one group reporting on each.
""")

md(r"""
**Q4.1.** Compare the two scenarios. For each, state the consequence of a false positive and of a false negative,
who bears it, and whether anything afterwards can correct the mistake.
""")
discuss(r"""
| | False positive | False negative |
|---|---|---|
| **Cancer screening** | A person without cancer is referred for follow-up testing: worry, cost, some procedural risk. Borne by the patient and the health system. *Corrected* by the follow-up test. | A person with cancer is reassured and sent home; diagnosis is delayed and treatment may be harder. Borne by the patient. *Not corrected* until symptoms or the next screening. |
| **Fire detection** | An alarm with no fire: disruption, evacuation, a wasted call-out, and growing distrust of the alarm. Borne by occupants, the company, responders. *Corrected* quickly if someone verifies. | A fire with no alarm: delayed response, damage, possible injury or death. Borne by occupants and owners. *Corrected* only if something else (a person, a smoke detector, sprinklers) notices. |

In both, the false negative is the graver error per case and is the one nothing follows up; the false positive is
milder per case, far more frequent, and is usually caught by the next step. The time scales differ (months vs
minutes), and so does the effect of false alarms on behavior.
""")

md(r"""
**Q4.2.** In both scenarios most people judge the false negative to be the more costly error. Explain why the
other type of mistake still matters. Use numbers from the tables to support your answer.
""")
discuss(r"""
Because cost per error is only half of the sum; the number of errors is the other half. At threshold A there are
180 false positives against 10 false negatives in screening, and 95 against 5 in fire detection. Caring about
false negatives alone leads to flagging everything — recall of 1, with precision equal to the base rate (⟦cA_prev⟧
and ⟦fA_prev⟧) — which is the same as having no model. False positives also spend the very resources that true
positives need (clinic appointments, fire crews) and wear down the trust that makes people act on a positive
(skipped screenings, ignored alarms). So false positives eventually produce harm of the first kind too.
""")

md(r"""
**Q4.3.** Using the idea of predicted probabilities, explain what happens to the number of predicted positives,
and then to recall and to precision, when the threshold is lowered. Check your explanation against the two tables.
What happens at the extremes — a threshold of 0, and a threshold close to 1?
""")
discuss(r"""
Lowering $t$ lets more cases clear the bar, so there are more predicted positives. Every actual positive that was
flagged before is still flagged, and some that were missed now get caught: TP can only rise and FN can only fall,
so **recall rises or stays the same**. But the newly flagged cases are the ones with lower predicted
probabilities, so more of them are negatives: FP rises, and **precision usually falls** (usually, not always — it
can tick up at a particular step). The tables agree: from B to A, recall goes ⟦cB_rec⟧ → ⟦cA_rec⟧ and precision
⟦cB_prec⟧ → ⟦cA_prec⟧ (screening); recall ⟦fB_rec⟧ → ⟦fA_rec⟧ and precision ⟦fB_prec⟧ → ⟦fA_prec⟧ (fire). At $t = 0$ everything is
flagged: recall 1, precision equal to the share of positives. With $t$ close to 1 almost nothing is flagged:
the few flags are usually right, and recall is near 0. Every threshold in between is a different trade, and the two
rows in each table are just two points on that curve.
""")

md(r"""
**Q4.4.** "Threshold B has the higher F1 score in both scenarios, so B is the better threshold." Respond to this
claim. In your answer, explain why the preferred threshold depends on the situation and on what happens after a
positive prediction.
""")
discuss(r"""
The arithmetic is right (F1: ⟦cA_f1⟧ vs ⟦cB_f1⟧ in screening, ⟦fA_f1⟧ vs ⟦fB_f1⟧ in fire) and the conclusion does not follow.
F1 answers "which threshold is better **if** precision and recall matter equally" and it ignores true negatives.
Nobody decided that a missed cancer and a needless referral matter equally. Choosing B means accepting 10 more
missed cancers to avoid 135 referrals, or 10 more undetected fires to avoid 76 false alarms; whether that is a good
trade depends on what each error costs, and that depends on what a positive prediction sets in motion. Where a
positive leads to a cheap, safe, accurate check (a diagnostic test, an operator looking at a camera), false
positives are inexpensive and a sensitive threshold can be justified. Where a positive leads directly to
something costly or hard to undo (an invasive procedure, evacuating a hospital), precision deserves more weight.
The same model can sensibly be run at different thresholds in different settings. A is not "the answer" either:
the threshold is a decision about consequences, and no statistic on this page makes it for us.
""")

md(r"""
**Q4.5.** You are asked to recommend a threshold for one of the two systems. List the additional information you
would want before making a recommendation, and say for each item how it could change your answer.
""")
discuss(r"""
Look for several of these, each with a "because":
- **The cost or harm of each error, at least relative to one another.** With costs one can compare thresholds
  directly: B is preferable in screening only if one missed cancer costs less than ⟦c_ratio⟧ needless referrals, and in
  fire detection only if one undetected fire costs less than ⟦f_ratio⟧ false alarms.
- **What happens after a positive prediction** — the follow-up test or verification step, its accuracy, cost, risk
  and delay. A good second step makes false positives cheap.
- **What happens after a negative prediction** — is there another chance to catch a miss (the next screening
  round, other detectors, sprinklers, people on site)?
- **Capacity.** How many positives can the clinic or the responders handle? 270 referrals may simply not be feasible.
- **How common the positive class is where the system will be used.** These tables assume 10% and 5%. If positives
  are rarer in practice, precision at every threshold will be lower than here.
- **How much timing matters.** Does catching the cancer or the fire earlier change the outcome a lot or a little?
- **How reliable these estimates are.** Recall in the fire table rests on only 50 fires; were the 1,000 cases
  representative of where the system will run?
- **More than two thresholds.** A and B are two points; the best trade may lie between or beyond them.
- **Who decides, and what the people affected prefer.** The trade involves values as well as numbers.
""")

md(r"""
---
## What to take away

- Every statistic here is a fraction; ask **"out of which group?"** before interpreting it.
- **Precision** is out of the predicted positives; **recall** is out of the actual positives. They answer
  different questions and usually pull in opposite directions as the threshold moves.
- **Accuracy** flatters a model when one class is rare; compare it with predicting the common class for everyone.
- **F1** and **balanced accuracy** are summaries with built-in weightings. A higher value is not automatically a
  better decision.
- A threshold turns a predicted probability into an action. Choosing it means weighing the two kinds of error
  and knowing what happens after a positive prediction.
""")

# ============================================================================ #
#  Build
# ============================================================================ #
METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

STUDENT_TITLE = (
    "# GSB 5544 — Topic 11.2: Classification Metrics in Context — Cancer Screening and Fire Detection\n"
    "*A guided class activity. We define ten statistics from the confusion matrix, then use them to compare two "
    "thresholds in each of two scenarios. Calculator only — no code.*"
)


def notebook(cell_list: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cell_list):
        c = {k: copy.deepcopy(v) for k, v in c.items() if k != "_kind"}
        c["id"] = f"cell-{i:03d}"
        out.append(c)
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def student_cells(cell_list: list[dict]) -> list[dict]:
    out = []
    for c in cell_list:
        c = copy.deepcopy(c)
        if c["_kind"] == "note":
            continue
        if c["_kind"] == "calc":
            c["source"] = CALC_SPACE
        elif c["_kind"] == "discuss":
            c["source"] = DISCUSS_SPACE
        out.append(c)
    out[0]["source"] = STUDENT_TITLE
    return out


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    leftovers = [c["source"][:60] for c in cells if "⟦" in c["source"]]
    assert not leftovers, leftovers
    WEEK.mkdir(parents=True, exist_ok=True)
    student = STEM.with_name(STEM.name + "-empty.ipynb")
    solution = STEM.with_name(STEM.name + "-solution.ipynb")
    s_cells = student_cells(cells)
    save(student, notebook(s_cells))
    save(solution, notebook(cells))
    n_q = sum(c["_kind"] in ("calc", "discuss") for c in cells)
    print(f"wrote {student.relative_to(ROOT)} ({len(s_cells)} cells) and {solution.relative_to(ROOT)} "
          f"({len(cells)} cells); {n_q} questions, no code cells")


if __name__ == "__main__":
    main()
