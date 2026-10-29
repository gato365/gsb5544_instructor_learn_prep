#!/usr/bin/env python3
"""Generate the Week 10 Topic 10.2 notebook (student + executed solution) and export its questions for the page.

Usage:  python3 tools/build_week10_reg_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week10_reg_topics.py --no-exec  # write both notebooks without executing
        python3 tools/build_week10_reg_topics.py --questions-js  # print the QUESTIONS array for "Shrink to Fit.html"

Topic 10.2 — Regularization: Ridge, LASSO, Tuning Parameters and Cross-Validation
  A short conceptual opener (15-20 minutes) placed before the penalized-regression code walkthrough and PA 10.2.
  Thirteen scaffolded think-and-discuss questions on one running example (forecasting a coffee shop's weekly
  sales from ten store characteristics, the example of the interactive page "Shrink to Fit"), each with a 2-4
  sentence answer / instructor note, the least-squares / ridge / LASSO objectives with their terms labelled, a
  table of candidate penalty strengths with hypothetical CV MSEs, the misconceptions to name out loud, one tiny
  simulated code demo, and a transition into the Ames walkthrough and the loans PA.
  The same 13 questions are exported (--questions-js) and pasted into the page's Think-Discuss-Reveal widget.
  Needs scikit-learn, pandas and numpy; nothing is read from the web.
Same markers as the earlier builders:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
"""

from __future__ import annotations

import copy
import html as htmlmod
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_10"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"
PAGE = SITE + "week_10/shrink_to_fit.html"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  The thirteen questions (shared by the notebook and the page)
# ============================================================================ #
# Each entry: (question, answer-in-markdown, optional follow-up or None). Markdown is kept simple (bold,
# italics, inline code, $...$ math) so it converts cleanly to the page's HTML.

QUESTIONS: list[tuple[str, str, str | None]] = [
    (
        "Plain linear regression. We fit a line (well, a plane) to the 60 training stores: weekly sales ≈ intercept + "
        "coefficients × the ten store characteristics. What exactly does least squares minimize, and why might a "
        "model that fits those 60 stores very well still forecast a *new* store poorly?",
        "Least squares picks the intercept and coefficients that make the **sum of squared training errors** as small as "
        "possible. It is judged only on the 60 stores it saw, so it will happily bend to whatever is peculiar about "
        "them — a few unusual stores, a lucky correlation — and those peculiarities do not repeat in the next store. Low "
        "training error is a fact about the past sample; forecasting is a claim about new stores.",
        "Which number did we trust last week instead of training error? (Cross-validated test error.)",
    ),
    (
        "Overfitting through the coefficients. Three of our ten predictors — seats, parking spaces and store size — "
        "all describe how big the store is, and two others may be pure noise. How can coefficients become overly "
        "sensitive to the particular 60 stores we happened to sample?",
        "When predictors overlap, many different combinations of coefficients give almost the same fitted values: a "
        "large positive coefficient on seats and a large negative one on store size can cancel. Least squares picks "
        "whichever combination fits *this* sample best, so a different 60 stores can flip the signs and sizes "
        "completely. Noise predictors get non-zero coefficients for the same reason — there is always some chance "
        "pattern to fit. Many predictors relative to rows makes all of this worse.",
        None,
    ),
    (
        "Why accept a worse fit? Suppose we deliberately pull the coefficients back toward zero, so training error "
        "goes *up* a little. Why could that make the forecasts for new stores *better*?",
        "Part of what the unconstrained coefficients capture is signal and part is sample noise. Pulling them toward "
        "zero gives up a little of the signal (a bit more training error) but throws away a lot of the noise, so the "
        "model's predictions vary less from sample to sample. For new stores the trade is usually favourable — up to a "
        "point. Last week's picture applies: training error can only get worse as we restrain the model, while "
        "validation error falls first and rises later.",
        "Does it *guarantee* better forecasts? No — it is a bet that the noise outweighs the lost signal, and "
        "cross-validation is how we check the bet.",
    ),
    (
        "The penalty. Regularization turns that idea into a single thing to minimize: training error **plus** a "
        "penalty on coefficient size. In words, what are the two terms fighting over, and what decides who wins?",
        "The first term rewards fitting the 60 stores closely; the second charges a price for large coefficients. "
        "Minimizing their sum finds coefficients that fit well *for their size*. The exchange rate between the two is "
        "the tuning parameter $\\lambda$ (scikit-learn calls it `alpha`): a larger $\\lambda$ makes coefficient size "
        "more expensive, so the solution shrinks. The intercept is left out of the penalty — it just sets the overall "
        "level of sales and shrinking it would bias every forecast.",
        None,
    ),
    (
        "Ridge regression penalizes the **sum of squared coefficients**, $\\lambda \\sum_j \\beta_j^2$. Why does that "
        "pull every coefficient toward zero, and why does it pull the biggest ones hardest?",
        "Squaring makes a coefficient of 4 cost sixteen times a coefficient of 1, so the cheapest way to lower the "
        "penalty is to trim the largest coefficients first. Every coefficient is shrunk somewhat, and the trimming of a "
        "coefficient that is already small buys almost nothing, so ridge rarely drives anything exactly to zero — it "
        "approaches zero smoothly. With overlapping predictors like seats, parking and size, ridge tends to spread the "
        "shared effect across the group instead of letting one coefficient balloon and another go negative.",
        None,
    ),
    (
        "LASSO penalizes the **sum of absolute coefficients**, $\\lambda \\sum_j |\\beta_j|$. How is the effect "
        "different, and why can LASSO set a coefficient to *exactly* zero when ridge does not?",
        "With absolute values, moving a coefficient from 0.1 to 0 saves the same amount of penalty as moving it from "
        "4.1 to 4 — the last step to zero is never cheaper than the others. If a predictor is not earning its keep in "
        "the fit, the penalty wins and the coefficient lands on zero and stays there. (Under ridge that last step from "
        "0.1 to 0 saves only 0.01, so it is almost never worth taking.) Predictors with a zero coefficient are out of "
        "the model, which is why LASSO is described as doing variable selection.",
        None,
    ),
    (
        "Compare them. What do ridge and LASSO share, and what is the practical difference between *shrinking* a "
        "coefficient and *zeroing* it?",
        "Both minimize training error plus a $\\lambda$-weighted penalty on coefficient size, both need standardized "
        "predictors, both have $\\lambda$ chosen by cross-validation, and neither penalizes the intercept. A shrunk "
        "coefficient keeps the predictor in the forecast with a muted effect; a zeroed one removes it, which gives a "
        "smaller model that is easier to explain and cheaper to feed with data. Among overlapping predictors LASSO "
        "often keeps one and drops the rest somewhat arbitrarily; ridge keeps them all with smaller coefficients. "
        "Which forecasts better depends on the data and is settled by cross-validation, not by preference.",
        "A LASSO zero on `store size` while `seats` is kept does **not** mean store size is irrelevant or has no "
        "effect — LASSO chose a stand-in. Regularization is a forecasting tool, not a hypothesis test. (A mixture of "
        "the two penalties, elastic net, exists for exactly this correlated-group situation; the walkthrough shows it.)",
    ),
    (
        "Units. Store size is measured in square feet (hundreds to thousands); manager tenure in years (0 to 20). "
        "Why does that matter once we start penalizing coefficient size, and what do we do about it?",
        "A coefficient's size depends on its predictor's units: sales per square foot is a tiny number, sales per year "
        "of tenure a large one, and neither says which predictor matters more. A penalty that treats every "
        "coefficient alike would therefore punish tenure heavily and square footage hardly at all. **Standardizing** "
        "each predictor (subtract its mean, divide by its standard deviation) puts every coefficient in the same "
        "units — dollars of sales per one standard deviation — so the penalty is fair and the coefficients are "
        "comparable. Large coefficients are not automatically a problem; they are only meaningful after this step.",
        "Inside cross-validation the scaler must be fitted on each training fold alone (a pipeline does this); "
        "scaling all the rows first leaks the validation rows into the fit.",
    ),
    (
        "The tuning parameter. What does the model become when $\\lambda = 0$, when $\\lambda$ is small, and when "
        "$\\lambda$ is enormous? And how is $\\lambda$ different in kind from the coefficients?",
        "At $\\lambda = 0$ the penalty vanishes and we are back to ordinary least squares. For small $\\lambda$ the "
        "coefficients are gently shrunk and predictions barely change. As $\\lambda$ grows the coefficients head to "
        "zero (LASSO drops predictors one by one) and the forecast collapses toward the average sales of the training "
        "stores — a model that ignores the store. The coefficients are **learned** by fitting, given a $\\lambda$; "
        "$\\lambda$ itself is not learned from the fit — it is a **tuning parameter**, set from outside and chosen by "
        "comparing candidate values on validation data.",
        "Software scales the error term differently (scikit-learn's `Lasso` divides it by $2n$, `Ridge` does not), "
        "so `alpha = 1` is not the same strength of penalty for ridge and LASSO, nor the same across data sets. The "
        "value is meaningless on its own; the cross-validation curve is what matters.",
    ),
    (
        "Choosing $\\lambda$ by cross-validation. Here are six candidate penalty strengths and the 5-fold "
        "cross-validation MSE each produced on the 60 training stores (hypothetical numbers, in millions of "
        "dollars²):\n\n"
        "| $\\lambda$ | 0 | 0.1 | 1 | 10 | 100 | 1000 |\n|---|---|---|---|---|---|---|\n"
        "| CV MSE | 9.8 | 9.1 | 8.4 | 8.2 | 9.6 | 15.0 |\n\n"
        "Which would you pick, and why? Why could training MSE not have made this choice?",
        "Pick $\\lambda = 10$: it has the lowest cross-validated error, and the curve has the expected shape — "
        "unpenalized at 9.8, improving as the penalty bites, then collapsing (15.0) when the penalty crushes the useful "
        "coefficients too. The strongest penalty is clearly not the best. Training MSE would have chosen $\\lambda = "
        "0$ every time, because any penalty can only make the fit to the training stores worse. One caution: 8.4 and "
        "8.2 are close, and five folds give a noisy average, so $\\lambda = 1$ would also be defensible; some analysts "
        "take the most penalized value whose error is within the noise of the minimum.",
        None,
    ),
    (
        "Grid search. We just compared six values of $\\lambda$ by hand. Describe what `GridSearchCV` does with "
        "that list, in one or two sentences.",
        "It takes the list of candidate values (the grid), and for each one runs the *same* K-fold cross-validation — "
        "same folds for every candidate — records the mean validation score, picks the candidate with the best mean, "
        "and refits that model on all the training rows. It is the table above, automated, with the bookkeeping "
        "(`cv_results_`, `best_params_`, `best_estimator_`) kept for you. Two tuning parameters (as in elastic net) "
        "just make the grid two-dimensional.",
        None,
    ),
    (
        "Validation versus test. The chain locked 40 stores away before we started. If cross-validation already tells "
        "us how well each $\\lambda$ forecasts unseen stores, why keep those 40 untouched until the very end?",
        "Because we *used* the validation scores to choose. The best of six candidates is partly the luckiest of six, "
        "so its cross-validated error is a little optimistic — and if we also chose features, methods and $\\lambda$ "
        "on it, more so. The 40 held-out stores took no part in any decision, so the one error we compute on them "
        "after refitting the chosen model is an honest estimate of performance on genuinely new stores. Open the test "
        "set once, report the number, and do not go back and re-tune.",
        None,
    ),
    (
        "Put it together. In four or five sentences, explain to a manager how regularization, ridge/LASSO, the "
        "tuning parameter and cross-validation work together to produce a sales forecast for a store the chain has "
        "never seen.",
        "We fit a linear model to the training stores but charge a price for large coefficients, which stops the model "
        "from memorizing quirks of our sample (regularization). Ridge charges by squared size and shrinks everything a "
        "little; LASSO charges by absolute size and can drop predictors entirely, giving a simpler model. How heavy the "
        "charge should be — the tuning parameter — is not something the fit can tell us, so we try a grid of values and "
        "let cross-validation on the training stores vote for the one that forecasts held-out stores best. We refit "
        "with that value on all the training stores, check it once on the stores we locked away, and that model, with "
        "standardized inputs, is what forecasts the new store.",
        None,
    ),
]

TABLE_Q_INDEX = 9  # the question that contains the markdown table (needs HTML conversion for the page)


def to_html(text: str) -> str:
    """Minimal markdown -> HTML for the page's Reveal panel (bold, italics, inline code, paragraphs, one table)."""
    def inline(s: str) -> str:
        s = htmlmod.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        s = s.replace("\\lambda", "λ").replace("\\sum_j", "Σ").replace("\\beta_j^2", "β²").replace("\\beta_j", "β")
        s = re.sub(r"\$(.+?)\$", r"<i>\1</i>", s)
        return s
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for b in blocks:
        if b.lstrip().startswith("|"):
            rows = [r.strip().strip("|").split("|") for r in b.strip().splitlines() if not re.match(r"^\s*\|[-| ]+\|\s*$", r)]
            head, body = rows[0], rows[1:]
            tr = "".join(f"<th>{inline(c.strip())}</th>" for c in head)
            trs = "".join("<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<table class="mini"><thead><tr>{tr}</tr></thead><tbody>{trs}</tbody></table>')
        else:
            out.append(f"<p>{inline(' '.join(b.split()))}</p>")
    return "".join(out)


def to_text(text: str) -> str:
    """Plain text for the page's question line (it is set with textContent): markdown stripped, math made readable,
    a markdown table rendered as one sentence."""
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for b in blocks:
        if b.lstrip().startswith("|"):
            rows = [r.strip().strip("|").split("|") for r in b.strip().splitlines() if not re.match(r"^\s*\|[-| ]+\|\s*$", r)]
            head = [c.strip() for c in rows[0]]
            out.append("; ".join(f"{head[0]} = {h}: {r[0].strip()} {v.strip()}" for h, v in zip(head[1:], rows[1][1:]) for r in [rows[1]]) + ".")
        else:
            out.append(" ".join(b.split()))
    s = " ".join(out)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)
    s = s.replace("`", "")
    s = s.replace("\\lambda", "λ").replace("\\sum_j", "Σ").replace("\\beta_j^2", "β²").replace("\\beta_j", "β")
    s = re.sub(r"\$(.+?)\$", r"\1", s)
    return s


def questions_js() -> str:
    """The QUESTIONS array for the page's Think-Discuss-Reveal widget (q = plain text, a = HTML)."""
    items = []
    for i, (q, a, follow) in enumerate(QUESTIONS):
        body = to_html(a) + (f'<p class="note"><b>Follow-up.</b> {to_html(follow)[3:-4]}</p>' if follow else "")
        if i == TABLE_Q_INDEX:   # show the candidate table in the reveal panel as well
            table = [b for b in re.split(r"\n\s*\n", q) if b.lstrip().startswith("|")][0]
            body = to_html(table) + body
        items.append("  { q: " + json.dumps(to_text(q)) + ",\n    a: " + json.dumps(body) + " }")
    return "var QUESTIONS = [\n" + ",\n".join(items) + "\n];"


# ============================================================================ #
#  Topic 10.2 notebook
# ============================================================================ #

t: list[dict] = []

md(t, """
# GSB 5544 — Topic 10.2: Regularization — Ridge, LASSO, Tuning Parameters and Cross-Validation — SOLUTION
*A 15–20 minute think-and-discuss opener before the penalized-regression walkthrough and PA 10.2. One running
example, thirteen short questions, no derivations.*
""")

md(t, f"""
## How to use this

**The running example.** A regional coffee chain has 100 locations. For each store we know **ten things** about
the store and its neighborhood — foot traffic, ad spend, store size, seats, parking spaces, neighborhood income,
competitor count, social-media posts, average temperature, manager tenure — and **last week's sales**. The chain
is opening new stores and wants a **sales forecast from a store's characteristics alone**. 60 stores are the
training data; 40 are locked away as a test set. (This is the example on the interactive page
[*Shrink to Fit*]({PAGE}), where the sliders show every number in this notebook moving; the questions below are
also in the page's *Think · Discuss · Reveal* box.)

**The rhythm.** For each question: think for a minute, discuss with a neighbour for two, then the instructor
reveals the answer. Thirteen questions, roughly 90 seconds each.

**The three objectives**, so they are in front of us the whole time. Each adds a term to the one before it.

$$\\text{{Least squares:}}\\quad \\min_{{\\beta_0,\\boldsymbol\\beta}} \\underbrace{{\\sum_{{i=1}}^{{n}} (y_i - \\hat y_i)^2}}_{{\\text{{prediction error on the training rows}}}}$$

$$\\text{{Ridge:}}\\quad \\min_{{\\beta_0,\\boldsymbol\\beta}} \\Big[\\underbrace{{\\sum_{{i=1}}^{{n}} (y_i - \\hat y_i)^2}}_{{\\text{{prediction error}}}} + \\underbrace{{\\lambda \\sum_{{j=1}}^{{p}} \\beta_j^2}}_{{\\text{{penalty: squared coefficients}}}}\\Big]$$

$$\\text{{LASSO:}}\\quad \\min_{{\\beta_0,\\boldsymbol\\beta}} \\Big[\\underbrace{{\\sum_{{i=1}}^{{n}} (y_i - \\hat y_i)^2}}_{{\\text{{prediction error}}}} + \\underbrace{{\\lambda \\sum_{{j=1}}^{{p}} |\\beta_j|}}_{{\\text{{penalty: absolute coefficients}}}}\\Big]$$

| symbol | what it is |
|---|---|
| $y_i$, $\\hat y_i$ | actual and predicted weekly sales for training store $i$; $n = 60$ |
| $\\beta_1, \\dots, \\beta_p$ | the **coefficients**, one per predictor ($p = 10$); learned by fitting |
| $\\beta_0$ | the intercept — **not penalized**; it only sets the overall level of sales |
| $\\lambda \\ge 0$ | the **tuning parameter**: how heavily coefficient size is charged. Not learned by fitting; chosen by cross-validation |

> **About `alpha`.** scikit-learn calls $\\lambda$ `alpha`, and it scales the error term differently for different
> models (`Lasso` and `ElasticNet` divide it by $2n$; `Ridge` does not). So `alpha=1` is **not** the same strength of
> penalty for ridge and for LASSO, and a good `alpha` on one data set says nothing about another. Compare penalty
> strengths by their cross-validation curves, never by the number itself.
""")

for i, (q, a, follow) in enumerate(QUESTIONS, start=1):
    md(t, f"---\n**Q{i}.** {q}")
    answer(t, a + (f"\n\n*Follow-up:* {follow}" if follow else ""))

md(t, """
---
## Misconceptions to name out loud

| students often think… | but… |
|---|---|
| "A big coefficient means that predictor is important / is a problem." | Coefficient size depends on the predictor's units until the predictors are standardized. Size is only comparable after that, and even then size is about the fit, not about importance in any causal sense. |
| "Regularization makes the model better." | It trades a little fit for less variance. Usually that helps on new data; sometimes (clean data, few predictors, many rows) it does nothing or hurts. Cross-validation tells you which. |
| "More penalty is safer." | Too much penalty flattens the model toward "predict the average sales": the CV curve rises again. The best penalty is the one with the lowest validation error, often moderate. |
| "Ridge and LASSO both do variable selection." | Ridge shrinks toward zero but essentially never reaches it; LASSO can land exactly on zero. Only LASSO (and elastic net) removes predictors. |
| "LASSO set `store size` to zero, so store size doesn't matter." | It means store size was not needed *for prediction given the other predictors* — very likely because `seats` carries the same information. It is not evidence of no effect and certainly not of no causal effect. |
| "Standardize once, then cross-validate." | The scaler is fitted; fit it on all rows and every validation fold has leaked its mean and spread into training. Put the scaler in the pipeline so it is refitted inside each training fold. |
""")

md(t, """
---
## One tiny demonstration (optional, two minutes)

A simulated coffee chain in eight lines: five real predictors, one correlated copy (`seats` tracks `size`), one
pure-noise column. Standardize, then fit least squares, ridge and LASSO at one penalty and compare the
coefficients. Watch which method shrinks and which zeros. (The `alpha` values differ between the two models on
purpose — see the note above.)
""")

code(t, """
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(5544)
n = 60
stores = pd.DataFrame({
    "traffic": rng.normal(800, 200, n),  "ads": rng.normal(300, 100, n),
    "size":    rng.normal(1800, 400, n), "income": rng.normal(65, 12, n),
    "competitors": rng.integers(0, 6, n),
})
stores["seats"]  = stores["size"] / 30 + rng.normal(0, 4, n)      # a correlated copy of size
stores["tenure"] = rng.normal(5, 3, n)                               # pure noise
sales = (6 * stores["traffic"] + 5 * stores["ads"] + 2 * stores["size"] + 60 * stores["income"]
         - 400 * stores["competitors"] + rng.normal(0, 2800, n))

fits = {
    "least squares": make_pipeline(StandardScaler(), LinearRegression()),
    "ridge":         make_pipeline(StandardScaler(), Ridge(alpha=«10»)),
    "LASSO":         make_pipeline(StandardScaler(), Lasso(alpha=«300»)),
}
coefs = pd.DataFrame({name: pipe.fit(stores, sales)[-1].coef_ for name, pipe in fits.items()}, index=stores.columns)
coefs.round(0) + 0.0        # + 0.0 turns LASSO's "-0.0" into a plain 0
""")

md(t, """
✅ **What to notice.** All three agree on the strong predictors (traffic, ads, income, competitors). Least squares
splits the "how big is the store" effect between `size` and `seats` in some arbitrary way and gives `tenure` a
non-zero coefficient it does not deserve; ridge shrinks every coefficient and shares the size effect across the
pair; LASSO shrinks too and sets some coefficients exactly to zero. Which of `size` and `seats` LASSO keeps is a
stand-in choice, not a finding. The penalty values here were set by hand; in the walkthrough and the PA they are
chosen by `GridSearchCV`.
""")

md(t, f"""
---
## Where this goes next

1. **The sliders.** [*Shrink to Fit*]({PAGE}): the same coffee chain; drag the penalty and watch the coefficients,
   the training and CV error curves, and the coefficient paths; run the grid search fold by fold; open the test
   set once at the end.
2. **The code walkthrough** (*Penalized Regression: Ridge and LASSO*, the first part of the PA 10.2 notebook):
   the Ames houses with *all* 67 predictors, where plain least squares with polynomial terms falls apart under
   cross-validation; then `Ridge`, `Lasso` and `ElasticNet` in a pipeline, counting zero coefficients, and
   `GridSearchCV` over `alpha` (and `l1_ratio`). Its twelve **DISCUSS** prompts are the questions above in Ames
   clothing — Q4–Q9 of this notebook answer them.
3. **PA 10.2.** The loans data again, predicting `interest_rate` — this time with ridge, LASSO and elastic net on
   many predictors, at least one parameter tuned with `GridSearchCV`. Bring the ideas: standardize inside the
   pipeline, choose `alpha` by cross-validation, read a zero coefficient as "not needed given the others", and
   keep the test estimate separate from the tuning.

Course site: [{SITE}]({SITE}).
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
            c["source"] = "**Your answer:** *(think, then discuss with a neighbour — write a sentence or two here)*"
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
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"unexpected errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


STEM = WEEK / "GSB5544_Topic_10_2_Regularization_Ridge_LASSO"
STUDENT_TITLE = (
    "# GSB 5544 — Topic 10.2: Regularization — Ridge, LASSO, Tuning Parameters and Cross-Validation  \n"
    "*A 15–20 minute think-and-discuss opener before the penalized-regression walkthrough and PA 10.2. One running "
    "example, thirteen short questions, no derivations. Think about each question for a minute, discuss it with a "
    "neighbour, and jot a sentence; the answers are revealed in class.*"
)


def main() -> None:
    if "--questions-js" in sys.argv:
        print(questions_js())
        return
    WEEK.mkdir(parents=True, exist_ok=True)
    student = STEM.with_name(STEM.name + "-empty.ipynb")
    solution = STEM.with_name(STEM.name + "-solution.ipynb")
    save(student, notebook(student_cells(t, STUDENT_TITLE)))
    save(solution, notebook(solution_cells(t)))
    print(f"wrote {student.relative_to(ROOT)} and {solution.relative_to(ROOT)} ({len(t)} cells)")
    if "--no-exec" not in sys.argv:
        execute(solution)


if __name__ == "__main__":
    main()
