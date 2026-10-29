#!/usr/bin/env python3
"""Generate the Week 10 Topic 10.2 notebook (student + executed solution) and export its questions for the page.

Usage:  python3 tools/build_week10_reg_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week10_reg_topics.py --no-exec  # write both notebooks without executing
        python3 tools/build_week10_reg_topics.py --questions-js  # print the QUESTIONS array for "Shrink to Fit.html"

Topic 10.2 — Regularization: Ridge, LASSO, Tuning Parameters and Cross-Validation
  A short conceptual opener (15-20 minutes) placed before the penalized-regression code walkthrough and PA 10.2.
  Five scaffolded think-and-discuss questions on one running example (forecasting a coffee shop's weekly
  sales from ten store characteristics, the example of the interactive page "Shrink to Fit"), each with a 2-4
  sentence answer / instructor note, the least-squares / ridge / LASSO objectives with their terms labelled, a
  table of candidate penalty strengths with hypothetical CV MSEs, the misconceptions to name out loud, one tiny
  simulated code demo, and a transition into the Ames walkthrough and the loans PA.
  The same 5 questions are exported (--questions-js) and pasted into the page's Think-Discuss-Reveal widget.
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
        "Plain linear regression first. We fit weekly sales ≈ intercept + coefficients × the ten store characteristics "
        "to the 60 training stores. What does least squares minimize — and why can a model that fits those 60 stores "
        "very well still forecast a *new* store poorly? (Three of the ten predictors — seats, parking, store size — all "
        "measure how big the store is, and two may be pure noise.)",
        "Least squares picks the intercept and coefficients that make the **sum of squared training errors** as small as "
        "possible, judged only on the 60 stores it saw. So it bends to whatever is peculiar about them: when predictors "
        "overlap, many coefficient combinations fit almost equally well (a big positive coefficient on seats cancelling "
        "a big negative one on store size), and least squares picks whichever suits *this* sample — a different 60 stores "
        "can flip the signs. Noise predictors get coefficients for the same reason. Low training error is a fact about the "
        "past sample; forecasting is a claim about new stores, and last week's cross-validated test error is the number "
        "that speaks to that.",
        None,
    ),
    (
        "The penalty. Regularization changes what we minimize to **training error + a penalty on coefficient size**. "
        "Why would deliberately accepting a slightly *worse* fit to the 60 stores make forecasts for new stores *better*, "
        "and what decides how the two terms trade off?",
        "Part of what unrestrained coefficients capture is signal and part is sample noise; pulling them toward zero gives "
        "up a little signal (more training error) but throws away a lot of noise, so predictions vary less from sample "
        "to sample — usually a favourable trade, though never a guaranteed one. The exchange rate is the tuning parameter "
        "$\\lambda$ (`alpha` in scikit-learn): a larger $\\lambda$ makes coefficient size more expensive, so the solution "
        "shrinks. The intercept is left out of the penalty — it only sets the overall level of sales, and shrinking it would "
        "bias every forecast.",
        "Training error can only get worse as $\\lambda$ grows; validation error falls first and rises later. That shape "
        "is the whole argument.",
    ),
    (
        "Two penalties. Ridge charges $\\lambda \\sum_j \\beta_j^2$ (squares); LASSO charges $\\lambda \\sum_j |\\beta_j|$ "
        "(absolute values). Why does ridge shrink every coefficient but almost never reach zero, while LASSO can set some "
        "coefficients to *exactly* zero? And why must the predictors be standardized before either penalty is applied?",
        "Squaring makes big coefficients very expensive and small ones nearly free, so ridge trims the largest hardest and "
        "the last step from 0.1 to 0 saves only 0.01 — almost never worth taking. Under absolute values that last step "
        "saves as much as any other, so a predictor that is not earning its keep lands on exactly zero and leaves the model: "
        "LASSO shrinks *and* selects, which gives a smaller, cheaper, more explainable model, though among overlapping "
        "predictors it keeps one somewhat arbitrarily (ridge shares the effect across the group). Both penalties treat a "
        "coefficient of 500 as \"big\" whatever the variable, yet sales per square foot and sales per year of tenure live "
        "on different scales; **standardizing** each predictor puts every coefficient in dollars per standard deviation so "
        "the penalty is fair — fitted inside each cross-validation training fold, or the validation rows leak into it.",
        "A LASSO zero on `store size` while `seats` is kept does **not** mean size is irrelevant, let alone causally "
        "inert: LASSO chose a stand-in. Regularization is a forecasting tool, not a hypothesis test. Large coefficients "
        "are not automatically a problem either; their size depends on units until the predictors are standardized.",
    ),
    (
        "The tuning parameter. What does the model become when $\\lambda = 0$, when it is small, and when it is enormous? "
        "Here are six candidates and the 5-fold cross-validation MSE each produced on the 60 training stores "
        "(hypothetical, millions of dollars²):\n\n"
        "| $\\lambda$ | 0 | 0.1 | 1 | 10 | 100 | 1000 |\n|---|---|---|---|---|---|---|\n"
        "| CV MSE | 9.8 | 9.1 | 8.4 | 8.2 | 9.6 | 15.0 |\n\n"
        "Which would you choose, and why could training MSE not have made this choice?",
        "At $\\lambda = 0$ the penalty vanishes and we have ordinary least squares; small $\\lambda$ shrinks gently; as it "
        "grows the coefficients head to zero (LASSO drops predictors one by one) and the forecast collapses toward the "
        "average sales of the training stores — a model that ignores the store. The coefficients are **learned** by "
        "fitting for a given $\\lambda$; $\\lambda$ itself is **tuned** from outside by comparing candidates on validation "
        "data. Pick $\\lambda = 10$: lowest CV error, and the curve has the expected shape — the strongest penalty (15.0) "
        "is clearly not the best. Training MSE would have chosen 0 every time, since any penalty worsens the fit to the "
        "training stores; and because 8.4 and 8.2 are within the noise of five folds, $\\lambda = 1$ is defensible too.",
        "scikit-learn scales the error term differently across models (`Lasso` divides by $2n$, `Ridge` does not), so "
        "`alpha = 1` is not the same penalty for ridge and LASSO, nor across data sets. The number means nothing on its "
        "own; the CV curve does.",
    ),
    (
        "Grid search, the test set, and the whole picture. `GridSearchCV` is handed a list of candidate $\\lambda$ values. "
        "What does it do with them? The chain locked 40 stores away before we began — if cross-validation already "
        "scores each candidate on unseen stores, why keep those 40 untouched until the end? Finish by explaining to a "
        "manager how regularization, ridge/LASSO, the tuning parameter and cross-validation work together to forecast a "
        "store the chain has never seen.",
        "Grid search runs the *same* K-fold cross-validation — same folds — for every candidate, records each mean "
        "validation score, keeps the best, and refits it on all the training rows; it is the table above, automated. "
        "The best of several candidates is partly the luckiest, so its validation score is a little optimistic — the 40 "
        "stores took no part in any decision, so the one error computed on them after refitting is the honest estimate; "
        "open them once, report, and do not re-tune. To the manager: we fit a linear model but charge a price for large "
        "coefficients, so it stops memorizing quirks of our sample (regularization); ridge charges by squared size and "
        "shrinks everything a little, LASSO charges by absolute size and can drop predictors, giving a simpler model; how "
        "heavy the charge should be is a tuning parameter the fit cannot tell us, so we try a grid of values and let "
        "cross-validation on the training stores vote for the one that forecasts held-out stores best; we refit with that "
        "value, check it once on the stores we locked away, and that model — with standardized inputs — forecasts the new "
        "store.",
        None,
    ),
]

TABLE_Q_INDEX = 3  # the question that contains the markdown table (needs HTML conversion for the page)


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
example, five questions, no derivations.*
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
reveals the answer. Five questions, three to four minutes each.

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
   clothing — Q2–Q4 of this notebook answer them.
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
    "example, five questions, no derivations. Think about each question for a minute, discuss it with a "
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
