# Week 10 — Model Selection, Hyperparameter Tuning, and Regularization

| Purpose | File | Built by |
|---|---|---|
| **Topic 10.1** (student) | `GSB5544_Topic_10_1_Model_Selection_and_Hyperparameter_Tuning-empty.ipynb` — deliberately straightforward, built to scaffold PA 10.1 on the same Ames data without doing it: the one rule (compare candidates on cross-validated test RMSE, same folds), two helpers (`preprocessing`, `cv_rmse`), model selection as a loop over feature sets, tuning k by hand, `GridSearchCV` (`param_grid`, the `step__parameter` naming convention, `best_params_`, negative `best_score_`, `cv_results_` → DataFrame → plot), two and three parameters at once (k × metric × weights) and the fit count, comparing model families on the same folds, `best_estimator_` to predict. Six ✅ checkpoints, five review questions, "lines to keep". `____` blanks. | `tools/build_week10_topics.py` |
| **Topic 10.1 Solutions** | `GSB5544_Topic_10_1_Model_Selection_and_Hyperparameter_Tuning-solution.ipynb` — completed and **executed**. | `tools/build_week10_topics.py` |
| **PA 10.1** (student) | `GSB5544_PA_10_1_Model_Section_and_Hyperparameter_Tuning.ipynb` — Ames: five feature sets with 10-NN, grid search over k with a plot, grid over k × metric, the best KNN model, linear regression on the same features; then the Lending Club loans competition (smallest test RMSE for `interest_rate`, `GridSearchCV` required). **Kept as received** (file name as received, "Section"). | — |
| **PA 10.1 Solutions** | `GSB5544_PA_10_1_Model_Section_and_Hyperparameter_Tuning-solution.ipynb` — every coding part in five beats (approach, code, what it does, expected output, common mistakes); the loans competition run twice — with application-time features (the real contest) and with the grade columns (to expose that `sub_grade` *is* the rate). **Executed** against the live data (a few minutes of grid searches). | `tools/build_week10_pas.py` |

| **Shrink to Fit** (interactive page) | `Shrink to Fit.html` — regularization on a simulated coffee-shop chain (100 stores, ten predictors, 60 train / 40 test): penalty slider with coefficients and train/CV error curves, ridge vs LASSO coefficient paths with a predict-then-reveal, grid search animated fold by fold with a one-time test-set reveal, elastic net heatmap, Python snippets, PA checklist. Received as an HTML page; its *Think · Discuss · Reveal* box now holds the five Topic 10.2 questions (pasted from `build_week10_reg_topics.py --questions-js`). Served at `week_10/shrink_to_fit.html`; the source keeps its original name. Loads one Google Fonts stylesheet; everything else is inline. | received; questions by `tools/build_week10_reg_topics.py` |
| **Topic 10.2** (student) | `GSB5544_Topic_10_2_Regularization_Ridge_LASSO-empty.ipynb` — a 15–20 minute conceptual opener on the coffee-shop example: the least-squares / ridge / LASSO objectives with each term labelled and the `alpha` scaling note, five scaffolded think-and-discuss questions (least squares and overfitting → the penalty and why a worse fit can forecast better → ridge vs LASSO and standardization → the tuning parameter, chosen from a CV table → grid search, validation vs test, and a synthesis for a manager), a misconceptions table, one eight-line simulated demo, and the hand-off into the walkthrough and PA 10.2. Student copy has *Your answer* prompts. | `tools/build_week10_reg_topics.py` |
| **Topic 10.2 Solutions** | `GSB5544_Topic_10_2_Regularization_Ridge_LASSO-solution.ipynb` — the short answers / instructor notes under every question, **executed**. Also answers the twelve DISCUSS prompts of the walkthrough (Q2–Q4 cover them). | `tools/build_week10_reg_topics.py` |
| **Penalized Regression walkthrough + PA 10.2** | `GSB_5544_PA_10_2_penalized_regression_solutions.ipynb` — received as is (file name as received): the Ames code walkthrough (all predictors, polynomial blow-up, `Ridge` / `Lasso` / `ElasticNet` pipelines, zero-coefficient counts, `GridSearchCV` over `alpha` and `l1_ratio`) with twelve DISCUSS prompts, then the loans competition with penalized models. Listed on the site as the student notebook. No separate loans key has been written for it. | — |

Edit the generators, not the `.ipynb` files, and re-run them. The PA refers to a "pre-class notebook" (a Model
Selection section and a note on grid-search naming conventions) that is not in this repository; Topic 10.1
covers both points, so the solution's cross-references point at the topic.

## Headline numbers

**Topic 10.1** (Ames, `KFold(5, shuffle=True, random_state=5544)`, 10-NN unless stated): feature sets — size only
57,893; size + quality 36,549; + year 33,509; + Neighborhood **32,980** (used for the rest). By-hand k: 1 → 36,740,
3 → 33,394, 5 → 33,150, 10 → 32,980, 20 → 33,296, 50 → 34,313. `GridSearchCV` over k = 1 … 25: best **k = 8,
32,928** (k = 7 – 10 within ~100; fold sd ≈ 2,700); 126 fits (125 + the refit). k × metric: Manhattan k = 6,
32,860 (250 fits). k × metric × weights: **Manhattan, k = 6, distance → 30,960**. Tree grid best `max_depth=6`
35,333; linear 35,627. `best_estimator_` on a 1,500 sq ft, quality-7, 2005 CollgCr house → ≈ 208,400.

**PA 10.1** (Ames, `KFold(10, shuffle=True, random_state=5544)`): part 1, 10-NN by set — A (base 4 + Neighborhood)
38,441; B (area + quality) 36,346; C (+ year + Neighborhood) 32,640; D (+ basement, garage) 29,649; **E** (+ lot,
full bath, rooms, fireplaces + Kitchen Qual, Exter Qual) **28,766**. Part 2, grid k = 1 … 30 on the five features:
**k = 3, 38,200** (plateau 38,200 – 38,650 for k = 3 … 10; k = 1 45,657). Part 3: **Manhattan k = 5, 37,451** vs
Euclidean k = 3, 38,200; 600 fits. Part 4: set E Manhattan k = 4 distance 27,670; 12 numeric + Neighborhood **Manhattan k = 8
distance 27,322**; 12 numeric + 3 categoricals ≈ 27,000 – 27,700; `weights="distance"` is the setting that helps.
Part 5: linear on the winner's features ≈ 30,900 – 32,600 (set E 30,864; 12 numeric + Neighborhood 32,601) vs
KNN ≈ 27,300.

**Loans** (10,000 loans, `KFold(5, shuffle=True, random_state=5544)`, RMSE in percentage points; sd of the rate
5.00): linear term + debt_to_income 4.63; 17 application-time numerics (logged amounts) 4.19; + 4 categoricals 4.10;
**degree-2 polynomial + categoricals 3.93** (best honest model); KNN k = 40 distance 4.18; tree (no depth limit, min 50 per
leaf) 4.27. Circular: grade only 1.11; **sub_grade only 0.19** (within-sub-grade sd ≈ 0.005); installment +
loan_amount + term 3.49 (post-loan leak).

## Rebuild checklist

```bash
python3 tools/build_week10_topics.py   # regenerate + execute Topic 10.1 (-empty and -solution; reads Ames)
python3 tools/build_week10_pas.py      # rebuild + execute the PA 10.1 solution (reads Ames + loans; a few minutes)
python3 tools/build_week10_reg_topics.py                  # regenerate + execute Topic 10.2 (no downloads)
python3 tools/build_week10_reg_topics.py --questions-js    # the QUESTIONS array to paste into "Shrink to Fit.html" if the questions change
python3 tools/build_site.py            # rebuild docs/index.html
```
