# Week 8 — Regression in Python and the first ideas of machine learning

| File | What it is | Built by |
|---|---|---|
| `GSB5544_Topic_8_2_Regression-empty.ipynb` | Student notebook. Students already know regression from other courses (assumptions, coefficients, polynomials, interactions), so nothing is re-taught. (1) **How to do it in Python**: the scikit-learn recipe (`X` as a DataFrame, `fit`, `intercept_`/`coef_`, `predict`, `score`), the 1-D-`X` error, more features / an interaction / `PolynomialFeatures` as "more columns". (2) **ML vocabulary on familiar ground**: features and target, parameters vs hyperparameters, `train_test_split`, RMSE on train vs test, a polynomial-degree loop in a scaled `make_pipeline` that shows overfitting. `____` blanks and three ✅ checks. 20 minutes. | `tools/build_week8_topics.py` |
| `GSB5544_Topic_8_2_Regression-solution.ipynb` | The same notebook, completed and executed. | `tools/build_week8_topics.py` |
| `GSB_5544_PA_8_2_Linear_Regression_Models.ipynb` | PA 8.2 student notebook: The Pudding's 90s song-recognition data (linear, log-outcome and quadratic models predicting Gen Z recognition from Millennial recognition) and the FEV data (a dummified categorical predictor, + age, + interaction). Kept as received. | — |
| `GSB_5544_PA_8_2_Linear_Regression_Models-solution.ipynb` | Instructor key: approach, code, what each piece does, expected output, common mistakes for every part; written answers for FEV-3, FEV-4 and FEV-9; an instructor-extra 5-fold cross-validation comparing the three song models in original units. Executed against the live data. | `tools/build_week8_pas.py` |

Edit the generator, not the `.ipynb` files, and re-run it.

## Scope note

Per the instructor team's overlap discussion: do **not** spend time reviewing regression itself (LINE
assumptions, interpreting coefficients, polynomial models, interactions). The job of this topic is only
(1) show how to do it in Python and (2) use linear regression as the familiar context for general
machine-learning concepts. The notebook is built to that brief.

## About the data

The **Advertising** data set from *An Introduction to Statistical Learning* (James, Witten, Hastie,
Tibshirani): 200 markets, TV / radio / newspaper budgets in thousands of dollars, sales in thousands of
units. Read from `https://www.statlearning.com/s/Advertising.csv` (`index_col=0`); a copy is in
`assignments/Data/advertising.csv` (no index column) in case that URL changes — if so, point `DATA_URL`
in the builder at the raw-GitHub copy and rebuild.

## Headline numbers quoted in the ✅ answers

Fitted on all 200 rows (scikit-learn 1.9, pandas 3.0):

| model | coefficients | R² |
|---|---|---|
| sales ~ TV | 7.033 + 0.0475 TV | 0.612 |
| sales ~ TV + radio | TV 0.0458, radio 0.1880 | 0.897 |
| + TV×radio | TV 0.0191, radio 0.0289, TV×radio 0.00109 | 0.968 |
| `PolynomialFeatures(degree=2)` on TV, radio | 5 terms | 0.986 |

Train/test split `test_size=0.25, random_state=5544` (150 / 50 rows), TV + radio: train RMSE 1.694,
test RMSE 1.617. Degree loop (scaled pipeline): training RMSE falls monotonically to 0.24 at degree 10;
test RMSE bottoms at degree 3 (≈ 0.37) and rises to ≈ 0.71 at degree 10. Changing `random_state`
changes the curve's shape; re-check the Checkpoint 3 answer if you do.

## PA 8.2 headline numbers

Songs (344): linear gen_z ≈ −0.048 + 0.707·millennial (R² 0.80, 11 negative predictions); log model
log(gen_z) ≈ −2.91 + 2.80·millennial; quadratic 0.109 − 0.273·m + 1.021·m² (R² 0.90). 5-fold CV RMSE in
original units: 0.095 / 0.064 / 0.067 — the log model wins narrowly. *Waterfalls* (m = 0.846, actual 0.497)
predicts 0.550 / 0.583 / 0.609; *Smells Like Teen Spirit* at m = 0.7 predicts 0.447 / 0.387 / 0.418.

FEV (654): smoker-only model 2.566 + 0.711·smoker (group means; R² 0.06); + age (months): −0.217·smoker +
0.0192·age (R² 0.58, sign flips); + interaction: smoker +1.870, age 0.0202, smoker×age −0.0131 (R² 0.59;
growth 0.24 vs 0.08 L/year). 5-fold CV RMSE 0.84 / 0.57 / 0.56 L.

## Rebuild checklist

```bash
python3 tools/build_week8_topics.py    # regenerate + execute Topic 8.2 (-empty and -solution)
python3 tools/build_week8_pas.py       # rebuild + execute the PA 8.2 instructor solution (reads both data sets live)
python3 tools/build_site.py            # rebuild docs/index.html
```
