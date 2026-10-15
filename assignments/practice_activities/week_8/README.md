# Week 8 — Regression in Python and the first ideas of machine learning

| File | What it is | Built by |
|---|---|---|
| `GSB5544_Topic_8_2_Regression-empty.ipynb` | Student notebook. Students already know regression from other courses (assumptions, coefficients, polynomials, interactions), so nothing is re-taught. (1) **How to do it in Python**: the scikit-learn recipe (`X` as a DataFrame, `fit`, `intercept_`/`coef_`, `predict`, `score`), the 1-D-`X` error, more features / an interaction / `PolynomialFeatures` as "more columns". (2) **ML vocabulary on familiar ground**: features and target, parameters vs hyperparameters, `train_test_split`, RMSE on train vs test, a polynomial-degree loop in a scaled `make_pipeline` that shows overfitting. `____` blanks and three ✅ checks. 20 minutes. | `tools/build_week8_topics.py` |
| `GSB5544_Topic_8_2_Regression-solution.ipynb` | The same notebook, completed and executed. | `tools/build_week8_topics.py` |

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

## Rebuild checklist

```bash
python3 tools/build_week8_topics.py    # regenerate + execute Topic 8.2 (-empty and -solution)
python3 tools/build_site.py            # rebuild docs/index.html
```
