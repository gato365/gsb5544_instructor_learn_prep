# Week 8 — Regression

| File | What it is | Built by |
|---|---|---|
| `GSB5544_Topic_8_2_Regression-empty.ipynb` | Student notebook: *Does advertising move sales?* Business question → summary statistics and `.corr()` → scatterplot with `geom_smooth(method="lm")` → simple linear regression with scikit-learn (`fit`, `intercept_`, `coef_`, `predict`, `score`, residuals) → **predict vs explain** → multiple linear regression with two quantitative features (TV + radio) → why newspaper's slope vanishes. `____` blanks and "Your answer" prompts. 20 minutes. | `tools/build_week8_topics.py` |
| `GSB5544_Topic_8_2_Regression-solution.ipynb` | The same notebook, completed and executed. | `tools/build_week8_topics.py` |

Edit the generator, not the `.ipynb` files, and re-run it.

## About the data

The **Advertising** data set from *An Introduction to Statistical Learning* (James, Witten, Hastie,
Tibshirani): 200 markets, TV / radio / newspaper budgets in thousands of dollars, sales in thousands of
units. The notebook reads it from the book's site (`https://www.statlearning.com/s/Advertising.csv`,
`index_col=0`); a copy is in `assignments/Data/advertising.csv` (same columns, no index column) in case
that URL changes — if so, point `DATA_URL` in the builder at the raw-GitHub copy and rebuild.

Headline numbers quoted in the ✅ answers (scikit-learn 1.9, pandas 3.0):

| model | intercept | slopes | R² | RMSE |
|---|---|---|---|---|
| sales ~ TV | 7.033 | TV 0.0475 | 0.612 | 3.24 |
| sales ~ TV + radio | 2.921 | TV 0.0458, radio 0.1880 | 0.897 | 1.67 |
| sales ~ TV + radio + newspaper | 2.939 | TV 0.0458, radio 0.1885, newspaper −0.0010 | 0.897 | — |

Correlations with sales: TV 0.78, radio 0.58, newspaper 0.23; TV–radio 0.05; radio–newspaper 0.35.

## Why this design

The whiteboard plan for Topic 8.2 was: Python (scikit-learn, plotnine); regression; data = any business
setting with a real relationship; (1) question → summary statistics → visualisation (scatterplot) →
analysis (predict, explain) with simple linear regression and one feature; (2) multiple linear regression
with two quantitative explanatory features. The notebook follows that order exactly, and adds the one
scikit-learn gotcha students always hit (`X` must be a DataFrame, shown as a deliberate error cell).

## Rebuild checklist

```bash
python3 tools/build_week8_topics.py    # regenerate + execute Topic 8.2 (-empty and -solution)
python3 tools/build_site.py            # rebuild docs/index.html
```
