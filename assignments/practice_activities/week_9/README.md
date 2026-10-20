# Week 9 — K-Nearest Neighbors and Decision Trees for Regression

The four deliverables, plus the animation page they lean on:

| Purpose | File | Built by |
|---|---|---|
| **Week 9 Topics** (student) | `GSB5544_Topic_9_1_KNN_and_Decision_Trees-empty.ipynb` — explanations, 🎬 links into the animation page, the same eight houses reproduced in pandas/scikit-learn (fitted line and residuals, k nearest neighbors and their average, SSE-scored splits), the scikit-learn review with a code comparison, the editable **Pipeline** definition box, five ✅ checkpoints and eight review questions. `____` blanks and *Your answer* prompts. | `tools/build_week9_topics.py` |
| **Week 9 Topics Solutions** | `GSB5544_Topic_9_1_KNN_and_Decision_Trees-solution.ipynb` — the same notebook completed, with every answer written out, **executed** (outputs render on GitHub). | `tools/build_week9_topics.py` |
| **Week 9 Practice Activity** | `GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression.ipynb` — the activity exactly as received (wine data: KNN regression functions and tree sketches; Ames: a categorical feature with KNN, then 10-NN and a depth-4 tree on five features). **Not modified.** | — |
| **Week 9 Practice Activity Solutions** | `GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression-solution.ipynb` — every sketch question answered in words *and* drawn by code, every coding question answered in five beats (approach, code, what it does, expected output, common mistakes), **executed** against the live data. | `tools/build_week9_pas.py` |
| Animation page (pairs with the Topic) | `algorithms.html` — **Algorithms in motion**: play / pause / step / restart animations of linear regression, KNN and decision-tree regression on eight houses, with captions, a slider for the new observation, `k` and `max_depth` selectors, the highlighted scikit-learn line for each step, the *Same recipe* comparison table and the *Pipeline* definition box. Self-contained (no external assets). | hand-written; checked by `tools/check_week9_page.py` |

Edit the generators, not the `.ipynb` files, and re-run them. The animation page is edited directly (never
the copy in `docs/`).

## How the page is published

As in Week 3, the page is listed in `site_config.json` with a `"page"` key and `tools/build_site.py` copies
it to `docs/week_9/algorithms.html`, so GitHub Pages serves it at
<https://gato365.github.io/gsb5544_instructor_learn_prep/week_9/algorithms.html>. The notebooks link to it
there (absolute links, so they work from Colab). Deep links: `#linear-regression`, `#knn`, `#decision-tree`
select an algorithm; `#same-recipe` and `#pipeline` jump to those sections.

## The animation data

Eight houses, chosen so that no tie or boundary case appears at the default new size (22.5):

| size (100s sq ft) | 10 | 12 | 15 | 18 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|
| price ($1,000s) | 150 | 180 | 200 | 260 | 250 | 320 | 360 | 410 |

Line: price ≈ 33.29 + 11.72 × size, SSE 715.1, prediction at 22.5 = 297.0. KNN at 22.5: k = 1 → 320,
k = 2 → 285, k = 3 → 276.67 (houses 24, 20, 18), k = 5 → 278, k = 8 → 266.25 (the mean). Tree: root split
size ≤ 22 (SSE 12,747); depth 2 predicts 340 (leaves 176.67 / 255 / 340 / 410); depth 3 predicts 320.
With a `garage` column, one-hot + scaled, 3-NN predicts 310.

Advertising data (Topic 8.2's), `TV` + `radio`, split `random_state=5544`: test RMSE linear 1.62, scaled
3-NN 0.90, tree depth 4 1.82.

## PA 9.1 headline numbers

Wine (27 training vintages, ages 12–40 with 36 and 38 missing): 1-NN jumps at the midpoints between ages,
2-NN at the ages themselves (tie at exactly 13), 27-NN is flat at mean log price 3.158. One split at 20.5:
means 2.830 (9) / 3.322 (18); splits at 20.5 and 30.5: 2.830 / 3.082 / 3.622.

Ames (2,930 houses): with `Neighborhood` one-hot and 10-NN, only **Blueste** (exactly 10 houses) has
predicted = mean; larger neighborhoods use an arbitrary 10 of their tied houses, smaller ones borrow from other
neighborhoods. Five features, scaled numerics, 10-NN: RMSE 34,695 on the fitted data; split
`random_state=5544`, test RMSE by k: 1 → 46,898, 3 → 37,954, 5 → 38,321, 10 → 40,113, 20 → 41,820,
50 → 45,028, 100 → 46,654 (unscaled 10-NN: 56,547). Tree, numerics unscaled, depth 4: root
`Gr Liv Area ≤ 1610.5`, 16 leaves from 73,888 to 573,954, RMSE 47,869 on the fitted data; test RMSE by
depth: 1 → 63,050, 4 → 49,104, 6 → 43,213, 10 → 39,593, 15 → 43,149, None → 42,102.

## Rebuild checklist

```bash
/opt/anaconda3/bin/python tools/check_week9_page.py    # page engine vs scikit-learn (needs node)
python3 tools/build_week9_topics.py                    # regenerate + execute Topic 9.1 (-empty and -solution; reads Advertising)
python3 tools/build_week9_pas.py                       # rebuild + execute the PA 9.1 solution (reads wine + Ames; ~1 min)
python3 tools/build_site.py                            # copy algorithms.html into docs/week_9/ and rebuild docs/index.html
```

## How to test the page

Open `assignments/practice_activities/week_9/algorithms.html` directly in a browser (no server needed).

1. **Linear regression** loads with eight points. *Play*: three grey candidate lines, the red fitted line with
   its equation, eight purple residuals drawn one by one (the table fills in), the SSE, then the orange new house
   rising to the line and across to the price axis (297.0). The code panel highlights `fit`, `intercept_`/`coef_`,
   the residual line, then `predict`.
2. **K-nearest neighbors**: distances are measured one by one as green brackets along the x-axis (grey once past
   k), the k neighbors get rings, the prediction appears, then the red staircase. Change *k* to 8: the last step
   is flat at 266.25 and the caption says so. Change *k* to 1: the staircase passes through every point.
3. **Decision tree**: seven candidate splits are tried with their SSE listed on the right, the best (22) is kept,
   the tree diagram grows one level per step, the new house follows the questions (path highlighted in the
   diagram and the region shaded), and the staircase appears. *max_depth* 1 and 3 change the tree accordingly.
4. **Slider**: drag *New house size* to 30 under KNN; every number recomputes. At 22 under the tree, the caption
   notes 22 ≤ 22 is true (left). Under KNN at 22 with k = 1 the tie note appears.
5. **Controls**: Back / Step / Restart, Space, ←, →, R; speed; *Theme* cycles Auto / Light / Dark and the
   tints stay readable. Narrow the window: the panels stack.
6. `#knn` and `#decision-tree` in the address bar select the algorithm; `#pipeline` scrolls to the definition.
7. `tools/check_week9_page.py` confirms the page's numbers equal scikit-learn's for every slider position.
