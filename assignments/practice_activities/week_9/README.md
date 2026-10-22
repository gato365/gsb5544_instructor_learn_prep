# Week 9 — K-Nearest Neighbors, Decision Trees, and Cross-Validation

Two topics share one interactive page. Topic 9.1 (first meeting) introduces the three algorithms; Topic 9.2
(second meeting) evaluates them: training vs validation error, K-fold cross-validation, model selection,
pipelines. The 9.1 deliverables first, then the 9.2 ones.

## Topic 9.1 deliverables

The four deliverables, plus the animation page they lean on:

| Purpose | File | Built by |
|---|---|---|
| **Week 9 Topics** (student) | `GSB5544_Topic_9_1_KNN_and_Decision_Trees-empty.ipynb` — explanations, 🎬 links into the animation page, the same eight houses reproduced in pandas/scikit-learn (fitted line and residuals, k nearest neighbors and their average, SSE-scored splits), the scikit-learn review with a code comparison, the editable **Pipeline** definition box, five ✅ checkpoints and eight review questions. `____` blanks and *Your answer* prompts. | `tools/build_week9_topics.py` |
| **Week 9 Topics Solutions** | `GSB5544_Topic_9_1_KNN_and_Decision_Trees-solution.ipynb` — the same notebook completed, with every answer written out, **executed** (outputs render on GitHub). | `tools/build_week9_topics.py` |
| **Week 9 Practice Activity** | `GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression.ipynb` — the activity exactly as received (wine data: KNN regression functions and tree sketches; Ames: a categorical feature with KNN, then 10-NN and a depth-4 tree on five features). **Not modified.** | — |
| **Week 9 Practice Activity Solutions** | `GSB_5544_PA_9_1_K_Nearest_Neighbors_for_Regression-solution.ipynb` — every sketch question answered in words *and* drawn by code, every coding question answered in five beats (approach, code, what it does, expected output, common mistakes), **executed** against the live data. | `tools/build_week9_pas.py` |
| Animation page (pairs with both Topics) | `algorithms.html` — **Algorithms in motion**: play / pause / step / restart animations of linear regression, KNN and decision-tree regression on eight houses, with captions, a slider for the new observation, `k` and `max_depth` selectors, the highlighted scikit-learn line for each step, a *Cross-validation in motion* section (below), a glossary of every number on the page, the *Same recipe* comparison table and the *Pipeline* definition box. Self-contained (no external assets). | hand-written; checked by `tools/check_week9_page.py` |

Edit the generators, not the `.ipynb` files, and re-run them. The animation page is edited directly (never
the copy in `docs/`).

## Topic 9.2 deliverables

| Purpose | File | Built by |
|---|---|---|
| **Pre-class reading** (student) | `GSB_5544_Topic_9_2b_Cross_Validation.ipynb` — *Estimating the Test Error*: validation error, 2-fold then K-fold cross-validation on the Bordeaux wines with `cross_val_score(scoring="neg_mean_squared_error")`. **Kept as received.** Listed on the site as a student notebook. | — |
| **Topic 9.2** (student) | `GSB5544_Topic_9_2_Model_Evaluation_and_Cross_Validation-empty.ipynb` — nine sections: (1) the three models and their scikit-learn syntax reviewed on the eight houses, (2) decision trees in depth (candidate splits and SSE, reading `plot_tree`'s `samples` / `value` / `squared_error`, `decision_path`, depth vs training RMSE), (3) training / validation / test data on the wines, (4) training vs validation RMSE (the 1-NN row), (5) K-fold CV written out as a loop — the page's numbers — plus the no-shuffle trap and the two-k's box, (6) `cross_val_score` / `cross_validate` and the negative-score convention, (7) model and hyperparameter selection on the Advertising data with a sealed test set, (8) pipelines and leakage, `GridSearchCV`, (9) review questions. `____` blanks, eight ✅ checkpoints, *Your answer* prompts, 🎬 deep links into the page. | `tools/build_week9_cv_topics.py` |
| **Topic 9.2 Solutions** | `GSB5544_Topic_9_2_Model_Evaluation_and_Cross_Validation-solution.ipynb` — completed, every answer written out, **executed** (reads the wines from dlsun.github.io and Advertising from statlearning.com). | `tools/build_week9_cv_topics.py` |
| **Instructor walkthrough** (phone) | `instructor_walkthrough.html` — 44 short cards, one at a time, big Back / Next buttons, position remembered: what each model does, every number on the page with its value, what to notice when changing each setting, *Say it aloud* scripts for train / validation / test and CV, misconceptions and tap-to-reveal checks per section, a 50-minute (or 20-minute) run of show, a rapid self-quiz. Each card's *Open on the site* link opens `algorithms.html` already set to the right state. Served **unlisted** (`"listed": false` in `site_config.json`): reachable at <https://gato365.github.io/gsb5544_instructor_learn_prep/week_9/instructor_walkthrough.html>, no card on the index. | hand-written |
| **Practice Activity 9.2** | **not yet provided** — no solutions written; nothing about it is assumed in the notebook or the walkthrough. | — |

### The cross-validation section of the page (`#cross-validation`, short link `#cv`)

Model (linear / KNN with `k` / tree with `max_depth`), **K folds** (2, 3, 4, 8 = leave-one-out) and a
**shuffle** checkbox (`KFold(shuffle=True, random_state=5544)`; off = consecutive rows, which on these
size-sorted houses means extrapolating folds). Steps: cut into folds → for each fold *fit* (fresh model on the
other folds; training RMSE) then *validate* (predictions, errors and validation RMSE on the held-out houses) →
average. A fold grid (blue training / orange validation, one row per iteration), the scatter with the fold's
own fitted model and its validation errors, a per-fold table with the mean row, a *settings* chart (mean
training vs validation RMSE by `k` or `max_depth`, hover for values), a *model comparison* table (same folds
for every candidate, lowest mean validation RMSE highlighted) and the scikit-learn code with the matching
lines highlighted (the loop, then `cross_val_score`). `k` choices larger than the smallest training set are
disabled. A † marks a number that depends on how a tie was broken (equal distances, or two candidate splits
with equal SSE): the page keeps the earlier row / lower threshold, scikit-learn may not. The checker verifies
every untied number against `KFold`, a hand loop and `cross_val_score`.

**Deep links** (used by the notebook's 🎬 links and the walkthrough): `#knn&k=1&x=22`,
`#decision-tree&depth=3`, `#cv&model=tree&depth=2&folds=4&shuffle=1`, `#numbers`, `#same-recipe`, `#pipeline`.

### Headline numbers (eight houses, K = 4 shuffled, `random_state=5544`)

Shuffled row order `[5, 6, 4, 1, 2, 0, 3, 7]` (0-based) → folds hold out houses {6, 7}, {2, 5}, {1, 3}, {4, 8}.
3-NN: validation RMSE 38.87 / 17.95 / 60.42 / 79.06, **mean 49.07**; training RMSE 45.28 / 29.12 / 32.66 /
25.53, mean 33.15. Comparison (mean training / mean validation RMSE): line **9.11 / 10.60** (wins), 1-NN 0 /
36.18, 2-NN 27.20 / 46.49, 3-NN 33.15 / 49.07, 5-NN 63.61 / 87.11, tree depth 1 36.81 / 62.90, depth 2
14.93 / 31.86, depth 3 2.89 / 36.18. Without shuffling, 3-NN: K = 4 → 55.91, K = 2 → 127.24. K = 8 (LOO), 3-NN:
39.17. Full-data training RMSE: line 9.45, 3-NN 23.57, tree depth 1 / 2 / 3 / None → 39.92 / 16.27 / 5.00 / 0
(2 / 4 / 7 / 8 leaves). Depth-2 tree nodes: root 8 / 266.25 / 7,248.4 (samples / value / squared_error), left
5 / 208 / 1,736, right 3 / 363.3 / 1,355.6, leaves 176.7 (3), 255 (2), 340 (2), 410 (1).

Wines (27 training vintages, `train_test_split(test_size=0.5, random_state=5544)`, 13 / 14): 1-NN training
0 / validation 0.606; 5-NN 0.514 / 0.502 (the validation RMSE comes out slightly *lower* — the notebook says so
and uses it to motivate CV); 13-NN 0.713 / 0.508. 4-fold shuffled CV, scaled 5-NN: 0.645 / 0.374 / 0.587 /
0.542, mean 0.537; scale-before-CV (leak) 0.5308 vs pipeline 0.5372.

Advertising (`TV` + `radio`, split 150 / 50 `random_state=5544`, `KFold(5, shuffle=True, random_state=5544)`):
linear CV 1.732; scaled KNN best k = 2 (0.878; k = 1 0.978, k = 20 1.544); tree best depth 7 (0.928; depth 10
training 0.012 / validation 0.931). Test RMSE of the refitted winners: linear 1.617, 2-NN **0.811**, tree 1.396.
`GridSearchCV` over k = 1…20 agrees (k = 2, test 0.811).

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
/opt/anaconda3/bin/python tools/check_week9_page.py    # page engine vs scikit-learn, incl. the K-fold section (needs node)
python3 tools/build_week9_topics.py                    # regenerate + execute Topic 9.1 (-empty and -solution; reads Advertising)
python3 tools/build_week9_cv_topics.py                 # regenerate + execute Topic 9.2 (reads wines + Advertising; ~1 min)
python3 tools/build_week9_pas.py                       # rebuild + execute the PA 9.1 solution (reads wine + Ames; ~1 min)
python3 tools/build_site.py                            # copy algorithms.html + instructor_walkthrough.html into docs/week_9/, rebuild docs/index.html
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
8. **Decision tree, node boxes**: at depth 2 every box shows the question (or `predict …` for a leaf), `samples`,
   `value` and `squared_error`; the root reads 8 / 266.3 / 7248.4. Under the diagram, the numbered path for the
   new house lists each question, the answer, and the node reached; with size 22.5 it ends "Prediction … = 340".
9. **Cross-validation** (scroll down or open `#cv`): defaults to 3-NN, K = 4, shuffled. *Play*: the fold grid
   (row 1 holds out houses 6 and 7), then fit / validate pairs with the fold's staircase and orange error
   segments, the table filling row by row (38.87, 17.95, 60.42, 79.06), the mean row (33.15 / 49.07), and a
   final step drawing the refit on all eight houses. The comparison table highlights `LinearRegression()` (10.6)
   with a ★. Switch to the tree: the settings chart shows the validation U (62.9 → 31.9 → 36.2) over a falling
   training curve; hover a point for its values. Untick *shuffle*: fold 1 becomes houses 1 and 2 and the mean
   rises to 55.91. Choose K = 2: `k` = 5 is disabled ("too few training houses"). Dark theme: the fold grid and
   both chart series stay readable.
10. **Deep links**: `#cv&model=tree&depth=2&folds=4&shuffle=1` lands on the section already configured;
    `#knn&k=1&x=22` sets the slider and shows the tie note.
11. **Walkthrough** (`instructor_walkthrough.html`, best on a phone): 44 cards; *Next* / *Back* and ← → move
    one card; the section menu jumps; reload returns to the same card; every *Open on the site* link opens the
    page in the described state. Checks reveal on tap.
