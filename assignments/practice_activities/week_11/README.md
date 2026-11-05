# Week 11 — Introduction to Classification

| Purpose | File | Built by |
|---|---|---|
| **Topic 11.1** (student) | `GSB5544_Topic_11_1_Classification_Scores-empty.ipynb` — *Classification: What Every Score Measures.* More mathematical than the regression topics: every score is defined from the four cells of the confusion matrix, computed by hand, checked with scikit-learn and discussed (what it ignores, when it misleads). Running example: Palmer penguins — sex from four body measurements (two-class), species (three-class) — so the PA's storms and Titanic stay fresh. Sections: regression → classification and the KNN vote (p̂ = share of neighbors); the confusion matrix and scikit-learn's row/column layout; accuracy and the majority baseline; precision, recall, specificity, NPV (the last two = recall/precision of the negative class, `pos_label=0`); F1 as harmonic mean, with the 2TP/(2TP+FP+FN) form and Fβ; multi-class per-class scores and macro vs weighted averages; probabilities and the threshold, including scikit-learn's tie rule at exactly 0.5; precision–recall and ROC curves, AP and AUC; log loss; which score to lead with. Eight ✅ checkpoints, six review questions (one is the F1 algebra), "lines to keep". `____` blanks. | `tools/build_week11_topics.py` |
| **Topic 11.1 Solutions** | `GSB5544_Topic_11_1_Classification_Scores-solution.ipynb` — completed and **executed** (needs `palmerpenguins`; a guarded install cell covers Colab). | `tools/build_week11_topics.py` |
| **PA 11.1** (student) | `GSB_5544_PA_11_1_Intro_to_Classification.ipynb` — storms: a hand-made hurricane rule and its accuracy / precision / recall / F1 for both classes, 10-NN reasoning about status; Titanic: 10-NN survival from age, sex and class, predicted probabilities, precision / recall / F1 for both classes, threshold 0.8, precision–recall curves for both classes. **Kept as received.** | — |
| **PA 11.1 Solutions** | `GSB_5544_PA_11_1_Intro_to_Classification-solution.ipynb` — every DISCUSS prompt answered in words and checked by code; every coding question in five beats (approach, code, what it does, expected output, common mistakes). **Executed** against the live data. | `tools/build_week11_pas.py` |
| **Topic 11.2** (student) | `GSB5544_Topic_11_2_Classification_Metrics_in_Context-empty.ipynb` — *Classification Metrics in Context: Cancer Screening and Fire Detection.* A guided, discussion-led class activity, **calculator only (no code cells)**. Part 1: the confusion matrix and ten statistics (accuracy, error rate, precision, recall, F1, specificity, FPR, FNR, NPV, balanced accuracy), each with its equation, plain-language definition, the question it answers, numerator and denominator, range and direction; a "which denominator" table. Parts 2 and 3: two hypothetical scenarios, each the same 1,000 cases at two thresholds; every calculation question runs equation → meaning of the values → calculation → interpretation. Part 4: costs of the two errors, thresholds as cutoffs on predicted probabilities (logistic regression), what else one would need to know. 38 questions (20 calculation, 18 discussion); labelled work space, not blanks. About 75 minutes; the key gives a 50-minute cut. | `tools/build_week11_metrics_activity.py` |
| **Topic 11.2 Answer key** (instructor) | `GSB5544_Topic_11_2_Classification_Metrics_in_Context-solution.ipynb` — every calculation worked, an interpretation for each, suggested discussion responses, instructor notes (timing, where the discussion is headed) and a table of all ten statistics for the four threshold rows. Nothing to execute. Listed on the site with the solution link withheld (`publish_solution: false`). | `tools/build_week11_metrics_activity.py` |

Edit the generators, not the `.ipynb` files, and re-run them.

Topic 11.1 includes fill-in-the-blank definitions in the prose and tables, alongside the code blanks.
The student version uses `____` for missing terms and phrases; the solution shows the matching completions
in **bold**. Formulas and worked examples provide context for the definitions.

## Headline numbers

**Topic 11.1** (penguins, 333 complete rows, stratified 25 % test split `random_state=5544`, scaled 10-NN). Sex
(female = 1), test n = 84: confusion `[[37, 5], [8, 34]]` → TP 34, FP 5, FN 8, TN 37; accuracy 0.845 (baseline
0.500); precision 0.872, recall 0.810, specificity 0.881, NPV 0.822; F1 0.840 (male class 0.851); thresholds
0.3 / 0.5 / 0.7 / 0.9 → precision 0.848 / 0.881 / 0.914 / 1.000, recall 0.929 / 0.881 / 0.762 / 0.571 (the ≥ 0.5
rule differs from `predict()` on the exact-0.5 ties); AUC 0.957, AP 0.955, log loss 0.266. Species: Adelie
precision 0.881 / recall 1.000, Chinstrap 1.000 / 0.706 (5 of 17 called Adelie), Gentoo 1.000 / 1.000; accuracy
0.940; macro recall 0.902, weighted 0.940.

**Topic 11.2** (hypothetical counts, TP / FP / FN / TN; every value is computed in the generator). Cancer
screening, 1,000 people, 100 with cancer — A 90 / 180 / 10 / 720: accuracy 0.810 ("no one has cancer" scores 0.900),
precision 0.333, recall 0.900, F1 0.486, specificity 0.800, FPR 0.200, FNR 0.100, NPV 0.986, balanced accuracy 0.850;
B 80 / 45 / 20 / 855: 0.935, 0.640, 0.800, 0.711, 0.950, 0.050, 0.200, 0.977, 0.875. A → B trades 135 fewer false
positives for 10 more misses (13.5 per miss). Fire detection, 1,000 periods, 50 with a fire — A 45 / 95 / 5 / 855:
accuracy 0.900 ("never alarm" 0.950), precision 0.321, recall 0.900, F1 0.474 (0.473 from rounded inputs),
specificity 0.900, FPR 0.100, FNR 0.100, NPV 0.994, balanced accuracy 0.900; B 35 / 19 / 15 / 931: 0.966, 0.648, 0.700,
0.673, 0.980, 0.020, 0.300, 0.984, 0.840. A → B trades 76 fewer false alarms for 10 more undetected fires (7.6 per
miss). Accuracy and F1 favor B in both scenarios; balanced accuracy favors B for screening and A for fire.

**PA 11.1 — storms** (19,066 records, 4,684 hurricanes). Rule *wind ≥ 64 knots*: TP 4,684, FN 0, FP 128 (125
extratropical), TN 14,254 → accuracy **99.33 %**, hurricane precision **97.34 %**, recall **100 %**, F1 **98.65 %**;
non-hurricane precision **100 %**, recall **99.11 %**, F1 **99.55 %**. (Pressure ≤ 980 would give recall 68 %.)
10-NN on (wind, pressure) at pressure 975, wind 39 → **extratropical** (9 of 10 neighbors unscaled, 10 of 10
scaled). 10-NN status recall: hurricane 0.99, tropical storm 0.95, tropical depression 0.90, extratropical 0.37,
subtropical depression 0.13, other low 0.08, disturbance / subtropical storm / tropical wave 0.00.

**PA 11.1 — Titanic** (2,205 people with age; 711 survived = 32 %; `class` has 7 levels incl. crew). 10-NN on
scaled age + one-hot gender, class, fitted on all rows. 20-year-old woman, 1st class → survives, P = 1.0; 3rd
class → does not, P(survive) = 0.2. `predict()` confusion `[[1427, 67], [332, 379]]`: survivors precision 0.850,
recall 0.533, F1 0.655; non-survivors 0.811 / 0.955 / 0.877; accuracy 0.819 (baseline 0.678). Rule ≥ 0.5 (119
people sit at exactly 0.5): survivors 0.781 / 0.620, non-survivors 0.835 / 0.917. Threshold 0.8: survivors
precision **0.919**, recall **0.415**; non-survivors **0.779** / **0.983**. PR curve (survivors), threshold →
(precision, recall): 0 (0.32, 1.00), 0.5 (0.78, 0.62), 0.8 (0.92, 0.41), 1.0 (1.00, 0.21); AP 0.775. PR curve
(non-survivors): 0 (0.68, 1.00), 0.5 (0.81, 0.96), 0.8 (0.89, 0.71), 1.0 (0.99, 0.25). AUC 0.865.

## Rebuild checklist

```bash
python3 tools/build_week11_topics.py   # regenerate + execute Topic 11.1 (palmerpenguins; no downloads)
python3 tools/build_week11_pas.py      # rebuild + execute the PA 11.1 solution (reads storms + Titanic live)
python3 tools/build_week11_metrics_activity.py   # write the Topic 11.2 activity and its answer key (no code, nothing executed)
python3 tools/build_site.py            # rebuild docs/index.html
```
