# GSB 5544 — Instructor Prep

Teaching materials for GSB 5544 (Cal Poly).

## Structure
- `assignments/Data/`
  - `coffee_purchases.csv` (pandas) and `coffee_numpy.csv` (same data, no commas in text, for `np.genfromtxt`)
  - `kobe_bryant_games.csv` — one row per game, Kobe Bryant's first 15 seasons (from `nba_goats.xlsx`, sheet `KB`)
  - `sc_crime_sample.csv` — 50,000-row random sample (seed 5544) of `sc_offenses_1991_2020_classroom.csv` (South Carolina NIBRS)
  - `sc_crime_by_year_wide.csv` — offenses by `crime_cat` × year from the full SC file, in *wide* (untidy) form for the tidy-data lesson
- `assignments/practice_activities/week_1/`
  - `pandas_practice_activity/` — pandas notebooks
  - `numpy_practice_activity/` — numpy notebooks
  - `GSB5544_PA_1_1_decode_message*.ipynb`
- `assignments/practice_activities/week_2/` — 2.1 plotnine visualization (Kobe), 2.2 tidy data + Big Five verbs (SC crime)
- `assignments/labs/`, `assignments/quizzes/`

`*-empty.ipynb` notebooks contain `____` blanks for students; `*-solution.ipynb` files are completed **and executed**, so GitHub renders them with outputs.

## Week 1 notebooks (click to view rendered)
| Topic | Empty (student) | Solution (with outputs) |
|---|---|---|
| 1.2 Intro to Pandas | [student](assignments/practice_activities/week_1/pandas_practice_activity/GSB5544_Topic_1_2_Intro_to_Pandas.ipynb) | — |
| 1.3 Pandas Fundamentals | [student](assignments/practice_activities/week_1/pandas_practice_activity/GSB5544_Topic_1_3_Pandas_Fundamentals-empty.ipynb) | [solution](assignments/practice_activities/week_1/pandas_practice_activity/GSB5544_Topic_1_3_Pandas_Fundamentals-solution.ipynb) |
| 1.4 NumPy Essentials | [student](assignments/practice_activities/week_1/numpy_practice_activity/GSB5544_Topic_1_4_NumPy_Essentials-empty.ipynb) | [solution](assignments/practice_activities/week_1/numpy_practice_activity/GSB5544_Topic_1_4_NumPy_Essentials-solution.ipynb) |

Notebooks load the coffee data directly from this repo's raw URL, so they run in Colab as-is.

## Week 2 notebooks
| Topic | Empty (student) | Solution (with outputs) |
|---|---|---|
| 2.1 Data Visualization (plotnine) | [student](assignments/practice_activities/week_2/GSB5544_Topic_2_1_Data_Visualization-empty.ipynb) | [solution](assignments/practice_activities/week_2/GSB5544_Topic_2_1_Data_Visualization-solution.ipynb) |
| 2.2 Data Wrangling (tidy data, Big Five verbs, crosstabs) | [student](assignments/practice_activities/week_2/GSB5544_Topic_2_2_Data_Wrangling-empty.ipynb) | [solution](assignments/practice_activities/week_2/GSB5544_Topic_2_2_Data_Wrangling-solution.ipynb) |

Week 2 student notebooks are front-loaded: the first sections are fully worked, middle sections have `____` blanks, and the last sections plus the Practice Activities are `# your answer here`. Every ✅ Check and PA has an answer in the solution file only.
