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

## Naming convention
`GSB5544_<Topic|PA>_<week>_<number>_<slug>-empty.ipynb` — student version (fill in the `____` blanks / "YOUR CODE HERE" cells)
`GSB5544_<Topic|PA>_<week>_<number>_<slug>-solution.ipynb` — completed **and executed** (GitHub renders outputs)
No suffix — complete lecture notes with no student version.

Notebooks live under `assignments/practice_activities/week_<N>/`.

## Course website
The site is a single generated page, `docs/index.html`, served by GitHub Pages.

- **Edit `site_config.json`** to add/retitle notebooks or change what is published.
- **Hide material** with `"publish": false` + a `"placeholder"` line (the page shows the
  placeholder, e.g. "Coming Thursday: ..."), or hide just a solution with
  `"publish_solution": false` + `"solution_placeholder"`. Files stay in the repo either way.
- **Rebuild** after editing: `python3 tools/build_site.py`, then commit `docs/`.

Note: the repository is public, so hiding removes material from the *website page* only —
a determined student can still browse the repo. Keep anything truly secret out of the repo.
