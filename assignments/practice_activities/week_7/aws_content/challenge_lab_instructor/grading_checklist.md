# Grading checklist: Cloudy with a Chance of Parquet

Each line maps to one automatic deduction in the handout. Check the boxes that pass.

## Functions work as specified

- [ ] `size_up("parquet/by_year/YEAR=2024/ELEMENT=TMAX/")` returns a dict with keys `prefix`, `objects`, `bytes`, `readable`; 9 objects, about 14.8 MB.
- [ ] `size_up("csv/by_station/USC0004")["objects"]` is about 1190, not 1000.
- [ ] `size_up` on a prefix with no matches returns 0 objects and 0 bytes without an error.
- [ ] `read_slice(2024, "TMAX", ["USC00047851"])` returns 366 rows with columns `station_id`, `date`, `element`, `value`; `date` is a datetime; first value 18.3.
- [ ] `read_slice(2024, "PRCP", [...], budget_mb=10)` raises `ValueError` and the message includes the readable size.
- [ ] `patio_summary` returns the seven named columns, twelve rows per city, with integer `days_reported` and `patio_days`.

## Named special cases

- [ ] Messy ids: Phoenix and Miami have 366 `days_reported` in the example run (not 0).
- [ ] Missing days: Seattle totals 364 `days_reported`.
- [ ] Failed quality checks: `len(read_slice(2024, "TMAX", ["USC00305606"]))` is 285, not 362; Mount Sinai NY totals 282 `days_reported`.
- [ ] No temperature: Atascadero rain gauge has `avg_tmax_c` `NaN`, a real `total_prcp_mm` (73.4 in January), and 0 patio days.
- [ ] Does not exist: Atlantis has twelve rows, both averages `NaN`, both counts 0.
- [ ] Units: San Luis Obispo January `avg_tmax_c` is 18.0, not 180.

## No hard-coding

- [ ] No sizes, ids, years, or row counts typed inside any function body.
- [ ] The receipt's two byte counts come from `size_up` and `os.path.getsize`.
- [ ] The surprise run changed only the URL or file name.

## No banned shortcuts

- [ ] No key under `csv/by_year/`, `csv.gz/by_year/`, or `csv/by_station/` is read. (Listing `csv/by_station/` in the Function 1 test is allowed.)
- [ ] No Athena client or SQL.
- [ ] `filters=` appears inside `pd.read_parquet`; there is no full-slice read followed by `.isin` or `==` on the id.
- [ ] `size_up` uses a paginator or a continuation-token loop.
- [ ] No `iterrows`, `itertuples`, or `for` loop over DataFrame rows.

## Functions build on each other

- [ ] `read_slice` calls `size_up`.
- [ ] `patio_summary` calls `read_slice` (for both elements).

## Complete final output

- [ ] `patio_summary.csv`: 84 rows; city totals match the solution (SLO 209, Miami 142, Phoenix 132, Mount Sinai NY 98, Seattle 86).
- [ ] `patio_summary_surprise.csv`: 84 rows; matches `patio_summary_surprise_expected.csv` (Honolulu 183, Chicago 100, Lima MT 83, Denver 82, Dannemora NY 51).
- [ ] Receipt present, shrink factor roughly 10,000 to 25,000.
- [ ] `patio_days.png` present, five lines, one per city with data.

## Run in the right place

- [ ] The computer name printed in the SageMaker notebook is not the student's laptop.
- [ ] The laptop chart output shows a different computer name.
- [ ] Screenshot shows the `gsb5544` space with status Stopped.

Small differences in counts (a row or two, a few bytes) are expected if NOAA has revised the archive since October 5, 2026. Rerun the instructor solution before grading to get current numbers.
