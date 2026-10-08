# Surprise dataset: how it was built and how to release it

`stations_surprise.csv` has the same two columns as the students' `stations.csv` (`city`, `station_id`) and seven rows. It was built by hand on October 5, 2026 from stations whose 2024 records in `noaa-ghcn-pds` show the same irregularities as the example list. It contains every named special case and no new one.

| Named special case | In `stations.csv` | In `stations_surprise.csv` |
|---|---|---|
| 1. Messy ids (spaces, lowercase) | Phoenix `" usw00023183"`, Miami `"USW00012839 "` | Chicago `"usw00094846 "`, Denver `" USW00003017"`, Lima MT `usc00245030` |
| 2. Missing days | Seattle (2 `PRCP` days), Mount Sinai NY | Lima MT (346 of 366 `TMAX` days) |
| 3. Failed quality checks | Mount Sinai NY (77 `TMAX` rows) | Dannemora NY (62), Lima MT (54) |
| 4. Station with no temperature | Atascadero rain gauge `US1CASL0033` | Atascadero second gauge `US1CASL0034` |
| 5. Station that does not exist | Atlantis `USW00000000` | El Dorado `USC00999999` |
| 6. Tenths units | every station | every station |

Clean controls: San Luis Obispo in the example list, Honolulu in the surprise list.

## Releasing it

Keep this folder out of anything students can see until the last day or two of the lab. Then either post the file in Canvas and have students upload it to their space, or publish it at a URL and give them that URL for `SURPRISE_URL`. Students change one line and rerun.

## If you rebuild it for another term

1. Pick a year and read the `TMAX` and `PRCP` Parquet slices with `ID`, `DATE`, and `Q_FLAG`.
2. Choose two or three well-known airport stations (`USW…`) with a full year as clean cases, and write some of their ids in lowercase or with a stray space.
3. Group `TMAX` by `ID` and count rows and non-empty `Q_FLAG` values. Pick one station with at least 50 flagged rows and one with fewer than 350 rows.
4. Pick one `US1…` (CoCoRaHS) station that appears in `PRCP` but not in `TMAX`.
5. Invent one id that matches nothing.
6. Run the instructor solution on the new list and save the output as the expected file.

`patio_summary_surprise_expected.csv` is the instructor solution's output for this list. NOAA revises the archive, so regenerate it shortly before grading and expect small differences from the October 5, 2026 numbers.
