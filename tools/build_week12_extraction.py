#!/usr/bin/env python3
r"""Generate the Week 12 Topic 12.2 notebook (student + executed solution): the Extract step, shown end to end.

Usage:  python3 tools/build_week12_extraction.py             # write -empty and -solution, execute the solution (needs the internet)
        python3 tools/build_week12_extraction.py --no-exec   # write both notebooks without executing

Topic 12.2 — Extraction Walkthrough: Building the Week 12 Data Files from Their Sources
  Topic 12.1 starts from two cached CSV files.  This read-along notebook shows where they came from and rebuilds
  them live, one step at a time: fetch the Wikipedia inductee page, pick the right table by its column names, save
  it with its access date; then read one nflverse season file, choose the rows and columns, loop over all 27
  seasons, check, save, and compare with the repository's cached copies.  Four kinds of question, labelled in the
  prose so the instructor can see them at a glance:
    **Explain, step by step.**   read a code cell aloud, line by line
    **Write the code.**          a one- or two-line cell the students write themselves (a prompt in the student
                                 version, the code in the solution)
    **Fill in the blank.**       ____ in code or prose
    **Discuss.**                 a judgement question with a written answer in the solution
  Needs pandas, requests and lxml; the solution reads Wikipedia and GitHub when it executes.

Markers (the same as the other builders, plus one):
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  «text»            inside markdown     -> bold answer in the solution, a ____ blank in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
  > **Instructor note.**               -> kept in the solution, dropped from the student version
  write_code(task, code)               -> a code cell: the code in the solution, "# Write the code: task" in the student version
  ⟦name⟧            anywhere            -> replaced by a value computed here (from the cached files and one live fetch)
"""

from __future__ import annotations

import copy
import io
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_12"
DATA_DIR = ROOT / "assignments" / "Data"
STEM = WEEK / "GSB5544_Topic_12_2_Extraction_Walkthrough_Building_the_NFL_Data_Files"
JUPYTER = "/opt/anaconda3/bin/jupyter"
DATA = "https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/"

HOF_URL = "https://en.wikipedia.org/wiki/List_of_Pro_Football_Hall_of_Fame_inductees"
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{season}.csv"
HOF_CSV = DATA_DIR / "hof_inductees.csv"
RB_CSV = DATA_DIR / "nfl_rb_seasons_1999_2025.csv"
ACCESS_DATE_FILE = DATA_DIR / "nfl_hof_access_date.txt"
FIRST_SEASON, LAST_SEASON = 1999, 2025


# ============================================================================ #
#  The numbers the prose quotes
# ============================================================================ #
V: dict[str, str] = {}


def compute_values() -> None:
    import requests

    hof = pd.read_csv(HOF_CSV)
    rb = pd.read_csv(RB_CSV)
    V["ACCESS_DATE"] = ACCESS_DATE_FILE.read_text().strip() if ACCESS_DATE_FILE.exists() else date.today().isoformat()
    V["HOF_ROWS"] = f"{len(hof):,}"
    V["HOF_PEOPLE"] = f"{hof['Inductee'].nunique():,}"
    V["HOF_MAX_TEAMS"] = str(int(hof["Inductee"].value_counts().max()))
    V["RB_ROWS"] = f"{len(rb):,}"
    V["N_SEASONS"] = str(rb["season"].nunique())
    per = rb.groupby("season").size()
    V["BACKS_MIN"], V["BACKS_MAX"] = str(int(per.min())), str(int(per.max()))
    fb = rb[rb["position"] == "FB"].groupby("season").size()
    V["FB_FIRST"], V["FB_LAST"] = str(int(fb.get(FIRST_SEASON, 0))), str(int(fb.get(LAST_SEASON, 0)))
    V["POSITIONS_SEEN"] = " and ".join(sorted(rb["position"].unique()))
    V["N_IDS"] = f"{rb['player_id'].nunique():,}"

    print("fetching the Wikipedia page once, to count its tables ...")
    html = requests.get(HOF_URL, headers={"User-Agent": "GSB5544-classroom-notebook/1.0"}, timeout=60).text
    tables = pd.read_html(io.StringIO(html))
    want = {"Inductee", "Class", "Position", "Team(s)", "Years"}
    hits = [i for i, t in enumerate(tables) if want.issubset(t.columns)]
    V["N_TABLES"] = str(len(tables))
    V["HITS"] = " and ".join(map(str, hits))
    V["NFL_IDX"] = str([i for i in hits if "League(s)" not in tables[i].columns][0])
    V["OTHER_IDX"] = str([i for i in hits if "League(s)" in tables[i].columns][0])
    V["OTHER_ROWS"] = str(len(tables[int(V["OTHER_IDX"])]))
    V["HTML_KB"] = f"{len(html) / 1000:,.0f}"

    print("fetching one nflverse season file once, to quote its shape ...")
    one = pd.read_csv(NFLVERSE_URL.format(season=2024), low_memory=False)
    V["ONE_ROWS"], V["ONE_COLS"] = f"{len(one):,}", str(one.shape[1])
    V["ONE_RB"] = str(int((one["position"] == "RB").sum()))
    V["ONE_FB"] = str(int((one["position"] == "FB").sum()))
    V["ONE_BACKS"] = str(int(one["position"].isin(["RB", "HB", "FB"]).sum()))


def T(text: str) -> str:
    return re.sub(r"⟦(\w+)⟧", lambda mo: V[mo.group(1)], text)


# ============================================================================ #
#  Cell helpers
# ============================================================================ #
cells: list[dict] = []
BLANK = re.compile(r"«(.+?)»", re.S)


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n")})


def write_code(task: str, text: str) -> None:
    """A cell the students write: the solution carries the code, the student version only the task."""
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"write_task": task.strip()}, "outputs": [],
                  "source": f"# Write the code: {task.strip()}\n" + text.strip("\n")})


def answer(text: str) -> None:
    md("**Answer:** " + text.strip())


def note(text: str) -> None:
    md("> **Instructor note.** " + text.strip().replace("\n", "\n> "))


# ============================================================================ #
#  Front matter
# ============================================================================ #
md(r"""
# GSB 5544 — Topic 12.2: Extraction Walkthrough — Building the Week 12 Data Files from Their Sources — SOLUTIONS
*Topic 12.1 starts from two cached CSV files. This read-along notebook shows where they came from and rebuilds them
live, one step at a time, with every question answered.*
""")

note(r"""
**Run of show (about 25 minutes, read together, projector on).** Part 0 the two sources (3 min) · Part 1 Wikipedia:
fetch, find the table, look, save, compare (10 min) · Part 2 nflverse: one season, then the loop, check, save,
compare (10 min) · Part 3 what the Extract step leaves behind (2 min).
**Four kinds of question**, labelled in bold so you can pick as you go: *Explain, step by step* (a student reads a
cell aloud line by line), *Write the code* (one or two lines; give them 60 seconds, then show the solution),
*Fill in the blank* (`____` in code or prose) and *Discuss* (judgement; the written answer is in this notebook).
**Needs the internet.** If the room's Wi-Fi fails, read from this executed solution; the outputs are the point.
**The point to land:** an extraction is five things — a source you are allowed to use, the code that reads it,
the rows and columns you keep, the saved file, and the date — and Topic 12.1 could start from one line of
`pd.read_csv` only because all five were done here first.
""")

md(r"""
## Part 0 — Where the Week 12 data came from

Topic 12.1 reads two files. Neither was typed in; each is the saved result of code that read a public source.

| File 12.1 reads | Source | What the source looks like | Permission |
|---|---|---|---|
| `hof_inductees.csv` (⟦HOF_ROWS⟧ rows) | Wikipedia, [*List of Pro Football Hall of Fame inductees*](https://en.wikipedia.org/wiki/List_of_Pro_Football_Hall_of_Fame_inductees) | one web page, about ⟦HTML_KB⟧ KB of HTML, ⟦N_TABLES⟧ tables; the NFL table has one row per inductee **and team** | text licensed CC BY-SA 4.0; identify your script in the request |
| `nfl_rb_seasons_1999_2025.csv` (⟦RB_ROWS⟧ rows) | nflverse-data on GitHub, release [`stats_player`](https://github.com/nflverse/nflverse-data/releases/tag/stats_player) | one CSV per season, `stats_player_reg_<season>.csv`, about 2,000 players × ⟦ONE_COLS⟧ columns each | data released CC BY 4.0 by the nflverse project |

The outline's first choice, Pro Football Reference, was dropped: its terms restrict automated access and the use of
its content for machine-learning methods. **Checking permission is part of the Extract step**, before any code.

An extraction is finished when five things exist: the source (a URL you may use), the «code» that reads it, the
rows and columns kept, the saved «file», and the access «date». This notebook produces all five for both files.
It writes into a folder of its own, `week12_data/`, and at the end compares what it built with the repository's
cached copies.
""")

md(r"""
**Discuss.** Topic 12.1 reads each file with one line of `pd.read_csv`. What is lost if a project *only* has that
line and the file, and not this notebook?
""")
answer(r"""
Nobody can say where the numbers came from, whether the source allowed the use, which rows were dropped on the way
in, or when it was read — so nobody can repeat it or update it. The file alone is a claim; the code plus the date
is evidence. (It is also the first thing a reader of your final project will ask: *where did this come from?*)
""")

code(f'''
import io
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

OUT = Path("week12_data")            # this notebook writes here; the repository's copies are not touched
OUT.mkdir(exist_ok=True)
CACHE = "{DATA}"   # the cached copies Topic 12.1 reads
print("writing to", OUT.resolve())
''')

# ============================================================================ #
#  Part 1 — Wikipedia
# ============================================================================ #
md(r"""
## Part 1 — Source 1: the Hall of Fame list (Wikipedia)

### 1.1 Fetch the page

A web page is text. `requests.get` asks the server for it and hands back a *response*; the HTML is in `.text`.
""")

code(r'''
HOF_URL = "https://en.wikipedia.org/wiki/List_of_Pro_Football_Hall_of_Fame_inductees"

response = requests.get(HOF_URL, headers={"User-Agent": "GSB5544-classroom-notebook/1.0"}, timeout=60)
print(response.status_code, "|", f"{len(response.text):,}", "characters of HTML")
response.text[:300]
''')

md(r"""
**Explain, step by step.** Read the `requests.get(...)` line aloud: what are the three arguments, what does the
status code tell you, and what is in `response.text`?
""")
answer(r"""
`HOF_URL` is the address to fetch. `headers={"User-Agent": ...}` names the program making the request — Wikipedia
asks automated clients to identify themselves, and a descriptive name is the polite minimum. `timeout=60` makes
the call fail after 60 seconds instead of hanging forever. A status code of `200` means "OK, here is the page"
(a `404` would mean the page does not exist, `429` that we are asking too often). `response.text` is the whole
page as one string of HTML — the first 300 characters are the `<!DOCTYPE html>` boilerplate, nothing like a table
yet.
""")

md(r"""
### 1.2 From HTML to tables

`pd.read_html` finds every `<table>` in a page and turns each into a DataFrame. It accepts a file or a file-like
object, so the string is wrapped in `io.StringIO`, which makes a string behave like an open «file».
""")

code(r'''
tables = pd.read_html(io.StringIO(response.text))
''')

write_code("print how many tables the page has, then the shape of each one", r'''
print(len(tables), "tables")
[t.shape for t in tables]
''')

md(r"""
⟦N_TABLES⟧ tables, most of them navigation boxes with a handful of rows. Two are large.
""")

md(r"""
### 1.3 Find the right table by its column names, not its position

The inductee table could be `tables[1]` today and `tables[2]` after the next edit to the page. A test on the
**column names** fails loudly if the page changes; an index fails silently.
""")

code(r'''
want = {"Inductee", "Class", "Position", "Team(s)", "Years"}
hits = [i for i, t in enumerate(tables) if «want.issubset(t.columns)»]
print("tables with those five columns:", hits)
for i in hits:
    print(i, tables[i].shape, list(tables[i].columns))
''')

md(r"""
**Explain, step by step.** Read the `hits = [...]` line aloud. What does `enumerate` give, what does `issubset`
test, and why is the result a list of numbers?
""")
answer(r"""
`enumerate(tables)` yields pairs `(i, t)`: the position of each table and the table itself. `want.issubset(t.columns)`
asks whether every name in the set `want` appears among that table's columns — a set test, so the order of the
columns does not matter and extra columns are allowed. The comprehension keeps the `i` of every table that passes,
so `hits` is the list of table numbers that *could* be the one we want: here ⟦HITS⟧.
""")

md(r"""
Two tables pass: table ⟦NFL_IDX⟧ (NFL inductees) and table ⟦OTHER_IDX⟧ (⟦OTHER_ROWS⟧ rows of pre-NFL and
other-league careers, with a `League(s)` column). The project wants the NFL table, so the extra column is the
tie-breaker.
""")

code(r'''
nfl_table = [tables[i] for i in hits if «"League(s)" not in tables[i].columns»][0]
hof = nfl_table[["Inductee", "Class", "Position", "Team(s)", "Years"]]
hof.shape
''')

md(r"""
**Discuss.** Why select the five columns by name again, when the table already has exactly those five?
""")
answer(r"""
Because it will not always. Selecting by name fixes the column order, drops anything the page adds later (a new
"Ref." column, say) and raises a `KeyError` if one of the five disappears. Topic 12.1 depends on exactly these
five names; this line is where that promise is made.
""")

md(r"""
### 1.4 Look before you save

Saving is cheap; saving the wrong thing is expensive. Look at the head, the types and the repeats.
""")

code(r'''
print(hof.dtypes, "\n")
hof.head(8)
''')

write_code("count the rows, the distinct inductees, and show the five most repeated names", r'''
print(len(hof), "rows;", hof["Inductee"].nunique(), "distinct inductees")
hof["Inductee"].value_counts().head(5)
''')

md(r"""
⟦HOF_ROWS⟧ rows but ⟦HOF_PEOPLE⟧ people: one row per inductee «and team», up to ⟦HOF_MAX_TEAMS⟧ rows for one
person. Names carry footnote marks (`Marcus Allen** [4]`), and `Position` mixes "Running back", "Halfback" and
long pre-modern labels. **None of that is fixed here.** The Extract step saves what the source said; cleaning
names, parsing years and collapsing to one row per person is «Transform» work and lives in Topic 12.1, Part 3,
where every change is visible and checkable.
""")

md(r"""
### 1.5 Save it, with the date
""")

code(r'''
hof.to_csv(OUT / "hof_inductees.csv", «index=False»)
(OUT / "nfl_hof_access_date.txt").write_text(date.today().isoformat() + "\n")
print("saved", OUT / "hof_inductees.csv", "| accessed", (OUT / "nfl_hof_access_date.txt").read_text().strip())
''')

md(r"""
**Fill in the blank (prose).** Without `index=False`, re-reading the CSV would show an extra column called
«Unnamed: 0» holding the old row numbers.
""")

write_code("re-read the saved file and check that it has the same shape as `hof`", r'''
pd.read_csv(OUT / "hof_inductees.csv").shape == hof.shape
''')

md(r"""
### 1.6 Compare with the cached copy that Topic 12.1 reads

The repository's copy was saved on ⟦ACCESS_DATE⟧. Wikipedia changes; the comparison says whether today's page
still agrees with it.
""")

code(r'''
cached_hof = pd.read_csv(CACHE + "hof_inductees.csv")
same = hof.reset_index(drop=True).astype(str).equals(cached_hof.astype(str))
print("live", hof.shape, "| cached", cached_hof.shape, "| identical:", same)
''')

md(r"""
**Discuss.** Suppose this prints `False` next August. What happened, which copy should the project use, and what
does "refresh the data" have to include?
""")
answer(r"""
A new class was inducted, or someone edited a name, a team or a footnote. The project keeps using the **cached**
copy with its recorded date, so every number in the notebook stays reproducible. Refreshing is a deliberate act:
re-run the extraction, write the new file *and* the new date, then re-run every downstream cell and re-read every
sentence that quotes a count, because some of them will have changed. (The repository does this with
`tools/build_week12_project_exploration.py --refresh-data`.)
""")

# ============================================================================ #
#  Part 2 — nflverse
# ============================================================================ #
md(r"""
## Part 2 — Source 2: season-by-season statistics (nflverse)

### 2.1 One file per season

The nflverse project publishes one CSV per season on a GitHub *release* page. The file names differ only in the
season, so one URL pattern with a `{season}` slot covers all of them.
""")

code(r'''
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{season}.csv"
print(NFLVERSE_URL.format(season=«2024»))

one = pd.read_csv(NFLVERSE_URL.format(season=2024), low_memory=False)
print(one.shape)
list(one.columns)[:30]
''')

md(r"""
**Explain, step by step.** What does `.format(season=2024)` do to the string? Why read the file with
`pd.read_csv` straight from a URL? What is `low_memory=False` for?
""")
answer(r"""
`.format` replaces the `{season}` slot with `2024`, giving the exact file name on the release page. `pd.read_csv`
accepts a URL as well as a path, so no separate download step is needed — pandas fetches the file and parses it in
one call. `low_memory=False` makes pandas read the whole file before deciding each column's type; with the default,
a large file is read in chunks and a column that looks numeric in one chunk and text in another produces a
"mixed types" warning and an `object` column.
""")

md(r"""
⟦ONE_ROWS⟧ players and ⟦ONE_COLS⟧ columns for 2024 alone — passing, rushing, receiving, defence, kicking, punting,
fantasy points. The project needs a dozen of them.
""")

write_code("show which positions appear in the 2024 file and how many players each has", r'''
one["position"].value_counts()
''')

md(r"""
### 2.2 Which rows, which columns

Two decisions, made once and written down: keep the players whose listed position is a running back, and keep
the identifying columns plus the rushing and receiving counts.
""")

code(r'''
KEEP = ["player_id", "player_display_name", "position", "recent_team", "season", "games",
        "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]

is_back = one["position"].isin(«["RB", "HB", "FB"]»)
one_rb = one.loc[is_back, KEEP]
print(one_rb.shape)
one_rb.sort_values("rushing_yards", ascending=False).head(5)
''')

md(r"""
**Explain, step by step.** Read `one_rb = one.loc[is_back, KEEP]` aloud. What is `is_back`, and what do the two
things inside the square brackets each select?
""")
answer(r"""
`is_back` is a Series of `True` / `False`, one per player, `True` when the position is one of the three labels.
`.loc[rows, columns]` takes a row selector and a column selector: the boolean Series keeps only the `True` rows
(⟦ONE_BACKS⟧ of ⟦ONE_ROWS⟧ for 2024: ⟦ONE_RB⟧ RB and ⟦ONE_FB⟧ FB), and the list `KEEP` keeps only those 13 columns,
in that order. The result is a new, smaller DataFrame; `one` is unchanged.
""")

md(r"""
**Discuss.** Three choices to defend: (a) `player_id` is kept although `player_display_name` is there; (b)
`receptions`, `targets` and `receiving_yards` are kept although Topic 12.1 only plots rushing; (c) `"HB"` is in
the list although no 2024 player has it.
""")
answer(r"""
(a) Names are not identifiers: two different players are called Adrian Peterson and two Ricky Williams. The id is
the key that Topic 12.1's crosswalk joins on; drop it here and the join can never be made right later.
(b) The project question promises "rushing *and receiving*", and keeping three columns now costs nothing, while
going back to the source for them later means redoing the extraction. Keep what the questions might need, no more.
(c) Position vocabularies change between seasons, and the list is a *rule*, not a description of one file;
writing `HB` costs nothing and the check in 2.4 shows which labels actually occurred (⟦POSITIONS_SEEN⟧, never HB).
The same rows come from `position_group == "RB"`; either is fine, as long as the notebook says which was used.
""")

md(r"""
### 2.3 All ⟦N_SEASONS⟧ seasons: the loop

The same three lines, once per season, collected in a list and stacked at the end. One line in the loop is a
**check**: if a season's file does not have the 13 columns, stop immediately and say which season and which
columns.
""")

code(r'''
FIRST_SEASON, LAST_SEASON = 1999, «2025»

frames = []
for season in range(FIRST_SEASON, «LAST_SEASON + 1»):
    d = pd.read_csv(NFLVERSE_URL.format(season=season), low_memory=False)
    missing = [c for c in KEEP if c not in d.columns]
    assert not missing, (season, missing)          # stop loudly if a season's file has changed shape
    frames.append(d.loc[d["position"].isin(["RB", "HB", "FB"]), KEEP])
    print(season, f"{len(d):,} players ->", len(frames[-1]), "backs")

rb = pd.concat(frames, «ignore_index=True»)
rb.shape
''')

md(r"""
**Explain, step by step.** Read the loop aloud, one line at a time: the `range`, the read, the `missing` list,
the `assert`, the `append`, the `print`, and the `concat` after the loop.
""")
answer(r"""
`range(1999, 2026)` produces 1999 to 2025 — the end is excluded, hence `LAST_SEASON + 1`. Each pass reads that
season's file into `d`. `missing` lists the names in `KEEP` that are *not* columns of `d`; `assert not missing`
passes when the list is empty and otherwise stops the notebook with the season and the missing names in the error,
so a changed file can never be stacked silently. `frames.append(...)` keeps that season's running-back rows and
13 columns; `print` shows the row counts as they go, which is how you notice a season with zero backs. After the
loop, `pd.concat(frames, ignore_index=True)` stacks the ⟦N_SEASONS⟧ small tables into one and renumbers the rows 0 to
⟦RB_ROWS⟧ instead of restarting at 0 for each season.
""")

md(r"""
### 2.4 Check before saving

One row per player per season is the assumption every later step rests on. Say it as an `assert`, and look at the
counts.
""")

code(r'''
assert not rb.duplicated(["player_id", "season"]).any()      # one row per player per season
print(rb["player_id"].nunique(), "distinct players |", rb["season"].nunique(), "seasons")
rb["position"].value_counts()
''')

write_code("count the backs in each season and print the smallest and largest counts", r'''
per_season = rb.groupby("season").size()
print("backs per season:", per_season.min(), "to", per_season.max())
per_season
''')

md(r"""
**Discuss.** Fullbacks fall from ⟦FB_FIRST⟧ in 1999 to ⟦FB_LAST⟧ in 2025. Does the extraction have a problem, and
what does the trend mean for Topic 12.1's 50-carry rule?
""")
answer(r"""
The extraction is fine: it kept every player the source *labels* RB or FB, and the label is a roster designation,
not a usage count. The trend is the game changing — fewer teams carry a fullback — and it is why Topic 12.1 defines
the comparison group by **usage** (at least 50 carries) rather than by label: a 2025 fullback with 6 carries and a
2025 running back with 6 carries are the same kind of row, and both drop out. The label stays in the file so the
rule can be revisited (Figure 4 in 12.1 does exactly that).
""")

md(r"""
### 2.5 Save, and compare with the cached copy
""")

code(r'''
rb.to_csv(OUT / "nfl_rb_seasons_1999_2025.csv", index=False)

cached_rb = pd.read_csv(CACHE + "nfl_rb_seasons_1999_2025.csv")
compare = pd.DataFrame({"live_rows":    rb.groupby("season").size(),
                        "cached_rows":  cached_rb.groupby("season").size(),
                        "live_yards":   rb.groupby("season")["rushing_yards"].sum(),
                        "cached_yards": cached_rb.groupby("season")["rushing_yards"].sum()})
differs = compare[(compare["live_rows"] != compare["cached_rows"]) | (compare["live_yards"] != compare["cached_yards"])]
print("live", rb.shape, "| cached", cached_rb.shape, "| seasons that differ:", len(differs))
differs
''')

md(r"""
**Discuss.** The 2026 season is being played as you read this, yet the loop stops at 2025. Why not include the
current season, and what has to change when it ends?
""")
answer(r"""
A season in progress changes every week, so a file read today and one read next Tuesday disagree, and a
percentile among *partial* seasons compares players who have played different numbers of games. Keep completed
seasons only. When 2026 ends: set `LAST_SEASON = 2026`, re-run this extraction, save a new file **with a new name**
(`..._1999_2026.csv`) and a new date, then re-run Topic 12.1 and re-read every sentence that quotes a number.
""")

# ============================================================================ #
#  Part 3 — What the Extract step leaves behind
# ============================================================================ #
md(r"""
## Part 3 — What the Extract step leaves behind
""")

code(r'''
sorted(p.name for p in OUT.iterdir())
''')

md(r"""
Three files: the two data sets and the access date. Together with this notebook they *are* the Extract step; Topic
12.1 can begin with `pd.read_csv` because they exist.

| What | Where | Why it matters |
|---|---|---|
| the source and its permission | Part 0 | someone else can judge whether the use was allowed |
| the code that read it | Parts 1 and 2 | the extraction can be repeated on purpose, not by accident |
| the rows and columns kept | 1.3 and 2.2 | the reader knows what was dropped on the way in |
| the saved files | `week12_data/` | every later cell reads the same bytes |
| the access «date» | `nfl_hof_access_date.txt` | "the page said so on 10 November" is a checkable claim |
""")

md(r"""
**Discuss (closing).** Where does E end and T begin? The filter to running backs and the 13 columns happened
*here*; stripping `**` from the names and splitting `1993–1995` into two years happens in Topic 12.1. Why draw the
line there?
""")
answer(r"""
Rule of thumb: **Extract saves what the source said; Transform changes it.** Choosing rows and columns does not
alter a single value, so it belongs with the extraction (and is documented there). Rewriting a name, parsing a
year or collapsing several rows into one *does* alter values, and every such change should be visible in the
analysis notebook where it can be checked against the raw file. If the cleaning were done here, Topic 12.1's
reader could never see what `Marcus Allen** [4]` looked like before it became `Marcus Allen`.
""")

md(r"""
---
## For your own final project — the Extract checklist

1. **Permission first.** Read the source's terms before writing code; write one sentence in the notebook saying
   what the terms allow. If the answer is no, find another source (the outline's first choice was replaced for
   exactly this reason).
2. **Identify yourself and fail fast.** A descriptive `User-Agent` and a `timeout` on every request.
3. **Find things by name, not by position.** Tables by their columns, files by a pattern, columns by a list.
4. **Look before saving.** `shape`, `dtypes`, `head`, repeats of the key.
5. **Check with `assert`.** One row per key; the columns you expect; stop loudly otherwise.
6. **Save what the source said** — no cleaning in the Extract step — with `index=False` and an access date.
7. **Compare with the cache** when you refresh, and re-read every sentence that quotes a number.
""")


# ============================================================================ #
#  Build
# ============================================================================ #
METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

STUDENT_TITLE = (
    "# GSB 5544 — Topic 12.2: Extraction Walkthrough — Building the Week 12 Data Files from Their Sources\n"
    "*Topic 12.1 starts from two cached CSV files. This read-along notebook shows where they came from and rebuilds "
    "them live, one step at a time. Four kinds of question appear in bold: **Explain, step by step** (read a cell "
    "aloud, line by line), **Write the code** (one or two lines, in the cell provided), **Fill in the blank** "
    "(`____` in code or prose) and **Discuss** (write a sentence or two). Needs the internet.*"
)


def notebook(cell_list: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cell_list):
        c = copy.deepcopy(c)
        c["id"] = f"cell-{i:03d}"
        out.append(c)
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def solution_cells(cell_list: list[dict]) -> list[dict]:
    out = []
    for c in cell_list:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub(lambda m: m.group(1), c["source"])
        else:
            c["source"] = BLANK.sub(lambda m: f"**{m.group(1)}**", c["source"])
        out.append(c)
    return out


def student_cells(cell_list: list[dict]) -> list[dict]:
    out = []
    for c in cell_list:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            if "write_task" in c["metadata"]:
                c["source"] = f"# Write the code: {c['metadata']['write_task']}\n\n"
            else:
                c["source"] = BLANK.sub("____", c["source"])
        elif c["source"].startswith("> **Instructor note.**"):
            continue
        elif c["source"].startswith("**Answer:**"):
            c["source"] = "**Your answer:** *(think, then discuss with a neighbour — write a sentence or two here)*"
        else:
            c["source"] = BLANK.sub("`____`", c["source"])
        out.append(c)
    out[0]["source"] = STUDENT_TITLE
    return out


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    """Execute the solution in place (live web reads; the cached copies are read locally), then restore the URLs."""
    nb = json.loads(path.read_text())
    local = copy.deepcopy(nb)
    for cell in local["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace(DATA, str(DATA_DIR) + "/")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        save(tmp_path, local)
        subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=900", str(tmp_path)], check=True)
        executed = json.loads(tmp_path.read_text())
    for original, ran in zip(nb["cells"], executed["cells"]):
        if original["cell_type"] == "code":
            original["outputs"] = ran.get("outputs", [])
            original["execution_count"] = ran.get("execution_count")
            for o in original["outputs"]:      # the temp folder's absolute path must not leak into the output
                if isinstance(o.get("text"), str):
                    o["text"] = o["text"].replace(tmp, "...")
                elif isinstance(o.get("text"), list):
                    o["text"] = [line.replace(tmp, "...") for line in o["text"]]
    nb["metadata"]["language_info"] = executed["metadata"].get("language_info", nb["metadata"]["language_info"])
    save(path, nb)
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c["outputs"] if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"Execution errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    compute_values()
    for c in cells:
        c["source"] = T(c["source"])
    leftovers = [c["source"][:60] for c in cells if "⟦" in c["source"]]
    assert not leftovers, leftovers
    WEEK.mkdir(parents=True, exist_ok=True)
    student = STEM.with_name(STEM.name + "-empty.ipynb")
    solution = STEM.with_name(STEM.name + "-solution.ipynb")
    save(student, notebook(student_cells(cells)))
    save(solution, notebook(solution_cells(cells)))
    n_blanks = sum(len(BLANK.findall(c["source"])) for c in cells)
    n_write = sum("write_task" in c["metadata"] for c in cells)
    n_explain = sum(c["source"].startswith("**Explain, step by step.**") for c in cells)
    n_discuss = sum(c["source"].startswith("**Discuss") for c in cells)
    print(f"wrote {student.relative_to(ROOT)} and {solution.relative_to(ROOT)} ({len(cells)} cells, "
          f"{n_blanks} blanks, {n_write} write-the-code cells, {n_explain} explain prompts, {n_discuss} discuss prompts)")
    if "--no-exec" not in sys.argv:
        execute(solution)


if __name__ == "__main__":
    main()
