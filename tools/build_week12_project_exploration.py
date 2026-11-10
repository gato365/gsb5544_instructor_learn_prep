#!/usr/bin/env python3
r"""Generate the Week 12 Topic 12.1 notebook (student + executed solution) and its two cached data files.

Usage:  python3 tools/build_week12_project_exploration.py                 # write -empty and -solution, execute the solution
        python3 tools/build_week12_project_exploration.py --no-exec       # write both notebooks without executing
        python3 tools/build_week12_project_exploration.py --refresh-data  # re-download the two cached CSVs first

Topic 12.1 — Project Exploration: NFL Running Backs and the Hall of Fame (ETV: Extract, Transform, Visualize)
  A 20-minute fill-in-and-discuss opener for the final project. The instructor's own project outline (compare Hall of
  Fame running backs with their contemporaries season by season, then look for current players with similar profiles)
  is used to show how a project idea becomes a coherent notebook:
    ETV   — every notebook step is an Extraction, a Transformation, or a Visualization; the thirteen steps of the
            outline collapse into those three stages.
    SSVA  — every project question is answered by a Summary Statistic, shown in a Visualization, and read in an
            Analysis sentence: question → statistic → figure → sentence.
  The notebook runs the first version end to end on cached, permitted data (nflverse player statistics 1999-2025 and
  the Wikipedia inductee list), with ____ blanks in the code and the prose, four figures (one per project question
  that a figure can answer), and a closing discussion on how to improve the project.
  Needs pandas and plotnine; nothing is read from the web when the two cached CSVs are present.

Data files written to assignments/Data (only when missing or with --refresh-data):
  hof_inductees.csv              the Wikipedia "List of Pro Football Hall of Fame inductees" NFL table, one row per
                                 inductee-team (text CC BY-SA 4.0; access date recorded in the notebook)
  nfl_rb_seasons_1999_2025.csv   regular-season player-season rows for RB / FB from nflverse stats_player_reg_<season>
                                 (nflverse-data releases; 13 columns kept)

Markers (the same as the earlier builders):
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  «text»            inside markdown     -> bold answer in the solution, a ____ blank in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
  > **Instructor note.**               -> kept in the solution, dropped from the student version
  ⟦name⟧            anywhere            -> replaced by a value computed here from the cached data, so the prose
                                          cannot drift from the numbers the cells print
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
STEM = WEEK / "GSB5544_Topic_12_1_Project_Exploration_NFL_Hall_of_Fame"
JUPYTER = "/opt/anaconda3/bin/jupyter"
DATA = "https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/"

HOF_URL = "https://en.wikipedia.org/wiki/List_of_Pro_Football_Hall_of_Fame_inductees"
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{season}.csv"
HOF_CSV = DATA_DIR / "hof_inductees.csv"
RB_CSV = DATA_DIR / "nfl_rb_seasons_1999_2025.csv"
FIRST_SEASON, LAST_SEASON = 1999, 2025
KEEP = ["player_id", "player_display_name", "position", "recent_team", "season", "games", "carries", "rushing_yards",
        "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
ACCESS_DATE_FILE = DATA_DIR / "nfl_hof_access_date.txt"


# ============================================================================ #
#  Data: download once, cache in assignments/Data
# ============================================================================ #
def fetch_hof() -> pd.DataFrame:
    import requests

    html = requests.get(HOF_URL, headers={"User-Agent": "GSB5544-classroom-notebook/1.0"}, timeout=60).text
    tables = pd.read_html(io.StringIO(html))
    want = {"Inductee", "Class", "Position", "Team(s)", "Years"}
    hits = [t for t in tables if want.issubset(t.columns) and "League(s)" not in t.columns]
    assert hits, "inductee table not found"
    return hits[0][["Inductee", "Class", "Position", "Team(s)", "Years"]]


def fetch_rb_seasons() -> pd.DataFrame:
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        d = pd.read_csv(NFLVERSE_URL.format(season=season), low_memory=False)
        missing = [c for c in KEEP if c not in d.columns]
        assert not missing, (season, missing)
        frames.append(d.loc[d["position"].isin(["RB", "HB", "FB"]), KEEP])
        print(f"  {season}: {len(frames[-1])} running back rows")
    rb = pd.concat(frames, ignore_index=True)
    assert not rb.duplicated(["player_id", "season"]).any()
    return rb


def ensure_data(refresh: bool) -> None:
    if refresh or not HOF_CSV.exists():
        print("downloading the Hall of Fame list ...")
        fetch_hof().to_csv(HOF_CSV, index=False)
        ACCESS_DATE_FILE.write_text(date.today().isoformat() + "\n")
    if refresh or not RB_CSV.exists():
        print("downloading nflverse player statistics ...")
        fetch_rb_seasons().to_csv(RB_CSV, index=False)
        ACCESS_DATE_FILE.write_text(date.today().isoformat() + "\n")


# ============================================================================ #
#  The numbers the prose quotes (computed exactly as the notebook computes them)
# ============================================================================ #
V: dict[str, str] = {}


def compute_values() -> None:
    hof = pd.read_csv(HOF_CSV)
    rb = pd.read_csv(RB_CSV)
    V["ACCESS_DATE"] = ACCESS_DATE_FILE.read_text().strip() if ACCESS_DATE_FILE.exists() else date.today().isoformat()
    V["HOF_ROWS"] = f"{len(hof):,}"
    V["RB_ROWS"] = f"{len(rb):,}"
    V["N_SEASONS"] = str(rb["season"].nunique())

    bad = hof["Position"].str.contains("RB", case=False, regex=False)
    V["BAD_RB_ROWS"] = str(int(bad.sum()))
    is_rb = hof["Position"].str.contains(r"\b(?:Running back|Halfback|Fullback)\b", case=False, regex=True)
    hof_rb = hof[is_rb].copy()
    hof_rb["name"] = hof_rb["Inductee"].str.replace(r"\[\d+\]|[*^§†]", "", regex=True).str.strip()
    hof_rb["first_year"] = hof_rb["Years"].str.findall(r"\d{4}").map(lambda ys: min(map(int, ys)))
    hof_rb["last_year"] = hof_rb["Years"].str.findall(r"\d{4}").map(lambda ys: max(map(int, ys)))
    people = hof_rb.groupby("name").agg(hof_class=("Class", "min"), first_year=("first_year", "min"),
                                        last_year=("last_year", "max")).reset_index()
    V["N_HOF_RB_ROWS"] = str(len(hof_rb))
    V["N_HOF_RB"] = str(len(people))
    V["N_PREMODERN"] = str(int(hof_rb.loc[hof_rb["Position"].str.contains("Pre-Modern"), "name"].nunique()))
    overlap = people[people["last_year"] >= FIRST_SEASON]
    V["N_OVERLAP"] = str(len(overlap))
    V["N_COMPLETE"] = str(int((overlap["first_year"] >= FIRST_SEASON).sum()))
    V["N_PARTIAL"] = str(int((overlap["first_year"] < FIRST_SEASON).sum()))
    V["OVERLAP_NAMES"] = ", ".join(overlap.sort_values("first_year")["name"])
    V["COMPLETE_NAMES"] = " and ".join(overlap[overlap["first_year"] >= FIRST_SEASON].sort_values("first_year")["name"])

    names = rb.groupby("player_display_name")["player_id"].nunique()
    xwalk = overlap.merge(rb[["player_id", "player_display_name"]].drop_duplicates(), left_on="name",
                          right_on="player_display_name", how="left")
    assert xwalk["player_id"].notna().all(), xwalk[xwalk["player_id"].isna()]
    assert not xwalk.duplicated("name").any()
    V["AP_IDS"] = str(int(names.get("Adrian Peterson", 0)))

    rb["hof"] = rb["player_id"].isin(xwalk["player_id"])
    q = rb[rb["carries"] >= 50].copy()
    grp = q.groupby("season")["rushing_yards"]
    q["season_median"] = grp.transform("median")
    q["pct"] = 100 * grp.rank(method="max", pct=True)
    V["N_QUALIFIED"] = f"{len(q):,}"
    V["PEERS_MIN"] = str(int(q.groupby("season").size().min()))
    V["PEERS_MAX"] = str(int(q.groupby("season").size().max()))
    s06 = q[q["season"] == 2006]
    lt = s06[s06["player_display_name"] == "LaDainian Tomlinson"].iloc[0]
    V["N_2006"] = str(len(s06))
    V["MED_2006"] = f"{s06['season_median'].iloc[0]:,.0f}"
    V["LT_YDS"] = f"{lt['rushing_yards']:,.0f}"
    V["LT_ABOVE"] = f"{lt['rushing_yards'] - lt['season_median']:,.0f}"
    V["LT_PCT"] = ordinal(lt["pct"])
    V["LT_RATIO"] = f"{lt['rushing_yards'] / lt['season_median']:.1f}"

    career = q.groupby("player_id").agg(
        player=("player_display_name", "first"), hof=("hof", "first"), n_qualified=("season", "size"),
        median_pct=("pct", "median"), max_pct=("pct", "max"), share_90=("pct", lambda s: (s >= 90).mean()),
    ).reset_index()
    last = rb.groupby("player_id")["season"].max().rename("last_season")
    career = career.merge(last, on="player_id")
    hof_c = career[career["hof"] & (career["n_qualified"] >= 5)]        # the Hall of Famers Figure 3 shows
    V["N_HOF_FIVE"] = str(len(hof_c))
    V["HOF_MEDIAN_PCT"] = ordinal(hof_c["median_pct"].median())
    V["HOF_SHARE90"] = f"{100 * hof_c['share_90'].median():.0f}"
    others = career[~career["hof"] & (career["n_qualified"] >= 5)]
    V["OTHERS_MEDIAN_PCT"] = ordinal(others["median_pct"].median())
    V["OTHERS_SHARE90"] = f"{100 * others['share_90'].median():.0f}"
    V["N_FIVE_PLUS"] = str(int((career["n_qualified"] >= 5).sum()))
    V["N_HOF_CAREER"] = str(int(career["hof"].sum()))
    no_row = set(xwalk["name"]) - set(career.loc[career["hof"], "player"])
    V["HOF_NO_ROW"] = ", ".join(sorted(no_row)) if no_row else "none"
    for who, key in [("Edgerrin James", "EJ"), ("LaDainian Tomlinson", "LT"), ("Frank Gore", "FG"),
                     ("Derrick Henry", "DH"), ("Adrian Peterson", "AP"), ("Christian McCaffrey", "CM")]:
        rows = career[career["player"] == who]
        if who == "Adrian Peterson":
            rows = rows.sort_values("n_qualified", ascending=False).head(1)
        r = rows.iloc[0]
        V[f"{key}_N"] = str(int(r["n_qualified"]))
        V[f"{key}_MED"] = ordinal(r["median_pct"])
        V[f"{key}_S90"] = f"{100 * r['share_90']:.0f}"
        V[f"{key}_LAST"] = str(int(r["last_season"]))

    sens = []
    for thr in (25, 50, 100):
        qq = rb[rb["carries"] >= thr].copy()
        qq["pct"] = 100 * qq.groupby("season")["rushing_yards"].rank(method="max", pct=True)
        c = qq.groupby("player_id").agg(hof=("hof", "first"), median_pct=("pct", "median"),
                                        share_90=("pct", lambda s: (s >= 90).mean()))
        sens.append((thr, c[c["hof"]]["median_pct"].median(), 100 * c[c["hof"]]["share_90"].median(),
                     int(qq.groupby("season").size().median()), c[~c["hof"]]["median_pct"].median()))
    for thr, med, s90, peers, oth in sens:
        V[f"SENS_MED_{thr}"], V[f"SENS_S90_{thr}"], V[f"SENS_PEERS_{thr}"] = ordinal(med), f"{s90:.0f}", str(peers)
        V[f"SENS_OTH_{thr}"] = ordinal(oth)


def ordinal(x: float) -> str:
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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


def answer(text: str) -> None:
    md("**Answer:** " + text.strip())


def note(text: str) -> None:
    md("> **Instructor note.** " + text.strip().replace("\n", "\n> "))


# ============================================================================ #
#  Front matter
# ============================================================================ #
md(r"""
# GSB 5544 — Topic 12.1: Project Exploration — NFL Running Backs and the Hall of Fame — SOLUTIONS
*The 20-minute opener with every blank filled, the answers to the discussion prompts, and instructor notes.*
""")

note(r"""
**Run of show (20 minutes).** Part 0 the idea and the five questions (2 min) · Part 1 ETV and SSVA, the two maps
(3 min) · Part 2 Extract (3 min) · Part 3 Transform (5 min) · Part 4 Visualize, one figure per question (5 min) ·
Part 5 how to improve the project (2 min). For 15 minutes skip 3.2 and Figure 4 and give Part 5 as homework.
**The point to land:** a project is coherent when every cell is an E, a T or a V, and every figure answers a
numbered question with a statistic you can name. Everything else is cut.
**Why this project:** it is my own final-project idea at the outline stage, so the class sees the planning
document and the first running version side by side, including what the data cannot answer yet.
""")

md(r"""
## Part 0 — The idea in one paragraph

We want to know how **Hall of Fame running backs** compared with the *other* running backs who played at the same
time, and then use those comparisons to find **current and recently retired** players whose statistical profiles
look like a Hall of Famer's.

**The five project questions**

1. Which running backs, halfbacks and fullbacks are in the Pro Football Hall of Fame?
2. How did each Hall of Famer's rushing (and receiving) statistics compare with other running backs **in each season
   they played**?
3. What patterns show up across Hall of Fame careers: exceptional **peak** seasons, **sustained** performance, or both?
4. Which active or recently retired players show similar patterns at the same stage of their careers?
5. How much do the answers change with different statistics or a different **comparison group**?

**Key idea.** Compare a player with the other running backs in the same «season» *first*. A 1,200-yard season in a
year when the median qualified back gained 800 yards is 400 yards, or 50 %, above that season's median, and we can
also report the player's «percentile» among the qualifying backs that year. This makes comparisons more informative
across periods with different schedules and offensive styles. It does **not** remove every difference between eras.
""")

md(r"""
## Part 1 — Two maps that make a project coherent

### ETV — every step is an Extraction, a Transformation or a Visualization

The project outline had thirteen steps. Label each one. (E = extract data from a source · T = transform it: clean,
join, filter, compute · V = visualize or tabulate for a reader.)

| # | Outline step | Stage |
|---|---|:---:|
| 1 | Set up the notebook: packages, data sources, analysis settings | — |
| 2 | Read the Wikipedia list of inductees and identify the running backs | «E» |
| 3 | Obtain player-season rushing and receiving statistics (permitted source) | «E» |
| 4 | Standardize columns and positions, resolve team changes, one row per player-season | «T» |
| 5 | Match each Hall of Famer to the correct player record (crosswalk) | «T» |
| 6 | Define each season's comparison group (≥ 50 carries) | «T» |
| 7 | Difference from the season median and percentile among contemporaries | «T» |
| 8 | Summarize Hall of Fame careers: typical, peak, longevity | «T» |
| 9 | Label players as inducted, active, retired-not-eligible, eligible | «T» |
| 10 | Compare active players with Hall of Famers at the same career stage | «T» |
| 11 | Tables and figures: within a season, across a career, current vs historical | «V» |
| 12 | Data checks, sensitivity checks, retrospective evaluation | «T» (and V) |
| 13 | Save results, state limitations, document next steps | — |

Three stages hide inside thirteen steps: **two** extractions, **eight** transformations, **one** visualization step
that should really be spread across the notebook, one figure per question. Steps 1 and 13 are bookkeeping.

### SSVA — every question is answered by a Summary Statistic, a Visualization, and an Analysis sentence

| Question | Summary statistic (the number) | Visualization (the figure) | Analysis (the sentence) |
|---|---|---|---|
| Q1 Who is in the Hall? | a count: ⟦N_HOF_RB⟧ running backs, halfbacks and fullbacks | a table, not a chart | *N* Hall of Famers; *n* of them played in the seasons we can observe |
| Q2 Player vs that season | season «median» and the player's «percentile» | «histogram» of one season's rushing yards with the median and the player marked | "In 2006 Tomlinson gained ⟦LT_YDS⟧ yards, ⟦LT_ABOVE⟧ above the median" |
| Q3 Peak or sustained? | median season percentile; **share** of seasons at or above the 90th | «line» plot of percentile by career year; scatter of median vs share ≥ 90 | "Hall of Famers sit at about the ⟦HOF_MEDIAN_PCT⟧ percentile in a typical season" |
| Q4 Who looks similar now? | the same two numbers for active players | the same scatter, active players highlighted | "Henry's profile sits inside the Hall of Fame cluster" |
| Q5 Does the answer move? | the same numbers under 25 / 50 / 100 carries | a sensitivity «table» | "The conclusion is stable" or "it depends on the threshold" |

A figure that does not sit in a row of this table does not belong in the notebook.
""")

note(r"""
Ask the room to fill the ETV column before revealing it; the usual surprise is how much of a project is T. Then ask
for the SSVA row of *their* first project question — most students have a question and a dataset but not yet the
statistic, and this table forces the statistic to be named before any chart is drawn.
""")

# ============================================================================ #
#  Part 2 — Extract
# ============================================================================ #
md(r"""
## Part 2 — E: Extract

Two sources, both permitted for this use, both **cached** so the notebook runs the same way every time:

| Need | Source | Cached file | What to watch |
|---|---|---|---|
| Hall of Fame names, positions, classes | Wikipedia, *List of Pro Football Hall of Fame inductees* (text CC BY-SA 4.0), accessed ⟦ACCESS_DATE⟧ | `hof_inductees.csv` — ⟦HOF_ROWS⟧ rows, one per inductee **and team** | repeated rows per person, mixed position labels, footnote marks in names |
| Season-by-season statistics | nflverse `stats_player_reg_<season>` releases, accessed ⟦ACCESS_DATE⟧ | `nfl_rb_seasons_1999_2025.csv` — ⟦RB_ROWS⟧ RB / FB player-seasons, ⟦N_SEASONS⟧ seasons | coverage starts in «1999»; a career that began earlier is only **partly** observed |

The outline's first choice, Pro Football Reference, restricts automated access and the use of its content for
machine-learning methods, so the first version uses nflverse and says so. Cache the extraction once (the code that
produced each file is in `tools/build_week12_project_exploration.py`), record the access date, and read the cache
in the notebook.
""")

code(f'''
import warnings
import pandas as pd
import numpy as np
from plotnine import *

warnings.filterwarnings("ignore")
DATA = "{DATA}"

LAST_COMPLETED_SEASON = 2025   # the 2026 season is in progress; compare completed seasons only
MIN_CARRIES = «50»               # starting assumption for "a meaningful rushing role"; tested in Part 4

hof_raw = pd.read_csv(DATA + "hof_inductees.csv")
rb_raw  = pd.read_csv(DATA + "nfl_rb_seasons_1999_2025.csv")
print(hof_raw.shape, rb_raw.shape)
hof_raw.head()
''')

md(r"""
The live extraction was one line, `pd.read_html(HOF_URL)`, followed by finding the right table **by its column
names** rather than by its index (the page has fifteen tables). The statistics file was built by reading
`stats_player_reg_<season>.csv` for each season from 1999 to 2025 and keeping the rows whose position is RB, HB or
FB.
""")

code(r'''
rb_raw.head()
''')

code(r'''
# One row per player per season?  The whole analysis assumes it.
assert not rb_raw.duplicated(["player_id", "season"]).any()
rb_raw.groupby("season")["player_id"].nunique().describe().round(0)
''')

md(r"""
**Discuss.** Why cache an extraction instead of scraping every time the notebook runs?
""")
answer(r"""
Reproducibility (the same rows every run, even after the website changes), respect for the source's access terms,
speed in class, and an auditable record: the file plus its access date *is* the extraction step. The code that
built the cache stays in the repository so the extraction can be repeated deliberately, not accidentally.
""")

# ============================================================================ #
#  Part 3 — Transform
# ============================================================================ #
md(r"""
## Part 3 — T: Transform

### 3.1 Q1 — Which inductees are running backs?  (clean the Hall of Fame list)

Three problems in the raw table: the position labels are mixed ("Halfback Pre-Modern Era: Two-Way Performer",
"Fullback", "Running back"); names carry footnote marks (`Marcus Allen** [4]`); and a person has one row **per
team**. First the trap: a careless filter.
""")

code(r'''
careless = hof_raw[hof_raw["Position"].str.contains("RB", case=False)]
careless["Position"].value_counts()
''')

md(r"""
`"RB"` with `case=False` matches ⟦BAD_RB_ROWS⟧ rows, almost all of them «Cornerback» and «Quarterback», because
both contain the letters *r-b*. Use the full words with word boundaries instead.
""")

code(r'''
is_rb = hof_raw["Position"].str.contains(«r"\b(?:Running back|Halfback|Fullback)\b"», case=False, regex=True)
hof_rb = hof_raw[is_rb].copy()

# strip footnote marks from the names:  "Marcus Allen** [4]" -> "Marcus Allen"
hof_rb["name"] = hof_rb["Inductee"].str.replace(«r"\[\d+\]|[*^§†]"», "", regex=True).str.strip()

# first and last playing year from strings like "1993–1995", "2005, 2008", "1991"
years = hof_rb["Years"].str.findall(r"\d{4}")
hof_rb["first_year"] = years.map(lambda ys: min(map(int, ys)))
hof_rb["last_year"]  = years.map(lambda ys: max(map(int, ys)))

# one row per PERSON: earliest first year, latest last year, induction class
hof_people = (hof_rb.groupby(«"name"»)
                    .agg(hof_class=("Class", "min"), first_year=("first_year", "min"), last_year=("last_year", "max"),
                         position=("Position", "first"))
                    .reset_index()
                    .sort_values("first_year"))
print(len(hof_rb), "rows ->", len(hof_people), "people")
hof_people.tail(12)
''')

md(r"""
**Q1 answered:** ⟦N_HOF_RB⟧ Hall of Famers are listed at running back, halfback or fullback (⟦N_PREMODERN⟧ of them as
pre-modern two-way players). Only **⟦N_OVERLAP⟧** of them played in a season our statistics cover (1999 or later):
⟦OVERLAP_NAMES⟧. And only **⟦N_COMPLETE⟧** of those — ⟦COMPLETE_NAMES⟧ — have their *whole* career inside the data.
Keep that number in mind for Part 5.
""")

md(r"""
### 3.2 Crosswalk — connect each Hall of Famer to a `player_id`

Names suggest matches; they are not identifiers. Check: how many distinct `player_id` values does the name
*Adrian Peterson* have in the statistics?
""")

code(r'''
ids_per_name = rb_raw.groupby("player_display_name")["player_id"].nunique()
print(ids_per_name.sort_values(ascending=False).head(5))

# match the Hall of Famers whose careers reach 1999, by display name, then CHECK the playing years
observable = hof_people[hof_people["last_year"] >= 1999]
xwalk = (observable.merge(rb_raw.groupby(["player_id", "player_display_name"])["season"].agg(["min", "max"]).reset_index(),
                          left_on="name", right_on=«"player_display_name"», how="left")
                   .rename(columns={"min": "first_observed", "max": "last_observed"}))
xwalk["coverage"] = np.where(xwalk["first_year"] >= 1999, "complete", "partial (pre-1999 seasons missing)")
xwalk[["name", "hof_class", "first_year", "last_year", "player_id", "first_observed", "last_observed", "coverage"]]
''')

md(r"""
Two players share the name Adrian Peterson (⟦AP_IDS⟧ ids: the Bears back of the 2000s and the Vikings back), and two
share Ricky Williams. A join on `name` alone would have merged two careers into one. Every match above was checked against the playing years, and
each Hall of Famer whose career started before 1999 is labelled **partial** rather than silently treated as a short
career.
""")

code(r'''
rb = rb_raw.merge(xwalk[["player_id", "hof_class", "coverage"]], on="player_id", how=«"left"»)
rb["hof"] = rb["hof_class"].notna()
rb["career_year"] = rb["season"] - rb.groupby("player_id")["season"].transform(«"min"») + 1
print(rb["hof"].sum(), "Hall of Fame player-seasons out of", len(rb))
''')

md(r"""
### 3.3 Q2 — Each season's comparison group, median and percentile

A qualified season has at least «50» carries. Within each season, the **median** rushing yards of the qualified backs
is the reference point; a player's **percentile** is the share of qualified backs at or below their total.
""")

code(r'''
qualified = rb.loc[(rb["season"] <= LAST_COMPLETED_SEASON) & (rb["carries"] >= «MIN_CARRIES»)].copy()

by_season = qualified.groupby("season")["rushing_yards"]
qualified["season_median"]     = by_season.transform(«"median"»)
qualified["yards_above_median"] = qualified["rushing_yards"] - qualified["season_median"]
qualified["pct"]               = 100 * by_season.rank(method="max", «pct=True»)   # % of peers at or below, ties included

peers = qualified.groupby("season").size()
print(f"{len(qualified):,} qualified player-seasons; {peers.min()}–{peers.max()} peers per season")
qualified.loc[qualified["player_display_name"] == "LaDainian Tomlinson",
              ["season", "career_year", "carries", "rushing_yards", "season_median", "yards_above_median", "pct"]].round(0)
''')

md(r"""
**Q2 for one player-season:** in 2006 Tomlinson gained ⟦LT_YDS⟧ rushing yards when the median of the ⟦N_2006⟧
qualified backs was ⟦MED_2006⟧ — ⟦LT_ABOVE⟧ yards above the median, ⟦LT_RATIO⟧ times the median, the
«⟦LT_PCT⟧» percentile.
""")

md(r"""
### 3.4 Q3 — Career summaries: typical season, best season, how often exceptional

One row per player. The two numbers the whole project rests on are the **median** season percentile (sustained) and
the **share** of qualified seasons at or above the 90th percentile (peak).
""")

code(r'''
career = (qualified.groupby("player_id")
                   .agg(player=("player_display_name", "first"), hof=("hof", "first"), coverage=("coverage", "first"),
                        n_qualified=("season", "size"), first_q=("season", "min"), last_q=("season", "max"),
                        median_pct=("pct", «"median"»), max_pct=("pct", "max"),
                        share_90=("pct", lambda s: «(s >= 90).mean()»),
                        rush_yards=("rushing_yards", "sum"))
                   .reset_index())
career["status"] = np.select([career["hof"], career["last_q"] == LAST_COMPLETED_SEASON],
                             ["Hall of Fame", "active (played 2025)"], "not seen in 2025")
career.loc[career["hof"]].sort_values("median_pct", ascending=False).round(1)
''')

md(r"""
Only ⟦N_HOF_CAREER⟧ of the ⟦N_OVERLAP⟧ observable Hall of Famers have a row: ⟦HOF_NO_ROW⟧ never reached 50 carries in
an observed season. A percentile summary of a *partial* career describes its tail (Davis: three seasons after the
knee injury), not the player — which is why `coverage` travels with every row.

Status is **not** retirement: a player absent from 2025 may be injured, unsigned or retired, and a retired player who
is eligible but not inducted is not a confirmed "no". The outline's step 9 (a roster or retirement source) is still
to do; until then the label says exactly what the data say.
""")

# ============================================================================ #
#  Part 4 — Visualize
# ============================================================================ #
md(r"""
## Part 4 — V: Visualize, one figure per question

Colors are fixed by **entity**, never by rank: Hall of Famers orange, active players blue, everyone else gray. The
caption of each figure names the question it answers and the statistic it shows.
""")

md(r"""
### Figure 1 (Q2) — One season: where did Tomlinson sit among the 2006 backs?
*Statistic: the season median and one player's percentile. Form: a histogram of the ⟦N_2006⟧ qualified backs' rushing yards.*
""")

code(r'''
HOF, ACTIVE, OTHER = "#eb6834", "#2a78d6", "#9a9892"

s06 = qualified[qualified["season"] == «2006»]
lt  = s06[s06["player_display_name"] == "LaDainian Tomlinson"].iloc[0]
med = s06["season_median"].iloc[0]
top = np.histogram(s06["rushing_yards"], bins=np.arange(0, s06["rushing_yards"].max() + 100, 100))[0].max()

fig1 = (ggplot(s06, aes(x="rushing_yards"))
        + geom_histogram(binwidth=100, boundary=0, fill=OTHER, color="white")
        + geom_vline(xintercept=«med», linetype="dashed")
        + annotate("text", x=med + 30, y=top, label=f"season median {med:,.0f}", ha="left", size=9)
        + geom_vline(xintercept=lt["rushing_yards"], color=HOF, size=1.2)
        + annotate("text", x=lt["rushing_yards"] - 30, y=top, label=f"Tomlinson {lt['rushing_yards']:,.0f}  ({lt['pct']:.0f}th pct)",
                   ha="right", color=HOF, size=9)
        + labs(x="Rushing yards, 2006 regular season", y="Qualified running backs",
               title=f"2006: {len(s06)} backs with at least {MIN_CARRIES} carries")
        + theme_minimal() + theme(figure_size=(8, 3.8)))
fig1
''')

md(r"""
**Interpret Figure 1.** Write the analysis sentence this figure supports: name the statistic, the comparison and
the conclusion. What does the dashed line add that the orange line alone would not? Is this a *peak* or a
*sustained* claim?
""")
answer(r"""
⟦LT_YDS⟧ yards against a median of ⟦MED_2006⟧ — ⟦LT_RATIO⟧ times the typical qualified back — is an exceptional
season, and the figure shows it without a league-wide ranking: the comparison is **within the season**. The dashed
median line is the reference point; without it the orange line is just a large number, with it the reader sees
most of the ⟦N_2006⟧ backs piled far to its left. One season is a **peak** claim only; sustained needs Figure 2.
""")

md(r"""
### Figure 2 (Q3) — Careers: percentile by career year, one line per player
*Statistic: the season percentile each year. Form: «line» plot, career year on the horizontal axis, one line per player.*

**You choose the players.** Career year 1 must be real. For the Hall of Famers, Wikipedia's playing years say
whether it is (`coverage == "complete"`); for everyone else the only safe rule is a first season of 2000 or later
(a 1999 start could be a career already in progress). The table below lists the candidates: pick the two complete
Hall of Fame careers plus **two players of your own** — one retired and one active makes the clearest picture —
and make sure your neighbour picks different ones.
""")

code(r'''
first_seen = rb.groupby("player_id")["season"].min().rename("first_seen")
candidates = (career.merge(first_seen, on="player_id")
                    .query("(first_seen >= 2000 or coverage == 'complete') and n_qualified >= 5")
                    .sort_values("median_pct", ascending=False))
print(len(candidates), "players with a real career year 1 and at least five qualified seasons")
candidates[["player", "status", "n_qualified", "first_q", "last_q", "median_pct", "share_90"]].head(25).round(1)
''')

code(r'''
show = «["Edgerrin James", "LaDainian Tomlinson", "Frank Gore", "Derrick Henry"]»   # two Hall of Famers + two of your choice

paths = qualified[qualified["player_display_name"].isin(show) & (qualified["coverage"] != "partial (pre-1999 seasons missing)")].copy()
paths["who"] = np.select([paths["hof"], paths["season"] == LAST_COMPLETED_SEASON], ["Hall of Fame", "active"], "not seen in 2025")
paths["who"] = paths.groupby("player_id")["who"].transform("last")
ends = paths.sort_values("career_year").groupby("player_id").tail(1)

fig2 = (ggplot(paths, aes(x="career_year", y="pct", group="player_id", color="who"))
        + geom_hline(yintercept=90, linetype="dotted")
        + geom_line(size=1) + geom_point(size=2)
        + geom_text(ends, aes(label="player_display_name"), ha="left", nudge_x=0.3, size=9, show_legend=False)
        + scale_color_manual(values={"Hall of Fame": HOF, "active": ACTIVE, "not seen in 2025": "#52514e"}, name="")
        + scale_x_continuous(breaks=range(1, 18), limits=(1, 19))
        + labs(x="Career year (qualified seasons only)", y="Rushing-yard percentile among that season's backs",
               color="", title="Season percentile by career year: " + ", ".join(show))
        + theme_minimal() + theme(figure_size=(9, 4.2), legend_position="top"))
fig2
''')

md(r"""
**Interpret Figure 2.** For the players *you* chose: which lines make a **peak** argument (years above the dotted
90th line) and which a **sustained** one (many years near it)? Where does each line start, and what does a gap in a
line mean? Write the analysis sentence, naming your players.
""")
answer(r"""
For the four in the solution: both complete Hall of Fame careers open at or near the top and stay above the median
for a decade; the dotted line at the 90th percentile shows the difference between a **peak** (a year above it) and
**sustained** performance (many years near it). Gore's line is the length argument; Henry's is the height argument.
Seasons below 50 carries are not plotted — a gap, not a zero (an injury year or a reduced role). A line that starts
high and falls is a different shape from one that climbs; both can be Hall of Fame shapes, which is why Figure 3
needs two numbers, not one.
""")

note(r"""
Give the room two minutes to pick, then collect three or four groups' figures on the projector and ask what changed
and what did not: the Hall of Fame lines are the same every time, the comparison lines are not, and the sentence
has to be rewritten for each choice — that is the SSVA point. Common picks: McCaffrey, Barkley, Taylor, Kamara,
Elliott, Peterson. If someone picks Adrian Peterson, two lines share the label (two ids for one name; the group is
by id so they do not merge) — a live demonstration of the crosswalk problem from 3.2.
Say the caveat aloud: of the ⟦N_OVERLAP⟧ observable Hall of Famers, only James and Tomlinson have career year 1 in
the data. The partial careers (Smith, Faulk, Bettis, Martin, Davis, Thomas) are excluded from this figure on purpose;
their "career year 1" would be 1999, which is false.
""")

md(r"""
### Figure 3 (Q3 and Q4) — Sustained vs peak for every back with at least five qualified seasons
*Statistics: median season percentile (x) and share of seasons at or above the 90th percentile (y). Form: scatter;
Hall of Famers and active players highlighted and labelled.*
""")

code(r'''
five = career[career["n_qualified"] >= «5»].copy()
five["coverage"] = five["coverage"].fillna("complete")
# selective direct labels: the Hall of Famers and four comparison players; names placed left or right to avoid collisions
named = five[five["hof"] | five["player"].isin(["Derrick Henry", "Christian McCaffrey", "Frank Gore", "Adrian Peterson"])]
to_left = named[named["player"].isin(["Adrian Peterson", "Curtis Martin", "Edgerrin James", "Jerome Bettis", "Derrick Henry"])]
to_right = named.drop(to_left.index)

fig3 = (ggplot(five, aes(x="median_pct", y="share_90"))
        + geom_point(five[five["status"] == "not seen in 2025"], color=OTHER, alpha=0.6, size=2)
        + geom_point(five[five["status"] != "not seen in 2025"], aes(color="status", shape="coverage"), size=3)
        + geom_text(to_right, aes(label="player"), size=7, ha="left", nudge_x=1, color="#0b0b0b")
        + geom_text(to_left, aes(label="player"), size=7, ha="right", nudge_x=-1, color="#0b0b0b")
        + scale_color_manual(values={"Hall of Fame": HOF, "active (played 2025)": ACTIVE})
        + scale_shape_manual(values={"complete": "o", "partial (pre-1999 seasons missing)": "^"})
        + scale_y_continuous(labels=lambda l: [f"{100*v:.0f}%" for v in l])
        + labs(x="Median season percentile (sustained)", y="Share of seasons at or above the 90th percentile (peak)",
               color="", shape="", title=f"{len(five)} running backs with at least five qualified seasons, 1999–2025")
        + theme_minimal() + theme(figure_size=(9, 5.2), legend_position="top"))
fig3
''')

md(r"""
**Interpret Figure 3.** Where do the orange points sit, and what do the two axes say about them? What are the
triangles, and why are two of them low? Then Q4: which blue points sit inside the Hall of Fame cluster, and what
exactly is being claimed about them?
""")
answer(r"""
The ⟦N_HOF_FIVE⟧ Hall of Famers with five or more qualified seasons sit toward the upper
right: a median season around the **⟦HOF_MEDIAN_PCT⟧** percentile and a median of **⟦HOF_SHARE90⟧ %** of seasons at
or above the 90th, against the ⟦OTHERS_MEDIAN_PCT⟧ percentile and ⟦OTHERS_SHARE90⟧ % for the other backs with five
or more qualified seasons (most backs never have a 90th-percentile season). The triangles are partial careers:
Bettis and Smith are plotted from their *last* seasons only, which is why they sit low. **Q4:** Henry (⟦DH_N⟧
qualified seasons, median ⟦DH_MED⟧, ⟦DH_S90⟧ % at or above the 90th) and McCaffrey (⟦CM_N⟧, ⟦CM_MED⟧, ⟦CM_S90⟧ %)
sit with the complete Hall of Fame careers; Gore (⟦FG_N⟧ seasons, median ⟦FG_MED⟧) makes the longevity case from a
lower peak. These are **statistical resemblances, not probabilities of induction**.
""")

code(r'''
cols = ["player", "status", "coverage", "n_qualified", "first_q", "last_q", "median_pct", "max_pct", "share_90", "rush_yards"]
(five[five["status"] != "Hall of Fame"]
     .sort_values(«"median_pct"», ascending=False)
     .head(12)[cols].round(1))
''')

md(r"""
The candidate table sorts on **one** clearly stated measure and shows the others beside it rather than hiding them
in a single score.
""")

md(r"""
### Figure 4 (Q5) — Does the answer move with the carry threshold?
*Statistic: the Hall of Fame group's median percentile and share ≥ 90 under 25, 50 and 100 carries. Form: a table.*
""")

code(r'''
rows = []
for thr in «(25, 50, 100)»:
    q = rb[(rb["season"] <= LAST_COMPLETED_SEASON) & (rb["carries"] >= thr)].copy()
    q["pct"] = 100 * q.groupby("season")["rushing_yards"].rank(method="max", pct=True)
    c = q.groupby("player_id").agg(hof=("hof", "first"), median_pct=("pct", "median"),
                                   share_90=("pct", lambda s: (s >= 90).mean()))
    rows.append({"min_carries": thr, "peers_per_season": int(q.groupby("season").size().median()),
                 "hof_median_pct": c[c["hof"]]["median_pct"].median(),
                 "hof_share_90": c[c["hof"]]["share_90"].median(),
                 "others_median_pct": c[~c["hof"]]["median_pct"].median()})
pd.DataFrame(rows).round(2)
''')

md(r"""
**Interpret Figure 4 (the table).** Read down each column: what happens to the peer group, to the Hall of Fame
median, and to the other backs' median as the threshold rises? Why does one move and the other not? Then say in
one sentence whether the Q3 conclusion is stable.
""")
answer(r"""
Raising the threshold shrinks the peer group (⟦SENS_PEERS_25⟧ → ⟦SENS_PEERS_50⟧ →
⟦SENS_PEERS_100⟧ backs per season). The Hall of Fame median falls from the ⟦SENS_MED_25⟧ to the ⟦SENS_MED_100⟧
percentile, because the seasons that leave the pool were below theirs; the other backs' median barely moves
(⟦SENS_OTH_25⟧ to ⟦SENS_OTH_100⟧) because the seasons that leave were mostly their own. The *ordering* is what to
check: if the same players stay on top under all three rules, the conclusion is stable; the exact percentile never was.
""")

# ============================================================================ #
#  Part 5 — Improve the project
# ============================================================================ #
md(r"""
## Part 5 — How do we improve this project?

Work through these in pairs. Each one is a gap in the first version that the SSVA table or the ETV map makes visible.
""")

md(r"""
**5.1 Coverage.** Only ⟦N_COMPLETE⟧ of ⟦N_HOF_RB⟧ Hall of Fame running backs have complete careers in the data. What does
that do to Q3 and Q4, and what are the two honest ways forward?
""")
answer(r"""
Q3 ("what patterns show up across Hall of Fame careers") is being answered from two complete careers and six partial
ones, which is a description of two players, not a pattern. Either (a) obtain a licensed historical source with
stable player ids back to the 1930s, repeat the Extract step and keep nflverse as a check for the overlap years, or
(b) restate the question as a *modern-era* question (1999 onward) and compare active players with the modern cohort
of backs who have been eligible long enough to be judged — which is Q4 with an honest label. Either is fine; mixing
the two silently is not.
""")

md(r"""
**5.2 The comparison group.** The 50-carry rule defines "a running back". Who disappears, and which project question
suffers?
""")
answer(r"""
Receiving specialists, blocking fullbacks and injured players drop out, so a receiving-oriented back's best seasons
never get a percentile and Q2 is answered for rushing only. Fix: a second peer group with its own rule (for example
at least 30 targets) and a receiving-yards percentile, then a scrimmage-yards percentile that uses both. Record the
qualifying rule beside every metric, exactly as Figure 4 does for carries.
""")

md(r"""
**5.3 Status.** "Not seen in 2025" is doing a lot of work. What is wrong with it, and what source fixes it?
""")
answer(r"""
It mixes retired, injured, unsigned and briefly-benched players, and it cannot tell "eligible but not inducted"
from "not yet eligible" (five full seasons of retirement, plus the Hall's other requirements). The outline's step 9
needs an explicit roster or transactions source with a retirement date per player, and the Hall's current rules,
before any player is labelled a candidate — and even then an eligible player who was not inducted is not a permanent
"no".
""")

md(r"""
**5.4 What the statistics cannot see.** Name two things voters weigh that no column in `rb` contains. Could they be
extracted?
""")
answer(r"""
Awards and honors (All-Pro, MVP, rushing titles), postseason performance, team success and narrative (a Super Bowl
run, a record). Awards and playoff statistics are extractable from licensed sources and would be two more E steps
and two more columns in the career table; narrative is not a column, which is one reason the project reports
resemblance rather than probability.
""")

md(r"""
**5.5 Why not fit a classifier now?** We have a labelled column (`hof`) and features. What stops a model from being
the next cell?
""")
answer(r"""
Four things. Few positives (eight observable Hall of Famers, two complete), so any model would mostly memorize them.
Leakage in time: the label is known only years after the career ends, so a fair evaluation trains on cohorts that
retired long enough ago to have been judged and tests on later cohorts. Data rights: the source must permit model
use (nflverse does; Pro Football Reference's terms do not). And calibration: a "72 % chance of induction" would be
read as a forecast, which these data cannot support. The descriptive comparison is the right first deliverable.
""")

md(r"""
**5.6 Coherence check.** Go back to the SSVA table in Part 1. Is there a cell in this notebook that does not sit in
one of its rows? Is there a row with no figure?
""")
answer(r"""
Every figure maps to a question (1 → Q2, 2 → Q3, 3 → Q3 and Q4, 4 → Q5). Q1 has a table and a count, which is the
right form. What is missing is receiving: Q2 promises "rushing and receiving" and only rushing is shown, so the next
version either adds a receiving percentile to Figures 1 to 3 or edits the question. Either way the table and the
notebook must agree.
""")

# ============================================================================ #
#  Takeaways
# ============================================================================ #
md(r"""
---
## For your own final project — the checklist

1. **One paragraph, then numbered questions.** If a question cannot be numbered it is not a question yet.
2. **ETV.** Label every planned step E, T or V. Cache every E with its access date and the code that made it.
   Write every T as a named step with a check (`assert` one row per key; print the counts you expect).
3. **SSVA.** For each question, name the **summary statistic** before choosing the **visualization**, and write the
   **analysis sentence** the figure is supposed to support. One figure per question; colors fixed by entity.
4. **Say what the data cannot answer** (coverage, definitions, status) in the notebook, not in the presentation's
   last slide.
5. **Sensitivity before modelling.** Change the one rule you chose (the threshold, the peer group) and report what
   moves. A model comes after the descriptive version is coherent, and only with data licensed for it.
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
    "# GSB 5544 — Topic 12.1: Project Exploration — NFL Running Backs and the Hall of Fame\n"
    "*A 20-minute fill-in-and-discuss opener for the final project. One project idea taken from an outline to a running "
    "first version. Two maps keep it coherent: **ETV** (every step is an Extraction, a Transformation or a "
    "Visualization) and **SSVA** (every question gets a Summary Statistic, a Visualization and an Analysis sentence). "
    "Fill each `____` in the code and the prose, run the cells, and jot a sentence for each discussion prompt.*"
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
    """Execute the solution in place, reading data locally, then restore the URL sources."""
    nb = json.loads(path.read_text())
    local = copy.deepcopy(nb)
    for cell in local["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"].replace(DATA, str(DATA_DIR) + "/")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        save(tmp_path, local)
        subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=600", str(tmp_path)], check=True)
        executed = json.loads(tmp_path.read_text())
    for original, ran in zip(nb["cells"], executed["cells"]):
        if original["cell_type"] == "code":
            original["outputs"] = ran.get("outputs", [])
            original["execution_count"] = ran.get("execution_count")
    nb["metadata"]["language_info"] = executed["metadata"].get("language_info", nb["metadata"]["language_info"])
    save(path, nb)
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c["outputs"] if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"Execution errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    ensure_data("--refresh-data" in sys.argv)
    compute_values()
    for c in cells:   # the cells were built before the values existed; substitute now
        c["source"] = T(c["source"])
    leftovers = [c["source"][:60] for c in cells if "⟦" in c["source"]]
    assert not leftovers, leftovers
    WEEK.mkdir(parents=True, exist_ok=True)
    student = STEM.with_name(STEM.name + "-empty.ipynb")
    solution = STEM.with_name(STEM.name + "-solution.ipynb")
    save(student, notebook(student_cells(cells)))
    save(solution, notebook(solution_cells(cells)))
    n_blanks = sum(len(BLANK.findall(c["source"])) for c in cells)
    n_q = sum(c["source"].startswith("**Answer:**") for c in cells)
    print(f"wrote {student.relative_to(ROOT)} and {solution.relative_to(ROOT)} ({len(cells)} cells, "
          f"{n_blanks} blanks, {n_q} discussion prompts)")
    if "--no-exec" not in sys.argv:
        execute(solution)


if __name__ == "__main__":
    main()
