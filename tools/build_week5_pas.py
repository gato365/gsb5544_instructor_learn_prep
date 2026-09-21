#!/usr/bin/env python3
"""Build the Week 5 practice-activity INSTRUCTOR SOLUTION notebooks (PA 5.1 and PA 5.2).

Usage:  python3 tools/build_week5_pas.py            # build + execute both -solution files
        python3 tools/build_week5_pas.py --no-exec  # build without executing

The student notebooks (`GSB5544_PA_5_1_JSON_Data_Format_and_APIs.ipynb`, `GSB5544_PA_5_2_HTML_and_Web_Scraping.ipynb`)
are the source of truth for the questions and are left untouched.  For each, this script writes a `-solution.ipynb`
sibling in which every `# YOUR CODE HERE` cell is replaced — by order — with an answer group: the approach, complete
runnable code, what each important piece does, the expected output, and the common mistakes to watch for in class.

Everything is executed against the live sources (dlsun.github.io, api.tvmaze.com, en.wikipedia.org,
scrapethissite.com), so executing needs a network connection and takes a couple of minutes (requests are staggered).

The Tasty API needs a personal RapidAPI key.  Set the environment variable RAPIDAPI_KEY before building and that
section runs live; without it, the section runs on a small, clearly labelled MOCK of the API's response structure,
so the processing code is still executed and visible.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_5"
JUPYTER = "/opt/anaconda3/bin/jupyter"

PA51_SRC = WEEK / "GSB5544_PA_5_1_JSON_Data_Format_and_APIs.ipynb"
PA52_SRC = WEEK / "GSB5544_PA_5_2_HTML_and_Web_Scraping.ipynb"

PLACEHOLDER = re.compile(r"^\s*#\s*YOUR CODE HERE")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


# ============================================================================ #
#  PA 5.1 — JSON and APIs
# ============================================================================ #

PA51_TITLE = "# GSB 5544 — PA 5.1: Hierarchical Data, the JSON Data Format, and APIs — INSTRUCTOR SOLUTION"

PA51_INTRO = md("""
**How to use this notebook.** Each question is answered in the same five beats: **Approach** (the idea, in a
sentence or two) → **code** (complete and executed) → **What the code does** (piece by piece) → **Expected
output** → **Common mistakes** (what students actually do, and the symptom you will see on their screen).

**The one idea to keep returning to:** *decide what one row should be.* A show? A season? An episode? A cast
member? A tag? Every question in this activity is "pick the unit of the row, flatten to that unit with
`json_normalize`, then do ordinary pandas."

**Live data.** The TVMaze answers come from a live API; numbers were correct when this notebook was executed
(September 2026) and may drift slightly if TVMaze edits its records.
""")

PA51_ANSWERS: list[list[dict]] = [
    # ---- Shows 1
    [
        md("""
### Solution 1 — networks

**Approach.** One row per **show** is already what `df_shows` is, and the network's name was flattened into the
column `network.name`. Counting the values of one column is `value_counts()`.
"""),
        code("""
# Orientation first: what did json_normalize give us?
print(type(data_shows), len(data_shows))          # a list of 10 shows
print(df_shows.shape)
df_shows[["id", "name", "premiered", "network.name", "webChannel.name"]]
"""),
        code("""
df_shows["network.name"].value_counts(dropna=False)
"""),
        md("""
**What the code does.**
- `data_shows` is a **list** of 10 dictionaries (one per show). `pd.json_normalize` made one row per element.
- Each show's `network` is a nested *dictionary* (`{"id": ..., "name": ..., "country": {...}}`), so it was
  spread into dotted columns: `network.id`, `network.name`, `network.country.name`, …
- `value_counts()` tabulates a column. `dropna=False` keeps the missing value visible instead of silently
  dropping a show.

**Expected output.** NBC 2 and Cartoon Network 2; HBO, Pop, Channel 4, Global, The CW 1 each; and **one `NaN`**.

**Points to make in class.**
- The `NaN` is *Chicken Girls*: it is a web series, so its `network` is `null` in the JSON and its home
  (Brat) is under `webChannel.name` instead. A fuller answer: `df_shows["network.name"].fillna(df_shows["webChannel.name"])`.
- There are **two different shows called _The Powerpuff Girls_** (the 1998 original and the 2016 reboot —
  different `id`, different `premiered`). That is why Cartoon Network has 2. This matters in the next two
  questions: **group by `id`, not by `name`.**

**Common mistakes.**
- `df_shows["network"]["name"]` or `df_shows.network.name` → `KeyError`/wrong thing. After flattening, the
  column's *name* is the single string `"network.name"`; the dot is just a character. Use `df_shows["network.name"]`.
- Forgetting `dropna=False` and reporting 9 shows without noticing one vanished.
- Trying `data_shows["network"]` → `TypeError: list indices must be integers` — `data_shows` is a list; pick a
  show first (`data_shows[0]["network"]["name"]`).
"""),
    ],
    # ---- Shows 2
    [
        md("""
### Solution 2 — number of seasons, two ways

**Approach.** `seasons` is a **list** inside each show, so `json_normalize` left it as a list in one cell.
- **Way 1** keeps one row per show and measures each list: `len`.
- **Way 2** changes the unit of the row to the **season** (`record_path="seasons"`) and then counts rows per show.
"""),
        code("""
# Way 1 — one row per show; the "seasons" cell holds a list, and len() of a list is its number of elements
df_shows["n_seasons"] = df_shows["seasons"].apply(len)
df_shows[["id", "name", "premiered", "n_seasons"]]
"""),
        code("""
# Way 2 — one row per SEASON, carrying the show's id and name down onto each season row
df_seasons = pd.json_normalize(
    data_shows,
    record_path="seasons",            # the list to unpack: each season becomes a row
    meta=["id", "name"],              # fields of the parent show to keep
    meta_prefix="show.",              # seasons ALSO have "id" and "name" -> rename the show's to avoid a clash
)
print(df_seasons.shape)
df_seasons[["show.id", "show.name", "number", "premiereDate", "endDate"]].head(8)
"""),
        code("""
df_seasons.groupby(["show.id", "show.name"]).size()
"""),
        md("""
**What the code does.**
- `.apply(len)` runs `len` on each cell of the column. The cell is a Python list of season dictionaries, so
  the result is the number of seasons.
- `record_path="seasons"` tells `json_normalize` to go *into* each show's `seasons` list and make **one row
  per season** (43 rows = 6 + 7 + 3 + 3 + 1 + 5 + 2 + 6 + 2 + 8).
- `meta=["id", "name"]` copies the show's id and name onto each of its season rows; without it you would have
  43 seasons and no idea which show each belongs to.
- `meta_prefix="show."` is **required here**: a season has its own `id` and `name`, and pandas refuses to
  create two columns with the same name.
- `groupby([...]).size()` counts rows per group — seasons per show.

**Expected output.** Girls 6 · The Golden Girls 7 · Good Girls 3 · The Powerpuff Girls (2016, id 6771) 3 ·
Florida Girls 1 · Chicken Girls 5 · Derry Girls 2 · The Powerpuff Girls (1998, id 1955) 6 · Bomb Girls 2 ·
Gilmore Girls 8. Both ways agree.

**Common mistakes.**
- Omitting `meta_prefix` → `ValueError: Conflicting metadata name id, need distinguishing prefix`. The error
  message is, for once, exactly the fix.
- `groupby("show.name")` alone → *The Powerpuff Girls* shows **9** seasons (3 + 6): two shows merged because
  they share a name. Group by the id (keep the name alongside for readability).
- `df_shows["seasons"].str.len()` happens to work on lists too, but `len(df_shows["seasons"])` returns 10 —
  the length of the *column*, not of each list.
- Using `.count()` on a column with missing values (e.g. `endDate`) under-counts; `.size()` counts rows.
"""),
    ],
    # ---- Shows 3
    [
        md("""
### Solution 3 — episode title lengths

**Approach.** The unit is now the **episode**, which is *two* lists down: show → `seasons` → `episodes`. Give
`record_path` the path as a list. Then it is Week 4 string work (`.str.len()`) plus a grouped summary.
"""),
        code("""
df_episodes = pd.json_normalize(
    data_shows,
    record_path=["seasons", "episodes"],   # walk into seasons, then into each season's episodes
    meta=["id", "name", "premiered"],
    meta_prefix="show.",
)
print(df_episodes.shape)
df_episodes[["show.name", "season", "number", "name"]].head()
"""),
        code("""
df_episodes["title_length"] = df_episodes["name"].str.len()

(df_episodes
 .groupby(["show.id", "show.name", "show.premiered"])["title_length"]
 .agg(["count", "mean", "median", "min", "max"])
 .sort_values("mean", ascending=False)
 .round(1))
"""),
        code("""
# The extremes, to make it concrete
df_episodes.sort_values("title_length")[["show.name", "name", "title_length"]].iloc[[0, -1]]
"""),
        md("""
**What the code does.**
- `record_path=["seasons", "episodes"]` — a *list* of keys means "follow them in order". Result: 741 rows, one
  per episode.
- In `df_episodes`, the column `name` is the **episode title** (the record's own field); the show's name
  arrives as `show.name` because of the prefix.
- `.str.len()` gives the number of characters of each title (Topic 4.2).
- `.agg([...])` computes several summaries at once; reporting the mean *and* the median guards against one
  very long title distorting the picture; `count` shows how many episodes each summary rests on.

**Expected output.** Longest titles: **The Powerpuff Girls, the 1998 original (id 1955)** — mean ≈ 28
characters, median 29 (its titles are often two segments joined, e.g. "A / B"). Shortest: **Derry Girls**
(mean ≈ 12.5, median 9 — mostly "Episode 1", "Episode 2", …), with **Girls** close behind (mean ≈ 13).

**Common mistakes.**
- `record_path="episodes"` → `KeyError: 'episodes'`: episodes are not directly under a show; they live inside
  each season.
- Measuring `df_episodes["show.name"].str.len()` — the length of the *show's* name — because both are called
  "name" in the raw JSON. Look at `.head()` before computing.
- Grouping by show name only, which again blends the two Powerpuff series (and here it matters: the original
  is the longest, the reboot is mid-table).
- Answering from `max` alone: Chicken Girls has the single longest title (70 characters) but is not the show
  that *tends* to have long titles. "Tends to" asks for a centre, not an extreme.
"""),
    ],
    # ---- Shows 4
    [
        md("""
### Solution 4 — a shared birthday

**Approach.** The unit is the **cast member** (`record_path="cast"`). A birthday is the month and day of
`person.birthday`, which is text like `"1986-05-13"` — so slice off the year (Topic 4.2) and compare.
"""),
        code("""
df_cast = pd.json_normalize(data_shows, record_path="cast", meta=["id", "name"], meta_prefix="show.")
print(df_cast.shape)
df_cast[["show.name", "person.name", "person.birthday", "character.name"]].head()
"""),
        code("""
my_birthday = "05-13"          # MM-DD  <- change to your own

df_cast["birth_mmdd"] = df_cast["person.birthday"].str.slice(5)        # "1986-05-13" -> "05-13"
df_cast.loc[df_cast["birth_mmdd"] == my_birthday,
            ["person.name", "person.birthday", "character.name", "show.name"]]
"""),
        code("""
# If nobody shares your birthday: which days ARE represented, and how many birthdays are missing?
print("missing birthdays:", df_cast["person.birthday"].isna().sum(), "of", len(df_cast))
df_cast["birth_mmdd"].value_counts().head()
"""),
        md("""
**What the code does.**
- Each element of a show's `cast` list is `{"person": {...}, "character": {...}}`; flattening gives
  `person.name`, `person.birthday`, `character.name`, and so on — 112 rows.
- `.str.slice(5)` keeps characters from position 5 onward: `"1986-05-13"` → `"05-13"`. Comparing month-day
  ignores the birth year, which is what "share a birthday" means.
- `.loc[condition, columns]` filters rows and picks the columns to show in one step.

**Expected output (for May 13).** Two people: **Lena Dunham** (Hannah Horvath, *Girls*) and **Bea Arthur**
(Dorothy Zbornak, *The Golden Girls*). July 13 is the most common date in the data (4 rows).

**Common mistakes.**
- Comparing the full date (`== "1986-05-13"`) — that requires sharing the birth *year* too.
- Writing the day without zero-padding (`"5-13"`) → no matches; the data is `MM-DD` with leading zeros.
- 41 of 112 birthdays are missing (`NaN`), so "nobody shares my birthday" is common and not a bug — hence the
  question's "try a different day". `.str.slice` on `NaN` just returns `NaN`, and `NaN == "05-13"` is `False`, so
  no special handling is needed.
- Alternative that students who know datetimes may try: `pd.to_datetime(df_cast["person.birthday"])` then
  `.dt.month` / `.dt.day`. Equally correct, slightly more typing.
"""),
    ],
    # ---- TVMaze 1
    [
        md("""
### Solution (TVMaze 1) — the longest show that aired on 4 February 2018

**Approach.** Read the documentation for the **Schedule** endpoint: `/schedule?country=US&date=2018-02-04`
returns a list of the *episodes* that aired that day, each with the `show` nested inside. One request, flatten,
sort by `runtime`.
"""),
        code("""
import requests

response = requests.get("https://api.tvmaze.com/schedule",
                        params={"country": "US", "date": "2018-02-04"})
print(response.url)
print(response.status_code)
"""),
        code("""
# Processing lives in its own cell, so re-running it never re-requests
schedule = response.json()
print(type(schedule), len(schedule))
print(schedule[0].keys())

df_schedule = pd.json_normalize(schedule)
df_schedule.shape
"""),
        code("""
(df_schedule
 .sort_values("runtime", ascending=False)
 [["show.id", "show.name", "name", "airtime", "runtime", "show.network.name"]]
 .head())
"""),
        md("""
**What the code does.**
- `params={...}` — `requests` builds the query string; `response.url` shows the result
  (`...schedule?country=US&date=2018-02-04`), which is worth printing so students see what was actually asked.
- `status_code` 200 = success. Check it *before* `.json()`; a failed request often still has a body, and
  `.json()` on an error page is a confusing way to find out.
- Each element of the reply is an **episode** with keys like `name`, `airtime`, `runtime`, and `show` (a nested
  dictionary). After flattening, the episode's own fields keep their names and the show's fields become
  `show.id`, `show.name`, `show.network.name`, … (41 rows × ~60 columns).
- `sort_values("runtime", ascending=False)` puts the longest broadcast first.

**Expected output.** **Super Bowl LII — New England Patriots vs. Philadelphia Eagles**, `runtime` = **210
minutes**, 18:30. Next: *Alaska: The Last Frontier* (180), then several 120-minute programmes.

**Common mistakes.**
- Date format: the API wants ISO `YYYY-MM-DD`. `"02-04-2018"` or `"2/4/2018"` returns an error or an empty list.
- Leaving out `country=US`: the endpoint then defaults to US today, but students who pass another country
  code get a different (valid-looking) answer.
- Sorting by `show.runtime` instead of `runtime`. `show.runtime` is the show's *typical* length (Puppy Bowl:
  typical 180, but that day's broadcast ran 120); `runtime` is the length of *this airing*. The question is
  about what aired that day. The top answer is the same either way here, but the ranking below it is not.
- Reading the question as "longest-running series" — it is about duration in minutes.
- Building the URL by hand with a typo (`?country=US?date=...`); passing `params` avoids this entirely.
"""),
    ],
    # ---- TVMaze 2
    [
        md("""
### Solution (TVMaze 2) — non-voice actors on more than one show

**Approach.** The schedule tells us *which shows* aired; the cast lives at a different endpoint,
`/shows/{id}/cast`, **one show per request**. So: collect the distinct show ids → loop, one request each,
pausing between requests → tag each cast record with its show → flatten once → filter to non-voice →
count **distinct shows** per person.
"""),
        code("""
show_ids = df_schedule["show.id"].unique()
len(show_ids)             # 41 airings, but some shows aired twice -> fewer distinct shows
"""),
        code("""
import time

# REQUEST CELL — run once. ~40 requests with a half-second pause takes under a minute.
cast_records = []
for show_id in show_ids:
    response = requests.get(f"https://api.tvmaze.com/shows/{show_id}/cast")
    if response.status_code != 200:                 # don't let one bad reply poison the list
        print("skipped", show_id, "status", response.status_code)
        continue
    members = response.json()                       # a list of {"person":..., "character":..., "self":..., "voice":...}
    for member in members:
        member["show_id"] = int(show_id)            # the cast endpoint does NOT say which show -> tag it ourselves
    cast_records.extend(members)                    # extend = add each member; append would add one list
    time.sleep(0.5)                                 # stagger the requests (TVMaze allows ~20 per 10 seconds)

len(cast_records)
"""),
        code("""
# PROCESSING CELL — safe to re-run as often as you like
df_cast_day = pd.json_normalize(cast_records)
print(df_cast_day.shape)
print(df_cast_day["voice"].value_counts())
df_cast_day[["show_id", "person.id", "person.name", "character.name", "voice", "self"]].head()
"""),
        code("""
non_voice = df_cast_day[~df_cast_day["voice"]]                    # keep rows where voice is False

shows_per_person = (non_voice
                    .groupby(["person.id", "person.name"])["show_id"]
                    .nunique()                                     # DISTINCT shows, not rows
                    .sort_values(ascending=False))
repeat_people = shows_per_person[shows_per_person > 1]
repeat_people
"""),
        code("""
# Which shows? Bring the show names back in from the schedule (a Week 3 merge)
show_names = df_schedule[["show.id", "show.name"]].drop_duplicates()

(non_voice[non_voice["person.id"].isin(repeat_people.index.get_level_values("person.id"))]
 .merge(show_names, left_on="show_id", right_on="show.id")
 [["person.name", "character.name", "self", "show.name"]]
 .drop_duplicates()
 .sort_values("person.name"))
"""),
        md("""
**What the code does.**
- `.unique()` — 41 airings come from 38 distinct shows; requesting a show's cast twice would waste requests
  *and* double-count its cast.
- The f-string `f".../shows/{show_id}/cast"` builds a different URL on each pass of the loop. Here the varying
  part is in the **path**, not a `?parameter`, so `params=` is not used.
- **Tagging**: the reply contains people and characters but no show id. Adding `member["show_id"] = ...`
  *before* `extend` is what makes the final question answerable. (`int(...)` converts numpy's integer type to a
  plain Python one.)
- `extend` vs `append`: `extend` adds the *elements* of the reply to our list (a flat list of cast members —
  what `json_normalize` wants). `append` would give a list of 38 lists.
- `time.sleep(0.5)` — the "stagger" the hint demands. TVMaze's limit is about 20 calls per 10 seconds; without a
  pause you get HTTP **429** (Too Many Requests) and, if you persist, a temporary block.
- `~df["voice"]` — `voice` is already `True`/`False`; `~` flips it, keeping non-voice credits.
- `.nunique()` counts **distinct** shows per person. This is the question's warning in code: a person credited
  twice on the *same* show (two characters) has 2 rows but 1 show.
- Grouping by `person.id` as well as the name guards against two different people with the same name.

**Expected output.** 38 shows → about 880 cast rows, roughly 690 of them voice credits (the day's schedule is
full of animation: *The Simpsons*, *Family Guy*, *Bob's Burgers*, …). Among ~186 non-voice credits, two people
appear on more than one show: **Tom Llamas** and **John Dickerson** — news anchors, credited as themselves
(`self == True`) on Sunday news programmes.

**Common mistakes.**
- **No `time.sleep`** → a burst of 429 responses; then `response.json()` returns an error dictionary rather than a
  list, and `extend` quietly adds its *keys* (strings) to the list, so `json_normalize` fails much later with a
  baffling message. The `status_code` check above prevents that.
- **`.size()` / `.count()` / `value_counts()` on names** instead of `nunique()` on shows → a long list of false
  positives: every actor who plays two characters on one show.
- **Forgetting to tag the show** → all the cast in one frame and no way to count shows. Students usually
  discover this only at the groupby; point them back to the loop.
- **Requests and processing in one cell** → every pandas typo costs another 40 requests. Split them.
- Looping over `df_schedule["show.id"]` without `.unique()` → duplicate requests and inflated counts.
- Filtering with `df["voice"] == "False"` (a string) → nothing matches; it is a Boolean.
- Students may ask whether "actors" should exclude people appearing as themselves. The question only says
  non-voice, so the two anchors are the intended answer; adding `& ~df["self"]` leaves nobody, which is a nice
  discussion of how the definition drives the answer.
"""),
    ],
    # ---- Tasty 1
    [
        md("""
### Solution (Tasty 1) — daikon recipes to a data frame

**Approach.** Same request pattern as TVMaze, with two additions: the **`headers`** carry the key, and the
search term goes in `params` (`q`). The reply is a **dictionary** with two keys — `count` (how many recipes
match in total) and `results` (this page of recipes) — so flatten `["results"]`, not the whole reply.
"""),
        code("""
if USE_LIVE:
    response = requests.get(f"{domain}/recipes/list", headers=headers,
                            params={"from": 0, "size": 40, "q": "daikon"})
    print(response.status_code)
    daikon_json = response.json()
else:
    daikon_json = MOCK_DAIKON                      # stand-in with the same structure — NOT real Tasty data

print(daikon_json.keys())
df_daikon = pd.json_normalize(daikon_json["results"])
df_daikon.shape
"""),
        code("""
df_daikon[["id", "name", "price.portion", "user_ratings.count_positive", "user_ratings.count_negative"]].head()
"""),
        md("""
**What the code does.**
- `headers=headers` must accompany **every** request to this API; that is how RapidAPI knows whose quota to
  charge. `params` are the search options from the documentation form: `from` (offset of the first result),
  `size` (results per page, maximum 40), `q` (search text).
- `daikon_json.keys()` → `count` and `results`. `results` is the list of recipes, and `json_normalize` wants a
  list of records — so pass `daikon_json["results"]`.
- Nested dictionaries inside a recipe flatten to dotted names: `price.portion`, `user_ratings.count_positive`, …
  Nested **lists** (`tags`, `sections`, `instructions`) stay as lists in a single cell — that is what Tasty 5 is about.

**Common mistakes.**
- `pd.json_normalize(response.json())` without `["results"]` → a **one-row** frame with columns `count` and
  `results`. One row is the giveaway.
- HTTP **401/403** with a message such as "You are not subscribed to this API": the key is fine but the student
  never clicked *Subscribe* on the Basic plan — or pasted the key with a stray space/quote. **429** means the
  monthly free quota is used up (it is small — another reason to separate request cells from processing cells).
- Leaving `headers` off the second and later requests.
- **Posting the key**: a notebook pushed to GitHub or shared in a forum exposes it. In this notebook the key is
  read from an environment variable for exactly that reason.
"""),
    ],
    # ---- Tasty 2
    [
        md("""
### Solution (Tasty 2) — how many daikon recipes, and the cheapest per portion

**Approach.** "How many" is the API's own `count` (compare it with the number of rows actually received).
"Cheapest per portion" is a sort on `price.portion` — after removing recipes with no price.
"""),
        code("""
print("API says:", daikon_json["count"], "| rows received:", len(df_daikon))
"""),
        code("""
priced = df_daikon[df_daikon["price.portion"] > 0]              # drop missing (NaN) and zero prices
print(len(priced), "of", len(df_daikon), "recipes have a per-portion price")

priced.sort_values("price.portion")[["name", "price.portion", "price.total", "num_servings"]].head()
"""),
        md("""
**What the code does.**
- `count` is the total number of matches on the server; `len(df_daikon)` is how many came back in *this* reply.
  They agree only when the total fits in one page (≤ 40). That distinction is the entire reason for the loop in
  the next question.
- `df["price.portion"] > 0` is `False` for `NaN`, so this one filter removes both the recipes with no price
  information and any recorded as 0.
- `sort_values("price.portion")` ascending → cheapest first. Prices appear to be recorded in **US cents**
  (e.g. `250` = $2.50) — worth saying aloud, since "250" looks expensive for a portion.

**Common mistakes.**
- Answering with `len(df_daikon)` when `count` is larger than 40 — the frame only holds the first page.
- Sorting without filtering: a recipe with price 0 (or `NaN`, which sorts last but confuses `idxmin` users
  less than zeros do) is reported as "cheapest".
- Using `price.total` (the whole dish) when the question asks **per portion**.
- `KeyError: 'price.portion'` — if *no* recipe in the reply has a `price` object the column is never created;
  check `df_daikon.columns`.
"""),
    ],
    # ---- Tasty 3
    [
        md("""
### Solution (Tasty 3) — all the avocado recipes (pagination)

**Approach.** There are hundreds of matches but at most 40 per reply, so request **pages**: `from=0`, `from=40`,
`from=80`, … The first reply's `count` tells us where to stop. Collect only each reply's `results`.

⚠️ **The hint in the activity has a typo.** It shows `.extend(response.json()).get("results")`. The parentheses
are in the wrong place: that extends the list with the reply *dictionary* (which adds its **keys**, the strings
`"count"` and `"results"`), and then calls `.get` on `None` (the return value of `extend`) →
`AttributeError: 'NoneType' object has no attribute 'get'`. The correct form is
**`.extend(response.json().get("results"))`**. Expect several students to paste the hint verbatim.
"""),
        code("""
PAGE_SIZE = 40

if USE_LIVE:
    # REQUEST CELL — run once. First request also tells us the total.
    first = requests.get(f"{domain}/recipes/list", headers=headers,
                         params={"from": 0, "size": PAGE_SIZE, "q": "avocado"}).json()
    total = first["count"]
    avocado = list(first["results"])

    for start in range(PAGE_SIZE, total, PAGE_SIZE):            # 40, 80, 120, ...
        response = requests.get(f"{domain}/recipes/list", headers=headers,
                                params={"from": start, "size": PAGE_SIZE, "q": "avocado"})
        if response.status_code != 200:
            print("stopped at", start, "status", response.status_code)
            break
        avocado.extend(response.json().get("results"))          # only the recipes, not the wrapper
        time.sleep(1)                                           # respect the rate limit
else:
    total = MOCK_AVOCADO["count"]
    avocado = list(MOCK_AVOCADO["results"])                     # stand-in — NOT real Tasty data

print("API count:", total, "| recipes collected:", len(avocado))
"""),
        code("""
# PROCESSING CELL
df_avocado = pd.json_normalize(avocado).drop_duplicates(subset="id")
df_avocado.shape
"""),
        md("""
**What the code does.**
- `range(PAGE_SIZE, total, PAGE_SIZE)` generates the offsets 40, 80, 120, … up to (not including) `total`; with
  the first page already in hand, that covers every recipe with no wasted request.
- `response.json().get("results")` — `.get("results")` is the dictionary method, the same as `["results"]` except
  that it returns `None` instead of raising an error if the key is missing.
- `avocado.extend(...)` grows **one flat list of recipe dictionaries** across all pages — the shape
  `json_normalize` needs.
- `time.sleep(1)` between requests; the `status_code` check stops the loop cleanly if the quota runs out
  rather than filling the list with error messages.
- `drop_duplicates(subset="id")`: results can shift between pages while you are paging, so a recipe
  occasionally arrives twice.

**Teaching suggestion (from the activity, and worth repeating):** first run the loop for 2–3 pages only —
`range(PAGE_SIZE, 3 * PAGE_SIZE, PAGE_SIZE)` — and check `len(avocado)` is 120 and the first names differ page to
page. Only then run the full loop. Every request counts against a small monthly quota.

**Common mistakes.**
- The hint's misplaced parenthesis (above).
- `append` instead of `extend` → a list of lists; `json_normalize` then produces nonsense columns `0, 1, 2, …`.
- Forgetting to change `from` inside the loop (a constant `params`) → the same 40 recipes, many times over;
  `drop_duplicates` exposes it (shape collapses to 40 rows).
- Asking for `size=100`: the API silently caps at 40, so stepping by 100 **skips** 60 recipes per page.
- Re-running the request cell to fix a pandas error → quota gone. Keep processing in a separate cell.
"""),
    ],
    # ---- Tasty 4
    [
        md("""
### Solution (Tasty 4) — proportion of positive reviews

**Approach.** Each recipe has `user_ratings.count_positive` and `user_ratings.count_negative`. The proportion
is positive ÷ (positive + negative). Then filter to recipes with more than 500 reviews in total and sort.
"""),
        code("""
df_avocado["n_reviews"] = (df_avocado["user_ratings.count_positive"]
                           + df_avocado["user_ratings.count_negative"])
df_avocado["prop_positive"] = df_avocado["user_ratings.count_positive"] / df_avocado["n_reviews"]

df_avocado[["name", "user_ratings.count_positive", "user_ratings.count_negative",
            "n_reviews", "prop_positive"]].head()
"""),
        code("""
(df_avocado[df_avocado["n_reviews"] > 500]
 .sort_values("prop_positive", ascending=False)
 [["name", "n_reviews", "prop_positive"]]
 .head())
"""),
        md("""
**What the code does.**
- Column arithmetic is element-wise: one proportion per recipe.
- A recipe with no reviews gives 0 / 0 = `NaN` (pandas does not raise an error). Those rows then fail the
  `> 500` filter on their own, so no special handling is needed — but students should know why `NaN`s appear.
- The 500-review threshold is the statistical point of the question: a recipe with 3 reviews, all positive,
  has a proportion of 1.0 and would otherwise "win". Small samples make extreme proportions.
- The API also supplies `user_ratings.score`, which is this same proportion — a handy check on the arithmetic.

**Common mistakes.**
- Dividing by `count_positive` alone, or by `count_negative` → proportions above 1.
- Filtering on `count_positive > 500` rather than on the **total** number of reviews.
- Sorting first and reading the top row *before* applying the 500 filter → a recipe with a handful of reviews.
- Reporting `idxmax()` (a row label) rather than the recipe's name.
"""),
    ],
    # ---- Tasty 5
    [
        md("""
### Solution (Tasty 5) — how many avocado recipes are vegetarian?

**Approach.** `tags` is a **list of dictionaries** inside each recipe, so change the unit of the row to the
**tag**: `record_path="tags"`, keeping the recipe's `id` and `name` as `meta`. Then filter the tag rows to
`vegetarian` and count **distinct recipes**. No new API requests — this reuses the `avocado` list.
"""),
        code("""
with_tags = [recipe for recipe in avocado if recipe.get("tags")]       # a few items have no tags list
print(len(avocado) - len(with_tags), "items without tags skipped")

df_tags = pd.json_normalize(with_tags, record_path="tags",
                            meta=["id", "name"], meta_prefix="recipe.")
print(df_tags.shape)
df_tags.head()
"""),
        code("""
vegetarian = df_tags[df_tags["name"] == "vegetarian"]
n_vegetarian = vegetarian["recipe.id"].nunique()
print(n_vegetarian, "of", df_avocado["id"].nunique(), "avocado recipes are tagged vegetarian")
"""),
        md("""
**What the code does.**
- `record_path="tags"` → one row per (recipe, tag) pair; a recipe with 12 tags contributes 12 rows. Each tag has
  its own `id`, `name` (machine name, e.g. `vegetarian`), `display_name`, and `type` (e.g. `dietary`).
- `meta=["id", "name"]` with `meta_prefix="recipe."` — again essential, because tags *also* have `id` and
  `name`. In `df_tags`, `name` is the **tag's** name and `recipe.name` is the recipe's.
- The list comprehension drops items with no `tags` (the search can return *compilations* as well as recipes);
  `json_normalize` raises `KeyError: 'tags'` if any record lacks the key.
- `.nunique()` on the recipe id counts recipes, not rows — robust even if a duplicate slipped through.

**Common mistakes.**
- `KeyError: "Key 'tags' not found"` → a record without tags; filter first, as above.
- `ValueError: Conflicting metadata name id` → missing `meta_prefix` (same fix as Shows 2).
- Filtering `df_tags["recipe.name"].str.contains("vegetarian")` — that searches recipe *titles*, and misses most
  vegetarian recipes.
- `str.contains("veg")` on the tag name also matches `vegan` and `vegetables` style tags; use `==`. (Whether
  vegan recipes should count as vegetarian is a fair discussion — in practice Tasty tags vegan recipes with
  both.)
- Counting rows of `df_tags` and calling it the number of recipes.
- Going back to the API for this question. The instruction says not to: the data is already in `avocado`.
"""),
    ],
]

# Replaces the activity's hard-coded-key cell with one that reads the key from the environment (and sets up the mock)
PA51_TASTY_SETUP: list[dict] = [
    md("""
> **Instructor note — how this section was run.** The key is read from the environment variable
> `RAPIDAPI_KEY` so that it never appears in the notebook (`export RAPIDAPI_KEY=...` before launching Jupyter,
> or before running `tools/build_week5_pas.py`). **If no key is set, the cells below run on a small MOCK** that
> imitates the structure of the API's reply (`count` + `results`, with `price`, `user_ratings`, and `tags`
> inside each recipe). The mock's recipes and numbers are **invented for illustration** — they are not Tasty
> data — so use the *code and explanations* in class, and rebuild with a key for real answers. The banner
> printed by the next cell says which mode was used.
"""),
    code("""
import os
import time
import requests

domain = "https://tasty.p.rapidapi.com"

API_KEY = os.environ.get("RAPIDAPI_KEY", "PUT-YOUR-KEY-HERE")
USE_LIVE = API_KEY != "PUT-YOUR-KEY-HERE"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "tasty.p.rapidapi.com",
}

print("MODE:", "LIVE Tasty API" if USE_LIVE else "MOCK data (no RAPIDAPI_KEY set) — numbers below are illustrative only")
"""),
    code("""
# MOCK replies — same shape as the real API, invented values. Used only when no key is available.
def _recipe(i, name, portion, pos, neg, tags):
    return {"id": i, "name": name, "num_servings": 4,
            "price": {"total": portion * 4, "portion": portion},
            "user_ratings": {"count_positive": pos, "count_negative": neg,
                             "score": round(pos / (pos + neg), 6) if pos + neg else None},
            "tags": [{"id": 1000 + k, "name": t, "display_name": t.replace("_", " ").title(), "type": "mock"}
                     for k, t in enumerate(tags)]}

MOCK_DAIKON = {"count": 4, "results": [
    _recipe(101, "(mock) Pickled Daikon", 75, 310, 22, ["vegetarian", "vegan", "sides"]),
    _recipe(102, "(mock) Daikon Noodle Soup", 260, 120, 9, ["dinner", "asian"]),
    _recipe(103, "(mock) Banh Mi With Daikon Slaw", 410, 980, 61, ["lunch", "sandwiches"]),
    _recipe(104, "(mock) Daikon Cake", 0, 12, 1, ["vegetarian", "snacks"]),          # price 0 = unknown
]}

MOCK_AVOCADO = {"count": 6, "results": [
    _recipe(201, "(mock) Avocado Toast", 180, 2400, 130, ["vegetarian", "breakfast"]),
    _recipe(202, "(mock) Chicken Avocado Wrap", 390, 820, 95, ["lunch"]),
    _recipe(203, "(mock) Guacamole", 120, 5100, 140, ["vegetarian", "vegan", "appetizers"]),
    _recipe(204, "(mock) Avocado Brownies", 95, 3, 0, ["vegetarian", "desserts"]),      # tiny sample, 100% positive
    _recipe(205, "(mock) Shrimp Avocado Salad", 450, 640, 38, ["dinner", "seafood"]),
    {"id": 206, "name": "(mock) Avocado Compilation", "user_ratings": {"count_positive": 0, "count_negative": 0}},
]}
"""),
    code("""
# The activity's connection test: list the tags the API recognises
if USE_LIVE:
    response = requests.get(f"{domain}/tags/list", headers=headers)
    print(response.status_code)
    tags_json = response.json()
    print(tags_json.keys(), "|", tags_json["count"], "tags")
else:
    print("skipped (mock mode). With a valid key this prints 200 and a dictionary with keys 'count' and 'results'.")
    print("Without a valid key the API answers 401/403 and a JSON message such as 'You are not subscribed to this API.'")
"""),
]


# ============================================================================ #
#  PA 5.2 — HTML and Web Scraping
# ============================================================================ #

PA52_TITLE = "# GSB 5544 — PA 5.2: HTML and Web Scraping — INSTRUCTOR SOLUTION"

PA52_INTRO = md("""
**How to use this notebook.** Each part is answered in the same beats: **Approach** → **code** (complete and
executed) → **What the code does** → **Expected output** → **Common mistakes**.

**The one idea to keep returning to:** *solve it for one, then loop.* Parts 7 – 11 extract a single city by
hand precisely so that part 12 is a copy-and-paste into a `for` loop. The same rhythm repeats for the hockey
pages: one page first, then all of them.

**Live pages.** Wikipedia is edited continuously. The numbers here were correct when this notebook was executed
(September 2026). If the table's `class` attribute or its column order changes, parts 5 – 13 need the
corresponding edit — the *Inspect* step in part 4 is how you find out.
""")

PA52_ANSWERS: list[list[dict]] = [
    # 1
    [
        md("""
### Solution 1 — get the HTML

**Approach.** `requests.get` downloads the page, exactly as it fetched JSON last time; the HTML is in
`response.text`. **Wikipedia rejects requests that do not identify themselves**, so send a `User-Agent` header.
"""),
        code("""
import requests

url = "https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population"
headers = {"User-Agent": "GSB5544-class-exercise/1.0 (Cal Poly; educational use)"}

response = requests.get(url, headers=headers)
print(response.status_code)
print(len(response.text), "characters of HTML")
response.text[:300]
"""),
        md("""
**What the code does.**
- `requests.get(url, headers=headers)` sends an HTTP GET, just like a browser's address bar.
- `headers={"User-Agent": ...}` says who is asking. By default `requests` announces itself as
  `python-requests/2.x`, and Wikipedia's servers answer that with **403 Forbidden**. Any honest descriptive
  string works; Wikimedia's policy asks for one that identifies the project.
- `response.status_code` — 200 means the page arrived. `response.text` is the page source as one long string
  (about 1.7 million characters); printing a slice is enough to see that it is HTML.

**Expected output.** `200`, then roughly 1.7 million characters beginning `<!DOCTYPE html>`.

**Common mistakes.**
- **No header → 403.** The student's `response.text` is then a ~100-character error page, and every later
  step fails in a misleading way: part 3 finds **0 tables**, part 6 raises `IndexError: list index out of range`.
  When a student reports either symptom, ask for `response.status_code` first.
- `response.json()` → `JSONDecodeError`: this is a web page, not an API.
- Using `response` (the object) where the *text* is needed in part 2.
"""),
    ],
    # 2
    [
        md("""
### Solution 2 — parse into a tree

**Approach.** `BeautifulSoup` converts the string into a searchable tree of tags.
"""),
        code("""
from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text, "html.parser")
print(type(soup))
soup.title.text
"""),
        md("""
**What the code does.**
- `BeautifulSoup(html_string, "html.parser")` — the first argument is the HTML **text**; the second names the
  parser (`"html.parser"` ships with Python, so nothing extra to install).
- The result, `soup`, represents the whole document. `soup.title` is a shortcut to the first `<title>` tag, and
  `.text` is the text inside it — a quick confirmation that we parsed the page we meant to.

**Expected output.** `<class 'bs4.BeautifulSoup'>` and `'List of United States cities by population - Wikipedia'`.

**Common mistakes.**
- `BeautifulSoup(response, ...)` → `TypeError` (it needs the string, `response.text`).
- `BeautifulSoup(url, ...)` — parses the *address* as if it were HTML; no error, but `soup` contains no tables.
- Omitting the parser argument works but prints a warning; name it.
- `from bs4 import beautifulsoup` — the class name is case-sensitive; and the package installs as
  `beautifulsoup4` but imports as `bs4`.
"""),
    ],
    # 3
    [
        md("""
### Solution 3 — how many tables?

**Approach.** `find_all("table")` returns a list of every `<table>` tag; `len` counts them.
"""),
        code("""
tables = soup.find_all("table")
len(tables)
"""),
        md("""
**What the code does.** `find_all(tag_name)` searches the entire tree beneath `soup` and returns a list-like
`ResultSet`. Because it is a list, `len()`, indexing (`tables[2]`), and `for` loops all work.

**Expected output.** **10** tables (as executed). The page holds the main cities table plus a sidebar, a legend,
tables for Puerto Rico, census-designated places, cities formerly over 100,000, and so on.

**Common mistakes.**
- `soup.find("table")` returns only the **first** table (a single tag); `len()` of a tag counts its children,
  giving a meaningless number rather than an error.
- Getting 0 → the download failed (part 1, status 403).
"""),
    ],
    # 4
    [
        md("""
### Solution 4 — the attributes of the cities table

**Approach.** In the browser: right-click inside the cities table → **Inspect**, then move up the highlighted
lines until the whole table is highlighted; read the opening `<table ...>` tag. The same information is
available in Python from each tag's `.attrs`:
"""),
        code("""
for i, t in enumerate(tables):
    print(i, "| class:", " ".join(t.attrs.get("class", [])), "| style:", t.attrs.get("style"))
"""),
        md("""
**What the code does.**
- Every tag has an `.attrs` dictionary of its attributes. `.get("style")` returns `None` for tables without a
  `style` rather than raising a `KeyError`.
- `class` comes back as a **list** (`['sortable', 'wikitable', ...]`) because an HTML element can carry several
  classes separated by spaces; `" ".join(...)` prints them as they appear in the source.
- `enumerate` numbers the tables so we can refer to them by position.

**Expected output.** The cities table is the one with
`class="sortable wikitable sticky-header-multi static-row-numbers sort-under col1left col2center"` and
`style="text-align:right"` (index 2 in the list as executed).

**Points to make in class.**
- *Inspect* shows the page **after** the browser's JavaScript has run; `requests` gets the **raw source**. On
  Wikipedia they can differ slightly (a sortable table gains `jquery-tablesorter` in the browser). If a class
  copied from Inspect fails to match, print `.attrs` as above and trust what Python received.
- Many tables share `wikitable`; it is the **combination** of classes (plus the style) that is unique.

**Common mistakes.** Inspecting a *cell* or a *row* and reporting its attributes instead of climbing up to the
`<table>` tag; copying the class string with a missing or extra space.
"""),
    ],
    # 5
    [
        md("""
### Solution 5 — how many tables have these attributes?

**Approach.** Give `find_all` an `attrs` dictionary; only tags whose attributes match are returned.
"""),
        code("""
cities_attrs = {
    "class": "sortable wikitable sticky-header-multi static-row-numbers sort-under col1left col2center",
    "style": "text-align:right",
}
len(soup.find_all("table", attrs=cities_attrs))
"""),
        md("""
**What the code does.** `attrs={"class": "...", "style": "..."}` requires **both** to match. When the class
value contains several space-separated classes, Beautiful Soup compares it with the tag's full `class`
attribute as written — so the string must match exactly, in the same order.

**Expected output.** **1** — the attributes identify the cities table uniquely, which is what makes part 6 safe.

**Common mistakes.**
- A typo anywhere in the long class string → 0 matches, and then `[0]` in part 6 raises `IndexError`.
- Matching on `{"class": "wikitable"}` alone → 7 tables: a *single* class name matches any tag that has that
  class among others.
- Writing `class="..."` as a keyword argument → `SyntaxError`, because `class` is a reserved word in Python.
  That is why we use the `attrs` dictionary (Beautiful Soup also accepts `class_=`).
"""),
    ],
    # 7
    [
        md("""
### Solution 7 — the row for New York City

**Approach.** In HTML a table row is a `<tr>` tag. Search **inside `table`** (not `soup`) for all rows, look at
the first few to see where the data starts, and take the first city.
"""),
        code("""
rows_all = table.find_all("tr")
print(len(rows_all), "rows")

for i in range(5):                                   # peek: which rows are headers?
    cells = rows_all[i].find_all(["th", "td"])
    print(i, [c.text.strip()[:14] for c in cells][:6])
"""),
        code("""
city = rows_all[3]             # rows 0-2 are header rows; row 3 is the first city
print(city.text.strip()[:80])
"""),
        md("""
**What the code does.**
- `table.find_all("tr")` — searching from `table` restricts the search to that table. `soup.find_all("tr")`
  would return the rows of all 10 tables.
- The peek loop prints the first few cells of rows 0 – 4. `find_all(["th", "td"])` accepts a **list** of tag
  names, so it shows header cells and data cells alike.
- `rows_all[3]`: rows 0 and 1 are the two-level header (*Municipality, ST, 2025 estimate…* and the *mi² / km²*
  sub-header), row 2 is an empty spacer row, and row 3 is New York.

**Expected output.** 351 rows (3 header rows + 348 cities). `city` is a single `<tr>` tag whose text begins
`New York[c] NY 8,584,629 …`.

**Common mistakes.**
- `table.find_all("tr")[0]` → the header row; part 8 then fails with `IndexError`, because header rows contain
  `<th>` cells and **no `<td>`**.
- `table.find("tr")` — also the header row (`find` = first match).
- Off-by-one on the index (2 → the empty spacer row → `IndexError` in part 8). Printing the rows, as above,
  beats counting in the browser.
"""),
    ],
    # 8
    [
        md("""
### Solution 8 — the city's name

**Approach.** Cells within a row are `<td>` tags. The name is in the first cell — but so is a footnote marker,
so go one level deeper to the `<a>` (link) tag, which holds only the name.
"""),
        code("""
cells = city.find_all("td")
print(len(cells), "cells")
print(repr(cells[0].text))            # the whole cell: includes the footnote marker

name = cells[0].find("a").text        # the link inside the cell: just the name
name
"""),
        md("""
**What the code does.**
- `city.find_all("td")` → the row's 10 data cells, in column order: 0 name · 1 state · 2 2025 estimate ·
  3 2020 census · 4 change · 5 land area mi² · 6 land area km² · 7 density /mi² · 8 density /km² · 9 location.
- `cells[0].text` gathers **all** text inside the cell: `'New York[c]'`. The `[c]` is a footnote link.
- `cells[0].find("a")` finds the first `<a>` tag inside that cell — the link to the city's article — whose
  text is exactly `'New York'`.
- `repr(...)` shows hidden characters (`\\n`, non-breaking spaces) that `print` would hide — a good debugging habit
  when scraping.

**Expected output.** `'New York'`.

**Common mistakes.**
- Stopping at `cells[0].text` → 28 of the 348 cities keep a footnote marker (`New York[c]`, `Philadelphia[d]`,
  `Jacksonville[e]`, …). It looks fine in `.head()` apart from the first row and then breaks a later merge on city
  name. Alternatives if students did not think of the `<a>` tag: `.text.split("[")[0]`, or Week 4's
  `re.sub(r"\\[.*\\]", "", text)`.
- `city.find("td").text` works for the name (first cell) but does not generalise to parts 9 – 11.
- Forgetting `.text` and storing the **tag** (`<a href=...>New York</a>`) — it prints plausibly but is not a string.
"""),
    ],
    # 9
    [
        md("""
### Solution 9 — the state

**Approach.** Second cell. Plain text, with `.strip()` for safety.
"""),
        code("""
state = cells[1].text.strip()
state
"""),
        md("""
**What the code does.** `cells[1]` is the *ST* column; `.text` extracts the text; `.strip()` removes any
surrounding whitespace or newline (Week 4). Wikipedia cells frequently end in `\\n`.

**Expected output.** `'NY'` — the table gives two-letter postal abbreviations, not state names.

**Common mistakes.** Forgetting that `cells` was defined in part 8 and re-searching from `table` (which gives
the first cells of the *whole table*); skipping `.strip()` and later failing to match `"NY\\n" == "NY"`.
"""),
    ],
    # 10
    [
        md("""
### Solution 10 — the population

**Approach.** Third cell — but scraped values are **text**. Remove the thousands separators, then convert.
"""),
        code("""
print(repr(cells[2].text))

population = int(cells[2].text.strip().replace(",", ""))
population
"""),
        md("""
**What the code does.** `cells[2].text` is the string `'8,584,629'`. `.replace(",", "")` deletes the commas
(→ `'8584629'`) and `int(...)` turns the digits into a number that can be summed, sorted, and plotted.

**Expected output.** `8584629` (the 2025 estimate, as executed).

**Common mistakes.**
- `int("8,584,629")` → `ValueError: invalid literal for int()`. The comma is the culprit.
- Leaving it as text. Nothing fails — until the data frame sorts populations **alphabetically**
  (`'999,999'` > `'8,584,629'`) or `.sum()` concatenates strings. Check `df.dtypes` in part 12.
- Taking `cells[3]` (the 2020 census) — the question asks for the 2025 estimate.
"""),
    ],
    # 11
    [
        md("""
### Solution 11 — the land area

**Approach.** The 2020 land area in **square miles** is the sixth cell (index 5; index 6 is km²). It has a
decimal point, so convert with `float`.
"""),
        code("""
print([c.text.strip() for c in cells[:8]])       # count along the row to find the right index

area = float(cells[5].text.strip().replace(",", ""))
area
"""),
        md("""
**What the code does.** Printing the row's cells is the reliable way to find the index: name, state, 2025
estimate, 2020 census, change, **land area mi²**, land area km², density… `float` rather than `int` because of
the decimal; `.replace(",", "")` because large areas are written like `1,706.8` (Anchorage).

**Expected output.** `300.5`.

**Common mistakes.**
- Counting columns from the **visible header**, where "2020 land area" is one header spanning two cells
  (mi² and km²) — students land on the wrong index. Count `<td>` cells, not header labels.
- `int("300.5")` → `ValueError`.
- Omitting the comma removal: it works for New York and then fails inside the loop at the first city with an
  area over 1,000 sq mi. A reminder that "works for one row" is a necessary test, not a sufficient one.
"""),
    ],
    # 12
    [
        md("""
### Solution 12 — every city: the loop

**Approach.** Parts 7 – 11 *are* the loop body. Replace "row 3" with "each row from 3 onward", collect one
dictionary per city in a list, and hand the list to `pd.DataFrame`.
"""),
        code("""
rows = []
for city in table.find_all("tr")[3:]:                 # every row after the 3 header rows
    cells = city.find_all("td")
    rows.append({
        "name": cells[0].find("a").text,
        "state": cells[1].text.strip(),
        "population": int(cells[2].text.strip().replace(",", "")),
        "area": float(cells[5].text.strip().replace(",", "")),
    })

df_cities = pd.DataFrame(rows)
print(df_cities.shape)
df_cities.head()
"""),
        code("""
# Always audit a scrape: types, missing values, and the two ends of the table
print(df_cities.dtypes)
print("missing values:", df_cities.isna().sum().sum(), "| footnote markers left in names:",
      df_cities["name"].str.contains(r"\\[").sum())
df_cities.tail(3)
"""),
        md("""
**What the code does.**
- `[3:]` slices off the three header rows — the loop version of choosing `[3]` in part 7.
- The loop variable is called `city` on purpose: the lines inside are the code from parts 8 – 11, unchanged.
- `rows.append({...})` adds one dictionary per city. A **list of dictionaries** converts directly to a data
  frame: the keys become the column names.
- The audit cell confirms numeric dtypes (`int64`, `float64`), no missing values, and no `[` left in any name.

**Expected output.** A data frame with **348 rows and 4 columns**, New York / Los Angeles / Chicago at the top,
cities of just over 100,000 at the bottom.

**Common mistakes.**
- `[0:]` or no slice → `IndexError: list index out of range` on the very first pass (header rows have no `<td>`).
  A defensive alternative some students find: `if len(cells) == 0: continue`.
- `rows = []` placed **inside** the loop → a data frame with one row (the last city).
- `pd.DataFrame(...)` called inside the loop — slow, and only the last one survives.
- Appending a plain list `[name, state, population, area]` is fine, but then the columns are `0, 1, 2, 3` unless
  `columns=[...]` is given.
- Re-using the single-city variables (`name`, `state`, …) from parts 8 – 11 in the dictionary instead of
  recomputing them from the current `cells` → 348 copies of New York.
- 349 or 350 rows → a header row slipped in; 0 rows → `soup.find_all` was used and a different table was looped.
"""),
    ],
    # 13
    [
        md("""
### Solution 13 — the same table with `pd.read_html`

**Approach.** `pd.read_html` parses `<table>` tags into data frames for you. It returns a **list** (one data
frame per table found), so narrow it with the same `attrs` and take element `[0]`. Pass it the HTML we already
downloaded, wrapped in `StringIO`.
"""),
        code("""
from io import StringIO

tables_pd = pd.read_html(StringIO(response.text), attrs=cities_attrs)
print(len(tables_pd), "table(s) matched")

df_cities_pd = tables_pd[0]
print(df_cities_pd.shape)
df_cities_pd.head(3)
"""),
        code("""
# Optional tidy-up: flatten the two-level header and drop the empty spacer row
tidy = df_cities_pd.copy()
tidy.columns = [top if top == bottom else f"{top} {bottom}" for top, bottom in tidy.columns]
tidy = tidy.dropna(how="all").reset_index(drop=True)
print(tidy.shape)
tidy[["Municipality", "ST", "2025 estimate", "2020 land area mi2"]].head(3)
"""),
        md("""
**What the code does.**
- `StringIO(response.text)` makes the string look like a file. Recent versions of pandas want HTML text passed
  this way (a bare string triggers a deprecation warning or is treated as a path).
- `attrs=cities_attrs` — the same dictionary as part 5 — so only the cities table is parsed. `match="Municipality"`
  (keep tables whose text contains that word) is an alternative.
- `[0]` takes the single data frame out of the list.
- The table has a two-row header, so pandas builds **MultiIndex columns** such as
  `('2020 land area', 'mi2')`. The tidy-up joins the two levels into one name.

**Expected output.** 1 table matched; shape **(349, 10)** — all ten columns, with numbers already converted. The
extra row (349 vs 348) is the empty spacer row, read as all-`NaN`; after `dropna(how="all")` it is 348.

**Points to make in class.** `read_html` is three lines instead of thirty — *when the data is a `<table>`*. But it
returns only the visible text: the names still carry their footnote markers (`New York[c]`), and links (`href`)
are lost. Beautiful Soup is for when you need something `read_html` does not give you, or when the data is not in
a table at all (the countries page in Topic 5.2).

**Common mistakes.**
- `pd.read_html(url)` directly → **`HTTPError: 403 Forbidden`**: pandas fetches the page itself *without* our
  `User-Agent` header. Download with `requests`, then pass the text.
- Forgetting that the result is a list: `tables_pd.head()` → `AttributeError: 'list' object has no attribute 'head'`.
- No `attrs`/`match` → a list of 10 data frames and guesswork about which index is right.
- `ImportError: lxml not found` on a fresh install — `pip install lxml` (or pass `flavor="bs4"`).
"""),
    ],
    # Hockey 1
    [
        md("""
### Solution (hockey 1) — the first page

**Approach.** Exactly parts 1 – 2 again: request, check, parse. This site is built for scraping practice, so no
special header is needed (sending one does no harm).
"""),
        code("""
base_url = "https://www.scrapethissite.com"

response = requests.get(base_url + "/pages/forms/")
print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")
soup.title.text.strip()
"""),
        md("""
**What the code does.** Same two steps as before. The site's address is kept in `base_url` because the page links
collected later are *relative* (`/pages/forms/?page_num=2`) and will need it in front. The variable is named
`soup` again because the activity's pagination cell further down expects that name.

**Expected output.** `200` and the title *Hockey Teams: Forms, Searching and Pagination | Scrape This Site …*.

**Common mistakes.** Re-using the Wikipedia `url` variable by accident; forgetting `.text`.
"""),
    ],
    # Hockey 2
    [
        md("""
### Solution (hockey 2) — the main table

**Approach.** Count the tables first. If there is only one, `find` is all we need.
"""),
        code("""
print(len(soup.find_all("table")), "table on the page")

table = soup.find("table")
table.attrs
"""),
        md("""
**What the code does.** `find_all` confirms the page has a single table; `find("table")` returns it as one tag.
Its only attribute is `class="table"`, so `soup.find("table", attrs={"class": "table"})` is an equivalent, more
explicit, way to write it.

**Expected output.** 1 table; `{'class': ['table']}`.

**Common mistakes.** `table = soup.find_all("table")` (no `[0]`) → `table` is a *list*, and the next step's
`table.find_all("tr")` raises `AttributeError: ResultSet object has no attribute 'find_all'` — Beautiful Soup's
message even suggests you probably treated a list of elements like a single element.
"""),
    ],
    # Hockey 3
    [
        md("""
### Solution (hockey 3) — the table as a data frame

**Approach.** Inspect a row: team rows are `<tr class="team">` and the header row has no class — so asking for
`class="team"` rows skips the header automatically. Column names come from the `<th>` cells. Solve one row,
then loop.
"""),
        code("""
header = [th.text.strip() for th in table.find_all("th")]
print(header)

team_rows = table.find_all("tr", attrs={"class": "team"})
print(len(team_rows), "team rows on this page")
[td.text.strip() for td in team_rows[0].find_all("td")]            # one row first
"""),
        code("""
records = []
for team in team_rows:
    records.append([td.text.strip() for td in team.find_all("td")])

df_page1 = pd.DataFrame(records, columns=header)
df_page1.head()
"""),
        md("""
**What the code does.**
- `[th.text.strip() for th in ...]` is a **list comprehension** — a one-line loop that builds a list. Here it
  collects the nine header labels.
- `find_all("tr", attrs={"class": "team"})` keeps only the data rows. This is sturdier than slicing `[1:]`,
  because it selects rows by what they *are* rather than where they sit — which matters in part 4.
- **`.strip()` is essential on this site**: each cell's text is surrounded by newlines and indentation
  (`'\\n        Boston Bruins\\n    '`).
- Each row becomes a list of nine strings; `columns=header` names them.

**Expected output.** 25 rows × 9 columns: *Team Name, Year, Wins, Losses, OT Losses, Win %, Goals For (GF), Goals
Against (GA), + / -*. First row: Boston Bruins, 1990, 44 wins, 24 losses.

**Common mistakes.**
- No `.strip()` → every value wrapped in whitespace; later `df["Team Name"] == "Boston Bruins"` matches nothing.
- Looping over all `<tr>` without skipping the header → one row of `None`/empty values, or a column-count
  mismatch error.
- All columns are still **text** at this point. Conversion is done once, after all pages are collected (next part).
- The `OT Losses` column is **empty** for early seasons (overtime losses were not recorded before 1999–2000), so
  `int(...)` inside the loop raises `ValueError: invalid literal for int() with base 10: ''`. Convert afterwards with
  `pd.to_numeric(..., errors="coerce")`.
"""),
    ],
    # Hockey 4
    [
        md("""
### Solution (hockey 4) — every page

**Approach.** Wrap the page-1 work in a loop over the pagination links collected above: build the full URL →
request → parse → extract the team rows → pause. Collect everything in one list and build the data frame once.
"""),
        code("""
for link in links[:2] + links[-2:]:                    # what do the links look like, first and last?
    print(repr(link.text.strip()), link.attrs["href"], "| aria-label:", link.attrs.get("aria-label"))
"""),
        code("""
import time

# REQUEST + EXTRACT — 24 pages with a half-second pause
records = []
for link in links:
    if link.attrs.get("aria-label") in ("Next", "Previous"):      # arrow buttons repeat a page we already have
        continue
    page_url = base_url + link.attrs["href"]                      # relative link -> full address
    page_soup = BeautifulSoup(requests.get(page_url).text, "html.parser")

    for team in page_soup.find("table").find_all("tr", attrs={"class": "team"}):   # team rows only -> no headers
        records.append([td.text.strip() for td in team.find_all("td")])

    time.sleep(0.5)

len(records)
"""),
        code("""
# PROCESSING — name the columns, convert the numbers, audit
columns = ["team", "year", "wins", "losses", "ot_losses", "win_pct", "goals_for", "goals_against", "goal_diff"]
df_hockey = pd.DataFrame(records, columns=columns)

numeric = columns[1:]
df_hockey[numeric] = df_hockey[numeric].apply(pd.to_numeric, errors="coerce")     # '' -> NaN instead of an error

print(df_hockey.shape, "| duplicated rows:", df_hockey.duplicated().sum())
print(df_hockey["year"].min(), "to", df_hockey["year"].max(), "|", df_hockey["team"].nunique(), "teams")
print(df_hockey.dtypes)
df_hockey.tail()
"""),
        md("""
**What the code does.**
- **Skipping the arrows.** The last link is the "»" (*Next*) button; it carries `aria-label="Next"` and points to
  a page that is already in the list. Without the `continue`, that page is scraped twice. (On page 1 there is no
  "Previous" button, but the test covers it for any starting page.) Alternatives students may find: keep only
  links whose text `.isdigit()`, or `links[:-1]`.
- **`base_url + href`.** The hrefs are relative (`/pages/forms/?page_num=2`); `requests` needs the full address.
- **A different variable, `page_soup`, inside the loop** so the original `soup` (and `links`) are not overwritten
  mid-loop.
- **`class="team"`** filters out each page's header row — the "don't keep repeating headers" technicality —
  without any index arithmetic.
- **One list for all pages**, one `pd.DataFrame` call after the loop.
- **`pd.to_numeric(errors="coerce")`** converts every numeric column at once and turns the blank `OT Losses`
  entries into `NaN` rather than failing.
- The audit line checks the result from three angles: size, duplicates, and the range of years.

**Expected output.** **582 rows × 9 columns**, 0 duplicated rows, seasons **1990 – 2011**, 35 distinct team names,
numeric dtypes (`ot_losses` is `float64` because it contains `NaN`s — 224 of them, all before the 1999 season).

**Common mistakes.**
- **Not skipping "Next"** → 607 rows, 25 of them duplicates. `df.duplicated().sum()` catches it.
- **`requests.get(link.attrs["href"])`** without the base → `MissingSchema: Invalid URL '/pages/forms/?page_num=2'`.
- **Passing the tag instead of its href**: `requests.get(link)` → the same error, less obviously.
- **Overwriting `soup` inside the loop**, then re-running the pagination cell and getting links from page 24.
- **Header rows in the data** (from `find_all("tr")` without the class filter): a row whose "team" is
  `Team Name`, and numeric conversion then fails or coerces a whole row to `NaN`.
- **`int()` inside the loop** dying on the first empty `OT Losses` cell.
- **No `time.sleep`**: this practice site tolerates it, but the habit is what gets students blocked elsewhere.

**The other route the activity mentions.** Instead of harvesting links, generate the URLs from the pattern:
`for page_num in range(1, 25): requests.get(base_url + "/pages/forms/", params={"page_num": page_num})`. It needs
the number of pages in advance (24 — read it off the last numbered link), whereas following the links discovers
it. The site also accepts `per_page=100`, which cuts 24 requests to 6 — a nice illustration that reading the URL
a site uses can save most of the work.
"""),
    ],
]


# ============================================================================ #
#  Build
# ============================================================================ #

def source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def build_solution(src_path: Path, title: str, intro: dict, answers: list[list[dict]],
                   replace: dict[str, list[dict]] | None = None) -> dict:
    nb = json.loads(src_path.read_text())
    out: list[dict] = []
    n = 0
    for cell in nb["cells"]:
        src = source(cell)
        if cell["cell_type"] == "code" and src.strip() == "":
            continue                                                   # stray empty cell in the source
        swap = next((cells for key, cells in (replace or {}).items() if cell["cell_type"] == "code" and key in src), None)
        if swap is not None:
            out.extend(copy.deepcopy(swap))
        elif cell["cell_type"] == "code" and PLACEHOLDER.match(src):
            if n >= len(answers):
                raise SystemExit(f"{src_path.name}: more placeholders than answer groups (at #{n})")
            out.extend(copy.deepcopy(answers[n]))
            n += 1
        else:
            out.append(copy.deepcopy(cell))
    if n != len(answers):
        raise SystemExit(f"{src_path.name}: found {n} placeholders, expected {len(answers)}")
    first_md = next(i for i, c in enumerate(out) if c["cell_type"] == "markdown")
    out[first_md]["source"] = re.sub(r"^#[^\n]*", lambda m: title, source(out[first_md]), count=1)
    out.insert(first_md + 1, copy.deepcopy(intro))
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:03d}"
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = out
    nb["nbformat"], nb["nbformat_minor"] = 4, 5          # cell ids need nbformat 4.5
    return nb


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=900", str(path)], check=True)
    nb = json.loads(path.read_text())
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"execution errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    jobs = [
        (PA51_SRC, PA51_TITLE, PA51_INTRO, PA51_ANSWERS, {"PUT-YOUR-KEY-HERE": PA51_TASTY_SETUP}),
        (PA52_SRC, PA52_TITLE, PA52_INTRO, PA52_ANSWERS, None),
    ]
    for src, title, intro, answers, replace in jobs:
        dest = src.with_name(src.stem + "-solution.ipynb")
        save(dest, build_solution(src, title, intro, answers, replace))
        print(f"wrote {dest.relative_to(ROOT)}")
        if "--no-exec" not in sys.argv:
            execute(dest)


if __name__ == "__main__":
    main()
