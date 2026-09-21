#!/usr/bin/env python3
"""Generate the Week 5 Topic notebooks (student + executed solution).

Usage:  python3 tools/build_week5_topics.py            # write -empty and -solution, execute the solutions
        python3 tools/build_week5_topics.py --no-exec  # write all notebooks without executing

Two 10–15 minute openers, one per practice activity:
  Topic 5.1 — JSON and APIs          -> pairs with PA 5.1 (shows data, TVMaze API, Tasty API)
  Topic 5.2 — HTML and Web Scraping  -> pairs with PA 5.2 (Wikipedia cities table, multi-page hockey stats)
Same markers as the earlier builders:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
Both notebooks make a few live requests (api.tvmaze.com, scrapethissite.com), so executing needs a network connection.
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
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 5.1 — JSON and APIs (pairs with PA 5.1)
# ============================================================================ #

t51: list[dict] = []

md(t51, """
# GSB 5544 — Topic 5.1: JSON and APIs — SOLUTION
*Data that is nested instead of rectangular, how to flatten it, and how to ask a web server for it*
""")

md(t51, """
## The next 15 minutes

| | Question | Where it lands in PA 5.1 |
|---|---|---|
| **1. Read** | What does JSON look like, and how do I reach a value inside it? | everything |
| **2. Flatten** | How does nested data become a data frame? (`pd.json_normalize`) | Shows data 1 – 4, Tasty 5 |
| **3. Request** | How do I ask an API for JSON? (`requests.get`, endpoints, parameters) | TVMaze 1, Tasty 1 |
| **4. Repeat** | How do I make many requests politely? (loop, `extend`, `time.sleep`) | TVMaze 2, Tasty 3 |

Until now every data set arrived as a rectangle: rows and columns. Data from the web usually arrives
**hierarchical** — a show *has* a network, and *has many* episodes, each of which *has* a title. JSON is the
format that holds that shape.
""")

code(t51, """
import pandas as pd
import requests
import time
""")

md(t51, """
---
## 1. JSON is dictionaries and lists, nested

Once loaded into Python, JSON is made of things you already know:

| JSON | Python | Reach inside with |
|---|---|---|
| object `{ "name": "Girls" }` | **dictionary** | a **key**: `d["name"]` |
| array `[ ..., ... ]` | **list** | a **position**: `x[0]` |
| string, number, `true`/`false`, `null` | `str`, `int`/`float`, `True`/`False`, `None` | — |

Two shows, written by hand. Each show is a dictionary; `network` is a dictionary *inside* it;
`episodes` is a *list of dictionaries* inside it.
""")

code(t51, """
shows = [
    {"name": "Derry Girls", "network": {"name": "Channel 4", "country": "GB"},
     "episodes": [{"title": "Episode 1", "runtime": 30}, {"title": "Episode 2", "runtime": 30}]},
    {"name": "Bomb Girls", "network": {"name": "Global", "country": "CA"},
     "episodes": [{"title": "Jumping Tracks", "runtime": 60}]},
]
type(shows), len(shows), shows[0].keys()
""")

md(t51, """
To reach a value, **walk down one level at a time**: list → position, dictionary → key.
""")

code(t51, """
shows[0]["network"]["name"]          # first show -> its network -> the network's name
""")

code(t51, """
shows[«1»][«"episodes"»][«0»]["title"]     # the title of Bomb Girls' first episode
""")

md(t51, """
✅ (a) Is `shows` a list or a dictionary? What about `shows[0]`? And `shows[0]["episodes"]`?
(b) Why does `shows["name"]` raise an error?
""")

answer(t51, """
(a) `shows` is a **list** (of two shows); `shows[0]` is a **dictionary** (one show); `shows[0]["episodes"]` is
a **list** again (of episode dictionaries). (b) `shows` is a list, and lists take positions, not keys — you
must pick a show first: `shows[0]["name"]`. When you are lost in real JSON, ask `type(...)` and, for a
dictionary, `.keys()`.
""")

md(t51, """
---
## 2. Flattening: `pd.json_normalize`

**One row per show.** Nested *dictionaries* become dotted column names. Nested *lists* cannot fit in one cell
of a rectangle, so they are left as lists.
""")

code(t51, """
df_shows = pd.«json_normalize»(shows)
df_shows
""")

md(t51, """
**One row per episode.** To open up a list, name it as the `record_path`. Each element of that list becomes
a row. Use `meta` to carry down fields from the parent so you know which show each episode belongs to.
""")

code(t51, """
df_episodes = pd.json_normalize(shows,
                                record_path=«"episodes"»,     # the list to unpack: one row per element
                                meta=[«"name"»])              # parent fields to keep on each row
df_episodes
""")

md(t51, """
✅ (a) `df_shows` has ____ rows and `df_episodes` has ____ rows — what decides each number? (b) Which data
frame would you use to count shows per network? Which to find the average episode runtime per show?
""")

answer(t51, """
(a) 2 rows (one per **show**) and 3 rows (one per **episode**: 2 + 1). The `record_path` decides what a row
*is*. (b) Shows per network → `df_shows` (`df_shows["network.name"].value_counts()`); runtime per show →
`df_episodes` (`groupby("name")["runtime"].mean()`). **Choosing the unit of the row is the key decision** —
after that it is ordinary pandas. If the list is two levels down, give the path as a list:
`record_path=["seasons", "episodes"]`.
""")

md(t51, """
---
## 3. Asking an API for JSON

An **API** is a web address built for programs rather than people: you send a request, it sends back JSON.
A request has three parts:

| Part | Example | |
|---|---|---|
| **base URL** | `https://api.tvmaze.com` | the service |
| **endpoint** | `/search/shows` | *which* kind of data (listed in the API's documentation) |
| **parameters** | `q=golden girls` | the details of *this* question; joined to the URL after a `?` |

`requests.get` sends it. Pass parameters as a dictionary and `requests` builds the `?q=...` part for you.
""")

code(t51, """
response = requests.«get»("https://api.tvmaze.com/search/shows", params={"q": "golden girls"})
response.url, response.status_code
""")

md(t51, """
`status_code` 200 means success (404 = no such page, 401/403 = not authorised, 429 = slow down). Always
look at it before trusting the data. `.json()` converts the reply into Python lists and dictionaries:
""")

code(t51, """
results = response.«json»()
type(results), len(results), results[0].keys()
""")

code(t51, """
pd.json_normalize(results)[["score", "show.id", "show.name", "show.premiered", "show.network.name"]]
""")

md(t51, """
✅ (a) Each element of `results` has two keys, `score` and `show`. Why do the column names start with
`show.`? (b) What would you change to search for *"powerpuff"* instead?
""")

answer(t51, """
(a) The show's details are a dictionary nested under the key `show`, and `json_normalize` names nested
fields `parent.child` — hence `show.name`, and two levels down, `show.network.name`. (b) Only the parameter:
`params={"q": "powerpuff"}`. The base URL and endpoint stay the same — that is the whole point of an endpoint.
""")

md(t51, """
---
## 4. Many requests: loop, collect, pause

Some questions need one request per item (the cast of *each* show) or per page of results. The pattern:

1. start an **empty list**;
2. **loop**, making one request per item;
3. add each reply to the list with **`.extend(...)`** (adds the *elements* of the reply — `.append` would add
   the whole reply as one element);
4. **`time.sleep(...)`** between requests, so the server does not block you;
5. flatten **once, after** the loop — in a *separate cell*, so you never re-request just to fix your pandas.
""")

code(t51, """
show_ids = [722, 33320]              # The Golden Girls, Derry Girls

cast = []
for show_id in show_ids:
    response = requests.get(f"https://api.tvmaze.com/shows/{show_id}/cast")
    cast.«extend»(response.json())
    time.«sleep»(0.5)                  # be polite: half a second between requests

len(cast)
""")

code(t51, """
pd.json_normalize(cast)[["person.name", "character.name"]].head()
""")

md(t51, """
✅ After this loop you cannot tell which show each cast member came from. Why not — and what one line inside
the loop would fix it?
""")

answer(t51, """
The cast endpoint returns people and characters but not the show, and `extend` pours every reply into one
list. Tag each record before adding it — for example
`members = response.json()`, then `for m in members: m["show_id"] = show_id`, then `cast.extend(members)`.
You will need exactly this in PA 5.1 (TVMaze part 2).
""")

md(t51, """
---
## 5. APIs that need a key

Some APIs (the Tasty API in the PA) only answer registered users. You sign up, receive an **API key**, and
send it in the request's **headers** — extra information that travels with the request but is not part of
the URL:

```python
headers = {"X-RapidAPI-Key": "your-key-here", "X-RapidAPI-Host": "tasty.p.rapidapi.com"}
response = requests.get(url, headers=headers, params={"q": "daikon"})
```

A key is a password: it counts *your* requests against *your* quota. Do not post it, and do not leave it in a
notebook you share or push to GitHub.

## The four lines to keep

| | |
|---|---|
| **Read** | JSON = dictionaries (`["key"]`) and lists (`[0]`), nested; when lost, `type()` and `.keys()` |
| **Flatten** | `pd.json_normalize(data)` = one row per top-level item; `record_path=` opens a list, `meta=` keeps the parent's fields |
| **Request** | `requests.get(url, params={...})` → check `.status_code` → `.json()` |
| **Repeat** | empty list → loop → `.extend()` → `time.sleep()`; request in one cell, process in another |

PA 5.1 is on the course site: [%s](%s).
""" % (SITE, SITE))


# ============================================================================ #
#  Topic 5.2 — HTML and Web Scraping (pairs with PA 5.2)
# ============================================================================ #

t52: list[dict] = []

md(t52, """
# GSB 5544 — Topic 5.2: HTML and Web Scraping — SOLUTION
*When there is no API: reading the web page itself, finding the tags that hold the data, and looping over them*
""")

md(t52, """
## The next 15 minutes

| | Question | Where it lands in PA 5.2 |
|---|---|---|
| **1. Read** | What is HTML made of? (tags, attributes, text) | parts 3 – 5 |
| **2. Find** | How do I locate the tags I want? (`find`, `find_all`, `attrs`) | parts 2 – 11 |
| **3. Collect** | How do rows of a web table become a data frame? (the loop) | part 12, hockey 3 |
| **4. Scale** | Real pages, `pd.read_html`, and many pages | part 1, part 13, hockey 4 |

An API hands you clean JSON. Most websites do not have one — the data is only *on the page*, wrapped in
HTML meant for a browser. **Scraping** is pulling the data back out of that HTML.
""")

code(t52, """
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
""")

md(t52, """
---
## 1. HTML is a tree of tags

Three things to recognise:

- a **tag** marks a piece of the page: `<td> ... </td>`. Tags sit inside other tags — a tree.
- **attributes** are labels on the opening tag: `<table class="stats">`, `<a href="/page2">`.
- the **text** is what sits between the opening and closing tag.

The tags that matter for tables: `<table>` › `<tr>` (a **row**) › `<th>` (a **header** cell) or `<td>` (a
**data** cell). And for links: `<a href="...">`. A small page, as a string:
""")

code(t52, """
html = \"\"\"
<html><body>
  <h1>Central Coast cities</h1>
  <table class="notes"><tr><td>Populations are estimates.</td></tr></table>
  <table class="stats" id="cities">
    <tr><th>City</th><th>County</th><th>Population</th></tr>
    <tr><td><a href="/wiki/SLO">San Luis Obispo</a></td><td>SLO</td><td>47,063</td></tr>
    <tr><td><a href="/wiki/Santa_Maria">Santa Maria</a>[a]</td><td>SB</td><td>109,707</td></tr>
    <tr><td><a href="/wiki/Paso_Robles">Paso Robles</a></td><td>SLO</td><td>31,490</td></tr>
  </table>
  <ul class="pagination"><li><a href="/cities?page=1">1</a></li><li><a href="/cities?page=2">2</a></li></ul>
</body></html>
\"\"\"
""")

md(t52, """
---
## 2. Beautiful Soup: parse, then find

`BeautifulSoup` turns the string into a tree you can search. Two search methods do almost everything:

| | returns | use when |
|---|---|---|
| `soup.find("tag")` | the **first** matching tag (or `None`) | you want one thing |
| `soup.find_all("tag")` | a **list** of every matching tag | you want to count or loop |

Both accept `attrs={...}` to narrow the search by attribute.
""")

code(t52, """
soup = «BeautifulSoup»(html, "html.parser")
len(soup.«find_all»("table"))                      # how many tables on the page?
""")

code(t52, """
table = soup.find("table", attrs={«"class"»: «"stats"»})     # the one we want, picked out by its attribute
table.attrs
""")

md(t52, """
From a tag you can pull out its **text** (`.text`) and its **attributes** (`.attrs["href"]`). A search
can start from *any* tag, not only from `soup` — `table.find_all("tr")` looks only inside that table.
""")

code(t52, """
rows = table.«find_all»("tr")
first_city = rows[1]                     # rows[0] is the header row
cells = first_city.find_all(«"td"»)
cells[0].text, cells[2].text, cells[0].find("a").attrs[«"href"»]
""")

md(t52, """
✅ (a) Why `rows[1]` rather than `rows[0]`? What would `rows[0].find_all("td")` return? (b) For Santa
Maria, `cells[0].text` is `'Santa Maria[a]'`. How does `cells[0].find("a").text` avoid the footnote?
""")

answer(t52, """
(a) `rows[0]` is the header row: its cells are `<th>`, not `<td>`, so `find_all("td")` returns an **empty
list** — and `[0]` on an empty list is the `IndexError` you will meet most often when scraping.
(b) `.text` of the whole cell gathers *all* text inside it, including the footnote marker; the `<a>` tag
contains only the city name, so searching one level deeper gives clean text. Wikipedia's city table has
exactly this issue.
""")

md(t52, """
---
## 3. From rows to a data frame: the loop

Work out the extraction for **one** row, then put it in a loop: empty list → one dictionary per row →
`pd.DataFrame`. Scraped values are always **text**, so clean and convert the numbers.
""")

code(t52, """
records = []
for row in table.find_all("tr")[«1:»]:                       # skip the header row
    cells = row.find_all("td")
    records.append({
        "city": cells[0].find("a").text,
        "county": cells[1].text,
        "population": int(cells[2].text.«replace»(",", "")),   # "47,063" -> 47063
    })

df = pd.«DataFrame»(records)
df
""")

md(t52, """
✅ Without the `.replace(",", "")`, what would `int("47,063")` do? And if you skipped `int(...)` entirely, what
would `df["population"].sum()` give you?
""")

answer(t52, """
`int("47,063")` raises a `ValueError` — the comma is not part of a number. Without `int` the column stays
text, and summing text **concatenates** it (`'47,063109,70731,490'`). Check `df.dtypes` after every scrape.
""")

md(t52, """
---
## 4. Real pages

**Get the HTML** with `requests`, exactly as with an API — but the reply is a page, so use `.text`, not
`.json()`. **Find the right tags** with your browser: right-click the thing you want → *Inspect*, and read off
the tag and its attributes.

Two habits: check `status_code` (some sites, including Wikipedia, refuse requests that do not identify
themselves — send a `User-Agent` header), and scrape politely: look at the site's terms, and pause between
requests.
""")

code(t52, """
headers = {"User-Agent": "GSB5544 class exercise"}
response = requests.get("https://www.scrapethissite.com/pages/simple/", headers=headers)
response.«status_code»
""")

md(t52, """
Inspecting that page shows each country in `<div class="col-md-4 country">`, with the name in
`<h3 class="country-name">` and the capital in `<span class="country-capital">`. Matching on one class
(`"country"`) is enough.
""")

code(t52, """
soup = BeautifulSoup(response.«text», "html.parser")
countries = soup.find_all("div", attrs={"class": "country"})

records = []
for country in countries:
    records.append({
        "name": country.find("h3", attrs={"class": "country-name"}).text.«strip»(),
        "capital": country.find("span", attrs={"class": "country-capital"}).text,
        "population": int(country.find("span", attrs={"class": "country-population"}).text),
    })

pd.DataFrame(records).head()
""")

md(t52, """
✅ Why `.strip()` on the name but not on the capital? (Look at the page source, or try it without.)
""")

answer(t52, """
In the source the name sits on its own line inside the `<h3>`, surrounded by line breaks and spaces, so
`.text` comes back as `'\\n   Andorra\\n   '`. The capital is written tightly inside its `<span>`. When in
doubt, `.strip()` — it never hurts.
""")

md(t52, """
**Shortcut for tables.** `pd.read_html` finds `<table>` tags and converts them for you. It returns a **list**
of data frames (one per table), so narrow it with `attrs=` or `match=` and then pick with `[0]`. Pass it HTML
you downloaded yourself, wrapped in `StringIO`:
""")

code(t52, """
from io import StringIO

tables = pd.«read_html»(StringIO(html), attrs={"id": "cities"})
len(tables), tables[0]
""")

md(t52, """
**Many pages.** When the data is spread over pages, collect the page links (`<a>` tags in the pagination
list), then loop: request → parse → extract → `time.sleep`. It is Topic 5.1's request loop with scraping
inside.
""")

code(t52, """
pagination = BeautifulSoup(html, "html.parser").find("ul", attrs={"class": "pagination"})
[link.attrs["href"] for link in pagination.find_all(«"a"»)]
""")

md(t52, """
✅ These `href`s start with `/`. What must you add before passing one to `requests.get`?
""")

answer(t52, """
They are **relative** links — relative to the site. `requests` needs the full address, so join the site's
base URL to it: `"https://example.com" + link.attrs["href"]`. In PA 5.2 the base is
`https://www.scrapethissite.com`.
""")

md(t52, """
## The four lines to keep

| | |
|---|---|
| **Read** | HTML = tags in a tree, with attributes and text; tables are `table › tr › th/td`; links are `a` with `href` |
| **Find** | `BeautifulSoup(html, "html.parser")`; `find` = first, `find_all` = list; narrow with `attrs={...}`; then `.text`, `.attrs[...]` |
| **Collect** | solve one row → loop → list of dictionaries → `pd.DataFrame`; skip header rows; clean and convert numbers |
| **Scale** | `requests.get(url, headers=...)` + check `status_code`; `pd.read_html` for whole tables; loop over page links with `time.sleep` |

PA 5.2 is on the course site: [%s](%s).
""" % (SITE, SITE))


# ============================================================================ #
#  Build
# ============================================================================ #

BLANK = re.compile(r"«(.*?)»", re.S)
METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def notebook(cell_list: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cell_list):
        c = copy.deepcopy(c)
        c["id"] = f"cell-{i:03d}"
        out.append(c)
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def solution_cells(cells: list[dict]) -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub(lambda m: m.group(1), c["source"])
        out.append(c)
    return out


def student_cells(cells: list[dict], student_title: str) -> list[dict]:
    out = []
    for c in cells:
        c = copy.deepcopy(c)
        if c["cell_type"] == "code":
            c["source"] = BLANK.sub("____", c["source"])
        elif c["source"].startswith("**Answer:**"):
            c["source"] = "**Your answer:** *(write it here — replace this line)*"
        out.append(c)
    out[0]["source"] = student_title
    return out


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def execute(path: Path) -> None:
    subprocess.run([JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=600", str(path)], check=True)
    nb = json.loads(path.read_text())
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"execution errors: {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


TOPICS = [
    (t51, WEEK / "GSB5544_Topic_5_1_JSON_and_APIs",
     "# GSB 5544 — Topic 5.1: JSON and APIs  \n"
     "*Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
    (t52, WEEK / "GSB5544_Topic_5_2_HTML_and_Web_Scraping",
     "# GSB 5544 — Topic 5.2: HTML and Web Scraping  \n"
     "*Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
]


def main() -> None:
    for cells, stem, student_title in TOPICS:
        student = stem.with_name(stem.name + "-empty.ipynb")
        solution = stem.with_name(stem.name + "-solution.ipynb")
        save(student, notebook(student_cells(cells, student_title)))
        save(solution, notebook(solution_cells(cells)))
        print(f"wrote {student.relative_to(ROOT)} and {solution.relative_to(ROOT)} ({len(cells)} cells)")
        if "--no-exec" not in sys.argv:
            execute(solution)


if __name__ == "__main__":
    main()
