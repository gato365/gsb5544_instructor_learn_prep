#!/usr/bin/env python3
"""Generate the Week 4 Topic notebooks (student + executed solution).

Usage:  python3 tools/build_week4_topics.py            # write -empty and -solution, execute the solutions
        python3 tools/build_week4_topics.py --no-exec  # write all notebooks without executing

Two 15–20 minute openers, one per practice activity:
  Topic 4.1 — Text as Data (bag of words, TF, TF-IDF, cosine distance) -> pairs with PA 4.1 (Enron spam)
  Topic 4.2 — Strings and Regular Expressions                          -> pairs with PA 4.2 (Decode a Message)
Same markers as the Week 3 builders:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
Topic 4.1 reads the Enron sample from the public data301 URL, so executing needs a network connection.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_4"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"

ENRON_URL = "https://raw.githubusercontent.com/kevindavisross/data301/refs/heads/main/data/enron_email.csv"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 4.1 — Text as Data (pairs with PA 4.1)
# ============================================================================ #

t41: list[dict] = []

md(t41, """
# GSB 5544 — Topic 4.1: Text as Data — SOLUTION
*How a pile of emails becomes a table of numbers: bag of words, TF, TF-IDF, and cosine distance*
""")

md(t41, """
## The next 20 minutes

| | Question | Where it lands in PA 4.1 |
|---|---|---|
| **a. Represent** | How do we turn free text into rows and columns? | parts 1–3 |
| **b. Weight** | Why downweight words that appear everywhere? (TF-IDF) | parts 4–5 |
| **c. Compare** | How do we measure "these two documents are similar"? | parts 6–11 |

Week 3 asked *which rows of a table are most alike?* and answered it with a **distance**.
This week the observations are **emails** — no columns at all, just text. The whole trick of
text analysis is: **build the columns yourself**, then everything from Week 3 works again.
""")

code(t41, """
import pandas as pd
import numpy as np
""")

md(t41, """
---
## 1. The vocabulary of text analysis

Three words you will see in every reading on this topic:

| Term | Meaning | In PA 4.1 |
|---|---|---|
| **document** | one unit of text — one email, one review, one tweet | one email body |
| **corpus** | the collection of documents | the `body` column |
| **bag of words** | represent a document by *which words it uses and how often*, ignoring word order | what `CountVectorizer` builds |

"Bag" is the honest part of the name: *"the dog bit the man"* and *"the man bit the dog"*
become the **same** bag. We throw away order and keep counts — a crude summary that turns
out to be surprisingly powerful for questions like "is this spam?"
""")

md(t41, """
---
## 2. The term-frequency (TF) matrix — by hand first

A tiny corpus of four "emails". Before running anything: what should the columns of our
table be, and what number goes in each cell?
""")

code(t41, """
corpus = pd.Series([
    "Win a FREE prize now!!",
    "Meeting moved to noon.",
    "Free free FREE — claim your prize",
    "Can we move the meeting?",
])
corpus
""")

md(t41, """
**The recipe** (this is the answer to PA 4.1 part 2):

1. **Normalize** — lowercase everything, so `Free`, `free`, and `FREE` are one word.
2. **Tokenize** — split each document into words (*tokens*), dropping punctuation.
3. **Build the vocabulary** — the set of distinct words across the whole corpus. These are the **columns**.
4. **Count** — cell (i, j) = how many times word j appears in document i. These counts are the **term frequencies (TF)**.

The result is called the **term-frequency matrix** (or *document-term matrix*):
one row per document, one column per vocabulary word. Text is now a table, and
pandas is back in business.

`scikit-learn` wraps all four steps in one object, `CountVectorizer`:
""")

code(t41, """
from sklearn.feature_extraction.text import «CountVectorizer»

vec = «CountVectorizer»()
tf = vec.«fit_transform»(corpus)               # learn the vocabulary, then count
tf_df = pd.DataFrame(tf.«toarray»(),           # sparse matrix -> regular array
                     columns=vec.«get_feature_names_out»())
tf_df
""")

md(t41, """
Read one row aloud to check it: document 2 (*"Free free FREE — claim your prize"*)
has `free = 3`, `prize = 1`, `claim = 1`, and 0 everywhere else. Word order is gone;
the counts remain.

✅ **Check:** why does document 0's row show `win = 1` and not `Win = 1`? And why is
`fit_transform` one step, not two separate calls here?
""")

answer(t41, """
`CountVectorizer` lowercases before tokenizing (step 1 of the recipe), so `Win` and `win`
are the same column. `fit` learns the vocabulary from the corpus, `transform` counts each
document against that vocabulary; `fit_transform` does both on the same corpus in one call —
exactly what we want when building the matrix for the corpus we learned the vocabulary from.
""")

md(t41, """
---
## 3. The problem with raw counts — and the TF-IDF fix

Look down the columns of a real TF matrix and two kinds of words appear:

- **Words that show up in almost every document** — `the`, `to`, `and`. Big counts, zero information: knowing an email contains `the` tells you nothing about it.
- **Words that show up in a few documents** — `prize`, `viagra`, `invoice`. Small counts, big signal.

Raw TF rewards exactly the wrong words. The fix weights each column by how *rare* the word is
across the corpus:

| Quantity | Definition | Intuition |
|---|---|---|
| **TF** — term frequency | count of word *j* in document *i* | "how much does *this* document use the word?" |
| **DF** — document frequency | number (or fraction) of documents containing word *j* | "how common is the word across the corpus?" |
| **IDF** — inverse document frequency | roughly `log(n_documents / DF)` | big for rare words, near 0 for words in every document |
| **TF-IDF** | TF × IDF | large only when a word is *frequent in this document* **and** *rare overall* |

A word that appears in **every** document has IDF ≈ log(1) = 0 — its column is wiped out,
no matter how large its counts. `TfidfVectorizer` is a drop-in replacement for `CountVectorizer`:
""")

code(t41, """
from sklearn.feature_extraction.text import «TfidfVectorizer»

tfidf_vec = «TfidfVectorizer»()
tfidf_df = pd.DataFrame(tfidf_vec.«fit_transform»(corpus).toarray(),
                        columns=tfidf_vec.get_feature_names_out())
tfidf_df.round(2)
""")

md(t41, """
✅ **Check:** look at row 0. In the TF matrix, `win`, `free`, and `prize` all had count 1 —
yet in the TF-IDF matrix `win` scores **higher** than the other two. Why?
""")

answer(t41, """
Equal TF, different DF: `free` and `prize` each appear in **2** of the 4 documents, while `win`
appears in only **1** — so `win` has the larger IDF and wins the tiebreak. Same use *within*
the document, but rarer *across* the corpus, so more distinctive. On a real corpus the effect
is dramatic: `the` appears in essentially every email (DF ≈ n), its IDF is driven toward the
floor, and its huge counts are wiped out — while `prize` keeps its weight. That is why we
reach for TF-IDF rather than raw TF when we care about *distinctive* words (PA 4.1 part 4).
""")

md(t41, """
---
## 4. Comparing documents: cosine similarity, not Euclidean distance

Each email is now a row of numbers — a **vector**. Week 3's instinct says: Euclidean distance.
For text there is a catch: **document length**. A 2,000-word email uses every one of its words
more often than a 50-word email about the *same topic*, so Euclidean distance calls them far apart
just because one is longer.

The fix: compare the **direction** of the two vectors, not their length.

$$\\text{cosine similarity}(x, y) = \\frac{x \\cdot y}{\\lVert x \\rVert \\, \\lVert y \\rVert}
\\qquad\\qquad \\text{cosine distance} = 1 - \\text{cosine similarity}$$

| Value | Meaning |
|---|---|
| similarity ≈ 1 (distance ≈ 0) | same mix of words in the same proportions |
| similarity ≈ 0 (distance ≈ 1) | no overlapping words at all (counts can't be negative, so ≈ 0 is the floor here) |

Scaling a document up — writing the same email twice as long — doesn't change its direction,
so cosine similarity ignores length and keeps topic. That is exactly what we want.
""")

code(t41, """
from sklearn.metrics.pairwise import «cosine_similarity»

sim = «cosine_similarity»(tf_df)                       # 4×4: every document vs every document
pd.DataFrame(sim, columns=corpus.str[:20], index=corpus.str[:20]).round(2)
""")

md(t41, """
✅ **Check:** which pair of documents is most similar, and does that match your reading of the
four sentences? Which document is *"Can we move the meeting?"* closest to?
""")

answer(t41, """
Documents 0 and 2 (the two prize/free spam-style messages) have the highest off-diagonal
similarity — they share `free` and `prize`. *"Can we move the meeting?"* is closest to
*"Meeting moved to noon."*: they share meeting words. Note `moved` vs `move` are different
tokens — bag of words does no stemming — yet the shared vocabulary still links them.
""")

md(t41, """
---
## 5. The payoff: classify by "who are your neighbours?"

Now stack the pieces exactly the way PA 4.1 will:

1. Corpus → **TF (or TF-IDF) matrix** — text becomes a table.
2. A new email arrives with **unknown** label.
3. Compute the **cosine distance** from the new email to every email whose label we know.
4. **Sort.** Look at the closest emails — if the nearest neighbours are spam, bet spam.

That is nearest-neighbour classification, built entirely from this notebook's parts.
A dry run on the real PA data (a sample of the Enron corpus — read the PA intro for the story):
""")

code(t41, """
emails = pd.read_csv("%s")
emails["spam"].value_counts(dropna=False)
""" % ENRON_URL)

md(t41, """
1,980 emails labelled spam (1) or not (0) — and **10 emails with no label**. PA 4.1's job is to
guess those 10 labels using nearest neighbours. One caution before you start: a few `body`
entries are missing, and `CountVectorizer` refuses `NaN` documents — `.fillna("")` first.
""")

code(t41, """
corpus = emails["body"].«fillna("")»               # CountVectorizer can't digest NaN
tf_real = CountVectorizer().fit_transform(corpus)
tf_real.shape
""")

md(t41, """
Nearly 35,000 columns — one per distinct word in 1,990 emails. You will never look at this
matrix directly; you compute with it.

## The three lines to keep

| | |
|---|---|
| **Represent** | `CountVectorizer()` / `TfidfVectorizer()` + `fit_transform(corpus)` → documents become rows, words become columns |
| **Weight** | TF-IDF = TF × IDF — downweights words that appear in most documents, keeps distinctive ones |
| **Compare** | `cosine_similarity(...)`; distance = 1 − similarity; sort and read the nearest neighbours |

PA 4.1 is on the course site: [%s](%s).
""" % (SITE, SITE))


# ============================================================================ #
#  Topic 4.2 — Strings and Regular Expressions (pairs with PA 4.2)
# ============================================================================ #

t42: list[dict] = []

md(t42, """
# GSB 5544 — Topic 4.2: Strings and Regular Expressions — SOLUTION
*The string methods you will actually use, when to reach for a regex, and how to wield `.str` on a pandas Series*
""")

md(t42, """
## The next 20 minutes

| | Question | Where it lands in PA 4.2 |
|---|---|---|
| **a. Strings** | What can plain Python do to one string? | warm-ups 1–5 |
| **b. Patterns** | How do I describe "words like this" instead of one exact word? | warm-up 6, decode 2–7 |
| **c. Series** | How do the same ideas apply to a whole column at once? | everything |

Topic 4.1 crushed text into counts. This topic is the opposite skill: **surgical edits** —
trim this, replace that, keep words matching a pattern. PA 4.2 hands you a scrambled movie
quote; every cleaning step is one of the tools below.
""")

code(t42, """
import pandas as pd
import re
""")

md(t42, """
---
## 1. One string: the core methods

A string is a **sequence of characters**, so `len`, indexing, and slicing all work.
Beyond that, six methods cover most real cleaning work:

| Method | What it does |
|---|---|
| `s.lower()` / `s.upper()` | change case |
| `s.strip()` | remove whitespace from **both ends** (not the middle) |
| `s.replace(old, new)` | replace every occurrence, exact text only |
| `s.split(sep)` | string → list of pieces |
| `sep.join(list_of_strings)` | list of pieces → one string (note: called **on the separator**) |
| `s.startswith(x)` / `s.endswith(x)` | True/False tests |
""")

code(t42, """
s = "   The DUDE abides.   "
(
    «len(s)»,             # length counts the spaces too
    s.«strip()»,          # trim the ends
    s.strip().«lower()»,  # methods chain left to right
    s.strip().«split(" ")»
)
""")

code(t42, """
"-".join(["some", "assembly", "required"])     # join is a method of the SEPARATOR
""")

md(t42, """
✅ **Check:** strings are *immutable* — `s.strip()` returns a **new** string and leaves `s`
alone. What does `s` contain after running the cell above? What would you write to actually
update it?
""")

answer(t42, """
`s` still contains `"   The DUDE abides.   "` — no string method modifies the original.
To keep a change you must reassign: `s = s.strip()`. Forgetting the reassignment (or, in
pandas, forgetting `message = message.str.strip()`) is the single most common bug in PA 4.2.
""")

md(t42, """
---
## 2. When exact text isn't enough: regular expressions

`replace("ugh!", "")` removes exactly `ugh!` — but what about `ughhh?` and `ughhhhh,`?
You cannot list every variant. A **regular expression** (regex) describes the whole *family*
of strings with a pattern. The pieces you need this week:

| Pattern | Matches | Example |
|---|---|---|
| `abc` | those literal characters | `ugh` |
| `[aeiou]` | any **one** character from the set | |
| `[^ ...]` | any one character **not** in the set | `[^\\w\\s]` = not word, not space → **punctuation** |
| `\\w` `\\s` `.` | word character; whitespace; *anything* | |
| `x+` `x*` `x{2}` | one-or-more; zero-or-more; exactly 2 | `ugh+` = `ugh`, `ughh`, `ughhh`, … |
| `^x` `x$` | at the **start** / at the **end** of the string | `^k` = starts with k; `b$` = ends with b |

The two functions from the `re` module you will use:

- `re.findall(pattern, s)` — list of every non-overlapping match
- `re.sub(pattern, replacement, s)` — replace every match
""")

code(t42, """
groan = "Sighugh! Ughhh, fine. That was rough — very rough, ughhhhh."
re.«findall»(r"ugh+", groan)             # one-or-more h's — catches every groan length
""")

code(t42, """
re.«sub»(r"[Uu]gh+[^\\w\\s]", "", groan)   # groan + trailing punctuation, gone
""")

md(t42, """
Two habits to keep: write patterns as **raw strings** (`r"..."`) so backslashes survive, and
test with `findall` *before* you `sub` — see what you are about to destroy.

✅ **Check:** compare the two outputs to the original sentence. Three surprises to explain:
`findall(r"ugh+")` **missed** `Ughhh,` entirely but matched **inside** `rough` (twice) —
and after the `sub`, the first `rough` survived while the second became `ro`. Why, why, and why?
""")

answer(t42, """
(1) Regex is **case-sensitive**: lowercase `u` in the pattern cannot match the capital in
`Ughhh` — hence `[Uu]` in the `sub` pattern (and PA 4.2's hint "look out for capitalization!").
(2) A pattern matches **anywhere** in the string, not whole words — regex doesn't know what a
word is unless you tell it, so the `ugh` hiding inside `rough` matches too.
(3) The `sub` pattern requires a trailing **punctuation** character: `rough —` has a space
after `rough`, so no match — but `rough,` ends in a comma, so its `ugh,` was deleted, leaving
`ro`. Patterns do exactly what they say, not what you meant; that is why PA 4.2 warns
"look out for punctuation!", and why anchors (`^`, `$`) are how you pin a pattern to the
start or end of a word.
""")

md(t42, """
---
## 3. A whole column at once: the `.str` accessor

PA 4.2's `message` is a **pandas Series** of strings, not one string. Two ways to apply
string tools to every element:

1. **`.str` accessor** — most string methods, vectorized: `s.str.strip()`, `s.str.len()`, `s.str.upper()`, `s.str.contains(pat)`, `s.str.replace(pat, repl, regex=True)`, `s.str.slice(0, n)`
2. **`.apply` + `lambda`** — for anything `.str` doesn't offer: `s.apply(lambda w: re.findall(..., w))`

And the crucial difference from the `re` module: in `.str.replace`, **regex is off by default** —
pass `regex=True` when your pattern is a pattern.
""")

code(t42, """
words = pd.Series(["  kite ", "Kayakugh!", "  banana  ", "clarab"])
words.«str.strip»()
""")

code(t42, """
cleaned = words.str.strip()
(
    cleaned.«str.len»(),                                  # length of each word
    cleaned[cleaned.«str.contains»(r"^k", case=False)],   # starts with k or K
)
""")

code(t42, """
cleaned.str.«replace»(r"b$", "y", «regex=True»)           # ends in b -> y : clarab -> claray
""")

md(t42, """
✅ **Check:** `words.str.replace("b$", "y")` (no `regex=True`) silently changes **nothing**.
Why no error, and why no change?
""")

answer(t42, """
Without `regex=True`, pandas looks for the literal two characters `b$` — which appear in no
word — so it "succeeds" at replacing zero occurrences. No error, wrong result: the worst kind
of bug. When the pattern contains regex machinery (`$ ^ + * [ ]`), say `regex=True`; when it's
plain text (like `aa` → `ee`), the default literal mode is exactly right.
""")

md(t42, """
---
## 4. The PA 4.2 pipeline in miniature

The decode activity is: **strip → surgical regex repairs → truncate → join**. Here is the
whole shape on a three-word message — the PA is this, nine steps instead of three:
""")

code(t42, """
mini = pd.Series(["  Hekko!  ", "ughhh!zhere", "buddb  "])

fixed = (mini
         .str.strip()                              # 1. trim ends
         .str.replace(r"ugh+[^\\w\\s]", "", regex=True)   # 2. delete the groans
         .str.replace("kk", "ll")                  # 3. literal swap — no regex needed
         .str.replace("z", "t")                    # 4. another literal swap
         .str.replace(r"b$", "y", regex=True))     # 5. pattern swap — regex needed
«" ".join(fixed)»                                  # 6. Series -> one string
""")

md(t42, """
## The three lines to keep

| | |
|---|---|
| **Strings** | `strip / lower / replace / split`, and `sep.join(pieces)` to reassemble — always reassign the result |
| **Patterns** | `[^\\w\\s]` punctuation, `+` repeats, `^`/`$` anchors; test with `findall` before you `sub` |
| **Series** | `.str.method()` for the whole column; `regex=True` when the pattern is a pattern; `.apply(lambda ...)` for the rest |

PA 4.2 is on the course site: [%s](%s).
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
    (t41, WEEK / "GSB5544_Topic_4_1_Text_as_Data",
     "# GSB 5544 — Topic 4.1: Text as Data  \n"
     "*Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
    (t42, WEEK / "GSB5544_Topic_4_2_Strings_and_Regex",
     "# GSB 5544 — Topic 4.2: Strings and Regular Expressions  \n"
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
