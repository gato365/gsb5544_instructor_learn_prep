#!/usr/bin/env python3
"""Generate the Week 6 Topic notebook (student + executed solution).

Usage:  python3 tools/build_week6_topics.py            # write -empty and -solution, execute the solution
        python3 tools/build_week6_topics.py --no-exec  # write both notebooks without executing

Topic 6.1 — Functions, lambda, and map(): "How repetitive is a song?"  -> pairs with PA 6.1 (Writing Functions)
  Core (Parts A–D): lyrics as a string / list of lines / list of words -> a named function (def, parameter, return)
  -> the same operation as a lambda -> map() over every line.  Then practice (predict, complete, explain, debug,
  apply, interpret), a two-song investigation, and clearly marked optional extensions.
Same markers as the earlier builders:
  «text»            inside a code cell  -> "text" in the solution, "____" in the student version
  **Answer:** ...   as a markdown cell  -> kept in the solution, replaced by a "Your answer" prompt
Cells built with raises=True demonstrate an error on purpose; they are tagged "raises-exception" so execution
continues past them, and every other cell must run cleanly.
All lyrics are public-domain texts typed into the notebook, so nothing needs a network connection or credentials.
(Optional LyricsGenius preparation for instructors is documented in week_6/README.md, not in the notebooks.)
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_6"
JUPYTER = "/opt/anaconda3/bin/jupyter"
SITE = "https://gato365.github.io/gsb5544_instructor_learn_prep/"


def md(cells: list[dict], text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(cells: list[dict], text: str, raises: bool = False) -> None:
    meta = {"tags": ["raises-exception"]} if raises else {}
    cells.append({"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n")})


def answer(cells: list[dict], text: str) -> None:
    md(cells, "**Answer:** " + text.strip())


# ============================================================================ #
#  Topic 6.1 — Functions, lambda, and map() (pairs with PA 6.1)
# ============================================================================ #

t61: list[dict] = []

md(t61, """
# GSB 5544 — Topic 6.1: Functions, lambda, and map() — SOLUTION
*How repetitive is a song, and how can Python help us investigate?*
""")

md(t61, """
## The plan

A song repeats itself — a chorus comes back, a word is sung three times in a row. To *measure* that, we need
to do the same small job (count the words in a line, lower-case a word) **over and over**. That is exactly
what functions are for.

| | Question | New Python | Where it lands in PA 6.1 |
|---|---|---|---|
| **A. Collect** | How do lyrics become something Python can count? | `splitlines()`, `split()`, `len()`, `lower()` | — |
| **B. Name** | How do I package "count the words in a line" so I can reuse it? | `def`, parameter, `return` | Q1 – Q4 (watch where `return` sits) |
| **C. Shorten** | Can a one-line job be written in one line? | `lambda` | — |
| **D. Repeat** | How do I apply a function to *every* line at once? | `map()`, `list()` | Q2 (a list as input) |
| **Practice** | Predict · Complete · Explain · Debug · Apply · Interpret | — | all |
| **E. Investigate** | Which of two songs is more repetitive? | `set()` | — |
| *Optional* | *Extensions for when the core is solid* | *`filter`, `sorted(key=)`, comprehensions, `Counter`* | — |

**Parts A–D, the practice, and Part E are the core.** The optional extensions at the end are clearly marked
and can be skipped.
""")

md(t61, """
### The lyrics

All three texts are in the **public domain** in the United States (published before 1929) and are typed in
below, so this notebook needs no internet connection and no account.

| Variable | Song | Words by | Published | Used for |
|---|---|---|---|---|
| `lyrics` | *Twinkle, Twinkle, Little Star* — first verse as commonly sung (the opening couplet returns at the end) | Jane Taylor (poem "The Star") | 1806 | Parts A–D, the practice |
| `ballgame_lyrics` | *Take Me Out to the Ball Game* — chorus | Jack Norworth (music: Albert Von Tilzer) | 1908 | Part E (comparison) |
| `rowboat_lyrics` | *Row, Row, Row Your Boat* | traditional (19th-century nursery round) | 1852 | Practice 5 – 6 (apply) |

We stay with *Twinkle, Twinkle* through all four core parts, so you can watch the **same job** being done
as the syntax changes. It is short on purpose: you can check every early answer by counting with your finger.
""")

code(t61, '''
lyrics = """Twinkle, twinkle, little star,
How I wonder what you are!
Up above the world so high,
Like a diamond in the sky.
Twinkle, twinkle, little star,
How I wonder what you are!"""

print(lyrics)
''')

# ---------------------------------------------------------------------------- A
md(t61, """
---
## Part A. Turn lyrics into a collection

Right now the whole song is **one string**. Python cannot count "lines" or "words" in it until we tell it
where one line (or word) ends and the next begins. Two string methods do that, and both return a **list**:

| Method | Splits at | Gives |
|---|---|---|
| `text.splitlines()` | each line break | a list of **lines** |
| `text.split()` | any run of spaces or line breaks | a list of **words** |
""")

code(t61, """
type(lyrics), len(lyrics)
""")

md(t61, """
`len()` of a **string** counts **characters** — letters, spaces, punctuation, and the 5 invisible line
breaks. 170 characters is true but not very useful. Split it into lines:
""")

code(t61, """
lines = lyrics.«splitlines»()
lines
""")

code(t61, """
type(lines), len(lines)
""")

md(t61, """
`len()` of a **list** counts **items**. `lines` is a list of 6 strings, so `len(lines)` is 6. Now split
the whole song into words:
""")

code(t61, """
words = lyrics.«split»()
print(len(words))
words[:8]
""")

md(t61, """
Notice what a "word" is to `split()`: whatever sits between spaces. So the punctuation stays attached
(`'Twinkle,'`, `'star,'`), and `'Twinkle,'` and `'twinkle,'` are **different strings** because of the
capital T. `lower()` standardizes capitalization:
""")

code(t61, """
print("Twinkle," == "twinkle,")                 # different to Python
print("Twinkle,".«lower»() == "twinkle,")        # the same once lower-cased
lines[0].lower()
""")

md(t61, """
The same text, three shapes. Knowing which one you are holding tells you what `len()` will count:

| Object | Type | Example element | `len()` counts | Here |
|---|---|---|---|---|
| `lyrics` | `str` — the whole song | a character, `'T'` | characters | 170 |
| `lines` | `list` of `str` | a line, `'Twinkle, twinkle, little star,'` | lines | 6 |
| `words` | `list` of `str` | a word, `'Twinkle,'` | words | 32 |

To count the words in **one** line, pick the line, then split it:
""")

code(t61, """
lines[0].split()
""")

code(t61, """
len(lines[0].split())          # words in line 1
""")

md(t61, """
✅ **Checkpoint A.** (a) How many lines does this excerpt have? (b) How many words are in the third line,
`lines[2]`? Write the code, and check by counting. (c) `len(lines[0])` is 30 but `len(lines[0].split())` is
4 — what is being counted in each case?
""")

code(t61, """
len(«lines[2]».split())
""")

answer(t61, """
(a) 6 — `len(lines)`. (b) 6 — `'Up above the world so high,'` splits into `['Up', 'above', 'the', 'world',
'so', 'high,']`. (c) `lines[0]` is a **string**, so `len` counts its 30 **characters** (spaces and commas
included); `lines[0].split()` is a **list**, so `len` counts its 4 **items** (words). *Same function, different
object, different meaning* — the most common slip in this topic is counting characters when you meant words.
""")

# ---------------------------------------------------------------------------- B
md(t61, """
---
## Part B. Write a named function

We just wrote `len(lines[0].split())`, and to count the next line we would write `len(lines[1].split())`,
and so on. When the same steps are repeated with only the input changing, **package the steps in a
function** and give the input a name:
""")

code(t61, """
def count_words(line):
    \"\"\"
    Count the words in one line of lyrics.

    Parameter
    ---------
    line : str
      One line of lyrics.

    Returns
    -------
    int
      The number of words (pieces of text separated by spaces).
    \"\"\"
    words_in_line = line.split()
    «return» len(words_in_line)
""")

md(t61, """
Running that cell prints nothing — it only **defines** the function. The pieces:

| Piece | Here | What it is |
|---|---|---|
| `def` | `def` | "I am defining a function" |
| **name** | `count_words` | how you will call it later; a verb that says what it does |
| **parameter** | `line` | a placeholder name for whatever input you hand over; it exists only *inside* the function |
| **docstring** | `\"\"\" ... \"\"\"` | the documentation `help(count_words)` shows (same layout as the textbook) |
| **body** | the indented lines | the steps, written in terms of the parameter |
| **`return`** | `return len(words_in_line)` | the value the function **hands back** to whoever called it |

Now **call** it. The value you pass in is the **argument**; inside the function, `line` refers to it.
""")

code(t61, """
count_words(«lines[0]»)             # argument: the first line  -> line = 'Twinkle, twinkle, little star,'
""")

code(t61, """
count_words("Up above the world so high,")      # any string works — the function doesn't care where it came from
""")

md(t61, """
### Why `return` and not `print()`?

`print()` **shows** a value on the screen and then throws it away. `return` **hands the value back**, so you
can store it, add it, or pass it to another function. Compare a version that prints instead of returning:
""")

code(t61, """
def count_words_printed(line):
    print(len(line.split()))          # shows the number ... but hands nothing back

result = count_words_printed(lines[0])
print("result is:", result)
""")

md(t61, """
The 4 appeared on screen, but `result` is **`None`** — Python's value for "nothing". A function without a
`return` hands back `None`. With `return`, the value is usable:
""")

code(t61, """
count_words(lines[0]) + count_words(lines[1])        # 4 + 6
""")

md(t61, """
Try that with `count_words_printed` and you get `TypeError: unsupported operand type(s) for +: 'NoneType' and
'NoneType'`. **Rule of thumb:** a function that *computes* something should `return` it; `print` is for
messages to a person (like PA 6.1's *"I love sevens!"*). And in PA 6.1 Questions 3 – 4, look closely at
**where** `return` is indented — a `return` that never runs is the same as no `return`.

### A loop as a bridge

To count the words in *every* line, one way is a `for` loop that calls the function once per line:
""")

code(t61, """
for line in lines[:3]:
    print(count_words(line), "<-", line)
""")

code(t61, """
counts = []                                  # start empty
for line in lines:
    counts.append(«count_words»(line))        # call the function, keep the result
counts
""")

md(t61, """
This works, and it is worth being able to read. Keep it in mind: **Part D does exactly this in one line.**

✅ **Checkpoint B.** In your own words: (a) what is the difference between the *parameter* `line` and the
*argument* `lines[0]`? (b) What would `counts` be if `count_words` used `print` instead of `return`?
""")

answer(t61, """
(a) The **parameter** is the name written in the `def` line — a placeholder used inside the body. The
**argument** is the actual value supplied when the function is called; for that call, `line` refers to it.
One definition, many calls, a different argument each time. (b) `[None, None, None, None, None, None]` — each
call would print a number on the screen but return `None`, and `append` would store six `None`s.
""")

# ---------------------------------------------------------------------------- C
md(t61, """
---
## Part C. The same job as a lambda expression

`count_words` has one real step: `len(line.split())`. For a job that small, Python offers a shorter way
to write a function — a **lambda expression**:
""")

code(t61, """
count_words_lambda = «lambda» line: len(line.split())

count_words_lambda(lines[0])
""")

md(t61, """
Same input, same output. Side by side:

| | Named function | Lambda expression |
|---|---|---|
| **Written as** | `def count_words(line):` ⏎ `    return len(line.split())` | `lambda line: len(line.split())` |
| **Input** | the parameter in parentheses, `(line)` | the name(s) before the colon, `line` |
| **The colon `:`** | starts the body | separates the input from the expression |
| **Output** | whatever follows `return` | the value of the expression — returned **automatically**, no `return` keyword |
| **Name** | always has one (`count_words`) | anonymous — no name unless you assign it |
| **Size** | any number of steps, loops, `if` blocks | **exactly one expression** |
| **Docstring** | yes | no |

Read `lambda line: len(line.split())` aloud as *"a function that takes `line` and gives back
`len(line.split())`."*

We stored the lambda in a variable above only to compare it with `count_words`. That is legal but not usual —
if you want a name, use `def`. A lambda is normally written **right where it is used**, most often as the
input to another function. That is Part D.

A second small job — standardizing capitalization:
""")

code(t61, """
(lambda line: line.«lower»())(lines[0])        # define a lambda and call it immediately on the first line
""")

md(t61, """
**When to use which.** Use a **lambda** for a short, one-expression job you need once, in place.
Use a **named function** when the job needs several steps (a lambda cannot hold them), a check on the input
(PA 6.1's `times_seven`), an explanation (a docstring), or will be reused in several places.

✅ **Checkpoint C.** (a) Without running it, what does `(lambda line: len(line.split()))("How I wonder what
you are!")` give? (b) Why could PA 6.1's `times_seven` *not* be written as a lambda?
""")

answer(t61, """
(a) `6` — the lambda receives the string as `line`, splits it into 6 words, and returns the count.
(b) `times_seven` has several statements: an `if` that exits on bad input, an `if` that prints, then a
`return`. A lambda can hold only **one expression**, so a multi-step function with checks needs `def`.
""")

# ---------------------------------------------------------------------------- D
md(t61, """
---
## Part D. Apply a function to every line: `map()`

`map(function, collection)` applies the function to **each item** of the collection, in order.

| Argument | Here | What it does |
|---|---|---|
| 1st: a **function** | `count_words` | the job to do — passed *by name*, **without parentheses** |
| 2nd: a **collection** | `tiny` | the items to do it to, one at a time |

Start with a tiny example — three lines — so each result can be traced:
""")

code(t61, """
tiny = lines[:3]
word_counts = «map»(count_words, tiny)
word_counts
""")

md(t61, """
That is not a list of numbers — it is a **map object**, a promise to compute the results *when asked*
(the `0x...` part is a memory address, different every run). To see the results, hand the map object to
`list()`, which asks for every result and collects them:
""")

code(t61, """
«list»(word_counts)
""")

md(t61, """
Trace it — `map` pairs each input with one output, in order:

| Input line (`tiny[i]`) | `count_words(tiny[i])` | Result |
|---|---|---|
| `'Twinkle, twinkle, little star,'` | 4 words | `4` |
| `'How I wonder what you are!'` | 6 words | `6` |
| `'Up above the world so high,'` | 6 words | `6` |

**Three lines in, three numbers out** — `map` never changes the number of items.

A map object can be read **only once**. Ask it again and it is empty:
""")

code(t61, """
list(word_counts)            # already used up by the previous cell
""")

md(t61, """
So the habit is: **wrap `map(...)` in `list(...)` right away** and store the list.

| | `map(count_words, tiny)` | `list(map(count_words, tiny))` |
|---|---|---|
| What it is | a map object: results computed on demand | a list: all results computed and stored |
| Displays as | `<map at 0x...>` | `[4, 6, 6]` |
| Can you index it (`[0]`), use `len()`, read it twice? | no | yes |

Now every line, with the named function and then with the equivalent lambda:
""")

code(t61, """
line_counts = list(map(count_words, «lines»))
line_counts
""")

code(t61, """
list(map(«lambda line: len(line.split())», lines))
""")

md(t61, """
**Why no parentheses after `count_words`?** `count_words` is the function itself — *the recipe*.
`count_words(lines[0])` is a call — *the dish*, already cooked (the number 4). `map` needs the recipe so it
can cook each line itself:
""")

code(t61, """
print(type(count_words))            # the function itself
print(type(count_words(lines[0])))  # the result of calling it: an int
""")

md(t61, """
A lambda needs no parentheses either: `lambda line: ...` already *is* a function.

Two quick checks that the results make sense. Adding the per-line counts should give the total number of
words (`sum()` adds up a list of numbers):
""")

code(t61, """
sum(line_counts), len(words)
""")

md(t61, """
And putting each line next to its count in a data frame makes the pairing visible:
""")

code(t61, """
import pandas as pd

pd.DataFrame({"line": lines, "n_words": line_counts})
""")

md(t61, """
You have met this idea before: in Week 5, `df_shows["seasons"].apply(len)` applied `len` to every cell of a
column. `.apply()` is pandas' version of `map()`.

And standardizing capitalization, line by line:
""")

code(t61, """
list(map(lambda line: line.lower(), lines))
""")

md(t61, """
✅ **Checkpoint D.** Explain, in two or three sentences to a classmate who missed class, what
`list(map(count_words, lines))` does, and why both `list` and the missing parentheses after `count_words` matter.
""")

answer(t61, """
`map` takes the function `count_words` and the list `lines`, and calls `count_words` on each line in turn —
the same as the loop in Part B, without writing the loop. We pass `count_words` without parentheses because
we are handing over the function for `map` to call; `count_words(...)` would call it once, immediately, and
give `map` a number instead. `map` produces a lazy map object, so `list(...)` is needed to compute and show
all six counts, `[4, 6, 6, 6, 4, 6]`.
""")

# ---------------------------------------------------------------------------- Practice
md(t61, """
---
## Practice

Six short tasks that move from supported to independent: **Predict → Complete → Explain → Debug → Apply →
Interpret.** Write your prediction or answer *before* running a cell.

### Practice 1 — Predict

Predict each output, then run the cell to check. (a) `len("star")` and `len(["star"])`.
(b) `list(map(count_words, ["Up above", "the world so high", ""]))` — careful with the last item.
""")

code(t61, """
len("star"), len(["star"])
""")

code(t61, """
list(map(count_words, ["Up above", "the world so high", ""]))
""")

answer(t61, """
(a) `(4, 1)` — `"star"` is a string of 4 characters; `["star"]` is a list with 1 item.
(b) `[2, 4, 0]` — three items in, three results out. The empty string `""` splits into an empty list `[]`,
whose length is 0. (A blank line in a lyric file counts as a *line* but contributes 0 *words* — keep that in
mind for Part E.)
""")

md(t61, """
### Practice 2 — Complete

(a) Fill in the blanks so `first_word` returns the **first word of a line, in lower case**.
(b) Use it with `map` to get the first word of every line. (c) Write the same thing with a lambda.
""")

code(t61, """
def first_word(line):
    \"\"\"Return the first word of a line of lyrics, in lower case.\"\"\"
    words_in_line = line.«split()»
    return words_in_line«[0]».lower()

first_word(lines[2])
""")

code(t61, """
list(«map»(first_word, «lines»))
""")

code(t61, """
list(map(«lambda line: line.split()[0].lower()», lines))
""")

answer(t61, """
(a) `line.split()` gives the list of words, `[0]` takes the first, `.lower()` standardizes it:
`first_word(lines[2])` is `'up'`. (b) `['twinkle,', 'how', 'up', 'like', 'twinkle,', 'how']` — one first word
per line (the comma stays attached; Part E deals with punctuation). (c) Same list: the lambda's single
expression chains the same three steps.
""")

md(t61, """
### Practice 3 — Explain

In plain language — no Python words allowed except the variable names — what does this produce, and what
does its output look like?
""")

code(t61, """
list(map(lambda line: line.lower().split(), lines[:2]))
""")

answer(t61, """
"For each of the first two lines, lower-case it and cut it into words; collect the results." The output is a
**list of two lists** — one list of words per line: `[['twinkle,', 'twinkle,', 'little', 'star,'], ['how',
'i', 'wonder', 'what', 'you', 'are!']]`. `map` returns whatever the function returns, one per item, so a
function that returns a list gives a list of lists.
""")

md(t61, """
### Practice 4 — Debug

Each cell below contains **one** realistic mistake. Run it, read the symptom, then fix it in the cell after.

**(a)** A word counter that "doesn't work":
""")

code(t61, """
def count_words_v2(line):
    len(line.split())

list(map(count_words_v2, lines[:3]))
""")

code(t61, """
def count_words_v2(line):
    «return» len(line.split())

list(map(count_words_v2, lines[:3]))
""")

answer(t61, """
**Forgot `return`.** The function computes the length and then throws it away, so every call returns `None`:
`[None, None, None]`. No error message — which is what makes this bug sneaky. Adding `return` gives `[4, 6, 6]`.
""")

md(t61, """
**(b)** "Words per line" — but the numbers look too big:
""")

code(t61, """
list(map(len, lines[:3]))
""")

code(t61, """
list(map(«count_words», lines[:3]))
""")

answer(t61, """
**Counting characters instead of words.** `len` applied to a *string* counts characters: `[30, 26, 27]`. The
line must be split into words first — `count_words` (or `lambda line: len(line.split())`) gives `[4, 6, 6]`.
Sanity check: a line cannot have 30 words in a 6-line nursery rhyme.
""")

md(t61, """
**(c)** Passing the function to `map`:
""")

code(t61, """
list(map(count_words(lines), lines))
""", raises=True)

code(t61, """
list(map(«count_words», lines))
""")

answer(t61, """
**Called the function instead of passing it.** Python evaluates `count_words(lines)` *first*, before `map`
starts — and `lines` is a list, which has no `.split()`, hence `AttributeError: 'list' object has no attribute
'split'`. (If it had worked, `map` would have received a number, not a function.) Pass the function by name,
with no parentheses: `map(count_words, lines)` → `[4, 6, 6, 6, 4, 6]`.
""")

md(t61, """
**(d)** Looking up the word count of the first line:
""")

code(t61, """
counts_by_line = map(count_words, lines)
counts_by_line[0]
""", raises=True)

code(t61, """
counts_by_line = «list»(map(count_words, lines))
counts_by_line[0]
""")

answer(t61, """
**Assumed `map` returns a list.** A map object cannot be indexed: `TypeError: 'map' object is not
subscriptable`. Convert it with `list(...)` first; then `counts_by_line[0]` is `4`.
""")

md(t61, """
### Practice 5 — Apply

A new song. Using only tools from Parts A–D: (a) split `rowboat_lyrics` into lines; (b) count the words in
each line with `map`; (c) check that the total matches `len(rowboat_lyrics.split())`; (d) lower-case every
line with `map` and a lambda.
""")

code(t61, '''
rowboat_lyrics = """Row, row, row your boat,
Gently down the stream.
Merrily, merrily, merrily, merrily,
Life is but a dream."""

rowboat_lines = rowboat_lyrics.«splitlines()»
rowboat_counts = list(map(«count_words», rowboat_lines))
rowboat_counts
''')

code(t61, """
«sum(rowboat_counts)», len(rowboat_lyrics.split())
""")

code(t61, """
list(map(«lambda line: line.lower()», rowboat_lines))
""")

answer(t61, """
(a–b) 4 lines with `[5, 4, 4, 5]` words. (c) Both totals are 18. (d) `['row, row, row your boat,', 'gently down
the stream.', 'merrily, merrily, merrily, merrily,', 'life is but a dream.']`. The same `count_words` function
works unchanged on a different song — that reuse is the point of writing it once.
""")

md(t61, """
### Practice 6 — Interpret

*Twinkle, Twinkle* gave per-line word counts `[4, 6, 6, 6, 4, 6]`. (a) What pattern do you see, and what might
it reflect in the song? (b) Can you conclude from these numbers alone that line 5 repeats line 1? Why or why
not? (c) *Row, Row, Row Your Boat* has a line with only 4 words but it feels very repetitive. What do the
word counts miss?
""")

answer(t61, """
(a) The pattern 4, 6 at lines 1–2 comes back at lines 5–6 — consistent with the opening couplet being sung
again at the end. (b) **No.** Equal counts do not mean equal lines: lines 2, 3, and 4 all have 6 words and
are all different. To show a line repeats we have to compare the lines themselves (`lines[0] == lines[4]` is
`True`). (c) Word counts measure *length*, not *repetition*: `'Merrily, merrily, merrily, merrily,'` is 4
words but one word said four times. To see repetition we need to compare the words themselves — Part E.
""")

# ---------------------------------------------------------------------------- E
md(t61, """
---
## Part E. Guided investigation: which song is more repetitive?

Compare *Twinkle, Twinkle* with the chorus of *Take Me Out to the Ball Game*.
""")

code(t61, '''
ballgame_lyrics = """Take me out to the ball game,
Take me out with the crowd;
Buy me some peanuts and Cracker Jack,
I don't care if I never get back.
Let me root, root, root for the home team,
If they don't win, it's a shame.
For it's one, two, three strikes, you're out,
At the old ball game."""
''')

md(t61, """
### Step 1 — Cleaning rules, stated up front

Before counting words across songs we must decide when two words are "the same". Our rules, applied to
every word of both songs:

1. **Lower-case** every word (`'Take'` and `'take'` are the same word).
2. **Remove punctuation from the two ends** of a word: `. , ; : ! ?` (`'star,'` → `'star'`).
3. Leave the inside of a word alone, so contractions survive: `"don't"` stays `"don't"`.

`word.strip(chars)` removes any of the characters in `chars` from the **start and end** of a string only —
exactly rule 2 and rule 3. Each rule is one step, so this is a job for a named function:
""")

code(t61, """
PUNCTUATION = ".,;:!?"

def clean_word(word):
    \"\"\"Lower-case a word and strip punctuation from its two ends.\"\"\"
    return word.«lower»().strip(PUNCTUATION)

clean_word("Twinkle,"), clean_word("crowd;"), clean_word("don't")
""")

md(t61, """
Now a function that turns a whole song into its list of clean words — `split` it, then `map` the cleaning
function over every word:
""")

code(t61, """
def song_words(text):
    \"\"\"Split a song into words and clean each one with clean_word.\"\"\"
    return list(map(«clean_word», text.split()))

song_words(lyrics)[:8]
""")

md(t61, """
### Step 2 — One new tool: `set()`

A **set** keeps **one copy of each distinct value**; duplicates disappear. So `len(set(...))` counts
*different* words:
""")

code(t61, """
set(["root", "root", "root", "for", "the", "home", "team"])
""")

code(t61, """
len(set(song_words(lyrics)))            # distinct words in Twinkle, Twinkle
""")

md(t61, """
(A set has no order, so its display order may look scrambled — that does not matter for counting.)

### Step 3 — The measure

For each song:

- **total words** = number of words after cleaning;
- **unique words** = number of *different* words after cleaning;
- **unique share** = unique words ÷ total words.

The unique share is a simple **vocabulary-diversity** measure. Near 1 → almost every word is new; lower →
the song reuses its words more. It is *one* lens on repetition — not a full measure of how repetitive a song
*sounds* (more on that below). Several steps and a result with three parts: a named function, returning a
dictionary.
""")

code(t61, """
def vocabulary_summary(text):
    \"\"\"Total words, unique words, and their ratio for one song (after clean_word).\"\"\"
    cleaned = song_words(text)
    total = len(cleaned)
    unique = len(«set»(cleaned))
    return {"total_words": total, "unique_words": unique, "unique_share": round(unique / total, 3)}

vocabulary_summary(lyrics)
""")

code(t61, """
songs = {"Twinkle, Twinkle, Little Star": lyrics,
         "Take Me Out to the Ball Game": ballgame_lyrics}

summaries = list(map(«vocabulary_summary», songs.values()))
pd.DataFrame(summaries, index=list(songs.keys()))
""")

md(t61, """
`map` applied `vocabulary_summary` to each song's text; `pd.DataFrame` turned the list of dictionaries into a
table (one row per song).

### Step 4 — Do the cleaning rules matter?

The same unique-word count under three rule sets — no cleaning, lower-case only, and our full rules:
""")

code(t61, """
for title, text in songs.items():
    raw_words = text.split()
    no_cleaning = len(set(raw_words))
    lower_only = len(set(map(lambda word: word.lower(), raw_words)))
    full_rules = len(set(map(clean_word, raw_words)))
    print(f"{title:32}  no cleaning: {no_cleaning}   lower only: {lower_only}   full rules: {full_rules}")
""")

md(t61, """
### Step 5 — Equal lengths

The ball-game chorus has 57 words; *Twinkle* has 32. A longer text has more chances to repeat, which tends to
*lower* its unique share — so comparing different lengths is not quite fair. One fix: compare the **first 32
cleaned words** of each.
""")

code(t61, """
n = len(song_words(lyrics))                 # 32
for title, text in songs.items():
    first_n = song_words(text)[:n]
    print(f"{title:32}  unique share of first {n} words: {len(set(first_n)) / n:.3f}")
""")

md(t61, """
### Step 6 — Interpret

✅ Use the tables above.

(a) Which excerpt appears more repetitive by the unique-share measure? What evidence supports that?
(b) How did capitalization and punctuation change the counts? Would *not* cleaning have changed the conclusion?
(c) Why might comparing excerpts of different lengths be misleading? Did it change the answer here?
(d) *Row, Row, Row Your Boat* has a unique share of 13 / 18 = 0.722 — *higher* than both. Does that mean it
is the least repetitive? What features of a song are missing when we analyze only its words?
""")

answer(t61, """
(a) **Twinkle, Twinkle**: 20 different words out of 32 (0.625) against 39 of 57 (0.684) for the ball-game
chorus. It reuses more of its words — mostly because two whole lines come back and "twinkle" appears four times.
(b) Without cleaning, `'Twinkle,'` / `'twinkle,'` and `'game,'` / `'game.'` count as different words, which
inflates the unique counts (21 vs 20; 44 vs 39) and makes both songs look *less* repetitive than they are.
Twinkle stays the more repetitive one here, but the gap and the numbers depend on the rules — which is why
they must be stated and applied the same way to every song.
(c) Longer texts tend to have lower unique shares simply because common words (*the*, *me*, *I*) come back.
Comparing the first 32 words of each: Twinkle 0.625, ball game 0.750 — the same conclusion, and a wider gap.
(d) No. Its 18 words are too few for the measure to be stable, and "row, row, row" / "merrily × 4" is
repetition packed into a *tiny* text. More importantly, words are only part of a song: the lyric sheet does not
show that *Row, Row* is sung as a **round** over and over, that a chorus repeats between verses, or anything
about **melody, rhythm, harmony, or instrumentation** — the main sources of musical repetition. Lyrics
copied from websites also add section labels like `[Chorus]` (which would count as words) and blank lines
between sections (extra "lines" with 0 words), so cleaning rules matter even more on real data.
""")

md(t61, """
**Substring vs. whole word.** Checking whether a line mentions a word needs care. `"the" in line` asks
whether the three **letters** appear anywhere — including inside `"they"`. To match the **whole word**,
compare against the line's list of clean words:
""")

code(t61, """
line = "If they don't win, it's a shame."
print("the" in line.lower())                              # substring: True, because of "they"
print("the" in list(map(clean_word, line.split())))       # whole word: False
""")

# ---------------------------------------------------------------------------- Optional
md(t61, """
---
## Optional extensions

*Not required.* Each one re-uses a job you already know — counting words, cleaning words — in a new form.

### `filter()` — keep only the items that pass a test

`filter(function, collection)` keeps the items for which the function returns **`True`**. The function
makes a yes/no decision for each item; the items themselves are kept unchanged. Like `map`, it returns a lazy
object, so wrap it in `list()`.
""")

code(t61, """
ballgame_lines = ballgame_lyrics.splitlines()

list(filter(lambda line: count_words(line) >= 8, ballgame_lines))        # lines with 8 or more words
""")

code(t61, """
list(filter(lambda line: "the" in song_words(line), ballgame_lines))    # lines containing the WORD "the"
""")

md(t61, """
### `sorted()` with `key=` — order items by a computed value

`key=` takes a function, applied to each item to decide the order — here, `count_words`. The result is the
**original lines**, rearranged from fewest to most words; the counts are only used for ordering.
""")

code(t61, """
sorted(ballgame_lines, key=count_words)
""")

md(t61, """
### List comprehensions — another way to write `map`

`[expression for item in collection]` builds a list directly. These two lines do the same thing:
""")

code(t61, """
print(list(map(count_words, lines)))
print([count_words(line) for line in lines])
""")

md(t61, """
Many Python programmers prefer the comprehension for everyday code; `map` makes the "apply a function"
idea explicit, which is why we learned it first.

### `collections.Counter` — how often each item appears

`Counter` counts every distinct item in a list; `.most_common(k)` lists the `k` most frequent.
""")

code(t61, """
from collections import Counter

Counter(song_words(lyrics)).most_common(3)
""")

code(t61, """
Counter(lines).most_common(2)              # repeated LINES: each of the first two lines is sung twice
""")

code(t61, """
len(lines), len(set(lines))                # 6 lines, only 4 different ones
""")

md(t61, """
---
## The lines to keep

| | |
|---|---|
| **Collect** | `text.splitlines()` → list of lines; `text.split()` → list of words; `len()` counts characters of a string but items of a list; `lower()` standardizes case |
| **Name** | `def name(parameter):` + indented body + `return value`; the argument is what you pass in; `return` hands a value back, `print` only shows it |
| **Shorten** | `lambda parameter: expression` — a one-expression function, value returned automatically; use `def` when the job needs steps, checks, or explanation |
| **Repeat** | `list(map(function, collection))` — pass the function **without** parentheses; `map` is lazy and single-use, so wrap it in `list()` |

PA 6.1 is on the course site: [%s](%s).
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
              and "raises-exception" not in c["metadata"].get("tags", [])
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    missing = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
               and "raises-exception" in c["metadata"].get("tags", [])
               and not any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if errors or missing:
        raise SystemExit(f"unexpected errors: {errors}; demo cells that did not raise: {missing}")
    print(f"executed {path.relative_to(ROOT)} (no unexpected errors)")


TOPICS = [
    (t61, WEEK / "GSB5544_Topic_6_1_Functions_Lambda_Map",
     "# GSB 5544 — Topic 6.1: Functions, lambda, and map()  \n"
     "*How repetitive is a song? Fill each `____` blank as you work; the ✅ checks ask for a sentence or two.*"),
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
