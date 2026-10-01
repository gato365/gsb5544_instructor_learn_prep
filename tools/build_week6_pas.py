#!/usr/bin/env python3
"""Build the Week 6 practice-activity INSTRUCTOR SOLUTION notebooks (PA 6.1 — Writing Functions, PA 6.2 — Iteration).

Usage:  python3 tools/build_week6_pas.py            # build + execute both -solution files
        python3 tools/build_week6_pas.py --no-exec  # build without executing

The student notebooks (`Practice_Activity_6_1_Writing_Functions.ipynb`, `GSB_5544_Practice_Activity_6_2_Iteration.ipynb`)
are the source of truth for the questions and are left untouched.  For each, this script writes a `-solution.ipynb`
sibling in which each question gets an answer group: the approach, complete runnable code, what each important piece
does, the expected output, and the common mistakes to watch for in class.  The questions are the check-ins of the
textbook chapters the activities come from:
  PA 6.1 — https://ds-ml-with-python.github.io/Course-Textbook/05-function_writing.html  (def, scope, unit tests, validation)
  PA 6.2 — https://ds-ml-with-python.github.io/Course-Textbook/06-iteration.html         (for loops, vectorizing, map, lambda, apply)

Cells built with raises=True demonstrate an error on purpose (tagged "raises-exception" so execution continues).
Executing needs the `palmerpenguins` package in the Anaconda kernel (`pip install palmerpenguins`); no network.
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

PA61_SRC = WEEK / "Practice_Activity_6_1_Writing_Functions.ipynb"
PA62_SRC = WEEK / "GSB_5544_Practice_Activity_6_2_Iteration.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str, raises: bool = False) -> dict:
    meta = {"tags": ["raises-exception"]} if raises else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n")}


# ============================================================================ #
#  PA 6.1 — Writing Functions
# ============================================================================ #

PA61_HEADER = [
    md("""
# GSB 5544 — PA 6.1: Writing Functions — INSTRUCTOR SOLUTION
*Textbook: [Chapter 6, Writing Functions](https://ds-ml-with-python.github.io/Course-Textbook/05-function_writing.html)
— scope, dynamic lookup, unit tests, and input validation*
"""),
    md("""
**How to use this notebook.** Each question is answered in the same five beats: **Approach** (the idea, in a
sentence or two) → **code** (complete and executed) → **What the code does** (piece by piece) → **Expected
output** → **Common mistakes** (what students actually do, and the symptom you will see on their screen).

**The one idea to keep returning to:** *a function is a small, sealed environment.* Arguments go in, one value
comes back through `return`, and nothing inside changes the objects outside (Questions 0 and 4). Everything
between the input and the `return` — including checking the input (Questions 1–3) — is your job as the author.

**Before class: a slip in the printed `add_or_subtract` (Questions 3–4).** Its final `return res` is indented
*inside* the `else` block, after `exit(...)`, so it can never run: as printed, `"add"` and `"subtract"` calls
return `None`. The answers below give the **intended** behaviour (the `return` one level out) and show what the
printed version does, so you can either fix the handout or use it as a live "where does `return` sit?" moment.
"""),
]

PA61_Q0 = [
    md("""
### Solution 0 — the Section 6.3.2 check-in: one plotting function, two ways

**Approach.** "Species and/or island" means each filter is **optional**: give both parameters the default
`None`, and filter only on the ones the user supplied. Then write the function twice — once reaching out to
the global `penguins` (**dynamic lookup**), once receiving the data as an argument (**everything passed in**) —
and compare how each behaves when the global data changes.
"""),
    code('''
from plotnine import labs

penguins = load_penguins()

# Version 1 — DYNAMIC LOOKUP: the function is never given the data; it finds `penguins` in the global environment
def plot_bills_dynamic(species = None, island = None):
  """
  Scatterplot of bill length vs. bill depth for a chosen species and/or island.

  Parameters
  ----------
  species : str, optional
    "Adelie", "Chinstrap", or "Gentoo". None (the default) keeps every species.
  island : str, optional
    "Biscoe", "Dream", or "Torgersen". None (the default) keeps every island.

  Returns
  -------
  ggplot
    The scatterplot.
  """
  subset = penguins                                      # <- looked up in the GLOBAL environment
  if species is not None:
    subset = subset[subset["species"] == species]
  if island is not None:
    subset = subset[subset["island"] == island]
  subset = subset.dropna(subset = ["bill_length_mm", "bill_depth_mm"])

  plot = (ggplot(subset, aes(x = "bill_length_mm", y = "bill_depth_mm"))
          + geom_point()
          + labs(title = f"species = {species}, island = {island}  (n = {len(subset)})",
                 x = "Bill length (mm)", y = "Bill depth (mm)"))
  return plot
'''),
    md("""
Unit tests — a species only, an island only, and both:
"""),
    code("""
plot_bills_dynamic(species = "Adelie")
"""),
    code("""
plot_bills_dynamic(island = "Dream")
"""),
    code("""
plot_bills_dynamic(species = "Adelie", island = "Torgersen")
"""),
    code('''
# Version 2 — EVERYTHING PASSED IN: the data is a parameter, and the inputs are checked before use
def plot_bills(data, species = None, island = None):
  """
  Scatterplot of bill length vs. bill depth for a chosen species and/or island.

  Parameters
  ----------
  data : pandas DataFrame
    Penguin data with columns species, island, bill_length_mm, bill_depth_mm.
  species : str, optional
    A species present in `data`. None (the default) keeps every species.
  island : str, optional
    An island present in `data`. None (the default) keeps every island.

  Returns
  -------
  ggplot
    The scatterplot.
  """
  if not isinstance(data, pd.DataFrame):
    exit("Please provide the penguin data as a pandas DataFrame.")
  if species is not None and species not in data["species"].unique():
    exit(f"species must be one of {sorted(data['species'].unique())}.")
  if island is not None and island not in data["island"].unique():
    exit(f"island must be one of {sorted(data['island'].unique())}.")

  subset = data                                          # <- only what was handed in
  if species is not None:
    subset = subset[subset["species"] == species]
  if island is not None:
    subset = subset[subset["island"] == island]
  subset = subset.dropna(subset = ["bill_length_mm", "bill_depth_mm"])

  plot = (ggplot(subset, aes(x = "bill_length_mm", y = "bill_depth_mm"))
          + geom_point()
          + labs(title = f"species = {species}, island = {island}  (n = {len(subset)})",
                 x = "Bill length (mm)", y = "Bill depth (mm)"))
  return plot
'''),
    code("""
plot_bills(penguins, species = "Gentoo")
"""),
    code("""
plot_bills(penguins, species = "adelie")          # lower-case typo -> stopped with a helpful message
""", raises=True),
    code("""
# A valid combination that simply has no penguins: Gentoo live only on Biscoe
plot_bills(penguins, species = "Gentoo", island = "Dream").data.shape
"""),
    md("""
**Why "pass everything in" is the safer default.** Suppose, later in the analysis, someone narrows the global
data frame. The dynamic-lookup version silently follows it; the passed-in version only ever uses what it is given:
"""),
    code("""
penguins = penguins[penguins["island"] == "Biscoe"]            # the global data changes ...

print("dynamic lookup, Adelie:", plot_bills_dynamic(species = "Adelie").data.shape[0], "penguins")
print("passed in,      Adelie:", plot_bills(load_penguins(), species = "Adelie").data.shape[0], "penguins")

penguins = load_penguins()                                     # restore the full data
"""),
    md("""
**What the code does.**
- **`species = None, island = None`** — default arguments make both filters optional, which is what "and/or"
  asks for. `if species is not None:` filters only when the user supplied a species.
- **`subset = subset[subset["species"] == species]`** — the Week 2 boolean filter, applied one condition at a
  time, so any combination of the two works without four separate branches.
- **`dropna(...)`** — two penguins have no bill measurements; dropping them avoids plotnine's
  *"Removed 2 rows containing missing values"* warning and makes the `n` in the title honest.
- **`return plot`** — the function hands the plot back. The notebook displays it because it is the cell's last
  value, and the caller can still add to it (`plot_bills(penguins) + labs(...)`), which a `print` would not allow.
- **Version 1's `subset = penguins`** is the dynamic lookup: `penguins` is not a parameter, so Python looks in
  the function's local environment, does not find it, and falls back to the global one (textbook §6.3.2).
- **Version 2's checks** follow textbook §6.4.2: `isinstance` for a Python type, `in data[...].unique()` for a
  valid value, and `exit(...)` with a message that tells the user how to fix the call.
- `.data.shape` reads the data frame stored inside a plotnine plot — a quick way to unit-test the filter
  without looking at a picture.

**Expected output.** Adelie: 151 penguins (152 minus 1 with missing bills) spread over all three islands.
Dream: 124 (56 Adelie + 68 Chinstrap; none missing). Adelie on Torgersen: 51. Gentoo: 123. The `"adelie"`
typo stops with `SystemExit: species must be one of ['Adelie', 'Chinstrap', 'Gentoo'].` Gentoo on Dream is a
legitimate empty result, `(0, 8)`. After the global data is narrowed to Biscoe, the dynamic version reports
**44** Adelie while the passed-in version still reports **151** — the same call, a different answer, and no
error to warn you.

**Common mistakes.**
- **Required arguments instead of defaults** (`def f(species, island)`) — then "species only" is impossible
  without passing a dummy island.
- **`if species:` / `== None`** instead of `is not None` — works here, but `is not None` is the idiom and
  says exactly what is meant.
- **Combining conditions with `and`**: `penguins[(penguins.species == s) and (penguins.island == i)]` →
  `ValueError: The truth value of a Series is ambiguous`. Element-wise conditions need `&` and parentheses.
- **Case-sensitive values**: `"adelie"` matches nothing, and without a check the result is a silent, empty plot.
- **Calling the "passed-in" version with a filtered copy by accident** — the point of Version 2 is that the
  caller decides what data goes in, so the call site should make that visible.
- **`print(plot)` inside the function** instead of `return plot` — the plot appears, but the function
  returns `None`, so the caller cannot modify or save it.
"""),
]

PA61_Q1 = [
    md("""
### Solution 1 — `times_seven()`

**Approach.** Three steps, in order: **check** the input (stop with a message if it is not a number),
**announce** if it is a 7, **return** the input times 7. The check comes first so nothing is computed on bad input.
"""),
    code("""
def times_seven(x):

  if not isinstance(x, (int, float)):
    exit("Please provide a single number (an int or a float) for x.")

  if x == 7:
    print("I love sevens!")

  return x * 7
"""),
    md("""
**What the code does.**
- **`isinstance(x, (int, float))`** is `True` when `x` is an `int` *or* a `float` — a tuple of types means
  "any of these". `not` reverses it, so the `exit` runs only for non-numbers (textbook §6.4.2).
- **`exit("...")`** (imported from `sys` at the top of the notebook) stops the function and shows the message.
  Say what was expected, so the user knows how to fix the call.
- **`print("I love sevens!")`** is a message *for the person* — it does not change what the function returns.
- **`return x * 7`** hands the result back to the caller.

**Common mistakes.**
- **`return print(x * 7)`** or printing instead of returning → the number shows on screen but the function
  returns `None`, so `times_seven(2) + 1` fails.
- **`type(x) == int`** → rejects floats (`times_seven(1.5)` exits). Use `isinstance` with both types.
- **`if x = 7:`** → `SyntaxError`; comparison is `==`.
- **Checking with `x.isnumeric()`** — that is a *string* method; on a number it raises `AttributeError`.
- **Putting the `return` before the `print`** → the announcement never happens (code after `return` does not run).
"""),
]

PA61_Q2 = [
    md("""
### Solution 2 — unit tests, and a list as input

**Approach.** Test the ordinary cases (a 7, other integers, a float, zero, a negative), then the unexpected
ones (a string, a list). A test is only useful if you know the right answer *before* running it, so the
`assert` statements write the expected value down.
"""),
    code("""
times_seven(7)                       # the special case: prints AND returns
"""),
    code("""
times_seven(2), times_seven(-1.5), times_seven(0)
"""),
    code("""
assert times_seven(2) == 14
assert times_seven(-1.5) == -10.5
assert times_seven(0) == 0
assert times_seven(7) == 49          # also prints the announcement
print("all ordinary-input tests passed")
"""),
    md("""
Unexpected input — each call should stop with our message rather than return something strange:
"""),
    code("""
times_seven("7")
""", raises=True),
    code("""
times_seven([1, 3, 5, 7])
""", raises=True),
    md("""
**Why the check matters for the list.** Without it, Python would happily run `[1, 3, 5, 7] * 7` — but `*` on
a list means *repeat the list*, and a list is never `== 7`, so there is no announcement either:
"""),
    code("""
repeated = [1, 3, 5, 7] * 7
print(len(repeated), repeated[:8], "...")
print([1, 3, 5, 7] == 7)
"""),
    md("""
**If the user really wants each number times 7**, apply the function to each element — this is `map()` from
Topic 6.1:
"""),
    code("""
list(map(times_seven, [1, 3, 5, 7]))
"""),
    md("""
A test that *expects* the error, so a whole test cell can run without stopping:
"""),
    code("""
try:
  times_seven([1, 3, 5, 7])
  print("TEST FAILED: a list was accepted")
except SystemExit as err:
  print("stopped as expected:", err)
"""),
    md("""
**Two edge cases worth showing.** A Python quirk: `True` is an `int` (it equals 1), so it passes the check. And a
number that came out of numpy — for example a value pulled from a pandas column — is *not* a Python `int`:
"""),
    code("""
times_seven(True)
"""),
    code("""
times_seven(np.int64(7))             # a numpy integer is not an instance of int
""", raises=True),
    code("""
from numbers import Number

def times_seven_v2(x):

  if isinstance(x, bool) or not isinstance(x, Number):
    exit("Please provide a single number for x.")

  if x == 7:
    print("I love sevens!")

  return x * 7

times_seven_v2(np.int64(7)), times_seven_v2(2.5)
"""),
    md("""
**What the code does.**
- `assert condition` does nothing when the condition is `True` and raises `AssertionError` when it is
  `False` — a test that writes down the expected answer.
- `try:` / `except SystemExit as err:` catches the stop that `exit` raises, so we can check that it *did* stop
  and read its message (`err`).
- `list(map(times_seven, [1, 3, 5, 7]))` calls `times_seven` once per element: four numbers in, four out.
- `numbers.Number` is the "any kind of number" type: it includes numpy's number types; excluding `bool`
  explicitly closes the `True` loophole.

**Expected output.** `times_seven(7)` prints *I love sevens!* and returns 49; the others give `(14, -10.5, 0)`;
the asserts pass. `"7"` and `[1, 3, 5, 7]` both stop with `SystemExit: Please provide a single number (an int or
a float) for x.` Unchecked, the list would become a 28-element list, with no announcement. Mapped over the list:
*I love sevens!* printed **once** (only for the 7) and `[7, 21, 35, 49]`. `times_seven(True)` returns 7;
`np.int64(7)` is rejected by the original check and accepted by `times_seven_v2`, which returns
`(np.int64(49), 17.5)` and prints the announcement.

**Common mistakes.**
- **"Testing" by eye** — running `times_seven(2)` and not deciding in advance that the answer should be 14.
- **Only testing the happy path.** The list case is the one that shows *why* the validation exists.
- **Putting several error-raising calls in one cell** — execution stops at the first, and the others never run.
  One call per cell, or `try`/`except` as above.
- **Answering "it multiplies each number by 7"** — that is true for a numpy array (`np.array([1, 3, 5, 7]) * 7`),
  not for a Python list.
"""),
]

PA61_Q3 = [
    md("""
### Solution 3 — predictions

**Approach.** Follow each call through the function with its defaults filled in: `second_num = 2` and
`type = "add"` unless the call says otherwise. Then ask: which branch runs, and *who* raises any error — the
function's own `exit`, or Python itself?

| Call | Filled in | Answer |
|---|---|---|
| `add_or_subtract(5, 6, type = "subtract")` | 5 − 6 | **b. −1** |
| `add_or_subtract("orange")` | `"orange" + 2` | **e.** an error from Python's `+`, raised inside our function |
| `add_or_subtract(5, 6, type = "multiply")` | no branch matches → `else` | **d.** the function's own `exit` message |

Options a (1) and c (30) are the distractors for "subtracted in the wrong order" and "multiplied anyway".

Each call runs in its own cell, because the second and third stop with an error. First, **with the function
exactly as printed**:
"""),
    code("""
print(add_or_subtract(5, 6, type = "subtract"))     # as printed: `return res` sits inside the else block
"""),
    code("""
add_or_subtract("orange")
""", raises=True),
    code("""
add_or_subtract(5, 6, type = "multiply")
""", raises=True),
    md("""
The first call shows **`None`**, not −1: the subtraction happens, but the only `return` is inside the `else`
block, after `exit(...)`, where it can never be reached. Moving `return res` out one level gives the intended
function. We keep the printed one under another name for Question 4:
"""),
    code("""
add_or_subtract_as_printed = add_or_subtract       # keep the printed version for comparison

def add_or_subtract(first_num, second_num = 2, type = "add"):

  if (type == "add"):
    res = first_num + second_num
  elif (type == "subtract"):
    res = first_num - second_num
  else:
    exit("Please choose `add` or `subtract` as the type.")

  return res                                     # one level out: runs after "add" or "subtract"

add_or_subtract(5, 6, type = "subtract")
"""),
    md("""
**What the code does.**
- **Call 1**: `type = "subtract"` → `res = 5 - 6 = -1` → returned (once `return` is in the right place).
- **Call 2**: only `first_num` is given, so `second_num = 2` and `type = "add"`. The function runs
  `"orange" + 2`, and Python's `+` refuses: `TypeError: can only concatenate str (not "int") to str`. The
  message is Python's, not ours — the error is defined in a different function (the `+` operation) that is
  called *inside* `add_or_subtract`. **Answer e.**
- **Call 3**: `"multiply"` matches neither branch → `else` → `exit(...)` → `SystemExit` with the function's own
  message. **Answer d.**

**Expected output.** As printed: `None`, a `TypeError`, and `SystemExit: Please choose `add` or `subtract` as the
type.` With the corrected function, call 1 returns **−1**; calls 2 and 3 are unchanged.

**Common mistakes.**
- **Predicting d for "orange"** — the function never checks its inputs' types, so its own `exit` is not what
  stops it. (A validation line like Question 1's would turn this into a d.)
- **Predicting 1** — subtracting in the wrong order: `first_num - second_num` is 5 − 6.
- **Running all three calls in one cell** (as the handout lays them out) → only the `TypeError` appears.
- **Not noticing the indentation** — in Python, indentation *is* the structure. This is also why Topic 6.1 treats
  "forgot `return`" as the most common silent bug: no error, just `None`.
- A style note worth making: naming a parameter **`type`** masks the built-in `type()` inside the function
  (name masking, textbook §6.3.1). `operation = "add"` would be clearer.
"""),
]

PA61_Q4 = [
    md("""
### Solution 4 — what is in the global environment?

**Approach.** Only assignments *in the global environment* change global objects. Inside a call,
`second_num = 4` sets the function's **parameter** in its own local environment; it does not touch the global
`second_num`. So: track each `=` at the top level, and evaluate each call with its defaults.
"""),
    code("""
print("first_num  =", first_num)
print("second_num =", second_num)
print("result     =", result)
print("result_2   =", result_2)
"""),
    code("""
# The same two calls with the function exactly as printed in the handout
print(add_or_subtract_as_printed(first_num, second_num = 4), add_or_subtract_as_printed(first_num))
"""),
    md("""
**What the code does.** The Question 4 cell above ran with the corrected `add_or_subtract` from Solution 3.

| Object | Value | Why |
|---|---|---|
| **a. `first_num`** | **5** | assigned once at the top level; passing it into a function does not change it |
| **b. `second_num`** | **3** | `second_num = 4` inside the call is the function's *parameter*, not the global object — "what happens in a function stays in the function" |
| **c. `result`** | **9** | first 8, then **overwritten** by `add_or_subtract(5, second_num = 4)` = 5 + 4 (default `type = "add"`) |
| **d. `result_2`** | **7** | `add_or_subtract(5)` uses both defaults: 5 + 2 |

**Expected output.** 5, 3, 9, 7. With the function **as printed**, both calls return `None`, so `result` and
`result_2` would be `None` — and note that `result` would *still* be overwritten (8 → `None`), because the
assignment happens regardless of what the function returns.

**Common mistakes.**
- **`second_num = 4`** — believing the keyword argument reassigns the global variable. It names which
  parameter receives 4, nothing more.
- **`result = 8`** — forgetting that the later `result = ...` replaces it.
- **`result_2` is an error** — overlooking the defaults; `second_num` has one (`2`), so one argument is enough.
- **`result_2 = 8`** — assuming the call reuses the `second_num = 4` from the previous call. Every call starts a
  fresh local environment (textbook §6.3.2).
"""),
]


# ============================================================================ #
#  PA 6.2 — Iteration
# ============================================================================ #

PA62_HEADER = [
    md("""
# GSB 5544 — PA 6.2: Iteration — INSTRUCTOR SOLUTION
*Textbook: [Chapter 7, Iteration](https://ds-ml-with-python.github.io/Course-Textbook/06-iteration.html)
— for loops, vectorized functions, `map()`, lambda functions, and `.apply()`*
"""),
    md("""
**How to use this notebook.** Each question is answered in the same five beats: **Approach** (the idea, in a
sentence or two) → **code** (complete and executed) → **What the code does** (piece by piece) → **Expected
output** → **Common mistakes** (what students actually do, and the symptom you will see on their screen).

**The one idea to keep returning to:** *first ask whether the function is vectorized.* If it works on a whole
array (`np.sqrt`, boolean masks), use it directly. If it only works on one value — anything with an `if` inside —
you must iterate: a `for` loop, `map()` for lists, or `.apply(axis=1)` for the rows of a data frame. A lambda is
the glue that fixes the arguments that should *not* change.

The questions refer to "the code above" in the textbook: `sing_verse()` from *99 Bottles of Beer*. It is
reproduced here so the notebook runs on its own.
"""),
    code("""
import numpy as np
import pandas as pd

# From the textbook (Section 7.2): one verse, returned as a string rather than printed
def sing_verse(num):
  song = str(num) + " bottles of beer on the wall \\n" + str(num) + " bottles of beer \\n" + " take one down, pass it around, \\n" + str(num-1) + " bottles of beer on the wall \\n"
  return song

print(sing_verse(99))
"""),
]

PA62_Q1 = [
    md("""
### Solution 1 — a list of verses instead of one long string

**Approach.** The textbook's loop starts with an empty **string** and adds each verse with `+`. Start with an
empty **list** instead and add each verse as a one-element list (or `.append()` it).
"""),
    code("""
verses = []                                  # empty LIST, not an empty string
for i in range(100, 97, -1):
  verses = verses + [sing_verse(i)]          # list + list -> a longer list

print(type(verses), len(verses))
verses
"""),
    code("""
# The same loop with .append(), which adds one element in place
verses = []
for i in range(100, 97, -1):
  verses.append(sing_verse(i))

print(verses[0])                             # the first verse is still one readable string
"""),
    code("""
print("".join(verses))                       # and the list can be glued back into the whole song
"""),
    md("""
**What the code does.**
- `verses = []` — the "empty object" to accumulate into is now a list.
- `verses + [sing_verse(i)]` — the brackets turn the verse string into a one-element list, so `+` is
  *list + list* and appends one element. `verses.append(sing_verse(i))` does the same thing in place.
- `range(100, 97, -1)` — 100, 99, 98: start, stop (not included), step.
- `"".join(verses)` — glues the elements together with nothing between them (each verse already ends in `\\n`).

**Expected output.** A list of **3 strings**, one verse each; `verses[0]` prints the 100-bottles verse.

**Common mistakes.**
- `verses = verses + sing_verse(i)` → `TypeError: can only concatenate list (not "str") to list`. Without
  the brackets Python tries to add a string to a list.
- `verses = verses.append(...)` → `verses` becomes `None` after the first step (`append` changes the list and
  returns `None`), and the second step fails with `AttributeError: 'NoneType' object has no attribute 'append'`.
- Keeping `song = ""` from the textbook and ending up with one long string again — check `type()` and `len()`.
- `sing_verse(i)` *printed* inside the loop instead of stored — nothing accumulates.
"""),
]

PA62_Q2 = [
    md("""
### Solution 2 — `sqrt_pos_unvec()` and a `for` loop

**Approach.** The function handles **one** value: if it is positive return its square root, otherwise return it
unchanged. Because of the `if`, the function cannot take a whole array, so a `for` loop feeds it one value at a
time and collects the results.
"""),
    code("""
def sqrt_pos_unvec(val):
  if val > 0:
    return np.sqrt(val)
  return val                                 # not positive: leave it alone

sqrt_pos_unvec(7), sqrt_pos_unvec(-2), sqrt_pos_unvec(0)
"""),
    code("""
a_vec = np.array([-2, 1, -3, -9, 7])

result = []
for val in a_vec:
  result = result + [sqrt_pos_unvec(val)]

result = np.array(result)                    # back to a numpy array
result
"""),
    code("""
sqrt_pos_unvec(a_vec)                        # why the loop is needed: the if cannot judge a whole array
""", raises=True),
    md("""
**What the code does.**
- `if val > 0:` compares **one number** to 0, so it gives one `True`/`False` — exactly what `if` needs.
- The second `return val` runs only when the `if` did not return — the non-positive values pass through.
- The loop is the textbook's accumulate pattern: empty list → add one result per value → convert to an array.
- Handing the whole array to the function raises `ValueError: The truth value of an array with more than one
  element is ambiguous` — the same error as Section 7.2.2, and the reason this function is *unvectorized*.

**Expected output.** `(np.float64(2.6457513110645907), -2, 0)` for the single-value tests, and
`array([-2., 1., -3., -9., 2.64575131])` from the loop.

**Common mistakes.**
- **No `else` path** — a function whose only `return` is inside the `if` returns `None` for negatives, so the
  result is `[-2 → None, ...]`. ("Returns the square root if positive" still needs a decision for the rest.)
- `result.append(...)` with `result = result.append(...)` — the `None` problem from Question 1.
- `np.sqrt(val)` on a negative value — `nan` plus a `RuntimeWarning`, which is what happens if the `if` is
  written as `if val > 0: val = np.sqrt(val)` and the function then returns `np.sqrt(val)` anyway.
- Writing the loop over `range(len(a_vec))` and indexing — works, but `for val in a_vec` is the Python idiom.
"""),
]

PA62_Q3 = [
    md("""
### Solution 3 — `sqrt_pos_vec()` without a loop

**Approach.** Replace the `if` with a **boolean mask** (Section 7.2.2): `vec > 0` is an array of `True`/`False`,
and it can select exactly the positive entries to overwrite. Two details make it safe: work on a **copy** so the
caller's array is not changed, and make it a **float** array so square roots are not truncated to integers.
"""),
    code("""
def sqrt_pos_vec(vec):
  out = np.array(vec, dtype = float)         # a float COPY: sqrt(7) must not become 2
  is_pos = out > 0                           # one True/False per element
  out[is_pos] = np.sqrt(out[is_pos])         # replace only the positive entries
  return out

a_vec = np.array([-2, 1, -3, -9, 7])
sqrt_pos_vec(a_vec)
"""),
    code("""
print(a_vec)                                 # the input is untouched ...
print(sqrt_pos_vec([-4, 16, 0, 2.25]))      # ... and a plain list works too
"""),
    md("""
Why the two details matter — the textbook's version, applied to an integer array **in place**:
"""),
    code("""
a_int = np.array([-2, 1, -3, -9, 7])
is_pos = a_int > 0
a_int[is_pos] = np.sqrt(a_int[is_pos])
a_int, a_int.dtype                          # sqrt(7) = 2.64... was stored as 2, and a_int itself changed
"""),
    md("""
**What the code does.**
- `np.array(vec, dtype=float)` makes a new float array from whatever came in (list or array). Without `dtype=float`,
  an integer input stays integer and numpy **truncates** 2.64 → 2 when storing it. Without the copy, the function
  silently modifies the caller's array (the `a_int` demonstration).
- `out > 0` is vectorized: it compares every element at once and returns a boolean array — no `if` needed.
- `out[is_pos]` on the left selects where to write; on the right it selects what to take the square root of, so
  `np.sqrt` never sees a negative number (no `nan`, no warning).

**Expected output.** `array([-2., 1., -3., -9., 2.64575131])` — the same values as Question 2, in one
vectorized step. The in-place integer version gives `array([-2, 1, -3, -9, 2])` with dtype `int64`.

**Common mistakes.**
- **`if vec > 0:` inside the function** → the ambiguous-truth-value `ValueError`. That is the loop-shaped
  thinking the question asks you to drop.
- **Integer truncation** — `array([-2, 1, -3, -9, 2])` looks plausible enough that students do not notice.
- **`np.sqrt(vec)` then masking** — computes square roots of the negatives first: `RuntimeWarning: invalid
  value encountered in sqrt`. Mask first, then square-root.
- **Modifying the input** — the caller's `a_vec` changes; a function should return a new object (Week 6 scope).
- `np.where(vec > 0, np.sqrt(vec), vec)` is a one-line alternative but still evaluates `np.sqrt` on the whole
  array (warning); `np.where(vec > 0, np.sqrt(np.clip(vec, 0, None)), vec)` avoids it.
"""),
]

PA62_Q4 = [
    md("""
### Solution 4 — `sing_verse_3()`, `map()` over three lists, and unequal lengths

**Approach.** Add a third parameter, `container`, and use it wherever the verse said "bottles". `map()` accepts
as many iterables as the function has parameters and walks them **in step**: the first number with the first
drink and the first container, and so on.
"""),
    code("""
def sing_verse_3(num, drink, container):
  song = str(num) + " " + container + " of " + drink + " on the wall \\n"
  song = song + str(num) + " " + container + " of " + drink + "\\n"
  song = song + " take one down, pass it around, \\n"
  song = song + str(num-1) + " " + container + " of " + drink + " on the wall \\n"
  return song

print(sing_verse_3(99, "beer", "bottles"))          # (a) one verse, to test the function
"""),
    code("""
nums = range(100, 97, -1)
drinks = ["beer", "milk", "lemonade"]
containers = ["bottles", "cartons", "cans"]

song = map(sing_verse_3, nums, drinks, containers)   # (b) three iterables -> three arguments, in step
print("".join(list(song)))
"""),
    code("""
containers_2 = ["bottles", "cans"]                   # (c) three drinks, two containers

song = list(map(sing_verse_3, nums, drinks, containers_2))
print(len(song), "verses")
print("".join(song))
"""),
    md("""
No error — `map()` quietly **stops at the end of the shortest iterable**: two containers, two verses; the
third drink ("lemonade") and the third number (98) are never used. If you want that to be an error instead,
Python 3.14 added `strict=True` to `map()` (in Colab, which runs an older Python, use
`zip(nums, drinks, containers_2, strict=True)` to get the same check):
"""),
    code("""
list(map(sing_verse_3, nums, drinks, containers_2, strict = True))     # Python 3.14+
""", raises=True),
    md("""
**What the code does.**
- `sing_verse_3(num, drink, container)` builds the verse one line at a time, `container` replacing the hard-coded
  "bottles". The textbook's spacing (`" on the wall \\n"`) is kept so the output matches the book.
- `map(sing_verse_3, nums, drinks, containers)` — the function, then **one iterable per parameter, in the same
  order as the parameters**. Step 1 calls `sing_verse_3(100, "beer", "bottles")`, step 2
  `sing_verse_3(99, "milk", "cartons")`, step 3 `sing_verse_3(98, "lemonade", "cans")`.
- `list(...)` realizes the lazy map object; `"".join(...)` glues the verses; `print` renders the `\\n`s.

**Expected output.** (a) The 99-bottles verse. (b) Three verses: 100 bottles of beer, 99 cartons of milk,
98 cans of lemonade. (c) **2 verses** (beer in bottles, milk in cans) — silently, no error. With `strict=True`:
`ValueError: map() argument 3 is shorter than arguments 1-2`.

**Common mistakes.**
- **Iterables in the wrong order** — `map(sing_verse_3, drinks, nums, containers)` → `"beer" - 1` →
  `TypeError: unsupported operand type(s) for -: 'str' and 'int'`. Match the order of the parameters.
- **Passing the lists as one argument** — `map(sing_verse_3, [nums, drinks, containers])` → `TypeError:
  sing_verse_3() missing 2 required positional arguments`.
- Expecting (c) to error or to recycle the containers (R-style recycling does not happen in Python).
- Forgetting `list()` and printing the map object itself: `<map object at 0x...>`.
"""),
]

PA62_Q5 = [
    md("""
### Solution 5 — a lambda to hold two arguments fixed

**Approach.** Only the number should change; drink and container stay "milk" and "glasses". A lambda wraps
`sing_verse_3` into a one-argument function for `map()`, so no throw-away `sing_verse_milk_glasses()` is needed.
"""),
    code("""
song = map(lambda i: sing_verse_3(i, "milk", "glasses"), range(100, 97, -1))
print("".join(list(song)))
"""),
    md("""
**What the code does.**
- `lambda i: sing_verse_3(i, "milk", "glasses")` is an anonymous function with **one** parameter, `i`. Each time
  `map` calls it with a number, it calls `sing_verse_3` with that number and the two fixed strings.
- Because the lambda takes one argument, `map` needs only **one** iterable — the numbers.

**Expected output.** Three verses: 100, 99, and 98 glasses of milk on the wall.

**Common mistakes.**
- `map(sing_verse_3, nums, "milk", "glasses")` — strings *are* iterables (of characters), so this runs and
  produces verses about `"m"` in `"g"`, then stops after 4 characters. A memorable bug to show in class.
- `map(lambda i: sing_verse_3(i, "milk", "glasses")(nums))` or calling the lambda immediately — the function
  must be *passed* to `map`, not called (Topic 6.1, Debug (c)).
- Writing `lambda i: return sing_verse_3(...)` → `SyntaxError`; a lambda's body is an expression, no `return`.
- Listing the fixed arguments in the wrong order: `sing_verse_3(i, "glasses", "milk")` sings about "milk of
  glasses".
"""),
]

PA62_Q6 = [
    md("""
### Solution 6 — a function that labels one penguin

**Approach.** Write a function for **one penguin** — the pieces of information it needs become the parameters —
and check the categories **in order**, returning as soon as one matches. The order matters only for the last two:
"Average Adelie" is defined as an Adelie that is *not* a Billy or a Daisy, so those are tested first.
"""),
    code("""
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
"""),
    code("""
def penguin_category(species, sex, bill_length, bill_depth, flipper_length, body_mass):
  \"\"\"
  Label one penguin.

  Parameters
  ----------
  species, sex : str
    From the penguins data ("Adelie", ...; "male"/"female").
  bill_length, bill_depth, flipper_length : float
    Measurements in mm.
  body_mass : float
    Body mass in grams.

  Returns
  -------
  str
    "Big Mouth Billy", "Dainty Daisy", "Average Adelie", or "Other".
  \"\"\"
  if sex == "male" and bill_length * bill_depth > 800:
    return "Big Mouth Billy"
  if sex == "female" and flipper_length < 0.05 * body_mass:
    return "Dainty Daisy"
  if species == "Adelie":
    return "Average Adelie"
  return "Other"
"""),
    code("""
# Unit tests: one hand-made penguin per category
print(penguin_category("Gentoo", "male", 50.0, 17.0, 230, 6000))     # 50*17 = 850 > 800 -> Billy
print(penguin_category("Adelie", "female", 36.0, 17.0, 180, 3800))    # 180 < 0.05*3800 = 190 -> Daisy
print(penguin_category("Adelie", "male", 39.0, 18.0, 190, 3700))      # 39*18 = 702 -> not Billy; Adelie -> Average
print(penguin_category("Chinstrap", "female", 46.0, 17.0, 195, 3500)) # 195 > 175 -> not Daisy; not Adelie -> Other
"""),
    code("""
# ... and the first real penguin in the data
first = penguins.iloc[0]
first["species"], first["sex"], penguin_category(first["species"], first["sex"], first["bill_length_mm"],
                                                 first["bill_depth_mm"], first["flipper_length_mm"], first["body_mass_g"])
"""),
    md("""
**What the code does.**
- Each parameter is one piece of "information about a penguin"; naming them after the columns keeps Question 7
  easy to write.
- `sex == "male" and bill_length * bill_depth > 800` — both conditions in one `if`; the first `return` that fires
  ends the function, so later tests are skipped.
- The two sexes make Billy and Daisy mutually exclusive, so their order does not matter; "Average Adelie" must
  come **after** them, and the final `return "Other"` catches everything else.
- A penguin with a **missing sex** (`NaN`, 11 of them) fails both `==` tests and lands in Adelie/Other —
  reasonable, and worth saying out loud.

**Expected output.** `Big Mouth Billy`, `Dainty Daisy`, `Average Adelie`, `Other` for the four test penguins; the
first real penguin (an Adelie male, 39.1 × 18.7 = 731) is an `Average Adelie`.

**Common mistakes.**
- **Writing the function for the whole column** — `if penguins["sex"] == "male"` → the ambiguous-truth-value
  error from Question 3. This function is deliberately *un*vectorized; that is why Question 7 needs `.apply`.
- **Testing `species == "Adelie"` first** — then every Adelie is "Average", including the 19 Billys and 6 Daisys.
- **Capitalization** — the data has `"male"`/`"female"` and `"Adelie"`; `"Male"` matches nothing.
- **Body-mass units** — 5% of body mass in *grams* (e.g. 190) is compared with flipper length in *mm*; that is
  the definition as given, even though the units differ. Students who "fix" the units get different counts.
- Returning the labels with inconsistent spelling/spaces, which splits the counts in Question 8.
"""),
]

PA62_Q7 = [
    md("""
### Solution 7 — `.apply()` across rows

**Approach.** Section 7.5: `.apply(..., axis=1)` hands the function **one row at a time** as a Series. Since
`penguin_category` wants six separate arguments, a lambda unpacks the row's columns into them.
"""),
    code("""
penguins["category_name"] = penguins.apply(
  lambda row: penguin_category(row["species"], row["sex"], row["bill_length_mm"], row["bill_depth_mm"],
                               row["flipper_length_mm"], row["body_mass_g"]),
  axis = 1)

penguins[["species", "sex", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "category_name"]].head(8)
"""),
    md("""
**What the code does.**
- `axis=1` — apply across **rows** (the default, `axis=0`, would hand the function a whole column).
- `lambda row: penguin_category(row["species"], ...)` — `row` is one row as a pandas Series; `row["species"]`
  picks a value out of it. The lambda does the same job as in Question 5: it adapts the function's signature to
  what `.apply` supplies.
- The result has one label per row, in order, so it can be assigned straight into a new column.

**Expected output.** A `category_name` column of strings. In the first eight rows: row 1 is a `Dainty Daisy`
(female, flipper 186 < 0.05 × 3800 = 190), row 5 a `Big Mouth Billy` (male, 39.3 × 20.6 = 809.6 > 800), row 6
another `Dainty Daisy` (181 < 181.25), and the rest — including row 3, whose measurements are all missing — are
`Average Adelie`.

**Common mistakes.**
- **`penguins.apply(penguin_category, axis=1)`** → `TypeError: penguin_category() missing 5 required positional
  arguments` — the textbook's exact error; `.apply` passes the row as *one* argument.
- **Forgetting `axis=1`** → the lambda receives a column, `row["species"]` raises `KeyError: 'species'`.
- **Looping over the data frame** with `for i in range(len(penguins))` and `.loc[i, ...]` — works, but slow and
  verbose; the question asks for an *iterable function*.
- **`penguins.category_name = ...`** on a column that does not exist yet creates an attribute, not a column
  (pandas warns). Use `penguins["category_name"] = ...`.
""" ),
]

PA62_Q8 = [
    md("""
### Solution 8 — counts per category

**Approach.** Counting the values of one column is `value_counts()`. A cross-tab by species is a useful check
that the definitions behaved as intended.
"""),
    code("""
penguins["category_name"].value_counts()
"""),
    code("""
pd.crosstab(penguins["species"], penguins["category_name"])
"""),
    md("""
**What the code does.** `value_counts()` tabulates a column, largest group first. `pd.crosstab` counts
species × category — it shows that every "Average Adelie" is an Adelie (by construction) and that Gentoos
dominate the Daisys (long flippers but heavy bodies).

**Expected output.** **Average Adelie 127, Other 84, Big Mouth Billy 71, Dainty Daisy 62** (344 total). By
species: Adelie 127 / 19 / 6 / 0; Chinstrap 0 / 34 / 2 / 32; Gentoo 0 / 18 / 54 / 52 (Average · Billy · Daisy ·
Other).

**Common mistakes.**
- `penguins.groupby("category_name").count()` — counts every column (and under-counts where values are
  missing); `.size()` or `value_counts()` is what is meant.
- Counts that do not sum to 344 → the `.apply` was run on a filtered or `dropna()`'d copy.
- Different counts from the key usually trace back to Question 6: test order (Adelie first), units, or a typo in
  a label — `value_counts()` will show the stray label as its own row.
"""),
]

PA62_ANSWERS = [PA62_Q1, PA62_Q2, PA62_Q3, PA62_Q4, PA62_Q5, PA62_Q6, PA62_Q7, PA62_Q8]


# ============================================================================ #
#  Practice Activity: Iteration (penguin categories) — the short stand-alone version of PA 6.2 Q6–Q8
# ============================================================================ #

ITER_HEADER = [
    md("""
# GSB 5544 — Practice Activity: Iteration — INSTRUCTOR SOLUTION
*Textbook: [Section 7.5, Iterating on datasets](https://ds-ml-with-python.github.io/Course-Textbook/06-iteration.html#iterating-on-datasets)
— a row-by-row function and `.apply(axis=1)`*
"""),
    md("""
**How to use this notebook.** This activity is the penguin part of PA 6.2 (Questions 6–8) on its own, with the
counting code supplied. The answers below are self-contained — they reload the data and redefine the function —
so this part runs on its own even if the PA 6.2 cells above were skipped. Same five beats: **Approach → code →
What the code does → Expected output → Common mistakes**.
"""),
]

ITER_Q0 = [
    md("""
### Solution 0 — load the data and the libraries

**Approach.** `pandas` for the data frame, `numpy` for the (optional) vectorized check, and `load_penguins` for
the data. In Colab, `palmerpenguins` is not pre-installed: run `!pip install palmerpenguins` once first.
"""),
    code("""
# !pip install palmerpenguins          # once, in Colab
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins

penguins = load_penguins()
print(penguins.shape)
penguins.head()
"""),
    md("""
**Expected output.** `(344, 8)` — 344 penguins; the columns used below are `species`, `sex`, `bill_length_mm`,
`bill_depth_mm`, `flipper_length_mm`, `body_mass_g`. Note the `NaN`s in row 3: two penguins have no measurements
and 11 have no recorded sex.

**Common mistakes.** `import palmerpenguins` then `load_penguins()` → `NameError`; it is
`from palmerpenguins import load_penguins` (or `palmerpenguins.load_penguins()`). Forgetting the `pip install`
in Colab → `ModuleNotFoundError: No module named 'palmerpenguins'`.
"""),
]

ITER_Q1 = [
    md("""
### Solution 1 — a function that labels one penguin

**Approach.** The function describes **one** penguin, so each piece of information is a parameter. Test the
categories in order and `return` at the first match: Billy and Daisy first (they are defined by sex and a
measurement), then "Average Adelie" (an Adelie that was *not* caught above), then "Other" for everyone left.
"""),
    code("""
def penguin_category(species, sex, bill_length, bill_depth, flipper_length, body_mass):
  \"\"\"
  Label one penguin as "Big Mouth Billy", "Dainty Daisy", "Average Adelie", or "Other".

  Parameters
  ----------
  species, sex : str
    "Adelie" / "Chinstrap" / "Gentoo" and "male" / "female", as in the penguins data.
  bill_length, bill_depth, flipper_length : float
    Measurements in mm.
  body_mass : float
    Body mass in grams.

  Returns
  -------
  str
    The category name.
  \"\"\"
  if sex == "male" and bill_length * bill_depth > 800:
    return "Big Mouth Billy"
  if sex == "female" and flipper_length < 0.05 * body_mass:
    return "Dainty Daisy"
  if species == "Adelie":
    return "Average Adelie"
  return "Other"
"""),
    code("""
# Unit tests — one hand-made penguin per category, with the arithmetic in the comment
print(penguin_category("Gentoo", "male", 50.0, 17.0, 230, 6000))      # 50 * 17 = 850 > 800          -> Billy
print(penguin_category("Adelie", "female", 36.0, 17.0, 180, 3800))     # 180 < 0.05 * 3800 = 190      -> Daisy
print(penguin_category("Adelie", "male", 39.0, 18.0, 190, 3700))       # 39 * 18 = 702; Adelie        -> Average Adelie
print(penguin_category("Chinstrap", "female", 46.0, 17.0, 195, 3500))  # 195 > 175; not Adelie        -> Other
"""),
    md("""
**What the code does.**
- The `if` lines read like the definitions: `sex == "male" and bill_length * bill_depth > 800` is "male **and**
  bill area over 800". `and` here joins two single `True`/`False` values — fine for one penguin.
- The first `return` that fires ends the function, so an Adelie Billy is a Billy, not an "Average Adelie".
- The last line, `return "Other"`, needs no `if`: anything that reached it matched nothing above.
- Because this function has `if`s inside, it is **not vectorized** — it cannot take whole columns. That is why
  the next question needs an iterable function.

**Expected output.** `Big Mouth Billy`, `Dainty Daisy`, `Average Adelie`, `Other`.

**Common mistakes.**
- **Checking `species == "Adelie"` first** → every Adelie becomes "Average", swallowing 19 Billys and 6 Daisys.
- **`"Male"` / `"Female"` / `"adelie"`** — the data is lower-case `"male"`/`"female"` and capitalized species.
- **Writing it for the whole column** (`if penguins["sex"] == "male":`) → `ValueError: The truth value of a Series
  is ambiguous`.
- **A missing `return "Other"`** → non-matching penguins get `None`, and `value_counts` silently drops them.
- Comparing flipper length (mm) with 5 % of body mass (g) looks odd, but it is the definition as written; keep it.
"""),
]

ITER_Q2 = [
    md("""
### Solution 2 — `.apply()` across rows

**Approach.** `.apply(..., axis=1)` hands the function one row at a time. The row arrives as a single Series, and
`penguin_category` wants six arguments, so a lambda pulls the six columns out of the row.
"""),
    code("""
penguins["category_name"] = penguins.apply(
  lambda row: penguin_category(row["species"], row["sex"], row["bill_length_mm"], row["bill_depth_mm"],
                               row["flipper_length_mm"], row["body_mass_g"]),
  axis = 1)

penguins[["species", "sex", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "category_name"]].head(8)
"""),
    md("""
**A vectorized cross-check (optional).** The same labels without `.apply`, using boolean masks and `np.select`,
which picks the first condition that is `True` for each row — the vectorized cousin of the `if` chain:
"""),
    code("""
is_billy = (penguins["sex"] == "male") & (penguins["bill_length_mm"] * penguins["bill_depth_mm"] > 800)
is_daisy = (penguins["sex"] == "female") & (penguins["flipper_length_mm"] < 0.05 * penguins["body_mass_g"])
is_adelie = penguins["species"] == "Adelie"

vectorized = np.select([is_billy, is_daisy, is_adelie],
                       ["Big Mouth Billy", "Dainty Daisy", "Average Adelie"], default = "Other")
(vectorized == penguins["category_name"]).all()           # True: both routes agree for every penguin
"""),
    md("""
**What the code does.**
- `axis=1` → iterate over **rows**. (The default `axis=0` would hand the lambda a whole *column*, and
  `row["species"]` would fail with `KeyError`.)
- `lambda row: penguin_category(row["species"], ...)` — the adapter between what `.apply` supplies (one row)
  and what the function expects (six values). It is the textbook's Section 7.5 pattern exactly.
- The result is a Series with one label per row, aligned by index, so it drops straight into a new column.
- `np.select(conditions, choices, default)` evaluates all the masks at once; `&` (not `and`) combines two
  boolean Series element-wise. `(vectorized == penguins["category_name"]).all()` confirms the two routes agree.

**Expected output.** In the first eight rows: rows 1 and 6 are `Dainty Daisy`, row 5 is `Big Mouth Billy`, the
rest are `Average Adelie` (including row 3, whose measurements are all `NaN` — every comparison with `NaN` is
`False`, so it falls through to the species test). The cross-check prints `True`.

**Common mistakes.**
- **`penguins.apply(penguin_category, axis=1)`** → `TypeError: penguin_category() missing 5 required positional
  arguments: ...` — `.apply` passes the row as one argument.
- **Leaving out `axis=1`** → `KeyError: 'species'`.
- **`map(penguin_category, penguins)`** — iterating a data frame yields its *column names*, not rows.
- **`for i in range(len(penguins)): penguins.loc[i, "category_name"] = ...`** — works but slow and not what
  "iterable function" asks for.
- **Using `and`/`or` in the vectorized version** → the ambiguous-truth-value error; element-wise needs `&`/`|`.
"""),
]

ITER_Q3 = [
    md("""
### Solution 3 — the counts

**What the code does.** `DataFrame.value_counts("category_name")` counts how many rows take each value of that
column, largest first — the same numbers as `penguins["category_name"].value_counts()`.

**Expected output.** **Average Adelie 127 · Other 84 · Big Mouth Billy 71 · Dainty Daisy 62** (sums to 344).

**Common mistakes.**
- Counts that do not add to 344 → the labels were computed on a filtered or `dropna()`'d copy of the data.
- An extra row with a misspelled label (`"Dainty Daisy "`, `"Big mouth Billy"`) → a typo in Question 1's strings.
- `NameError: name 'penguins' is not defined` → the given cell was run before Question 0 (or after a kernel restart).
- Different numbers from the key almost always trace back to **test order** in Question 1 (Adelie checked first
  gives Average Adelie 152, Billy 52, Daisy 56, Other 84).
"""),
]

ITER_SRC = WEEK / "Practice_Activity_Iteration.ipynb"
ITER_ANSWERS = [ITER_Q0, ITER_Q1, ITER_Q2]                     # one per empty code cell, in order
ITER_GIVEN = ("code", lambda s: 'value_counts("category_name")' in s, "after", ITER_Q3)   # the supplied cell


def iter_rules() -> list:
    return empty_code_rules(ITER_ANSWERS) + [ITER_GIVEN]


COMBINED_DEST = WEEK / "GSB_5544_Week_6_Iteration_Keys-solution.ipynb"
COMBINED_HEADER = [
    md("""
# GSB 5544 — Week 6 Iteration Keys — INSTRUCTOR SOLUTION
*Two activities from textbook [Chapter 7, Iteration](https://ds-ml-with-python.github.io/Course-Textbook/06-iteration.html), in one notebook*

| | Activity | Questions |
|---|---|---|
| **Part 1** | PA 6.2 — Iteration | 1 – 8: *99 Bottles* verses as a list, `sqrt_pos_unvec` / `sqrt_pos_vec`, `sing_verse_3` with `map()` and a lambda, penguin categories with `.apply` |
| **Part 2** | Practice Activity: Iteration | 0 – 3: the penguin categories on their own (the same task as Part 1, Questions 6 – 8, with the counting code supplied) |

Each part is self-contained — Part 2 reloads the data and redefines its function — so either can be run alone.
Every answer follows the same five beats: **Approach → code → What the code does → Expected output → Common mistakes**.
"""),
]


# ============================================================================ #
#  Build
# ============================================================================ #

METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


# (cell type, test on the source, "replace" the cell or insert "after" it, answer cells) — each must match once
PA61_RULES = [
    ("code", lambda s: s.strip() == "", "replace", PA61_Q0),
    ("code", lambda s: "def times_seven(x):" in s and "____" in s, "replace", PA61_Q1),
    ("markdown", lambda s: "unit tests" in s and "times_seven" in s, "after", PA61_Q2),
    ("code", lambda s: 'add_or_subtract("orange")' in s, "replace", PA61_Q3),
    ("markdown", lambda s: "In your Global Environment" in s, "after", PA61_Q4),
]


def empty_code_rules(groups: list[list[dict]]) -> list:
    """One rule per answer group: the n-th empty code cell of the source is replaced by the n-th group."""
    counter = {"n": 0}

    def make(k: int):
        def test(s: str) -> bool:
            if s.strip() != "":
                return False
            hit = counter["n"] == k
            if hit:
                counter["n"] += 1
            return hit
        return test

    return [("code", make(k), "replace", group) for k, group in enumerate(groups)]


def build_solution(src_path: Path, header: list[dict], rules: list) -> dict:
    nb = json.loads(src_path.read_text())
    out = copy.deepcopy(header)
    used = [0] * len(rules)
    for cell in nb["cells"]:
        src = source(cell)
        hit = next((i for i, (kind, test, _, _) in enumerate(rules) if cell["cell_type"] == kind and test(src)), None)
        if hit is None:
            out.append(copy.deepcopy(cell))
            continue
        used[hit] += 1
        _, _, how, cells = rules[hit]
        if how == "after":
            out.append(copy.deepcopy(cell))
        out.extend(copy.deepcopy(cells))
    if used != [1] * len(rules):
        raise SystemExit(f"{src_path.name}: each rule must match exactly one cell, got {used}")
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:03d}"
        c["source"] = source(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = out
    nb["metadata"] = copy.deepcopy(METADATA)
    nb["nbformat"], nb["nbformat_minor"] = 4, 5          # cell ids need nbformat 4.5
    return nb


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


def combine(header: list[dict], parts: list[tuple[str, dict]]) -> dict:
    """Join several solution notebooks into one. Each part's title cell becomes a '## Part n — ...' heading."""
    out = copy.deepcopy(header)
    for n, (label, part) in enumerate(parts, start=1):
        cells = copy.deepcopy(part["cells"])
        first = cells[0]
        assert first["cell_type"] == "markdown" and first["source"].startswith("# ")
        first["source"] = re.sub(r"^# GSB 5544 — ", f"---\n## Part {n} — ", first["source"], count=1)
        out.extend(cells)
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:03d}"
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return {"cells": out, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    jobs = [
        (PA61_SRC, PA61_HEADER, PA61_RULES),
        (PA62_SRC, PA62_HEADER, empty_code_rules(PA62_ANSWERS)),
    ]
    for src, header, rules in jobs:
        dest = src.with_name(src.stem + "-solution.ipynb")
        save(dest, build_solution(src, header, rules))
        print(f"wrote {dest.relative_to(ROOT)}")
        if "--no-exec" not in sys.argv:
            execute(dest)

    # One notebook holding both iteration keys: PA 6.2, then the stand-alone Practice Activity: Iteration
    combined = combine(COMBINED_HEADER, [
        ("PA 6.2", build_solution(PA62_SRC, PA62_HEADER, empty_code_rules(PA62_ANSWERS))),
        ("Iteration", build_solution(ITER_SRC, ITER_HEADER, iter_rules())),
    ])
    save(COMBINED_DEST, combined)
    print(f"wrote {COMBINED_DEST.relative_to(ROOT)}")
    if "--no-exec" not in sys.argv:
        execute(COMBINED_DEST)


if __name__ == "__main__":
    main()
