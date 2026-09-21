#!/usr/bin/env python3
r"""Generate the Topic 4.2 *extra practice* notebooks: strings and regular expressions.

Usage:  python3 tools/build_week4_regex_practice.py            # write both notebooks, execute the solutions
        python3 tools/build_week4_regex_practice.py --no-exec  # write both without executing

Output (assignments/practice_activities/week_4/):
  GSB5544_Topic_4_2_Strings_and_Regex_Extra_Practice-empty.ipynb     the practice section: numbered exercises with ____ blanks
  GSB5544_Topic_4_2_Strings_and_Regex_Extra_Practice-solution.ipynb  the solutions section: same numbering, completed code,
                                                                     executed output, and a short explanation of each answer
Covers every method, function, and regex piece introduced in Topic 4.2, plus two tools PA 4.2 needs
(`.str.count` and the `\1` back-reference), which are labelled as previews.

«text» in an exercise's code is the blank: "text" in the solutions, "____" in the practice notebook.
Every exercise ends with an expression; before writing anything this script runs the completed code and
checks that expression against the exercise's `expect`, so the "Intended result" lines cannot drift.
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
STEM = "GSB5544_Topic_4_2_Strings_and_Regex_Extra_Practice"
JUPYTER = "/opt/anaconda3/bin/jupyter"

ITEMS: list[dict] = []


def part(title: str, text: str) -> None:
    ITEMS.append({"kind": "part", "title": title, "text": text.strip()})


def setup(code: str) -> None:
    ITEMS.append({"kind": "setup", "code": code.strip("\n")})


def ex(title: str, task: str, code: str, expect, why: str) -> None:
    ITEMS.append({"kind": "ex", "title": title, "task": task.strip(), "code": code.strip("\n"),
                  "expect": expect, "why": why.strip()})


# ============================================================================ #
#  Part 1 — one string: the core methods
# ============================================================================ #

part("Part 1 — One string: the core methods", r"""
Plain Python, one string at a time: `len`, indexing and slicing, `lower` / `upper`, `strip`, `replace`,
`split`, `join`, `startswith` / `endswith` — and the habit of **reassigning** the result.
""")

ex("`len`", r"""
Count the characters in `s`, spaces included.
""", r"""
s = "  Data Science  "
«len»(s)
""", 16, r"""
`len` is a built-in *function* (not a method), so the string goes inside the parentheses. It counts every
character — the 2 leading spaces, the 12 characters of `Data Science` (the inner space included), and the 2 trailing spaces.
""")

ex("Indexing", r"""
A string is a sequence, so positions work. Get the first and the last character of `word`.
""", r"""
word = "regex"
(word[«0»], word[«-1»])
""", ("r", "x"), r"""
Positions start at `0`, and negative positions count from the end, so `-1` is always the last character
whatever the length.
""")

ex("Slicing from the start", r"""
Keep only the first five characters of `title`.
""", r"""
title = "vaudevillian"
title[«:5»]
""", "vaude", r"""
`[:5]` means "from the start up to, but not including, position 5" — positions 0, 1, 2, 3, 4. This is how you
truncate a word to a maximum length (PA 4.2, step 3).
""")

ex("Slicing out of the middle", r"""
Pull the four-digit year out of the invoice code.
""", r"""
code = "INV-2024-00931"
code[«4:8»]
""", "2024", r"""
`INV-` occupies positions 0–3, so the year starts at position 4; the slice stops *before* position 8, giving
positions 4, 5, 6, 7.
""")

ex("`lower` and `upper`", r"""
Produce an all-lowercase and an all-uppercase version of `name`.
""", r"""
name = "McDonald"
(name.«lower»(), name.«upper»())
""", ("mcdonald", "MCDONALD"), r"""
Both are *methods*: they come after the string and a dot, and need the empty parentheses to actually run.
""")

ex("`strip`", r"""
Remove the spaces at both ends of `raw`.
""", r"""
raw = "   hello world   "
raw.«strip»()
""", "hello world", r"""
`strip()` removes whitespace from the **two ends only**; the space between `hello` and `world` is in the middle,
so it stays.
""")

ex("`strip` changes the length", r"""
Compare the length of `raw` before and after stripping.
""", r"""
raw = "   hello world   "
(len(raw), len(raw.«strip»()))
""", (17, 11), r"""
17 characters with padding, 11 without: three spaces came off each end. Methods can be used inside a function
call — `raw.strip()` produces a new string, and `len` measures that.
""")

ex("`replace`", r"""
Change every `cats` to `dogs`.
""", r"""
text = "I like cats. cats are great."
text.«replace»("cats", «"dogs"»)
""", "I like dogs. dogs are great.", r"""
`replace(old, new)` swaps **every** occurrence of the exact text `old`, not just the first one.
""")

ex("`replace` to delete", r"""
Delete the dashes from the phone number by replacing them with *nothing*.
""", r"""
phone = "805-555-0199"
phone.replace(«"-"», «""»)
""", "8055550199", r"""
Replacing with the empty string `""` is how you delete text. The result is still a string, not a number.
""")

ex("`split`", r"""
Break the comma-separated line into a list of colours.
""", r"""
line = "red,green,blue"
line.«split»(«","»)
""", ["red", "green", "blue"], r"""
`split(sep)` cuts the string wherever `sep` occurs and returns a **list** of the pieces; the separators
themselves are thrown away.
""")

ex("`split` to count words", r"""
Count the words in `sentence` by splitting on spaces and measuring the list.
""", r"""
sentence = "the quick brown fox"
len(sentence.«split»(" "))
""", 4, r"""
Splitting on `" "` gives `['the', 'quick', 'brown', 'fox']`, and `len` of a list is its number of elements.
""")

ex("`join`", r"""
Glue the pieces into one date string with dashes between them.
""", r"""
pieces = ["2026", "09", "17"]
«"-"».«join»(pieces)
""", "2026-09-17", r"""
`join` is a method of the **separator**, and the list goes in the parentheses: `"-".join(pieces)`. It is the
reverse of `split`.
""")

ex("`split`, then `join`", r"""
Replace the spaces in `text` with underscores by splitting on `" "` and joining with `"_"`.
""", r"""
text = "week four notes"
"_".join(text.«split»(" "))
""", "week_four_notes", r"""
`split(" ")` gives `['week', 'four', 'notes']`; `"_".join(...)` stitches the list back together with the new
separator. (`text.replace(" ", "_")` gets the same answer — two routes, one result.)
""")

ex("`startswith` and `endswith`", r"""
Test whether the file name starts with `report` and whether it ends with `.csv`.
""", r"""
fname = "report_final.csv"
(fname.«startswith»("report"), fname.«endswith»(".csv"))
""", (True, True), r"""
Both return `True` or `False`. They test exact text at one end of the string — no patterns involved.
""")

ex("Strings are immutable", r"""
The second line *looks* like it strips `s` but throws the result away. Fix it so that `s` really is stripped.
""", r"""
s = "  padded  "
s = «s.strip()»
len(s)
""", 6, r"""
No string method changes the original; each returns a **new** string. Writing `s.strip()` alone computes the
answer and discards it — you must reassign with `s = s.strip()`. Without the fix, `len(s)` would still be 10.
""")

ex("Chaining methods", r"""
Strip `messy` and make it lowercase in a single expression.
""", r"""
messy = "   ThE DuDe AbIdEs.  "
messy.«strip»().«lower»()
""", "the dude abides.", r"""
Each method returns a string, so another method can follow immediately. Chains run **left to right**: strip
first, then lowercase the stripped result.
""")

ex("A longer chain", r"""
From `messy`, get the **last word**, in capitals.
""", r"""
messy = "   ThE DuDe AbIdEs.  "
messy.strip().split(" ")[«-1»].«upper»()
""", "ABIDES.", r"""
`strip()` first (otherwise the trailing spaces would create empty pieces), `split(" ")` makes a list, `[-1]`
takes its last element, and `.upper()` works on that string. The full stop is part of the "word" — `split`
knows nothing about punctuation.
""")

# ============================================================================ #
#  Part 2 — patterns with re.findall
# ============================================================================ #

part("Part 2 — Regular expressions: finding with `re.findall`", r"""
`re.findall(pattern, s)` returns a list of every match. Write patterns as **raw strings**, `r"..."`.
The pieces: literal text, sets `[...]`, negated sets `[^...]`, `\w`, `\s`, `.`, the quantifiers `+`, `*`, `{n}`,
and the anchors `^` and `$`.
""")

setup("import re")

ex("A literal pattern", r"""
Find every occurrence of the text `the`.
""", r"""
text = "the cat sat on the mat with the hat"
re.«findall»(r"«the»", text)
""", ["the", "the", "the"], r"""
Ordinary letters in a pattern match themselves. `findall` returns one list element per match, in order.
""")

ex("A pattern matches anywhere", r"""
Predict first, then count: how many times does the pattern `the` match in `text`?
""", r"""
text = "other theory, the end"
len(re.findall(r"«the»", text))
""", 3, r"""
Three: inside `o-the-r`, inside `the-ory`, and the standalone `the`. A regex has no idea what a *word* is — a
literal pattern matches wherever those characters appear.
""")

ex("A set of characters", r"""
Find every vowel, one at a time.
""", r"""
text = "regular expression"
re.findall(r"«[aeiou]»", text)
""", ["e", "u", "a", "e", "e", "i", "o"], r"""
Square brackets mean "any **one** character from this set". Each vowel is its own match, so the list has
seven single-character elements.
""")

ex("A set of digits", r"""
Find every digit in `text`, one at a time.
""", r"""
text = "Room 4B, floor 12"
re.findall(r"«[0123456789]»", text)
""", ["4", "1", "2"], r"""
The set can hold any characters, here the ten digits. `12` comes back as `'1'` and `'2'` because a set matches
**one** character. (Shorthand you will see elsewhere: `[0-9]` is the same set.)
""")

ex("A negated set", r"""
Find every character of `banana` that is **not** a vowel.
""", r"""
re.findall(r"«[^aeiou]»", "banana")
""", ["b", "n", "n"], r"""
A `^` as the **first thing inside the brackets** flips the set: "any one character *except* these".
""")

ex("The punctuation class", r"""
Find every punctuation mark: any character that is neither a word character nor whitespace.
""", r"""
text = "Wait... what?! No way; really."
re.findall(r"«[^\w\s]»", text)
""", [".", ".", ".", "?", "!", ";", "."], r"""
`\w` is a word character (letter, digit, underscore) and `\s` is whitespace. `[^\w\s]` is "not a word character
and not whitespace" — which leaves punctuation. The backslashes are why the pattern is a raw string: in
`r"..."` Python passes `\w` to the regex engine untouched.
""")

ex("`\\s` — whitespace", r"""
Count the whitespace characters in `text` (a space, a tab, and a newline are all whitespace).
""", r"""
text = "a b\tc\nd"
len(re.findall(r"«\s»", text))
""", 3, r"""
`\s` matches any single whitespace character: the space, the tab `\t`, and the newline `\n`. This is how the
PA's "how many characters are white space?" question is answered.
""")

ex("`\\w` — word characters", r"""
Count the word characters in `text`.
""", r"""
text = "hi there!"
len(re.findall(r"«\w»", text))
""", 7, r"""
`\w` matches letters, digits, and the underscore — `h i t h e r e` is seven. The space and the `!` are not
word characters.
""")

ex("`.` — any character", r"""
Match `c`, then **any one character**, then `t`.
""", r"""
text = "cat cot cut c_t"
re.findall(r"«c.t»", text)
""", ["cat", "cot", "cut", "c_t"], r"""
The dot is a wildcard for exactly one character of any kind, so `a`, `o`, `u`, and `_` all qualify.
""")

ex("`+` — one or more", r"""
Match `g` followed by **one or more** `o`s.
""", r"""
text = "go goo gooooal g"
re.findall(r"«go+»", text)
""", ["go", "goo", "goooo"], r"""
`+` applies to the single character before it: `o+` is "one or more `o`". The `+` is greedy — it takes as many
`o`s as it can. The lone `g` at the end has no `o`, so it does not match.
""")

ex("`*` — zero or more", r"""
Same text, but now allow **zero or more** `o`s. What extra match appears?
""", r"""
text = "go goo gooooal g"
re.findall(r"«go*»", text)
""", ["go", "goo", "goooo", "g"], r"""
`o*` also accepts *no* `o` at all, so the lone `g` now matches. `+` = at least one; `*` = optional and repeatable.
""")

ex("`\\w+` — whole words", r"""
Extract the words, leaving the punctuation and spaces behind.
""", r"""
text = "Voila! In view, a humble veteran."
re.findall(r"«\w+»", text)
""", ["Voila", "In", "view", "a", "humble", "veteran"], r"""
`\w+` is "a run of one or more word characters". A run ends at the first space or punctuation mark, so each
match is one word.
""")

ex("`{n}` — exactly n", r"""
Find every occurrence of exactly two `e`s in a row.
""", r"""
text = "meet me at the beet tree"
re.findall(r"«e{2}»", text)
""", ["ee", "ee", "ee"], r"""
`e{2}` is "the character `e`, exactly twice" — the same as writing `ee`. It matches in `meet`, `beet`, and
`tree`; the single `e`s in `me` and `the` do not qualify.
""")

ex("A literal followed by a quantifier", r"""
Match the groans: `ug` followed by one or more `h`.
""", r"""
groans = "ugh ughh ughhhh! Ugh"
re.findall(r"«ugh+»", groans)
""", ["ugh", "ughh", "ughhhh"], r"""
`ug` is literal; `h+` is one or more `h`. The `+` belongs only to the `h`, not to the whole word. The last
groan, `Ugh`, is missed — see the next exercise.
""")

ex("Regex is case-sensitive", r"""
Change the pattern so that the capitalised `Ugh` is caught too.
""", r"""
groans = "ugh ughh ughhhh! Ugh"
re.findall(r"«[Uu]»gh+", groans)
""", ["ugh", "ughh", "ughhhh", "Ugh"], r"""
A lowercase `u` in a pattern only matches a lowercase `u`. The set `[Uu]` accepts either case for that one
position; the rest of the pattern is unchanged.
""")

ex("`^` — at the start", r"""
Match `the` only when it is at the **start** of the string.
""", r"""
text = "the end of the road"
re.findall(r"«^the»", text)
""", ["the"], r"""
Outside brackets, `^` is an **anchor**: it matches a position (the start of the string), not a character. The
second `the` is in the middle, so only one match comes back. (Inside brackets, `^` means "not" — same symbol,
different job.)
""")

ex("`$` — at the end", r"""
Match `ing` only when it is at the **end** of the string.
""", r"""
text = "singing and dancing"
re.findall(r"«ing$»", text)
""", ["ing"], r"""
`$` anchors to the end of the string. `singing` contains `ing` twice, but neither is at the very end; only the
`ing` of `dancing` is.
""")

ex("Escaping a special character", r"""
Match the version number `1.2` **literally**. An unescaped dot would also match `1x2`.
""", r"""
text = "v1.2 and v1x2"
re.findall(r"1«\.»2", text)
""", ["1.2"], r"""
`.` normally means "any character", so `1.2` would match both `1.2` and `1x2`. A backslash turns a special
character back into a literal: `\.` is "an actual full stop". In a raw string you type one backslash; in an
ordinary string you would need `"1\\.2"`.
""")

ex("Escaping `$`", r"""
Find the prices: a literal dollar sign followed by one or more word characters.
""", r"""
text = "cost: $5 and $12"
re.findall(r"«\$»\w+", text)
""", ["$5", "$12"], r"""
`$` is the end-of-string anchor, so to match a real dollar sign it must be escaped as `\$`. Then `\w+` takes
the digits that follow. The same rule applies to the other special characters: `^ $ . + * [ ] { }`.
""")

ex("Combining a quantifier and a class", r"""
Match each groan **together with the punctuation mark that follows it**.
""", r"""
text = "rough day, ugh, enough!"
re.findall(r"«ugh+[^\w\s]»", text)
""", ["ugh,", "ugh!"], r"""
`ugh+` then exactly one punctuation character. The `ugh` in `rough` is followed by a space, so it is left
alone — but the one in `enough!` is followed by `!` and **is** caught. Testing with `findall` first shows you
that this pattern would damage `enough` before you ever run a replacement.
""")

# ============================================================================ #
#  Part 3 — replacing with re.sub
# ============================================================================ #

part("Part 3 — Regular expressions: replacing with `re.sub`", r"""
`re.sub(pattern, replacement, s)` replaces every match. Same patterns as Part 2 — now they change the string.
""")

ex("`re.sub` with a literal", r"""
Turn the dashes in the date into slashes.
""", r"""
date = "2026-09-17"
re.«sub»(r"-", «"/"», date)
""", "2026/09/17", r"""
The argument order is pattern, replacement, string. With a purely literal pattern this does the same job as
`date.replace("-", "/")`.
""")

ex("Delete all punctuation", r"""
Remove every punctuation mark by replacing it with the empty string.
""", r"""
text = "Hello, world! Ready?"
re.sub(r"«[^\w\s]»", "", text)
""", "Hello world Ready", r"""
`[^\w\s]` matches one punctuation character at a time; replacing each with `""` deletes it. The spaces survive
because `\s` is excluded from the match.
""")

ex("Collapse repeated spaces", r"""
Replace every **run** of whitespace with a single space.
""", r"""
text = "too    many   spaces"
re.sub(r"«\s+»", " ", text)
""", "too many spaces", r"""
`\s+` matches a whole run of whitespace as one match, and the run is replaced by one space. With `\s` alone
(no `+`) each space would be replaced by a space and nothing would change.
""")

ex("Delete the groans", r"""
Delete every groan together with its punctuation mark — either capitalisation.
""", r"""
text = "Fineugh! I will go.Ughhh? Ok"
re.sub(r"«[Uu]gh+[^\w\s]»", "", text)
""", "Fine I will go. Ok", r"""
`[Uu]` = either case, `gh+` = `g` and one or more `h`, `[^\w\s]` = the punctuation mark after it. The full stop
after `go` is kept because it comes *before* the groan and is not part of any match.
""")

ex("Anchored replacement", r"""
Change a `k` to a `v` — but only a `k` at the **start** of the word.
""", r"""
word = "kicking"
re.sub(r"«^k»", "v", word)
""", "vicking", r"""
`^k` matches a `k` only at position 0, so the second `k` is untouched. `word.replace("k", "v")` would have
produced `vicving`.
""")

ex("Anchored replacement at the end", r"""
Change a final `b` into a `y`, leaving other `b`s alone.
""", r"""
word = "bubblb"
re.sub(r"«b$»", "y", word)
""", "bubbly", r"""
`b$` is "a `b` that is the last character". The three earlier `b`s are not at the end, so they stay.
""")

# ============================================================================ #
#  Part 4 — a whole column: the .str accessor
# ============================================================================ #

part("Part 4 — A whole column at once: the `.str` accessor", r"""
The same tools applied to every element of a pandas Series: `.str.strip`, `.str.len`, `.str.upper`,
`.str.slice`, `.str.contains`, `.str.replace` (remember `regex=True`), and `.apply` + `lambda` for the rest.
Run the set-up cells as they appear; later exercises use `words` and `clean`.
""")

setup(r"""
import pandas as pd

words = pd.Series(["  Koila! ", "kiew,", "  humble  ", "kezeranugh!", "bozh", "kicarb"])
words
""")

ex("`.str.strip`", r"""
Strip the padding from every element of `words`.
""", r"""
words.«str.strip»()
""", ["Koila!", "kiew,", "humble", "kezeranugh!", "bozh", "kicarb"], r"""
`.str` is the doorway from a Series to the string methods: `words.str.strip()` runs `strip()` on each element
and returns a new Series. `words.strip()` without `.str` is an error.
""")

setup(r"""
clean = words.str.strip()      # set-up: run this whether or not you finished the previous exercise
""")

ex("`.str.len`", r"""
Get the length of every word in `clean`.
""", r"""
clean.«str.len»()
""", [6, 5, 6, 11, 4, 6], r"""
There is no `len(...)` for "each element", so pandas supplies `.str.len()`. `len(clean)` would instead return
6 — the number of *elements* in the Series.
""")

ex("Total characters", r"""
How many characters are there in the whole Series?
""", r"""
clean.str.len().«sum»()
""", 38, r"""
`.str.len()` gives a Series of numbers, and `.sum()` adds them — the pattern for PA 4.2's "how many characters
are in the scrambled message?"
""")

ex("`.str.upper`", r"""
Put every word in capitals.
""", r"""
clean.«str.upper»()
""", ["KOILA!", "KIEW,", "HUMBLE", "KEZERANUGH!", "BOZH", "KICARB"], r"""
Same idea as `.str.strip()`: the plain string method `upper`, applied element by element.
""")

ex("`.str.slice`", r"""
Truncate every word to at most four characters.
""", r"""
clean.str.«slice»(0, «4»)
""", ["Koil", "kiew", "humb", "keze", "bozh", "kica"], r"""
`.str.slice(0, 4)` is the Series version of `word[0:4]`. Words that are already short (`bozh`) are returned
unchanged.
""")

ex("`.str.contains` with an anchor", r"""
Which words start with a lowercase `k`? Return `True`/`False` for each.
""", r"""
clean.str.«contains»(r"«^k»")
""", [False, True, False, True, False, True], r"""
`.str.contains` treats its argument as a regex by default, so `^k` means "starts with `k`". `Koila!` is
`False` because the pattern is case-sensitive.
""")

ex("`case=False`, and filtering", r"""
Keep only the words that start with `k` **or** `K`.
""", r"""
clean[clean.str.contains(r"^k", «case=False»)]
""", ["Koila!", "kiew,", "kezeranugh!", "kicarb"], r"""
`case=False` makes the match ignore capitalisation. The `True`/`False` Series goes inside `clean[...]` to keep
only the rows where it is `True` — ordinary pandas filtering.
""")

ex("Words that end in punctuation", r"""
Keep the words whose **last** character is a punctuation mark.
""", r"""
clean[clean.str.contains(r"«[^\w\s]$»")]
""", ["Koila!", "kiew,", "kezeranugh!"], r"""
`[^\w\s]` = one punctuation character, `$` = at the end. Both pieces are from Part 2; here they run on every
element.
""")

ex("`.str.replace` with literal text", r"""
Replace every `z` with a `t`.
""", r"""
clean.str.«replace»("z", "t")
""", ["Koila!", "kiew,", "humble", "keteranugh!", "both", "kicarb"], r"""
For plain text, `.str.replace(old, new)` behaves like the string method, on every element. No `regex=True`
is needed because `z` is just a letter.
""")

ex("`.str.replace` with a pattern", r"""
Change a leading lowercase `k` to `v`.
""", r"""
clean.str.replace(r"^k", "v", «regex=True»)
""", ["Koila!", "view,", "humble", "vezeranugh!", "bozh", "vicarb"], r"""
In `.str.replace` regex is **off** by default, so a pattern needs `regex=True`. `Koila!` still starts with a
capital `K` and is untouched — a second replacement with `^K` → `V` would handle it.
""")

ex("The silent bug", r"""
Predict: without `regex=True`, does this replacement change anything? `.equals` answers `True` if the result
is identical to `clean`.
""", r"""
clean.str.replace("^k", "v").«equals»(clean)
""", True, r"""
`True` — nothing changed. Without `regex=True`, pandas searches for the two literal characters `^k`, finds
none, and "succeeds" at replacing zero occurrences. No error message, wrong answer.
""")

ex("Delete the groans from a column", r"""
Remove each `ugh` (any number of `h`s) **and** the punctuation mark after it.
""", r"""
clean.str.replace(r"«ugh+[^\w\s]»", "", regex=True)
""", ["Koila!", "kiew,", "humble", "kezeran", "bozh", "kicarb"], r"""
The same pattern as in Part 3, now on a column. Only `kezeranugh!` contains a groan followed by punctuation;
it becomes `kezeran`. The `!` of `Koila!` is not preceded by `ugh`, so it stays.
""")

ex("`.apply` + `lambda`", r"""
`.str` has no "run `re.findall` with my own function" button. Use `.apply` with a `lambda` to list the
punctuation marks in each word.
""", r"""
clean.«apply»(lambda w: re.«findall»(r"[^\w\s]", w))
""", [["!"], [","], [], ["!"], [], []], r"""
`.apply` hands each element to the function; `lambda w: ...` is a one-line function whose argument `w` is one
word. Every element of the result is a list — empty when the word has no punctuation.
""")

ex("`.apply` to count", r"""
Count the vowels in each word.
""", r"""
clean.apply(lambda w: «len»(re.findall(r"«[aeiou]»", w)))
""", [3, 2, 2, 4, 1, 2], r"""
Inside the `lambda`, `re.findall` returns the list of vowels in one word and `len` counts them. Anything you
can do to one string, `.apply` + `lambda` does to the whole Series.
""")

ex("Preview for PA 4.2 — `.str.count`", r"""
`.str.count(pattern)` counts the matches of a pattern in each element. Count the whitespace characters in the
**original, unstripped** `words`, then add them up.
""", r"""
words.str.«count»(r"«\s»").sum()
""", 7, r"""
`.str.count(r"\s")` does what the previous exercise did with `.apply`, in one step: matches of `\s` per element
(3, 0, 4, 0, 0, 0). `.sum()` totals them. This answers "how many characters are white space?"
""")

ex("Preview for PA 4.2 — keeping part of the match with `\\1`", r"""
Change a final `b` to `y` **even when punctuation follows it**. Parentheses capture the punctuation, and `\1`
in the replacement puts it back.
""", r"""
endings = pd.Series(["kicarb", "kanizb,", "bb.", "bubble"])
endings.str.replace(r"b(«[^\w\s]*»)$", r"y«\1»", regex=True)
""", ["kicary", "kanizy,", "by.", "bubble"], r"""
`b` = the letter to change; `([^\w\s]*)` = zero or more punctuation marks, **captured** by the parentheses;
`$` = end of the word. In the replacement, `\1` means "whatever the first pair of parentheses matched", so the
comma or full stop is preserved. `*` (not `+`) lets the pattern also match when there is no punctuation at all.
The replacement is a raw string too, so that `\1` reaches pandas intact.
""")

ex("Series to one string", r"""
Join the words of `clean` into a single string separated by spaces.
""", r"""
«" ".join»(clean)
""", "Koila! kiew, humble kezeranugh! bozh kicarb", r"""
`join` accepts any collection of strings, including a Series. The separator comes first, as always:
`" ".join(clean)`.
""")

# ============================================================================ #
#  Part 5 — mixed exercises
# ============================================================================ #

part("Part 5 — Mixed exercises: you choose the tools", r"""
The blanks are larger now — whole expressions. Decide which methods and patterns the job needs.
""")

ex("Tidy product codes", r"""
Make every product code tidy: no padding, all capitals, and at most six characters.
""", r"""
codes = pd.Series(["  ab-101 ", "AB-102", " ab-103x "])
codes.«str.strip()».«str.upper()».«str.slice(0, 6)»
""", ["AB-101", "AB-102", "AB-103"], r"""
Three `.str` methods chained. Order matters: strip **before** slicing, otherwise the leading spaces would use
up some of the six characters.
""")

ex("Normalise a messy string", r"""
One plain string: remove the padding, collapse the inner runs of spaces, and lowercase it.
""", r"""
messy = "   Too   MANY    Spaces  "
re.sub(«r"\s+"», «" "», messy.«strip()»).«lower()»
""", "too many spaces", r"""
`strip()` handles the ends, `re.sub(r"\s+", " ", ...)` turns each inner run of whitespace into one space, and
`.lower()` finishes. `re.sub` returns a string, so a string method can be chained onto it.
""")

ex("Extract hashtags", r"""
Pull out every hashtag: a `#` followed by one or more word characters.
""", r"""
post = "Loving #python and #regex, not #Mondays!"
re.findall(«r"#\w+"», post)
""", ["#python", "#regex", "#Mondays"], r"""
`#` is not special in a regex, so it needs no escaping; `\w+` takes the letters after it and stops at the comma,
space, or `!`.
""")

ex("Count the -ing words", r"""
How many words in the sentence end in `ing`, allowing for punctuation after the word?
""", r"""
sentence = "Singing, dancing and acting are tiring things."
pd.Series(sentence.split(" ")).str.contains(«r"ing[^\w\s]*$"»).«sum()»
""", 4, r"""
Split into a Series of words, then test each: `ing`, then zero or more punctuation marks (`[^\w\s]*`), then the
end (`$`). Without the punctuation part, `Singing,` would be missed. `things.` ends in `ings.`, not `ing`, so it does not
count. Summing `True`/`False` counts the `True`s.
""")

setup(r"""
secret = pd.Series(["  Xhe ", "ughh!qreasure", "  is", "buriedugh?", "  undez  ", "xhe", "flooz."])
""")

ex("Warm-up on a scrambled message", r"""
For the Series `secret` (set-up cell above): how many characters are there in total, and how many of them are
whitespace? Return both numbers.
""", r"""
(secret.«str.len().sum()», secret.«str.count(r"\s").sum()»)
""", (51, 9), r"""
`.str.len().sum()` totals the lengths; `.str.count(r"\s").sum()` totals the whitespace characters (3 + 0 + 2 +
0 + 4 + 0 + 0).
""")

ex("The longest word, in capitals", r"""
After stripping, which word of `secret` is longest? Show it in capitals.
""", r"""
stripped = secret.str.strip()
stripped[stripped.str.len() == «stripped.str.len().max()»].«str.upper()»
""", ["UGHH!QREASURE"], r"""
Compare each length with the maximum length to get a `True`/`False` filter, keep the matching row, and
capitalise it. Strip first — otherwise padding would count towards the length.
""")

ex("Decode the message", r"""
Decode `secret`: (1) strip each word; (2) delete every `ugh` (any number of `h`s) followed by a punctuation
mark; (3) change every `q` to `t`; (4) a word starting with `x` becomes `t`, and `X` becomes `T`; (5) a word
ending in `z` — possibly before punctuation — gets `r` instead. Then join with spaces.
""", r"""
decoded = (secret
           .str.«strip()»
           .str.replace(«r"ugh+[^\w\s]"», "", regex=True)
           .str.replace(«"q"», «"t"»)
           .str.replace(«r"^x"», "t", regex=True)
           .str.replace(«r"^X"», "T", regex=True)
           .str.replace(«r"z([^\w\s]*)$"», «r"r\1"», regex=True))
" ".join(decoded)
""", "The treasure is buried under the floor.", r"""
The PA 4.2 pipeline in miniature. Step 2 uses `ugh+[^\w\s]`; step 3 is plain text, so no `regex=True`; step 4
needs two anchored replacements because regex is case-sensitive; step 5 captures the optional punctuation with
`([^\w\s]*)` and restores it with `\1`, which is what turns `flooz.` into `floor.` rather than leaving it alone.
""")

ex("Pieces of words", r"""
In the decoded message, find every piece of a word that starts with `t` and ends with `e`.
""", r"""
found = decoded.apply(lambda w: re.findall(«r"t\w*e"», w))
[piece for pieces in found for piece in pieces]
""", ["treasure", "the"], r"""
`t`, then zero or more word characters (`\w*`), then `e`. Because `*` is greedy, `treasure` is matched whole
rather than stopping at the first `e`. `The` is not matched: its `T` is a capital. The last line flattens the
Series of lists into one list.
""")


# ============================================================================ #
#  Verify, then build
# ============================================================================ #

BLANK = re.compile(r"«(.*?)»", re.S)
METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def verify() -> None:
    ns: dict = {}
    bad = []
    n = 0
    for item in ITEMS:
        if item["kind"] == "setup":
            exec(item["code"], ns)
        elif item["kind"] == "ex":
            n += 1
            src = BLANK.sub(lambda m: m.group(1), item["code"])
            *body, last = src.split("\n")
            if last.startswith((" ", ")")):          # multi-line final expression: evaluate the tail
                k = max(i for i, line in enumerate(body) if not line.startswith((" ", ")")))
                body, last = body[:k], "\n".join(body[k:] + [last])
            exec("\n".join(body), ns)
            got = plain(eval(last, ns))
            if got != item["expect"]:
                bad.append((n, item["title"], got))
    if bad:
        raise SystemExit("expected values are wrong:\n" + "\n".join(f"  {b}" for b in bad))
    print(f"verified {n} exercises")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def build(solutions: bool) -> dict:
    if solutions:
        cells = [md("# GSB 5544 — Topic 4.2 Extra Practice: Strings and Regular Expressions — SOLUTIONS\n"
                    "*Same numbering as the practice notebook. Each solution shows the completed code, its output, "
                    "and why it works — for regex, what each part of the pattern does.*")]
    else:
        cells = [md("# GSB 5544 — Topic 4.2 Extra Practice: Strings and Regular Expressions\n"
                    "*Fill each `____` blank, run the cell, and compare with the **Intended result**. One blank is one "
                    "missing method, argument, or pattern. The exercises run in order — set-up cells (no blanks) must be "
                    "run as you reach them. Solutions, with the same numbering, are in the companion `-solution` notebook.*")]
    n = 0
    for item in ITEMS:
        if item["kind"] == "part":
            cells.append(md(f"---\n## {item['title']}\n\n{item['text']}"))
        elif item["kind"] == "setup":
            cells.append(code(item["code"]))
        else:
            n += 1
            if solutions:
                cells.append(md(f"### Solution {n} — {item['title']}"))
                cells.append(code(BLANK.sub(lambda m: m.group(1), item["code"])))
                cells.append(md(f"**Why it works.** {item['why']}"))
            else:
                cells.append(md(f"### Exercise {n} — {item['title']}\n{item['task']}\n\n"
                                f"**Intended result:** `{item['expect']!r}`"))
                cells.append(code(BLANK.sub("____", item["code"])))
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    return {"cells": cells, "metadata": copy.deepcopy(METADATA), "nbformat": 4, "nbformat_minor": 5}


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


def main() -> None:
    verify()
    practice, solution = WEEK / f"{STEM}-empty.ipynb", WEEK / f"{STEM}-solution.ipynb"
    save(practice, build(solutions=False))
    save(solution, build(solutions=True))
    print(f"wrote {practice.relative_to(ROOT)} and {solution.relative_to(ROOT)}")
    if "--no-exec" not in sys.argv:
        execute(solution)


if __name__ == "__main__":
    main()
