#!/usr/bin/env python3
"""Build the Week 4 practice-activity SOLUTION notebooks (PA 4.1 and PA 4.2).

Usage:  python3 tools/build_week4_pas.py            # build + execute both -solution files
        python3 tools/build_week4_pas.py --no-exec  # build without executing

Unlike Week 3, the student notebooks (`GSB_5544_PA_4_1_Text_Data.ipynb` and
`GSB_5544_PA_4_2_Decode_a_message.ipynb`) arrived without an -empty suffix and are left
byte-for-byte untouched — they are the source of truth for the questions.  This script
reads them and writes a `-solution.ipynb` sibling for each:
  * the title cell gains the course prefix and a "— SOLUTION" suffix,
  * every placeholder cell — `# ENTER YOUR CODE HERE`, `**TYPE YOUR RESPONSE HERE.**`,
    or an *empty* code cell — is replaced by the answer cells listed below, in order.
Placeholders are matched by *order*, so a question edit only requires keeping the
ANSWERS lists in step.

Both solutions read their data from public URLs (the data301 Enron sample; the scrambled
message on Dropbox), so executing needs a network connection.
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

PA41_SRC = WEEK / "GSB_5544_PA_4_1_Text_Data.ipynb"
PA42_SRC = WEEK / "GSB_5544_PA_4_2_Decode_a_message.ipynb"

PLACEHOLDER_CODE = re.compile(r"^\s*#\s*ENTER YOUR CODE HERE")
PLACEHOLDER_MD = re.compile(r"^\s*\*\*TYPE YOUR RESPONSE HERE\.?\*\*")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


def answer(text: str) -> dict:
    return md("**Answer:** " + text.strip())


# ============================================================================ #
#  PA 4.1 — Text Data: Spam Emails (Enron corpus)
# ============================================================================ #

PA41_TITLE = "# GSB 5544 — PA 4.1: Text Data — Spam Emails — SOLUTION"

PA41_ANSWERS: list[list[dict]] = [
    # 1. Read the body column in to create the corpus
    [
        code("""
import pandas as pd
import numpy as np

emails = pd.read_csv("https://raw.githubusercontent.com/kevindavisross/data301/refs/heads/main/data/enron_email.csv")
emails
"""),
        code("""
# The corpus is the collection of documents: one email body per document.
# A few dozen bodies are missing (NaN) — CountVectorizer refuses NaN documents,
# so replace them with the empty string before vectorizing.
emails["body"].isna().sum()
"""),
        code("""
corpus = emails["body"].fillna("")
corpus
"""),
        answer("""
1,990 emails. The corpus is the `body` column; 61 bodies are missing, and since
`CountVectorizer` raises an error on `NaN` documents we treat a missing body as an
empty document with `.fillna("")` (an empty document simply gets a row of zeros).
"""),
    ],
    # 2. Describe the steps to clean the emails and create the TF matrix
    [
        md("""
**Answer:** to turn the pile of email bodies into tabular data:

1. **Normalize the text** — lowercase every document, so `Free`, `free`, and `FREE` count as the same word.
2. **Tokenize** — split each document into individual words (*tokens*), treating punctuation and whitespace as separators (so `prize!` and `prize` are the same token).
3. **Build the vocabulary** — collect the set of distinct words appearing anywhere in the corpus. Each distinct word becomes a **column**.
4. **Count** — for each email (row) and each vocabulary word (column), record how many times that word appears in that email. These counts are the **term frequencies**.

The result is the **term-frequency (TF) matrix**: one row per email, one column per word,
counts in the cells — text is now a data frame. (Optional refinements at each step: drop
very common "stop words", strip word endings/stemming, only keep words above a minimum
document frequency — `CountVectorizer` has options for all of these, but the defaults are
fine here.)
"""),
    ],
    # 3. CountVectorizer -> TF matrix as a data frame
    [
        code("""
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
tf = vec.fit_transform(corpus)                       # learn vocabulary, then count
tf_df = pd.DataFrame(tf.toarray(), columns=vec.get_feature_names_out())
tf_df
"""),
        answer("""
1,990 rows (one per email) by ~35,000 columns (one per distinct word in the corpus).
`fit_transform` learns the vocabulary and produces a sparse count matrix; `.toarray()`
plus `get_feature_names_out()` turn it into a regular data frame with words as
column names.
"""),
    ],
    # 4. DF, IDF, and why TF-IDF rather than TF
    [
        md("""
**Answer:**

- The **document frequency (DF)** of a word is the number (or fraction) of *documents in the corpus* that contain the word — not how often it appears within any one document. `the` has a DF near 1,990 here; `pharmacy` has a small DF.
- The **inverse document frequency (IDF)** flips this into a weight, roughly `log(number of documents / DF)`: near zero for a word that appears in almost every document, large for a rare word. **TF-IDF** multiplies each term frequency by its word's IDF.
- **Why TF-IDF rather than TF:** raw counts are dominated by exactly the words that carry the least information. Every email uses `the`, `to`, and `and` a lot, so with raw TF two emails look "similar" mostly because they are both written in English. Words that appear in only a few documents — `viagra`, `invoice`, `nomination` — are what actually distinguish one email from another, and TF-IDF boosts them while flattening the everywhere-words. When we later measure similarity between emails, TF-IDF makes the comparison depend on the *distinctive* vocabulary rather than the filler.
"""),
    ],
    # 5. TfidfVectorizer
    [
        code("""
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vec.get_feature_names_out())
tfidf_df
"""),
        answer("""
Same shape as the TF matrix, but each entry is now the term frequency weighted by the
word's IDF (and each row is normalized to unit length by default), so common-everywhere
words are shrunk toward zero and distinctive words stand out.
"""),
    ],
    # 6. Open-ended comparison of spam vs non-spam
    [
        md("""
**Brainstormed questions.** (Yours may differ — that is the point of an open-ended part.)

1. Are spam emails *longer or shorter* than non-spam emails?
2. Which words does each class use most (by average TF)?
3. Which words *separate* the classes best (biggest gap in average TF-IDF)?
"""),
        code("""
# Keep only the emails with a known spam label for this part
known = emails["spam"].notna()
spam = emails.loc[known, "spam"]

tf_known = tf_df[known]
tfidf_known = tfidf_df[known]
spam.value_counts()
"""),
        code("""
# Q1: email length — total word count is the row sum of the TF matrix
tf_known.sum(axis=1).groupby(spam).describe().round(1)
"""),
        code("""
# Q2: most-used words in each class, by average term frequency
mean_tf = tf_known.groupby(spam).mean()
pd.DataFrame({
    "top not-spam (0)": mean_tf.loc[0.0].sort_values(ascending=False).head(10).index,
    "top spam (1)":     mean_tf.loc[1.0].sort_values(ascending=False).head(10).index,
})
"""),
        code("""
# Q3: the words with the biggest gap in average TF-IDF between the classes
mean_tfidf = tfidf_known.groupby(spam).mean()
gap = mean_tfidf.loc[1.0] - mean_tfidf.loc[0.0]     # positive -> spam-leaning

leaning = pd.concat([gap.sort_values().head(10), gap.sort_values().tail(10)]).reset_index()
leaning.columns = ["word", "gap"]
leaning["leans"] = np.where(leaning["gap"] > 0, "spam", "not spam")
leaning
"""),
        code("""
from plotnine import ggplot, aes, geom_col, coord_flip, labs

(ggplot(leaning, aes(x="reorder(word, gap)", y="gap", fill="leans"))
 + geom_col()
 + coord_flip()
 + labs(x="", y="mean TF-IDF, spam minus not-spam",
        title="Which words separate spam from not-spam?"))
"""),
        answer("""
Spam emails run **longer** on average (mean ≈ 176 words vs ≈ 135, with a much heavier tail).
By raw average TF both classes are topped by filler (`the`, `to`, `and`) — exactly the
raw-count problem from part 4. The TF-IDF gap is far more interpretable: the not-spam side
is dominated by Enron business vocabulary (`ect`, `enron`, `hou`, `xls`, `hpl`, `attached`,
`2000` — mail routing codes, spreadsheet attachments, deal talk), while the spam side is
marketing language (`your`, `http`, `here`, `click`, `www`, `free`). Spam talks *at you*
and wants a click; work email talks about *the business*.
"""),
    ],
    # 7. How to classify the 10 unknown emails — brainstorm
    [
        md("""
**Answer:** ideas that use what we have built:

- **Look for spam-y words** — from part 6 we know words like `click`, `http`, `free` lean spam and `enron`, `hou`, `attached` lean not-spam; we could score each unknown email by which side of the vocabulary it uses.
- **Nearest neighbours (the approach we take below):** the TF / TF-IDF matrices put every email — known and unknown — in the *same* columns, so we can compute the **distance** between an unknown email and every labelled email. If the emails most similar to the unknown one are overwhelmingly spam, guess spam; if they are overwhelmingly legitimate, guess not-spam. Because email lengths vary wildly, **cosine distance** (which compares the *mix* of words, ignoring document length) is the right ruler — this is exactly Week 3's "find the most similar rows", with columns we built ourselves.
"""),
    ],
    # 8. Cosine distance from email 1980 (TF) to every known email
    [
        code("""
from sklearn.metrics.pairwise import cosine_similarity

emails.loc[1980, "body"]
"""),
        code("""
# cosine distance = 1 - cosine similarity, from email 1980 to every labelled email
sims = cosine_similarity(tf_df.loc[[1980]], tf_df[known])[0]
dist_tf = pd.Series(1 - sims, index=emails.index[known], name="cosine_dist_tf")
dist_tf
"""),
    ],
    # 9. Sort, inspect the nearest neighbours, and decide
    [
        code("""
nearest_tf = dist_tf.sort_values().head(10)
pd.DataFrame({"cosine_dist_tf": nearest_tf.round(3),
              "spam": emails.loc[nearest_tf.index, "spam"].astype(int),
              "subject": emails.loc[nearest_tf.index, "subject"]})
"""),
        answer("""
Email 1980 is a *"thank you for requesting more information from quicken loans"*
marketing message. Of its 10 nearest labelled emails by TF cosine distance, the large
majority are spam (and the closest ones are other loan/marketing pitches), so I would
call it **spam (1)**. The reasoning: emails that use the same mix of words as known
spam are, absent other evidence, most plausibly spam themselves.
"""),
    ],
    # 10. Repeat with TF-IDF
    [
        code("""
sims = cosine_similarity(tfidf_df.loc[[1980]], tfidf_df[known])[0]
dist_tfidf = pd.Series(1 - sims, index=emails.index[known], name="cosine_dist_tfidf")

nearest_tfidf = dist_tfidf.sort_values().head(10)
pd.DataFrame({"cosine_dist_tfidf": nearest_tfidf.round(3),
              "spam": emails.loc[nearest_tfidf.index, "spam"].astype(int),
              "subject": emails.loc[nearest_tfidf.index, "subject"]})
"""),
        answer("""
The distances themselves grow (TF-IDF zeroes out the shared filler words, so overlaps are
rarer and everything sits further apart) and the exact neighbour list shuffles — the
TF-IDF neighbours match on *distinctive* loan-marketing words rather than on generic
vocabulary. But the **verdict is unchanged**: the nearest emails are still mostly spam,
so email 1980 is still classified spam. When both rulers agree, that is reassuring.
"""),
    ],
    # 11. Guess the spam status of all unknown emails
    [
        code("""
unknown = emails.index[emails["spam"].isna()]

def nn_vote(i, X, k=5):
    \"\"\"Fraction of email i's k nearest labelled neighbours (cosine distance) that are spam.\"\"\"
    sims = cosine_similarity(X.loc[[i]], X[known])[0]
    dist = pd.Series(1 - sims, index=emails.index[known])
    return emails.loc[dist.sort_values().head(k).index, "spam"].mean()

guesses = pd.DataFrame({
    "subject": emails.loc[unknown, "subject"],
    "frac_spam_tf": [nn_vote(i, tf_df) for i in unknown],
    "frac_spam_tfidf": [nn_vote(i, tfidf_df) for i in unknown],
})
guesses["guess"] = np.where(guesses["frac_spam_tfidf"] >= 0.5, "spam", "not spam")
guesses
"""),
        answer("""
For each unknown email we take a **majority vote of its 5 nearest labelled neighbours** by
cosine distance. TF and TF-IDF agree on every email: **1980, 1985, 1986, 1987, 1988 → spam**
(loan marketing, a fake "receipt", an internet pharmacy, "victory at last", "free pay per
view") and **1981–1984, 1989 → not spam** (the eastrans nomination and meter emails are
unmistakably Enron gas-scheduling traffic). The interesting case is **1989** — the subject
(*"87% off for all new software"*) screams spam, but its body is stuffed with random dictionary
words to fool word-count filters, so its 5-neighbour vote is only 0.4 and the body-only
classifier narrowly calls it not-spam. A good reminder that the guess depends on our choices:
including the `subject` text, using a different k, or weighting votes by distance could all
flip a borderline case.
"""),
    ],
]


# ============================================================================ #
#  PA 4.2 — Decode a Message (strings and regular expressions)
# ============================================================================ #

PA42_TITLE = md("# GSB 5544 — PA 4.2: Decode a Message — SOLUTION")

PA42_WARMUPS: list[dict] = [
    code("""
import re

# 1. How many characters are in the scrambled message?
message.str.len().sum()
"""),
    code("""
# 2. How many of these characters are white space?
message.str.count(r"\\s").sum()
"""),
    code("""
# 3. How many words are in the scrambled message?  (one "word" per element)
len(message)
"""),
    code("""
# 4. Show all the punctuation marks in the scrambled message.
punctuation = message.apply(lambda w: re.findall(r"[^\\w\\s]", w))
all_punct = [p for lst in punctuation for p in lst]
print(all_punct)
print("distinct marks:", sorted(set(all_punct)))
"""),
    code("""
# 5. Print out, in all capitals, the longest word in the scrambled message.
trimmed = message.str.strip()
longest = trimmed[trimmed.str.len().idxmax()]
print(longest.upper())
"""),
    code("""
# 6. Every piece of a word that starts with "m" and ends with "z".
pieces = message.apply(lambda w: re.findall(r"m\\w*z", w))
[p for lst in pieces for p in lst]
"""),
    answer("""
2,544 characters in total, of which 1,652 are whitespace (the words come padded with
spaces); 127 words. The punctuation is limited to `! , . ; ?`. The longest word is
`KAUDEVILLIANUGH?AOGHAJDBN` (25 characters), and the m…z pieces are `mosz` and `maaz` —
which, once decoded (z → t, aa → ee), will become *most* and *meet*.
"""),
]

PA42_DECODE: list[dict] = [
    code("""
# 1. Remove any spaces before or after each word.
words = message.str.strip()

# 2. "ugh" with any number of h's, followed by a punctuation mark: delete it.
words = words.str.replace(r"ugh+[^\\w\\s]", "", regex=True)

# 3. No word should be longer than 16 characters: drop the extras off the end.
words = words.str.slice(0, 16)

# 4. Replace all instances of exactly 2 a's with exactly 2 e's.
words = words.str.replace("aa", "ee")            # literal text -> no regex needed

# 5. Replace all z's with t's.
words = words.str.replace("z", "t")

# 6. Every word that ends in b: change it to a y.  (The b may hide before punctuation!)
words = words.str.replace(r"b([^\\w\\s]*)$", r"y\\1", regex=True)

# 7. Every word that starts with k: change it to a v.  (Mind the capital K's!)
words = words.str.replace(r"^k", "v", regex=True).str.replace(r"^K", "V", regex=True)

words
"""),
    code("""
# 8. Recombine the words into a message.
decoded = " ".join(words)
print(decoded)
"""),
    answer("""
> *"Voila! In view, a humble vaudevillian veteran, cast vicariously as both victim and villain
> by the vicissitudes of fate. … The only verdict is vengeance; a vendetta … and you may call me V."*

**9.** The quote is V's alliterative introduction speech from ***V for Vendetta*** (2005).

Notes on the trickier steps: in step 2 the pattern `ugh+[^\\w\\s]` needs the *punctuation
class* after the h's — `ugh` also hides inside legitimate words (`kaudevillian…` →
*vaudevillian*), and only the groans are followed by punctuation. Step 6's `b([^\\w\\s]*)$`
keeps any trailing punctuation while swapping the letter (a bare `b$` would miss `b,`).
Step 7 needs both `^k` and `^K` (regex is case-sensitive — `Koila!` must become `Voila!`).
And step 3's instructions mention both 16 and 13; we keep 16 characters, matching "no word
should be longer than 16."
"""),
]


# ============================================================================ #
#  Build
# ============================================================================ #

def is_placeholder(cell: dict) -> bool:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "code":
        return bool(PLACEHOLDER_CODE.match(src)) or src.strip() == ""
    return bool(PLACEHOLDER_MD.match(src))


def build_solution(src_path: Path, title, answers: list[list[dict]], prepend_title: bool = False) -> dict:
    nb = json.loads(src_path.read_text())
    out_cells: list[dict] = []
    n = 0
    for cell in nb["cells"]:
        if is_placeholder(cell):
            if n >= len(answers):
                raise SystemExit(f"{src_path.name}: more placeholders than answer groups (at #{n})")
            out_cells.extend(copy.deepcopy(answers[n]))
            n += 1
        else:
            out_cells.append(copy.deepcopy(cell))
    if n != len(answers):
        raise SystemExit(f"{src_path.name}: found {n} placeholders, expected {len(answers)}")
    if prepend_title:
        out_cells.insert(0, copy.deepcopy(title))
    else:  # replace the heading line of the first markdown cell
        src = "".join(out_cells[0]["source"]) if isinstance(out_cells[0]["source"], list) else out_cells[0]["source"]
        out_cells[0]["source"] = re.sub(r"^#[^\n]*", title, src, count=1)
    for i, c in enumerate(out_cells):
        c["id"] = f"cell-{i:03d}"
        if c["cell_type"] == "code":
            c.setdefault("outputs", [])
            c.setdefault("execution_count", None)
    nb["cells"] = out_cells
    return nb


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
    jobs = [
        (PA41_SRC, PA41_TITLE, PA41_ANSWERS, False),
        (PA42_SRC, PA42_TITLE, [PA42_WARMUPS, PA42_DECODE], True),
    ]
    for src, title, answers, prepend in jobs:
        dest = src.with_name(src.stem + "-solution.ipynb")
        save(dest, build_solution(src, title, answers, prepend_title=prepend))
        print(f"wrote {dest.relative_to(ROOT)}")
        if "--no-exec" not in sys.argv:
            execute(dest)


if __name__ == "__main__":
    main()
