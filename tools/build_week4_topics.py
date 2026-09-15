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
*From raw text to a table of numbers, and from the table to "how alike are these two emails?"*
""")

md(t41, """
## The road map

Week 3 asked *which rows of a table are most alike?* and answered it with a distance.
This week the observations are **emails** — there are no columns yet, only text. The whole
lesson is one journey, in three legs:

| Leg | Question | Sections | Where it lands in PA 4.1 |
|---|---|---|---|
| **Represent** | How does text become rows and columns? | 1 – 5 | parts 1 – 3 |
| **Weight** | Which columns deserve to count for more? (TF-IDF) | 6 | parts 4 – 5 |
| **Compare** | Given two rows of numbers, how similar are the emails? | 7 – 9 | parts 6 – 11 |

Every section ends with ✅ questions. Answer them *from the matrix in front of you* — the
point of this notebook is to keep looking at what the code produced, not just to run it.
""")

code(t41, """
import pandas as pd
import numpy as np
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 1. Documents and corpora

Two words you will meet in every reading on this topic:

- A **document** is one unit of text — whatever we are treating as *one observation*: one email, one review, one tweet, one contract.
- A **corpus** is the collection of documents — the whole data set of text.

Our running example is four short emails (two from work, two that look like spam):
""")

code(t41, """
corpus = pd.Series([
    "Send the invoice to Sam and the price to Kim",
    "The invoice and the price are attached",
    "Win your FREE prize now and claim the prize",
    "Free free FREE: click the link to win",
])
corpus
""")

md(t41, """
✅ **Fill in the blanks.**

1. Each element of `corpus` (for example `corpus[2]`) is one ____.
2. The whole Series `corpus` is the ____.
3. In PA 4.1, one document is one ____, and the corpus is the ____ column of the data frame.
4. In the table we are about to build, each document will become one ____ (row / column).
""")

answer(t41, """
(1) **document**; (2) **corpus**; (3) one **email body**, and the corpus is the **`body`**
column — 1,990 documents; (4) each document becomes one **row**. Deciding what counts as a
document is the first modelling choice: in PA 4.1 it is the body alone, but subject + body
together would be an equally valid document.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 2. From text to tokens

Before we can count anything, the text has to be made **consistent** and then **cut into pieces**.

### 2a. Normalization — make the text consistent

Normalization means applying rules so that things that *should* count as the same actually
*look* the same. The most common rule is **lowercasing**: `Free`, `free`, and `FREE` are one
word, and we want them counted as one.
""")

code(t41, """
corpus.str.«lower»()
""")

md(t41, """
✅ In documents 2 and 3, which words were *different strings* before normalizing and are
*identical* after? How many distinct spellings collapsed into one?
""")

answer(t41, """
`FREE` (document 2), `Free` and `FREE` (document 3) all become `free` — three spellings
collapse into one token. Without lowercasing the table would have separate `Free`, `FREE`,
and `free` columns, splitting one signal three ways.
""")

md(t41, """
### 2b. Tokenization — cut the text into units

Tokenization splits each document into **tokens** — the units we will count. Here a token is
a *word*: a run of letters or digits, with punctuation and spaces acting as the cuts.
""")

code(t41, """
tokens = corpus.str.lower().str.«findall»(r"\\w+")     # \\w+ = a run of word characters
tokens
""")

code(t41, """
tokens.apply(len)                                       # how many tokens in each document
""")

md(t41, """
✅ **Document 0** has ____ tokens in total, but only ____ *distinct* tokens, because the
words ____ and ____ each appear twice.
""")

answer(t41, """
10 tokens in total, 8 distinct: `the` and `to` each appear twice. Keep this document in
mind — its row of the matrix will have two 2s in it.
""")

md(t41, """
### 2c. Tokens are words, not phrases

A token is a single word. A phrase such as *free prize* is **not** one token — it is two
adjacent tokens. If we want phrases to be counted, we count **combinations** of neighbouring
tokens: pairs are called **bigrams**, triples **trigrams**. (`CountVectorizer(ngram_range=(1, 2))`
would add every bigram as an extra column; the default counts single words only, and that is
what we use this week.)

✅ Write out the bigrams of the phrase *"click the link"*. How many are there?
""")

answer(t41, """
Two bigrams: `click the` and `the link`. A phrase of *n* tokens has *n − 1* bigrams. Notice
that the bigram `the link` keeps some word order — exactly the information single-word
tokens throw away.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 3. Vocabulary and counting: the algorithm

Put the pieces in order and you have an **algorithm** — a fixed sequence of steps that turns
any corpus into a table:

| Step | Action | Result |
|---|---|---|
| 1 | **Normalize** each document (lowercase) | consistent text |
| 2 | **Tokenize** each document into words | a list of tokens per document |
| 3 | **Build the vocabulary** — the set of *distinct* tokens across the whole corpus | the **columns** of the table |
| 4 | **Count** how many times each vocabulary word appears in each document | one **row** per document |

The output is the **term-frequency (TF) matrix**, also called the document-term matrix:
documents down the side, vocabulary words across the top, counts in the cells.

- The **vocabulary** decides the columns — so the number of columns is the number of distinct words in the corpus.
- The **documents** decide the rows — one each.

✅ **Count by hand first.** Document 1 is *"The invoice and the price are attached"*.
Fill in its row of the table:

| `and` | `are` | `attached` | `invoice` | `price` | `the` | every other column |
|---|---|---|---|---|---|---|
| ____ | ____ | ____ | ____ | ____ | ____ | ____ |

And the whole matrix will have ____ rows and ____ columns. (Count the distinct words in
`tokens` above — or trust the code in Section 4 to tell you.)
""")

answer(t41, """
`and` 1, `are` 1, `attached` 1, `invoice` 1, `price` 1, `the` **2**, and **0** in every other
column. The matrix has **4 rows** (four documents) and **18 columns** (18 distinct words
across the four emails). Zeros will outnumber everything else — most words appear in only
one or two documents.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 4. The algorithm in code

`scikit-learn`'s `CountVectorizer` runs all four steps. We take it in pieces so that each
call can be matched to a step of the algorithm — and to the matrix it produces.

**Chunk 1 — create the vectorizer.** Nothing has touched the corpus yet; this object just
holds the settings (lowercase on, one-word tokens).
""")

code(t41, """
from sklearn.feature_extraction.text import «CountVectorizer»

vec = «CountVectorizer»()
vec
""")

md(t41, """
**Chunk 2 — `fit`: learn the vocabulary.** `fit` reads the corpus, normalizes and tokenizes
every document, and collects the distinct words. It performs steps 1 – 3 and remembers the
vocabulary; it does **not** count anything yet.
""")

code(t41, """
vec.«fit»(corpus)
vocab = vec.«get_feature_names_out»()
vocab, len(vocab)
""")

md(t41, """
✅ (a) `len(vocab)` is the number of ____ the matrix will have. (b) Is `vocab` in the order
the words appeared in the emails? (c) Which step of the algorithm has *not* happened yet?
""")

answer(t41, """
(a) **columns** — 18. (b) No: it is alphabetical (`and`, `are`, `attached`, …); the vectorizer
sorts the vocabulary. (c) Step 4, counting. `fit` learned *what to count*; nothing has been
counted.
""")

md(t41, """
**Chunk 3 — `transform`: count.** Now each document is counted against the learned vocabulary
— step 4. The result has one row per document and one column per vocabulary word.
""")

code(t41, """
tf = vec.«transform»(corpus)
tf.shape
""")

md(t41, """
✅ Read `tf.shape` as (____, ____). What decided the first number, and what decided the second?
""")

answer(t41, """
(4, 18): **4 rows** because there are four documents (the corpus decides the rows); **18
columns** because there are 18 words in the vocabulary (the vocabulary decides the columns) —
exactly the answer from the hand count in Section 3.
""")

md(t41, """
**Chunk 4 — make it a data frame.** `tf` is stored as a *sparse* matrix (only the non-zero
cells are kept, because most cells are zero). `.toarray()` expands it, and the vocabulary
supplies the column names.
""")

code(t41, """
tf_df = pd.DataFrame(tf.«toarray»(), columns=«vocab»)
tf_df
""")

md(t41, """
✅ Compare row 1 of `tf_df` with your hand count from Section 3. Do they agree — including
the 2 for `the`? Now check row 0 against your token count from Section 2b: which two columns
hold a 2?
""")

answer(t41, """
They agree: row 1 has 1s for `and`, `are`, `attached`, `invoice`, `price`, a **2** for `the`,
and 0 elsewhere. Row 0 has a 2 in `the` and a 2 in `to` — the two words that appeared twice
in document 0's token list. The code did exactly the algorithm, no more and no less.
""")

md(t41, """
**Chunk 5 — the shorthand.** `fit_transform` runs `fit` and then `transform` on the same
corpus in one call. It is what you will write in PA 4.1:
""")

code(t41, """
tf_df2 = pd.DataFrame(vec.«fit_transform»(corpus).toarray(), columns=vec.get_feature_names_out())
tf_df2.equals(tf_df)          # same matrix as the two-step version
""")

md(t41, """
✅ When would you call `transform` **without** `fit`? (Hint: in PA 4.1 there are emails whose
spam status is unknown, and we will want to compare them with the known ones.)
""")

answer(t41, """
When a *new* document arrives and must be represented with the **existing** columns: fitting
again would rebuild the vocabulary and change the columns, so the new row could no longer be
compared with the old rows. In PA 4.1 all 1,990 emails — known and unknown — are vectorized
together, so every row shares the same columns.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 5. Reading the TF matrix

The code is done; now interpret the object it made. Answer each question by looking at
`tf_df`, then check with code.
""")

code(t41, """
tf_df.loc[«2», «"prize"»]           # the cell for document 2, word "prize"
""")

md(t41, """
✅ (a) In words, what does the value in that cell mean? (b) Why is `tf_df.loc[3, "invoice"]`
zero? (c) What fraction of the whole matrix is zeros — and why is that fraction so high?
""")

code(t41, """
(tf_df == 0).mean().mean()          # fraction of cells that are zero
""")

answer(t41, """
(a) The word `prize` appears **2 times in document 2**. (b) Document 3 never uses the word
`invoice` — a zero means "this document does not contain this word". (c) About 61 % of the
cells are zero, because every document uses only a handful of the 18 vocabulary words.
On the real Enron corpus (1,990 emails, ~35,000 words) the zeros are over 99 % — which is why
`scikit-learn` stores the matrix sparsely.
""")

code(t41, """
tf_df.«sum»().sort_values(ascending=False).head(3)     # column totals: most-used words overall
""")

md(t41, """
✅ (d) `the` is the most-used word in the corpus. Does knowing an email contains `the` help you
tell the work emails (0, 1) from the spam-like emails (2, 3)? Which word in the top three
*does* help? (e) Document 0 and the shuffled sentence *"Kim the price and send to the invoice
to Sam"* — would they get the same row? What has the matrix thrown away?
""")

answer(t41, """
(d) No — `the` appears in all four documents, so its column says nothing about *which*
group an email belongs to. `free` (4 uses, all in documents 2 and 3) does help.
(e) Yes, identical rows: the matrix counts *which* words and *how often*, and throws away
**word order**. This is why the representation is called a *bag of words* — and why (d) matters:
the biggest column is the least useful one. Section 6 fixes that.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 6. TF-IDF, one ingredient at a time

Section 5 ended with a problem: the largest counts belong to words like `the`, `and`, `to` that
say nothing about what an email is *about*, while the words that do — `invoice`, `price`,
`prize` — have small counts. We fix it by building a **weight** for each word out of three
ingredients.

### 6a. Term frequency (TF)

This is the matrix we already have: TF(*document*, *word*) = how many times the word appears
in that document. It answers *"how much does **this** document use the word?"*

✅ Documents 0 and 1 share the word `the` (twice each) **and** the word `invoice` (once each).
Which of the two shared words is the stronger clue that they are about the same thing? Why
can't TF alone tell you?
""")

answer(t41, """
`invoice` — only work emails talk about invoices, while every email uses `the`. TF cannot
tell you because it only looks *inside* one document at a time; the reason `invoice` is a
better clue is about how the word behaves *across the corpus*. That is the next ingredient.
""")

md(t41, """
### 6b. Document frequency (DF)

DF(*word*) = the number of **documents** that contain the word (not how many times). It is a
property of the *column*, not of a cell: one number per word.
""")

code(t41, """
df = (tf_df «> 0»).sum()            # True where the word appears; summing Trues counts documents
df.sort_values(ascending=False)
""")

md(t41, """
✅ Fill in: DF(`the`) = ____, DF(`and`) = ____, DF(`invoice`) = ____, DF(`price`) = ____,
DF(`prize`) = ____. Which words are common *across the corpus* and which are rare?
""")

answer(t41, """
`the` 4 (every document), `and` 3, `invoice` 2, `price` 2, `prize` 1. The filler words `the`,
`and` are the common ones; the topic words `invoice`, `price`, `prize` are rare across the
corpus even when they appear more than once inside a document (`prize` has TF = 2 in document
2 but DF = 1).
""")

md(t41, """
### 6c. Inverse document frequency (IDF)

We want a weight that is **large for rare words and small for common ones** — the opposite of
DF. The textbook definition is

$$\\text{IDF}(\\text{word}) = \\log\\!\\left(\\frac{N}{\\text{DF}(\\text{word})}\\right), \\qquad N = \\text{number of documents.}$$

A word in every document has DF = N, so IDF = log(1) = **0**: its column is switched off.
A word in one document out of four has IDF = log(4) ≈ 1.39, the maximum.
""")

code(t41, """
N = «len(tf_df)»
idf_simple = np.log(«N / df»)
idf_simple.sort_values()
""")

md(t41, """
✅ (a) Which word has IDF exactly 0, and why? (b) Which words tie for the highest IDF, and what
do they have in common? (c) Put `the`, `and`, `invoice`, `prize` in order from smallest to
largest IDF.
""")

answer(t41, """
(a) `the` — it appears in all N = 4 documents, so N/DF = 1 and log(1) = 0. (b) Every word that
appears in exactly **one** document (`are`, `attached`, `claim`, `click`, `prize`, …) ties at
log(4) ≈ 1.39. (c) `the` (0) < `and` (0.29) < `invoice` (0.69) < `prize` (1.39): the rarer
across the corpus, the larger the weight.
""")

md(t41, """
### 6d. Put them together: TF × IDF

TF-IDF(*document*, *word*) = TF(*document*, *word*) × IDF(*word*). Each **cell** of the TF
matrix is multiplied by the weight of its **column**. A value is large only when the word is
used a lot in *this* document (TF) **and** is rare across the corpus (IDF).
""")

code(t41, """
tfidf_manual = tf_df «* idf_simple»           # every column scaled by its own IDF
tfidf_manual.round(2)
""")

md(t41, """
✅ Look at **row 0** before (`tf_df`) and after (`tfidf_manual`). (a) `the` was the largest value
in the row (2). What is it now, and why? (b) `to` also had count 2. What happened to it, and why
is it different from `the`? (c) Which words in row 0 now score highest?
""")

answer(t41, """
(a) `the` is now **0.0**: count 2 × IDF 0 = 0. It appeared in every document, so it can't
distinguish anything and is switched off. (b) `to` keeps a score of 2 × 0.69 = **1.39**: it
appears in only two documents (0 and 3), so it still carries information. (c) `kim`, `sam`,
`send` — count 1 each but the maximum IDF of 1.39 — tie with `to` at the top; `invoice` and
`price` sit at 0.69. The row now emphasises what is *particular* about this email.
""")

md(t41, """
### 6e. `TfidfVectorizer` — the same idea, two small differences

`scikit-learn` builds TF-IDF in one call, exactly like `CountVectorizer`. Its numbers differ
from `tfidf_manual` for two practical reasons:

1. **Smoothed IDF**: it uses $\\log\\frac{1 + N}{1 + \\text{DF}} + 1$ instead of $\\log\\frac{N}{\\text{DF}}$, so no word is switched off completely (a word in every document gets weight 1, the minimum, rather than 0).
2. **Row normalization**: each row is rescaled to have length 1, so long documents and short documents are on the same footing — this will matter again in Section 7.
""")

code(t41, """
from sklearn.feature_extraction.text import «TfidfVectorizer»

tfidf_vec = «TfidfVectorizer»()
tfidf_df = pd.DataFrame(tfidf_vec.«fit_transform»(corpus).toarray(),
                        columns=tfidf_vec.get_feature_names_out())

pd.DataFrame({"idf_simple": idf_simple, "idf_sklearn": tfidf_vec.idf_}, index=vocab).round(2).sort_values("idf_simple")
""")

code(t41, """
tfidf_df.round(2)
""")

md(t41, """
✅ (a) Under `scikit-learn`'s IDF, `the` gets weight ____ (the smallest) instead of 0 — but is
the **ordering** of the words from least to most informative the same as in `idf_simple`?
(b) In **row 2**, TF said `prize` = 2 and `the` = 1. What does `tfidf_df` say for the same two
cells? (c) In one sentence: how do the values change when we move from TF to TF-IDF?
""")

answer(t41, """
(a) Weight **1.0**; yes, the ordering is identical (`the` < `and` < `free`/`invoice`/`price`/`to`/`win`
< the one-document words). (b) `prize` ≈ 0.67 and `the` ≈ 0.17: `prize` was twice `the`
under TF and is about four times `the` under TF-IDF. (c) Moving from TF to TF-IDF shrinks
the values of words that appear in many documents and keeps or boosts the values of words
that are concentrated in a few — the row stops describing *how much English* the email
contains and starts describing *what it is about*.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 7. Comparing two documents: cosine similarity

Sections 1 – 6 built a representation: **one row of numbers per document**. Now the Week 3
question returns — *how alike are two rows?* — and we need a ruler.

**What the symbols mean.** Take document 0 and document 1:

- $x$ = the **row of document 0** in the matrix: 18 numbers, one per vocabulary word ($x_{\\text{the}} = 2$, $x_{\\text{invoice}} = 1$, $x_{\\text{free}} = 0$, …). A row of numbers is a **vector**.
- $y$ = the row of document 1, the same 18 columns in the same order.

**The dot product** $x \\cdot y = \\sum_{\\text{word}} x_{\\text{word}} \\, y_{\\text{word}}$: multiply the two rows
column by column and add up. A word contributes only if **both** documents use it — a zero on
either side kills that term. So the dot product measures **how much vocabulary the two
documents share**, weighted by how heavily each uses it.
""")

code(t41, """
x = tf_df.loc[«0»]
y = tf_df.loc[«1»]
(x * y)[(x * y) > 0]                 # only the words BOTH documents use survive
""")

code(t41, """
dot = «(x * y).sum()»
dot
""")

md(t41, """
✅ (a) Which four words contribute to $x \\cdot y$? (b) The dot product is 7 — how much of the 7
comes from the single word `the`, and is that the word you would *want* driving the answer?
""")

answer(t41, """
(a) `and`, `invoice`, `price`, `the` — the words in both documents. (b) `the` contributes
2 × 2 = **4 of the 7**, more than half, even though it is the least informative word in the
corpus. Keep this in mind for Section 8: it is exactly what TF-IDF changes.
""")

md(t41, """
**The lengths** $\\lVert x \\rVert = \\sqrt{\\sum_{\\text{word}} x_{\\text{word}}^2}$ and $\\lVert y \\rVert$
likewise: the size of each vector (Euclidean length, as in Week 3, measured from the origin).
A long email has bigger counts, so a bigger length **and** bigger dot products with everything
— length is exactly the thing we do *not* want the comparison to depend on.

**Cosine similarity** divides it out:

$$\\text{cosine similarity}(x, y) = \\frac{x \\cdot y}{\\lVert x \\rVert \\, \\lVert y \\rVert}$$

- the numerator rewards **shared vocabulary**,
- the denominator removes the effect of **document length**,
- what is left depends only on the *mix* of words — the direction of the vectors, not their size.

**Cosine distance** = 1 − cosine similarity, so that "small = similar", as in Week 3.
""")

code(t41, """
len_x = np.sqrt(«(x ** 2).sum()»)
len_y = np.sqrt(«(y ** 2).sum()»)
cos_sim = dot / «(len_x * len_y)»
len_x.round(3), len_y.round(3), cos_sim.round(3)
""")

md(t41, """
✅ **Fill in the blanks.**

1. $\\lVert x \\rVert = \\sqrt{1+1+1+1+1+1+2^2+2^2} = \\sqrt{\\,____\\,} \\approx 3.74$ and $\\lVert y \\rVert = \\sqrt{\\,____\\,} = 3$.
2. Cosine similarity of documents 0 and 1 = $7 / (3.74 \\times 3) \\approx$ ____.
3. For count vectors (no negatives), cosine similarity ranges from ____ (no shared words) to ____ (identical mix of words).
4. The cosine similarity of a document with **itself** is ____; the cosine *distance* of a document to itself is ____.
""")

answer(t41, """
(1) $\\sqrt{14}$ and $\\sqrt{9}$. (2) ≈ **0.62**. (3) from **0** to **1**. (4) similarity
**1**, distance **0** — a vector points in exactly its own direction.
""")

md(t41, """
`scikit-learn` computes every pair at once:
""")

code(t41, """
from sklearn.metrics.pairwise import «cosine_similarity»

sim_tf = pd.DataFrame(«cosine_similarity»(tf_df), index=corpus.str[:22], columns=corpus.str[:22])
sim_tf.round(2)
""")

md(t41, """
✅ (a) Confirm the (0, 1) entry matches your hand calculation. (b) Which pair of documents is
**most** similar, and which is **least**? Read the sentences — does it make sense? (c) Why is
the diagonal all 1s?
""")

answer(t41, """
(a) 0.62 ✔. (b) Most similar: documents 0 and 1 (0.62) — the two work emails, sharing
`invoice`, `price`, `and`, `the`. Least: documents 1 and 3 (0.18) — the attached-invoice email
and the *free free free* spam share only `the`. Documents 2 and 3 (0.40) pair up as the two
spam-like emails through `free` and `win`. (c) Every document has similarity 1 with itself.
""")

md(t41, """
**Why not Euclidean distance, as in Week 3?** Write document 3 out twice — same email, twice as
long. Every count doubles. Euclidean distance says it moved; cosine says it did not:
""")

code(t41, """
doubled = tf_df.loc[[3]] * 2
euclid = np.sqrt(((doubled.values - tf_df.loc[[3]].values) ** 2).sum())
cosine_similarity(doubled, tf_df.loc[[3]])[0, 0].round(3), euclid.round(2)
""")

md(t41, """
✅ Which of the two rulers treats the doubled email as "the same email"? Why is that the right
behaviour for text?
""")

answer(t41, """
Cosine (similarity 1.0 — same direction). Euclidean distance is about 3.7 units, purely because
the counts got bigger. Emails vary enormously in length, and length is not what we mean by
"similar", so cosine is the ruler for text.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 8. How Section 6 and Section 7 fit together

They answer different questions, and it is worth saying so plainly:

| | Section 6 — TF-IDF | Section 7 — cosine similarity |
|---|---|---|
| **Question** | *What number goes in each cell?* | *How alike are two rows?* |
| **Acts on** | **one document at a time** (a row), using column-wide information (DF) | **two documents at a time** (a pair of rows) |
| **Output** | a **matrix** — the representation | a **number per pair** — the comparison |
| **Role** | decides what the vectors $x$ and $y$ *are* | decides how $x$ and $y$ are *compared* |

Section 7 builds on Section 6 because the vectors $x$ and $y$ **are rows of whichever matrix
you built** — feed it `tf_df` and you compare raw counts; feed it `tfidf_df` and you compare
weighted, length-normalized rows. Same ruler, different vectors, different similarities:
""")

code(t41, """
sim_tfidf = pd.DataFrame(cosine_similarity(«tfidf_df»), index=corpus.str[:22], columns=corpus.str[:22])
pd.concat({"cosine on TF": sim_tf.round(2), "cosine on TF-IDF": sim_tfidf.round(2)}, axis=1)
""")

md(t41, """
✅ (a) The similarity of documents 0 and 1 falls from 0.62 (TF) to 0.43 (TF-IDF). Using your
answer from Section 7 about the word `the`, explain why. (b) Does the **most similar pair**
change? (c) Complete the sentence: *"TF-IDF changes the ____ that go into the comparison;
cosine similarity is the ____ we compare them with."*
""")

answer(t41, """
(a) Under TF, more than half of the dot product between 0 and 1 came from `the` (4 of 7).
TF-IDF gives `the` the smallest weight, so that shared-but-meaningless word almost drops out
of the numerator; what remains is the genuinely shared vocabulary (`invoice`, `price`), and the
similarity falls to an honest 0.43. (b) No — 0 and 1 are still the closest pair, now for the
right reason. (c) TF-IDF changes the **vectors** (the numbers in the rows) that go into the
comparison; cosine similarity is the **ruler** we compare them with.
""")

# ---------------------------------------------------------------------------- #
md(t41, """
---
## 9. The payoff: classify an email by its nearest neighbours

Stack the three legs exactly the way PA 4.1 will:

1. Corpus → **TF or TF-IDF matrix** (Sections 3 – 6): text becomes rows of numbers.
2. A new email arrives with an **unknown** label — it is vectorized with the *same columns* (Section 4, chunk 5).
3. Compute the **cosine distance** from the new email to every email whose label is known (Section 7).
4. **Sort.** If the closest emails are mostly spam, bet spam.

That is nearest-neighbour classification, assembled from this notebook. A first look at the
real corpus (a sample of Enron email — the PA introduction tells the story):
""")

code(t41, """
emails = pd.read_csv("%s")
emails["spam"].value_counts(dropna=False)
""" % ENRON_URL)

md(t41, """
1,980 emails labelled spam (1) or not (0) — and **10 with no label**, which PA 4.1 asks you to
guess. One practical warning: a few `body` entries are missing, and `CountVectorizer` refuses
`NaN` documents, so `.fillna("")` first.
""")

code(t41, """
real_corpus = emails["body"].«fillna("")»
tf_real = CountVectorizer().fit_transform(real_corpus)
tf_real.shape
""")

md(t41, """
✅ Using Sections 3 – 4: what decided the first number in `tf_real.shape`, and what decided the
second? Roughly what fraction of that matrix do you expect to be zeros?
""")

answer(t41, """
1,990 rows — one per email (the documents); ~35,000 columns — one per distinct word in the
whole corpus (the vocabulary). Almost all of it is zeros: a typical email uses a couple of
hundred distinct words out of 35,000, so well over 99 % of the cells are 0 — which is why
you compute with this matrix rather than look at it.
""")

md(t41, """
## The three lines to keep

| | |
|---|---|
| **Represent** | normalize → tokenize → vocabulary (columns) → count (rows) = the TF matrix; `CountVectorizer().fit_transform(corpus)` |
| **Weight** | TF-IDF = TF × IDF, IDF = log(N / DF): shrinks words that appear in most documents, keeps the distinctive ones; `TfidfVectorizer()` |
| **Compare** | cosine similarity = $x \\cdot y \\,/\\, (\\lVert x \\rVert \\lVert y \\rVert)$ — shared vocabulary over document length; distance = 1 − similarity; sort and read the neighbours |

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
