#!/usr/bin/env python3
"""Build the Week 3 practice-activity notebooks (PA 3.1 and PA 3.2): student + executed solution.

Usage:  python3 tools/build_week3_pas.py            # rewrite both -empty files, build + execute both -solution files
        python3 tools/build_week3_pas.py --no-exec  # same, without executing

Each `-empty.ipynb` is the source of truth for the *questions*.  This script
  * normalizes the title (course prefix, "— SOLUTION" suffix on the key),
  * inserts a data-loading cell for the Ames section of PA 3.2 (the source notebook had none),
  * and, for the solution, replaces every placeholder cell — `# YOUR CODE HERE` or
    `**YOUR RESPONSE HERE**` — with the answer cells listed below, in placeholder order.
Placeholders are matched by *order*, so adding or moving questions only requires
keeping the ANSWERS lists in step.

The solution notebooks read their data from public URLs (MovieLens 1M via dlsun.github.io,
Ames housing, College Scorecard), so executing needs a network connection.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_3"
JUPYTER = "/opt/anaconda3/bin/jupyter"

PA31 = WEEK / "joining_and_merging" / "GSB5544_PA_3_1_Concatenating_Joining_Pivoting"
PA32 = WEEK / "distances_between_observations" / "GSB5544_PA_3_2_Distances_Between_Observations"

PLACEHOLDER_CODE = re.compile(r"^\s*#\s*YOUR CODE HERE")
PLACEHOLDER_MD = re.compile(r"^\s*\*\*YOUR RESPONSE HERE\.?\*\*")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}


def answer(text: str) -> dict:
    return md("**Answer:** " + text.strip())


# ============================================================================ #
#  PA 3.1 — Concatenating, Joining, and Pivoting (MovieLens 1M)
# ============================================================================ #

PA31_TITLE = "GSB 5544 — PA 3.1: Concatenating, Joining, and Pivoting Data"

PA31_ANSWERS: list[list[dict]] = [
    # 1. Read in the data files
    [
        md("""
The README says fields are separated by `::`. A multi-character separator needs `engine="python"`, and because the files have no header row we pass `header=None` and supply `names=` ourselves (column names taken from the README). `movies.dat` contains a few non-UTF-8 bytes in titles, hence `encoding_errors="ignore"`.
"""),
        code("""
base = "https://dlsun.github.io/pods/data/ml-1m/"

movies  = pd.read_csv(base + "movies.dat",  sep="::", engine="python", header=None,
                      names=["MovieID", "Title", "Genres"], encoding_errors="ignore")
ratings = pd.read_csv(base + "ratings.dat", sep="::", engine="python", header=None,
                      names=["UserID", "MovieID", "Rating", "Timestamp"])
users   = pd.read_csv(base + "users.dat",   sep="::", engine="python", header=None,
                      names=["UserID", "Gender", "Age", "Occupation", "Zip"])

print(movies.shape, ratings.shape, users.shape)
"""),
        code("""
movies.head()
"""),
        code("""
ratings.head()
"""),
        code("""
users.head()
"""),
    ],
    # 2. Which age group gives the highest ratings?
    [
        md("""
The rating and the rater's age live in different tables, so join `ratings` to `users` on `UserID`. Every rating has exactly one user (`validate="many_to_one"`). `Age` is a *code* (1, 18, 25, …) for an age range — map it to the README's labels before summarizing.
"""),
        code("""
age_labels = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}

ratings_users = ratings.merge(users, on="UserID", how="left", validate="many_to_one")
ratings_users["AgeGroup"] = ratings_users["Age"].map(age_labels)

by_age = (ratings_users.groupby("AgeGroup")["Rating"]
                       .agg(n_ratings="size", mean_rating="mean")
                       .sort_values("mean_rating", ascending=False)
                       .round(3))
by_age
"""),
        answer("""
Average ratings rise steadily with age. The **56+** group is the most generous (mean about 3.77) and **18–24** the harshest (about 3.51), with every group in between falling in order. The differences are small in absolute terms, but with tens of thousands of ratings per group they are very stable.
"""),
    ],
    # 3. Highest / lowest average rating among movies with >= 100 ratings
    [
        md("""
Summarize `ratings` to one row per movie (count and mean), then join the titles on — a lookup join, one movie per `MovieID`. Filter to 100+ ratings *after* counting.
"""),
        code("""
movie_stats = (ratings.groupby("MovieID")["Rating"]
                      .agg(n_ratings="size", avg_rating="mean")
                      .reset_index()
                      .merge(movies, on="MovieID", how="left", validate="one_to_one"))

popular = movie_stats[movie_stats["n_ratings"] >= 100]
popular.sort_values("avg_rating", ascending=False)[["Title", "n_ratings", "avg_rating"]].head(10)
"""),
        code("""
popular.sort_values("avg_rating")[["Title", "n_ratings", "avg_rating"]].head(10)
"""),
        answer("""
Highest: *Seven Samurai* (4.56), *The Shawshank Redemption* (4.55), *The Godfather* (4.52), *A Close Shave* (4.52), *The Usual Suspects* (4.52). Lowest: *Kazaam* (1.47), *Battlefield Earth* (1.61), *Pokémon the Movie 2000* (1.62), *Aces: Iron Eagle III* (1.64), *Police Academy 6* (1.66). The 100-rating floor matters: without it the extremes are movies rated once or twice.
"""),
    ],
    # 4. Average rating vs. share of 18-24 raters, scatterplot
    [
        md("""
Reuse `ratings_users` (ratings joined to users). A 0/1 indicator for "rater is 18–24" has a mean equal to the *share* of such raters, so one `groupby().agg()` gives the average rating, the 18–24 share, and the count per movie. Join the titles on afterwards.
"""),
        code("""
ratings_users["is_18_24"] = (ratings_users["Age"] == 18).astype(int)

by_movie = (ratings_users.groupby("MovieID")
                         .agg(avg_rating=("Rating", "mean"),
                              share_18_24=("is_18_24", "mean"),
                              n_ratings=("Rating", "size"))
                         .reset_index()
                         .merge(movies, on="MovieID", how="left", validate="one_to_one"))
by_movie.head()
"""),
        code("""
from plotnine import *

(ggplot(by_movie, aes(x="share_18_24", y="avg_rating", size="n_ratings"))
 + geom_point(alpha=0.25)
 + labs(x="Share of ratings from users aged 18-24", y="Average rating",
        size="Number of ratings", title="MovieLens: each point is one movie")
)
"""),
        code("""
by_movie[["avg_rating", "share_18_24"]].corr().round(3)
"""),
        answer("""
There is a moderate **negative** relationship (correlation about −0.35): movies whose audience skews 18–24 tend to have lower average ratings, consistent with part 2 (younger users rate more harshly, and they also gravitate to different films). Heavily-rated movies (big points) cluster at a 10–30% share with ratings of 3–4.5; the extreme shares (0% or 100%) are movies with a handful of ratings.
"""),
    ],
    # 5. Number of ratings by movie; how many movies had zero ratings?
    [
        md("""
An inner join between `movies` and a per-movie count keeps only movies that *have* ratings — the zero-rating movies are exactly the rows an inner join throws away. A **left** join from `movies` keeps them, with `NaN` in the count column, which we then fill with 0.
"""),
        code("""
counts = ratings.groupby("MovieID").size().rename("n_ratings").reset_index()

movie_counts = movies.merge(counts, on="MovieID", how="left", validate="one_to_one")
movie_counts["n_ratings"] = movie_counts["n_ratings"].fillna(0).astype(int)

print("movies in movies.dat:", len(movies))
print("movies with ratings :", len(movies.merge(counts, on="MovieID", how="inner")))
print("movies with zero    :", (movie_counts["n_ratings"] == 0).sum())
"""),
        code("""
movie_counts[movie_counts["n_ratings"] == 0].head(10)
"""),
        answer("""
3,883 movies are listed but only 3,706 were rated, so **177 movies have zero ratings**. An inner join returns 3,706 rows and never shows the missing 177; the left join keeps every movie and reveals them as `NaN` counts (filled with 0). The `_merge` indicator or a left anti join (`movies[~movies["MovieID"].isin(ratings["MovieID"])]`) gives the same 177.
"""),
    ],
    # 6. Movies with both a 1 and a 5
    [
        md("""
Build two small tables — the number of 1-star ratings per movie and the number of 5-star ratings per movie — and **inner join** them: a movie appears in the result only if it is in both, i.e. it received both a 1 and a 5. The joined columns answer "how many of each type".
"""),
        code("""
ones  = ratings[ratings["Rating"] == 1].groupby("MovieID").size().rename("n_1").reset_index()
fives = ratings[ratings["Rating"] == 5].groupby("MovieID").size().rename("n_5").reset_index()

both = (ones.merge(fives, on="MovieID", how="inner", validate="one_to_one")
            .merge(movies, on="MovieID", how="left", validate="one_to_one"))

print("movies with both a 1 and a 5:", len(both))
both.sort_values("n_1", ascending=False)[["Title", "n_1", "n_5"]].head(10)
"""),
        code("""
# The most polarizing: many 1s AND many 5s
both["min_count"] = both[["n_1", "n_5"]].min(axis=1)
both.sort_values("min_count", ascending=False)[["Title", "n_1", "n_5"]].head(10)
"""),
        code("""
# Same answer by pivoting instead of joining: one row per movie, one column per star rating
star_counts = pd.crosstab(ratings["MovieID"], ratings["Rating"])
((star_counts[1] > 0) & (star_counts[5] > 0)).sum()
"""),
        answer("""
**2,986 movies** received at least one 1-star *and* at least one 5-star rating (out of 3,706 rated movies). The `n_1` and `n_5` columns give the number of each; for example *Wild Wild West* has 314 ones and 16 fives, while *The Blair Witch Project* is genuinely polarizing with 219 ones and 180 fives. A crosstab of `MovieID` by `Rating` (a pivot) reaches the same 2,986 without a join.
"""),
    ],
]

# ============================================================================ #
#  PA 3.2 — Distances Between Observations (Ames housing, College Scorecard)
# ============================================================================ #

PA32_TITLE = "GSB 5544 — PA 3.2: Distances Between Observations"

PA32_AMES_DATA = [
    md("""
The Ames data set (2,930 home sales in Ames, Iowa; tab-separated) is at the URL below. House 0 is the first row. The variables we need for part 1: `Gr Liv Area` (above-ground living area, sq ft), `Bedroom AbvGr`, `Full Bath`, `Half Bath`, and `SalePrice`; part 2 adds `House Style`.
"""),
    code("""
df_ames = pd.read_csv("https://dlsun.github.io/pods/data/AmesHousing.txt", sep="\\t")

df_ames.shape
"""),
]

PA32_ANSWERS: list[list[dict]] = [
    # Ames 1 — code
    [
        md("""
**Plan.** (1) Build the three variables (combine full and half baths into one `Bathrooms` count). (2) **Standardize** each variable so a difference of one standard deviation counts the same for every variable — otherwise living area, measured in the thousands, swamps bedrooms and bathrooms, measured in ones. (3) Compute the distance from every house to house 0. (4) Keep only houses cheaper than house 0 and sort by distance.
"""),
        code("""
df_ames["Bathrooms"] = df_ames["Full Bath"] + 0.5 * df_ames["Half Bath"]

house0 = df_ames.loc[0]
show = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "House Style", "Neighborhood", "Year Built", "SalePrice"]
house0[show]
"""),
        code("""
features = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms"]

X = df_ames[features].astype(float)
X_z = (X - X.mean()) / X.std()          # standardized: every column has mean 0, SD 1

diff = X_z - X_z.loc[0]                 # row-wise difference from house 0
df_ames["dist_euclid"]    = np.sqrt((diff ** 2).sum(axis=1))
df_ames["dist_manhattan"] = diff.abs().sum(axis=1)

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
cheaper.sort_values("dist_euclid")[show + ["dist_euclid"]].head(10)
"""),
        code("""
cheaper.sort_values("dist_manhattan")[show + ["dist_manhattan"]].head(10)
"""),
        md("""
**Sensitivity check.** Wrap the calculation in a function and compare the five nearest cheaper houses under different metrics and scalings.
"""),
        code("""
def nearest_cheaper(features, i=0, metric="euclidean", scaling="z", k=5):
    \"\"\"Indices of the k houses nearest to house i on `features`, among houses cheaper than house i.\"\"\"
    X = df_ames[features].astype(float)
    if scaling == "z":
        X = (X - X.mean()) / X.std()
    elif scaling == "minmax":
        X = (X - X.min()) / (X.max() - X.min())
    diff = X - X.loc[i]
    dist = np.sqrt((diff ** 2).sum(axis=1)) if metric == "euclidean" else diff.abs().sum(axis=1)
    dist = dist[df_ames["SalePrice"] < df_ames.loc[i, "SalePrice"]]
    return dist.sort_values().head(k).index.tolist()

for scaling in ["z", "minmax", "none"]:
    for metric in ["euclidean", "manhattan"]:
        print(f"{scaling:>7} {metric:>10}: {nearest_cheaper(features, scaling=scaling, metric=metric)}")
"""),
        code("""
# What the *unscaled* distance picks: matches on square feet only, ignoring bedrooms and baths
df_ames.loc[nearest_cheaper(features, scaling="none"), show]
"""),
    ],
    # Ames 1 — response
    [
        answer("""
House 0 is a 1,656 sq ft, 3-bedroom, 1-bath one-story ranch in North Ames, built 1960, sold for $215,000. With standardized variables the nearest cheaper houses (rows 1226, 1940, 1357, 758, 291, …) are all 1,640–1,670 sq ft, 3 bedrooms, 1 bath, several of them also in North Ames and built in the 1950s–60s, selling for $100,000–$165,000 — sensible "same house, lower price" matches.

The results are **insensitive to the metric** (Euclidean and Manhattan return the same five houses in slightly different order) and to **z-score vs. min-max scaling**. They are **very sensitive to not scaling at all**: on raw units living area (SD ≈ 500 sq ft) dominates bedrooms and baths (SD < 1), so the unscaled distance just matches square footage and returns houses with different bedroom and bath counts.

**Sale price should not be in the distance.** The goal is a house that is *like* house 0 but *cheaper*: price is the constraint we filter on, not a dimension of similarity. Including it would pull the matches toward houses priced like house 0 — the opposite of what we want.
"""),
    ],
    # Ames 2 — code
    [
        md("""
`House Style` is categorical, so it cannot be subtracted. **One-hot encode** it (`pd.get_dummies`): one 0/1 column per style. Two houses with the same style differ by 0 on all those columns; two with different styles differ by 1 in two columns, which adds √2 ≈ 1.41 to the Euclidean distance — large compared with typical z-score differences, so a style mismatch is heavily penalized.
"""),
        code("""
style_dummies = pd.get_dummies(df_ames["House Style"], dtype=float)
style_dummies.head()
"""),
        code("""
X2 = pd.concat([X_z, style_dummies], axis=1)     # 3 standardized numeric columns + 8 style indicators

diff2 = X2 - X2.loc[0]
df_ames["dist_style"] = np.sqrt((diff2 ** 2).sum(axis=1))

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
cheaper.sort_values("dist_style")[show + ["dist_style"]].head(10)
"""),
        code("""
# Which styles did the part-1 top 10 have, and which do the part-2 top 10 have?
top1 = cheaper.sort_values("dist_euclid").head(10)["House Style"].value_counts()
top2 = cheaper.sort_values("dist_style").head(10)["House Style"].value_counts()
pd.DataFrame({"part 1 (no style)": top1, "part 2 (with style)": top2}).fillna(0).astype(int)
"""),
    ],
    # Ames 2 — response
    [
        answer("""
Every one of the ten nearest houses is now a `1Story` like house 0 (rows 1940, 618, 2700, 314, 788, …), whereas in part 1 several of the closest matches were 1.5- or 2-story houses that merely had similar square footage and room counts. Because a style mismatch costs √2 in distance, the dummies act almost like a filter: the algorithm first finds houses of the same style, then orders them by size and rooms. If that is too strict, scale the dummy columns down (multiply them by 0.5, say) so style is a preference rather than a requirement. Metric and scaling choices again barely change the list; the encoding of the categorical variable is what matters here.
"""),
    ],
    # Ames 3 — code
    [
        md("""
**Variables chosen.** Quantitative: `Gr Liv Area`, `Bedroom AbvGr`, `Bathrooms`, `Year Built`, `Overall Qual`, `Lot Area`, `Garage Cars`. Categorical: `House Style`, `Neighborhood`, `Bldg Type`. These capture size, age, quality, land, and location — what a buyer who "likes house 0" is probably reacting to — without the dozens of near-duplicate basement/porch/garage columns. `Garage Cars` has one missing value; fill it with 0 so every house has a distance.
"""),
        code("""
quant = ["Gr Liv Area", "Bedroom AbvGr", "Bathrooms", "Year Built", "Overall Qual", "Lot Area", "Garage Cars"]
categ = ["House Style", "Neighborhood", "Bldg Type"]

X3 = pd.get_dummies(df_ames[quant + categ], columns=categ, dtype=float)
X3[quant] = (X3[quant] - X3[quant].mean()) / X3[quant].std()
X3 = X3.fillna(0)

diff3 = X3 - X3.loc[0]
df_ames["dist_full"] = np.sqrt((diff3 ** 2).sum(axis=1))

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
show3 = show + ["Overall Qual", "Lot Area", "Garage Cars", "Bldg Type"]
cheaper.sort_values("dist_full")[show3 + ["dist_full"]].head(10)
"""),
        code("""
house0[show3]
"""),
        code("""
# Sensitivity: drop Lot Area (house 0 sits on an unusually large lot) and see how much the list changes
quant_no_lot = [q for q in quant if q != "Lot Area"]
X4 = pd.get_dummies(df_ames[quant_no_lot + categ], columns=categ, dtype=float)
X4[quant_no_lot] = (X4[quant_no_lot] - X4[quant_no_lot].mean()) / X4[quant_no_lot].std()
X4 = X4.fillna(0)
df_ames["dist_no_lot"] = np.sqrt(((X4 - X4.loc[0]) ** 2).sum(axis=1))

cheaper = df_ames[df_ames["SalePrice"] < house0["SalePrice"]]
a = set(cheaper.sort_values("dist_full").head(10).index)
b = set(cheaper.sort_values("dist_no_lot").head(10).index)
print("top-10 overlap with / without Lot Area:", len(a & b), "of 10")
cheaper.sort_values("dist_no_lot")[show3 + ["dist_no_lot"]].head(10)
"""),
    ],
    # Ames 3 — response
    [
        answer("""
House 0 is a 1960 one-story, 3-bed, 1-bath, quality-5 house on a very large lot (31,770 sq ft — the 99th percentile). With size, age, quality, lot, garage, style, neighborhood, and building type all in the distance, the nearest cheaper houses are one-story single-family homes of 1,470–1,770 sq ft, 3 bedrooms, 1–1.5 baths, quality 5–6, on similarly huge lots (20,000–35,000 sq ft), mostly built in the 1950s–70s. They read as the same *kind* of property, which is what the added variables buy us.

Sensitivity is now noticeably higher than in parts 1–2: dropping `Lot Area` alone replaces most of the top ten, because the lot is house 0's most unusual feature and standardization makes "unusual" expensive to match. The general lesson: with many variables, *which* variables you include (and how you scale them) matters more than Euclidean vs. Manhattan. Sale price is still left out for the reason given in part 1.
"""),
    ],
    # College 1 — code
    [
        md("""
Both variables are quantitative but on wildly different scales (admission rate 0–1; undergraduates 5–119,000), so standardize before computing Euclidean distance to Cal Poly. The unscaled version is shown for contrast.
"""),
        code("""
num = ["AdmissionRate", "Undergraduates"]

A = df_college[num]
A_z = (A - A.mean()) / A.std()
dist1 = np.sqrt(((A_z - A_z.loc[school_name]) ** 2).sum(axis=1))

df_college.assign(dist=dist1.round(3)).sort_values("dist")[num + ["CarnegieClassification", "Ownership", "dist"]].head(11)
"""),
        code("""
# Without scaling, Undergraduates (SD ≈ 7,800) dominates and AdmissionRate is effectively ignored
dist1_raw = np.sqrt(((A - A.loc[school_name]) ** 2).sum(axis=1))
df_college.assign(dist=dist1_raw.round(1)).sort_values("dist")[num + ["dist"]].head(6)
"""),
    ],
    # College 1 — response
    [
        answer("""
Using **standardized Euclidean distance** on admission rate and undergraduate enrollment, the schools nearest Cal Poly (33% admit, 21,090 undergraduates) are UC Santa Barbara, DeVry University–Illinois, UNC Chapel Hill, Clemson, the University of Virginia, CUNY Hunter, Stony Brook, and Boston University — all selective-ish schools with roughly 15,000–23,000 undergraduates. Without scaling, the list becomes Iowa, East Carolina, VCU, Buffalo, and Kentucky: schools with ~21,000 undergraduates regardless of admission rate (many admit 70–80%), because a 0.4 difference in admit rate is invisible next to a difference of hundreds of students. Standardizing is the decision that makes both variables count.
"""),
    ],
    # College 2 — code
    [
        md("""
Add one-hot columns for the two categorical variables. Note: `Institution` is not unique in this file (15 duplicated names), so instead of `pd.concat` on the index — which fails on duplicate labels — call `pd.get_dummies` on a single frame and pass `columns=` to encode only the categorical ones. Then standardize the two numeric columns as before.
"""),
        code("""
B = pd.get_dummies(df_college[num + ["CarnegieClassification", "Ownership"]],
                   columns=["CarnegieClassification", "Ownership"], dtype=float)
B[num] = (B[num] - B[num].mean()) / B[num].std()

dist2 = np.sqrt(((B - B.loc[school_name]) ** 2).sum(axis=1))
df_college.assign(dist=dist2.round(3)).sort_values("dist")[num + ["CarnegieClassification", "Ownership", "dist"]].head(11)
"""),
    ],
    # College 2 — response
    [
        answer("""
Cal Poly is a *public* "Master's Colleges & Universities: Larger Programs" school, and the dummies make those two facts count for a lot (a mismatch on either adds √2 to the distance). The nearest schools are now CUNY Hunter, Baruch, John Jay, and Brooklyn College, then Cal Poly Pomona and CUNY Queens — public master's-level institutions with 12,000–27,000 undergraduates and moderate admission rates. UCSB, UNC, and UVA drop down the list because they are research doctoral universities, and DeVry drops because it is private for-profit. Whether that is "more similar" depends on the question: for *how selective and how big*, part 1's list is better; for *what kind of institution*, this one is.
"""),
    ],
    # College 3 — code
    [
        md("""
The 38 `PCIP` columns are all proportions on the same 0–1 scale, so this time we do **not** standardize: doing so would inflate tiny, rare fields (a 1% difference in library science would count as much as a 20% difference in engineering). Plain Euclidean distance on the raw proportions — and, as a check, cosine similarity, which compares the *mix* regardless of magnitude — give the same neighbours.
"""),
        code("""
P = df_college.filter(like="PCIP")
cp_fields = P.loc[school_name]
cp_fields.sort_values(ascending=False).head(6)      # 14 = engineering, 52 = business, 01 = agriculture, 45 = social sciences
"""),
        code("""
dist3 = np.sqrt(((P - cp_fields) ** 2).sum(axis=1))
df_college.assign(dist=dist3.round(3)).sort_values("dist")[["State", "Undergraduates", "CarnegieClassification", "dist"]].head(11)
"""),
        code("""
# Cosine similarity as a cross-check (1 = identical mix of fields)
norms = np.sqrt((P ** 2).sum(axis=1))
cosine = (P @ cp_fields) / (norms * norms.loc[school_name])
cosine.sort_values(ascending=False).head(8).round(3)
"""),
    ],
    # College 3 — response
    [
        answer("""
Judged only by *what students study*, Cal Poly's neighbours are the big public land-grant universities: NC State, Iowa State, Illinois Urbana-Champaign, Mississippi State, Texas A&M, Clemson, Purdue, Virginia Tech, Auburn, and West Virginia. That is exactly Cal Poly's field mix — engineering (23%), business (16%), and agriculture (11%) — which is rare outside the land-grant system. None of these schools appeared in parts 1–2, because they are larger, doctoral, and mostly less selective; which list is "right" depends entirely on which notion of similarity you meant, and that choice — the variables, not the metric — is the real decision.
"""),
    ],
]


# ============================================================================ #
#  Build
# ============================================================================ #

def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, nb: dict) -> None:
    for index, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{index:03d}"
    nb["nbformat"] = 4
    nb["nbformat_minor"] = max(5, nb.get("nbformat_minor", 0))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def is_placeholder(cell: dict) -> bool:
    text = source(cell)
    return bool(PLACEHOLDER_CODE.match(text)) if cell["cell_type"] == "code" else bool(PLACEHOLDER_MD.match(text))


def clear_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def insert_after_heading(nb: dict, heading: str, new_cells: list[dict]) -> None:
    """Insert cells after the markdown cell whose source starts with `heading` (idempotent)."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and source(cell).startswith(heading):
            following = source(nb["cells"][i + 1]) if i + 1 < len(nb["cells"]) else ""
            if following.startswith(source(new_cells[0])):
                return
            nb["cells"][i + 1:i + 1] = copy.deepcopy(new_cells)
            return
    raise SystemExit(f"heading not found: {heading!r}")


def build(stem: Path, title: str, answers: list[list[dict]], extra: list[tuple[str, list[dict]]] = ()) -> None:
    student_path = stem.with_name(stem.name + "-empty.ipynb")
    solution_path = stem.with_name(stem.name + "-solution.ipynb")

    student = load(student_path)
    student["cells"][0]["source"] = f"# {title}"
    for heading, new_cells in extra:
        insert_after_heading(student, heading, new_cells)
    for cell in student["cells"]:
        cell["source"] = source(cell)
    clear_outputs(student)
    save(student_path, student)

    solution = copy.deepcopy(student)
    solution["cells"][0]["source"] = f"# {title} — SOLUTION"
    placeholders = [i for i, c in enumerate(solution["cells"]) if is_placeholder(c)]
    if len(placeholders) != len(answers):
        raise SystemExit(f"{stem.name}: {len(placeholders)} placeholders but {len(answers)} answer groups")
    for index, group in sorted(zip(placeholders, answers), reverse=True):
        solution["cells"][index:index + 1] = copy.deepcopy(group)
    solution["cells"] = [c for c in solution["cells"] if not (c["cell_type"] == "code" and not source(c).strip())]
    save(solution_path, solution)
    print(f"wrote {student_path.relative_to(ROOT)}\n      {solution_path.relative_to(ROOT)}")


def execute(path: Path) -> None:
    subprocess.run(
        [JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=900", str(path)],
        check=True,
    )
    nb = load(path)
    errors = [(i, o.get("ename")) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"{path.name}: execution errors {errors}")
    print(f"executed {path.relative_to(ROOT)} (no errors)")


def main() -> None:
    build(PA31, PA31_TITLE, PA31_ANSWERS)
    build(PA32, PA32_TITLE, PA32_ANSWERS, extra=[("## Ames - Recommending Similar Homes", PA32_AMES_DATA)])
    if "--no-exec" not in sys.argv:
        execute(PA31.with_name(PA31.name + "-solution.ipynb"))
        execute(PA32.with_name(PA32.name + "-solution.ipynb"))


if __name__ == "__main__":
    main()
