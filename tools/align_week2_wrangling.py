#!/usr/bin/env python3
"""Align Week 2 notebook organization and Data Wrangling scaffolding.

Run from the repository root. The script is intentionally deterministic so the
student and solution copies can be refreshed together without notebook drift.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEEK = ROOT / "assignments" / "practice_activities" / "week_2"
WRANGLING = WEEK / "data_wrangling"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, notebook: dict) -> None:
    if notebook.get("nbformat") == 4:
        notebook["nbformat_minor"] = max(5, notebook.get("nbformat_minor", 0))
    for index, cell in enumerate(notebook.get("cells", [])):
        cell.setdefault("id", f"cell-{index:03d}")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value,
    }


def clear_outputs(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def update_topic() -> None:
    solution_path = WRANGLING / "GSB5544_Topic_2_2_Data_Wrangling-solution.ipynb"
    student_path = WRANGLING / "GSB5544_Topic_2_2_Data_Wrangling-empty.ipynb"
    solution = load(solution_path)

    set_source(
        solution["cells"][0],
        "# GSB 5544 — Topic 2.2: Data Wrangling — SOLUTION  \n"
        "*Tidy data, the Big Five verbs, grouped calculations, and reshaping*",
    )

    # Insert a direct bridge to every operation students meet in PA 2.2.
    insert_at = next(
        i for i, cell in enumerate(solution["cells"])
        if "## 10. Practice Activities" in source(cell)
    )
    bridge = [
        markdown(
            "## 10. Bridge to PA 2.2: grouped transformations and wide data\n\n"
            "The practice activity combines the verbs above. Use this pattern: "
            "**identify the observational unit → filter/select → mutate → group → summarize → arrange**."
        ),
        markdown(
            "### A. Several grouping variables\n\n"
            "Pass a list to `groupby` when each answer needs more than one label. "
            "The mean of a 0/1 indicator is a proportion."
        ),
        code(
            "# One survival proportion for every gender/type combination\n"
            "titanic_example = pd.DataFrame({\n"
            "    \"gender\": [\"female\", \"female\", \"male\", \"male\"],\n"
            "    \"type\": [\"passenger\", \"crew\", \"passenger\", \"crew\"],\n"
            "    \"survived\": [1, 1, 0, 1]\n"
            "})\n"
            "titanic_example.groupby([\"gender\", \"type\"])[\"survived\"].mean()"
        ),
        markdown(
            "### B. Weighted means and `transform`\n\n"
            "A weighted mean is `(value * weight).sum() / weight.sum()`. "
            "`transform` returns one value per original row, so it is useful when each row needs its group's total."
        ),
        code(
            "college_example = pd.DataFrame({\n"
            "    \"state\": [\"CA\", \"CA\", \"OR\"],\n"
            "    \"earnings\": [70000, 50000, 60000],\n"
            "    \"enrollment\": [20000, 5000, 10000]\n"
            "})\n"
            "college_example[\"state_enrollment\"] = (\n"
            "    college_example.groupby(\"state\")[\"enrollment\"].transform(\"sum\")\n"
            ")\n"
            "college_example[\"state_weight\"] = (\n"
            "    college_example[\"enrollment\"] / college_example[\"state_enrollment\"]\n"
            ")\n"
            "college_example[\"weighted_earnings\"] = (\n"
            "    college_example[\"earnings\"] * college_example[\"state_weight\"]\n"
            ")\n"
            "college_example.groupby(\"state\")[\"weighted_earnings\"].sum()"
        ),
        markdown(
            "### C. Missing values, replacing, and vectorized standardization\n\n"
            "Use `.dropna(subset=[...])` to filter missing rows and `.replace(0, pd.NA)` when a sentinel "
            "value means missing. DataFrame arithmetic works column-by-column, without loops."
        ),
        code(
            "scores_example = pd.DataFrame({\"floor\": [12.0, 13.0, 0], \"rings\": [11.0, 14.0, 15.0]})\n"
            "scores_example = scores_example.replace(0, pd.NA)\n"
            "standardized = (scores_example - scores_example.mean()) / scores_example.std()\n"
            "standardized[\"total_z\"] = standardized.sum(axis=1)\n"
            "standardized.sort_values(\"total_z\", ascending=False)"
        ),
        markdown(
            "### D. Reshape: index, wide, and long\n\n"
            "`.set_index([...])` identifies rows without deleting the score columns. `.melt()` stacks several "
            "score columns into an event column and a score column. `.pivot()` spreads values back into columns."
        ),
        code(
            "gym_example = pd.DataFrame({\n"
            "    \"year\": [2020, 2020], \"gymnast\": [\"A\", \"B\"],\n"
            "    \"floor\": [14.1, 13.8], \"rings\": [13.9, 14.2]\n"
            "})\n"
            "gym_indexed = gym_example.set_index([\"year\", \"gymnast\"])[[\"floor\", \"rings\"]]\n"
            "gym_long = gym_indexed.reset_index().melt(\n"
            "    id_vars=[\"year\", \"gymnast\"], var_name=\"event\", value_name=\"score\"\n"
            ")\n"
            "gym_long"
        ),
        markdown(
            "### PA 2.2 readiness check\n\n"
            "Before starting, make sure you can explain when to use `value_counts(normalize=True)`, "
            "the mean of an indicator, `groupby([...])`, `describe()`, `dropna`, `transform`, "
            "DataFrame arithmetic, `sum(axis=1)`, `set_index`, and `melt`."
        ),
    ]
    solution["cells"][insert_at:insert_at] = bridge
    for cell in solution["cells"]:
        if "## 10. Practice Activities" in source(cell):
            set_source(cell, source(cell).replace("## 10.", "## 11.", 1))

    save(solution_path, solution)

    # Make the student copy genuinely participatory from the opening minutes.
    student = copy.deepcopy(solution)
    clear_outputs(student)
    set_source(
        student["cells"][0],
        "# GSB 5544 — Topic 2.2: Data Wrangling  \n"
        "*Fill each `____` blank during class; later checks ask you to write complete expressions.*",
    )
    replacements = {
        'import pandas as pd': 'import ____ as pd',
        'pd.read_csv("https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/sc_crime_sample.csv")': 'pd.________("https://raw.githubusercontent.com/gato365/gsb5544_instructor_learn_prep/main/assignments/Data/sc_crime_sample.csv")',
        'crime.head()': 'crime.______()',
        'crime.shape': 'crime.____',
        'wide.melt(id_vars = "crime_cat"': 'wide.____(id_vars = "____"',
        'value_vars = [str(y) for y in range(1991, 2021)]': 'value_vars = [str(y) for y in range(____, ____)]',
        'var_name = "year"': 'var_name = "____"',
        'value_name = "n_offenses"': 'value_name = "____"',
        'crime["crime_cat"].unique()': 'crime["____"].unique()',
        'crime["crime_cat"].nunique()': 'crime["____"].nunique()',
        'crime["crime_cat"].value_counts()': 'crime["____"].value_counts()',
        'crime["crime_cat"].value_counts(normalize = True)': 'crime["____"].value_counts(normalize = ____)',
        'crime["year"].describe()': 'crime["____"].describe()',
        'crime[["incident_date", "crime_cat", "crime", "location_type", "arrest_made"]]': 'crime[["____", "____", "____", "____", "____"]]',
        'groupby(["gender", "type"])["survived"].mean()': 'groupby(["____", "____"])["____"].mean()',
        'groupby("state")["enrollment"].transform("sum")': 'groupby("____")["____"].transform("____")',
        'college_example["enrollment"] / college_example["state_enrollment"]': 'college_example["____"] / college_example["____"]',
        'scores_example.replace(0, pd.NA)': 'scores_example.replace(____, pd.____)',
        '(scores_example - scores_example.mean()) / scores_example.std()': '(scores_example - scores_example.____()) / scores_example.____()',
        'standardized.sum(axis=1)': 'standardized.sum(axis=____)',
        'set_index(["year", "gymnast"])': 'set_index(["____", "____"])',
        'id_vars=["year", "gymnast"], var_name="event", value_name="score"': 'id_vars=["____", "____"], var_name="____", value_name="____"',
    }
    for cell in student["cells"]:
        text = source(cell)
        for old, new in replacements.items():
            text = text.replace(old, new)
        set_source(cell, text)
    save(student_path, student)


def update_pa() -> None:
    source_path = WRANGLING / "GSB5544_PA_2_2_data_wrangling.ipynb"
    student_path = WRANGLING / "GSB5544_PA_2_2_data_wrangling-empty.ipynb"
    solution_path = WRANGLING / "GSB5544_PA_2_2_data_wrangling-solution.ipynb"
    original = load(source_path if source_path.exists() else student_path)
    student = copy.deepcopy(original)
    set_source(student["cells"][0], "# GSB 5544 — PA 2.2: Wrangling and Summarizing Data")
    intro = source(student["cells"][1]).split("\n\nUse the workflow and readiness check", 1)[0]
    set_source(
        student["cells"][1],
        intro + "\n\n"
        "Use the workflow and readiness check from **Topic 2.2**. Each section builds from one-row "
        "summaries to grouped calculations and reshaping. Replace every `# YOUR CODE HERE` or "
        "`**YOUR RESPONSE HERE.**` before submitting."
    )
    clear_outputs(student)
    save(student_path, student)

    sol = copy.deepcopy(student)
    set_source(sol["cells"][0], "# GSB 5544 — PA 2.2: Wrangling and Summarizing Data — SOLUTION")
    set_source(
        sol["cells"][1],
        intro + "\n\nThis completed version follows the Topic 2.2 workflow and includes rendered outputs."
    )
    answers = {
        6: 'import numpy as np\ndf_titanic["type"] = np.where(df_titanic["crew"].notna(), "crew", "passenger")\ndf_titanic[["class", "type"]].value_counts()',
        8: 'df_titanic["type"].value_counts(normalize=True)',
        10: 'df_titanic["survived"].mean()',
        12: 'pd.crosstab(df_titanic["survived"], df_titanic["type"], margins=True)',
        14: 'df_titanic.groupby("type")["survived"].mean()',
        16: 'df_titanic.groupby(["gender", "type"])["survived"].mean()',
        18: '# Passenger/crew survival reverses after accounting for gender (Simpson\'s paradox).\npd.crosstab(df_titanic["gender"], df_titanic["type"], normalize="columns").round(3)',
        24: 'df_college[df_college["Institution"].str.contains("California Polytechnic", case=False, na=False)]',
        26: 'df_college_complete = df_college.dropna(subset=["Median Earnings"]).copy()\ndf_college_complete.shape',
        28: 'df_college_complete["Median Earnings"].describe()\ndf_college_complete["Median Earnings"].hist(bins=30)',
        30: 'df_college_complete["log_earnings"] = np.log(df_college_complete["Median Earnings"])\ndf_college_complete["log_earnings"].hist(bins=30)',
        32: 'df_college_complete["weight"] = df_college_complete["Undergraduates"] / df_college_complete["Undergraduates"].sum()\n(df_college_complete["weight"] * df_college_complete["Median Earnings"]).sum()',
        34: 'state_mean = df_college_complete.groupby("State")["Median Earnings"].mean().sort_values(ascending=False)\nstate_mean.plot.bar(figsize=(12, 4))',
        36: 'df_college_complete["state_total"] = df_college_complete.groupby("State")["Undergraduates"].transform("sum")\ndf_college_complete["state_weight"] = df_college_complete["Undergraduates"] / df_college_complete["state_total"]\ndf_college_complete["weighted_earnings"] = df_college_complete["Median Earnings"] * df_college_complete["state_weight"]\nstate_weighted_mean = df_college_complete.groupby("State")["weighted_earnings"].sum().sort_values(ascending=False)\nstate_weighted_mean.plot.bar(figsize=(12, 4))',
        42: 'event_cols = df_gym.loc[:, "Floor Exercise":"Pommelled Horse"].columns.tolist()\ngym_scores = df_gym.set_index(["Round", "Year", "Gymnast"])[event_cols].copy()\ngym_scores',
        44: 'gym_scores.describe()',
        50: 'gym_scores = gym_scores.replace(0, np.nan)\ngym_scores.isna().sum()',
        52: 'gym_scores.describe()',
        54: 'gym_scores.corr().round(3)',
        56: 'gym_z = (gym_scores - gym_scores.mean()) / gym_scores.std()\ngym_z.head()',
        58: 'gym_z["total_z"] = gym_z.sum(axis=1, min_count=len(event_cols))\ngym_z.sort_values("total_z", ascending=False).head(10)',
        60: 'gym_long = gym_scores.reset_index().melt(id_vars=["Round", "Year", "Gymnast"], var_name="Event", value_name="Score")\nyearly_event_means = gym_long.groupby(["Year", "Event"])["Score"].mean().unstack()\nyearly_event_means',
        62: 'gym_long[gym_long["Event"] == "Floor Exercise"].boxplot(column="Score", by="Year")',
        64: 'year_means = gym_scores.groupby(level="Year").transform("mean")\nyear_sds = gym_scores.groupby(level="Year").transform("std")\ngym_year_z = (gym_scores - year_means) / year_sds\ngym_year_z.head()',
        66: 'gym_year_z["total_z"] = gym_year_z.sum(axis=1, min_count=len(event_cols))\ngym_year_z.sort_values("total_z", ascending=False).head(10)',
    }
    responses = {
        22: '**Solution:** The observational unit is one college. `earnings` is the median annual earnings, ten years after entry, among federally aided former students whose tax records are available.',
        46: '**Solution:** Missing event scores are excluded from each column separately, so each event can have a different non-missing sample size.',
        48: '**Solution:** `NaN` is more appropriate because not competing is missing information, not a genuine score of zero. Treating it as zero would distort means, standard deviations, correlations, and totals.',
    }
    for index, value in {**answers, **responses}.items():
        set_source(sol["cells"][index], value)
    clear_outputs(sol)
    save(solution_path, sol)
    if source_path.exists():
        source_path.unlink()


if __name__ == "__main__":
    update_topic()
    update_pa()
    print("Aligned Topic 2.2 and PA 2.2 student/solution notebooks.")
