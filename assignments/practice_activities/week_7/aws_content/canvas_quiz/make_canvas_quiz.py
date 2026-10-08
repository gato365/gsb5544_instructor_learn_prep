#!/usr/bin/env python3
"""
Session 1 Canvas quiz builder — GSB 5544.

Posts a short vocabulary-and-concepts quiz for the AWS session directly to Canvas
through the Canvas API (canvasapi), and writes a human-readable answer key next to
this script. Same structure as the STAT/DATA 1810 lab quiz builders.

15 easy items, all selected response: 11 multiple choice (one correct) and
4 multiple answers (several correct). Content: S3 vocabulary (bucket, object, key,
prefix), storage vs computation, disk vs RAM, total vs available RAM, cores and vCPUs,
units, where code runs and what a RUNS ON comment does, anonymous data vs your own
account, GHCN conventions, cost, and the shutdown checklist. Everything comes from
the pre-class reading and the topics notebook.

SETUP (one time):
    pip install canvasapi python-dotenv

    Put your Canvas token in a file named .env in THIS folder, one line:
        CANVAS_API_KEY=your_token_here
    This folder is private (not in the course repository). A template is in .env.example.

    The course id lives in this script: edit COURSE_ID below.
    (Or override without editing: CANVAS_COURSE_ID=... in .env.)

USAGE:
    python3 make_canvas_quiz.py --dry-run   # print payloads, post nothing
    python3 make_canvas_quiz.py             # create quiz as an unpublished draft
    python3 make_canvas_quiz.py --publish   # create and publish immediately

Quiz settings: 15 questions, 1 point each, 2 attempts (highest kept), 30-minute limit.
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Course id: EDIT THIS
# ---------------------------------------------------------------------------
COURSE_ID = 194426         # <- your GSB 5544 Canvas course id (the number in the course URL)

# ---------------------------------------------------------------------------
# Credentials (token is read from .env, never from this file)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / ".env").exists() or (p / "_quarto.yml").exists() or (p / ".git").exists():
            return p
    return HERE


ROOT_DIR = _find_project_root()

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass  # fine if the vars are already exported in the shell

API_URL   = os.getenv("CANVAS_API_URL", "https://canvas.calpoly.edu").rstrip("/")
API_KEY   = os.getenv("CANVAS_API_KEY")
COURSE_ID = int(os.getenv("CANVAS_COURSE_ID", COURSE_ID))

# ---------------------------------------------------------------------------
# Quiz metadata
# ---------------------------------------------------------------------------
QUIZ_TITLE = "Session 1 Quiz — Data Bigger Than Your Laptop"
QUIZ_DESCRIPTION = (
    "<p>Fifteen short questions on the pre-class reading and the topics notebook: S3 vocabulary, "
    "storage versus computation, memory, units, where code runs, and cost. "
    "Questions marked <em>select all that apply</em> have more than one correct answer.</p>"
    "<p>2 attempts, 30 minutes each. Your highest score is kept.</p>"
)
ALLOWED_ATTEMPTS = 2
TIME_LIMIT_MIN   = 30
SHUFFLE_ANSWERS  = True

# ---------------------------------------------------------------------------
# Questions
#   type "mc" -> multiple choice, ONE correct  (choices, answer = index)
#   type "ma" -> multiple answers, SEVERAL correct (choices, answer = [indices])
# Every item has 5 choices. 1 point each -> 15 points.
# ---------------------------------------------------------------------------
QUESTIONS = [
    dict(type="mc", points=0, source="Reading §3 · bucket",
         text="In S3, what is a <strong>bucket</strong>?",
         choices=[
             "A named container for objects; its name is unique across all of AWS",
             "A rented computer that runs your notebook",
             "The beginning of an object's name, used to list things 'under' it",
             "One stored file plus its metadata",
             "A geographic cluster of data centers",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §3 · key",
         text="The weather file for 2024 is stored as <code>csv/by_year/2024.csv</code> in the bucket <code>noaa-ghcn-pds</code>. In S3 terms, what is <code>csv/by_year/2024.csv</code>?",
         choices=[
             "The object's key: its full name inside the bucket, one string",
             "Three folders and a file",
             "The bucket's name",
             "A region",
             "An instance type",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §3 · prefix",
         text="When S3 shows what looks like a folder called <code>csv/</code>, what is it really showing you?",
         choices=[
             "A prefix: the shared beginning of many keys",
             "A directory on a disk, exactly like a laptop folder",
             "A separate bucket",
             "A compressed archive",
             "A SageMaker space",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §1, §3 · storage vs computation",
         text="Which statement about S3 is correct?",
         choices=[
             "S3 stores data; it cannot run your code. Some computer has to read the data to do anything with it",
             "S3 stores data and also runs pandas on it when asked",
             "S3 is the name of the rented machine your notebook runs on",
             "Data in S3 is automatically loaded into your laptop's RAM",
             "S3 can hold at most one file per bucket",
         ], answer=0),

    dict(type="mc", points=0, source="Reading, 'what a RUNS ON comment does' · where code runs",
         text="You open a notebook in JupyterLab inside your SageMaker space and run a cell. Which computer runs that cell?",
         choices=[
             "The SageMaker instance; your laptop is only showing a browser tab",
             "Your laptop, because that is where the browser is",
             "S3, because that is where the data is",
             "Whichever computer the cell's <code># RUNS ON:</code> comment names",
             "Both computers at once",
         ], answer=0),

    dict(type="mc", points=0, source="Reading, 'what a RUNS ON comment does'",
         text="A cell begins with <code># RUNS ON: SageMaker (remote)</code>, but you run it in a notebook you opened in Positron on your laptop. What happens?",
         choices=[
             "It runs on your laptop. The comment is a label for people; it does not move the code",
             "Positron sends the cell to SageMaker because of the comment",
             "The cell refuses to run",
             "The cell runs on S3",
             "It runs on your laptop only if the space is stopped",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §1 · total vs available RAM",
         text="A laptop reports 16 GB of <strong>total</strong> RAM and 4 GB <strong>available</strong>. Which number decides whether a new dataset can be loaded, and why are they different?",
         choices=[
             "4 GB; the browser, chat apps, and other running programs already hold the rest",
             "16 GB; total is always the number that matters",
             "4 GB; the other 12 GB is disk space",
             "16 GB; available RAM only matters on cloud machines",
             "Neither; it depends on free disk space",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §1 · disk vs RAM",
         text="A file is 3 GB on disk and your laptop has 500 GB of free disk space and 4 GB of available RAM. Why might <code>pd.read_csv</code> on the whole file still fail?",
         choices=[
             "pandas works in RAM, and a loaded DataFrame is often larger than the file; 4 GB of RAM may not be enough",
             "500 GB is not enough disk space for a 3 GB file",
             "CSV files cannot be read by pandas",
             "The file must first be copied to S3",
             "It cannot fail: the file fits on the disk",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §1 · cores and vCPUs",
         text="A cloud machine is sold as <strong>2 vCPUs, 4 GB</strong>. What is a vCPU?",
         choices=[
             "One logical CPU; two vCPUs may be a single physical core presenting itself as two",
             "Always one physical core",
             "A unit of RAM",
             "A unit of disk space",
             "The number of notebooks you can open at once",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §2 · units",
         text="A drive sold as \"1 TB\" shows up as about 931 GB in your file browser. Why?",
         choices=[
             "The seller counts in thousands (decimal units) and the operating system counts in 1,024s (binary units)",
             "About 7% of the drive is broken",
             "The operating system hides system files",
             "The drive is actually 931 GB and the label is wrong",
             "Files were already saved on it",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §4 · anonymous data, your own account",
         text="NOAA's weather bucket can be read anonymously, without signing in. So why does this module require you to have your own AWS account?",
         choices=[
             "The account lets you rent the remote computer (the SageMaker space); the data itself is free to read",
             "The account is needed to unlock the weather data",
             "Anonymous reads are only allowed from Positron",
             "AWS requires an account to download any file",
             "You do not need an account; the reading is wrong",
         ], answer=0),

    dict(type="mc", points=0, source="Reading §4 · GHCN conventions",
         text="In the GHCN-Daily data, a <code>TMAX</code> value is stored as <code>266</code> and its quality flag is empty. What is the temperature, and what would you do if the quality flag were <strong>not</strong> empty?",
         choices=[
             "26.6 °C (values are in tenths); a flagged value failed NOAA's checks and is dropped, leaving that day missing",
             "266 °C; a flagged value is kept as is",
             "26.6 °C; a flagged value is replaced with 0",
             "2.66 °C; flags can be ignored",
             "266 °F; a flagged value is averaged with its neighbors",
         ], answer=0),

    dict(type="ma", points=0, source="Reading §6 · cost (select all that apply)",
         text="<em>Select all that apply.</em> In the required Session 1 workflow, which of these cost you money (from your credits)?",
         choices=[
             "A SageMaker space whose status is <em>Running</em>, for every hour it is running",
             "The small disk attached to your space, per GB per month, even when the space is stopped",
             "Reading NOAA's public bucket from inside the same AWS region",
             "A notebook that is open in a browser tab while the space is <em>Stopped</em>",
             "Looking at Console Home",
         ], answer=[0, 1]),

    dict(type="ma", points=0, source="Reading, workflow table (select all that apply)",
         text="<em>Select all that apply.</em> In the eight-step workflow, which steps run code on the <strong>SageMaker</strong> machine?",
         choices=[
             "Reading NOAA data from S3",
             "Filtering, transforming, and aggregating the data",
             "Saving a small summary CSV",
             "Opening the summary in Positron and making the chart",
             "Signing in to your AWS account",
         ], answer=[0, 1, 2]),

    dict(type="ma", points=0, source="Reading §6 · shutdown checklist (select all that apply)",
         text="<em>Select all that apply.</em> You are finished for the day. Which of these actually stops the hourly charge for your SageMaker space?",
         choices=[
             "Clicking <strong>Stop space</strong> on the space's page and waiting for the status to read <em>Stopped</em>",
             "Idle Shutdown turning the space off after 60 minutes with no activity",
             "Closing the JupyterLab browser tab",
             "Saving the notebook",
             "Signing out of the AWS console",
         ], answer=[0, 1]),
]

assert len(QUESTIONS) == 15, len(QUESTIONS)
assert all(len(q["choices"]) == 5 for q in QUESTIONS)

# ---------------------------------------------------------------------------
# Canvas payload builders
# ---------------------------------------------------------------------------
def build_canvas_question(q: dict, idx: int) -> dict:
    payload = {
        "question_name":   f"Q{idx}",
        "question_text":   q["text"],
        "points_possible": q["points"],
    }
    if q["type"] == "mc":
        payload["question_type"] = "multiple_choice_question"
        payload["answers"] = [
            {"answer_html": c, "answer_weight": 100 if i == q["answer"] else 0}
            for i, c in enumerate(q["choices"])
        ]
    elif q["type"] == "ma":
        correct = set(q["answer"])
        payload["question_type"] = "multiple_answers_question"
        payload["answers"] = [
            {"answer_html": c, "answer_weight": 100 if i in correct else 0}
            for i, c in enumerate(q["choices"])
        ]
    else:
        raise ValueError(f"Unknown question type {q['type']!r}")
    return payload


def answer_key() -> str:
    total = sum(q["points"] for q in QUESTIONS)
    lines = ["# Session 1 Quiz — Answer Key\n",
             f"Total points: {total}  ·  {len(QUESTIONS)} questions  ·  "
             f"{ALLOWED_ATTEMPTS} attempts  ·  {TIME_LIMIT_MIN} min\n"]
    for i, q in enumerate(QUESTIONS, 1):
        lines.append(f"**Q{i}** ({q['type']}, {q['points']} pt, {q['source']}) {q['text']}")
        if q["type"] == "mc":
            lines.append(f"   Correct: {q['choices'][q['answer']]}")
        else:
            lines.append("   Correct: " + " | ".join(q["choices"][i] for i in q["answer"]))
        lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Post
# ---------------------------------------------------------------------------
def post_to_canvas(publish: bool, dry_run: bool):
    quiz_payload = {
        "title":            QUIZ_TITLE,
        "description":      QUIZ_DESCRIPTION,
        "quiz_type":        "assignment",
        "published":        publish,
        "allowed_attempts": ALLOWED_ATTEMPTS,
        "scoring_policy":   "keep_highest",
        "time_limit":       TIME_LIMIT_MIN,
        "shuffle_answers":  SHUFFLE_ANSWERS,
        "show_correct_answers": False,
    }
    total = sum(q["points"] for q in QUESTIONS)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating quiz: {QUIZ_TITLE}")
    print(f"  Course          : {COURSE_ID} @ {API_URL}")
    print(f"  Questions       : {len(QUESTIONS)}  ({total} points)")
    print(f"  Allowed attempts: {ALLOWED_ATTEMPTS}")
    print(f"  Time limit      : {TIME_LIMIT_MIN} min")
    print(f"  Published       : {publish}\n")

    quiz = None
    if not dry_run:
        if COURSE_ID == 0:
            sys.exit("COURSE_ID is still 000000. Edit it at the top of this script (or set CANVAS_COURSE_ID in .env).")
        try:
            from canvasapi import Canvas
        except ImportError:
            sys.exit("canvasapi not installed. Run: pip install canvasapi python-dotenv")
        if not API_KEY:
            sys.exit(f"Missing CANVAS_API_KEY. Put it in {ROOT_DIR / '.env'} (see .env.example).")
        course = Canvas(API_URL, API_KEY).get_course(COURSE_ID)
        quiz = course.create_quiz(quiz_payload)
        print(f"  Quiz created -> ID {quiz.id}")

    for i, q in enumerate(QUESTIONS, 1):
        cq = build_canvas_question(q, i)
        if dry_run:
            print(f"  Q{i:02d} [{cq['question_type']}] {q['source']}  pts={cq['points_possible']}")
            for a in cq["answers"]:
                mark = "*" if a["answer_weight"] == 100 else " "
                print(f"       [{mark}] {a['answer_html'][:90]}")
        else:
            quiz.create_question(question=cq)
            print(f"  Posted Q{i:02d} ({q['source']})")

    if dry_run:
        print("\n[DRY RUN] No changes made to Canvas.")
    else:
        status = "published" if publish else "draft (unpublished)"
        print(f"\nDone — '{QUIZ_TITLE}' posted as {status}.")
        print(f"   {API_URL}/courses/{COURSE_ID}/quizzes/{quiz.id}")


def main():
    ap = argparse.ArgumentParser(description="Post the Session 1 quiz to Canvas.")
    ap.add_argument("--publish", action="store_true", help="publish immediately (default: draft)")
    ap.add_argument("--dry-run", action="store_true", help="print payloads; post nothing")
    args = ap.parse_args()

    key_path = HERE / "session1_quiz_answer_key.md"
    key_path.write_text(answer_key())
    print(f"Wrote {key_path.name}")

    post_to_canvas(publish=args.publish, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
