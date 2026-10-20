"""Build the grading rubric PDF for the GSB 5544 AWS lab (04_lab_remote + 04_lab_local).

100 points, 5 elements x 20 points. Each element is scored at one of four levels:
3 = full score (20), 2 = 14, 1 = 7, 0 = left blank (0).

Run:  python3 make_lab_rubric.py   ->  writes lab_rubric.pdf next to this file.  (needs: pip install reportlab)
"""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent / "lab_rubric.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, leading=19, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=9, leading=11.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CENTER = ParagraphStyle("center", parent=CELL, alignment=TA_CENTER)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.white)

ELEMENTS = [
    dict(
        name="1. Remote analysis",
        where="04_lab_remote.html, L2 to L5",
        l3="Sizes measured before any read (L2). wx has the five required columns, real units (divided by 10), dates parsed, and quality-flagged rows dropped. hot_days includes days_reported; monthly_rain applies the 90% completeness rule and reports years_used. Output shown for every cell, and the setup cell shows a SageMaker computer name.",
        l2="All summaries present and run on SageMaker, with one error or omission: units left in tenths, Q_FLAG not dropped, days_reported or years_used missing, or the completeness rule skipped.",
        l1="Partial: some summaries missing or clearly wrong, missing days treated as zero, or much of the work done on the laptop instead of the remote machine.",
        l0="Left blank, or the remote notebook was not submitted.",
    ),
    dict(
        name="2. Export and hand-off",
        where="04_lab_remote.html L6; 04_lab_local.html L8; the two CSVs",
        l3="lab_hot_days.csv, lab_monthly_rain.csv and lab_remote_measurements.json written on the remote disk with sizes printed; the JSON has every required key; both CSVs submitted; L8 reads all three locally and prints shapes and sizes.",
        l2="Files exported and read locally, with a minor gap: one JSON key missing, a CSV not submitted, or sizes not printed.",
        l1="Only some files exported or read; the local notebook cannot find them; or the summaries were recomputed locally instead of downloaded.",
        l0="Left blank: nothing exported or nothing read locally.",
    ),
    dict(
        name="3. Local visualizations",
        where="04_lab_local.html, L9 and L10",
        l3="Two charts made in Positron: hot days per year (one line per market) and typical monthly rainfall (both markets comparable). Axes labelled with units, informative titles, PNGs saved, wettest month printed, days_reported checked before plotting.",
        l2="Both charts present and readable, with one or two labelling or saving omissions, or wettest month not printed.",
        l1="One chart missing, a chart that does not show both markets, unlabelled axes, or charts made on SageMaker rather than locally.",
        l0="Left blank, or the local notebook was not submitted.",
    ),
    dict(
        name="4. Interpretation",
        where="04_lab_local.html, L11, L13(a), L13(b)",
        l3="Business read (6 to 8 sentences) cites numbers from both charts, addresses heat and rain, flags any coverage caution, and names two datasets to join with where each lives. All five questions answered with specifics from this lab. Weekly-refresh choice gives two reasons tied to measured values.",
        l2="All parts attempted; answers mostly specific, with one weak spot: a question answered generically, a dataset named without a location, or refresh reasons not tied to measurements.",
        l1="Several answers missing or generic; no numbers from the charts; one of L11 or L13 absent.",
        l0="Left blank.",
    ),
    dict(
        name="5. Workflow evidence and shutdown",
        where="Setup cells; L7; L12; L14; submission",
        l3="Remote HTML shows a SageMaker computer name and local HTML shows a different, laptop name. L7 cost estimate printed. L12 shows the resource table and the four sizes. Screenshot shows the space Stopped. Both HTML files and both CSVs uploaded.",
        l2="All present with one gap: cost cell left on its default, one size missing from L12, or the screenshot shows the space page without a clear Stopped status.",
        l1="Evidence incomplete: same computer name in both notebooks, no screenshot, or one HTML file missing.",
        l0="Left blank: no evidence of the workflow or the shutdown.",
    ),
]

GRID = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#808080")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3863")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]


def rubric_table():
    head = [Paragraph("Element (20 pts)", HEAD), Paragraph("3 = Full score (20 pts)", HEAD),
            Paragraph("2 = Nearly complete (14 pts)", HEAD), Paragraph("1 = Partial (7 pts)", HEAD),
            Paragraph("0 = Left blank (0 pts)", HEAD)]
    rows = [head] + [[[Paragraph(e["name"], CELLB), Paragraph(e["where"], SMALL)],
                      Paragraph(e["l3"], CELL), Paragraph(e["l2"], CELL), Paragraph(e["l1"], CELL), Paragraph(e["l0"], CELL)]
                     for e in ELEMENTS]
    t = Table(rows, colWidths=[1.55 * inch, 2.45 * inch, 2.35 * inch, 2.15 * inch, 1.5 * inch], repeatRows=1)
    t.setStyle(TableStyle(GRID + [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")])]))
    return t


def score_sheet():
    rows = [[Paragraph("Element", HEAD), Paragraph("Level (circle one)", HEAD), Paragraph("Points", HEAD), Paragraph("Grader comments", HEAD)]]
    for e in ELEMENTS:
        rows.append([Paragraph(e["name"], CELLB),
                     Paragraph("3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0", CENTER),
                     Paragraph("&nbsp;", CELL), Paragraph("&nbsp;", CELL)])
    rows.append([Paragraph("Total", CELLB), Paragraph("3 = 20, 2 = 14, 1 = 7, 0 = 0", CENTER), Paragraph("/ 100", CELLB), Paragraph("&nbsp;", CELL)])
    t = Table(rows, colWidths=[2.3 * inch, 2.2 * inch, 0.9 * inch, 4.6 * inch], rowHeights=[None] + [0.62 * inch] * 5 + [0.4 * inch])
    t.setStyle(TableStyle(GRID + [("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E7E6E6"))]))
    return t


def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(letter), leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch, title="GSB 5544 AWS Lab Rubric", author="GSB 5544")
    story = [
        Paragraph("GSB 5544 Lab Rubric: Weather as a Business Covariate", H1),
        Paragraph("<b>100 points. 5 elements, 20 points each.</b> Score every element at one of four levels: "
                  "<b>3</b> = full score (20 points), <b>2</b> = nearly complete (14), <b>1</b> = partial (7), <b>0</b> = left blank (0). "
                  "Grade from the two HTML files (<i>04_lab_remote.html</i>, run on SageMaker; <i>04_lab_local.html</i>, run in Positron), "
                  "the two CSV files, and the screenshot of the stopped space. Question labels (L2, L9, ...) match the notebooks.", BODY),
        Spacer(1, 6),
        rubric_table(),
        Spacer(1, 6),
        Paragraph("<b>Quick checks before scoring.</b> (a) The computer names printed by the two setup cells must differ; if they match, element 5 is at most level 1, and the element whose notebook ran in the wrong place (1 or 3) is capped at level 1. "
                  "(b) A cell that stops with <i>NameError: name '____' is not defined</i> was never completed; treat everything downstream of it as blank. "
                  "(c) Missing observations treated as zero (no days_reported, no completeness rule) cap element 1 at level 2.", SMALL),
        PageBreak(),
        Paragraph("GSB 5544 Lab Score Sheet", H1),
        Paragraph("Student: ______________________________________ &nbsp;&nbsp;&nbsp; Grader: ____________________________ &nbsp;&nbsp;&nbsp; Date: ______________", BODY),
        Spacer(1, 10),
        score_sheet(),
        Spacer(1, 10),
        Paragraph("Submission checklist", H2),
        Paragraph("[ ] 04_lab_remote.html &nbsp;&nbsp; [ ] 04_lab_local.html &nbsp;&nbsp; [ ] lab_hot_days.csv &nbsp;&nbsp; [ ] lab_monthly_rain.csv &nbsp;&nbsp; [ ] screenshot: space Stopped &nbsp;&nbsp; [ ] computer names differ", BODY),
        Spacer(1, 8),
        Paragraph("Overall comments", H2),
        Table([[Paragraph("&nbsp;", CELL)]], colWidths=[10 * inch], rowHeights=[1.6 * inch],
              style=TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#808080"))])),
    ]
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
