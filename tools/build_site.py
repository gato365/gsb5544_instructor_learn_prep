#!/usr/bin/env python3
"""Build docs/index.html for the course website from site_config.json.

Usage:  python3 tools/build_site.py        (run from the repo root, then commit)

Publishing rules, per item in site_config.json:
  "publish": false          -> the whole item shows only its "placeholder" text
  "publish_solution": false -> student links show, solution shows "solution_placeholder"
  "extended": <notebook path> -> a third row, "Extended", for a fully worked walkthrough notebook
  "page": <repo path to an .html file> -> the file is copied into docs/ (at "page_dest",
      default: its path relative to assignments/practice_activities/) and linked as an
      interactive page served by GitHub Pages, next to any student/solution notebooks.
  "listed": false           -> (pages only) the file is still copied into docs/ and served at its
      URL, but no card is shown on the index page; for instructor-facing pages.
Files always stay in the repository; this only controls what the website links to.
Each week renders as a collapsible <details> block (id "week-N"): the most recent week is open by default,
the others collapsed; the page remembers which weeks a visitor opened (localStorage) and offers Expand all /
Collapse all. A link to index.html#week-3 opens that week.
NOTE: the repo itself is public, so hiding here hides from the *page*, not from
someone browsing the repository directly.
"""
import json, html, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = json.load(open(ROOT / "site_config.json"))
REPO, BR = cfg["repo"], cfg["branch"]

def links(path):
    colab = f"https://colab.research.google.com/github/{REPO}/blob/{BR}/{path}"
    view  = f"https://github.com/{REPO}/blob/{BR}/{path}"
    return (f'<a class="btn colab" href="{colab}">Open in Colab</a>'
            f'<a class="btn" href="{view}">View notebook</a>')

rows = []
n_weeks = len(cfg["weeks"])
for w_i, wk in enumerate(cfg["weeks"]):
    # each week is a collapsible <details>; the most recent week starts open, the rest collapsed
    if w_i:
        rows.append('</div></details>')
    is_open = " open" if w_i == n_weeks - 1 else ""
    rows.append(f'<details class="week" id="week-{w_i + 1}"{is_open}><summary><h2>{html.escape(wk["label"])}</h2>'
                f'<span class="count">{len(wk["items"])} item{"s" if len(wk["items"]) != 1 else ""}</span></summary><div class="items">')
    for it in wk["items"]:
        title = html.escape(it["title"])
        if not it.get("publish", True):
            ph = html.escape(it.get("placeholder", "Coming soon."))
            rows.append(f'<div class="card placeholder"><h3>{title}</h3>'
                        f'<p class="soon">&#128274; {ph}</p></div>')
            continue
        body = ""
        if it.get("page"):
            src = ROOT / it["page"]
            dest_rel = it.get("page_dest") or str(pathlib.Path(it["page"]).relative_to("assignments/practice_activities"))
            dest = ROOT / "docs" / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            if not it.get("listed", True):
                print(f"copied unlisted page {it['page']} -> docs/{dest_rel}")
                continue
            view = f"https://github.com/{REPO}/blob/{BR}/{it['page']}"
            body += (f'<div class="row"><span class="tag page">Interactive</span>'
                     f'<a class="btn colab" href="{dest_rel}">Open page</a>'
                     f'<a class="btn" href="{view}">View source</a></div>')
        if it.get("student"):
            body += f'<div class="row"><span class="tag">Student</span>{links(it["student"])}</div>'
        if it.get("solution"):
            if it.get("publish_solution", True):
                body += f'<div class="row"><span class="tag sol">Solution</span>{links(it["solution"])}</div>'
            else:
                ph = html.escape(it.get("solution_placeholder", "Solution not posted yet."))
                body += f'<div class="row"><span class="tag sol">Solution</span><em class="soon">{ph}</em></div>'
        if it.get("extended"):
            body += f'<div class="row"><span class="tag ext">Extended</span>{links(it["extended"])}</div>'
        rows.append(f'<div class="card"><h3>{title}</h3>{body}</div>')
rows.append('</div></details>')

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg["title"])}</title>
<link rel="icon" type="image/png" sizes="64x64" href="favicon.png">
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
        background:#f6f7f9; color:#1c2733; }}
 main {{ max-width: 860px; margin: 0 auto; padding: 2rem 1rem 4rem; }}
 h1 {{ border-bottom: 3px solid #154734; padding-bottom:.4rem; }}
 h2 {{ color:#154734; margin:0; font-size:1.3rem; }}
 details.week {{ margin-top:1.4rem; border-top:2px solid #d5ded9; }}
 details.week summary {{ list-style:none; cursor:pointer; display:flex; align-items:center; gap:.7rem;
        padding:.9rem .2rem; user-select:none; }}
 details.week summary::-webkit-details-marker {{ display:none; }}
 details.week summary::before {{ content:"\25B6"; font-size:.75rem; color:#154734; transition:transform .15s; }}
 details.week[open] summary::before {{ transform:rotate(90deg); }}
 details.week summary:hover h2 {{ text-decoration:underline; }}
 details.week summary:focus-visible {{ outline:3px solid #154734; outline-offset:2px; border-radius:6px; }}
 .count {{ margin-left:auto; font-size:.78rem; color:#5a6a79; white-space:nowrap; }}
 .items {{ padding-bottom:.6rem; }}
 .controls {{ display:flex; gap:.5rem; margin:1.2rem 0 0; }}
 .controls button {{ font:inherit; font-size:.85rem; color:#154734; background:#fff; border:1px solid #b9c8c0;
        border-radius:6px; padding:.3rem .7rem; cursor:pointer; }}
 .controls button:hover {{ filter:brightness(1.08); }}
 .card {{ background:#fff; border:1px solid #e2e6ea; border-radius:10px;
         padding: .9rem 1.1rem; margin:.7rem 0; }}
 .card h3 {{ margin:.1rem 0 .6rem; font-size:1.05rem; }}
 .row {{ margin:.35rem 0; display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; }}
 .tag {{ font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
        background:#e8f0ec; color:#154734; border-radius:4px; padding:.15rem .45rem; }}
 .tag.sol {{ background:#fdf3d7; color:#7a5b00; }}
 .tag.page {{ background:#e4ecfa; color:#1f4e9c; }}
 .tag.ext {{ background:#efe3f7; color:#5b2a86; }}
 .btn {{ font-size:.85rem; text-decoration:none; color:#154734; border:1px solid #b9c8c0;
        border-radius:6px; padding:.25rem .6rem; background:#fff; }}
 .btn.colab {{ background:#154734; color:#fff; border-color:#154734; }}
 .btn:hover {{ filter:brightness(1.12); }}
 .placeholder {{ background:#eef1f4; border-style:dashed; }}
 .soon {{ color:#5a6a79; }}
 footer {{ color:#8a97a3; font-size:.8rem; margin-top:3rem; }}
</style></head><body><main>
<h1>{html.escape(cfg["title"])}</h1>
<p>Course notebooks. <em>Student</em> versions have blanks to fill in; <em>Solution</em>
versions are completed and rendered with outputs. <em>Interactive</em> pages run in the
browser and pair with the notebook listed beside them. <em>Extended</em> notebooks are fully worked
walkthroughs: the question behind each step, the tables before and after.
Click a week to open or close it.</p>
<div class="controls"><button id="expand" type="button">Expand all</button><button id="collapse" type="button">Collapse all</button></div>
{''.join(rows)}
<footer>Generated by <code>tools/build_site.py</code> from <code>site_config.json</code>.</footer>
</main>
<script>
(function () {{
  var weeks = document.querySelectorAll('details.week'), KEY = 'gsb5544-open-weeks';
  var saved = null;
  try {{ saved = JSON.parse(localStorage.getItem(KEY)); }} catch (e) {{}}
  if (Array.isArray(saved)) weeks.forEach(function (d) {{ d.open = saved.indexOf(d.id) !== -1; }});
  if (location.hash) {{ var target = document.querySelector(location.hash); if (target && target.classList.contains('week')) target.open = true; }}
  function remember() {{
    var open = []; weeks.forEach(function (d) {{ if (d.open) open.push(d.id); }});
    try {{ localStorage.setItem(KEY, JSON.stringify(open)); }} catch (e) {{}}
  }}
  weeks.forEach(function (d) {{ d.addEventListener('toggle', remember); }});
  document.getElementById('expand').onclick = function () {{ weeks.forEach(function (d) {{ d.open = true; }}); remember(); }};
  document.getElementById('collapse').onclick = function () {{ weeks.forEach(function (d) {{ d.open = false; }}); remember(); }};
}})();
</script>
</body></html>
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "index.html").write_text(page)
(ROOT / "docs" / ".nojekyll").write_text("")
print("wrote docs/index.html")
