# Week 6 — Writing Functions (and lambda / map)

| File | What it is | Built by |
|---|---|---|
| `GSB5544_Topic_6_1_Functions_Lambda_Map-empty.ipynb` | Student notebook: *How repetitive is a song?* Lyrics → lists → `def` → `lambda` → `map()`, then practice (predict, complete, explain, debug, apply, interpret), a two-song investigation, and optional extensions (`filter`, `sorted(key=)`, comprehensions, `Counter`). `____` blanks and "Your answer" prompts. | `tools/build_week6_topics.py` |
| `GSB5544_Topic_6_1_Functions_Lambda_Map-solution.ipynb` | The same notebook, completed and executed. | `tools/build_week6_topics.py` |
| `Practice_Activity_6_1_Writing_Functions.ipynb` | PA 6.1 student notebook (penguins check-in, `times_seven`, `add_or_subtract`, scope). Kept as received. | — |
| `Practice_Activity_6_1_Writing_Functions-solution.ipynb` | Instructor key: approach, code, what each piece does, expected output, common mistakes. Executed. | `tools/build_week6_pas.py` |

Edit the generators, not the `.ipynb` files, and re-run them. Cells that demonstrate an error on purpose are
tagged `raises-exception`; every other cell must run cleanly or the build stops.

**Heads-up for PA 6.1, Questions 3–4.** In the handout's `add_or_subtract`, `return res` is indented inside the
`else` block, so as printed the function returns `None` for `"add"` and `"subtract"`. The solution gives the
intended answers (−1; 5, 3, 9, 7) and also shows the as-printed behaviour. Fix the handout or use it as a
discussion point.

## The lyrics in the notebooks

All three texts are in the public domain in the US (published before 1929), so they can live in this public
repository:

| Variable | Song | Words by | Published |
|---|---|---|---|
| `lyrics` | *Twinkle, Twinkle, Little Star* (first verse as commonly sung) | Jane Taylor, "The Star" | 1806 |
| `ballgame_lyrics` | *Take Me Out to the Ball Game* (chorus) | Jack Norworth | 1908 |
| `rowboat_lyrics` | *Row, Row, Row Your Boat* | traditional | 1852 |

The student notebook needs no internet connection, API key, or extra package.

## Optional instructor prep: fetching other lyrics with LyricsGenius

Use this only if you want to swap in a different song *for your own preparation or a live class demo*. It is not
part of the student sequence, and the notebooks never depend on it.

Checked September 2026 against the [PyPI page](https://pypi.org/project/lyricsgenius/) and the
[project README](https://github.com/johnwmillr/LyricsGenius):

1. **Requirements.** Python 3.11 or newer; `pip install lyricsgenius`.
2. **Access token.** Create a free Genius account, register an API client at <https://genius.com/api-clients>,
   and generate a **client access token** on that page.
3. **Keep the token out of the notebooks and this repo.** Put it in an environment variable. LyricsGenius reads
   `GENIUS_ACCESS_TOKEN` automatically when no token is passed:
   ```bash
   export GENIUS_ACCESS_TOKEN="paste-your-token-here"     # in your shell, not in a notebook
   ```
4. **Fetch, on your own computer.**
   ```python
   from lyricsgenius import Genius

   genius = Genius(remove_section_headers=True,        # drops "[Chorus]", "[Verse 1]", ...
                   skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"])
   song = genius.search_song("Song Title", "Artist Name")
   print(song.title, "-", song.artist)
   text = song.lyrics
   ```
   The Genius API returns song metadata but not lyrics, so the package reads the lyrics from the song's web
   page. That step is often **blocked (HTTP 403 / captcha) from cloud and shared networks, including Google
   Colab**. Run it from your own machine, and expect it to break occasionally when Genius changes its pages.
5. **Clean before use.** Fetched text can contain a header line (e.g. "… Lyrics"), blank lines between
   sections, and section labels if `remove_section_headers` is off. Print it, trim it to a short excerpt, and
   state your cleaning rules as the notebook does in Part E.
6. **Copyright.** Most lyrics on Genius are under copyright. **Do not commit fetched lyrics to this repository**
   (it is public) or put them in student notebooks. Paste a short excerpt into a local copy for a class demo, and
   keep the public-domain texts as the version students download.

**Fallback.** If the token or the fetch fails, nothing changes for students: the notebooks already contain
three complete public-domain songs.

## Rebuild checklist

```bash
/opt/anaconda3/bin/python -m pip install palmerpenguins   # once: the PA 6.1 solution needs it
python3 tools/build_week6_topics.py    # regenerate + execute Topic 6.1 (-empty and -solution)
python3 tools/build_week6_pas.py       # rebuild + execute the PA 6.1 instructor solution
python3 tools/build_site.py            # rebuild docs/index.html
```
