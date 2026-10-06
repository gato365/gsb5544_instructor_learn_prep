# Week 7 — Association Rules

| File | What it is | Built by |
|---|---|---|
| `GSB5544_Topic_7_1_Association_Rules-empty.ipynb` | Student notebook: *Who gets played together?* Simulated playlists of Latin and hip-hop artists → one-hot table → support → confidence and lift by hand → `apriori` + `association_rules` → why confidence alone misleads when one item (Bad Bunny) is very popular. `____` blanks and "Your answer" prompts. 20 minutes. | `tools/build_week7_topics.py` |
| `GSB5544_Topic_7_1_Association_Rules-solution.ipynb` | The same notebook, completed and executed. | `tools/build_week7_topics.py` |
| `GSB5544_PA_7_1_Association_Rules.ipynb` | PA 7.1 student notebook (Groceries data, 19 parts). Kept as received. | — |
| `GSB5544_PA_7_1_Association_Rules-solution.ipynb` | Instructor key: approach, code, what each piece does, expected output, common mistakes; written answers for parts 5 and 18 and an instructor note on the part-10 predictions. Executed against the live data file. | `tools/build_week7_pas.py` |

Edit the generators, not the `.ipynb` files, and re-run them.

## About the Topic's data

The sixteen artist names are real (eight Latin, eight hip-hop), but the **400 playlists are simulated** inside the
notebook with a fixed seed (`np.random.default_rng(5544)`), and the notebook says so. Each simulated listener has a
taste (Latin / hip-hop / mixed), Bad Bunny is added to 75 % of playlists regardless of taste (the "whole milk" of
the data), and four pairings are planted: J. Cole → Kendrick Lamar, Karol G → Shakira, Future → Travis Scott, and
the cross-genre Cardi B → J Balvin. The point of simulating is that students can check whether the method recovers
what was planted — it does: the four planted pairs are the four highest-lift rules (2.49, 2.43, 2.16, 1.77), while
sorting by confidence puts "→ Bad Bunny" rules (lift ≈ 1) on top. Nothing in the notebook is a claim about real
listening habits. Changing the seed changes the numbers quoted in the ✅ answers; re-check them if you do.

## PA 7.1 notes

- The data (`groceries.csv`, 9,835 baskets, 169 items) is read live from GitHub; building the key needs a network
  connection.
- `make_rules` (part 13) is the helper from the reading. The key defines it as
  `association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)` so the notebook runs
  on its own; if the reading's version differs cosmetically, the results are the same.
- Headline numbers: 2,159 single-item baskets (22 %); whole milk support 0.256; {yogurt} → {whole milk}
  confidence 0.402, lift 1.57; {soda} → {whole milk} lift 0.90; 333 frequent itemsets at 1 % support; 234 rules at
  confidence ≥ 0.20; 108 "interesting" rules (confidence ≥ 0.30 and lift ≥ 1.5); beef → root vegetables lift 3.04.

## Rebuild checklist

```bash
/opt/anaconda3/bin/python -m pip install mlxtend     # once
python3 tools/build_week7_topics.py    # regenerate + execute Topic 7.1 (-empty and -solution)
python3 tools/build_week7_pas.py       # rebuild + execute the PA 7.1 instructor solution (downloads groceries.csv)
python3 tools/build_site.py            # rebuild docs/index.html
```
