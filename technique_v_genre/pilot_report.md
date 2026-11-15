# Full-corpus run — Experiments A, B, C

Run on the complete corpus: 21 files, 11952 passages, 6 authors, 21 titles, 72167 total (passage, reader, run) rows. Every passage is read by gpt-5.4 at k=3; nearly all are also read by qwen3-max at k=3. 33 rows (11 passages) across Ulysses, The Sound and the Fury, and The Ambassadors were permanently rejected by DashScope's content moderation (data_inspection_failed) and are instead read by a local open-weight model (reader_id `qwen3.6-local`, Qwen/Qwen3.6-27B served via vLLM — see `fill_qwen_gaps.slurm`), substituted in place of the missing qwen3-max reading for just those passages. This is a genuinely different model, correctly labeled as its own reader, not merged into the qwen3-max column.

## Experiment A — per-author fingerprints

| author | n (passage x reader x run) | mean level | std (spread) | level 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| Faulkner | 3539 | 2.07 | 1.23 | 21% | 10% | 10% | 59% |
| Ford | 2335 | 0.79 | 1.19 | 64% | 13% | 4% | 20% |
| James | 10602 | 0.44 | 0.58 | 60% | 36% | 3% | 1% |
| Joyce | 10386 | 1.37 | 1.22 | 31% | 32% | 5% | 31% |
| Richardson | 5118 | 0.91 | 0.71 | 26% | 61% | 8% | 4% |
| Woolf | 24957 | 0.82 | 0.88 | 43% | 40% | 10% | 7% |

**By title** (finer grain, previews the Joyce/Woolf developmental-arc question):

| author | title | year | n | mean level | std |
|---|---|---|---|---|---|
| Faulkner | As I Lay Dying | 1930 | 1480 | 1.89 | 1.28 |
| Faulkner | The Sound and the Fury | 1929 | 2059 | 2.21 | 1.18 |
| Ford | The Good Soldier | 1915 | 2335 | 0.79 | 1.19 |
| James | The Ambassadors | 1903 | 4826 | 0.48 | 0.63 |
| James | The Wings of the Dove | 1902 | 5776 | 0.40 | 0.53 |
| Joyce | A Portrait of the Artist as a Young Man | 1916 | 2182 | 0.69 | 0.76 |
| Joyce | Dubliners | 1914 | 1262 | 0.38 | 0.60 |
| Joyce | Ulysses | 1922 | 6942 | 1.77 | 1.22 |
| Richardson | Backwater | 1916 | 1651 | 0.88 | 0.71 |
| Richardson | Honeycomb | 1917 | 1653 | 0.96 | 0.72 |
| Richardson | Pointed Roofs | 1915 | 1814 | 0.87 | 0.69 |
| Woolf | Between the Acts | 1941 | 1272 | 1.02 | 0.80 |
| Woolf | Flush | 1933 | 842 | 0.37 | 0.52 |
| Woolf | Jacob's Room | 1922 | 1505 | 0.59 | 0.68 |
| Woolf | Mrs Dalloway | 1925 | 2224 | 0.89 | 0.52 |
| Woolf | Night and Day | 1919 | 4638 | 0.37 | 0.59 |
| Woolf | Orlando | 1928 | 2106 | 0.51 | 0.69 |
| Woolf | The Voyage Out | 1915 | 3310 | 0.39 | 0.63 |
| Woolf | The Waves | 1931 | 2411 | 2.50 | 0.81 |
| Woolf | The Years | 1937 | 3611 | 0.98 | 0.72 |
| Woolf | To the Lighthouse | 1927 | 3038 | 0.76 | 0.52 |

**Predictions (HANDOFF sec. 5, Experiment A):**

- Faulkner (genre pole): predicted spread >1.2 — full corpus shows std=1.23 (CONFIRMED).
- Woolf (technique author): predicted spread ~0.5, peak at level 1 — full corpus shows std=0.88, mean=0.82.
- Richardson (technique author): predicted spread ~0.5, peak at level 1 — full corpus shows std=0.71, mean=0.91 (now all three Pilgrimage volumes in the corpus: Pointed Roofs, Backwater, Honeycomb).
- James (control, floor): predicted mass at levels 0-1 — full corpus mean=0.44 (CONFIRMED, lowest or near-lowest mean in panel).
- Ford (control, boundary case): full corpus mean=0.79, std=1.19.
- Joyce developmental arc (Dubliners -> Portrait -> Ulysses marching rightward): now testable with all three works at full or near-full completion.
  - A Portrait of the Artist as a Young Man: mean=0.69, n=2182
  - Dubliners: mean=0.38, n=1262
  - Ulysses: mean=1.77, n=6942
  - Marches rightward by publication order: CONFIRMED (Dubliners -> A Portrait of the Artist as a Young Man -> Ulysses).

## Author variance decomposition — within-oeuvre texture

For authors with multiple titles in the corpus, splits total variance in mediation_level into a between-title component (which book you're reading) and a within-title component (passage-to-passage noise inside one book). Each passage is first collapsed to its mean level across readers/runs, so titles aren't weighted by how many reader-runs happen to exist for them. eta^2 = between-title SS / total SS: the fraction of an author's overall spread that is explained by which book you're reading, rather than by ordinary passage-to-passage variation.

| author | titles | n passages | std of title means | between-title variance share (eta^2) |
|---|---|---|---|---|
| Faulkner | 2 | 728 | 0.238 | 0.020 |
| James | 2 | 2107 | 0.046 | 0.004 |
| Joyce | 3 | 2077 | 0.683 | 0.217 |
| Richardson | 3 | 912 | 0.043 | 0.003 |
| Woolf | 10 | 4462 | 0.639 | 0.530 |

Per-title means, low to high, within each author:

- **Faulkner** (2 titles, n=728 passages):
    - As I Lay Dying: 1.64
    - The Sound and the Fury: 1.97
- **James** (2 titles, n=2107 passages):
    - The Wings of the Dove: 0.37
    - The Ambassadors: 0.44
- **Joyce** (3 titles, n=2077 passages):
    - Dubliners: 0.32
    - A Portrait of the Artist as a Young Man: 0.65
    - Ulysses: 1.63
- **Richardson** (3 titles, n=912 passages):
    - Backwater: 0.85
    - Pointed Roofs: 0.85
    - Honeycomb: 0.92
- **Woolf** (10 titles, n=4462 passages):
    - Night and Day: 0.35
    - The Voyage Out: 0.36
    - Flush: 0.36
    - Orlando: 0.46
    - Jacob's Room: 0.56
    - To the Lighthouse: 0.79
    - Mrs Dalloway: 0.87
    - The Years: 0.91
    - Between the Acts: 1.00
    - The Waves: 2.49

Authors with only one title in the corpus (Ford) can't be scored on this axis at all — not evidence of a fixed style, just no within-oeuvre comparison available yet.

## Experiment B — geometry of the cline

Pooled mediation_level density across all authors/readers/runs in the full corpus:

| level | name | proportion | raw n |
|---|---|---|---|
| 0 | psycho-narration | 41.9% | 23850 |
| 1 | narrated monologue / FID | 36.8% | 20936 |
| 2 | quoted monologue | 7.6% | 4300 |
| 3 | autonomous monologue | 13.8% | 7851 |

Valleys detected at level(s): [2]. Monotonic across 0-3: False.
Interpretation: with only 4 discrete bins, this is still a coarse read, not a real dip-test/GMM fit (per HANDOFF sec. 5 Experiment B, that would want a continuous soft score). Reported here as a first look at full-corpus scale, not a final verdict.

`fit_category == boundary_tension` rate by mediation_level (bonus check from HANDOFF sec. 5):

| level | boundary_tension rate |
|---|---|
| 0 | 64.1% |
| 1 | 44.6% |
| 2 | 74.6% |
| 3 | 44.3% |

## Experiment C — moving frontier, diachronic arm

Year range: 1902-1941 (21 titles, 6 authors).

| year | title | no_type_fits | boundary_tension | antimimetic | n |
|---|---|---|---|---|---|
| 1902 | The Wings of the Dove | 0.2% | 45.8% | 0.03% | 7008 |
| 1903 | The Ambassadors | 0.3% | 44.4% | 0.13% | 6282 |
| 1914 | Dubliners | 0.1% | 28.5% | 0.00% | 2448 |
| 1915 | Pointed Roofs | 0.0% | 47.4% | 0.00% | 2064 |
| 1915 | The Voyage Out | 0.3% | 37.2% | 0.00% | 4614 |
| 1915 | The Good Soldier | 0.8% | 44.3% | 0.19% | 2640 |
| 1916 | Backwater | 0.3% | 41.5% | 0.00% | 2160 |
| 1916 | A Portrait of the Artist as a Young Man | 0.4% | 39.2% | 0.16% | 3114 |
| 1917 | Honeycomb | 0.1% | 47.3% | 0.00% | 1968 |
| 1919 | Night and Day | 0.1% | 42.2% | 0.02% | 5370 |
| 1922 | Ulysses | 1.7% | 42.0% | 1.32% | 9736 |
| 1922 | Jacob's Room | 1.0% | 52.6% | 0.38% | 1824 |
| 1925 | Mrs Dalloway | 0.0% | 52.5% | 0.00% | 2280 |
| 1927 | To the Lighthouse | 0.3% | 49.5% | 0.22% | 3181 |
| 1928 | Orlando | 0.9% | 42.2% | 0.68% | 2664 |
| 1929 | The Sound and the Fury | 0.6% | 30.9% | 0.22% | 3576 |
| 1930 | As I Lay Dying | 0.3% | 40.6% | 0.14% | 2118 |
| 1931 | The Waves | 0.8% | 48.8% | 0.60% | 2502 |
| 1933 | Flush | 0.0% | 46.9% | 0.00% | 1008 |
| 1937 | The Years | 0.0% | 47.6% | 0.02% | 4122 |
| 1941 | Between the Acts | 0.6% | 59.2% | 0.27% | 1488 |

Spearman correlation, year vs. no_type_fits_rate across titles: rho=0.104, p=0.655. No clear trend at the title level — this is a by-title (21 titles) summary; HANDOFF sec. 5's passage-level diachronic curve would be a finer-grained follow-up.
