# Non-Woolf career coverage — what was missing, what's now fixed

## The problem

The chapter's Section 2.4 and Section 3 argue about Woolf's *fingerprint
relative to other authors*, and the coauthor's ask is for work-to-work
development across careers. The scored corpus could not support that
symmetrically. Coverage before this expansion:

| author | works | span | mean mediation level, first → last |
|---|---|---|---|
| Woolf | 10 | 1915–1941 (26y) | 0.60 → 1.03 (peak 2.28 at *The Waves*) |
| Joyce | 3 | 1914–1922 (8y) | 0.55 → 1.67 |
| Richardson | 3 | 1915–1917 (**2y**) | 0.93 → 0.99 |
| James | 3 | 1902–1903 (**1y**) | 0.62 → 0.67 |
| Faulkner | 2 | 1929–1930 (**1y**) | 1.98 → 1.88 |
| Ford | **1** | 1915 (**0y**) | 0.90 |

Woolf had a 26-year arc; everyone else had a snapshot. Richardson's three
volumes are consecutive years, James's "three works" are really two books
(*Wings of the Dove* is one novel split across two Gutenberg volumes), and
Ford was a single point. Only Joyce had anything resembling a trajectory —
and his is the one the field already tells as a development story.

The asymmetry is not just aesthetic. Woolf's arc is the only one where a
*change* in mediation level can be distinguished from a *difference between
books*, so any claim that her drift is distinctive currently has no
comparison class. Richardson in particular was doing the FID-sharp thing
Woolf is measured against, and three adjacent years cannot show whether she
held that position or moved.

## What was added

All public domain, verified downloadable from Gutenberg, downloaded and
segmented with the existing pipeline (`expand_nonwoolf.py` → `segment.py`).

**Richardson — *Pilgrimage* vols 4–9.** The headline fix. Volumes 4 through
9 are all on Gutenberg, taking her from 3 works / 2 years to **9 works /
1915–1927 (12 years)** — the only non-Woolf arc long enough to be read
against Woolf's on its own terms.

| vol | year | title | passages |
|---|---|---|---|
| 4 | 1919 | The Tunnel | 608 |
| 5 | 1919 | Interim | 356 |
| 6 | 1921 | Deadlock | 487 |
| 7 | 1923 | Revolving Lights | 394 |
| 8 | 1925 | The Trap | 218 |
| 9 | 1927 | Oberland | 235 |

**Ford — *Parade's End* vols 1–2.** Takes the Ford control from n=1 to
n=3, 1915–1925: *Some Do Not...* (1924, 705 passages) and *No More
Parades* (1925, 567).

**May Sinclair — 8 works, 1897–1922.** She used "stream of consciousness"
as a critical term in 1918, reviewing Richardson's *Pilgrimage* in *The
Egoist*, and then wrote in the mode herself. The arc is picked to straddle
that coinage rather than to sample her output evenly — *Audrey Craven*
(1897) and *The Helpmate* (1907) are the conventional "before", *The Tree
of Heaven* (1917) through *Harriett Frean* (1922) the after.

| year | title | passages |
|---|---|---|
| 1897 | Audrey Craven | 469 |
| 1907 | The Helpmate | 696 |
| 1913 | The Combined Maze | 592 |
| 1914 | The Three Sisters | 596 |
| 1917 | The Tree of Heaven | 597 |
| 1919 | Mary Olivier: A Life | 730 |
| 1920 | The Romantic | 278 |
| 1922 | Life and Death of Harriett Frean | 129 |

At 25 years she is the closest match in the corpus to Woolf's 26, and the
only author whose arc crosses the moment the term itself was coined —
which makes her the natural test case for Section 1's genre-vs-technique
framing and Experiment C's "moving frontier".

Total: **7,657 new passages**, ~69k scored rows at 3 readers × k=3 — about
59% on top of the 116,203 rows already in `scored/`. Sinclair is over half
of that on her own; `score_expansion.sh` keeps her files in a separate
array so the arc can be trimmed to four points (*Audrey Craven*, *The Tree
of Heaven*, *Mary Olivier*, *Harriett Frean*) if the budget is tight. Segmentation is
verified additive: all 25 pre-existing passage files are byte-identical
after re-running `segment.py`, so `scored/` passage IDs still align.

## Verification

Every text is now checked against what the manifest claims it is, in three
layers, because the corpus has two kinds of file and they admit different
kinds of proof:

1. **Pinning** (offline, all 38 works) — sha256, word count and a recorded
   opening excerpt. Proves a file has not changed since it was confirmed;
   does not prove it was ever right.
2. **Provenance** (online, the 28 Gutenberg-sourced works) — the filename's
   `pg<id>` is resolved against Gutenberg's RDF metadata and title, author
   and volume must match. This is the layer that catches "wrong ID,
   plausible filename".
3. **Corroboration** (offline) — the title must appear in the text's own
   front matter, for the files that have one.

`python verify_corpus.py --online` → **38 works, 0 failures, 0 warnings**.
`python -m unittest test_corpus_integrity` → **22 tests, all passing**.

Three things worth knowing about what this does and doesn't establish:

- **Ten files are not Gutenberg-sourced** (nine Woolf texts plus
  *As I Lay Dying*) and their original download is not recorded anywhere in
  the repo. Layer 2 is unavailable for them. They now carry an explicit
  `provenance_note` saying so — pinned by hash against the state in which
  every result to date was produced, which detects later drift but cannot
  confirm the original acquisition. That is a real limit, recorded rather
  than papered over.
- **Publication years cannot be verified from Gutenberg.** `dcterms:issued`
  is the date PG posted the file, often the 2000s. Years are asserted from
  scholarship; the checker only bounds them by the author's lifespan.
- **Two Woolf title checks were passing for the wrong reason.**
  `orlando_cleaned.txt` matched on "Orlando's father" — the character — and
  `the_years.txt` on the phrase "the days the weeks the years passed".
  Both are now marked `title_on_titlepage: false` rather than left as
  checks that pass coincidentally.

The negative tests matter more than the passing corpus here. Writing them
surfaced two genuine bugs in the checker: the duplicate detector was keyed
on filename, so a repeated manifest entry compared equal to itself and was
skipped; and the volume check read `extract_volume(title) or volume_field`,
so a title reading "(vol. 1)" silently masked a `volume: 2` field — exactly
the swap the check exists to catch. Both are fixed and both now have tests.

`score_expansion.sh` runs `verify_corpus.py --online` and aborts if it
fails, so a bad download cannot reach the scorer.

## Second pass: the male authors

The first pass gave arcs to Woolf, Richardson and Sinclair — three women in
the same SoC lineage — and left all four male authors as 1–8 year
snapshots. That is a confound, not just uneven coverage: every diachronic
claim would have been measured almost entirely within one lineage, with no
way to separate a period effect from an author-selection effect. It lands
hardest where the chapter is most exposed, since Section 1.2 makes
Woolf-vs-Joyce the central contrast and Joyce is the one author who
structurally cannot be given an arc.

Of the four, only Joyce was actually blocked. The other three were
judgement calls presented as constraints:

| author | first pass said | actually |
|---|---|---|
| Joyce | maxed out | correct — *Finnegans Wake* (1939) not US PD until 2035, *Stephen Hero* (1944) likewise |
| James | "trivially extendable", parked as a control | available all along; 1875–1904 is a 29-year arc, longer than Woolf's |
| Ford | tetralogy incomplete | true, but his 1906–13 work was written into the downloader behind an unused flag |
| Faulkner | "not on Gutenberg" | wrong stopping point — *As I Lay Dying* is already non-Gutenberg-sourced, and Standard Ebooks has *Soldiers' Pay* (1926) |

**James added at four career points** (1875–1904, 4,188 passages across 5
files — *The Portrait of a Lady* is two Gutenberg volumes, the same split
*Wings of the Dove* already has):

| year | title | passages |
|---|---|---|
| 1875 | Roderick Hudson | 826 |
| 1881 | The Portrait of a Lady (2 vols) | 1,458 |
| 1897 | What Maisie Knew | 594 |
| 1904 | The Golden Bowl | 1,310 |

Four points rather than his full bibliography: the same 29-year span at
roughly half the cost. *The Bostonians* (1886), *The Tragic Muse* (1890)
and *The Awkward Age* (1899) would densify the "major phase" transition if
the budget later allows.

## Third pass: off-Gutenberg sources

Four sources were checked for the five works that are public domain but not
on Gutenberg (Faulkner 1926–29; the last two *Parade's End* volumes).

- **Princeton's online holdings** — nothing. Faulkner's 7 online records are
  all criticism, a filmed stage production, and a translation; Ford's novels
  all resolve to Gutenberg passthroughs. The catalog JSON API is public, so
  this can be re-run for any author without a login; `electronic_portfolio_s`
  names the provider per record, which is the field that separates a real
  e-text from an audiobook.
- **Institutional Books (Harvard)** — zero Faulkner, no *Parade's End*. Only
  3,504 English fiction records dated 1900–1930 in 983k books, because its
  public-domain determination predates those works entering the public
  domain. Also note its `date1_src` is edition date, not first publication.
- **Faded Page** — reachable only indirectly, but it is upstream of the
  Standard Ebooks texts below.
- **Standard Ebooks** — has three of the five, hand-proofread.

**Added (Standard Ebooks):**

| work | year | passages |
|---|---|---|
| Faulkner, *Soldiers' Pay* | 1926 | 537 |
| Ford, *A Man Could Stand Up—* | 1926 | 410 |
| Ford, *The Last Post* | 1928 | 413 |

*Parade's End* is now complete (4/4) and Ford spans 1915–1928 across 5
works. Faulkner moves to 3 works, 1926–1930, breaking his one-year span.
`verify_corpus.py` gained a Standard Ebooks provenance layer: each title's
canonical `content.opf` from its public GitHub repo, checked the way
Gutenberg RDF is (SE's own OPDS feeds are gated behind a patron login).

**Deliberately not ingested:** Faulkner's *Mosquitoes* (1927) and
*Sartoris* (1929) exist only as Internet Archive scans. They would be the
only OCR'd texts in a corpus of clean e-text — a source difference
correlating perfectly with author, on the exact feature the cline measures.
See the OCR control below before deciding.

**Still unaddressed:** Ford's 1906–13 work (~1,800 passages, cheapest
remaining gain).

## The OCR control

Faulkner's *Soldiers' Pay* exists both ways, and not coincidentally:
Standard Ebooks names the Internet Archive scan as its upstream source, so
the pair is one edition lineage — raw OCR against its own proofread
descendant. Same words, same edition, differing only in transcription.

The quotation-mark damage is already measurable without scoring anything:

| | words | opening “ | closing ” |
|---|---|---|---|
| IA raw OCR | 92,009 | 2,574 | 2,696 |
| SE proofread | 89,620 | 2,948 | 2,920 |

Two things to read there. The OCR text has **374 fewer opening quotes than
its own proofread descendant** — 12.7% of them lost or misread. And its
closing quotes *exceed* its opening quotes by 122, which no typographic
convention produces; that direction is only reachable through corruption.

A correction to an earlier claim in this document's first draft: raw
open-vs-close counts are not a validity test on their own. English sets a
multi-paragraph quotation with an opening mark on every paragraph and a
closing mark only at the end, so opens legitimately exceed closes in clean
prose — the proofread text does exactly this, by 28. Only the reverse
direction is diagnostic. By that corrected test, *Mosquitoes* is corrupted
(100 excess closes) while *Sartoris*'s 145 excess opens is ambiguous.

`ocr_control/run_ocr_control.py` scores 260 evenly-spaced passages from each
arm with the corpus's own readers and prompt. **Prediction, recorded before
scoring:** if OCR is harmless the two distributions are indistinguishable;
if it matters, level-2 mass migrates down to level 1 or 0, because framed
interior speech stripped of its quotation marks reads as unframed. Nothing
in the control is written to `positives/` or the manifest.

## Three caveats that belong in the writeup

**1. Richardson's *Pilgrimage* is not nine works.** Richardson insisted it
was one novel in chapters, and the volumes were published serially as parts
of a continuing whole. "Work-to-work development" therefore means something
structurally different for her than for Woolf, whose novels are discrete
projects. This is worth stating rather than smoothing over — it may in fact
be the more interesting comparison, since it sets Woolf's discrete
re-inventions against Richardson's continuous one.

**2. Ford's tetralogy stays incomplete.** *A Man Could Stand Up—* (1926)
and *Last Post* (1928) are not on Gutenberg. Ford's arc runs 1915 → 1925
with the war novels' resolution missing.

**3. Faulkner and Joyce cannot be extended this way.** Joyce is effectively
maxed out — *Finnegans Wake* (1939) is not US public domain, and *Stephen
Hero* was published posthumously in 1944. Faulkner has only *The Sound and
the Fury* on Gutenberg; *Soldiers' Pay* (1926), *Mosquitoes* (1927) and
*Sartoris* (1929) are US public domain now but would need another source.
Faulkner therefore remains a 2-work, 1929–30 snapshot, which limits what
can be said about the "genre pole" developing over time.

## Still open

- **James is trivially extendable** (1875–1904, all on Gutenberg) if the
  control should have a curve rather than a point. His "major phase"
  development is a well-known story the model could be asked to confirm —
  a useful ruler-calibration check, since it is a development claim with a
  known answer.
- **Sinclair changes what Section 1 can do.** She is now the only author
  in the corpus whose career crosses the coinage of the term the chapter is
  about, which makes her available as evidence in the theory section rather
  than only in the results. Worth deciding with your coauthor whether she
  belongs in the Section 2 corpus description as a fifth SoC author or is
  introduced separately in Section 1.
- **Faulkner remains the weak point.** Two works, one year, and the genre
  pole of the argument rests on him. If Section 2.4's author comparison
  leans on Faulkner as the high-spread case, that claim has no
  developmental dimension at all.

## Running the scoring (launched 2026-09-29 15:06)

The expansion needs three readers, because every work it will be compared
against carries three (gpt-5.4, qwen3-max, llama). Measuring the new works'
cross-reader spread over a smaller ensemble than the baseline would break
the very comparison this expansion exists to enable. That splits into two
arms, which append to the same `scored/<name>.jsonl` and are resumable by
`(passage_id, reader_id, run)`, so order and overlap do not matter.

**Cloud arm — running.** `score_expansion.sh` on della-vis2, gpt-5.4 +
qwen3-max over all 16 new works, ~46k rows. It runs
`verify_corpus.py --online` first and aborts if the corpus does not verify,
so a wrong-book download cannot reach the scorer.

```bash
cd technique_v_genre && nohup ./score_expansion.sh > logs/expansion_cloud.log 2>&1 &
```

**Llama arm — blocked on weights, then one command.** This must be a SLURM
job, not another client: `serve_score_model.slurm` runs the vLLM server and
the scoring client inside the *same* job by design, and both its header and
`fill_qwen_gaps.slurm` say never to split those across separately queued
jobs. The script now takes a `FILES` env var so it can target the 16 new
works instead of its hardcoded original 21; unset, it behaves exactly as
before.

The blocker: the HF cache was cleared some time after the August run, so
`meta-llama/Llama-3.3-70B-Instruct` is gone and compute nodes are offline
(`HF_HUB_OFFLINE=1`). The token still has access to the gated repo, so it
is a re-download (~140GB, 30 shards), running now. Once it lands:

```bash
cd technique_v_genre && ./submit_llama_expansion.sh
```

which refuses to submit unless all 30 shards are present — checked before
the job queues rather than discovered 30 minutes into a GPU allocation.

## Checking the corpus at any time

```bash
cd technique_v_genre
python verify_corpus.py --online        # all three layers
python -m unittest test_corpus_integrity -v
python verify_corpus.py --bless         # pin hashes after adding texts
```
