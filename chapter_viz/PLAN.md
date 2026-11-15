# Chapter figures — plan

Four figures. Figure 1 carries the argument; 2 and 3 support it; 4 is optional.

## What the data will and will not support

Three constraints come out of the scoring work and they shape every figure.

**The scale is ordinal and uncalibrated between readers.** On the same 13,348
passages, gpt-5.4 puts 52% of judgements at level 0 and qwen3-max puts 31%. A
mean pooled across readers is an average over instruments, not a measurement.
Every figure therefore names its reader, and Figure 1 shows all three.

**Spread is statistic-dependent, and one author is reader-dependent.** Woolf
leads on *range* under all three readers (2.22 / 2.11 / 1.70). Faulkner's SD
swings 4x by reader (0.17 gpt-5.4, 0.66 qwen3-max) on only 3 works. A chart
built on one reader and one summary statistic would say "Faulkner is the most
variable author," which the other two readers contradict. This is why Figure 1
plots every work as a mark and faceted by reader rather than reducing to a bar
per author.

**n varies 3 to 10.** Range is an extreme-value statistic, so n must be visible
in the figure, not just the caption. (Checked: range does not actually track n
here, r = -0.07 — but the reader cannot know that from the chart.)

**llama is excluded from all figures.** It failed the calibration gate — 90.7%
of its judgements are level 1, entropy 0.93 bits — so its per-work means sit at
1.00 regardless of the text. It appears only in Figure 3's reliability note as a
documented failure.

---

## Figure 1 — The fingerprint is a shape, not a level  *(flagship)*

**Claim:** Woolf's distinctiveness is the *distribution* of her work-level
mediation, not its height. Eight novels cluster; *The Waves* departs utterly.
Richardson writes nine volumes of one project and never leaves a 0.30 band.

**Form:** horizontal strip plot (one row per author, one mark per work), in
**small multiples — three panels, one per reader**. Authors share a row order
across panels, sorted by median mediation under qwen3-max.

Why this form:
- Every work is a mark, so the reader sees Woolf's *cluster-plus-outlier* shape
  against Faulkner's even spread. A bar of "range" or "SD" hides exactly that,
  and picks a different winner depending on which statistic you choose.
- Faceting by reader turns the uncalibrated-scale problem into the figure's
  second message: the ordering survives the instrument. Faulkner's instability
  becomes visible rather than suppressed.

**Color: emphasis, not categorical.** Woolf in accent slot 1; the other six
authors in de-emphasis gray. Seven categorical hues would fail the all-pairs CVD
gate (the method caps all-pairs forms at three) and would bury the one row that
is the point — the "eight hues when the story is one number" anti-pattern.

**Annotations:** direct-label *The Waves*, *Night and Day* and *The Years* in
the Woolf row (the span that makes the claim); IQR bracket behind each author
row so cluster-vs-spread is readable without arithmetic; `n=` beside each author
name.

**Honest caption must state:** reader per panel, ordinal scale, n per author,
and that *The Waves* is the highest-mediation work in the corpus — above
*Ulysses* — which complicates the received "Woolf is quieter than Joyce."

---

## Figure 2 — Career trajectories  *(diachronic)*

**Claim:** the arcs differ in kind. Richardson is flat across 12 years; Woolf is
flat then spikes; Joyce climbs monotonically; James is flat across 29 years.

**Form:** small multiples, one panel per author, publication year on x, work-level
mediation on y, shared axes, works connected in date order. 7 panels, one series
each (so no categorical-hue problem). Single hue, Woolf's panel accented.

Shared y-axis is essential — independent axes would make every author look
equally variable, which is the opposite of the finding.

**Caveat to encode:** Joyce ends at 1922 because *Finnegans Wake* is in copyright
until 2035. Mark the panel so his arc is not read as complete.

---

## Figure 3 — Why the ordinal scale, not the categories  *(methodological)*

**Claim:** Humphrey's typology does not fail for lack of signal; it fails at one
seam. Re-expressing the same judgements on Cohn's cline raises reliability from
fair to moderate.

**Form:** NOT a chart for the headline numbers — two kappas are a **stat pair**
(κ 0.386 categorical → 0.514 ordinal), per the method's "handful of headline
numbers → stat tiles, not a grouped bar."

Beside them, one ranked horizontal bar chart: sources of reader disagreement,
sequential single hue, showing `indirect_interior_monologue` vs
`omniscient_description` at **17.3%** towering over the rest (next is 2.9%).
That single bar is the argument — the typology's weak point is exactly Cohn's
L0/L1 transition.

Add the supporting figure: **95.5% of ordinal disagreements are one level apart**
— as a stat tile, not a pie.

---

## Figure 4 — Ambiguity, measured directly  *(optional)*

**Claim:** ambiguity is a property of passages, not a failure of readers.

**Form:** histogram of Jev's top-two mediation probability gap across 21,162
passages, single hue sequential, with the 0.10 threshold marked — **12.4%** of
passages are genuinely undecidable.

This is the only figure measuring ambiguity directly rather than inferring it
from reader disagreement, which `AMBIGUITY.md` warns conflates cline-adjacency,
detection uncertainty and model noise. Worth including if Section 1's
"lean into ambiguity" framing is load-bearing.

---

## Build notes

- Author the figures as one self-contained HTML page per figure (hover tooltips
  by default; a table view behind each for accessibility), plus static SVG/PDF
  export for the print chapter.
- Light and dark both explicitly stepped, not an automatic flip.
- Run `validate_palette.js` on the accent+gray pair before shipping; do not
  eyeball CVD safety.
- All figures read `data/work_mediation.csv` and `data/author_spread.csv` so the
  numbers cannot drift between figures.

## Data status at time of writing (2026-10-01)

- qwen3-max and Jev: complete, all 46 works
- gpt-5.4: 12 works still short (gapfill running) — Figure 1's gpt panel will
  be missing Faulkner's *Soldiers' Pay*, both new Ford volumes and 3 Sinclair
  works until it finishes. Rebuild after.
