# Stream-of-consciousness experiments — handoff for Claude Code on Della

You are picking this up in VSCode / Claude Code, connected to an authenticated
Della session where the full data and the project's scoring model live. This
document is self-contained: it carries the framework, the corpus, the scoring
schema, and three experiments with runnable steps and falsifiable predictions.
Nothing here depends on the sandbox that produced it.

Project root on Della: `/scratch/gpfs/MM4/apjanco/penelope`
Host login node needs Duo + VPN, which is why this moved off the automated
sandbox and onto your interactive session.

---

## 1. What this project is

The `penelope` project reads modernist novels with an ensemble of models and
asks, per passage, **how consciousness is represented** — then uses the pattern
of those judgments to adjudicate a century-old literary-theory debate:

- **Technique view** (Bowling 1950, Humphrey 1954): stream of consciousness is
  a nameable set of devices with a finite taxonomy. If it's a technique, every
  passage should fit one type cleanly.
- **Genre/mode view** (the SoC-as-unbounded-form counter-tradition): it's an
  open mode that exceeds any taxonomy. If it's a genre, some passages resist
  every single type.

A completed literature review (`soc_literature_review.md`, in the project
artifacts) established the key finding: the field **re-drew this binary as a
graded cline** decades ago (Cohn 1978, Fludernik 1993). So the productive
target is not a label but a **position on a mediation cline**, and the
disagreement *between readers* is the signal — not raw disagreement rate, which
an earlier run (`AMBIGUITY.md` / the Woolf-Richardson comparison) proved
conflates three different things (cline-adjacency, detection uncertainty, model
noise).

### The one methodological shift that drives everything

The earlier model card scored a **flat 16-way categorical** `soc_type` and got
~3–4% weighted F1. That number means the *target is mis-specified*, not that
training failed. The fix (detailed in `soc_skill_update_proposal.md`): replace
the flat label with **three orthogonal outputs**, of which the ordinal
mediation level is primary.

**The mediation cline (0–3), the primary measurement:**

| level | name | gloss |
|---|---|---|
| 0 | psycho-narration | narrator reports the mind ("she felt uneasy") |
| 1 | narrated monologue / FID | character's thought bends the narrator's sentence, no quotes |
| 2 | quoted monologue | framed direct interior speech ("I am uneasy, she thought") |
| 3 | autonomous monologue | unframed raw interior voice (Molly Bloom) |

Two readers one level apart are near-agreement; 0-vs-3 is real instability.
Ordinal spread distinguishes them; categorical agreement cannot. **This is the
fix for the Woolf-Richardson artifact.**

---

## 2. The corpus

`positives.zip` (seen in the sandbox) is 23 full **novel** texts (~2.13M words),
1902–1941. These are whole novels, not pre-segmented passages — segmentation is
step 1 of the pipeline below. **There is more data on Della you should inventory
first** (a negatives set is implied by the `positives` name; the strategy doc
also references material this subset doesn't sample). Start by listing
`/scratch/gpfs/MM4/apjanco/penelope`.

Per-author composition of the positives subset (drop the duplicate
`pg05670_jacobs_room.txt`; it repeats `jacobs_room.txt`):

| author | works | words | expected cline role |
|---|---|---|---|
| Woolf | 11 | 925k | core; early→late arc |
| Joyce | 3 | 427k | *Dubliners*→*Portrait*→*Ulysses*, wide by development |
| James | 3 | 362k | **control** — figural, not canonical SoC (narrator-dominated, levels 0–1) |
| Richardson | 3 | 173k | technique, sharp FID peak |
| Faulkner | 2 | 161k | genre end, wide + heavy tail |
| Ford | 1 | 77k | **control** — impressionist boundary case |

James and Ford are the built-in controls: they anchor the narrator-mediated
floor so Joyce's climb toward level 3 is measured against something. Full
manifest with dates and word counts: `corpus_manifest.json` (project artifacts).

---

## 3. Scoring schema (per passage)

Emit this JSON per passage (drop-in from `soc_skill_update_proposal.md`). The
three fields that carry the experiments are `mediation_level`, `fit_category`,
and `fit_alternatives`:

```json
{
  "passage": "Exact verbatim quote.",
  "is_soc": true,
  "mediation_level": 1,
  "fit_category": "boundary_tension",
  "fit_alternatives": ["indirect_interior_monologue", "omniscient_description"],
  "antimimetic": false,
  "antimimetic_defense": "",
  "soc_type": "indirect_interior_monologue",
  "secondary_devices": ["free_association", "imagery"],
  "narrator_position": "minimal",
  "character_pov": "Clarissa Dalloway",
  "confidence": "medium"
}
```

- `fit_category` ∈ {technique_fit, boundary_tension, no_type_fits} — your
  three-way sort, promoted to an explicit field.
- `fit_alternatives` names the two adjacent types in contention. This is what
  makes boundary tension **measurable**: at the corpus level you can check
  whether reader disagreement is one-step-on-the-cline (tension) or scattered
  (a genre candidate).
- `antimimetic` = true only with a non-empty `antimimetic_defense`. This is the
  genre pole given a real unnatural-narratology criterion, not just "unstable
  under resampling."

---

## 4. The pipeline — one substrate, three slices

All three experiments are the **same scored table sliced three ways**. Build the
table once:

```
segment novels into passages
        │
        ▼
score each passage with the ensemble  →  one long table:
   (text, author, year, passage_id, reader_id, mediation_level,
    fit_category, fit_alternatives, antimimetic, ...)
        │
   ┌────┼─────────────────────┐
   ▼    ▼                     ▼
by author   pooled (all)      by year
= fingerprints   = geometry   = moving frontier
```

### Step 0 — inventory Della

`ls -R /scratch/gpfs/MM4/apjanco/penelope | head -100`. Find: the full corpus
(positives + negatives + anything the subset didn't include), the ensemble /
GRPO-trained scoring code, and any existing scored output so you don't recompute.

### Step 1 — segment (decision required)

Whole novels → passages. **Segment size is the single most consequential
upstream choice** and every downstream result depends on it. Options, in order
of preference for this task:

1. **Paragraph-based windows** (recommended start): split on blank lines,
   then merge/split to a target of ~80–200 words so a "passage" is roughly one
   unit of narration. Record char offsets so you can trace any passage back.
2. **Fixed sliding window** (200 words, 50% overlap): more uniform for the
   diachronic curve, but cuts across rhetorical units.
3. **Scene/section breaks**: truest to the text, hardest to automate uniformly.

Emit a passages table: `passage_id, text, author, year, char_start, char_end,
n_words, passage_text`. Keep it as parquet.

### Step 2 — score with the ensemble

Two reader options — **run both if you can, they answer different questions**:

- **The project's GRPO-trained model on Della** (the reason to be on the
  cluster). Score every passage; this is the "trained specialist reader."
- **A frontier ensemble** (several models each scoring every passage). This
  gives the *multi-reader distribution* the experiments actually need — the
  spread across readers is the convergence signal. The lit review argues this
  ensemble is arguably the better instrument here than the single trained model,
  because the finding is about cross-reader convergence, not one classifier's
  verdict.

For each (passage × reader) store the full schema JSON. This is the master
table. **Checkpoint it** — it's the expensive artifact everything else reads.

### Step 3 — the three slices

Once the master table exists, all three experiments are cheap groupbys + plots.

---

## 5. The three experiments (with falsifiable predictions)

### Experiment A — Per-author fingerprints

**Question:** does "technique author" vs "genre author" show up as the *shape*
of each author's mediation-level distribution?

**Do:** for each author, histogram `mediation_level` over all their passages
(pool readers, or show per-reader bands). Compute the spread (variance) and
modality of each distribution.

**Predict (falsifiable):**
- Technique authors (Richardson, Woolf) → **single sharp peak**, low spread
  (~0.5), concentrated at level 1 (FID).
- Genre author (Faulkner) → **flat/multimodal**, high spread (>1.2), heavy tail
  at level 3.
- Controls (James, Ford) → mass at levels 0–1, confirming the ruler's floor.
- Joyce → **wide, but for a different reason than Faulkner**: split his panel by
  book and it should resolve into three narrower peaks marching rightward
  (*Dubliners*→*Ulysses*), a developmental signature, not a within-book genre
  performance. If Joyce's *per-book* spread is as wide as Faulkner's, the
  fingerprint claim is in trouble.

**Kills the hypothesis if:** spreads don't separate technique from genre
authors, or the controls don't sit at the floor. Schematic of the expected
shapes: `author_distribution_schematic.png` (project artifacts) — *illustrative,
replace with measured values*.

### Experiment B — Geometry of the cline

**Question:** are Humphrey's types discrete natural kinds (Humphrey) or a smooth
continuum (Cohn/Fludernik)?

**Do:** pool **all** passages across authors into one mediation-level density.
Also inspect the continuous version if the scorer emits a soft score, not just
the 0–3 bin. Test for multimodality (dip test / GMM with model selection).

**Predict:**
- Discrete attractor basins at 0/1/2/3 with thin valleys between → Humphrey's
  types are natural kinds.
- Smooth unimodal or plateaued smear → Cohn/Fludernik continuum wins.
- Either result is publishable. **Bonus:** the low-density valleys are exactly
  the boundary-tension zones, so this map also tells you *where* Humphrey's
  taxonomy is under-specified — cross-check that those regions are where
  `fit_category == "boundary_tension"` concentrates.

### Experiment C — The moving frontier (flagship)

**Question:** are "technique evidence" (clean fit) and "genre evidence" (no type
fits) fixed properties of passages, or positions on a *moving frontier of
reading familiarity* that shifts as techniques conventionalize (Fludernik 2012)?

**Two arms:**
- **Diachronic (data you have):** order passages by publication year (1902–1941)
  and plot the `no_type_fits` rate and the reader-disagreement rate over time.
  Does the frontier recede (techniques conventionalize, no-type-fits declines)
  or hold constant (each generation invents new unnatural forms)?
- **Experimental (needs the manipulable-reader setup):** take today's
  `antimimetic` / `no_type_fits` passages and re-score them with **increasing
  few-shot exposure** to canonical SoC examples in context. Watch whether they
  migrate from `no_type_fits` → `technique_fit` as the reader's exposure grows.

**Predict:** if the same passage migrates from genre to technique both in the
historical record and under the exposure manipulation, the debate is reframed
from "what is SoC" to "how does a reading community domesticate the unreadable"
— and made measurable.

**The confound is the asset:** the readers are models trained on the critical
tradition, so their agreement may reflect *canonized consensus* rather than
fresh reading. Control for it — but note that the passages where they still
disagree are, by definition, the parts of the tradition **not yet
conventionalized**, which is precisely Fludernik's frontier measured at the
level of the reading community. Controlling for the confound *is* Experiment C.

---

## 6. The validation gate (do this before believing any of A–C)

**Experiment D — convergence tracks surface features (clears the Da bar).**

Nan Da's critique (2019) is the methodological bar: a computational literary
result must correspond to something interpretable, not model idiosyncrasy.

**Do:** correlate `mediation_level` (and cross-reader convergence on it) with
surface features the linguistic tradition already identified: reporting-verb
density, first/third-person pronoun ratio, tense distribution, punctuation
density (Dahl 1970; Gast 2023 has a ready *Ulysses* register yardstick).

**Predict:** if convergence tracks these measurable features, the signal is
interpretable and real. If it doesn't, it's model noise and A–C are built on
sand. **Run D first on a sample** before scaling the full scoring.

---

## 7. Suggested order of work

1. Inventory Della; locate full corpus + scoring code + any existing scores.
2. Segment a **sample** (one Woolf, one Faulkner, one James) and eyeball the
   passages — is the segment size producing sane units?
3. Score that sample with both readers; run **Experiment D** on it. If
   convergence tracks surface features, proceed; if not, fix scoring first.
4. Scale to the full corpus; checkpoint the master table.
5. Run A, B, C off the master table (cheap once it exists).
6. Write up: A and B are bankable standalone results; C is the flagship.

## 8. Reference artifacts (in the project, pull them in Claude Code)

- `soc_literature_review.md` — the referee-grade review, verified DOIs, the
  technique-vs-genre assessment.
- `soc_sources.csv` — 26 sources annotated by camp, relevance, DOI.
- `soc_skill_update_proposal.md` — the flat-label→three-output fix and the full
  signal-extraction chain, with the drop-in schema.
- `soc_research_targets.md` — why these experiments (the three sources of
  leverage: uniform coverage, multi-reader distributions, manipulable readers).
- `corpus_manifest.json` — the 23-text manifest with authors/years/word counts.
- `author_distribution_schematic.png` — expected fingerprint shapes
  (illustrative).
