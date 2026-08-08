# Technique vs. genre: mapping where Humphrey's typology holds and where it doesn't

## Why this document exists

The v3 "ambiguity" line of work (`AMBIGUITY.md`, the CHR 2027 draft, the GRPO
training run) treated cross-model *disagreement* itself as the object of
study — the hope was that a model trained to represent disagreement would be
a more honest instrument than one trained to resolve it. Two things have
since made that framing hard to keep using as the project's spine:

1. **The training signal never converged.** v2 (GRPO) regresses against v1
   (SFT) on every binary-detection metric (`sft_vgrpo-v3.md`), and type
   classification F1 is in the low single digits for both (`model_card.md`:
   weighted F1 3.10% → 3.78%, 16-class). Two training stages have not
   produced a model whose disagreement is trustworthy enough to build an
   argument on.
2. **The one clean qualitative result didn't generalize the way the
   corpus-level statistics predicted it should.** `woolf_richardson_
   comparison.md` found a real, resampling-confirmed effect for Woolf
   (habitual aspect + elaborate syntax pulls toward
   `omniscient_description`) — but when the same design was run on
   Richardson, her aspect-stratified passages came out *more* stable across
   models than Woolf's, even though `AMBIGUITY.md`'s corpus-wide numbers say
   Richardson has the *lowest* cross-model agreement of any author (7.7–8.1%
   for Pointed Roofs and Backwater). Both findings are individually solid;
   together they mean "cross-model disagreement rate" is not, by itself, a
   stable proxy for "the typology doesn't fit this author." Something else
   is driving Richardson's corpus-wide instability, and the aspect-sample
   didn't happen to select for it.

That second point is the real reason to pivot rather than just keep scaling
the same design. Model disagreement conflates two very different things: (a)
*genuine boundary tension inside Humphrey's system* — is this passage
`indirect_interior_monologue` or `omniscient_description`, a real question
with a real answer on either side — and (b) *disagreement because the
passage doesn't belong to any of Humphrey's categories in the first place*,
which is a different claim entirely and the one the genre side of the
debate is actually making. AMBIGUITY.md's pipeline never separated these.
This document proposes a framing that does.

## The reframing

Stop trying to prove the typology right or the genre argument right. Both
are correct descriptions of *something* — the question worth asking of this
corpus is **which passages, sections, and works each one is the correct
description of**, and whether that sorts by author in a way a literary
critic would find meaningful (e.g., "Woolf is better read through
Humphrey's technique typology; Faulkner is better read as an unbounded
genre performance").

This is not a new pipeline. It is a new way of reading data the project
already has, plus one deliberate extension (Faulkner) and one new scoring
step the existing pipeline never computed.

### Two poles, not one axis of disagreement

- **Technique evidence (Humphrey's side).** A passage where a specific,
  named type in the taxonomy fits cleanly: multiple independent readers
  (models, and ideally human readers) converge on the *same* type, for the
  *same* textual reasons, and the passage doesn't need a residual category.
  The existing `clean-consensus` track in `consensus.yaml` /
  `build_ambiguity_dataset.py` is most of this already — it just needs to be
  read as *evidence for the typology* rather than as "the easy training
  data."

- **Genre evidence (the unbounded-form side).** A passage, section, or
  whole work that is unambiguously SoC by any reasonable reading — nobody
  would call it conventional narration — but that resists assignment to any
  single Humphrey type even under repeated, careful, multi-model scrutiny.
  This is *not* the same as "models disagree between type A and type B."
  The operational signature is different: `other_soc` used and *defended*
  (not just used as a default), `hybrid` where the hybridity itself is the
  point rather than a compromise between two clean readings, notes fields
  that describe the passage as exceeding the schema, and — the sharpest
  version — passages where confidence stays low and the type assignment
  keeps *changing* even under resampling of a single stable-seeming model,
  because there is no stable attractor to find.

- **Boundary tension (neither, or both — its own category, not evidence for
  genre).** This is what most of `AMBIGUITY.md`'s "systematic disagreement"
  actually is: two models each stably and defensibly pick *different named
  types*. That's still typology working — it just means the taxonomy's
  boundary between two adjacent categories is drawn in a contestable place.
  It's interesting (it's most of what `confounds_report.md` and
  `woolf_richardson_comparison.md` already document, e.g. the
  indirect/omniscient boundary), but it argues for *refining* Humphrey, not
  for abandoning him. Don't let it get counted as genre evidence — that was
  the conflation in the v3 framing.

So every passage needs a three-way sort, not the current two-way
consensus/disagreement split: **clean type fit**, **contested boundary
between named types**, **no type fits**. Only the first and third are
directly load-bearing for the technique-vs-genre argument; the second is
worth keeping as a secondary, Humphrey-internal finding (it's already good
material — see the indirect/omniscient discussion below).

### Why Faulkner is the deliberate extension

Every close-reading test so far has been Woolf vs. Richardson — two authors
whose SoC is famously the "quiet," grammatically-intact end of the
spectrum, which is probably why the typology fits them reasonably well most
of the time and why the interesting cases are boundary tensions rather than
total category failure. Faulkner is already in `positives/` and
`chunking_positives/` (*As I Lay Dying*, *The Sound and the Fury*) and
already shows the lowest binary-support numbers in the v2 model card's
per-class table and mid-pack agreement in `AMBIGUITY.md` (11.4%, 14.5% —
not the *lowest*, notably, which is itself worth investigating rather than
assuming). He's the natural test of the genre hypothesis: long unpunctuated
interior monologue, italicized time-shifts, a narrator (Benjy, Quentin) whose
own cognitive organization is part of what's being represented, not just his
content. If the genre argument is right anywhere in this corpus, Faulkner's
sections should show *category failure* (type keeps changing, `other_soc`
gets defended, hybrid explanations describe irreducibility) rather than
*boundary tension* (stable disagreement between two named types). If instead
Faulkner sorts the same way Richardson's aspect-sample did — locally stable,
just landing on different types than Woolf — that's an important negative
result and revises the hypothesis rather than confirming it.

Joyce is worth a lighter-touch pass for the same reason at the other
extreme: *Ulysses* generally, and especially the Penelope episode this whole
project is named after, is the text most commonly cited by both sides of
the debate — Humphrey uses it as his central example, and it's also the
text most often invoked as exceeding any typology. It's a useful third data
point precisely because expert opinion is already split on it going in.

## Operationalizing the three-way sort

This needs one new computed field per passage that doesn't exist yet in
`silver_data/` or `ambiguity_index.jsonl`: a **fit category**, derived from
the same resampled multi-model data `build_ambiguity_dataset.py` already
consumes, but classified against the three-way scheme instead of the
current two-way (`typological-boundary` / `detection-boundary` /
`clean-consensus` / `noisy`) scheme. Concretely:

| Existing `ambiguity_index.jsonl` type | Reframed as |
|---|---|
| `clean-consensus` | **Technique evidence** |
| `typological-boundary` (models split between exactly 2 named, stable types) | **Boundary tension** (Humphrey-internal, secondary finding) |
| `noisy` (3+ distinct readings, or resampling shows no stable attractor even within a single model) | **Genre evidence candidate** — needs the qualitative check below before counting |
| `detection-boundary` (some models say not-SoC at all) | Discard for this analysis — different question (is it SoC at all, not what type of SoC) |

The existing index gets most of the way there for free once it's built for
Faulkner (it currently isn't — resampling priority order in `README.md` is
"Richardson > Faulkner > Woolf > Joyce > Sinclair," so Faulkner coverage
should already be closer to complete than Woolf's). What's missing is the
**qualitative confirmation step** that `confounds_report.md` already
established as necessary: a `noisy` label from the index is a *candidate*
for genre evidence, not proof of it, until someone (or a targeted resample)
checks whether the instability is (a) a real absence of a stable attractor —
genre evidence — or (b) simple model unreliability (mercury-2's known
general noisiness, already flagged and controlled for in
`woolf_richardson_comparison.md`) that happens to land in the `noisy`
bucket by accident. The existing 0.8-consistency-threshold resampling
design is the right tool for this; it just needs to be pointed at a
different question than it was built for.

### A per-work "fit score" — the one genuinely new artifact

Once passages are sorted three ways, the useful summary statistic per work
(and per author) is a **fit ratio**: technique-evidence passages ÷
(technique-evidence + genre-evidence passages), with boundary-tension
passages reported separately rather than folded into either side. This is
the number that lets the eventual argument say "Woolf's ratio is X, and
Faulkner's is Y" rather than staying anecdotal. No script currently computes
it — it's a straightforward extension of the type-distribution tables
already in `AMBIGUITY.md` (which report raw type percentages per work but
never compute `other_soc`/`hybrid` share or cross-reference against the
ambiguity index), plus the qualitative-confirmation adjustment above.

## What to keep, what to shelve

**Keep, unchanged:**
- The taxonomy itself (`scripts/prompts.py`) — the whole point is to test
  it against the corpus, not redesign it prematurely.
- The resampling infrastructure (`scripts/resample.py`,
  `build_ambiguity_dataset.py`, `consensus.yaml`) — same data, different
  question asked of it.
- The close-reading methodology from `confounds_report.md` and
  `woolf_richardson_comparison.md` — isolated passages, sent independently
  of chunk context, to 3 frontier models, 3 runs each, checked for stability
  under resampling before anything gets called a finding. This is exactly
  the right rigor bar for genre-evidence candidates and should be applied
  to Faulkner the same way.
- `AMBIGUITY.md`'s per-work agreement and type-distribution tables — reusable
  as a first-pass signal, with the caveat (now proven, not hypothetical)
  that raw agreement rate alone is not a reliable stand-in for genre-fit and
  needs the qualitative check.

**Shelve, not delete:**
- The v3 GRPO / "teach the model to hold two readings" training objective.
  It's not needed to make this argument — the argument can be made directly
  from the frontier-model ensemble plus close reading, which is already the
  actual evidentiary instrument in every finding that has held up so far.
  Training a deployable classifier is a plausible *downstream* product once
  the fit categories stabilize (a "technique-fit detector" is a well-posed
  ML task in a way "detect ambiguity" turned out not to be), but it
  shouldn't be the thing blocking progress on the literary argument.
- `AMBIGUITY.md`'s framing of disagreement-as-signal itself — supersede
  rather than continue, since the Richardson puzzle shows it's not doing
  the sorting work it was supposed to do. Its data and infra survive; its
  central claim doesn't need defending anymore.

**Revise:**
- `ANDY.md`'s steering doc currently scopes the project to *Mrs Dalloway*
  and *To the Lighthouse* only ("surface examples... that problematize
  existing models and typologies"). That framing is one-sided — it only
  looks for genre evidence, never technique evidence, and only in Woolf. It
  should broaden to the two-sided, cross-author comparison described here.
- `chapter_intro.md` is actually already close to this framing at the
  conceptual level (it cites Long & So's "no one can agree on what \[SoC\]
  is, yet everyone seems to know it when they see it," and Bowler &
  Drewery's "definitive definition... fraught with contradictions") — but
  its own results section describes only the failure mode ("what applying a
  typology to an entire text produces is extreme interpretive ambiguity"),
  which is the same one-sided outcome as the current AMBIGUITY.md line. It
  should be rewritten to report the two-sided finding once the fit scores
  exist: which passages/authors validate the typology, which don't, and
  why that split falls where it does.
- The CHR 2027 lightning talk draft (`CHR2027_lightning_talk.md`) is built
  entirely on the disagreement-as-signal framing and the (not yet
  submitted, deadline Aug 14 2026) abstract will need a rewrite once this
  reframing is worked through — worth flagging now rather than after a
  draft is finalized around the old framing.

## Concrete next steps, in order

1. **Confirm Faulkner's resampling coverage** in `silver_data/resampled/`
   (README says it's second priority after Richardson) and run
   `build_ambiguity_dataset.py` if not already current.
2. **Write the fit-score script.** Extend the per-work tables in
   `AMBIGUITY.md` / `soc_type_analysis.ipynb` with `other_soc`/`hybrid`
   share and a fit ratio per the three-way sort above, for every author
   currently in `silver_data/` (Woolf, Richardson, Joyce, Faulkner,
   Sinclair). This is pure analysis over existing data — no new API calls.
3. **Qualitatively confirm the `noisy` bucket for Faulkner**, using the
   same design as `confounds_report.md`: pick the passages the fit-score
   step flags as genre-evidence candidates, isolate them from chunk
   context, send to gpt-5.4 / mercury-2 / qwen3-max at k=3 each, and check
   whether the instability is a real absence of an attractor or ordinary
   model noise. This produces the Faulkner analogue of
   `woolf_richardson_comparison.md` — call it `faulkner_typology_test.py`
   / `faulkner_comparison.md` following the existing naming pattern.
4. **Revisit the boundary-tension findings already in hand** (the
   indirect/omniscient boundary, the habitual+elaborate→omniscient pull) as
   the secondary, Humphrey-internal thread — these remain good scholarship
   regardless of the genre question and don't need to be redone.
5. **Update `ANDY.md` and `chapter_intro.md`** once step 2–3 produce
   numbers, so the steering doc and the chapter draft argue the two-sided
   claim instead of the one-sided "typology fails" claim currently in
   `chapter_intro.md`'s results paragraph.
6. **Decide the CHR 2027 abstract's fate** — rewrite around the two-sided
   finding (a stronger, more defensible claim than "we trained a model to
   disagree") or hold it for a later venue if step 2–4 aren't done by the
   August 14 deadline.

## Open question worth naming, not solving yet

If Faulkner turns out to sort more like Richardson's aspect-sample (locally
stable, just landing on unexpected types) than like a genuine genre-evidence
case, that's a real possibility worth sitting with rather than engineering
away: it would suggest Humphrey's typology, suitably extended, may cover
more of the modernist SoC canon than the genre argument gives it credit for,
and that the genre argument's real evidence lies somewhere this corpus
hasn't sampled yet (later, more radically fragmented texts; non-English
modernism; post-1960s stream-of-consciousness heirs). That would still be a
publishable, interesting result — it just wouldn't be the clean two-sided
map the working hypothesis predicts. Worth stating in whatever gets written
up either way.
