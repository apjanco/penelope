# "Time Passes": where the typology holds, and the two passages where it doesn't

Follow-up to the Lighthouse+Dalloway scoring run, prompted by the single
`antimimetic=true` hit that surfaced in the full k=3 pass (out of 5,001
valid reader-runs) — on `to_the_lighthouse_cleaned_0274`, the first passage
of "Time Passes," Part Two of *To the Lighthouse*. That section — the
house standing empty for ten years, Mrs Ramsay's death disposed of in a
bracketed aside, almost no human characters present — is already a
recognized crux in Woolf criticism for exactly this kind of narratorial
strangeness, so it was worth a properly-powered look rather than resting
on one low-probability sample. All 33 passages of the section (source
offsets 230393–263117 in `to_the_lighthouse_cleaned.txt`) were rescored at
k=10 per reader (gpt-5.4, qwen3-max — mercury-2 excluded, per this
project's established practice), using AMBIGUITY.md's own 0.8-consistency
bar for calling a reader "stable" on a passage, which is only meaningful at
this k (unsatisfiable at k=3 except by unanimity).

## The section as a whole: more mediated, not less

Going in, the naive expectation was that "Time Passes" would show up as
*less* mediated than the rest of the novel — that its famous strangeness
would look like a push toward the unmediated end of Cohn's cline. It's the
opposite.

| | mean mediation_level | technique_fit | boundary_tension | noisy/other |
|---|---|---|---|---|
| Lighthouse + Dalloway overall (k=3) | 0.80 / 0.90 | 63.2% | 30.9% | 5.9% |
| Time Passes only (k=10) | **0.41** | 45.5% | 21.2% | 33.3%* |

*mostly `detection_boundary` (9.1%, one reader finds no SoC at all — the
purely descriptive shadow/light passages like `_0280`) and `noisy`
(15.2%), not genre-pole evidence; see below.

Of the 22 stable, `is_soc=true` passages, all but a handful land as
`omniscient_description` at mediation level 0 — psycho-narration, the
*most* narratorially mediated point on the cline, not the least. The
section's strangeness isn't that Woolf's narrator recedes here. It's that
the narrator's normal move — interpreting a mind from outside, "she felt,"
"it seemed as if" — keeps firing on content that has no mind to interpret:
a sleeper nobody names ("should any sleeper, fancying that he might find on
the beach an answer to his doubts..."), "the minds of men" as a collective
abstraction, Mrs McNab's habitual drudgery narrated as if it were
philosophical meditation. Both readers, independently, keep naming the same
strain: gpt-5.4's `alternative_reasoning` on passage `_0278` calls the
runner-up "other_soc, because this is not neatly a standard individual
mind-rendering scene... the hypothetical 'any sleeper' universalizes
consciousness." qwen3-max reads the same passage as a clean `technique_fit`
— it's willing to let "omniscient_description" cover a generalized,
characterless meditation without treating that as a strain at all. That
disagreement — not about *which* type fits, but about *whether* the
passage needed a human mind to fit any type — turns out to be the section's
real fault line, and it's exactly where the two antimimetic passages sit.

## The two passages: personified, non-human interiority

Across all 658 valid runs, `antimimetic=true` appears exactly 7 times, and
only from gpt-5.4, on exactly two passages — both built around the same
device: mental predicates assigned to something that isn't a person.

### `to_the_lighthouse_cleaned_0274`

> "...Only through the rusty hinges and swollen sea-moistened woodwork
> certain airs, detached from the body of the wind (the house was
> ramshackle after all), crept round corners and ventured indoors. Almost
> one might imagine them, as they entered the drawing-room, questioning
> and wondering, toying with the flap of hanging wallpaper, asking, would
> it hang much longer, when would it fall?... asking: Were they allies?
> Were they enemies? How long would they endure?"

gpt-5.4's own ten runs split, almost exactly down the middle, between two
*stable* positions — not scattered across many labels, which would read as
noise, but two coherent readings each independently reproduced:

- **5 of 10 runs**: `simulation_state_of_mind`, `fit_category:
  boundary_tension`, `antimimetic: false` — the passage still gets a
  Humphrey label, with the taxonomy under some strain but holding.
- **5 of 10 runs**: `soc_type: other_soc`, `fit_category: no_type_fits`,
  `antimimetic: true`. Representative defense: *"The passage attributes
  quasi-conscious questioning to disembodied 'airs' 'detached from the
  body of the wind,' creating an interiority that is not securely
  locatable in any human character and is instead distributed across a
  personified atmospheric force. Because the syntax renders thought-like
  inquiry ('Were they allies? Were they enemies?') without a plausible
  single consciousness to own it, the scene resists the taxonomy's
  assumption that rendered consciousness belongs to a character mind."*

qwen3-max, on the same 10 passages, is not split at all: `omniscient_
description` / `technique_fit` in 9 of 10 runs, `boundary_tension` in the
other. It never once returns `antimimetic`. Its own reasoning explicitly
treats the personification as ordinary narratorial imagery rather than
something the taxonomy can't hold: the airs are, for this reader,
authorially-described figures for atmosphere, fully coverable by
"omniscient description," the same category it uses for Mrs Ramsay's or
Mr Ramsay's very human interiority elsewhere in the novel.

### `to_the_lighthouse_cleaned_0281`

> "...among the shrouded jugs and sheeted chairs even the prying of the
> wind, and the soft nose of the clammy sea airs, rubbing, snuffling,
> iterating, and reiterating their questions — 'Will you fade? Will you
> perish?' — scarcely disturbed the peace... Nothing it seemed could break
> that image, corrupt that innocence, or disturb the swaying mantle of
> silence which, week after week, in the empty room, wove into itself the
> falling cries of birds, ships hooting, the drone and hum of the
> fields... and folded them round the house in silence."

Same device, extended further: not just the wind but the room's own
*silence* is given active, continuous, self-directed mental verbs — it
"weaves," "folds," maintains an "indifference" and "air of pure integrity"
against an implied question. gpt-5.4: 6 of 10 runs hold at `imagery` /
`boundary_tension`; 2 of 10 escalate to `other_soc` / `no_type_fits` /
`antimimetic: true` ("the room/house a quasi-conscious, self-maintaining
inwardness... without anchoring that experience in any human or
personified mind"); 2 of 10 return no SoC instance at all. qwen3-max again:
`omniscient_description` / `technique_fit` in 9 of 10 runs, no strain
registered.

## What this is evidence of

Both passages fail the stability bar for a confirmed antimimetic finding —
neither is a queue of unanimous, or even majority, verdicts. That's the
honest first read. But the AMBIGUITY.md methodology this project already
uses treats a *stable split within one model* as a different, and still
meaningful, category from noise: gpt-5.4's 5/10 on passage `_0274` is not
a model flailing across many answers, it's a model landing on exactly two
coherent, independently-reproduced positions and alternating between them
— which is closer to the signature of a genuinely contested case than to
randomness. Combined with the cross-reader split (gpt-5.4 sometimes sees
the taxonomy break; qwen3-max never does, on the identical text), the
finding is not "this passage is antimimetic" but something more precise
and more defensible: **this is the one device in two full novels of Woolf
where whether the typology holds depends on the reader's threshold for
naturalizing personified, non-human interiority** — which is close to
exactly the question Alber (2010) and Fludernik (2012) argue about in the
abstract, operationalized here on a specific, quotable pair of sentences.
Fludernik's position — that "unnatural" features get conventionalized over
time, so that a reader with a stronger prior for figurative psycho-
narration (arguably qwen3-max's behavior here) simply naturalizes what a
stricter reader (gpt-5.4, some of the time) does not — is a live
description of what these two models are actually doing to the same text,
not just a theoretical caution.

For the Woolf chapter, this sharpens rather than complicates the headline
result. Woolf's SoC is overwhelmingly technique-legible — 94% of Lighthouse
and Dalloway sort cleanly into `technique_fit` or argued `boundary_tension`
between named Humphrey categories, and even "Time Passes," the section
built to be strange, resolves mostly to heavily-mediated `omniscient_
description` rather than genre-pole breakdown. The place the typology
actually strains to its edge is narrow and specific: personified,
non-human interiority with no character to own it, appearing in exactly
two sentences' worth of the entire sample. That is a far more precise and
defensible claim than "Woolf sometimes exceeds the typology" — it names
the device, the location, and the exact condition (which reader, how
often) under which the exception shows up.

(Supporting data: `scored/time_passes.jsonl` — 660 rows, k=10, both
readers; `scored/time_passes_report.md` — full classification and all
technique_fit/boundary_tension examples; passage offsets traceable via
`passages/time_passes.jsonl`'s `char_start`/`char_end` against
`positives/to_the_lighthouse_cleaned.txt`.)
