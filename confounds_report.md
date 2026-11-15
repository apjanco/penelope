# Identifying typology-confounding passages

This is a companion to `AMBIGUITY.md`, and it approaches the same question
from the opposite direction. AMBIGUITY.md finds boundary cases
statistically, from the rate at which different frontier models disagree on
a passage's type across `silver_data/`. This document finds them
qualitatively, by close-reading a single strong model's structured output
and its own reasoning trace against the taxonomy's stated definitions in
`scripts/prompts.py`. The two methods should agree with each other where
they overlap, and in the one sample run so far, they do. Every claim below
is grounded in a specific chunk from `training_data/teacher.json.jsonl`,
quoted directly, alongside the model's own label and explanation for it, so
the confound can be checked against the primary text rather than taken on
description.

The first instinct, when asked to find passages the model found confusing,
is to filter the data mechanically: pull every instance marked
`confidence: low`, every `hybrid` or `other_soc` label, every non-empty
`notes` field, and call that the confound set. In the first sample batch —
56 confirmed instances across five texts — this yields almost nothing.
Fifty-two of the fifty-six were marked `confidence: high`, four `medium`,
none `low`. There were no `hybrid` or `other_soc` labels at all, and only
one instance out of fifty-six had anything written in `notes`. The model is
fluent and decisive even in places where, on inspection, its call is
genuinely contestable. Its own uncertainty signals turn out not to be a
reliable proxy for where the taxonomy is under strain. Confounds mostly
have to be found by reading the passage and the model's `explanation`
against the taxonomy's definitions directly, not by filtering the schema's
confidence fields for self-reported doubt.

Reading through the first batch this way, six recurring patterns emerged as
the actual criteria worth using.

**Definitional self-contradiction.** The taxonomy defines
`direct_interior_monologue` as having no narrator framing — no "he
thought." In chunk `pg02413-madame-bovary_003`, the model labels the
following passage `direct_interior_monologue`, `narrator_position:
minimal`, confidence high:

> "So it is for this," she said to herself, "that his face beams when he
> goes to see her, and that he puts on his new waistcoat at the risk of
> spoiling it with the rain. Ah! that woman! That woman!" And she detested
> her instinctively.

Its explanation reads: *"The quoted sections present unmediated, fragmented
thoughts with immediate emotional outbursts ('Ah! that woman! That
woman!', 'Get out!'), fitting direct interior monologue. It is not
indirect_interior_monologue because the use of quotation marks and raw,
ungrammatical exclamations signals a direct transcription of her mental
voice rather than narrator-blended free indirect discourse."* The passage
opens with the words "she said to herself" — explicit narratorial framing —
and the model's own `narrator_position` field already concedes "minimal"
rather than "absent." The label doesn't follow the rule its own definition
states, and the field the model filled in next to the label already shows
it noticed the framing without letting that observation change the label.

**Field-internal inconsistency.** The same chunk produces a second,
related case a few lines earlier. The passage —

> As to Charles, he did not stop to ask himself why it was a pleasure to
> him to go to the Bertaux. Had he done so, he would, no doubt, have
> attributed his zeal to the importance of the case, or perhaps to the
> money he hoped to make by it.

— is labeled `indirect_interior_monologue`, `narrator_position: present`,
with the explanation *"The third-person syntax merges seamlessly with
Charles's subjective rhythm and rhetorical self-questioning, hallmarks of
free indirect discourse."* But `indirect_interior_monologue` is defined in
`scripts/prompts.py` as requiring the narrator to "hover invisibly." A
narrator marked "present" is, by the taxonomy's own vocabulary, not
invisible. The same pairing recurs independently in a different author: in
`pg03019-pointed-roofs_002`, the passage

> The sense of that sudden little embrace recurred often to Miriam during
> the course of the first day. It was unlike any contact she had known--more
> motherly than her mother's.

is again `indirect_interior_monologue`, again `narrator_position: present`,
with the model noting *"It could be mistaken for omniscient_description,
but the phrasing remains tightly anchored to her subjective realization
rather than external narrator analysis."* The model names the correct
neighboring category and argues past it, but the `narrator_position` field
it fills in for both instances sits on the wrong side of its own type
definition. AMBIGUITY.md independently identifies indirect/omniscient as
"the most contested typological boundary" in the corpus, from cross-model
disagreement rates; these two instances are the same boundary, visible
inside one model's own structured fields.

Naming a pattern like this doesn't yet say what kind of problem it is.
There's a real difference between a model simply getting something wrong —
an execution failure a better pass would fix — and a case where the
category system itself doesn't have a meaningful answer to give, so that
any answer a reader lands on will look, from another angle, like a mistake.
The two look identical from a single label sitting next to a passage. The
distinguishing test is stability under repetition and independence: a
model's own execution error tends to be noisy — inconsistent even with
itself, or correctable once the contradiction is pointed out — while a
genuine gap in the rubric produces disagreement that is *stable*, in the
sense AMBIGUITY.md already uses the word: multiple independent readers each
give a different answer, and each of them keeps giving that same answer
when asked again. That's not one reader being careless. That's several
careful readers running into the same wall from different directions.

So the Bertaux and embrace passages were put to that test directly, rather
than left as a suspicion. Both were sent, isolated from their surrounding
chunk context, to three frontier models with no connection to the teacher
pipeline — gpt-5.4, mercury-2, qwen3-max — three times each. Every model
was perfectly consistent with itself, and no two models agreed with each
other. gpt-5.4 called the Bertaux passage `omniscient_description` on all
three runs, every time: *"The passage renders Charles's attraction through
an authoritative narrator who interprets motives he himself does not fully
examine. It moves close to his consciousness via rhetorical questioning and
accumulative sensory [detail]."* mercury-2 returned `{"instances": []}` —
no stream of consciousness at all — on all three runs. qwen3-max called it
`indirect_interior_monologue`, `narrator_position: minimal`, on all three
runs: *"The passage conveys Charles's inner experience through third-person
syntax that closely mirrors his thoughts and feelings without direct
quotation or overt narration of mental states."* The teacher's own reading
— `indirect_interior_monologue`, `narrator_position: present` — is a
*fourth* stable position. The embrace passage from Pointed Roofs reproduced
the identical three-way split, independently, down to the wording of each
model's reasoning. Four models, four internally perfect, mutually
incompatible readings, on two unrelated passages by two different authors.
That is the stable-disagreement signature, not noise.

The teacher model was also resampled against itself, using its actual
chunk-level methodology rather than an isolated passage, to check whether
its own reading is a reliable position or a coin flip. This is a weaker
test than it should have been — the resampling job ran out of its time
allocation partway through and only completed for the Madame Bovary chunk,
three runs at temperature 0.2 (its production setting) and three at 0.8;
Pointed Roofs got one run before the job was cut off, not enough to say
anything about it. Within the six completed Bovary runs, the Bertaux
passage (or a sub-span of it) was the model's own choice of leading
instance in three of them. Its `soc_type` was `indirect_interior_monologue`
in all three — a stable attractor even under resampling. Its
`narrator_position`, though, split two "present" to one "minimal" across
those same three runs — the exact field that disagreed with the taxonomy's
own definition in the first place turns out not to be fully stable even
within one model's repeated judgment of itself. That's consistent with,
though weaker evidence than, the cross-model result: the label the model
converges on is closer to a real position than an error, but the field
meant to justify or disqualify that label is the part still wobbling.

Put together, this passage — and by extension, this pattern of "present"
narrator paired with "indirect" framing — is genuine category-2 evidence:
not a model getting the taxonomy's rule wrong, but the taxonomy's rule
not having a determinate answer for what free indirect discourse actually
does, which the field of narrative theory has described as a fusion of
narrator and character voice for a hundred years, and which a category
built on "hovers invisibly" cannot represent without forcing a binary
choice onto something that isn't binary. This is a different verdict from
the "she said to herself" case earlier in this same chunk (the definitional
self-contradiction pattern): that one is a model failing to apply a rule
the rubric states clearly enough to apply. This one is the rubric not
having a rule to apply in the first place. The two look the same on the
page — a label sitting next to a passage — and only the resampling test
tells them apart.

(Supporting data: `cross_model_confound_test.py` and its output
`cross_model_confound_results.jsonl`, run from `della-vis2` against
gpt-5.4/mercury-2/qwen3-max via the project's existing
`scripts/config.py`/`models.yaml`; `resample_teacher_chunks.py` and its
output `resample_teacher_results.jsonl`, run against the teacher's own
`teacher_relabel()` on a GPU node. Both are one-off scripts, not yet
promoted to `scripts/`, and the teacher-resampling script has a known flaw
worth fixing before reusing it: it selects the "leading" instance per run
by passage length rather than by matching a specific passage across runs,
which is why three of its six Bovary runs picked a different passage in
the same chunk entirely — a second finding in its own right, since it
means the model's segmentation of a chunk into distinct instances isn't
fully stable either, a separate question from what any one instance gets
labeled.)

**Genre and period mismatch.** In `pg00600-notes-from-underground_004`,
the opening of Dostoevsky's novel —

> I am a sick man.... I am a spiteful man. I am an unattractive man. I
> believe my liver is diseased. However, I know nothing at all about my
> disease, and do not know for certain what ails me.

— is labeled `direct_interior_monologue`, `narrator_position: absent`,
explained as *"First-person unmediated declaration of psychological and
physical state without narrative framing. Rules out
indirect_interior_monologue because there is no third-person syntax or
narrator hovering; the voice is entirely the character's own."* Two
sentences later, in the same voice, the text reads: "No, I refuse to
consult a doctor from spite. That you probably will not understand." The
model labels this second passage `direct_interior_monologue` as well, on
the same grounds. But the "you" being addressed there is explicit in the
text itself — this is formally a written confession addressed to a reader,
not thought caught unawares in the moment, and the taxonomy's
direct-monologue category was built to describe a technique that didn't
exist yet in 1864. Both instances satisfy every surface marker the
definition lists — first person, fragmentary, no explicit "he thought" —
while doing something structurally different, and the model's explanation
in both cases stops at the surface markers.

**Uniform labeling against an explicit instruction.** The system prompt
instructs the model to differentiate within a chunk rather than apply one
label throughout it. Chunk `pg00600-notes-from-underground_004` produces
six confirmed instances, and five of them share `narrator_position:
absent` with `orthographic_marker` as the secondary device; four of those
five are `direct_interior_monologue`. The one exception is instructive: a
passage reading

> To live longer than forty years is bad manners, is vulgar, immoral. Who
> does live beyond forty? Answer that, sincerely and honestly I will tell
> you who do: fools and worthless fellows.

is labeled `free_association` instead, with the model reasoning that *"the
breathless numerical escalation followed by a sudden physiological
interruption mimics the character's frantic, unfiltered mental pacing and
logic-defying leaps."* That is a real distinction, but it raises the
question the instruction is meant to force: is this span actually a
different technique from the rest of the chunk, or a continuation of the
same rhetorical performance, addressed to the same "you," that simply
happens to escalate numerically? The model differentiated once in six
instances and gave a real reason for it; whether that one differentiation
is doing real work or just responding to surface rhythm is exactly the
kind of thing worth checking by hand against the source when it recurs at
scale.

**Taxonomy coverage gaps.** In `pg03019-pointed-roofs_003`, the passage

> She flushed and fumbled--thought of dressing-tables and the little
> objects of which she had made so many hanging to the mirror by ribbons;
> "toilet-tidies" haunted her--but that was not it--she smoothed out her
> work as if to show it to Fräulein--"Na, na," came the delicate caustic
> voice. "Was wird das wohl sein?" Then she remembered.

is labeled `simulation_state_of_mind`, secondary device `orthographic_marker`,
explained as rendering *"acute social anxiety through fragmented syntax,
bodily reaction, and interrupted cognitive retrieval rather than coherent
propositional thought."* The `evidence` field the model supplies quotes
only the English portion around the interruption. The German dialogue
breaking into Miriam's anxious interiority is the most notable feature of
the passage, and there is no field in the schema for code-switching as a
technique, so the model's own evidence for its label silently steps around
the thing that makes the passage distinctive.

**A defensible close call, which is not a failure.** Not every boundary
case is a problem to fix. In `to-the-lighthouse-cleaned_004`, the passage

> And, what was even more exciting, she felt, too, as she saw Mr Ramsay
> bearing down and retreating, and Mrs Ramsay sitting with James in the
> window and the cloud moving and the tree bending, how life, from being
> made up of little separate incidents which one lived one by one, became
> curled and whole like a wave which bore one up with it and threw one
> down with it, there, with a dash on the beach.

is labeled `imagery` rather than `indirect_interior_monologue`, with the
reasoning that *"the cognitive event is entirely carried by the extended
wave metaphor; there is no expository gloss explaining the feeling, only
the image itself functioning as thought. It rules out
indirect_interior_monologue because the passage lacks propositional
reasoning and instead relies on sensory-kinetic imagery to convey the
mental shift."* A reader could reasonably call this indirect interior
monologue that happens to be imagery-heavy rather than imagery that happens
to function as thought; the model names the real alternative and argues a
specific, checkable reason for its choice instead of asserting it. That is
the taxonomy's genuine edge, correctly navigated, and it is exactly the
kind of calibration data AMBIGUITY.md's cross-model resampling is meant to
surface at scale — recording it here from a single model's reasoning is a
second, independent line of evidence that the boundary is real rather than
an artifact of cross-model noise, not a defect to correct.

## From cataloguing confounds to testing a hypothesis about them

Everything above is a catalogue: examples found, sorted into six ways they
went wrong, with the Bertaux/embrace pair upgraded from suspicion to
confirmed category-2 evidence by resampling. A catalogue like that has a
ceiling on how useful it can be. It tells you something is happening
without telling you why, or what kind of literary claim it supports beyond
"the taxonomy is imperfect here," which is not, on its own, a finding
anyone needs a corpus to demonstrate. The six criteria differ in a way the
catalogue format doesn't surface: some of them are about a real boundary
being genuinely graded (the narrator-position pattern), and some are about
a dimension the taxonomy simply doesn't have (code-switching, confessional
address). Those call for different responses — accepting gradience versus
adding a dimension — and telling them apart from a pile of examples, after
the fact, isn't reliable. The alternative is to state a hypothesis about
*why* a given kind of ambiguity happens, in terms of a textual feature that
can be checked against passages nobody has classified yet, and see whether
the pattern holds.

The clearest candidate hypothesis came out of noticing that both confirmed
category-2 passages — Flaubert's Bertaux passage, Richardson's embrace
passage — describe a habitual or recurring state rather than a single
moment: "he liked going into the courtyard... he liked the granary... he
liked old Rouault," "the sense of that sudden little embrace recurred often
... during the course of the first day." The hypothesis: free indirect
discourse rendering a habitual disposition is harder to place than free
indirect discourse rendering a punctual, single-moment thought, because
"the narrator hovers invisibly" presupposes something happening in a
moment, and generalizing about what a character is habitually like is
already most of the way to the kind of authoritative character-summary the
taxonomy calls `omniscient_description`. If that's right, it should show
up as a *directional* effect — habitual passages pulled specifically toward
`omniscient_description`, not just toward disagreement in general — and it
should hold across different authors, since it's a claim about grammar and
mode, not about any one writer's style.

To test it without circularity, four passages of each kind were hand-picked
from Woolf — two from To the Lighthouse, two from Mrs Dalloway — chosen by
grammatical aspect markers in the raw prose (habitual: "for having lived...
one feels," "it would come over her," "that was his fate, his gift";
punctual: "she thought," a single dated exchange) before any model saw any
of them, and run through the same cross-model test as the Bertaux/embrace
passages: gpt-5.4, mercury-2, qwen3-max, three times each, in isolation
from chunk context.

The habitual group showed disagreement on all four passages, no exceptions.
Two — both from To the Lighthouse — pulled gpt-5.4 and qwen3-max to the
same place independently: *"the sense of that sudden little embrace..."*
sits in Dalloway, but the Lighthouse pair (James Ramsay's "great clan"
passage, and Mr Ramsay's "it was his power, his gift") drew, from both
models, on their first try: gpt-5.4 — *"The passage renders Mr. Ramsay's
mental habit and existential stance through an authoritative narrator
rather than raw unmediated thought."* qwen3-max, independently, on the same
passage: *"The narrator articulates [Mr Ramsay's psychological condition]
with... an authoritative, interpretive narrator rather than through
unmediated thought."* Two models neither of which is our teacher, neither
of which saw the other's answer, converging on the same reasoning in
almost the same words. Every habitual passage got at least one
`omniscient_description` or `hybrid` verdict from at least one model; only
two of four punctual passages did. Within a single model's own three
repeated runs on the same passage, answers were unstable 33% of the time
(4 of 12 model-passage pairs) for the habitual group against 17% (2 of 12)
for the punctual group.

The punctual group also produced the cleanest possible negative case: *"There!
Out it boomed. First a warning, musical; then the hour, irrevocable. The
leaden circles dissolved in the air. Such fools we are, she thought,
crossing Victoria Street"* — got `indirect_interior_monologue` from all
three models, all three runs, no exceptions, and all three explanations
independently cite the same textual anchor, the explicit "she thought"
marking a single located moment: gpt-5.4 — *"The narrator recedes and the
sentence conveys her private judgment..."* mercury-2 — *"reports a thought
with the attribution 'she thought'... merging narrator and character voice
in a free-indirect style."* qwen3-max — *"blends third-person narration
with Clarissa's interior reflection ('Such fools we are'), rendered in free
indirect discourse."* Total agreement, on a punctual passage, is exactly
what the hypothesis predicts and exactly what none of the habitual passages
produced.

One selection turned out to be flawed, and it's worth reporting rather than
dropping quietly, because the failure is informative. *"Had there been an
axe handy, a poker, or any weapon that would have gashed a hole in his
father's breast and killed him, there and then, James would have seized
it"* was picked as a punctual moment — a specific, present impulse — but
its grammar is counterfactual ("had there been... would have"), not simple
past, and every model's explanation noticed that instead of noticing the
single-moment aspect I'd selected for: gpt-5.4 called it `reverie_fantasy`
on one run — *"renders James's inward violent wish through a hypothetical
scenario... a fantasy of patricidal desire"* — and `indirect_interior_
monologue` on another, unstable with itself; qwen3-max called it
`omniscient_description` — *"the narrator articulates James's violent
impulse with psychological authority... not James's [own voice]."* This
passage has its own confounding grammatical feature (hypothetical mood)
independent of the one I was testing for, and it produced disagreement for
what looks like a different reason. That's a real limitation of a four-per-
group sample: one badly chosen passage is a quarter of the group, and
aspect isn't the only grammatical feature that can pull a passage away from
a stable reading. Excluding it, the punctual group's instability rate drops
to 1 of 9 (11%), against the habitual group's 33% — a sharper contrast, but
also a smaller and more curated sample than an honest result should rest
on. And mercury-2 specifically was the least self-consistent model across
both groups (unstable on 4 of 8 passages total, including one punctual
one), which raises the question of how much of the habitual-group
instability is aspect and how much is just mercury-2 being a noisier
annotator in general, evenly distributed rather than concentrated by
aspect — the current sample is too small to fully separate those two
explanations from each other.

With those caveats, the directional pattern held on both texts, for both
models that were internally reliable enough to trust (gpt-5.4, qwen3-max),
and the convergent reasoning quoted above is closer to two independent
critics agreeing on *why* a passage resists the category than to either
noise or a coincidence. That's a stronger and more specific claim than
"models disagree here": free indirect discourse describing what a
character habitually is or does sits closer to the omniscient end of the
taxonomy's spectrum than free indirect discourse describing what a
character thinks in one moment, because habitual description is already a
form of authorial generalization, which is what `omniscient_description`
is defined by. That's a claim a narrative theorist could evaluate, agree
with, refine, or reject on its own terms — which a raw disagreement count
never was.

## Scaling the aspect test — one signal held, one didn't

The sample was extended from two Woolf novels to five — adding Jacob's
Room, The Voyage Out, and Between the Acts, one habitual and one punctual
passage from each, hand-selected the same way and with more care taken
this time to exclude conditional/hypothetical mood after the axe-passage
lesson — bringing the total to seven passages per group. The honest result
is that scaling confirmed one part of the original finding and mostly
erased the other, which is worth reporting exactly as it happened rather
than folded quietly into a bigger success.

The part that got stronger is the directional one. Every habitual passage
that pulled at least one reliable model toward `omniscient_description` or
`hybrid` — six of seven, 86% — against two of seven, 29%, for punctual
passages, and that holds even restricted to the two models that were
self-consistent enough to trust (gpt-5.4, qwen3-max), excluding mercury-2
entirely. Both new habitual passages from Jacob's Room and The Voyage Out
reproduced the pull independently of the original Lighthouse pair,
including one case — `voyageout_iterative_1` — where qwen3-max itself
wasn't even consistent with its own three runs, splitting
`indirect_interior_monologue` once and `omniscient_description` twice, which
is instability *in the direction the hypothesis predicts* rather than
against it.

The part that mostly didn't survive is the cruder claim: that habitual
passages are simply noisier overall. In the original four-per-group sample
that looked like a real gap — 33% within-model instability for habitual
against 17%, or 11% excluding the flawed passage. At seven per group the
gap nearly closed: 33% against 29% overall, 29% against 21% restricted to
the two reliable models. That's a small enough difference, at this sample
size, to not lean on. Worth noting separately: mercury-2's own instability
turned out to be evenly split across both groups (three unstable passages
of seven in each), which means its noise isn't tracking aspect at all — it's
a generally unreliable annotator, not one that specifically struggles with
habitual description, and excluding it from the instability count (rather
than from the omniscient-pull count, where it wasn't doing much work
either way) is the right call rather than a convenient one.

So the finding that survives scaling, precisely stated, is narrower than
first reported: habitual/iterative free indirect discourse gets pulled
toward `omniscient_description` specifically, at a rate that held up and
even sharpened across five authors' worth of Woolf's own range (early,
middle, and late), while the broader claim that habitual passages produce
more disagreement in general does not have enough support to make yet. The
directional version is also the more useful one for the reason the whole
exercise was undertaken — it's a claim with actual content ("habitual
description reads as authorial generalization") rather than a claim about
noise, and it's the kind of thing a narrative theorist can agree or
disagree with on its merits.

(Supporting data: `cross_model_aspect_test.py` and its output
`cross_model_aspect_results.jsonl`, same methodology and models throughout,
now 14 hand-selected passages across five novels — Mrs Dalloway, To the
Lighthouse, Jacob's Room, The Voyage Out, Between the Acts — split evenly
between habitual and punctual aspect.)

One correction belongs here too, since it affects how much weight to put on
any of this. Two of the seven texts in the first sample turned out to be
misidentified by filename before I ever read their actual content.
`chunking_positives/pg32220_lieutenant_gustl.txt` is not Schnitzler's
Lieutenant Gustl; its opening line — "Whoever has been at Friedrichshafen
on beautiful Lake Constance... To this region the story of little Bissula
leads us" — identifies it as Felix Dahn's Bissula, a nineteenth-century
historical novel told in ordinary third-person narration, and the
pipeline's verdict of "no stream of consciousness" on its first five
chunks was correct, not a miss. `chunking_positives/pg34219_mary_olivier_a_life.txt`
is not Sinclair's Mary Olivier; its opening pages ("THE ENCHANTED CASTLE...
FOR CHILDREN... The Treasure Seekers, The Would-be-Goods") identify it as
E. Nesbit's The Enchanted Castle, a children's fantasy novel, and the same
thing happened. A genuine Sinclair dataset does exist elsewhere in the
project, per AMBIGUITY.md's own author table, so this looks like a
filename or download error specific to these two files rather than
evidence the real texts are missing. It's worth auditing the rest of
`chunking_positives/` before trusting another title there at face value; I
haven't done that yet.

The ten examples selected for the first round of review were drawn from the
first five chunks each of the five correctly-identified texts — Ulysses, To
the Lighthouse, Notes from Underground, Madame Bovary, and Pointed Roofs —
run through the fixed extraction pipeline, with the full reasoning trace
retained in `raw_reasoning` alongside the structured label for each
instance. All ten are quoted above or were shown in full in the
conversation that produced this file; this document is meant to hold the
method and its textual grounding together, not to summarize the dataset at
a distance from it.

A few questions are worth carrying into the next batch. Scaling from four to
seven passages per group already did useful work: it separated a real,
strengthening signal (the directional pull toward `omniscient_description`)
from one that mostly turned out to be small-sample noise (the raw
instability-rate gap), and confirmed mercury-2's instability isn't
aspect-specific, so it can be set aside rather than agonized over. The
natural next step is more of the same kind of scaling rather than a
different design: fifteen to twenty passages per group would give the
directional claim a real effect-size estimate instead of a proportion out
of seven, and a pair of authors outside Woolf — ideally one with a very
different sentence style, to check the pull toward `omniscient_description`
isn't somehow a fact about her syntax specifically rather than about
habitual aspect — would matter more now than adding more Woolf. `mercury-2`
is worth dropping from future runs of this specific test, not because it's
uninformative in general but because two runs' worth of evidence now show
its noise doesn't track the thing being measured. A control for raw length
and syntactic complexity, independent of aspect, is still missing and still
worth adding before trusting the effect size further.

Two things from the earlier catalogue are still open in the same way the
aspect hypothesis was before this test: the "she said to herself"
definitional-contradiction pattern and the Dostoevsky genre-mismatch
pattern haven't been put through resampling, and shouldn't be assumed to be
category-2 findings just because they were found by the same close-reading
method that first flagged the narrator-position pattern. The
self-contradiction case in particular looks more like model error on its
face — the rubric's rule is stated clearly enough to apply, and the model
just didn't apply it — which predicts it should resolve under resampling
rather than reproduce. That's a specific, checkable prediction, not a
guess, and it's the natural next test now that the aspect hypothesis has a
result to compare it against. The genre-mismatch case needs a different
kind of test entirely, closer to the aspect design than to plain
resampling: pick passages that vary systematically in proximity to the
technique's canonical period (a stratified sample from, say, Flaubert,
Dostoevsky, and Richardson, roughly matched for what they're describing)
and check whether disagreement rate tracks distance from 1910s-20s
high modernism the way it tracked habitual-versus-punctual aspect here.

The teacher-resampling script (`resample_teacher_chunks.py`) still needs
its instance-selection fixed — matching a specific passage across runs by
token overlap, the way `resample.py`/`consensus.py` already do elsewhere in
this project, rather than picking whichever instance happens to be
longest — and rerun with concurrency so a single chunk's resampling doesn't
time out before finishing, which is what happened to the Pointed Roofs half
of the first resampling run. And whether Pointed Roofs' density in the
first sample (thirty-five confirmed instances across five chunks, against
single digits everywhere else) is about technique saturation or partly
about chunk length is still an open question from before this test, worth
checking against AMBIGUITY.md's Richardson disagreement rates once there's
bandwidth to return to it.
