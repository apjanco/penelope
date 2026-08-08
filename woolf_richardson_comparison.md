# Woolf and Richardson against the typology: a passage-by-passage comparison

This collects the full set of passages tested so far under the
habitual/punctual-aspect design first described in `confounds_report.md` —
seven habitual and seven punctual passages from five Woolf novels, five
habitual and five punctual from Richardson's three Pilgrimage volumes, plus
three further passages added afterward specifically to separate aspect
from sentence architecture once the first round made the two look
entangled — with, for each, what specific feature of the taxonomy it puts
pressure on and what three independent models (gpt-5.4, mercury-2,
qwen3-max; three runs each, passages sent in isolation from chunk context)
actually said about it. The point of listing all twenty-seven is that the
pattern only shows up in the set, not in any single passage — and the
pattern that showed up was not the one expected going in, first when Woolf
was compared to Richardson and again when the follow-up test was designed
specifically to overturn the first revision. Both rounds are kept in
rather than silently updated, because watching a hypothesis get corrected
twice in one document is more informative than presenting only the final,
tidier version.

## Woolf

### To the Lighthouse

**Habitual.** *"Since he belonged, even at the age of six, to that great
clan which cannot keep this feeling separate from that... James Ramsay,
sitting on the floor cutting out pictures... endowed the picture of a
refrigerator as his mother spoke with heavenly bliss."* Tests: whether
generalizing a child into a "clan" or type — narrated character-summary in
the service of a specific scene — reads as interiority or as authorial
placement. Result: `omniscient_description`, stable across all three runs,
both reliable models. gpt-5.4: *"an authoritative narrator... rather than
raw unmediated thought."*

**Habitual.** *"It was his power, his gift, suddenly to shed all
superfluities... that was his fate, his gift."* Tests the same boundary a
second time, on Mr Ramsay directly. Result: `omniscient_description`,
stable, both models, near-identical reasoning independently reached:
gpt-5.4 — *"renders Mr. Ramsay's mental habit and existential stance
through an authoritative narrator."* qwen3-max — *"an authoritative,
interpretive narrator... rather than through unmediated thought."*

**Punctual.** *"Had there been an axe handy... James would have seized
it."* Intended as a punctual impulse; turned out to test conditional/
hypothetical mood instead, which the models caught and I hadn't controlled
for. Result: unstable within gpt-5.4 itself (`reverie_fantasy` then
`indirect_interior_monologue`), `omniscient_description` from qwen3-max.
Kept in the record as the sample's one clear methodological miss, not
excised.

**Punctual.** *"A bit of a hypocrite? she repeated. Oh no — the most
sincere of men... he is absorbed in himself, he is tyrannical, he is
unjust."* Tests direct-vs-indirect at the "she thought"/quotation-marked
boundary. Result: `hybrid` (gpt-5.4) or `indirect_interior_monologue`
(mercury-2, qwen3-max) — real disagreement, but a different, milder kind:
all three explanations independently anchor on the same textual feature
(the fused third-person frame plus unmediated exclamation), disagreeing
about how to name the fusion rather than about what's happening in the
text.

### Mrs Dalloway

**Habitual.** *"For having lived in Westminster—how many years now? over
twenty,—one feels... before Big Ben strikes."* Tests habitual/generalized
feeling against the taxonomy's implicit assumption of a punctual thought
event. Result: `hybrid`/`indirect_interior_monologue`, unstable within
gpt-5.4 (hybrid twice, indirect once) — disagreement, but not the clean
omniscient-pull of the Lighthouse pair; qwen3-max: *"blending memory,
hypothetical reflection, and sensory-triggered recollection."*

**Habitual.** *"For they might be parted for hundreds of years, she and
Peter... they came back in the middle of St James's Park on a fine
morning—indeed they did."* Same test. Result: `hybrid` (gpt-5.4, stable
across all three runs) vs. `indirect_interior_monologue` (mercury-2,
qwen3-max) — disagreement in the same register as the previous passage,
not a shift to authorial reading.

**Punctual.** *"There! Out it boomed... Such fools we are, she thought,
crossing Victoria Street."* Tests the clean case: explicit "she thought,"
one bounded moment. Result: total agreement, all three models, all three
runs, `indirect_interior_monologue`, each independently citing the "she
thought" tag as the anchor. The cleanest data point in the whole set.

**Punctual.** *"Was Evelyn ill again?... Not the right hat for the early
morning, was that it?"* Same test, longer passage, specific conversational
exchange. Result: `indirect_interior_monologue` from gpt-5.4 and qwen3-max,
stable; mercury-2 unstable (`none` twice, `indirect_interior_monologue`
once) — the instability here is mercury-2's general unreliability, not a
real boundary tension.

### Jacob's Room

**Habitual.** *"For the landscape needed it. It was too pale... too pale as
usual... a favourite with his landladies' children... which they often
did."* Tests generalized professional/social standing (Steele, the
painter) against punctual-moment presumption. Result:
`omniscient_description`, both reliable models on first run, though
gpt-5.4 wavered to `indirect_interior_monologue` on a later run. gpt-5.4:
*"enters the painter's social and psychological situation through an
interpretive narrator... 'The critics would say' shades toward interior
anticipation, but the dominant [mode is authorial]."*

**Punctual.** *"No, but not in lodgings, thought Mrs Flanders. It's a great
experiment coming so far with young children..."* Clean "thought" tag
case, matching the Dalloway punctual control. Result:
`indirect_interior_monologue`, all three models, essentially stable
(mercury-2 wobbled once). Confirms the punctual/explicit-tag case reads
consistently regardless of author.

### The Voyage Out

**Habitual.** *"As for the mass of streets... though thirty of her forty
years had been spent in a street. She knew how to read the people who
were passing her; there were the rich... there were the bigoted
workers... there were the poor..."* Tests generalized, categorical
social perception (a taxonomy of *other people*, not just of the
character herself) against the punctual-moment assumption. Result:
`indirect_interior_monologue`, all three models, stable — no
omniscient pull despite being maximally generalizing in content. The
first real counter-example inside the Woolf set itself.

**Punctual.** *"But this lady looked neither up nor down... Yes, she knew
she must go back to all that, but at present she must weep."* Result: mild
disagreement (`simulation_state_of_mind` vs. `indirect_interior_monologue`),
mercury-2 unstable — ordinary noise, not a directional pattern.

### Between the Acts

**Habitual.** *"She had met him at a Bazaar; and at a tennis party... in
his ravaged face she always felt mystery... Now a third time... she felt
it again."* The most explicitly, grammatically habitual passage in the
whole Woolf set — it counts its own repetitions. Result:
`indirect_interior_monologue`, both reliable models, fully stable, no
omniscient pull at all. The strongest counter-example to the
Lighthouse-derived hypothesis.

**Punctual.** *"Isa raised her head. The words made two rings, perfect
rings... his snow-white breast was circled with a tangle of dirty
duckweed; and she too, in her webbed feet was entangled."* Tests
imagery-as-thought against propositional-thought categories. Result:
`imagery` (gpt-5.4) vs. `indirect_interior_monologue` (qwen3-max) — a real
but different boundary (imagery vs. indirect monologue, not indirect vs.
omniscient).

## Richardson

### Pointed Roofs

**Habitual.** *"Why had he come to tea every Sunday—never missing a
single Sunday—all the winter?... Eve said he looked perfectly
miserable."* The Richardson analogue of the Between the Acts passage —
explicit, counted repetition ("every Sunday," "never missing a single").
Result: `direct_interior_monologue`, both reliable models, fully stable.
gpt-5.4: *"a rapid sequence of unmediated rhetorical questions that render
a character's thought directly, without narrator framing."* Not
`indirect`, not `omniscient` — *direct*, the opposite end of the taxonomy
from where the Woolf hypothesis predicted habitual content would land.

**Habitual.** *"The sense of that sudden little embrace recurred often to
Miriam... during the course of the first day."* (This is the original
passage from the first resampling test.) Result: `indirect_interior_
monologue` (qwen3-max, stable) / `omniscient_description` (gpt-5.4,
stable) / `none` (mercury-2, stable) — the one Richardson passage that
does reproduce a real, stable three-way split, and the one most similar in
register to the Woolf passages (third-person "the sense... recurred," no
dashes or fragmentation).

**Punctual.** *"She drew back from the window as the bass chords began
thumping gently in the darkness. It was better that it should come now
than later on, at dinner-time."* Result: `indirect_interior_monologue`,
all three models, essentially stable. Ordinary punctual case, as expected.

**Punctual.** *"'Toilet-tidies' haunted her... 'Na, na,' came the delicate
caustic voice. 'Was wird das wohl sein?' Then she remembered."* Tests
embedded German dialogue against the taxonomy's lack of a code-switching
dimension (see `confounds_report.md`, criterion 5). Result:
`indirect_interior_monologue`/`simulation_state_of_mind` depending on run —
mild disagreement, `evidence` field still quotes only the English portion
in every run.

### Backwater

**Habitual.** *"That darkness was dreadful. It was his own life. She would
never know it. However well they got to know each other they would always
be strangers. Probably he never thought about her when he was alone."*
As explicitly and totalizingly generalizing as anything in the Woolf
set — a permanent, unconditional claim about the relationship's whole
future. Result: `indirect_interior_monologue`, both reliable models,
fully stable. gpt-5.4: *"third-person-compatible narrative language...
closely tracking private judgment."* No omniscient pull whatsoever,
despite content that generalizes further into the future and further
past the present moment than any Woolf passage tested.

**Punctual.** *"Miriam turned sharply to her mother... How had she heard
about the little classes? Had there been little classes? She could not
remember them."* Result: `indirect_interior_monologue`, stable, both
reliable models. Ordinary case.

### Honeycomb

**Habitual.** *"It meant never being able to agree with people about
books, never liking them for the same reasons as other people.... It
meant... knowing, absolutely, everything about the author."* Result:
`direct_interior_monologue`, both reliable models, fully stable. gpt-5.4:
*"unmediated, first-person-inflected interior style... moving
associatively rather than through orderly exposition."*

**Habitual.** *"He had never thought for a moment in his life of the world
as anything than what it appeared to be. He had no suspicion that anyone
ever did."* The one Richardson passage that clearly reproduces the
Woolf-style omniscient pull. Result: `omniscient_description`, mercury-2
and qwen3-max; `none` from gpt-5.4. qwen3-max: *"an authoritative,
interpretive narrator rather than through the character's own unfiltered
thoughts."* Notably, this is the only Richardson habitual passage in the
set that generalizes about *another* character rather than about Miriam's
own ongoing state — see the discussion below.

**Punctual.** *"It sounds like a housekeeper giving an order to a servant
she knows won't obey her, thought Miriam..."* Result: `indirect_interior_
monologue`, all three models, fully stable. Clean case.

**Punctual.** *"'This dress is a nightmare in this room,' she thought...
'Never mind, I mustn't think about it,' she added hurriedly."* Result:
`direct_interior_monologue` (gpt-5.4) / `indirect_interior_monologue`
(qwen3-max) — the quotation-marked "she thought" boundary again, same
tension as the Lighthouse "bit of a hypocrite" passage, independent of
author.

## What the comparison actually shows

Going in, the expectation — stated plainly in `confounds_report.md` — was
that habitual/iterative content would pull toward `omniscient_description`
regardless of author, because generalizing about what a character is
permanently like is already most of the way to authorial summary. Woolf
supported that, mostly: two of four Lighthouse-and-Jacob's-Room-style
passages showed a clean, independently-reproduced pull, though two others
(the Voyage Out and Between the Acts habitual passages) did not, which
should have been a warning sign even inside the Woolf set alone. Richardson
does not support it at all. Of her five habitual passages, two get
`direct_interior_monologue` with total model agreement — the *opposite*
end of the taxonomy from omniscient — two get `indirect_interior_monologue`
with no pull whatsoever despite content that generalizes as far or further
than anything in Woolf, and only one reproduces the Woolf pattern.
Meanwhile Richardson's punctual passages behave almost identically to her
habitual ones: overwhelmingly stable, overwhelmingly
`indirect_interior_monologue`, with instability at or below what her
habitual passages show. The aspect distinction, which organized the entire
Woolf-only sample into two visibly different populations, does almost no
organizing work in Richardson at all.

Reading the models' explanations across both authors together suggested a
first-pass diagnosis: every Richardson `direct_interior_monologue` verdict
cites fragmentation, dashes, rhetorical-question sequences, syntax that
"moves associatively rather than through orderly exposition"; every Woolf
`omniscient_description` verdict cites the opposite — long, subordinated,
essayistic construction, "since he belonged... since to such people...
James Ramsay, sitting on the floor" as one uninterrupted grammatical unit.
That looked like a clean substitution — swap sentence architecture in for
aspect as the real predictor — and it's a fair first correction to
`confounds_report.md`'s original hypothesis. But stated that way it's still
too simple, and a second, targeted round of testing shows why.

### Testing architecture on its own — the substitution doesn't hold either

If sentence architecture alone were doing the work, elaborate syntax should
pull toward `omniscient_description` regardless of what the content is
doing, aspect included. Three passages were chosen specifically to check
that, crossing architecture against aspect in ways the first two rounds
hadn't cleanly separated: a Woolf passage that is long and periodic but
renders one punctual moment, not a habitual truth (*"He had followed her
into the drawing-room, that young man they laughed at; he was standing by
the table, fidgeting with something, awkwardly, feeling himself out of
things, as she knew without looking round"* — Mrs Ramsay's real-time,
unspoken awareness of Charles Tansley); and, as the sharper test, two
passages from Richardson written in a mode she almost never uses — long,
accumulative, catalog-like sentences — one punctual (*"It meant the large
dark room, the dark invisible picture, the big pieces of strange dark
furniture in gloomy corners... the little dishes stuck about, sweets,
curiously crusted brown almonds, sheeny grey-green olives"* — what a laugh,
heard just now, evokes) and one explicitly, maximally habitual (*"They had
never known anything else. For years life had been for them just what it
was to-day--breakfast in bed, chirping at their mother... They would grow
up and be exactly like their parents"*).

The result rules out architecture as an independent predictor as cleanly
as it ruled out aspect. The Woolf elaborate-but-punctual passage got total
agreement on `indirect_interior_monologue` — no pull, despite the
long periodic construction that supposedly does the pulling. Richardson's
elaborate-but-punctual passage got `imagery` (gpt-5.4) and
`indirect_interior_monologue` (mercury-2, qwen3-max) — again no pull,
confirming that even when Richardson borrows Woolf's characteristic
sentence-building, punctual content resists the drift toward authorial
reading. But Richardson's elaborate *and* habitual passage — the one
crossing both features at once — got `omniscient_description` from
qwen3-max, stable across all three runs, with `narrator_position: dominant`,
the single strongest narrator-presence rating anywhere in this whole study,
stronger than any Woolf passage produced. gpt-5.4 and mercury-2 both called
this passage `none` rather than converging on the same reading, so it isn't
the clean two-model agreement the Lighthouse passages showed — but a model
with no stake in either author's reputation reading Richardson, of all
writers, as maximally omniscient the one time she writes a long
accumulative sentence about a permanent truth is about as strong a
confirmation as a three-passage check can give.

Put together as a table (27 passages now, crossing architecture against
aspect):

| | iterative (habitual) | punctual |
|---|---|---|
| **elaborate** | 7/8 (88%) | 1/6 (17%) |
| **fragmented** | 1/5 (20%) | 2/8 (25%) |

Only one cell is elevated. The other three — elaborate+punctual,
fragmented+habitual, fragmented+punctual — sit within a few points of each
other, all in the 17–25% range, indistinguishable from baseline
disagreement. Neither habitual content nor elaborate syntax predicts the
omniscient pull on its own; each one alone produces roughly the same low
rate as having neither. It's specifically the *conjunction* — generalizing
content delivered in long, periodic, subordinated syntax — that produces
the strong effect, and that conjunction is what Woolf's habitual passages
happened to be (which is why aspect looked causal on her alone) and almost
never is in Richardson (which is why architecture looked causal once a
second author was added, until architecture was tested on its own and
mostly failed too). The mechanism this actually supports is closer to the
original intuition in `confounds_report.md` than either single-factor
revision was: `omniscient_description` is defined by *interpretive
authority* — a narrator standing back and asserting something about a
character rather than rendering their immediate experience — and that
authority seems to require both an atemporal claim (something true beyond
the moment) and the syntactic room to make an assertion of it (a
sentence built to hold a generalization, not a fragment built to render a
single sensation). Take away either ingredient and the passage reads as
interior, regardless of author.

There is a second, weaker pattern worth flagging without overclaiming it: the
one Richardson passage that does show the omniscient pull ("He had never
thought for a moment in his life...") is the one habitual passage in her
set that generalizes about *another* character's fixed nature rather than
about Miriam's own continuous state. Three of Woolf's omniscient-pulled
passages are also other-directed (Ramsay described by the narrator,
Steele's professional insecurity), while the one Woolf habitual passage
that stayed cleanly `indirect_interior_monologue` with no pull at all — Isa's
recurring feeling about Haines in Between the Acts — is self-directed, like
most of Richardson's non-pulled set. Self-generalization may read as
continuous with a character's ongoing self-awareness (still interior,
whatever the syntax), while generalizing about someone *else's* fixed
nature already resembles the kind of external, authoritative judgment
`omniscient_description` is defined by, independent of sentence
architecture. This isn't confirmed — the sample crossing self/other with
syntax-type is too thin to separate the two variables — but it's a
concrete, testable second hypothesis rather than a vague caveat, and the
next scaling round could stratify on both axes at once (self- vs.
other-directed, hypotactic vs. paratactic) rather than aspect alone.

A genuine puzzle, reported rather than resolved: `AMBIGUITY.md` finds
Richardson's Pilgrimage novels have the lowest cross-model agreement of
any author in the full corpus (7.7–8.1% for Pointed Roofs and Backwater,
using the same three models as this test). This sample finds close to the
opposite — Richardson's ten passages were, if anything, *more* stable
across models than Woolf's fourteen, once mercury-2's general noise is set
aside. Both can be true at once only if whatever drives Richardson's
documented corpus-wide ambiguity is a different axis than habitual/punctual
aspect — something this hand-picked, aspect-stratified sample didn't
happen to select for. That's worth taking seriously as a limitation of
this design rather than explaining away: the passages here were chosen to
be clean instances of a grammatical category, which may have systematically
excluded whatever *does* make Richardson's prose hard to classify at scale
(possibly passage length, possibly a different boundary — AMBIGUITY.md
also flags `simulation_state_of_mind`/`indirect_interior_monologue` overlap
as distinctively common in Richardson, which none of these ten passages
were selected to test directly). Finding that axis is a more promising next
target than further scaling this one.

## Where this points next

The architecture-versus-aspect question that opened this round is now
answered well enough to stop re-testing it: neither factor predicts alone,
their conjunction does, and that held up under a targeted three-passage
check designed specifically to break it, including the single cleanest
piece of evidence in the study (Richardson, writing against her own grain,
still producing the strongest omniscient reading in the dataset the moment
both ingredients were present). Three follow-ups remain open, in rough
order of cost versus payoff. First, cheap: re-read `AMBIGUITY.md`'s
existing Richardson disagreement data (already collected, no new API calls
needed) for what those passages have in common that the twelve tested here
don't — length, the indirect/simulation_state_of_mind boundary specifically,
something else — since this design's own passages turned out to be
unusually stable and clearly aren't sampling whatever drives her
corpus-wide ambiguity. Second, moderate: the self- versus other-directed
generalization thread above is still just an observation, not a tested
hypothesis — a sample crossing self/other with the now-confirmed
elaborate+habitual conjunction, independent of author, would show whether
it's a real third ingredient or an artifact of which passages happened to
be available. Third, more expensive but probably eventually necessary:
extend to a third author with yet another sentence style — Faulkner's long
unpunctuated interior monologue is architecturally closer to Woolf's
periodic construction than to Richardson's paratactic one, and would test
whether "elaborate + generalizing" predicts the omniscient pull outside
these two authors too, or whether something author-specific is still
riding along with the conjunction.

(Supporting data: `cross_model_aspect_test.py`/`cross_model_aspect_
results.jsonl` for the Woolf set, `cross_model_richardson_test.py`/
`cross_model_richardson_results.jsonl` for the Richardson set,
`cross_model_architecture_test.py`/`cross_model_architecture_results.jsonl`
for the three-passage disentangling check. Same
methodology and models throughout.)
