# *The Waves*, measured: what the mediation-cline data actually supports

*A note explaining the finding, with textual examples, for discussion — revised after a methodological review (`technique_v_genre/critique.md`) caught that the original headline statistic (53% between-book variance) was doing more work than the data could support. This version leads with the claims the data actually carries.*

## The finding, stated plainly

Of Woolf's ten novels in the corpus, nine sit within a narrow, unremarkable band — they are not distinguishable from each other on the measure described below. The tenth, *The Waves*, is not on that curve at all: it is more radically unmediated, on average, than either of William Faulkner's novels in our comparison set, and more unmediated than *Ulysses*. That is a specific, falsifiable, single-book claim, and it is the part of this finding that survives close scrutiny.

An earlier version of this note led with a different, broader-sounding number — that 53% of the total variation in how "mediated" Woolf's prose is comes from which book you're reading, versus 22% for Joyce and under 2% for Faulkner, James, and Richardson — and used it to argue that Woolf's stream-of-consciousness technique is unusually book-dependent across her whole career. That statistic is real, but it is almost entirely the arithmetic shadow of *The Waves* alone: remove that one novel and the figure collapses from 53% to roughly 20%. The 53% was never a finding about ten novels. It was a finding about one, wearing the syntax of a career-wide claim. This version corrects that.

## What "mediated" means here, in her own sentences

The measurement comes from Dorrit Cohn's account in *Transparent Minds* of narratorial distance — not a binary between "inside the character's head" and "not," but a graded position running from the narrator clearly organizing and reporting a character's thought (Cohn's *psycho-narration*), through *narrated monologue* (what's usually called free indirect discourse), to *quoted monologue* (thought explicitly tagged and quoted, "she thought"), to *autonomous monologue* (interiority given with no narratorial frame at all — nothing standing between the reader and the character's mind). Two independent language models read each passage and placed it along this four-point scale; where they agree, we treat that as a reasonably stable reading, in the same spirit as inter-rater agreement in manual coding — though see the Limitations section below for what that comparison does and doesn't license.

Four passages from the corpus, one from each end of the scale, make the distinction concrete — and all four happen to be Woolf's own prose, which is useful, since it means the ladder itself is internal to her range rather than a comparison across authors:

**Psycho-narration** (*Night and Day*): "From an acute consciousness of herself as an individual, Mary passed to a conception of the scheme of things in which, as a human being, she must have her share. She half held a vision; the vision shaped and dwindled." The narrator is visibly doing the organizing — summarizing what Mary's mind is doing, from outside it, rather than giving us her actual idiom.

**Narrated monologue / free indirect discourse** (*Mrs Dalloway*): "Hugh was becoming an intolerable ass... And yet Richard couldn't have said that to save his life! Why these people stood that damned insolence he could not conceive." No tag announces this as thought — the syntax is still third-person, past-tense narration — but the evaluative charge ("that damned insolence," "an intolerable ass") is unmistakably Richard's own voice bleeding into the narration. This is Cohn's own paradigm case for the technique, and it is Woolf's default mode across most of her career.

**Quoted monologue** (*To the Lighthouse*): "he was thinking, Go on reading. You don't look sad now, he thought. And he wondered what she was reading..." Here the thought is explicitly tagged and quoted — closer to the character's own idiom, but the narrator still marks the boundary each time. Tellingly, this particular passage was one the models themselves flagged as unstable between this category and free indirect discourse — exactly the kind of boundary case Cohn's own close reading of a different *Lighthouse* sentence (which she calls "ambiguously suspended between narration and narrated monologue") already anticipates. The instability is not model error; it is the text refusing to resolve, which is the whole reason the scale is graded rather than a set of hard boxes.

**Autonomous monologue** (*The Waves*): "I am arrayed, I am prepared... I glance, I peep, I powder. All is exact, prepared. My hair is swept in one curve. My lips are precisely red." No narrator, no tag, no past-tense frame — first person, present tense, nothing mediating between the reader and Rhoda's (or whichever voice's) mind at all.

## The shape of a career: nine novels, then one

Ranking her ten novels by their average position on this scale:

| novel | year | mean position on the cline |
|---|---|---|
| *Night and Day* | 1919 | 0.35 |
| *The Voyage Out* | 1915 | 0.36 |
| *Flush* | 1933 | 0.36 |
| *Orlando* | 1928 | 0.46 |
| *Jacob's Room* | 1922 | 0.56 |
| *To the Lighthouse* | 1927 | 0.79 |
| *Mrs Dalloway* | 1925 | 0.87 |
| *The Years* | 1937 | 0.91 |
| *Between the Acts* | 1941 | 1.00 |
| *The Waves* | 1931 | **2.49** |

The more interesting claim here is arguably the negative one: nine books, written across 1915–1941 — spanning the invention of her mature method — land inside a band roughly 0.65 wide on a 0–3 scale. Adjacent titles in that band are separated by amounts smaller than the passage-to-passage noise within any one of them. If the standard account of Woolf's development treats the move from *Night and Day* to *Mrs Dalloway* as a change in narratorial distance, this measure doesn't find that change — whatever *did* shift between those books, on this specific axis it looks close to flat. That's a claim the data can actually carry, and it's the load-bearing context for what *The Waves* then does.

*The Waves* is not on that curve. Its average is two and a half times higher than *Between the Acts*, her next-most-interior novel, and higher than either of Faulkner's novels in our comparison set (*As I Lay Dying*, 1.89; *The Sound and the Fury*, 2.21) — the two texts used to define what an extreme, genre-defining commitment to unmediated interiority looks like in this corpus. By this specific measure, the average passage of *The Waves* sits further toward pure, unframed interiority than the average passage of *Ulysses* (1.77).

That comparison is worth stating carefully rather than triumphantly, for two reasons. First, the scale may partly be measuring grammar as much as radicalism: a first-person, present-tense soliloquy scores toward "unmediated" almost by construction, and *The Waves* is built entirely from that form. Second, and more interestingly, a follow-up check on how the *level itself moves through each text* (lag-1 autocorrelation of mediation-level along reading order — see `technique_v_genre/woolf_variance_robustness_report.md`) shows *Ulysses* sustaining a register in much longer runs (0.57) than *The Waves* does (0.24). That's consistent with what any reader of both books already knows: *Ulysses* spends long stretches in modes — Eumaeus's exhausted cliché, Ithaca's catechism, Cyclops's parody — that are heavily mediated in ways orthogonal to Cohn's cline, while *The Waves* holds its single unmediated register far more uniformly throughout. The finding may be less "*Waves* out-radicalizes *Ulysses*" than "*The Waves* is uniformly one thing across its whole length, and *Ulysses* is deliberately many things" — which is itself a real, measurable difference between the two books, just not quite the same claim. (This specific ordering also hasn't yet been stress-tested against the memorization objection discussed in Limitations below — the masking test run for this note establishes that passage-level scores generally track a passage's own grammar rather than the book's identity, but wasn't designed to isolate the *Waves*-vs-*Ulysses* comparison specifically.)

## Restraint as a choice — the claim that doesn't need the statistic

The strongest interpretive move available here doesn't actually depend on any variance percentage at all. It only needs two facts already established above: nine of Woolf's ten novels sit in a flat, mutually indistinguishable band, and the tenth sits beyond anything else in the corpus, Faulkner and Joyce included. Put together, those two facts say that Woolf had the extreme register available to her — demonstrably, since she used it once, to greater effect than anyone else in this comparison — and chose, in nine other books, not to use it. That is a claim about *restraint as a deliberate choice*, not about a "quiet" style she simply had. It is compatible with, and arguably sharpens, Dora Zhang's argument that Woolf's innovations "announce themselves more quietly" than Joyce's: quiet was optional for her, not constitutional.

## The cross-author comparison, corrected

An earlier version of this note reported eta-squared (the between-book share of total variance) for each author: Woolf 53%, Joyce 22%, Faulkner 2%, James and Richardson both under 1%. Two problems with that statistic, both caught in review:

1. **Eta-squared is biased by how many titles an author has.** An author with ten titles is structurally advantaged over one with two, independent of any real difference in how their style varies — Woolf's ten-title sample and Faulkner's two-title sample are not being compared on equal footing. The corrected version below uses a variance-components intraclass correlation (ICC) instead, which accounts for this, computed with a bootstrap confidence interval (`technique_v_genre/woolf_variance_robustness.py`).
2. **The comparison set had two corpus-composition errors.** *The Wings of the Dove* was counted as two of James's "three titles" — it is one novel, split across two files purely for corpus-building reasons, not two independent works. James's true title count is 2, not 3. Separately, Richardson's three *Pilgrimage* volumes are installments of one continuous narrative (one protagonist, one uninterrupted story), not independent books — her near-zero between-title variance is close to tautological and shouldn't be read as a "fixed style" finding the way James's or Faulkner's can be.

| author | titles | ICC (95% CI) | ICC without highest-mean title |
|---|---|---|---|
| Faulkner | 2 | 0.037 [0.006, 0.085] | — (needs ≥3 titles) |
| James | 2 | 0.008 [0.000, 0.022] | — (needs ≥3 titles) |
| Joyce | 3 | 0.358 [0.322, 0.391] | 0.117 (without *Ulysses*) |
| Richardson\* | 3 | 0.001 [0.000, 0.022] | 0.000 (without *Honeycomb*) |
| Woolf | 10 | 0.561 [0.534, 0.588] | **0.199 (without *The Waves*)** |

\* Not a fair "does style vary by book" comparison case — see above.

Woolf's corrected figure (ICC 0.56) is close to the original naive eta-squared (0.53), so the ICC correction itself doesn't overturn the headline number — what overturns it is the leave-one-out check: removing her single highest-scoring novel drops the figure by nearly two-thirds, to 0.20, which is the same story the ranking table above already tells in plainer terms. Joyce shows the same pattern at smaller scale: without *Ulysses*, his figure drops from 0.36 to 0.12. The number that should travel with "53%" (or its corrected version, "56%") is always "and mostly one book."

One robustness check in Woolf's favor: *Flush* and *Orlando* are genre-heterogeneous relative to her other eight novels (a dog's-eye-view biography, a mock-biography) — dropping both from the comparison actually raises her ICC slightly, to 0.586, so the finding is not fragile in that direction. But the choice to include or exclude genre-adjacent texts should be stated, which it now is.

## Connecting this to the reception argument

This bears on the argument that Woolf's contribution to stream-of-consciousness technique has been under-recognized because it "announces itself more quietly" than Joyce's, as Dora Zhang puts it — and on the related cluster of claims (that her prose is too grammatically regular to read as organic, that she is persistently ranked second to Joyce as an innovator, that her primary mode, free indirect discourse, reads as the more conventional, less revolutionary technique). Cohn's own account already supplies part of the reason why: narrated monologue "needed no Joycean revolution" and instead formed "a stylistic bridge" from nineteenth- to twentieth-century fiction, which is a fair description of nine of Woolf's ten novels — and, per the corrected framing above, only nine of them.

That *The Waves* is generically discontinuous with the rest of Woolf's fiction is not, on its own, a new claim — it is close to consensus in Woolf scholarship, and has been argued on close-reading grounds (its abandonment of the narratorial positioning that organizes her other novels; its treatment as a different genre, the "playpoem," rather than a more extreme novel; Woolf's own description of it as "abstract, eyeless"). What this measurement adds is not the discovery that *The Waves* is different — critics already knew that — but a specific, falsifiable claim about *how* different, expressed on the same graded scale used across the whole comparison set, which lets the difference be stated in the same terms as Joyce's and Faulkner's, rather than only in the vocabulary of Woolf criticism's own syllabus.

## How the numbers were computed (for reference)

For each Woolf novel, every scored passage's mediation position (0–3) was averaged to a single passage-level score, collapsing across the two independent model readings. Two statistics were then computed on that per-passage data: a one-way random-effects intraclass correlation with a bootstrap confidence interval (resampling passages within each fixed title, 2,000 iterations), correcting the group-count bias in the originally-reported eta-squared; and a leave-one-title-out sensitivity check, recomputing the statistic with each title removed in turn. The same calculations were run for every other author with more than one title in the corpus. Full per-title numbers, the complete leave-one-out table, within-title level distributions, and the autocorrelation figures are in `technique_v_genre/woolf_variance_robustness_report.md`; the underlying script is `technique_v_genre/woolf_variance_robustness.py`.

## Limitations

**The scale is ordinal, not interval.** Averaging codes 0–3 assumes the step from psycho-narration to narrated monologue is the same size as the step from quoted to autonomous monologue. Cohn does not claim that, and the *To the Lighthouse* example above — models unstable between two adjacent categories — is itself evidence the intervals aren't uniform. The within-title level distributions in `woolf_variance_robustness_report.md` (full 0–3 proportions per title, not just the mean) are the more honest version of this data; a 0.87 mean for *Mrs Dalloway*, for instance, turns out to come from a text sitting at level 1 74% of the time, not from wide oscillation between 0 and 2.

**Memorization.** GPT-4-class models demonstrably know popular public-domain fiction well enough to reproduce passages, and *Mrs Dalloway*, *To the Lighthouse*, *The Waves*, and *Ulysses* are among the most-discussed novels in existence — a model asked how mediated a *Waves* passage is could in principle be retrieving critical consensus about the book rather than reading the passage. We tested this directly: 177 passages across the six titles carrying this note's headline claims (*The Waves*, *Ulysses*, *As I Lay Dying*, *The Sound and the Fury*, *Mrs Dalloway*, *To the Lighthouse*), stratified across mediation levels, had every character name replaced with a generic placeholder (*Mrs Ramsay* → *Character 01*, etc.) and the source filename anonymized — the two only channels through which the scoring prompt could carry a book's identity — then rescored with the same two readers. Masked scores correlated strongly with the originals at the individual-passage level (r=0.90, mean absolute shift 0.25 on the 0–3 scale, no meaningful net bias), which is evidence the classifier is substantially responding to a passage's own grammar rather than retrieving the book it belongs to. This test is *not*, however, a check on the specific *Waves*-vs-*Ulysses* aggregate ordering — the stratified sampling needed for the paired comparison flattens out exactly the distributional imbalance that produces that ordering in the first place, so a title-mean comparison on this sample isn't meaningful in either direction. Full results in `technique_v_genre/memorization_test_report.md`; a proportionally-sampled follow-up would be needed to test the aggregate ordering under masking directly.

**Two correlated models agreeing is not independent inter-rater reliability.** The two readers share training data, prompt framing, and plausibly correlated failure modes — their agreement is evidence of *something* stable, but not the same kind of evidence human-human inter-rater agreement would be. The remaining open item beyond this note is validating a sample against trained human coders working from the same rubric; that hasn't been done yet.

Treat this note as a data-driven complement to close reading — a way of finding where to look, and a way of stating a known difference (*The Waves* vs. everything else) in falsifiable, cross-author-comparable terms — not a replacement for the reading itself.
