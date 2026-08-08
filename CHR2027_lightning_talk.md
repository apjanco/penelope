# CHR 2027 Lightning Talk Proposal (draft)

**Submission type:** Lightning Talk — abstract up to 750 words, 3-minute presentation
**Deadline:** submissions close August 14, 2026 (UTC-12); notify Oct 23, 2026
**Note:** CHR review is double-blind via EasyChair — strip author names/affiliations and
self-identifying links before submission. Use the ACH LaTeX template.

---

## Working title

*The Moving Frontier: Is Stream-of-Consciousness a Technique or a Genre?*

Alternate titles to consider:
- *Woolf's Quiet Ambiguity: Testing a Reception History Computationally*
- *Penelope's Loom: A Continuous Model of Stream-of-Consciousness Mediation*
- *Three Predictions for Stream of Consciousness: Genre Pole, Technique, or Historical Frontier?*
- *Training a Model to Disagree: Stream-of-Consciousness Typology as a Site of Productive Ambiguity* (earlier framing, kept as an option if the committee wants the disagreement angle foregrounded over the case-study structure)

---

## Draft abstract (~1040 words)

Stream-of-consciousness (SoC) narration remains, a century into its
critical life, notoriously difficult to pin down. As Hoyt Long and Richard
Jean So put it, "no one can agree on what it is, yet everyone seems to know
it when they see it" — not, we think, a failure of criticism so much as a
structural feature of the object itself. Robert Humphrey's foundational
1954 taxonomy treats SoC as a toolkit of nameable techniques — direct and
indirect interior monologue, soliloquy, omniscient description — each
presumed to have a defensible boundary. A counter-tradition argues this
gets the object wrong: Monika Fludernik's natural narratology suggests that
what readers actually experience as SoC resists discrete sorting
altogether, shading continuously into ordinary narration rather than
snapping into Humphrey's categories. A third position reframes the
question rather than settling it: Dorrit Cohn's psychic-distance model
asks not which technique a passage belongs to, but how much mediation
stands between narrator and character consciousness — an ordinal scale
from psycho-narration, through narrated and quoted monologue, to
autonomous monologue. We build our method on Cohn's scale precisely
because it lets us ask empirically what neither Humphrey nor Fludernik
could settle by close reading alone.

To test this, we built a pipeline, Penelope, that scores passages from
roughly twenty modernist novels — Virginia Woolf, Dorothy Richardson,
James Joyce, William Faulkner, Henry James, Ford Madox Ford — against this
same cline. Each passage is read independently by multiple frontier
language models (GPT-5.4, Qwen3-Max), resampled at temperature, with
explicit fields for the places where the typology itself seems to strain:
boundary tension between adjacent categories, passages where no category
quite fits, and what we call, borrowing from Richardson's and Alber's
unnatural narratology, antimimetic markers — passages that resist realist
consciousness-representation altogether.

Woolf is our pivotal case, and here we lean on our co-author's own account
of her reception. She is frequently cited as one of the foremost pioneers
of SoC narration, yet, as Dora Zhang has counterintuitively noted, her
innovations have not been clearly recognized: compared to Joyce's, her
technique "announces itself more quietly." We trace this discrepancy to
three recurring explanations: her relative adherence to grammatical
structure, which makes her writing seem constructed rather than organic;
her persistent second billing to Joyce, a more radical innovator by
reputation if not by any settled measure; and her predominant use of free
indirect discourse — what Cohn calls "narrated monologue," a technique
that, unlike Joyce's, "needed no Joycean revolution" and instead formed "a
stylistic bridge" from nineteenth- to twentieth-century fiction. We are
not interested in confirming or overturning this reception history so much
as testing it: is Woolf's "quietness" a fixed personal style, or a choice
deployed differently novel to novel?

The corpus answers, and not quite in the way we expected. In aggregate,
Woolf's profile is exactly as quiet as predicted — a mean mediation well
below Faulkner's — and the rest of the panel falls into place around her
almost as tidily: Faulkner's spread confirms the genre-pole prediction,
James sits at the floor, and Joyce's own developmental arc marches
rightward across his career, from *Dubliners* through *Portrait* to
*Ulysses*, exactly as his reputation would predict. But Woolf's aggregate
turns out to be hiding something. Broken down title by title, nine of her
ten novels sit in a narrow, mutually indistinguishable band, and the
tenth, *The Waves*, is not on that curve at all — more radically
unmediated, on this measure, than either of Faulkner's novels. Ask how
much of an author's total variance simply comes down to which book you're
reading, and Woolf's share is the largest of anyone in the panel with a
real multi-title career, by a wide margin over Joyce, the only other
author whose books differ enough to make the comparison meaningful. But in
neither case is that share a stable, career-long trait: set *The Waves*
aside and Woolf's figure falls by nearly two-thirds; set *Ulysses* aside
and Joyce's falls by the same proportion. What's load-bearing here isn't a
percentage, which by itself would just be the arithmetic shadow of one
book — it's the shape: nine flat novels, then one that goes further than
anything else in the corpus. Woolf's "quiet" reputation, in other words,
is not a style at all. It is nine genuinely similar deployments of
restraint and one deliberate departure from it, closer in spirit to what
Cohn was already finding by close reading, when she described a single
sentence in *To the Lighthouse* as "ambiguously suspended between
narration and narrated monologue," than to any fixed authorial voice.

Two further checks turn from authors back to the cline itself. Pooled
across the corpus, the mediation distribution shows a real valley at
quoted monologue, not the smooth continuum we had expected — a shape
closer to Humphrey's discrete types than to Fludernik's gradient, though a
coarse four-bin count is not yet the dip-test our fuller design calls for.
And the typology's breakdown rate shows no significant trend across our
1902–1941 window, which leaves open a question Fludernik's account of
conventionalization and Alber's and Richardson's unnatural narratology
would answer differently: does the antimimetic remainder shrink as a
technique gets absorbed into the tradition, or does it stay put?

We close against Nan Da's critique of computational literary studies: that
much distant-reading work is statistically fragile or, worse,
interpretively hollow. We take that risk seriously because our own
project keeps finding versions of it in its own results and correcting
for them rather than reporting past them: an earlier, cruder measure, raw
cross-model disagreement, looked at first like a stable misfit signal,
until closer comparison across Woolf and Richardson showed it wasn't; the
variance figure above first came to us as a flatter, uncaveated
percentage, until a proper accounting of how much one book was carrying
it — and a check on whether the scoring was reading passages or recalling
criticism about them — showed what was actually load-bearing. What we
offer instead is resampled, falsifiable, multi-model scoring, tested
against predictions drawn from the field's own scholarship rather than a
single ground-truth label — a way, we hope, to ask whether a century of
calling one writer "quiet" was ever a description of her technique, or
only ever a way of measuring her against Joyce.

---

## Notes for revision

- **This draft now directly incorporates your co-author's argument and
  citations** (Zhang on Woolf's under-recognition; Cohn's "narrated
  monologue" / "stylistic bridge" reading and the *To the Lighthouse* close
  reading; the three-part reception-history explanation) as the motivating
  frame for Experiment A. Since it now makes specific interpretive claims in
  their name, **this needs their review before submission** — please verify
  the quoted fragments against the actual source text and get the missing
  bibliographic details (Zhang's article title/venue/year aren't in the
  pasted draft, only page numbers 133/141).
- **Voice pass (this revision):** rewritten throughout to match your
  co-author's register — longer, hedged sentences, "we suggest/argue/trace"
  framing, and the disclaiming move their draft uses ("we are not interested
  in creating a definitive typology... nor in verifying...") now echoed in
  the Woolf paragraph ("we are not interested in confirming or overturning
  this reception history so much as testing it"). Same claims and citations
  throughout, no content changes.
- **Structure/voice (this revision):** reverted the cold statistical
  opening — it read like a science-poster hook, not your co-author's
  register, which earns its claims through critical context before stating
  them. Back to the original essayistic opening (Long & So, then
  Humphrey/Fludernik/Cohn), with the 53%/22%/2%/<1% variance-decomposition
  finding now disclosed mid-essay, in prose, as the "twist" the corpus turns
  up after the reception-history setup — "the corpus answers, and not quite
  in the way we expected" — rather than as a bracketed statistic
  ("std=1.23, confirming...") or a headline number. That bracketed-stat
  phrasing is gone throughout; results are stated in sentences now.
- **Word count:** now ~1040 words — the variance-finding correction added
  more than the dropped Faulkner/James/Richardson percentages saved back.
  A compression pass is still owed before Aug 14 and is now more urgent;
  cut sentences, don't revert to punchier science-abstract phrasing (that
  register was already tried and reverted once — see earlier notes).
- **Corpus mismatch still open:** your co-author's draft lists the
  comparative corpus as Richardson, Joyce, Faulkner, and May Sinclair; the
  actual scored corpus is Richardson, Joyce, Faulkner, Henry James, and Ford
  Madox Ford — no Sinclair. This abstract uses the real (James/Ford) corpus
  since that's what the pilot numbers are drawn from. Sinclair appears in
  your co-author's draft in a different role (the critic who coined the term
  reviewing Richardson) — worth keeping as a citation/historical anchor for
  Experiment C's diachronic window, but not currently a scored author. Still
  needs an explicit decision with your co-author: drop her from their
  "comparative set" sentence, or actually add her fiction to the corpus.
- **Methods-paragraph reconciliation:** the co-author's draft describes
  fine-tuning a local model (Qwen3-4B) against a ten-class taxonomy
  synthesizing Humphrey and Erwin Steinberg. This abstract describes the
  current ensemble-scoring / mediation-cline pipeline instead, which is a
  different (newer) stage of the project. Still unresolved: does the paper
  present both as sequential stages, or fully migrate the methods section to
  the cline-based approach? Worth flagging to your co-author directly, since
  their intro paragraph will need the same update this abstract just got.
- **Anonymization:** remove any GitHub links / model names tied to your HF
  account (`apjanco/...`) before submission, per double-blind policy — or
  phrase as "a publicly released model checkpoint" without the identifying
  path. `technique_v_genre/HANDOFF_della.md` and `pilot_report.md` should not
  be linked directly in the submission for the same reason. Citing Zhang,
  Cohn, Fludernik, etc. by name is fine — they're not submission authors.
- **Numbers are now full-corpus, not pilot.** All 21 files (11,952 passages,
  72,167 passage/reader/run rows) are scored and `pilot_experiments_abc.py`
  has been re-run against the complete set — the Faulkner/Woolf/Joyce/James
  figures above are final unless the underlying scoring changes. One
  provenance detail worth a methods-section footnote: 11 passages across
  Ulysses, *The Sound and the Fury*, and *The Ambassadors* were permanently
  rejected by DashScope's content moderation (`data_inspection_failed`) and
  were instead read by a local open-weight model (`qwen3.6-local`,
  Qwen/Qwen3.6-27B) substituted in place of the missing qwen3-max reading for
  just those passages — a commercial safety filter silently excluding
  exactly the most extreme end of the distribution is itself a small,
  citable methodological wrinkle, not just a data gap.
- **Variance finding corrected (this revision), per an external methodological
  review** (`technique_v_genre/critique.md`) and the response to it
  (`woolf_variance_decomposition.md`, `woolf_variance_robustness.py`,
  `memorization_masking_test.py`). What changed and why it matters for this
  abstract:
  - The bare "53%" was almost entirely the arithmetic shadow of *The Waves*
    — removing it drops the figure to ~20%. The abstract no longer states
    the percentage without the leave-one-out fact attached ("set *The
    Waves* aside and Woolf's figure falls by nearly two-thirds"), and now
    parallels it with Joyce's own figure showing the identical pattern
    (falls by the same proportion without *Ulysses*) rather than presenting
    Woolf as a uniquely book-dependent outlier.
  - Dropped the precise Faulkner/James/Richardson percentages from the
    abstract prose entirely — *The Wings of the Dove* was miscounted as 2 of
    James's "3 titles" (it's one novel split across two files), and
    Richardson's *Pilgrimage* volumes are installments of one continuous
    work, not independent books, making her figure close to tautological.
    Both are fixed in the underlying data now, but neither author has a
    real enough multi-title career to state a comparison figure for in
    750 words without the caveats those figures need — the abstract now
    just says so, rather than reporting misleading precision.
  - Removed the implicit "*The Waves* is the single most extreme text in
    the entire corpus" claim (i.e., beats *Ulysses*) that the previous
    draft carried — a memorization/masking test (177 passages, character
    names and source files anonymized, rescored) found scores are strongly
    correlated with the originals (r=0.90) and not badly biased in
    aggregate, which is reassuring evidence the model reads passages rather
    than recalling criticism about famous books — but the test wasn't
    designed to validate that specific *Waves*-vs-*Ulysses* ordering, so
    the abstract no longer asserts it. If asked in Q&A: the honest answer
    is "we checked the classifier isn't just pattern-matching to book
    identity in general, but haven't yet isolated this one comparison."
  - The closing paragraph now names this correction itself as a second
    instance (after the earlier cross-model-disagreement lesson) of the
    project catching and fixing its own Da-critique-shaped risk, which
    strengthens rather than weakens the methodological-rigor pitch.
  Full numbers: `technique_v_genre/woolf_variance_robustness_report.md`
  (ICC + bootstrap CI + leave-one-out) and
  `technique_v_genre/memorization_test_report.md`.
- **Where the v3 dual-reading / GRPO model fits:** still a "future work" beat
  at most, not the centerpiece — see prior revision notes if this needs
  reviving.
- **Fit to CFP:** the pitch maps onto CHR's "modelling bias, uncertainty, and
  conflicting interpretation" interest, and the Woolf case-study framing adds
  a concrete literary-historical stake (a reception-history claim tested
  computationally) that broadens appeal beyond a pure-methods talk.
- **Talk structure for 3 minutes:** (1) the critical claim — disagreement
  tracks something real, testable against literary-historical predictions
  (20s); (2) the pipeline + mediation cline in one breath (25s); (3) Woolf
  as the pivotal case — the reception-history claim, stated as a prediction
  (40s); (4) the turn — aggregate confirms it, then the nine-flat-novels-
  plus-one shape, landing on "set *The Waves* aside and the number falls by
  two-thirds — same story for Joyce and *Ulysses*," ideally with a chart
  showing the flat band and the outlier together (50s); (5) B/C in one line
  each, then the closing provocation — a century of calling one writer
  "quiet" may have been measuring her technique, or just measuring Joyce
  (25s).
