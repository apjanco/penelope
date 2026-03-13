---
name: soc-classification
description: >
  Classify and analyze stream of consciousness (SoC) passages in modernist literature.
  Use this skill whenever the user asks to identify, label, classify, or analyze interior
  consciousness techniques in fiction — especially works by Joyce, Woolf, Faulkner,
  Richardson, or other modernist authors. Also trigger when the user mentions Humphrey's
  typology, interior monologue, free indirect discourse, free association, space-montage,
  or any technique for representing consciousness in narrative prose. Useful for building
  training datasets, annotating passages, or developing computational approaches to
  literary analysis of stream of consciousness.
---

# Stream of Consciousness Classification

This skill provides a taxonomy and decision framework for classifying stream of consciousness
(SoC) techniques in modernist fiction. It draws on two primary sources:

- **Robert Humphrey**, *Stream of Consciousness in the Modern Novel* (1954) — provides
  the foundational four-type taxonomy plus additional devices
- **Erwin R. Steinberg** — extends the analysis with attention to affective simulation,
  mixed-mode passages, and the distinction between memory-based and constructive
  (fantasy/reverie) interior monologue

The taxonomy below synthesizes both frameworks.

## Why this matters

Stream of consciousness is not a single technique — it is a family of narrative strategies
for representing the pre-speech levels of consciousness. Misclassifying these techniques
(or treating them as interchangeable) obscures the very thing that makes modernist prose
innovative: the *degree* and *manner* in which narrative authority is ceded to a
character's inner life. Accurate classification supports close reading, computational
literary analysis, and the development of AI models that can reason about narrative voice.

## Taxonomy

### Core Categories (Humphrey's Four Types)

#### 1. Direct Interior Monologue

The character's consciousness is presented directly, without narratorial mediation.
There is no guiding intelligence between reader and character. No "he thought" or
"she felt" — the reader is dropped into the mind itself.

**Key markers:**
- First person or character-anchored perspective with no narrator frame
- No reporting verbs ("thought," "wondered," "felt")
- Syntax may break down, mimicking raw thought
- Can range from highly lyrical (Stephen Dedalus) to colloquial (Molly Bloom)

**Spectrum within this category:**
Direct interior monologue is not monolithic. Compare Molly Bloom's unpunctuated,
associative rush with Stephen Dedalus's compressed, poetic impressions. Both lack
a mediating narrator, but the texture of consciousness differs vastly — one mimics
the rhythms of spoken thought, the other the fleeting perceptions of an artistic
sensibility. When classifying, note where on this spectrum a passage falls.

**Humphrey ref:** pp. 24–29

#### 2. Indirect Interior Monologue

An author presents unspoken material as if directly from a character's consciousness,
but maintains an authorial presence through third-person references, occasional
commentary, or narrative framing. The narrator is there, but yields significant
ground to the character's voice.

**Key markers:**
- Third-person pronouns, but the *content* and *diction* belong to the character
- Occasional reporting cues ("thought Clarissa," "she felt"), but these are light
  and often parenthetical
- The passage could almost be direct interior monologue if you swapped pronouns
- Often slides between narrator and character within a single sentence

**Relationship to free indirect discourse:**
Indirect interior monologue overlaps heavily with what narratologists call "free
indirect discourse" (FID). In practice, for SoC classification, treat them as
closely related — the distinction is that indirect interior monologue specifically
targets *pre-speech* levels of consciousness, while FID can also represent more
composed, articulable thought. When a passage uses FID to render raw, unformed
consciousness, classify it as indirect interior monologue.

**Humphrey ref:** pp. 29–33

#### 3. Omniscient Description of Consciousness

The author describes the character's inner experience from the outside, using
conventional third-person narration. The narrator retains full authority — we are
*told about* consciousness rather than *immersed in* it.

**Key markers:**
- Clear narratorial voice distinct from the character
- Psycho-narration: "She considered," "He felt," "The impression colored her world"
- The narrator may interpret or contextualize the character's mental states
- Syntax is typically conventional and well-formed

**Why it counts as SoC:**
Although this is the least "interior" of the core types, Humphrey includes it because
the *subject matter* is consciousness itself. The technique is conventional; the
content is not. Richardson's narration of Miriam Henderson's perceptions in
*Backwater* is a key example — the narrator describes states of mind that Miriam
herself cannot yet articulate.

**Humphrey ref:** pp. 33–36

#### 4. Soliloquy

A character addresses themselves (or an implied audience) in a manner that is more
organized than raw thought but not externally spoken. It has the feel of someone
"talking to themselves" — more coherent than interior monologue, but not dialogue.

**Key markers:**
- First person, addressed to self or absent other
- More syntactically organized than direct interior monologue
- Rhetorical quality — the character is *making a case*, expressing anger, grief, etc.
- Often emotionally heightened

**Distinguishing from direct interior monologue:**
The key distinction is the level of rhetorical organization. Soliloquy has a
discernible argumentative or emotional arc; direct interior monologue tends to
drift associatively. Faulkner's characters in *As I Lay Dying* often use soliloquy —
they are clearly *performing* their inner speech for an implied listener, even
though no one is there.

**Humphrey ref:** pp. 36–39

### Additional Devices

These are not separate "types" in Humphrey's core taxonomy but are important
techniques that operate within or alongside the four core categories.

#### 5. Free Association

The linking of thoughts, memories, or perceptions through non-logical connections —
by sensory similarity, phonetic echo, emotional resonance, or contiguity rather
than by rational argument. Free association is a *mechanism* that can appear within
any of the four core types.

**Key markers:**
- Abrupt topic shifts connected by sensory or emotional threads
- Chains of thought where each link is associative, not logical
- Often triggered by external stimuli (a sight, sound, word) that unlocks memory

**Humphrey ref:** pp. 65–72

#### 6. Space-Montage

Multiple perspectives or locations are juxtaposed in rapid succession, often within
a single scene. The narrative cuts between characters or viewpoints cinematically.
This is a *structural* device — it organizes how consciousness is presented across
multiple minds.

**Key markers:**
- Rapid shifts between characters' perspectives
- A shared external event perceived differently by multiple observers
- Cinematic cutting — the "camera" moves between minds
- External scene serves as connective tissue between interior views

**Humphrey ref:** pp. 49–57

#### 7. Orthographic/Typographic Markers of Consciousness Shifts

Authors use visual features of the text — italics, parentheses, capitalized headlines,
lack of punctuation, etc. — to signal transitions in the level or mode of
consciousness being represented.

**Sub-types:**
- **Italics**: Faulkner's Benjy sections use italics to mark temporal shifts in memory
- **Parentheses**: Woolf uses parenthetical asides for sudden intrusions of thought
- **Headlines/Capitals**: Joyce uses typographic disruptions to signal external
  text intruding on consciousness
- **Punctuation removal**: Joyce's Penelope episode strips punctuation to mimic
  the unbroken flow of Molly's thought

**Humphrey ref:** pp. 57–63

#### 8. Imagery / Literary Impressionism

Sustained sensory imagery that renders a character's subjective perception of the
world. Not merely descriptive — the imagery itself *is* the consciousness. The
character's emotional and psychological state is expressed through how they perceive
external reality.

**Key markers:**
- Dense, often synesthetic sensory language
- The external world is filtered entirely through the character's subjectivity
- Perceptions are emotionally charged, not neutral description
- Boundaries between perceiver and perceived blur

**Humphrey ref:** pp. 73–82

#### 9. Simulation of a State of Mind (Steinberg)

A passage whose formal properties — rhythm, syntax, diction, density — are organized
to simulate a specific emotional or psychological state in the reader. The technique
is not simply *about* an emotion; it formally *enacts* it.

**Key markers:**
- Syntactic structure mirrors the emotional state: sensuous and flowing for happiness,
  clipped and abruptly halted for distress
- Sensory richness increases or decreases to match the affective register
- The passage's rhythm is doing expressive work beyond its semantic content
- Often identifiable by contrast: compare how the same character's monologue changes
  texture when the emotional valence shifts

**Sub-types by affect:**
- **Happiness/pleasure**: flowing syntax, dense sensory imagery, present-tense
  immediacy. Example: Bloom's seedcake memory ("Softly she gave me in my mouth the
  seedcake warm and chewed... Joy: I ate it: joy")
- **Unhappiness/distress**: truncated phrases, abrupt stops, imperative self-commands.
  Example: Bloom on the Tolka walk ("Stop. Stop. If it was it was. Must.")
- **Anxiety/avoidance**: fragmented perception, staccato questions and self-answers,
  physical sensations intruding ("His heart quopped softly... Quick. Cold statues:
  quiet there. Safe in a minute.")

**Why this is distinct from the other categories:**
All SoC techniques convey emotion to some degree, but simulation of a state of mind
foregrounds the *formal mimicry* of affect as the primary literary strategy. The
classification question is: is the passage primarily organized around representing
a *thought process* (→ interior monologue) or around *enacting an emotional state*
through form (→ simulation)?

**Steinberg ref:** p. 41, 45–46

#### 10. Reverie / Constructive Fantasy

A sustained passage in which a character constructs an imagined scene rather than
recalling a real memory or processing present perception. The character is
daydreaming, projecting, or fantasizing.

**Key markers:**
- Conditional or future-oriented diction, or a present tense that describes events
  that are not actually happening
- Sustained coherence unusual for free association — the character is *building*
  a scene, not drifting between fragments
- Sensory details are invented rather than remembered or currently perceived
- Often triggered by a desire, fear, or idle speculation

**Distinguishing from memory-based free association:**
In free association, the mind is pulled backward by sensory triggers to real memories.
In reverie, the mind projects *forward* or *outward* into imagined scenarios. The
Bloom passage imagining a walk through an Eastern city ("Walk along a strand, strange
land...") is constructive — he has never been there. Compare with the seedcake
memory, where he is recalling an actual experience.

**Steinberg ref:** p. 57

### Hybrid and Transitional Forms

Many passages do not fit neatly into a single category. This is expected and
analytically valuable — the transitions between modes are often where the most
interesting literary work happens. Steinberg explicitly flags this for the extended
Stephen Dedalus passage in Ulysses (his pp. 30–31), noting that "not all" sentences
are strictly SoC and that the passage "merges soliloquy and 3rd person omniscient."

**Common hybrid patterns:**
- Omniscient narration sliding into direct interior monologue (Richardson's *Honeycomb*)
- Indirect interior monologue layered with free association (Woolf's *Mrs Dalloway*)
- Soliloquy punctuated by imagery (Faulkner's Dewey Dell sections)
- Third-person omniscient interwoven with soliloquy (Joyce's Stephen passages)
- Simulation of emotional state overlaid on any of the core types

**Sentence-level annotation:**
For extended passages that shift modes, consider annotating at the sentence or clause
level rather than labeling the whole passage. This captures the *transitions* — which
are often the most analytically interesting feature. When a passage merges multiple
modes, note:
- The primary mode at the passage level
- The secondary mode(s)
- Where the transitions occur and what triggers them (often a sensory stimulus,
  emotional shift, or intrusion of memory)

## Negative Examples & Boundary Cases

Accurate SOC classification requires knowing what SOC is *not* — and recognizing the
gray areas where reasonable analysts may disagree. The following examples illustrate
common false positives, near-misses, and genuinely ambiguous cases. When classifying,
use these to calibrate your judgment.

### What Is NOT Stream of Consciousness

#### 1. Conventional Third-Person Narration About Inner States

> *Emma was not sorry to be pressed. She read, and was surprized to find how much the style
> of the letter, though cheerful, shewed real feeling.*

**Why not SOC:** The narrator reports Emma's mental state ("was not sorry," "was surprized")
using standard psycho-narration. The narrator retains full authority, interpreting Emma's
response rather than rendering it. The syntax is orderly and belongs entirely to the narrator's
voice, not Emma's. This is *about* consciousness but does not simulate or immerse us in it.

**What might tempt misclassification:** The passage concerns a character's inner response.
But describing an inner state is not the same as rendering one. The test: could you remove
the narrator without the passage still conveying consciousness? Here, no — without the
narrator, there is no passage.

#### 2. Dialogue Expressing Inner Feeling

> *"I am half agony, half hope. Tell me not that I am too late, that such precious feelings
> are gone for ever."*

**Why not SOC:** Although this is intensely interior and emotional, it is *spoken* (or
written, in the case of Wentworth's letter). SOC techniques represent *unspoken* mental
content. Dialogue can reveal consciousness, but it is mediated through the conventions of
social communication — the character has composed and organized their expression for an
audience. SOC bypasses that organizing step.

**What might tempt misclassification:** The emotional intensity and first-person voice
resemble direct interior monologue. But the presence of an addressee ("Tell me not…") and
the rhetorical polish distinguish this from the unedited texture of thought.

#### 3. First-Person Retrospective Narration

> *I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe; nor
> am I one of your Hollywood-movie ectoplasms. I am a man of substance, of flesh and bone,
> fiber and liquids — and I might even be said to possess a mind.*

**Why not SOC:** Despite the first person and the philosophical self-reflection, this is a
narrator *telling a story about himself* to a reader. It is composed, rhetorically structured,
and temporally distanced from the experience it describes. The narrator has had time to
shape and polish. Compare with Molly Bloom's monologue, where there is no sense of
retrospective craft — the words feel simultaneous with the thinking.

**What might tempt misclassification:** First person + introspection. But first-person
narration is not the same as first-person *thought*. The narrator is performing for a
reader; interior monologue has no audience.

#### 4. Lyrical Description / Purple Prose

> *The autumn leaves drifted in spirals of amber and rust, catching the last light of the
> declining sun, each one a small bright death against the gray November sky.*

**Why not SOC:** This is a narrator (or a character functioning as narrator) producing
polished literary description. The syntax is controlled and balanced. The imagery serves
an aesthetic purpose — it is craft, not cognition. There is no sense of a mind in the
act of perceiving; instead, a mind has already perceived and is now *writing about it*
beautifully.

**What might tempt misclassification:** Dense sensory imagery can resemble a sensory
impression stream. The key difference: in SOC, the imagery is *fragmentary, subjective,
and anchored in the perceiving consciousness*. Here, the imagery is *composed* and
*narrator-controlled*.

#### 5. Philosophical Reflection / Essay Voice

> *Time is the substance I am made of. Time is a river which sweeps me along, but I am
> the river; it is a tiger which destroys me, but I am the tiger; it is a fire which
> consumes me, but I am the fire.*

**Why not SOC:** Although this concerns consciousness, it is discursive and argumentative —
the writer is developing an idea through metaphor. The syntax is parallel and rhetorical.
This is *thinking about* consciousness, not *enacting* it. SOC renders the experience of
thinking; this renders the *product* of thought.

**What might tempt misclassification:** Self-referential content about mind, time, or
identity. But SOC is defined by *technique*, not *topic*. A passage can be about
consciousness without using consciousness-rendering techniques.

#### 6. Dream Sequences Told Conventionally

> *That night she dreamed she was back at Manderley. She walked through the great iron gates
> and up the long drive, and the house was before her, just as she remembered it.*

**Why not SOC:** The dream is narrated in conventional past-tense third person. The narrator
is *describing* a dream, not *simulating* the experience of dreaming. An SOC dream passage
would render the dream's own logic (or illogic) — its distortions, jumps, and surreal
imagery — without the narrator stepping in to organize the account.

**What might tempt misclassification:** Dreams are altered states of consciousness. But
the *content* being about a dream does not make the *technique* stream of consciousness.
The technique must itself enact the qualities of the mental state.

### Gray Areas & Genuinely Ambiguous Cases

These examples illustrate passages where classification is legitimately uncertain. The
correct response to ambiguity is not to force a label but to **acknowledge the ambiguity,
identify the competing interpretations, and explain what textual evidence supports each.**

#### Gray Area 1: Free Indirect Discourse — SOC or Not?

> *She would not say of any one in the world now that they were this or were that. She felt
> very young; at the same time unspeakably aged.*

**The case for SOC (indirect interior monologue):** The diction ("unspeakably aged") feels
like Clarissa's own, not the narrator's. The thought is rendered in her register. The
content — a self-contradictory emotional state — is pre-articulate, the kind of thing
one feels but does not say aloud.

**The case against SOC:** The narrator is present ("She felt," "She would not say"). The
syntax is perfectly composed. This could be elegant psycho-narration — the narrator
describing Clarissa's feelings in language that happens to be sympathetic to her idiom.

**The signal:** Look at the surrounding context. If this passage is embedded in a longer
stretch where the narrator progressively recedes and the character's voice dominates,
it is likely SOC (indirect interior monologue). If the narrator reasserts control
immediately after, it may be a brief moment of free indirect discourse within otherwise
conventional narration. **The classification depends on the mode of the surrounding text.**

#### Gray Area 2: Soliloquy vs. Direct Interior Monologue

> *I will not think of it. I will not. I will think of something else. I will think of
> the money. What did he say? Fifty dollars a month. Fifty dollars...*

**The case for soliloquy:** The character is addressing themselves with imperative
commands ("I will not"). There is a rhetorical structure — self-command followed by
deliberate redirection. The character is *managing* their thought.

**The case for direct interior monologue:** Despite the self-commands, the passage
quickly fragments ("What did he say? Fifty dollars a month. Fifty dollars..."). The
attempted control is breaking down, and the mind drifts into repetition — a hallmark
of uncontrolled thought flow.

**The signal:** This is a *transition* from soliloquy to direct interior monologue.
The self-commands are soliloquy; the fragmentation is interior monologue. **The most
accurate classification is hybrid**, noting where the shift occurs. A passage that
starts organized and ends fragmented tells a story about the character's failing
attempt at mental discipline — and the classification should capture that narrative.

#### Gray Area 3: Sensory Description vs. Sensory Impression Stream

> *The sea was indistinguishable from the sky, except that the sea was slightly creased as
> if a cloth had wrinkles in it. Gradually as the sky whitened a dark line lay on the
> horizon dividing the sea from the sky and the grey cloth became barred with thick strokes
> moving, one after another, beneath the surface, following each other, pursuing each other,
> perpetually.*

**The case for SOC (sensory impression stream):** The imagery is organized by perception,
not by logic. The repetitions ("one after another," "following each other, pursuing each
other, perpetually") mimic the hypnotic quality of watching waves. The passage is rendering
the *experience* of seeing.

**The case against SOC:** The prose is highly controlled and literary. A narrator is clearly
composing this — the metaphor of cloth and wrinkles is crafted, not spontaneous. There is
no specific character perceiving; this is Woolf's narratorial voice at its most lyrical.

**The signal:** Check whether the passage is *anchored in a character's perception* or is
*free-standing description*. If a character is identified as the perceiver (even implicitly),
and the sensory details reflect their subjective state, lean toward SOC. If the passage
is a narratorial set piece — beautiful but unanchored — lean toward literary impressionism
rather than SOC. In the case of Woolf's interludes in *The Waves*, this is a genuinely
unresolved question that scholars disagree on.

#### Gray Area 4: Omniscient Description of Consciousness vs. Non-SOC Psycho-Narration

> *He had the sensation of stepping off the top step of a staircase when there was none. He
> fell endlessly, and with each fall the years peeled away from him, and a gulf opened at
> the pit of his stomach, and the wind of recollection blew bitterly across it.*

**The case for SOC (omniscient description):** The passage describes an interior experience
using subjective metaphors ("stepping off the top step," "wind of recollection"). The
content is consciousness — not action, not dialogue, not setting. Humphrey would include
this because the *subject matter* is the character's inner life.

**The case against SOC:** The narrator is fully in control. The metaphors are the narrator's,
not the character's. This is skillful psycho-narration — the narrator *describing* a
mental event — but it does not simulate consciousness or immerse the reader in it.

**The signal:** Humphrey's category of "omniscient description of consciousness" has the
widest boundary and the most disagreement. The question is: **does the passage aim to
represent the *texture* of mental experience, or merely to report that a mental event
occurred?** If the former (metaphors that convey how the experience *felt*), lean toward
SOC. If the latter ("He remembered his childhood and felt sad"), it is conventional
narration.

#### Gray Area 5: Interior Monologue vs. Simulation of State of Mind

> *Softly she gave me in my mouth the seedcake warm and chewed. Mawkish pulp her mouth
> had mumbled sweetsour of her spittle. Joy: I ate it: joy.*

**The case for direct interior monologue:** First person, no narrator, associative flow.
Bloom is recalling a moment in present-tense memory. The passage is his thought.

**The case for simulation of state of mind:** The syntax — short clauses, the colon-linked
"Joy: I ate it: joy" — formally enacts pleasure. The rhythmic structure is doing expressive
work beyond representing thought. The passage doesn't just *report* joy; its form *is*
joyful.

**The signal:** This is correctly classified as **both** — direct interior monologue that
is *also* a simulation of a state of mind (happiness). These categories are not mutually
exclusive. Interior monologue is the *mode*; simulation is the *function*. When both
apply, note both and explain the relationship.

### Summary: Signals for Discrimination

| Feature | Points TOWARD SOC | Points AWAY from SOC |
|---|---|---|
| Narrator presence | Absent or minimal | Clearly present and organizing |
| Syntax | Fragmentary, associative, mimics thought rhythm | Controlled, balanced, rhetorically polished |
| Pronouns | Shift from narrator to character idiom | Consistently narrator's voice |
| Audience | No implied reader/listener | Addressed to reader, interlocutor, or self-as-audience |
| Temporal stance | Present-tense immediacy of thought | Retrospective, composed after the fact |
| Content organization | Associative, non-linear, driven by perception or memory | Logical, argumentative, narratively structured |
| Sensory detail | Subjective, anchored in a character's perception | Decorative, narrator-controlled, unanchored |
| Emotional texture | Enacted through form (rhythm, syntax, density) | Reported ("she felt sad") |
| Rhetorical polish | Raw, unedited, sometimes ungrammatical | Crafted, metaphorically rich in a composed way |

**When in doubt:** If a passage exhibits *some* SOC features but not others, classify it
with the appropriate type and a **medium or low confidence** level. In the notes, explain
which features are present and which are absent, and identify what would tip the
classification one way or the other. **Acknowledging ambiguity is more valuable than
forcing certainty.**

## Classification Procedure

When presented with a passage to classify:

1. **Identify the narrator position.** Is there a narrator distinct from the character?
   If no → likely direct interior monologue or soliloquy. If yes → indirect interior
   monologue or omniscient description.

2. **Assess rhetorical organization.** Is the passage associative and drifting, or does
   it have an argumentative/emotional arc? Drifting → interior monologue.
   Arc → soliloquy.

3. **Check for affective simulation.** Is the passage's *form* organized primarily to
   enact an emotional state? If the syntax, rhythm, and density are doing expressive
   work that goes beyond representing thought content, consider classifying as
   simulation of a state of mind (noting which affect).

4. **Distinguish memory from fantasy.** If the passage involves sustained scene-building,
   ask: is this recalled or constructed? Recalled → free association / memory.
   Constructed → reverie / fantasy.

5. **Check for additional devices.** Does the passage use free association, montage,
   orthographic markers, or sustained imagery? Note these as layered on top of the
   primary classification.

6. **Look for transitions.** Does the mode shift within the passage? If so, identify
   where and how. Mark these as hybrid. For extended passages, consider sentence-level
   annotation.

7. **Produce a structured annotation** with:
   - Primary SoC type
   - Secondary devices (if any)
   - Affective register (if simulation of state of mind is present)
   - Narrator position (absent / minimal / present / dominant)
   - Character POV
   - Key textual evidence for the classification
   - Confidence level (high / medium / low)
   - Notes on edge cases or ambiguity

## Output Format

For each classified passage, produce an annotation in this structure:

```
Primary type: [one of the core types or hybrid]
Secondary devices: [any of: free association, space-montage, orthographic marker, imagery,
                    simulation of state of mind, reverie/fantasy, or none]
Affective register: [if simulation of state of mind: happiness, distress, anxiety, etc.; else n/a]
Narrator position: absent | minimal | present | dominant
Character POV: [character name]
Evidence: [2-3 specific textual features supporting the classification]
Confidence: high | medium | low
Notes: [any ambiguity, hybrid transitions, or analytical observations]
```

## Reference Corpus

See `references/example_corpus.md` for annotated example passages organized by type,
drawn from Joyce, Woolf, Faulkner, and Richardson. These serve as calibration
examples for classification decisions.

## Processing a Full Novel

### Why chunk?

Even models with very large context windows produce better fine-grained annotations when
working on focused passages rather than entire novels. The reasons are specific to SoC
classification:

1. **Attention degradation.** SoC classification depends on noticing subtle textual
   features — pronoun shifts, punctuation changes, syntactic fragmentation. These
   signals get lost when the model is attending across hundreds of thousands of tokens.
   Research consistently shows that retrieval and analytical accuracy decline in the
   middle of long contexts ("lost in the middle" problem).

2. **Annotation density.** You want annotations at the passage or sentence level, not
   a single label for 250,000 words. Generating dense, structured output for every
   classifiable passage in a novel requires the model to sustain focused analytical
   attention — something that degrades over long inputs.

3. **Output limits.** Even models with 1M+ input windows have output caps (8K–128K
   tokens). A full novel's worth of passage-level annotations will exceed most output
   limits.

### Novel sizes for reference

| Novel                          | ~Words   | ~Tokens (×1.3) |
|-------------------------------|----------|-----------------|
| *Mrs Dalloway* (Woolf)        | 63,000   | ~82K            |
| *As I Lay Dying* (Faulkner)   | 57,000   | ~74K            |
| *To the Lighthouse* (Woolf)   | 70,000   | ~91K            |
| *The Sound and the Fury*      | 107,000  | ~139K           |
| *Ulysses* (Joyce)             | 266,000  | ~346K           |
| *Pilgrimage* (Richardson, 13 vols) | ~500,000 | ~650K      |

Token estimates use the ~1.3 words-per-token ratio typical for literary English, which
tends to run slightly higher than average due to unusual vocabulary, neologisms, and
non-English words. Joyce's *Ulysses* in particular may tokenize at a higher ratio due
to compound neologisms ("scrotumtightening," "strandentwining") and multilingual text.

### Model context windows and recommended chunk sizes

The guiding principle: **use no more than 40–50% of the context window for the input
text chunk.** The remainder is needed for the system prompt (including the skill
instructions and example corpus), the overlap/context buffer, and the output
(structured annotations). For smaller models, be even more conservative.

| Model                        | Context Window | Usable for Text | Recommended Chunk | Notes |
|-----------------------------|---------------|-----------------|-------------------|-------|
| **Qwen3 8B**               | 32K native (131K w/ YaRN) | ~12–14K | **8K–10K tokens (~6–8K words)** | Smaller models need tighter chunks for quality. Stick to native 32K; YaRN-extended context degrades quality on short-text tasks. At this chunk size, *Ulysses* = ~35–43 chunks. |
| **Qwen3 14B/32B**          | 32K native (131K w/ YaRN) | ~14–16K | **10K–12K tokens (~8–9K words)** | Slightly more capable; same context constraints. |
| **Llama 4 Scout (17B)**    | 10M (trained at 256K) | ~50–60K | **10K–15K tokens** | The 10M window is aspirational; reliable quality is at much shorter lengths. Treat 128K as the practical ceiling; chunk well under it for annotation density. |
| **Llama 4 Maverick (400B MoE)** | 1M | ~100K | **15K–20K tokens** | More capable model; larger chunks viable but still well short of the window for annotation quality. |
| **Gemini 2.5 Pro**         | 1M (2M available) | ~200K | **15K–25K tokens** | Strong long-context performance. Could go up to 30K chunks for simpler classification, but keep smaller for hybrid/transition detection. |
| **Gemini 2.5 Flash**       | 1M | ~150K | **10K–15K tokens** | Faster but less precise; keep chunks tighter. |
| **Claude Sonnet 4.5/4.6**  | 200K (1M beta) | ~80K | **15K–25K tokens (~12–19K words)** | Strong at nuanced textual analysis. At 20K-token chunks, *Ulysses* = ~17 chunks, *Mrs Dalloway* = ~4 chunks. |
| **Claude Opus 4.5/4.6**    | 200K (1M beta) | ~80K | **20K–30K tokens (~15–23K words)** | Best for this task; highest quality on literary analysis. Larger chunks viable because of superior attention to textual detail. *Mrs Dalloway* may fit in 3 chunks. |

**Important caveats:**
- These are recommendations for *annotation quality*, not what the models can technically
  accept. You could feed 200K tokens to Claude, but the annotations for passages in the
  middle would be less reliable than for passages at the beginning and end.
- Token counts are approximate. Always tokenize with the target model's tokenizer for
  exact numbers. Literary English tokenizes differently than code or technical prose.
- For **local/self-hosted models** (Qwen, Llama via llama.cpp or vLLM), actual effective
  context depends heavily on quantization level and available VRAM. A 4-bit quantized
  Qwen3 8B on a 16GB GPU will struggle with contexts over 16K even if 32K is the
  theoretical max.

### Two-pass workflow

#### Pass 1: Segmentation

Before classifying, segment the novel into analytically meaningful units. Do NOT
chunk at arbitrary token boundaries.

**Segmentation strategies by author:**

- **Joyce (*Ulysses*)**: Episodes provide natural top-level divisions. Within episodes,
  segment at shifts in narrative technique (e.g., the transition from initial style to
  gigantism in "Cyclops," or the shift between characters in "Wandering Rocks").
  The Penelope episode (Molly's monologue) can be segmented by the eight "sentences."

- **Faulkner (*The Sound and the Fury*, *As I Lay Dying*)**: Named sections/chapters
  are natural units. Within Benjy's section, italicized vs. roman text marks temporal
  shifts that may warrant separate chunks.

- **Woolf (*Mrs Dalloway*, *To the Lighthouse*)**: Harder — Woolf's transitions are
  fluid and mid-sentence. Segment at perspective shifts (Clarissa → Septimus →
  Peter Walsh), using character names as anchors. "Time Passes" in *To the Lighthouse*
  is a distinct section with different SoC characteristics.

- **Richardson (*Pilgrimage*)**: Individual volumes and chapter breaks provide structure.
  Segment within chapters at shifts between omniscient narration and Miriam's interiority.

**Model-assisted segmentation:**
You can use a lighter/faster model (Gemini Flash, Qwen3 8B) for Pass 1. The prompt
should ask the model to identify *where narrative mode shifts occur* without classifying
the modes. This is a simpler task that tolerates larger chunks and less precise models.

#### Pass 2: Classification with overlap

For each segment from Pass 1:

1. **Include context buffer.** Prepend ~500–1000 tokens from the end of the previous
   segment and append ~500–1000 tokens from the beginning of the next segment. This
   lets the model see what the passage is transitioning *from* and *to*.

2. **Include the skill instructions.** The classification procedure, taxonomy, and
   2–3 calibration examples from the reference corpus should be in the system prompt.
   Don't include the entire example corpus — select examples relevant to the author
   and likely techniques.

3. **Request structured output.** Ask for annotations in the format specified in the
   Output Format section. For passages that shift modes, request sentence-level labels.

4. **Track metadata.** Each annotation should include: chunk_id, novel, chapter/episode,
   character_POV, position_in_text (start/end word or token offsets), plus the
   classification fields.

#### Pass 3 (optional): Reconciliation

After classifying all chunks, run a reconciliation pass:

- Check that classifications at chunk boundaries are consistent (the end of chunk N
  should agree with the beginning of chunk N+1, since they share overlap).
- Flag passages where confidence was "low" for human review.
- Generate summary statistics: distribution of SoC types by chapter, character, and
  position in the novel.

### Cost and time estimates

Rough estimates for processing a full novel through Pass 2 (annotation):

| Novel              | ~Tokens | Chunks (20K) | API calls | Est. cost (Claude Sonnet) | Est. cost (Gemini Flash) |
|-------------------|---------|-------------|-----------|--------------------------|-------------------------|
| *Mrs Dalloway*    | 82K     | 4–5         | 4–5       | ~$0.50–1.00              | ~$0.05–0.10            |
| *As I Lay Dying*  | 74K     | 4–5         | 4–5       | ~$0.50–1.00              | ~$0.05–0.10            |
| *Sound and Fury*  | 139K    | 7–8         | 7–8       | ~$1.00–2.00              | ~$0.10–0.20            |
| *Ulysses*         | 346K    | 17–20       | 17–20     | ~$3.00–5.00              | ~$0.30–0.50            |

These are input-token-dominated costs. Output costs depend on annotation density.
Prices are approximate and based on early 2026 API pricing.

### Practical tip: start small

Before processing a full novel, validate the approach on a single well-known
chapter — e.g., the Penelope episode of *Ulysses*, the opening of *Mrs Dalloway*,
or Benjy's section of *The Sound and the Fury*. Compare the model's classifications
against your own and against Humphrey/Steinberg's. Adjust chunk size, prompt
wording, and calibration examples based on where the model diverges from
expert annotation.

### Using Chonkie for chunking

The `scripts/soc_chonkie.py` script wraps the [Chonkie](https://github.com/chonkie-inc/chonkie)
chunking library with SoC-specific defaults. Chonkie handles tokenization, sentence
splitting, overlap context, and JSON export out of the box.

**Install:**
```bash
pip install chonkie                # base: Token, Sentence, Recursive chunkers
pip install chonkie[semantic]      # adds SemanticChunker + embedding support
```

**Recommended chunkers for SoC work:**

| Chunker | Best for | Why |
|---------|----------|-----|
| `SentenceChunker` | **Default for classification (Pass 2)** | SoC transitions often happen between sentences. Respects these boundaries naturally. |
| `RecursiveChunker` | Texts with structural markers | Splits at chapter/episode headings first, then paragraphs, then sentences. Good for Faulkner (character sections) and structured editions of Joyce. |
| `SemanticChunker` | **Segmentation (Pass 1)** | Groups text by semantic similarity — mode transitions often correlate with semantic register shifts. Requires an embedding model but can surface natural breakpoints. |
| `SlumberChunker` | Small-scale validation | Uses an LLM to find meaningful boundaries. Most expensive but could directly identify SoC mode transitions. |

**Quick start with model presets:**
```bash
# List available presets
python scripts/soc_chonkie.py --list-presets

# Chunk for Claude Sonnet (20K chunks, 1K overlap)
python scripts/soc_chonkie.py mrs_dalloway.txt --model claude-sonnet --chunker sentence

# Chunk for Qwen3 8B (9K chunks, 600 overlap)  
python scripts/soc_chonkie.py mrs_dalloway.txt --model qwen3-8b --chunker sentence

# Chunk for Claude Opus (25K chunks — Mrs Dalloway fits in ~3 chunks)
python scripts/soc_chonkie.py mrs_dalloway.txt --model claude-opus --chunker recursive
```

**Pipeline mode** (chains chunking → overlap refinement → JSON export):
```bash
python scripts/soc_chonkie.py ulysses.txt --pipeline --model claude-sonnet
```

**Or use Chonkie directly in your own code:**
```python
from chonkie import Pipeline

# SoC-optimized pipeline for Claude Sonnet
doc = (
    Pipeline()
    .process_with("text")
    .chunk_with("sentence", tokenizer="gpt2", chunk_size=20000)
    .refine_with("overlap", context_size=1000)
    .export_with("json", file="novel_chunks.json")
    .run(texts=open("mrs_dalloway.txt").read())
)

for chunk in doc.chunks:
    print(f"Chunk: {chunk.token_count} tokens")
    print(f"  Context before: {len(chunk.context_before or '')} chars")
    print(f"  Text: {chunk.text[:80]}...")
```

**Advanced: semantic segmentation for Pass 1:**
```python
from chonkie import SemanticChunker

# Use semantic similarity to find natural breakpoints
# where narrative mode is likely shifting
chunker = SemanticChunker(
    tokenizer="gpt2",
    chunk_size=20000,
    threshold="auto",  # auto-detect similarity threshold
)
segments = chunker(novel_text)

# Then classify each segment in Pass 2 using SentenceChunker
# at a finer grain if needed
```

The `scripts/soc_chunker.py` (custom, no dependencies) is also available as a
lightweight alternative if you prefer not to install Chonkie.

## Considerations for Computational Classification

If building a dataset or training a model for automatic SoC classification:

- **Passage boundaries matter.** SoC techniques often shift mid-paragraph. Sentence-level
  or clause-level annotation may be more useful than paragraph-level labels.
- **Context dependency.** Some classifications require knowing what comes before and after.
  A parenthetical aside only registers as an orthographic consciousness marker if
  the surrounding text is in a different narrative mode.
- **Author-specific conventions.** Joyce, Woolf, Faulkner, and Richardson each have
  distinctive signatures within the same category. A model trained only on Joyce's
  direct interior monologue may not recognize Richardson's. Include cross-author
  examples in training data.
- **Labeling granularity.** Consider whether your task needs the full taxonomy or a
  simplified version. For some purposes, a binary "narrator-mediated vs. unmediated"
  distinction is sufficient.
