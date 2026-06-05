"""System prompt constants for training and inference.

TRAINING_SYSTEM_PROMPT  — includes condensed taxonomy (~600-800 tokens).
                           Used in build_dataset.py to construct training examples.

INFERENCE_SYSTEM_PROMPT — schema only, no taxonomy.
                           Used in analyze.py for production inference.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output schema (shared by both prompts)
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA = """\
## Output format

Return ONLY valid JSON (no markdown fences) matching this schema:

{
  "instances": [
    {
      "is_soc": true,
      "passage": "<exact quoted text>",
      "soc_type": "<type>",
      "secondary_devices": ["<device>", ...],
      "affective_register": "<emotion or n/a>",
      "narrator_position": "absent | minimal | present | dominant",
      "character_pov": "<character name or empty>",
      "explanation": "<1-2 sentence reasoning>",
      "evidence": ["<textual feature 1>", "<textual feature 2>"],
      "confidence": "high | medium | low",
      "notes": "<optional ambiguity notes>"
    }
  ]
}

Return {"instances": []} if the passage contains no stream-of-consciousness.
Every field is required; use empty string or empty list when not applicable.
"""

# ---------------------------------------------------------------------------
# Condensed taxonomy (training only, ~600-800 tokens)
# ---------------------------------------------------------------------------

_TAXONOMY = """\
## Stream-of-consciousness taxonomy

Classify passages using the PRIMARY type that best fits. Assign additional
techniques in secondary_devices.

1. direct_interior_monologue
   Key markers: first-person present-tense unmediated thought; no narrator
   framing ("he thought"); sentence fragments, dash interruptions, immediate
   sensory-thought blend; character's raw voice without grammatical smoothing.

2. indirect_interior_monologue
   Key markers: third-person syntax with interior focalization; free indirect
   discourse—narrator merges with character voice; past tense but thought-like
   rhythm; "she wondered", "wasn't it odd that"; narrator hovers invisibly.

3. omniscient_description
   Key markers: narrator articulates character psychology with authority;
   interpretive rather than raw; "He was the kind of man who…"; analytic
   distance; reader sees mind from outside but deeply.

4. soliloquy  [DEPRECATED → other_soc if below 30 examples]
   Key markers: character speaks aloud alone; theatrical address; rhetorical
   questions to self; dramatic apostrophe; structured argument.

5. free_association
   Key markers: logic-defying leaps; sound or image chaining over sense;
   single-sentence paragraph bursts; word association sequences; abrupt topic
   shifts without transition.

6. space_montage  [DEPRECATED → other_soc if below 30 examples]
   Key markers: spatial rather than temporal juxtaposition; place-triggered
   consciousness; geography as memory map; topography structures thought.

7. orthographic_marker
   Key markers: typography encodes mental state — italics, ellipsis, ALL-CAPS,
   em-dashes, unconventional spacing, absent punctuation, mid-sentence breaks.

8. imagery
   Key markers: extended metaphor or sensory image that IS thought, not
   describes it; synesthesia; the image carries the entire cognitive event;
   no expository gloss.

9. simulation_state_of_mind
   Key markers: renders emotional/perceptual state rather than propositional
   thought; reader feels the affect directly; bodily sensation as narrative
   medium; affect register prominent.

10. reverie_fantasy
    Key markers: daydream / memory / wish diverging from present action;
    conditional or hypothetical mood ("if only…", "she imagined…");
    temporal drift; shift from present to recalled/imagined scene.

11. hybrid
    Key markers: TWO OR MORE primary types inseparably fused in a single
    passage; neither type alone is sufficient; annotate both in secondary_devices.

12. other_soc
    Key markers: clearly interior but resists the above categories; explain in
    notes; includes rare soliloquy and space_montage instances.
"""

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT: str = (
    "You are a literary analyst specialising in stream-of-consciousness "
    "(SoC) narrative technique. Analyse the supplied text and identify ALL "
    "passages that use SoC technique.\n\n"
    + _TAXONOMY
    + "\n"
    + _OUTPUT_SCHEMA
)

INFERENCE_SYSTEM_PROMPT: str = (
    "You are a literary analyst specialising in stream-of-consciousness "
    "(SoC) narrative technique. Analyse the supplied text and identify ALL "
    "passages that use SoC technique.\n\n"
    + _OUTPUT_SCHEMA
)

# ---------------------------------------------------------------------------
# Judge prompt (used by scripts/train_grpo.py reward function)
# ---------------------------------------------------------------------------

JUDGE_PROMPT: str = """\
You are an expert evaluator of literary interpretation quality.
You will score a model's analysis of a literary passage against an interpretive rubric.

Score conservatively. A trace that names a SoC type without arguing for it scores 0.2
on type_coherence, not 0.5. Reward genuine deliberation, not superficial pattern-matching.

## Taxonomy keyword reference (for specificity scoring)
direct_interior_monologue, indirect_interior_monologue, omniscient_description,
soliloquy, free_association, space_montage, orthographic_marker, imagery,
simulation_state_of_mind, reverie_fantasy, hybrid, other_soc

## Scoring criteria

**grounding** (0.0–1.0): Does the <think> trace cite at least one verbatim phrase
that appears word-for-word in the PASSAGE and that drives the is_soc verdict?
- 1.0 = yes, phrase is present in passage and clearly motivates the verdict
- 0.5 = phrase is paraphrased or partially matches
- 0.0 = no verbatim citation, or citation does not appear in passage

**skepticism** (0.0–1.0): Does the trace engage the skepticism gate?
- If is_soc=true: does it explain what specifically marks this as interior consciousness
  rather than conventional narration? (1.0 = yes, 0.0 = no)
- If is_soc=false: does it articulate what the passage would need to qualify as SoC?
  (1.0 = yes with specific criterion named, 0.5 = vague, 0.0 = no)

**specificity** (0.0–1.0): Does the trace name a specific taxonomy type (from the list above)?
- 1.0 = specific type named and used to frame the argument
- 0.5 = type named but only in passing
- 0.0 = only generic terms like \u201cstream of consciousness\u201d used

**type_coherence** (0.0–1.0, only meaningful when is_soc=true):
- 1.0 = trace argues FOR the assigned soc_type with textual evidence AND considers
  at least one alternative and explains why the chosen type is stronger
- 0.7 = argues for assigned type but does not consider alternatives
- 0.4 = assigned type mentioned but argument is thin or circular
- 0.2 = type named without argument
- 0.0 = JSON type contradicts or is unrelated to the <think> trace

## Input

PASSAGE:
{passage}

THINK TRACE:
{think_trace}

JSON OUTPUT:
{json_output}

## Output

Return ONLY valid JSON with no markdown fences:
{{
  "grounding": <float 0.0–1.0>,
  "skepticism": <float 0.0–1.0>,
  "specificity": <float 0.0–1.0>,
  "type_coherence": <float 0.0–1.0>,
  "rationale": "<one sentence explaining the type_coherence score>"
}}
"""
