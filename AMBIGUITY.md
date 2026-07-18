# Ambiguity: Training v3 on Model Disagreement

## The core idea

Every previous version of this model has been trained on consensus. The current `consensus.py` pipeline is explicitly a filter: it discards examples where frontier models disagree and keeps only those where they converge. The conservative track requires full agreement on type; the moderate track requires a majority. The result is a clean dataset of unambiguous cases — and a model that has never learned to recognise uncertainty in the text itself.

v3 inverts this. The discarded disagreements are not noise. They are the most interesting data we have.

**A productive boundary case** shows *systematic* disagreement: model A consistently reads a passage as `indirect_interior_monologue`, model B consistently reads it as `omniscient_description`, and each is internally coherent across resamples. That is not one model being wrong. That is two defensible readings of the same passage — which is our signal that the typology's boundary between those types is under genuine pressure at that point in the text.

The v3 goals:
1. **Lower false positives.** The model should be genuinely uncertain about passages that multiple strong readers disagree on, rather than defaulting to a positive SoC prediction. However, that both examples identify the passage as SoC is still a strong signal that it is SoC, but that the type is ambiguous.
2. **Surface boundary cases.** For a given passage, the model should be able to report not just a type but the *shape of the disagreement* — which readings are defensible and why.

---

## Corpus analysis findings (from `soc_type_analysis.ipynb`)

Before designing the resampling strategy, we ran a type distribution analysis across all works in `silver_data/` and `training_data/`. The key findings are recorded here to inform pipeline priorities.

### Cross-model agreement is uniformly low — and that is good news

Using the same token-overlap grouping logic as `consensus.py` (MIN_OVERLAP_RATIO = 0.35), we computed the fraction of multi-model passage groups where all three silver models assigned the same `soc_type`:

| Work | groups | multi-model | agree % |
|---|---|---|---|
| Richardson: Backwater | 25 | 13 | **7.7%** |
| Richardson: Pointed Roofs | 80 | 37 | **8.1%** |
| Faulkner: As I Lay Dying | 92 | 44 | 11.4% |
| Woolf: Orlando | 153 | 88 | 12.5% |
| Woolf: The Voyage Out | 198 | 98 | 14.3% |
| Faulkner: Sound & Fury | 152 | 83 | 14.5% |
| Joyce: Portrait | 190 | 100 | 16.0% |
| Joyce: Dubliners | 111 | 55 | 16.4% |
| Richardson: Honeycomb | 113 | 55 | 18.2% |
| Sinclair: Mary Olivier | 102 | 34 | 20.6% |
| Woolf: Between the Acts | 155 | 77 | 22.1% |
| Woolf: Jacob's Room | 172 | 89 | 23.6% |
| Woolf: The Waves | 208 | 93 | 24.7% |
| Joyce: Ulysses | 732 | 312 | 25.3% |
| Woolf: The Years | 293 | 163 | **26.4%** |

Agreement never exceeds 27%. The *vast majority* of multi-model passage groups already show disagreement in the single-pass data. This resolves the "how many boundary cases exist" open question: the answer is most of them. The resampling step is necessary to separate systematic from random disagreement, but the raw material for a large ambiguity dataset already exists.

**Implication for the pipeline:** do not start with a subset of the data. Run resampling across the full positive corpus. The priority order for resampling (most disagreement first) is: Richardson > Faulkner > Woolf > Joyce > Sinclair.

**Implication for the dataset mix:** the proposed 60/25/15 split (clean SoC / negative / boundary) was based on an unknown quantity of boundary cases. Given that ~75–92% of multi-model groups are disagreements, even after resampling filters out random disagreement, boundary cases will likely outnumber clean consensus cases. The mix should be decided empirically after `build_ambiguity_dataset.py` produces `ambiguity_index.jsonl`. A starting point of 40/30/30 is more realistic, with clean-consensus cases potentially becoming the scarce resource.

### The most contested typological boundary is indirect vs. omniscient

From the author-level type distributions:

| Author | indirect % | omniscient % |
|---|---|---|
| Joyce | 27.6% | 3.7% |
| Woolf | 44.8% | 11.0% |
| Richardson | 42.1% | 15.8% |
| Sinclair | 24.9% | 16.5% |
| Faulkner | 14.1% | 4.0% |

Richardson and Sinclair have the highest `omniscient_description` fractions, and Richardson also has the lowest cross-model agreement. The `indirect_interior_monologue` / `omniscient_description` boundary is the most structurally ambiguous pair in this corpus: both involve third-person syntax and character interiority; the distinction depends on whether the narrator or the character is felt as the source of interpretive language. This is precisely where typological definitions are most likely to generate systematic disagreement between models.

When inspecting `ambiguity_index.jsonl` after resampling, expect this pair to dominate the `type_split` distribution for Richardson and Woolf.

### Mrs Dalloway is a Woolf outlier — and absent from silver_data

`Mrs_Dalloway.txt` is only present in `training_data/` (1,125 deduped records, 4 models including gemini-3-pro-preview and original gpt-5) and nearly absent from `silver_data/` (1 record, filename mismatch). Its type profile:

| type | % |
|---|---|
| indirect | **72.5%** |
| omniscient | 17.5% |
| direct | 3.8% |

This makes *Mrs Dalloway* the most extreme indirect-dominant work in the corpus — more than any Richardson or Woolf work in `silver_data/`. It should be added to the silver pipeline under the correct filename (`pg05765_mrs_dalloway.txt`) and re-run. Until then, training data for Dalloway comes from a different model set than the rest of the corpus, which is a source of inconsistency.

*To the Lighthouse* is also absent from `silver_data/` and should be added at the same time.

### Richardson's Pilgrimage texts are the primary source of boundary cases

All three Pilgrimage volumes (Pointed Roofs, Backwater, Honeycomb) have agreement rates of 8–18% — the lowest in the corpus — and a distinctive type profile dominated by `indirect_interior_monologue` (42–49%) and `simulation_state_of_mind` (14–19%). The `simulation_state_of_mind` type is itself a contested boundary: it describes passages where the syntax and rhythm enact an emotional state rather than representing a thought event, which frequently overlaps with how different models read `indirect_interior_monologue`. Richardson is therefore the most important author for the ambiguity dataset both in quantity and in kind.

---

## What the existing silver data contains

The silver data in `silver_data/` has one pass per frontier model (currently gpt-5.4, mercury-2, qwen3-max) on each chunk of the positive corpus. The consensus tracks in `consensus.yaml` currently treat cross-model disagreement in one of two ways:

- **Discard** (conservative/moderate): passages where models disagree on type are filtered out.
- **Keep all rows** (liberal, `resolve_type: all`): every model's row is retained, but the disagreement structure is not represented — the rows are independent records, not a linked comparison.

Neither treatment records *which* models disagreed, *what* they each said, or *how consistently* each held its position. That metadata is currently lost.

The key addition needed before v3 training can begin is a **disagreement extraction step** that recovers this structure from the raw per-model JSON files.

---

## Distinguishing systematic from random disagreement

Not all disagreement is equal. Two sources of disagreement need to be separated:

**Random / noisy disagreement:** Model A labels a passage `indirect_interior_monologue` on one run and `omniscient_description` on another. The model is uncertain — its reads are unstable. This is not a productive boundary case; it is a hard example that the current typology does not anchor well, and training on it risks teaching the model to be randomly inconsistent.

**Systematic disagreement:** Model A *consistently* labels the passage `indirect_interior_monologue` across resamples. Model B *consistently* labels it `omniscient_description` across resamples. Each model has a stable, coherent reading. The disagreement is between two positions, not within one uncertain model. This is the gold.

### Operationalising the distinction

The existing silver data has only one pass per model, so within-model consistency cannot be measured from what we have. Producing a proper v3 dataset requires re-running each frontier model k=3–5 times on each passage at non-zero temperature and recording all outputs.

For each passage and each frontier model, compute:
- **Within-model consistency score**: the mode `soc_type` across k runs, and the fraction of runs that match the mode. A model with ≥ 80% of runs returning the same type is considered *stable* on this passage.
- **Cross-model divergence**: among the stable models, how many distinct mode types appear? If all stable models converge → clean case. If stable models split between two types → systematic disagreement → candidate for the ambiguity dataset.

A passage qualifies as a **productive boundary case** if:
1. At least two models are stable (within-model consistency ≥ 0.8).
2. The stable models split between exactly two type assignments (not three or more — that is just chaos).
3. Each side of the split has at least one stable model.

### New `consensus.yaml` track

Add a new track `ambiguous` to `consensus.yaml`:

```yaml
ambiguous:
  agreement: systematic_disagreement   # new mode (see below)
  min_models_per_side: 1               # at least one model stably on each side
  min_within_model_consistency: 0.8    # fraction of resamples that match the mode
  max_type_split: 2                    # reject if more than two distinct types
  output_formats: [jsonl]
```

This requires a new `agreement` mode in `consensus.py` / `scripts/consensus.py` that reads the resampled per-model outputs rather than single-pass outputs.

---

## What the training example looks like

A clean consensus example has a single `soc_type` label. A productive boundary case needs a different representation: it should encode both readings, why each is defensible, and that the correct answer is to *hold both* rather than choose.

Proposed training format for ambiguous examples:

```json
{
  "messages": [
    {"role": "system", "content": "<taxonomy system prompt>"},
    {"role": "user", "content": "<passage text>"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n{\n  \"is_soc\": true,\n  \"ambiguous\": true,\n  \"readings\": [\n    {\n      \"soc_type\": \"indirect_interior_monologue\",\n      \"confidence\": \"high\",\n      \"evidence\": \"...\",\n      \"rationale\": \"...\"\n    },\n    {\n      \"soc_type\": \"omniscient_description\",\n      \"confidence\": \"high\",\n      \"evidence\": \"...\",\n      \"rationale\": \"...\"\n    }\n  ],\n  \"boundary\": \"The passage sits at the typological boundary between indirect interior monologue and omniscient description. The third-person syntax and interpretive distance admit both readings.\"\n}"}
  ]
}
```

The key fields:
- `ambiguous: true` — explicit signal to downstream consumers that this is a boundary case.
- `readings` — an ordered list of the two defensible interpretations, each with its own evidence and rationale.
- `boundary` — a sentence that characterises *where* in the typology the boundary falls and *what* feature of the passage generates the ambiguity.

For the `<think>` block, the model should reason through both positions rather than arguing toward one. The training traces for ambiguous examples should model a different deliberative pattern: *here is what makes type A compelling; here is what makes type B equally compelling; here is why no further evidence in the passage resolves the question.*

---

## How this addresses the false-positive problem

The current model defaults to `is_soc: true` because every training example is either a confirmed SoC passage or a clean negative. The model has never seen a passage where the right answer is "this is SoC, but I genuinely cannot tell which kind, and that uncertainty is the correct response."

Training on ambiguous cases teaches three things the model currently lacks:

1. **Typological humility.** Not every passage resolves to a single type. Forcing a choice where none exists is a form of false confidence.

2. **Calibrated positive detection.** If a passage generates systematic disagreement between two SoC types, it is clearly SoC — but the ambiguity is informative. If a passage generates systematic disagreement between a SoC type and `none`, that is a different signal: it may be a false positive, and the model should be encouraged to return `is_soc: false` with a rationale.

3. **Useful output for downstream research.** A corpus-scale run that flags passages as `ambiguous: true` with their competing readings is more useful to a literary scholar than a run that forces a single type label on every passage.

---

## Pipeline changes

### Step 1: Resample the silver models

For each passage in the positive corpus, re-run each of the three frontier models k=5 times at temperature ≥ 0.7. Store all runs in a new directory `silver_data/resampled/<model_name>/<passage_id>.jsonl`.

This is the most expensive step. At k=5 and ~4,762 training passages and 3 models, it requires ~71,430 model calls. With batching this is feasible but not cheap. Consider starting with a subset: run resampling only on the passages that `consensus.py --track liberal` currently marks as disagreements (i.e., the ones the conservative track has already discarded). This is the highest-priority subset for ambiguity labelling.

### Step 2: Compute consistency scores

`scripts/build_ambiguity_dataset.py`:
- Load all resampled runs for each passage.
- Compute within-model consistency (mode type + fraction).
- Identify stable models (consistency ≥ 0.8).
- Classify each passage as: clean-consensus / clean-negative / productive-boundary / noisy (three or more types among stable models).
- Write a summary to `silver_data/ambiguity_index.jsonl` with one record per passage: `{passage_id, clean_types, boundary_types, noisy, stable_models, unstable_models}`.

### Step 3: Generate training traces for boundary cases

For productive-boundary passages, generate the training-format assistant turn (above). This requires a frontier model to write both the `<think>` block and the dual-reading JSON, given:
- The passage text.
- The two competing type assignments from the stable models.
- The taxonomy definitions for those two types.

This is a new trace-generation task, distinct from the single-reading traces in `scripts/generate_traces.py`. Create `scripts/generate_ambiguity_traces.py` for this.

### Step 4: Mix into the training dataset

The v3 training set is a mixture:
- **Clean SoC examples** (from conservative/moderate consensus tracks) — teaches confident recognition.
- **Clean negative examples** (from `negatives/`) — teaches genuine non-SoC.
- **Productive boundary cases** (from ambiguity dataset) — teaches calibrated uncertainty and dual-reading output.

Recommended *starting point* mix: 60% clean SoC / 25% clean negative / 15% boundary. However, given that cross-model agreement is 7–26% across all works (see *Corpus analysis findings*), clean-consensus cases may turn out to be the scarce resource after resampling. Recompute the available counts from `ambiguity_index.jsonl` before fixing the final mix; a 40/30/30 split (clean SoC / negative / boundary) may better reflect what the data can support.

The `build_dataset.py` script needs a `--ambiguity` flag that adds the boundary examples from `silver_data/ambiguity_index.jsonl`.

### Step 5: Update the evaluator

The current `scripts/evaluate.py` measures binary SoC F1 and per-class type F1. For v3 it needs an additional metric:

**Ambiguity precision/recall:** For passages in the test set that were labelled as productive boundary cases, how often does the model return `ambiguous: true`? How often does it falsely return `ambiguous: true` on clean cases?

This requires a test split that includes some labelled boundary cases. Reserve 10% of the productive-boundary passages for the test set before building the training dataset.

---

## What success looks like

A successful v3 model:
- Maintains binary SoC recall at or above v2 (≥ 98%).
- Improves binary SoC precision — fewer false positives — by at least 5 pp over v2.
- Returns `ambiguous: true` with two coherent readings on ≥ 70% of held-out productive-boundary cases.
- Returns `ambiguous: false` on ≥ 90% of held-out clean-consensus cases (low false-ambiguity rate).

The last two metrics are new and have no v1/v2 baseline. They define the capability that is genuinely new in v3.

---

## Open questions

**How many boundary cases exist in the current data?** The cross-model type agreement analysis (see *Corpus analysis findings* above) shows that agreement across the three silver models is 7–26% across all works — meaning the majority of multi-model passage groups already show disagreement in the single-pass data. The boundary case pool is large. The resampling step is needed to separate systematic from random disagreement, but the raw material exists at scale. Prioritise Richardson for resampling first (lowest agreement rates). After resampling and running `build_ambiguity_dataset.py`, revisit the 60/25/15 dataset mix — boundary cases will likely be abundant enough to support a 40/30/30 split.

**What is the right output schema for boundary cases?** The dual-reading JSON above is a proposal. It may need iteration — in particular, the `boundary` field requires the model to characterise the typological fault line, which is a harder generation target than a type label. An alternative is to keep `boundary` as an optional field and accept its absence without penalty in early training runs. Include a step in our process to revisit this issue and evaluate the quality of the `boundary` explanations. If they are consistently poor, we may need to simplify the output schema for v3 and defer this feature to a later version.

**Does the ambiguity concept extend to `is_soc` itself?** The productive-boundary definition above assumes both competing readings are SoC types. But there will also be passages where one stable model returns SoC and another returns `none`. These are a different kind of boundary case — not a typological ambiguity but a detection threshold ambiguity. They deserve their own treatment and are likely the primary driver of the false-positive problem. Flag these separately as `detection_boundary` cases in `ambiguity_index.jsonl`.

---

## Appendix: schemas, interfaces, and passage grouping

### A. Resampled run file schema

Each file `silver_data/resampled/<model_name>/<chunk_id>.jsonl` contains one JSON object per line, one line per resample run:

```json
{
  "chunk_id": "pg04300_ulysses_ch03",
  "run_index": 0,
  "model_label": "gpt-5.4",
  "source_file": "pg04300_ulysses.txt",
  "passage": "<exact quoted text>",
  "soc_type": "indirect_interior_monologue",
  "secondary_devices": ["imagery"],
  "confidence": "high",
  "explanation": "...",
  "evidence": ["phrase 1", "phrase 2"],
  "temperature": 0.8
}
```

`chunk_id` is the primary key and must match the `chunk_id` field in `silver_data/*.json`. `run_index` is 0-based within the k runs for that model/passage pair. `temperature` is recorded so that later analysis can filter by sampling conditions.

### B. Passage grouping across models

The existing codebase does **not** use `chunk_id` as a hard key for cross-model passage matching. `scripts/consensus.py` groups passages by token-overlap similarity using `_token_overlap(a, b) = |a ∩ b| / min(|a|, |b|)`, with `MIN_OVERLAP_RATIO = 0.35` (i.e., 35% of the smaller passage's tokens must overlap). Passages with normalised text overlap ≥ 0.35 are considered the same span.

`scripts/build_ambiguity_dataset.py` must use the same logic when joining single-pass records across models to form groups. Import or replicate `_normalise_text` and `_token_overlap` from `scripts/consensus.py`. Within each group, use the `chunk_id` from whichever model produced it; if `chunk_id` values differ across models for the same span, use the lexicographically first.

For the resampled data, `chunk_id` is written explicitly during resampling (read from the original silver JSON), so the fuzzy match is only needed when working from single-pass data.

### C. `ambiguity_index.jsonl` schema

One record per passage group that passes the productive-boundary filter:

```json
{
  "chunk_id": "pg04300_ulysses_ch03",
  "source_file": "pg04300_ulysses.txt",
  "passage": "<exact quoted text>",
  "boundary_type": "typological | detection",
  "stable_models": ["gpt-5.4", "qwen3-max"],
  "unstable_models": ["mercury-2"],
  "type_split": {
    "indirect_interior_monologue": ["gpt-5.4"],
    "omniscient_description": ["qwen3-max"]
  },
  "within_model_consistency": {
    "gpt-5.4": 0.8,
    "qwen3-max": 1.0,
    "mercury-2": 0.4
  },
  "noisy": false,
  "split": "train"
}
```

`boundary_type` is `"typological"` when both competing type assignments are SoC types, and `"detection"` when one side is `"none"` (a detection threshold disagreement). `noisy` is `true` when stable models disagree on three or more types — these are excluded from training but retained in the index for analysis. `split` is assigned at index-build time by reserving 10% of records (stratified by `boundary_type`) for test.

### D. V3 unified output schema

The v3 system prompt in `scripts/prompts.py` must support both clean and ambiguous instances in the same response. Extend `_OUTPUT_SCHEMA` to add optional fields on each instance:

```json
{
  "instances": [
    {
      "is_soc": true,
      "passage": "<exact quoted text>",
      "soc_type": "indirect_interior_monologue",
      "ambiguous": false,
      "readings": null,
      "boundary": null,
      "secondary_devices": ["imagery"],
      "affective_register": "melancholic",
      "narrator_position": "minimal",
      "character_pov": "Stephen",
      "explanation": "...",
      "evidence": ["phrase 1"],
      "confidence": "high",
      "notes": ""
    },
    {
      "is_soc": true,
      "passage": "<exact quoted text>",
      "soc_type": null,
      "ambiguous": true,
      "readings": [
        {
          "soc_type": "indirect_interior_monologue",
          "confidence": "high",
          "evidence": "phrase supporting this reading",
          "rationale": "..."
        },
        {
          "soc_type": "omniscient_description",
          "confidence": "high",
          "evidence": "phrase supporting this reading",
          "rationale": "..."
        }
      ],
      "boundary": "The third-person syntax and interpretive distance admit both readings.",
      "secondary_devices": [],
      "affective_register": "n/a",
      "narrator_position": "present",
      "character_pov": "",
      "explanation": "",
      "evidence": [],
      "confidence": "high",
      "notes": ""
    }
  ]
}
```

Rules:
- When `ambiguous: false`, `soc_type` is required, `readings` and `boundary` are `null`.
- When `ambiguous: true`, `soc_type` is `null`, `readings` has exactly two entries, `boundary` is a non-empty string.
- `parse_response()` in `evaluate.py` must be updated to handle both cases: extract `soc_type` directly for clean instances; for ambiguous instances, extract both types from `readings` and record `ambiguous: true` alongside them.

### E. CLI interfaces for new scripts

**`scripts/build_ambiguity_dataset.py`**

```
python scripts/build_ambiguity_dataset.py \
    --resampled-dir silver_data/resampled/ \
    --silver-dir    silver_data/ \
    --output        silver_data/ambiguity_index.jsonl \
    [--consistency-threshold 0.8]   # min fraction of runs matching mode type
    [--max-type-split 2]            # reject if > N distinct stable types
    [--test-ratio 0.10]             # fraction of boundary cases held for test
    [--seed 42]
```

When `--resampled-dir` is absent or empty, falls back to single-pass data with consistency threshold forced to 1.0 (trivially stable), producing an upper-bound estimate of cross-model disagreements only.

**`scripts/generate_ambiguity_traces.py`**

```
python scripts/generate_ambiguity_traces.py \
    --index   silver_data/ambiguity_index.jsonl \
    --output  training_data/ambiguity_traces/ \
    --model   <label-from-models.yaml> \
    [--split  train|test|all]        # default: train
    [--candidates 4]                 # traces generated per passage
    [--temperature 0.8]
```

For each passage in the index (filtered by `split`), calls the frontier model `--candidates` times with a prompt that includes the passage text, the two competing type definitions from `_TAXONOMY`, and the dual-reading trace format. Scores each candidate on grounding (automatic) and selects the highest-scoring trace. Writes one file per `chunk_id` to `--output`.
