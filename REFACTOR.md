# Penelope — Fine-Tuned Model Refactor

## Overview

The current pipeline sends literary chunks to external LLM APIs
(OpenAI, Gemini, Qwen, etc.) for SoC detection and classification.
This refactor replaces those external calls with a small, locally-hosted
reasoning model fine-tuned specifically on the SoC classification task.

**Goals:**
- Eliminate dependency on paid external APIs and network availability.
- Produce a domain-specific model that has internalized the SKILL.md taxonomy.
- Enable reproducible, versioned inference: the model *is* the annotation logic.
- Reduce per-passage latency and cost for large corpora.
- Generate a publishable, citable artifact (the fine-tuned model and dataset).

---

## Current Architecture (to be replaced)

```
Chunk → prompt construction → OpenAI-compatible API → JSON parse → results
```

- `scripts/analyze.py` — constructs prompts, calls external API, parses responses.
- `models.yaml` — profiles for each external LLM backend.
- `run.py` — orchestrates chunk → model → results.
- `training_data/` — silver-label outputs from four large LLMs (GPT-5, Gemini,
  Qwen3-max, Mercury-2). These become the foundation for the training set.

---

## New Architecture

```
Chunk → local inference (fine-tuned SoC model) → structured output → results
```

The fine-tuned model accepts a passage as input and returns a structured
interpretation: whether SoC is present, the primary type, supporting evidence,
and a confidence level. No external API call; no prompt engineering at runtime.

---

## Phase 1 — Dataset Construction

### 1.1 Silver-label consolidation

The four `training_data/*.json` files (one per large LLM) are the starting point.
`build_dataset.py` reads these files directly — the existing `consensus.py` script
globs `results/*.json` and cannot be run as-is against `training_data/`. The merging
logic is ported into `build_dataset.py` with `training_data/` as the explicit source path.
Tracks:
- **Conservative** — all four models agree on `soc_type`.
- **Moderate** — three of four agree.
- **Liberal** — majority vote, with disagreement fields preserved.

Output: `dataset/silver_consensus.jsonl`

Each record should carry:
```json
{
  "text": "<passage>",
  "label": "<soc_type or null>",
  "is_soc": true,
  "secondary_devices": [],
  "narrator_position": "minimal",
  "confidence": "high",
  "agreement_track": "conservative",
  "source_models": ["gpt-5", "qwen3-max", "gemini-3-pro-preview", "mercury-2"]
}
```

### 1.2 Negative examples

The current dataset contains only positive SoC instances (LLMs were prompted
to find SoC). Add **non-SoC** passages drawn from the same source texts.
Two categories:

- **Clean negatives** — chunks where *no* model identified any SoC. Label
  `is_soc: false`, `label: null`.
- **Hard negatives** — chunks where exactly one model found SoC and the other
  three did not. These near-miss passages are the most informative negatives
  for training. Label `is_soc: false` (below the conservative threshold) but
  flag with `"hard_negative": true` so they can be up-weighted during training
  or analysed separately in evaluation.

Target ratio: roughly 1:2 positive to negative to reflect real-world base rates.

### 1.3 Human review pass (recommended before training)

Run inter-annotator agreement (IAA) on a stratified sample (~150–200 passages)
using Cohen's κ or Krippendorff's α across the `soc_type` field. Categories
with κ < 0.6 should be revised or collapsed before training. Document results
in `dataset/annotation_notes.md`.

### 1.4 Dataset splits

| Split | Size (target) | Notes |
|---|---|---|
| Train | ~1,600 | Stratified by `soc_type` and source work |
| Validation | ~200 | Same stratification |
| Test | ~200 | Held out; never seen during training or tuning |

Minimum per-class count: 30 instances. Collapse `soliloquy` and `space_montage`
into `other_soc` if below threshold (see Phase 2.1 rationale).

---

## Phase 2 — Model: `Qwen/Qwen3-4B`

### 2.1 Selected model

**`Qwen/Qwen3-4B`** — 4.0B parameters (3.6B non-embedding), Apache 2.0 license,
available on Hugging Face.

Rationale:
- The Qwen3-max variant produced the silver labels for this corpus; fine-tuning
  the 4B version on Qwen-derived data minimises distribution mismatch.
- Native 32,768-token context window (extendable to 131,072 via YaRN) — sufficient
  for full chapter-level chunks without truncation.
- Uniquely supports **both** thinking mode (`<think>…</think>` reasoning blocks
  before the final answer) and non-thinking mode within the same weights, switchable
  at inference time. This is a key design driver for Phases 3 and 4.
- Strong instruction-following and structured-output capabilities out of the box.

Requires `transformers >= 4.51.0` (earlier versions raise `KeyError: 'qwen3'`).

### 2.2 Training approach

**Primary — Instruction fine-tuning (SFT)**
Format each training example as a three-turn chat (see Phase 3.2 for the full schema).
Fine-tune with **QLoRA** (4-bit base + LoRA rank 16–64) on the Qwen3 chat template.
Tooling: `transformers` + `trl` (`SFTTrainer`) + `peft`.

**Optional — DPO alignment**
After SFT, apply DPO to improve calibration on ambiguous passages. DPO requires
*same-model* preference pairs — cross-model silver disagreements alone are not
sufficient. Construct pairs by running the SFT checkpoint on passages below the
conservative threshold to generate multiple rollouts, then rank them by
majority-vote agreement with the silver data. The higher-ranked completion is
"chosen"; the lower-ranked is "rejected". Apply with `trl.DPOTrainer`. Optional
but valuable if downstream use depends on calibrated confidence scores.

---

## Phase 3 — Training Pipeline

### 3.1 New scripts

| File | Purpose |
|---|---|
| `scripts/build_dataset.py` | Merge silver JSONs → JSONL splits; add negatives |
| `scripts/train.py` | SFT training loop (SFTTrainer + LoRA config) |
| `scripts/evaluate.py` | Per-class F1, confusion matrix, IAA comparison |
| `scripts/export_model.py` | Merge LoRA adapter → base model; push to HF Hub |

### 3.2 Training record schema (JSONL)

Each record is a three-turn conversation. The **training** system prompt includes a
condensed taxonomy derived from `SKILL.md` — type names plus the key diagnostic
markers for each type. This is critical: without definitions in the prompt, the model
learns label-to-example associations but not what each label *means*, which produces
fragile generalisation. The full SKILL.md (scholarly discussion, Humphrey page
references, extended examples) is omitted to keep per-example token cost manageable;
the markers alone are sufficient and budget around 600–800 tokens.

At **inference time**, the condensed taxonomy is dropped from the system prompt
(the definitions are now in the model weights) and only the output schema is kept.
See Phase 4.1 for the inference prompt.

The `build_dataset.py` script should maintain the condensed taxonomy string as a
constant (e.g. `TRAINING_SYSTEM_PROMPT`) built from `SKILL.md`'s key-marker bullets,
so it stays in sync with the taxonomy if SKILL.md is updated.

The assistant turn uses Qwen3's native `<think>…</think>` format. The thinking
content is built from the silver data's `explanation` and `evidence` fields
concatenated into prose. The final JSON array follows immediately after `</think>`.
Loss is computed **only on tokens after `</think>`** — thinking tokens are masked
to `-100` in the data collator (see Phase 3.4).

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a literary analyst specialising in stream of consciousness (SoC) in modernist fiction. Given a text chunk, identify every SoC passage and return them as a JSON array. If none are present, return an empty array.\n\nTaxonomy (from Humphrey / Steinberg):\n- direct_interior_monologue: No narrator frame; no reporting verbs; reader dropped into the mind; syntax may fragment\n- indirect_interior_monologue: Third-person syntax but character's diction/content; light reporting cues ('thought X'); slides between narrator and character voice; overlaps with free indirect discourse when rendering pre-speech consciousness\n- omniscient_description: Narrator retains full authority; psycho-narration ('she felt', 'he considered'); consciousness described from outside, not rendered from inside\n- soliloquy: First-person address to self; more syntactically organised than direct monologue; rhetorical/emotional arc; character is 'making a case'\n- free_association: Non-logical links between thoughts via sensory, phonetic, or emotional contiguity; abrupt topic shifts; mechanism that can layer on any core type\n- space_montage: Rapid juxtaposition of multiple characters' perspectives on a shared external event; cinematic cutting between minds\n- orthographic_marker: Italics, parentheses, missing punctuation, or typographic disruption used to signal consciousness shifts or represent the stream's unbroken flow\n- imagery: Dense sensory language where subjective perception *is* the consciousness; boundaries between perceiver and perceived blur\n- simulation_state_of_mind: Syntax/rhythm/diction formally enacts an emotional state rather than merely describing it; form mimics the affect\n- reverie_fantasy: Character constructs an imagined scene (not recalled memory); future/conditional diction; sustained coherence distinct from free association\n- hybrid: Passage shifts between two or more of the above; note primary and secondary modes\n- other_soc: Recognised SoC technique not captured above (includes soliloquy and space_montage when below training-data threshold)\n\nOutput schema: {\"instances\": [{\"is_soc\": bool, \"soc_type\": str, \"passage\": str, \"secondary_devices\": [str], \"narrator_position\": str, \"character_pov\": str, \"confidence\": str, \"notes\": str}]}"
    },
    {
      "role": "user",
      "content": "Source: Mrs_Dalloway.txt — Chunk mrs-dalloway_000\n\n<full chunk text>"
    },
    {
      "role": "assistant",
      "content": "<think>\nThe passage uses third-person syntax but immediately adopts Clarissa's idiom. The reporting verb is minimal and yields at once to character-anchored diction. Sensory detail triggers associative memory — a hallmark of indirect interior monologue. Evidence: exclamatory interjection ('What a lark!'), sensory-triggered memory transition, non-linear time.\n</think>\n{\"instances\": [{\"is_soc\": true, \"soc_type\": \"indirect_interior_monologue\", \"passage\": \"And then, thought Clarissa Dalloway...\", \"secondary_devices\": [\"free_association\", \"imagery\"], \"narrator_position\": \"minimal\", \"character_pov\": \"Clarissa Dalloway\", \"confidence\": \"high\", \"notes\": \"\"}]}"
    }
  ]
}
```

Note that `soc_type` uses the collapsed taxonomy (see Phase 1.4): `soliloquy` and
`space_montage` are represented as `other_soc` in the training labels.

### 3.3 Structured output at inference

Constrained decoding tools (e.g. `outlines`) conflict with Qwen3's thinking mode
because the `<think>…</think>` block is free-form text — constraining generation
from token one would corrupt the reasoning trace. Use **post-generation parsing**
instead:

1. Strip everything up to and including `</think>` from the raw output string.
2. Extract the JSON substring that follows (from the first `{` to the last `}`).
3. Parse with `json.loads` and validate against the Pydantic schema.
4. On parse failure, log the raw output and return an empty `instances` list —
   fail-safe, not a crash.

For non-thinking mode (`--fast` flag), the same parsing logic applies; the model
emits no `<think>` block so step 1 is a no-op.

### 3.4 Loss masking for thinking tokens

Configure the `SFTTrainer` data collator to mask all tokens inside (and including)
the `<think>…</think>` span by setting their labels to `-100`. Only tokens from
the opening `{` of the JSON output through `</s>` contribute to the loss. This is
the standard approach for Qwen3 SFT: the model learns to produce correct JSON
outputs while the thinking content is guided by the pre-training signal already
in the base weights, keeping training efficient and the gradient signal clean.

### 3.5 Key dependencies

Add to `requirements.txt`:

```
transformers>=4.51.0   # required for Qwen3 chat template support
trl>=0.8.0
peft>=0.10.0
datasets>=2.19.0
torch>=2.2.0
```

---

## Phase 4 — Inference Integration

### 4.1 Replace `scripts/analyze.py`

The new `analyze.py` will:
1. Load the fine-tuned model once at startup (`transformers >= 4.51.0` required).
2. For each chunk, format the prompt using the **inference system prompt** —
   the output schema only, without the condensed taxonomy. After fine-tuning,
   the type definitions are in the model weights; re-including them at inference
   consumes context budget without adding accuracy. The inference prompt is
   maintained as `INFERENCE_SYSTEM_PROMPT` in `analyze.py`.
3. Generate output with `enable_thinking=True` (default) or `False` (`--fast`).
4. Strip the `<think>` block, parse the trailing JSON, validate against the
   Pydantic schema, and return results. Parse failures are logged; return `[]`.

External API logic (`openai` client, `models.yaml`, retry/rate-limit handling)
is removed. `models.yaml` is replaced by `model_config.yaml`.

### 4.2 Config changes

`models.yaml` → `model_config.yaml`:
```yaml
model:
  path: ./models/penelope-soc-v1        # local merged model
  # or: hf_id: apjanco/penelope-soc-v1  # HF Hub
  device: cuda                           # or: cpu, mps
  max_new_tokens: 4096          # chapters with multiple passages + think block
  enable_thinking: true          # set to false for --fast mode
  batch_size: 4
```

### 4.3 Hardware requirements

- **GPU (recommended):** Qwen3-4B in bf16 requires ~8 GB VRAM. With 4-bit
  quantization at inference time it runs in ~4–5 GB VRAM.
- **CPU fallback:** Export a GGUF-quantized version via `llama.cpp`. Q4_K_M
  quantization (~2.5 GB file) gives a practical balance of speed and accuracy
  for machines without a GPU.

---

## Phase 5 — Evaluation & Release

### 5.1 Metrics

- Per-class **precision, recall, F1** on held-out test split.
- **Macro-F1** as the primary headline metric.
- **Confusion matrix** — focus on which subtypes the model conflates
  (expected: `direct` vs. `indirect` interior monologue).
- Compare against: (a) majority-vote of the four original large LLMs on
  the test set, (b) zero-shot Qwen3-4B before fine-tuning.

### 5.2 Model card & HF Hub release

Publish the fine-tuned model to Hugging Face with:
- Model card documenting dataset, taxonomy, training config, and metrics.
- `SocInstance` JSON schema as the model's output format.
- `deploy_hf.sh` updated to push model weights and adapter.

### 5.3 Dataset release

Publish `dataset/silver_consensus.jsonl` (conservative track) as a
standalone HF dataset — the first annotated SoC corpus with multi-model
provenance.

---

## Migration Path

| Current component | Fate |
|---|---|
| `scripts/analyze.py` | Rewritten — local model inference replaces API calls |
| `scripts/models.py` | Updated — add `is_soc: bool` to `SocInstance`; rename `LLMResponse.soc_instances` → `instances`; add `OTHER_SOC` to `SocType` enum, deprecating `SOLILOQUY` and `SPACE_MONTAGE` |
| `models.yaml` | Replaced by `model_config.yaml` |
| `run.py` | Minimal changes — replaces model-loading logic only |
| `chunk.py`, `scripts/extract.py` | Unchanged |
| `consensus.py` | Logic ported into `scripts/build_dataset.py`; original script reads from `results/` and is not used in the training pipeline |
| `training_data/*.json` | Become training set inputs |
