# Passage-Grounded Teacher Gold for Penelope

*How to adapt the image-grounded teacher-gold method from the `nara_trainer`
project to fix Penelope's "learns the schema, not the typology" problem.*

status: design + implementation guide · companion to `POST-TRAINING.md` and `TRAINING_APPROACH.md`

---

## 0. The shared failure, stated precisely

Both projects hit the **same wall**, for the same reason.

`nara_trainer` trained a vision model to date archival page sequences. Its first
gold set stamped **one coarse record-level date on every page**. An SFT student
trained on it reached a beautiful training loss — and then evaluation showed:

| Diagnostic | Value | Meaning |
|---|---|---|
| `schema_valid` | 92% | the JSON *shape* was learned ✅ |
| `echoes_gold_reasoning` | 100% | it reproduced the gold's boilerplate reasoning ❌ |
| `all_same_date` | 100% | it copied one label onto every item ❌ |
| `distinct_evidence` | 30% | the "evidence" was boilerplate ❌ |

The verdict was: **the student learned the output template, not the reasoning.**
Low teacher-forced loss does not prove free-generation reasoning — the model was
rewarded for copying a schema, and the gold gave it nothing else to copy.

Penelope is in the identical position. Its silver labels are consensus JSON from
frontier APIs; the `<think>` trace is masked out during training
(`ThinkMaskingCollator`), and the `explanation`/`evidence` fields are short,
often generic, and correlated with class frequency. So the Qwen3-4B student
learns:

- the `{"instances": [...]}` shape (easy, always rewarded),
- a bias toward the majority `soc_type` (the label distribution),
- **not** the Humphrey/Steinberg *distinctions* — the discriminations between
  `direct_interior_monologue` and `indirect_interior_monologue`, between
  `imagery` and `simulation_state_of_mind`, that are the entire point.

The lesson `nara_trainer` learned the hard way is the one to import:

> **Do not train more epochs on schema-echo gold. Fix the gold first.**
> The cure is gold that *forces differentiated, source-grounded reasoning* —
> produced by a single strong teacher shown the primary source in the same
> representation the student sees.

---

## 1. What "image-grounded" becomes for a text project

The `nara_trainer` method has four load-bearing properties. None of them are
actually about *images* — they are about **grounding, differentiation, shared
representation, and schema-validated structure**. Here is the transposition.

| `nara_trainer` (vision) | Penelope (text) |
|---|---|
| Teacher is shown the **page images** in sequence order | Teacher is shown the **verbatim chunk text** (the same `chunk_text` the student sees) |
| `<t.t seconds>` markers give each page its position so the teacher reasons about **progression** | The full **taxonomy + chunk framing** (`TRAINING_SYSTEM_PROMPT`) is given so the teacher reasons within the **same rubric** the student is trained on |
| Teacher must **read what is visible** — dates, stamps, letterhead | Teacher must **quote verbatim** — the exact phrase, punctuation, syntax that licenses each classification |
| **One dating per page, differentiated** — never copy a coarse date to all pages | **One label per instance, differentiated** — never copy the same `soc_type` / boilerplate `explanation` across a chunk's instances |
| Output validated against `TemporalTarget` pydantic with **corrective retries** | Output validated against `LLMResponse` / `SocInstance` pydantic (`scripts/models.py`) with **corrective retries** |
| Reasoning references **specific pages and visible evidence** | Reasoning references **specific quoted spans** and contrasts the chosen type against a plausible alternative |
| Resumable `<output>.jsonl`, keyed by `na_id` | Resumable `<output>.jsonl`, keyed by `chunk_id` |

The essence you are importing: **a single strong teacher re-labels every chunk
from the source text, emitting per-instance, quote-anchored, type-contrastive
gold that a schema-echoing student cannot fake.** This is stronger than
consensus silver, because consensus filtering removes *disagreement* but does
not add *grounded discriminative reasoning* — it can still leave you a pile of
short, generic explanations that all look alike.

### Why not just keep the consensus-silver + `generate_traces.py` path?

`generate_traces.py` regenerates only the `<think>` block while **keeping the
silver `soc_type` labels fixed**. That improves trace *prose* but inherits the
silver labels' distribution and any systematic silver errors. The teacher-gold
method is more radical and is what `nara_trainer` proved necessary: the teacher
**re-derives the labels themselves from the text**, so both the label and its
justification are grounded in the same evidence. Keep `generate_traces.py` as a
trace-polishing step *after* teacher gold if you like, but the label source of
truth should move to the teacher.

---

## 2. Target architecture

```
chunking/  (verbatim chunk text — same as the student sees)
     │
     ▼
 A. TEACHER RELABEL            B. DIAGNOSE               C. TRAIN                 D. EVAL FOR ECHO
  scripts/teacher_gold.py  ──▶ scripts/diagnose_gold.py ─▶ scripts/train.py     ─▶ scripts/evaluate.py
  one strong local teacher     before/after signal        --mask-thinking False    schema-echo &
  (Qwen3-32B / 72B via vLLM)   table (§5)                  (train on the trace)     grounding metrics (§7)
     │                                                          │
     ▼                                                          ▼
 training_data/teacher.json  ────────────────────────────▶ dataset/train.jsonl
 (+ .jsonl resume checkpoint)   build_dataset.py --silver-dir  (differentiated gold)
```

Stages A, B, D are new. Stage C reuses `scripts/train.py` with think-masking
**off** (POST-TRAINING.md §4 Step 3). This slots into the existing SLURM layout
(`train_della.slurm`, `judge_della.slurm`) — the teacher server is stood up the
same way the GRPO judge server already is.

---

## 3. The teacher

### 3.1 Model choice

You already run **Qwen3-72B-Instruct** as the GRPO judge (`judge_della.slurm`,
`POST-TRAINING.md` §4). Reuse that infrastructure. Two viable options:

- **Qwen3-72B-Instruct** (TP2 on 2× A100-80GB, `--constraint=gpu80`) — highest
  quality; identical serving script to the judge.
- **Qwen3-32B-Instruct** (single A100-80GB) — cheaper, still far stronger than
  the 4B student. `nara_trainer` deliberately chose the **32B dense Instruct**
  (not Thinking) teacher and found it sufficient — "no reason to upgrade" — so
  start here if GPU budget is tight.

Use an **Instruct** model, not a Thinking model: you want clean, controllable
JSON, and you are supplying the reasoning format yourself. Serve it offline with
vLLM exactly as `judge_della.slurm` does, publishing the endpoint to a shared
file on `/scratch/gpfs/` so the client job can discover it.

### 3.2 Same representation as the student

Critical rule from `nara_trainer`: **show the teacher the source in the same
frame the student is trained in.** For Penelope that means the teacher's system
prompt is built from the **same `TRAINING_SYSTEM_PROMPT`** (taxonomy + schema in
`scripts/prompts.py`) the student uses, and the user turn carries the chunk with
the same `Source:` / context framing `scripts/silver.py` already constructs
(minus the `Source:` line if you adopt POST-TRAINING §2's leak fix — do that
here so the teacher never sees a filename cue).

Do **not** invent a different taxonomy or a different output schema for the
teacher. The whole point is that the gold lives in the student's own
representational space.

---

## 4. Implementation: `scripts/teacher_gold.py`

Model it directly on `create_training_data.py::vision_teacher` in `nara_trainer`.
The mechanics that made it work, transposed one-to-one:

### 4.1 The teacher system prompt

Reuse `TRAINING_SYSTEM_PROMPT` and **append the differentiation + grounding
rules** — the text analog of the vision teacher's "differentiate per page, do
not copy a coarse date onto every page" clause:

```
You are creating training labels for a stream-of-consciousness classifier.
You are shown ONE chunk of a literary text. Identify every SoC passage and,
for EACH, classify it using the taxonomy above.

Rules:
- GROUND every classification in the text. The `evidence` array must contain
  VERBATIM phrases copied exactly from the passage (matching words, punctuation,
  and any orthographic markers). Do not paraphrase evidence.
- DIFFERENTIATE. A chunk usually contains more than one technique. Give two
  instances the same `soc_type` only when the text genuinely supports it. Do not
  apply one label to the whole chunk.
- ARGUE the type. In `explanation`, name the ONE alternative type a careful
  reader might choose instead, and say what in the text rules it out. A label
  with no contrast is not acceptable.
- ABSTAIN natively. If a span reads like conventional narrated psychology rather
  than unmediated consciousness, mark `is_soc: false` and say what it would need
  to qualify. Skepticism about negatives matters as much as confidence about
  positives.
- Use `confidence: low` with a wide interpretive hedge rather than fabricating a
  crisp label you cannot defend from the text.
Return ONLY the JSON object matching the schema. One entry per distinct passage.
```

This is the exact spirit of the `nara_trainer` `VISION_TEACHER_SYSTEM` prompt —
differentiate, ground in visible evidence, abstain natively, contrast against
alternatives — retargeted from EDTF dates to SoC types.

### 4.2 Schema-validated output with corrective retries

Copy the `_vision_teacher_target` retry loop. Validate each response against the
existing pydantic models in `scripts/models.py` (`LLMResponse` →
`list[SocInstance]`). On failure, send a **corrective follow-up turn that keeps
the chunk in context** and states exactly what was wrong, then re-parse. Two
repair retries was enough in `nara_trainer`.

```python
# sketch — mirrors create_training_data.py::_vision_teacher_target
for _ in range(retries + 1):
    raw = call_teacher(messages)                 # vLLM OpenAI-compatible endpoint
    obj, err = extract_first_json(raw)           # balanced-brace scan, tolerant of prose
    if obj is not None:
        try:
            resp = LLMResponse.model_validate(obj)
            if _passes_grounding_gate(resp, chunk_text):   # see §4.3
                return resp, None
            err = "evidence not verbatim in passage"
        except ValidationError as e:
            err = f"schema: {str(e).splitlines()[0]}"
    messages += [
        {"role": "assistant", "content": raw[:4000]},
        {"role": "user", "content":
            f"That was invalid ({err}). Return ONLY corrected JSON: each instance "
            f"needs a soc_type from the taxonomy, an is_soc flag, an evidence array "
            f"whose strings appear VERBATIM in the passage, and an explanation that "
            f"names and rules out one alternative type."},
    ]
```

Reuse `_extract_first_json` verbatim from `create_training_data.py` — it is a
domain-agnostic balanced-brace JSON extractor.

### 4.3 A grounding gate the teacher can't dodge

The single highest-leverage addition. `nara_trainer` required the teacher to
read visible evidence; here, **enforce it programmatically**: reject any instance
whose `evidence` strings are not substrings of `chunk_text` (case/whitespace-
normalized). This is the same automatic grounding check already implemented in
`generate_traces.py::_grounding_score` — lift that logic into a hard gate. If the
gate fails, it becomes the `err` message for a corrective retry. This is what
prevents the teacher from producing plausible-but-ungrounded justifications,
which is exactly the boilerplate the student would otherwise learn to echo.

### 4.4 Resumable checkpoint

Copy the resume mechanism: append each validated chunk result to
`training_data/teacher.json.jsonl` keyed by `chunk_id`, skip keys already
present on restart, and compile the final `training_data/teacher.json` array at
the end. Teacher passes over hundreds of chunks are long and preemptible on
Della — resumability is not optional.

### 4.5 CLI shape

```bash
python scripts/teacher_gold.py \
    --input chunking/ \
    --output training_data/teacher.json \
    --model Qwen/Qwen3-32B-Instruct \
    --base-url $TEACHER_URL \        # published by teacher_della.slurm, like judge
    --temperature 0.2 \             # low, for grounded discriminative labeling
    --max-tokens 2048 \
    --retries 2 \
    --resume
```

Add a `teacher_della.slurm` by copying `judge_della.slurm` and swapping the
served model + the client command (endpoint-file discovery + health poll are
identical).

---

## 5. Prove the gold is better *before* training (Stage B)

`nara_trainer` did not retrain blindly — it built a **before/after signal table**
that demonstrated the teacher gold carried reasoning the coarse gold lacked.
Reproduce this with `scripts/diagnose_gold.py`, comparing the old silver
(`training_data/*.json` frontier outputs) against `training_data/teacher.json`.

Penelope-appropriate metrics (direct analogs of the `nara_trainer` table):

| Signal | Old silver | Teacher gold (target) | How to measure |
|---|---|---|---|
| **Boilerplate-echo rate** | high | ~0 | fraction of `explanation`s that are near-duplicates (normalized edit distance) across instances |
| **Evidence grounded (verbatim)** | low | ~1.0 | fraction of `evidence` strings that appear verbatim in `chunk_text` |
| **Type differentiation** | ~0 | high | unique `soc_type` count ÷ instance count, **within multi-instance chunks** (analog of `nara`'s per-page date differentiation) |
| **Distinct-evidence fraction** | low | ~1.0 | fraction of instances whose `evidence` set is unique in the chunk |
| **Type-contrast present** | ~0 | high | fraction of `explanation`s that name an alternative type (regex over taxonomy keywords) |
| **Type distribution** | spiky (majority-class) | spread across taxonomy | histogram + entropy over the 12 `SocType` values |
| **Native abstention** | rare | present & legit | fraction with `is_soc: false` that give a "would need…" rationale |
| **Confidence spread** | all `medium` | high/med/low mix | histogram over `Confidence` |

Spot-check a handful of chunks by hand, exactly as `nara_trainer` did (it
confirmed the teacher read three *different* dates off three scans). For Penelope,
confirm the teacher assigns, e.g., `direct_interior_monologue` vs
`indirect_interior_monologue` to two spans in the same Woolf chunk **and quotes
the pronoun/reporting-verb difference that distinguishes them**. If it can do
that, the gold carries the typology. If every instance in a chunk gets the same
type with interchangeable evidence, the teacher (or its prompt) has failed —
fix the prompt, not the student.

---

## 6. Train on the gold — and on the reasoning (Stage C)

Two changes from the current Penelope SFT:

1. **Point `build_dataset.py` at the teacher gold.** `training_data/teacher.json`
   is in the same silver record schema (`build_dataset.py` already ingests
   `training_data/*.json`), so consensus/passage-grouping still applies — but with
   a single high-quality teacher you can relax consensus filtering (you no longer
   need three models to vote; you need one grounded teacher). Keep the negative
   examples and class-rebalancing from `POST-TRAINING.md` §3.

2. **Turn off think-masking** (`POST-TRAINING.md` §4 Step 3):
   `python scripts/train.py --mask-thinking False`. This is the direct analog of
   `nara_trainer` computing loss over the differentiated target rather than a
   copyable template. The interpretive `<think>` deliberation — where the type
   contrast and the verbatim grounding live — must receive gradient, or the
   student will again optimize only the trivially-copyable JSON tail.

Have the teacher emit the `<think>` trace too (the differentiation/contrast
reasoning), followed by the JSON. Then the student is trained on *how the type
was argued from the text*, not just the label. Keep `nara_trainer`'s warning in
view: **a low SFT loss still does not prove reasoning** — it is teacher-forced
next-token prediction. Only the held-out echo eval (Stage D) tells you the truth.

---

## 7. Evaluate for schema-echo, not just F1 (Stage D)

This is where `nara_trainer` caught the failure the loss curve hid, and where
Penelope's own `POST-TRAINING.md` "Perspective" section already argues F1 is the
wrong target. Extend `scripts/evaluate.py` to run the student **free-generation**
on the held-out `dataset/test.jsonl` and compute echo/grounding diagnostics — the
Penelope analog of `evaluate_pilot.py`'s schema-echo check:

- **`echoes_gold_reasoning`** — does the student reproduce teacher `explanation`
  boilerplate verbatim? Target: low.
- **`all_same_type`** — fraction of multi-instance chunks where the student emits
  one identical `soc_type` for every instance (the `all_same_date` analog).
  Target: well below 100%.
- **`evidence_grounded`** — fraction of the student's `evidence` strings that
  appear verbatim in the source chunk. Target: high. (Reuse the §4.3 gate as a
  metric.)
- **`type_contrast_rate`** — fraction of outputs whose reasoning names and rules
  out an alternative type. Target: high.
- **`schema_valid`** — should stay high; it was never the problem.

Success criteria (mirroring the `nara_trainer` bar): `schema_valid` stays high,
`echoes_gold_reasoning` ≈ 0, `all_same_type` far below 100%, `evidence_grounded`
high. If those hold, the student learned the *typology*, not the template.

**Decoding note from `nara_trainer`:** when you run free-generation eval, greedy
decoding degenerated into repeat-loops; it used `repetition_penalty ≈ 1.15` and
**avoided `no_repeat_ngram_size`** because that corrupts legitimately repeated
JSON keys across `instances[]`. Apply the same decoding settings in
`scripts/evaluate.py` / `scripts/infer.py`.

---

## 8. Relationship to the existing GRPO plan

Teacher gold and GRPO are complementary, and the order matters:

1. **Teacher gold + SFT (this document)** gives the student a *policy that can
   already produce grounded, differentiated, type-contrastive reasoning*. GRPO
   cannot bootstrap a skill the base policy never exhibits; starting GRPO from a
   schema-echo SFT checkpoint mostly teaches reward-hacking. Fix the SFT gold
   first — this is precisely why `nara_trainer` stopped and rebuilt gold instead
   of adding epochs or jumping to RL.
2. **GRPO (POST-TRAINING.md §4)** then *sharpens* interpretive quality on top of
   a good policy. Notably, the GRPO reward components you already designed —
   grounding, skepticism, specificity, type-coherence — are the **same signals**
   the teacher-gold diagnostics (§5) and echo eval (§7) measure. That is not a
   coincidence: the teacher gold front-loads those properties into the SFT data,
   and GRPO amplifies them. You can even reuse the `Qwen3-72B` judge server as
   the teacher server (§3.1).

---

## 9. Concrete next steps

1. `scripts/teacher_gold.py` — port `create_training_data.py::vision_teacher`:
   reuse `TRAINING_SYSTEM_PROMPT` + differentiation/grounding rules (§4.1),
   `LLMResponse` schema validation with corrective retries (§4.2), the verbatim
   grounding gate (§4.3), and resumable `.jsonl` (§4.4).
2. `teacher_della.slurm` — copy `judge_della.slurm`; serve `Qwen3-32B-Instruct`
   (or reuse the 72B judge); publish endpoint to a shared file; run the client.
3. Run a **32-chunk spot-check** first (the `MAX_SEQUENCES=32` habit from
   `nara_trainer`) and hand-inspect differentiation before the full pass.
4. `scripts/diagnose_gold.py` — build the before/after table (§5). Gate the full
   run on it: if differentiation/grounding aren't clearly better than silver,
   fix the teacher prompt, not the student.
5. Full teacher pass → `training_data/teacher.json`.
6. `build_dataset.py --silver-dir training_data/` (teacher gold in place) →
   `train.py --mask-thinking False` (§6).
7. Extend `scripts/evaluate.py` with the echo/grounding eval (§7); confirm the
   success criteria before spending GPU on GRPO.

The one-line version: **stop feeding the student a schema it can copy; feed it
differentiated, verbatim-grounded, type-contrastive gold from a strong teacher
reading the same text — then verify with a free-generation echo eval, not the
loss curve.**
