Need a harness/cli for the model? Control system prompt (same as during training) and model for output. 

Training data contains "Source:" in user input. Should we retrain without? 


Problem: **High false-positive rate.** The model is biased toward predicting SoC.
  Post-hoc confidence thresholding or a secondary filtering step is recommended
  for precision-sensitive applications.

  Expand on interpretive AI frame, not just classifier. But goal is to name and recognize types of SoC. Specifically in ways that can be learned and reproduced by a model. The goal is not accurate recognition and classification, but rather the act of defining and automaticallying testing those definitions. How to evaluate in that context? Not against gold standard, but against plausible-coherent interpretations that corespond with the underlying typology. 

Are there better models than Qwen for this task? How might we productively compare models to improve on interpretive AI task? Perhaps add GRPO and a reward model to training? 

---

## Implementation Plan

### 1. Inference CLI / Harness

**Problem:** No standardized entry point to run the fine-tuned model with the exact training-time system prompt. The output should also be loaded into a pydantic model for structured downstream use. This should be a cli tool using Typer.

**Code changes:**
- Create `scripts/infer.py` that wraps `load_model_and_tokenizer` from `scripts/evaluate.py`, applies the system prompt from `SKILL.md`, and returns structured JSON output.
- CLI flags: `--model`, `--adapter`, `--base-model`, `--input-text`, `--input-file`, `--format [json|csv|markdown]`.


**Training curriculum:** No change needed — this is purely an inference artifact.

---

### 2. Strip "Source:" from Training Data

**Problem:** User turns in the training JSONL include `Source: <filename>`, a spurious feature the model may have learned to rely on as a cue for positive predictions.

**Code changes:**
- In `scripts/build_dataset.py`, add a normalization pass that removes lines matching `^Source:.*` from user message content before writing JSONL.
- Re-run: `python scripts/build_dataset.py` → inspect `dataset/train.jsonl` to confirm removal.
- Re-train from scratch with the cleaned dataset; compare eval metrics (`logs/eval_*_results.json`) before and after.



---

### 3. Reduce False-Positive Rate

**Problem:** Model is biased toward predicting SoC; precision is too low for precision-sensitive use.


**Negative-example rebalancing (medium cost):**
- Check class balance: `python -c "import json; d=open('dataset/train.jsonl'); [print(json.loads(l)['label']) for l in d]" | sort | uniq -c`
- The `negatives/` directory contains non-SoC texts. If `label: false` examples are underrepresented, draw more negatives in `scripts/build_dataset.py` until the true/false split is roughly 1:2.
- This is the most durable fix — it teaches the model to be skeptical rather than patching output at inference time.


---

### 4. Interpretive Reasoning Traces and GRPO

**Core reframe:** The `<think>` block is not a scratchpad for arriving at the "correct" answer — it is the interpretive act itself. A good reading weighs multiple plausible classifications, stays grounded in verbatim evidence, acknowledges genuine ambiguity, and produces a defensible judgment rather than a determined one. The goal is a model that interprets, not one that merely detects.

The current `ThinkMaskingCollator` in `scripts/train.py` sets labels to `-100` for all `<think>...</think>` tokens, giving the model no gradient signal on its reasoning. The JSON output is trained; the thinking that produces it is not. That must change.

**Step 1 — Define the interpretive reasoning format in `SKILL.md`**

Add a section specifying what a good `<think>` trace looks like. It should not be a deterministic checklist — it should be a structured *deliberation*:

```
Interpretive reasoning trace format:
- First reading: describe the immediate texture of the passage — voice, syntax, register.
  What is the reader's relationship to interiority here?
- Candidate types: name 1-3 types from the taxonomy that could plausibly apply.
  For each, cite a verbatim phrase that supports it.
- Strongest reading: argue for the primary type. What makes this type the most
  productive lens — not just the most accurate label?
- Counter-reading: what would a reasonable alternative interpretation say?
  Why does the chosen reading hold up against it?
- Skepticism gate: could this passage appear in a non-SoC text? If yes, what
  specifically marks it as interior consciousness rather than narrated psychology?
  If you cannot answer, set is_soc: false.
```

This format is **exploratory, not procedural**. The model should not be checking boxes; it should be practicing interpretation. The deliberation is the output that matters, and the JSON is a structured summary of the conclusion.

**Step 2 — Generate interpretive thinking traces for training data**

Create `scripts/generate_traces.py` that takes existing `dataset/train.jsonl` records and regenerates the `<think>` block using a frontier model (e.g., via `models.yaml` with `gpt-5.4` or `qwen3-max`). Prompt the frontier model with the interpretive reasoning format above and the passage + gold label. Generates multiple candidate traces per passage (e.g., 4–8 completions with `temperature=0.8`).

Store all candidates in `training_data/traces/<passage_id>.jsonl`. Selection is automated: score each candidate with the automatic reward components (grounding + typological specificity) and keep the highest-scoring trace per passage. This makes `generate_traces.py` a self-contained pipeline step with no manual bottleneck. The stored candidates remain on disk for audit — a researcher can inspect the full set at any time, but the pipeline does not wait on human review.

Rebuild `dataset/train.jsonl` from selected traces: `python scripts/build_dataset.py --traces training_data/traces/`.

**Step 3 — Remove think-masking; train on the full reasoning chain**

In `scripts/train.py`, retire `ThinkMaskingCollator` (or expose it as `--mask-thinking` flag defaulting to `False`) so the model trains on the complete assistant turn, including the `<think>` block. The reasoning trace is now part of what the model is learning, not a free prefix.

This is the key architectural change: loss is computed over both the deliberation and the JSON. The model learns *how to think about SoC*, not just what JSON shape to output.

**Step 4 — GRPO with an interpretive reward model**

Once the SFT run on rubric-structured traces is complete, apply GRPO to sharpen interpretive quality. Create `scripts/train_grpo.py` using `trl.GRPOTrainer`.

**Reward function design:** Do not score `is_soc` or `soc_type` against silver labels — that reproduces the gold-standard framing we are moving away from. Instead, score the *quality of the interpretation* on two dimensions, corresponding to the two decisions the model makes:

**Dimension 1 — `is_soc` verdict quality:**
- **Grounding score** (0–1): does the `<think>` trace cite at least one verbatim phrase from the passage that drives the verdict? (Automatic, string-match: phrase appears in passage.)
- **Skepticism score** (0–1): if `is_soc: false`, does the trace articulate *why* this passage falls short — naming what it would need to qualify? If `is_soc: true`, does the trace address the skepticism gate ("could this appear in a non-SoC text?")? Rewarding good negative reasoning is as important as rewarding good positive reasoning.

**Dimension 2 — `soc_type` argument quality (only scored when `is_soc: true`):**
- **Typological specificity score** (0–1): does the reasoning name a specific Humphrey/Steinberg marker (from the taxonomy keyword list), not just a generic interior label like "stream of consciousness"? (Automatic, keyword match.)
- **Type coherence score** (0–1): does the `<think>` trace explicitly argue *for* the assigned `soc_type` and *against* at least one alternative? Does the final JSON `soc_type` follow from the deliberation rather than appearing without support? (LLM-as-judge: prompt a frontier model with the trace, the passage, and the taxonomy.)

**The judge prompt lives in `scripts/prompts.py` as `JUDGE_PROMPT`.** It does not exist yet and must be written. It receives: (1) the original passage, (2) the full `<think>` trace, (3) the final JSON output, and (4) the taxonomy from `_TAXONOMY`. It returns a JSON object:

```json
{
  "grounding": 0.0–1.0,
  "skepticism": 0.0–1.0,
  "specificity": 0.0–1.0,
  "type_coherence": 0.0–1.0,
  "rationale": "<one sentence explaining the type_coherence score>"
}
```

The judge prompt should instruct the model to score conservatively — a trace that names a type without arguing for it scores 0.2 on `type_coherence`, not 0.5. The `rationale` field is logged per completion for qualitative monitoring during GRPO runs.

**The judge runs as a local vLLM server on Della, not via an external API.** This eliminates per-call cost, removes the external dependency during training, and gives throughput high enough for batched judgment across k=8 completions.

**Deployment and inter-node connectivity:**

Della compute nodes (`della<N>.princeton.edu`) are on the same internal network and can reach each other directly by hostname — no SSH tunnel or VPN needed. vLLM binds to `0.0.0.0` by default, so any node in the cluster can call `http://della<N>.princeton.edu:8000/v1` once the server is up.

The two concrete problems are **discovery** (which node did SLURM allocate?) and **readiness** (vLLM takes 5–10 minutes to load a 72B model). Both are solved with a shared file on `/scratch/gpfs/`:

`judge_della.slurm`:
```bash
# Start vLLM, wait for it to be ready, then publish the endpoint
ENDPOINT_FILE=/scratch/gpfs/MM4/apjanco/penelope/judge_endpoint.txt
rm -f $ENDPOINT_FILE

vllm serve Qwen/Qwen3-72B-Instruct \
    --port 8000 --tensor-parallel-size 2 --max-model-len 8192 &

# Poll until the server responds, then write the endpoint
until curl -sf http://localhost:8000/health; do sleep 10; done
echo "http://$(hostname -f):8000/v1" > $ENDPOINT_FILE
wait  # keep the job alive
```

`train_grpo.slurm` submits with `--dependency=after:<judge_job_id>` (starts after judge *begins*, not finishes) and then waits for the endpoint file before launching training:
```bash
ENDPOINT_FILE=/scratch/gpfs/MM4/apjanco/penelope/judge_endpoint.txt
until [ -f $ENDPOINT_FILE ]; do sleep 15; done
export JUDGE_URL=$(cat $ENDPOINT_FILE)
python scripts/train_grpo.py --judge-url $JUDGE_URL ...
```

`scripts/train_grpo.py` reads `JUDGE_URL` from the environment (or `--judge-url` flag) and sets it as `base_url` for the judge client — no entry in `models.yaml` needed; the URL is resolved at runtime.

Qwen3-72B fits in BF16 across 2×A100 80GB (`--constraint=gpu80`). Fallback: Qwen3-32B on 1 GPU, or any 70B model with 4-bit quantization.

Combine as:
```
reward = 0.2 * grounding + 0.2 * skepticism + 0.2 * specificity + 0.4 * type_coherence
```
When `is_soc: false`, `specificity` and `type_coherence` are set to their neutral value (0.5) rather than 0, so the model is not penalized for correctly declining to classify.

**Group sampling:** For each passage, sample *k=8* completions. The group relative advantage (GRPO's baseline subtraction) means the model is pushed toward *better interpretations of the same passage*, not toward any fixed target. This matches the interpretive frame: there is no single correct reading, but some readings are more defensible than others.

**Training curriculum:**
1. **SFT on interpretive traces** (`scripts/train.py --mask-thinking False`) — gives the model a policy that can produce deliberative reasoning.
2. **GRPO on top of SFT adapter** (`scripts/train_grpo.py`) — sharpens interpretive quality using the composite reward. Run for 1–2 epochs; monitor coherence score trend, not F1.
3. **Model comparison:** run the same SFT + GRPO pipeline on Qwen3-4B (current) and one alternative base (Llama-3.1-8B or Mistral-7B). Compare final coherence scores, not classification accuracy. The better model is the one whose reasoning traces are more interpretively productive.

Add `--method [sft|grpo]` to `train_della.slurm` to dispatch either pipeline from the same job script.

---

### 5. Documentation Updates

This refactor changes the training pipeline substantially — new scripts, new training regime, new evaluation frame, new SLURM jobs. All documentation should be updated **after** the code changes land, not before.

**`README.md`**

The training pipeline diagram needs a new stage and corrected descriptions:

```
training_data/*.json          negatives/
        │                         │
        └──────────┬──────────────┘
                   ▼
         build_dataset.py  →  dataset/train.jsonl (silver labels + boilerplate traces)
                   ▼
       generate_traces.py   →  training_data/traces/   (interpretive <think> blocks)
                   ▼
         build_dataset.py   →  dataset/train.jsonl (rebuilt with interpretive traces)
          --traces ...
                   ▼
             train.py       →  adapter-sft/        (full sequence loss, no think-masking)
          --mask-thinking False
                   ▼
          train_grpo.py     →  adapter-grpo/       (interpretive reward sharpening)
          (judge server: judge_della.slurm)
                   ▼
          evaluate.py       →  eval_results.json   (coherence scores, not just F1)
```

The "Quick start (inference only)" section should reference `scripts/infer.py` (once written) as the recommended inference entry point in place of the manual `run.py` invocation.

The opening description currently says "No external API calls; no network dependency at inference time" — this remains true for the fine-tuned model at inference. The vLLM judge is a training-time dependency only; clarify this in the README.

**`SKILL.md`**

Add two new sections (as specified in Step 1 of Section 4 above):
1. **Interpretive reasoning trace format** — the five-step deliberation format that defines what a good `<think>` block looks like. This is the primary rubric for the project and belongs in `SKILL.md` as the authoritative reference.
2. **Interpretive evaluation criteria** — brief description of the four reward dimensions (grounding, skepticism, specificity, type coherence) so that the rubric is human-readable, not just code.

**`requirements.txt`**

Add before the fine-tuning block:
```
typer>=0.12.0       # scripts/infer.py CLI
trl>=0.8.0          # scripts/train_grpo.py (GRPOTrainer)
vllm>=0.4.0         # judge_della.slurm (install separately on Della; not required for inference)
```
Note `vllm` in a comment as Della-only; it should not be a hard dependency for local inference users.

**`model_card.md`**

Update to reflect:
- Training regime: SFT with full sequence loss (interpretive traces) → GRPO with interpretive reward
- Evaluation: coherence-based, not precision/recall against gold standard — explain why
- Known limitations: remove or revise the false-positive warning once the negative trace fix is validated
- Intended use: frame explicitly as an interpretive AI tool, not a classifier

**`POST-TRAINING.md`** (this file)

Once the refactor is complete, this file should be archived or converted into a proper `CHANGELOG.md` entry. It is working notes; the decisions it records should be promoted into the documents above and this file retired.

---

## Perspective: What We Are Actually Evaluating

The goal of this project is not to improve a classifier. It is to evaluate a rubric and a typology.

The central question is: **can a model learn and implement the Humphrey/Steinberg framework for stream of consciousness?** Not just output the right label, but reason within the vocabulary of the framework — use its categories as tools for reading, distinguish where the framework draws distinctions, and produce interpretations that would be recognizable to a literary scholar working in this tradition.

This reframes what success and failure mean.

**Success** does not look like high F1 against a gold standard. It looks like a model whose reasoning traces consistently use the framework's distinctions in coherent, defensible ways — whose reading of a Woolf passage would be recognizable as a *reading*, not a classification. The coherence scoring from the judge model is a proxy for this, but the deeper test is whether a literary scholar can read the model's output and say: "this is how someone who understands the typology would approach this passage."

**Failure is also informative.** If the model consistently fails to distinguish `indirect_interior_monologue` from `omniscient_description`, that may not be a model limitation — it may be a signal that the typology's boundary between those categories is underspecified, or that the key markers in `SKILL.md` are not sufficient to operationalize the distinction. When the model struggles, examine the rubric before concluding the model is inadequate.

**The training pipeline is an experiment on the typology.** Each training run is testing whether the framework — as specified in `SKILL.md` and implemented in the trace format and reward function — is coherent enough to be learned and reproduced. The reward function is not a performance metric; it is an operationalization of what it means to interpret well within this framework. If the reward function turns out to be systematically gameable or to reward the wrong things, that reveals a gap in how the framework has been specified.

**Evaluation, accordingly, is not benchmark-driven.** The primary output of an evaluation run is not a number but a set of readings: the model's `<think>` traces on a held-out set of passages. These should be read, not just scored. Are the candidate types named with appropriate uncertainty? Is the evidence cited verbatim and relevant? Is the counter-reading actually a plausible alternative? Does the skepticism gate engage with the right question? These are humanistic criteria, and they require humanistic judgment to assess — the judge model approximates this judgment, but the final word is a reader's.

**What the GRPO loop is doing.** Group sampling with an interpretive reward does not optimize the model toward any fixed reading. It selects, within a group of candidate interpretations of the same passage, the ones that are most consistent with the framework's norms. Over many training steps, this should push the model toward readings that are more grounded, more typologically specific, more willing to engage with alternative readings, and more willing to decline to classify when the evidence is insufficient. This is not gradient descent toward a target — it is closer to sustained practice within a disciplinary tradition.

---

## Research Claims and Analysis

This work sits at the intersection of computational literary studies, interpretive AI, and literary theory. A paper arising from it is not primarily an NLP paper — it does not claim state-of-the-art performance on a benchmark. It is a claim about the operationalizability of a humanistic framework and about what the training process reveals about that framework.

**The central claim** is that interpretive frameworks in literary studies — typologies, rubrics, reading practices — can be operationalized as training objectives for language models, and that the training process itself constitutes a test of the framework's internal coherence. Where the model learns to apply a distinction reliably, that is evidence the distinction is well-specified. Where it consistently fails to separate two categories, that is evidence of underspecification in the framework, not (only) model inadequacy.

**Supporting analysis — what to measure and report:**

*Per-type learning curves.* Track mean coherence score per `soc_type` across SFT and GRPO training steps. Types that plateau at low coherence despite training are candidates for rubric refinement. Types that separate cleanly early are evidence of well-drawn category boundaries. This analysis answers: which distinctions in the Humphrey/Steinberg typology are most and least operationalizable?

*Confusion patterns as typological evidence.* When the model conflates two types — say, `indirect_interior_monologue` and `omniscient_description` — that conflation is not merely an error. It is evidence that the typology's boundary between those categories depends on distinctions that are not reliably encoded in the text itself (or not in the markers specified in `SKILL.md`). Report these as findings about the typology, not as model failures.

*The counter-reading corpus.* The `<think>` traces contain a counter-reading step — the model's account of what a reasonable alternative interpretation would say. Collect and analyze these at scale: what alternatives does the model consistently consider? Do they align with what literary scholars would identify as the live interpretive alternatives for those passages? This is qualitative evidence that the model is reasoning within the tradition, not just pattern-matching.

*Ambiguity and refusal.* Cases where the model declines to classify (`is_soc: false` with a well-argued skepticism trace) are valuable data. Compare them against the passages where human annotators disagreed in the original silver-labeling process. If the model's refusals correlate with annotator disagreement, that is evidence it is learning genuine ambiguity, not just being conservative. A model that confidently misclassifies is less interesting than one that correctly identifies where the framework's concepts are under strain.

*Ablation across training stages.* Report coherence scores at three points: (1) base Qwen3-4B zero-shot, (2) SFT with full sequence loss on interpretive traces, (3) SFT + GRPO. Each stage is a separate contribution: (1) establishes what the base model already knows about SoC, (2) shows the effect of training on deliberative reasoning, (3) shows the effect of group-relative reward pressure. The differences between stages are the empirical claims.

*Corpus variation.* The training corpus spans Woolf, Joyce, Faulkner, Richardson, James, Tolstoy, and others. Analyze coherence scores by author and by SoC type × author. If the model reasons better about Woolf's `indirect_interior_monologue` than Faulkner's `direct_interior_monologue`, that may reflect training data coverage — or it may reflect that some authors' SoC techniques are more formally consistent and thus more learnable. Either finding is publishable.

**Framing the contribution:**

The paper is not "we built a classifier." It is closer to: "we used a language model as an instrument for testing a literary-critical framework, and we report what the instrument revealed." The analogy is not to NLP benchmarking but to the use of computational tools in corpus linguistics to surface patterns that inform theoretical claims.

This places the work in dialogue with:
- Franco Moretti's distant reading and the question of what computational methods can and cannot access in literary texts
- The interpretive AI literature (Ramsey, Ramsay's *Reading Machines*; McGann on radiant textuality) on the limits and possibilities of algorithmic interpretation
- The Humphrey/Steinberg typology itself — this is probably the most sustained computational test that typology has received, and the findings speak directly to its theoretical coherence

**What the paper should not claim:**
- That the model *reads* in any phenomenological sense
- That coherence scores are a ground-truth measure of interpretive quality
- That high reward means the interpretation is correct — only that it is well-argued within the framework

The honest frame: a model trained this way produces readings that are *internally consistent with the Humphrey/Steinberg framework as operationalized here*. Whether that operationalization captures what Humphrey actually meant is a separate scholarly question — and one the paper can productively leave open.


