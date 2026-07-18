# Penelope

**Detect and classify stream of consciousness in literary texts using a fine-tuned local model.**

Penelope is a pipeline that reads a corpus of literary works, segments them into
chunks, and runs each chunk through a locally-hosted Qwen3-4B model fine-tuned
specifically on the SoC classification task. Every instance is classified
according to a detailed taxonomy drawn from Robert Humphrey and Erwin R.
Steinberg. No external API calls; no network dependency at inference time.

Named after the final episode of Joyce's *Ulysses* — Molly Bloom's unbroken
interior monologue — the canonical example of stream of consciousness in fiction.

---

## Architecture

```
input/                         model_config.yaml
  ├── novel_1.txt              (local model path,
  ├── novel_2.pdf               device, sampling)
  └── novel_3.docx                    │
        │                             │
        ▼                             ▼
   ┌─────────┐             ┌──────────────────────┐
   │ chunk.py │────────────▶│       run.py         │
   │ extract  │  chunking/  │  fine-tuned Qwen3-4B │
   │ + markup │             │  (local inference)   │
   └─────────┘             └──────────────────────┘
                                       │
                              results/*.json
                                       │
                            ┌──────────┴──────────┐
                            ▼                     ▼
                      consensus.py           streamlit app.py
                   (multi-run merging)    (comparison dashboard)
```

**Training pipeline** (separate, run once to produce the model):

```
positives/                     negatives/
  (SoC texts →          +   (Gutenberg
   silver.py →                plain text)
   training_data/*.json)           │
        │                         │
        └──────────┬──────────────┘
                   ▼
         build_dataset.py  →  dataset/train.jsonl
                   ▼
             train.py       →  adapter-sft/
                   ▼
          train_grpo.py     →  adapter-grpo/
       + judge_della.slurm     (Qwen3.6-27B vLLM, concurrent SLURM job)
                   ▼
          export_model.py   →  models/penelope-soc-v1
                   ▼
          evaluate.py       →  eval_results.json
```

---

## Quick start (inference only)

If you already have the fine-tuned model:

```bash
# 1. Clone and set up environment
git clone https://github.com/apjanco/penelope.git
cd penelope
conda create -n penelope python=3.12 -y
conda activate penelope
pip install -r requirements.txt

# 2. Configure the model path in model_config.yaml
#    path: apjanco/penelope-soc-v1    # HF Hub (auto-download)
#    path: ./models/penelope-soc-v1   # local merged model

# 3. Extract and chunk your texts. Most large language models are not able to process an entire work at once. This step breaks a work into chunks that the model is able to process.

python chunk.py --input input/
# Review and adjust <chunk-N> boundaries in chunking/ before proceeding

# 4. Run inference
python run.py --input chunking/ --output results/

# 5. (Optional) Compare runs with the dashboard
streamlit run app.py
```

---

## Step-by-step usage

### Step 1 — Chunk (`chunk.py`)

Extract text and produce annotated files with `<chunk-N>` markup:

```bash
# All files in input/
python chunk.py --input input/

# Single file
python chunk.py --input input/mrs_dalloway.txt

# Custom chunk size
python chunk.py --input input/ --chunk-size 15000 --chunk-overlap 800

# Custom output directory
python chunk.py --input input/ --output my_chunks/
```

Creates plain-text files in `chunking/` with tags like:

```
<chunk-0 label="Chapter I">
...text of chunk 0...
</chunk-0>

<chunk-1 label="Chapter II">
...text of chunk 1...
</chunk-1>
```

Open these in any editor. You can move, merge, split, or rename chunks before
running inference — the pipeline parses the markup at runtime.

### Step 2 — Inference (`run.py`)

```bash
# Full corpus
python run.py --input chunking/ --output results/

# Dry run — list chunks without running the model
python run.py --input chunking/ --dry-run

# Verbose logging
python run.py --input chunking/ -v

# CSV output only
python run.py --input chunking/ --format csv
```

The model is loaded once from `model_config.yaml` and held in memory for the
run. Results are written to `results/results.csv` and `results/results.json`.

### Step 3 — Consensus (`consensus.py`)

Merge results from multiple runs into filtered datasets:

```bash
python consensus.py                              # default track (moderate)
python consensus.py --track conservative
python consensus.py --track conservative --track liberal
python consensus.py --list                       # show available tracks
python consensus.py --track moderate --output consensus/
```

Tracks are configured in `consensus.yaml`:

| Track | Agreement | Min models | Confidence |
|---|---|---|---|
| **conservative** | full | 3 | high |
| **moderate** | partial | 2 | medium |
| **liberal** | any | 1 | low |

---

## Model configuration (`model_config.yaml`)

```yaml
model:
  path: apjanco/penelope-soc-v1   # HF Hub ID or local path
  device: auto                     # auto, cuda, cpu, mps
  max_new_tokens: 4096             # thinking block + JSON output
  enable_thinking: true            # Qwen3 <think>...</think> mode
  batch_size: 1
  temperature: 0.6
  top_p: 0.95
  top_k: 20
```

Set `enable_thinking: false` for faster inference without the reasoning trace.
Accuracy may be slightly lower on ambiguous passages.

**Hardware requirements:**
- GPU: Qwen3-4B in bf16 requires ~8 GB VRAM; with 4-bit quantization ~4–5 GB.
- CPU: Use a GGUF export (`llama.cpp` Q4_K_M, ~2.5 GB) for machines without a GPU.

---

## Training pipeline

### 0. Prerequisites

```bash
# On Princeton Della — interactive GPU login node
ssh della-gpu.princeton.edu
module load anaconda3/2025.12
module load cudatoolkit/12.8      # verify with: sbatch test_cuda.slurm
conda create -n penelope python=3.12 -y
conda activate penelope
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # recommended on A100; ~10 min
```

### 1. Download negative examples (`scripts/download_gutenberg.py`)

Downloads a curated set of public-domain texts from Project Gutenberg for use
as non-SoC training examples. Included authors have predominantly externalized
narration (Conan Doyle, Dickens, Trollope, Twain, Verne, Dumas, etc.).
Authors who pioneered free indirect discourse (Austen, Eliot, James, Flaubert)
are explicitly excluded.

```bash
# Download all curated texts (~50 works) to negatives/
python scripts/download_gutenberg.py --output-dir negatives/

# Download specific Gutenberg IDs only
python scripts/download_gutenberg.py --output-dir negatives/ --ids 98 766 1661

# Print the curated list without downloading
python scripts/download_gutenberg.py --list

# Slower rate to avoid Gutenberg throttling
python scripts/download_gutenberg.py --output-dir negatives/ --delay 3.0
```

Files are saved as `negatives/pg<ID>_<slug>.txt` with boilerplate stripped.
Already-downloaded files are skipped on re-runs.

### 1.5 — Download positive examples (`scripts/download_positives.py`)

Downloads texts where stream-of-consciousness technique is prominent — Joyce,
Woolf, Richardson, Schnitzler, late James, Flaubert, Stein, and others.
Grouped by primary SoC technique (direct interior monologue, free indirect
discourse, reverie/fantasy, soliloquy, hybrid). All works are US public domain.

```bash
# Download all curated SoC texts (~25 works) to positives/
python scripts/download_positives.py --output-dir positives/

# Print the catalog without downloading
python scripts/download_positives.py --list

# Filter by author name
python scripts/download_positives.py --output-dir positives/ --author woolf

# Download specific Gutenberg IDs only
python scripts/download_positives.py --output-dir positives/ --ids 4300 5765

# Slower rate to avoid Gutenberg throttling
python scripts/download_positives.py --output-dir positives/ --delay 3.0
```

The `positives/` directory also includes several non-Gutenberg texts
(Faulkner, Woolf later works) that must be sourced separately and placed
there manually before chunking.

After downloading, generate silver labels:

```bash
# Chunk the positive texts
python chunk.py --input positives/ --output chunking_positives/

# Run four LLMs to generate silver-label annotations
python scripts/silver.py --input chunking_positives/ --output-dir training_data/
```

Silver labels are written to `training_data/<model_label>.json`. Run
`scripts/silver.py --dry-run` first to verify chunk boundaries.

### 1.6 — Unified annotation + resampling (`scripts/annotate.py`)

`annotate.py` runs both `silver.py` (single-pass annotations) and
`resample.py` (k=10 resamples per passage) in one resumable step. Prefer
this over running the two scripts separately for any new batch of texts.

```bash
# Annotate all files in chunking_positives/ (silver + k=10 resampling)
python scripts/annotate.py --input chunking_positives/

# Restrict to specific models
python scripts/annotate.py --input chunking_positives/ --model mercury-2 --model qwen3-max

# Silver extraction only (skip resampling — faster, useful when iterating)
python scripts/annotate.py --input chunking_positives/ --skip-resample

# Dry run — show call counts without hitting any API
python scripts/annotate.py --input chunking_positives/ --dry-run
```

For large corpora on Della, run from a visualisation node inside `screen` or
`tmux` — compute nodes have no internet access. See `resample_run.sh` for
the recommended workflow when running each model separately.

Resampling is **required for v3**: `build_ambiguity_dataset.py` (step 2.5)
needs at least two models with k≥10 runs to distinguish systematic from
random disagreement.  Priority order (lowest agreement first):
Richardson > Faulkner > Woolf > Joyce > Sinclair.

Output:
- `silver_data/<model>.json` — single-pass annotations
- `silver_data/resampled/<model>.jsonl` — k resamples per passage

### 2. Build the dataset (`scripts/build_dataset.py`)

Merges silver-label JSON files in `training_data/` (produced by `silver.py`
from the `positives/` corpus) with Gutenberg negatives, applies consensus
filtering, collapses rare types, and splits into train / val / test JSONL files.

```bash
python scripts/build_dataset.py \
    --silver-dir silver_data/ \
    --neg-dir    negatives/ \
    --output     dataset/ \
    --track      moderate     # conservative | moderate | liberal
```

Output: `dataset/train.jsonl`, `dataset/val.jsonl`, `dataset/test.jsonl`.
Each record is an OpenAI-format chat message with a `<think>…</think>` reasoning
trace followed by a JSON `instances` array.

Consensus tracks applied during build:

| Track | Models must agree | Min models |
|---|---|---|
| conservative | all four | 4 |
| moderate | majority | 3 |
| liberal | any | 1 |

**For v3:** after step 2.5–2.7 below, rebuild with `--ambiguity-index` to
fold in multi-label boundary-case records alongside the consensus positives.

### 2.5 — Build ambiguity index (`scripts/build_ambiguity_dataset.py`)

Required for v3 training. Loads all resampled JSONL files from
`silver_data/resampled/`, measures within-model consistency (mode type as
fraction of k runs), classifies each passage as one of four boundary types,
and writes `silver_data/ambiguity_index.jsonl`.

```bash
# Build from resampled data (normal mode)
python scripts/build_ambiguity_dataset.py

# Preview stats without writing output
python scripts/build_ambiguity_dataset.py --dry-run

# Fallback when resampling is not yet complete — uses single-pass silver data
# (upper-bound estimate of disagreements; some random disagreement included)
python scripts/build_ambiguity_dataset.py --single-pass

# Adjust the consistency threshold (default 0.8 — model must agree 80% of runs)
python scripts/build_ambiguity_dataset.py --consistency-threshold 0.7
```

Boundary types in the output index:

| Type | Meaning | Training use |
|---|---|---|
| `typological-boundary` | Models agree it is SoC; disagree on which type | Multi-label positive |
| `detection-boundary` | Some models say SoC, others say no | Uncertainty / skepticism signal |
| `clean-consensus` | All models agree on type | Standard single-label positive |
| `noisy` | 3+ distinct type readings | Excluded from training |

> **Current status:** ~3 553 records available from mercury-2 (complete) and
> gpt-5.4 (complete); qwen3-max resampling is in progress (5/41 source files
> done as of 2026-06-16). Run `build_ambiguity_dataset.py --dry-run` to check
> current coverage before building the index.

### 2.6 — Generate interpretive traces for boundary cases (`scripts/generate_traces.py`)

Optional but recommended for v3. Replaces the boilerplate stub
`<think>` content in negative and boundary records with genuine interpretive
reasoning generated by a frontier model (scored on grounding + typological
specificity).

```bash
# Generate traces for all splits (prioritises negatives and boundary cases)
python scripts/generate_traces.py \
    --dataset    dataset/ \
    --output     training_data/traces/ \
    --model      mercury-2 \
    --candidates 6 \
    --temperature 0.8

# Specific split only
python scripts/generate_traces.py --dataset dataset/ --output training_data/traces/ --split train
```

After generating traces, rebuild the dataset to fold them in:

```bash
python scripts/build_dataset.py \
    --silver-dir silver_data/ \
    --neg-dir    negatives/ \
    --output     dataset/ \
    --ambiguity-index silver_data/ambiguity_index.jsonl
```

> **Note:** the `--ambiguity-index` flag is the v3 addition to
> `build_dataset.py`. It folds `typological-boundary` passages into the
> training set as multi-label records (multiple `instances` per passage) and
> `detection-boundary` passages as low-confidence positives or explicit
> uncertainty examples. This flag is not yet implemented — see REFACTOR.md.

### 3. Fine-tune (`scripts/train.py`)

QLoRA supervised fine-tuning of `Qwen/Qwen3-4B` on the generated dataset.
For v3, the `<think>` reasoning trace is included in the loss at a reduced
weight (default `--think-loss-weight 0.4`), so the model learns the
interpretive reasoning process as well as the structured JSON output.

| Flag | Behaviour |
|---|---|
| *(default)* `--think-loss-weight 0.4` | Reasoning trace at 40% weight; JSON at 100% — **v3 default** |
| `--think-loss-weight 1.0` | Equal weight on trace and output |
| `--think-loss-weight 0.0` or `--mask-thinking` | Trace excluded from loss — v1/v2 behaviour |

```bash
# v3 training (reasoning trace included at reduced weight)
python scripts/train.py \
    --dataset          dataset/ \
    --base-model       Qwen/Qwen3-4B \
    --adapter-output   adapter-v3/ \
    --epochs           3 \
    --batch-size       2 \
    --grad-accum       8 \
    --lr               2e-4 \
    --lora-rank        64 \
    --max-seq-len      4096 \
    --think-loss-weight 0.4

# Reproduce v1/v2 behaviour (reasoning trace masked from loss)
python scripts/train.py \
    --dataset          dataset/ \
    --adapter-output   adapter-sft/ \
    --mask-thinking
```

On Princeton Della, submit as a batch job:

```bash
sbatch train_della.slurm     # A100 80 GB, ~2 h for 3 epochs on ~500 examples
```

### 3.5 — GRPO training (`scripts/train_grpo.py`)

Reinforcement learning from a vLLM-hosted Qwen3.6-27B judge. Requires the SFT
adapter from step 3 and a running judge server (1×A100 80GB).

```bash
# 1. Start the judge server (~3–5 min to load Qwen3.6-27B)
JUDGE_JOB=$(sbatch --parsable judge_della.slurm)

# 2. Submit GRPO training — starts after judge is allocated
sbatch --dependency=after:$JUDGE_JOB train_grpo.slurm
```

Reward function (defined in `SKILL.md`):

| Component | Weight | Scorer |
|---|---|---|
| grounding | 0.2 | auto (verbatim phrase match) |
| skepticism | 0.2 | LLM judge |
| specificity | 0.2 | auto (taxonomy keyword match) |
| type coherence | 0.4 | LLM judge |

Output adapter is written to `adapter-grpo/`. Monitor KL divergence during
training — a spike indicates reward hacking.

### 4. Evaluate (`scripts/evaluate.py`)

Per-class F1, confusion matrix, and comparison against zero-shot Qwen3-4B
and the silver majority-vote baseline:

```bash
python scripts/evaluate.py \
    --base-model Qwen/Qwen3-4B \
    --adapter    adapter/ \
    --dataset    dataset/ \
    --output     eval_results.json
```

On Della:

```bash
sbatch evaluate_della.slurm
``` 

### 5. Export and publish (`scripts/export_model.py`)

Merge the LoRA adapter into the base model and optionally push to HF Hub:

```bash
python scripts/export_model.py \
    --base-model Qwen/Qwen3-4B \
    --adapter    adapter/ \
    --output     models/penelope-soc-v1 \
    --push-to-hub \
    --hub-id     apjanco/penelope-soc-v1
```

After export, update `model_config.yaml` to point at the new path, then run
inference as normal with `run.py`.

---

## Output

`results/results.csv` and `results/results.json` — one row per SoC instance:

| Column | Description |
|---|---|
| `source_file` | Original filename |
| `chunk_id` | Chunk identifier |
| `passage` | Verbatim quoted SoC passage |
| `soc_type` | Primary classification |
| `secondary_devices` | Layered techniques |
| `affective_register` | Emotional register (for simulation_state_of_mind) |
| `narrator_position` | absent / minimal / present / dominant |
| `character_pov` | Character whose consciousness is rendered |
| `explanation` | Model's 1–2 sentence reasoning |
| `evidence` | Textual features supporting the classification |
| `confidence` | high / medium / low |
| `notes` | Ambiguity or hybrid transition notes |
| `is_soc` | Boolean — always true in results (false rows filtered out) |

---

## SoC taxonomy

Defined in [`SKILL.md`](SKILL.md), synthesizing Humphrey (1954) and Steinberg (1973):

| Type | Description |
|---|---|
| `direct_interior_monologue` | Unmediated first-person thought; no narrator frame; may fragment |
| `indirect_interior_monologue` | Third-person syntax, character's diction; free indirect discourse |
| `omniscient_description` | Narrator describes psychology from outside with authority |
| `free_association` | Logic-defying leaps via sensory, phonetic, or emotional links |
| `orthographic_marker` | Italics, missing punctuation, or typography signals consciousness |
| `imagery` | Dense sensory language where perception *is* the consciousness |
| `simulation_state_of_mind` | Syntax/rhythm formally enacts an emotional state |
| `reverie_fantasy` | Character constructs an imagined scene; future/conditional diction |
| `hybrid` | Passage shifts between two or more types |
| `other_soc` | Recognised technique not captured above (incl. soliloquy, space-montage) |

---

## Chunking tools

Two chunkers are available:

| Tool | Install | Best for |
|---|---|---|
| `scripts/soc_chunker.py` | no extra deps | Chapter/heading detection with sentence-boundary fallback |
| `scripts/soc_chonkie.py` | `pip install chonkie` | Model-specific presets, semantic chunking, pipeline mode |

```bash
# Built-in chunker
python -m scripts.soc_chunker input/mrs_dalloway.txt -o chunks.json

# Chonkie with model preset
python -m scripts.soc_chonkie input/mrs_dalloway.txt --chunker sentence
```

---

## Streamlit dashboard (`app.py`)

Compare results across multiple runs side-by-side:

```bash
streamlit run app.py
```

Passages from different runs are matched by token overlap (35% threshold).

## Deploy to HuggingFace Spaces (`deploy_hf.sh`)

```bash
hf login           # once
./deploy_hf.sh                  # pushes to apjanco/penelope
./deploy_hf.sh myorg/my-space   # custom space
```

---

## Project structure

```
penelope/
├── SKILL.md                  # SoC taxonomy and classification rules
├── SPEC.md                   # Full project specification
├── REFACTOR.md               # Architecture notes for the fine-tuning refactor
├── AMBIGUITY.md              # v3 design: training on model disagreement
├── TRAINING_APPROACH.md      # Training rationale and two-stage pipeline
├── model_config.yaml         # Inference model configuration
├── consensus.yaml            # Consensus track definitions
├── requirements.txt          # Python dependencies
├── chunk.py                  # Step 1: extract + chunk → chunking/
├── run.py                    # Step 2: local model inference → results/
├── consensus.py              # Step 3: merge multi-run results
├── app.py                    # Streamlit comparison dashboard
├── deploy_hf.sh              # Push app to HuggingFace Spaces
├── resample_run.sh           # Login-node helper: run resample.py per model
├── train_della.slurm         # SLURM script: SFT training on Della A100
├── train_grpo.slurm          # SLURM script: GRPO training (depends on judge)
├── judge_della.slurm         # SLURM script: vLLM judge server (1×A100)
├── test_cuda.slurm           # SLURM probe to find best cudatoolkit version
├── input/                    # Raw literary texts (.txt, .docx, .pdf)
├── chunking/                 # Annotated texts with <chunk-N> markup
├── negatives/                # Gutenberg plain-text negatives (pg<ID>_<slug>.txt)
├── positives/                # SoC-positive texts (Gutenberg + manually placed)
├── chunking_positives/       # Chunked positive texts (input to silver.py / annotate.py)
├── training_data/            # Silver-label JSON from 4 large LLMs (legacy)
├── silver_data/              # Silver labels for the current corpus
│   └── resampled/            # k=10 resamples per passage per model
├── dataset/                  # Built JSONL splits (train / val / test)
├── adapter/                  # v1 adapter (think-masking; kept for reference)
├── adapter-sft/              # v2 SFT adapter (full-sequence loss)
├── adapter-v3/               # v3 adapter (weighted think loss + multi-label)
├── adapter-grpo/             # GRPO adapter after interpretive reward training
├── models/                   # Merged model after export_model.py
├── results/                  # CSV/JSON inference output
└── scripts/
    ├── prompts.py                # system prompt constants + JUDGE_PROMPT
    ├── models.py                 # Pydantic schemas (SocInstance, Chunk, etc.)
    ├── config.py                 # Config loader
    ├── extract.py                # Text extraction (.txt, .docx, .pdf)
    ├── soc_chunker.py            # Chunker + markup writer/parser
    ├── soc_chonkie.py            # Chonkie library wrapper
    ├── analyze.py                # Local model inference (replaces external API calls)
    ├── export.py                 # CSV/JSON export + summary
    ├── consensus.py              # Consensus filtering and merging logic
    ├── download_gutenberg.py     # Download curated Gutenberg negatives
    ├── download_positives.py     # Download curated SoC-positive texts
    ├── silver.py                 # Single-pass LLM annotation → silver_data/<model>.json
    ├── resample.py               # k-run resampling → silver_data/resampled/<model>.jsonl
    ├── annotate.py               # Unified silver + resample pipeline (prefer over running separately)
    ├── clean_silver.py           # Post-processing / dedup utilities for silver data
    ├── build_dataset.py          # Silver + negatives → JSONL training splits
    ├── build_ambiguity_dataset.py # Resampled data → ambiguity_index.jsonl (v3)
    ├── generate_traces.py        # Replace boilerplate traces with interpretive reasoning
    ├── infer.py                  # CLI for single-passage inference (Typer)
    ├── train.py                  # QLoRA SFT training (WeightedThinkCollator, --think-loss-weight)
    ├── train_grpo.py             # GRPO training with vLLM judge reward
    ├── evaluate.py               # Per-class F1, confusion matrix
    └── export_model.py           # Merge adapter and push to HF Hub
```

---

## Requirements

- Python 3.12 (recommended; 3.11 also works)
- CUDA 12.x and a GPU with ≥ 8 GB VRAM for training or bf16 inference
  (4-bit quantization lowers inference requirement to ~4–5 GB)
- `transformers>=4.51.0` — required for Qwen3 chat template support
- For training: `bitsandbytes>=0.43.0`, `peft>=0.10.0`, `trl>=0.8.0`

---

## References

- Humphrey, Robert. *Stream of Consciousness in the Modern Novel*. University of
  California Press, 1954.
- Steinberg, Erwin R. *The Stream of Consciousness and Beyond in Ulysses*.
  University of Pittsburgh Press, 1973.
