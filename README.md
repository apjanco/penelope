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
training_data/*.json          negatives/
  (silver labels from    +   (Gutenberg
   4 large LLMs)              plain text)
        │                         │
        └──────────┬──────────────┘
                   ▼
         build_dataset.py  →  dataset/train.jsonl
                                       dataset/val.jsonl
                                       dataset/test.jsonl
                   ▼
             train.py       →  adapter/
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

# 3. Extract and chunk your texts
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

### 2. Build the dataset (`scripts/build_dataset.py`)

Merges the four silver-label JSON files in `training_data/` with Gutenberg
negatives, applies consensus filtering, collapses rare types, and splits into
train / val / test JSONL files.

```bash
python scripts/build_dataset.py \
    --silver-dir training_data/ \
    --neg-dir    negatives/ \
    --output     dataset/ \
    --track      conservative     # conservative | moderate | liberal
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

### 3. Fine-tune (`scripts/train.py`)

QLoRA supervised fine-tuning of `Qwen/Qwen3-4B` on the constructed dataset.
A custom `ThinkMaskingCollator` masks `<think>…</think>` tokens from the loss
so that only the JSON output tokens receive gradient signal.

```bash
python scripts/train.py \
    --dataset        dataset/ \
    --base-model     Qwen/Qwen3-4B \
    --adapter-output adapter/ \
    --epochs         3 \
    --batch-size     2 \
    --grad-accum     8 \
    --lr             2e-4 \
    --lora-rank      64 \
    --max-seq-len    4096
```

On Princeton Della, submit as a batch job:

```bash
sbatch train_della.slurm     # A100 80 GB, ~2 h for 3 epochs on ~500 examples
```

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
huggingface-cli login           # once
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
├── model_config.yaml         # Inference model configuration
├── consensus.yaml            # Consensus track definitions
├── requirements.txt          # Python dependencies
├── chunk.py                  # Step 1: extract + chunk → chunking/
├── run.py                    # Step 2: local model inference → results/
├── consensus.py              # Step 3: merge multi-run results
├── app.py                    # Streamlit comparison dashboard
├── deploy_hf.sh              # Push app to HuggingFace Spaces
├── train_della.slurm         # SLURM batch script for Princeton Della
├── test_cuda.slurm           # SLURM probe to find best cudatoolkit version
├── input/                    # Raw literary texts (.txt, .docx, .pdf)
├── chunking/                 # Annotated texts with <chunk-N> markup
├── negatives/                # Gutenberg plain-text negatives (pg<ID>_<slug>.txt)
├── training_data/            # Silver-label JSON from 4 large LLMs
├── dataset/                  # Built JSONL splits (train / val / test)
├── adapter/                  # LoRA adapter weights after training
├── models/                   # Merged model after export_model.py
├── results/                  # CSV/JSON inference output
└── scripts/
    ├── prompts.py                # TRAINING_ and INFERENCE_SYSTEM_PROMPT constants
    ├── models.py                 # Pydantic schemas (SocInstance, Chunk, etc.)
    ├── config.py                 # Config loader
    ├── extract.py                # Text extraction (.txt, .docx, .pdf)
    ├── soc_chunker.py            # Chunker + markup writer/parser
    ├── soc_chonkie.py            # Chonkie library wrapper
    ├── analyze.py                # Local model inference (replaces external API calls)
    ├── export.py                 # CSV/JSON export + summary
    ├── consensus.py              # Consensus filtering and merging logic
    ├── download_gutenberg.py     # Download curated Gutenberg negatives
    ├── build_dataset.py          # Silver + negatives → JSONL training splits
    ├── train.py                  # QLoRA SFT training loop
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
