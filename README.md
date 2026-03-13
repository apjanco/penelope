# Penelope

**Detect and classify stream of consciousness in literary texts using LLMs.**

Penelope is a pipeline that reads a corpus of literary works, segments them into
meaningful chunks, and sends each chunk to one or more LLMs for exhaustive
identification of stream of consciousness (SOC) passages. Every instance is
classified according to a detailed taxonomy drawn from Robert Humphrey and
Erwin R. Steinberg, with support for hybrid forms, ambiguity, and confidence
levels. Results from multiple models can be compared side-by-side.

Named after the final episode of Joyce's *Ulysses* — Molly Bloom's unbroken
interior monologue — the canonical example of stream of consciousness in fiction.

## How it works

```
input/               SKILL.md               models.yaml
  ├── novel_1.txt    (SOC taxonomy &        (LLM profiles:
  ├── novel_2.pdf     classification         gpt-4o, claude,
  └── novel_3.docx    rules)                 local models…)
        │                  │                       │
        ▼                  ▼                       ▼
   ┌─────────┐     ┌────────────┐     ┌──────────────────┐
   │ Extract  │────▶│   Chunk    │────▶│  LLM Analysis    │
   │ .txt     │     │ (chapters/ │     │  (per model,     │
   │ .docx    │     │  headings) │     │   per chunk)     │
   │ .pdf     │     └────────────┘     └──────────────────┘
   └─────────┘                                  │
                                                ▼
                                     ┌────────────────────┐
                                     │  results.csv/.json │
                                     │  (with model_label │
                                     │   for comparison)  │
                                     └────────────────────┘
```

## Quick start

```bash
# Clone
git clone https://github.com/apjanco/penelope.git
cd penelope

# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Test chunking (no API calls)
python run.py --input input/ --dry-run

# Run analysis
python run.py --input input/ --output results/
```

## Configuration

### Single model (`.env` only)

If you just want to run one model, set these in `.env`:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=sk-...
```

### Multiple models (`models.yaml`)

To compare models, define profiles in `models.yaml`:

```yaml
models:
  - label: gpt-4o
    base_url: https://api.openai.com/v1
    model_name: gpt-4o
    api_key: ${OPENAI_API_KEY}

  - label: claude-sonnet
    base_url: https://api.anthropic.com/v1
    model_name: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  - label: qwen3-8b
    base_url: http://localhost:11434/v1
    model_name: qwen3:8b
    api_key: not-needed
```

API keys use `${ENV_VAR}` syntax and are resolved from your `.env` or shell environment.

## Usage

```bash
# Full corpus, all models
python run.py --input input/ --output results/

# Single file
python run.py --input input/mrs_dalloway.txt --output results/

# Specific models only
python run.py --input input/ --model gpt-4o --model claude-sonnet

# CSV only
python run.py --input input/ --format csv

# Custom chunk size
python run.py --input input/ --chunk-size 15000 --chunk-overlap 800

# Verbose logging
python run.py --input input/ -v
```

### Dry run

Test text extraction and chunking without making any API calls:

```bash
python run.py --input input/ --dry-run
```

### Chunking tools

Two chunkers are included:

| Tool | Install | Best for |
|---|---|---|
| `scripts/soc_chunker.py` | No dependencies | Chapter/heading detection with sentence-boundary fallback |
| `scripts/soc_chonkie.py` | `pip install chonkie` | Model-specific presets, semantic chunking, pipeline mode |

```bash
# Built-in chunker
python -m scripts.soc_chunker input/mrs_dalloway.txt -o chunks.json

# Chonkie with model preset
python -m scripts.soc_chonkie input/mrs_dalloway.txt --model claude-sonnet --chunker sentence
```

## Output

Results are written to `results/results.csv` and `results/results.json` with these columns:

| Column | Description |
|---|---|
| `model_label` | Which LLM produced this row |
| `source_file` | Original filename |
| `chunk_id` | Chunk identifier |
| `passage` | Verbatim quoted SOC passage |
| `soc_type` | Primary classification (see taxonomy below) |
| `secondary_devices` | Layered techniques |
| `narrator_position` | absent / minimal / present / dominant |
| `character_pov` | Character whose consciousness is rendered |
| `explanation` | LLM's reasoning |
| `confidence` | high / medium / low |
| `notes` | Ambiguity, hybrid transitions |

## SOC Taxonomy

The classification system in [`SKILL.md`](SKILL.md) synthesizes Humphrey (1954) and Steinberg:

**Core types (Humphrey)**
1. Direct Interior Monologue
2. Indirect Interior Monologue
3. Omniscient Description of Consciousness
4. Soliloquy

**Additional devices**
5. Free Association
6. Space-Montage
7. Orthographic/Typographic Markers
8. Imagery / Literary Impressionism
9. Simulation of a State of Mind (Steinberg)
10. Reverie / Constructive Fantasy

See `SKILL.md` for full definitions, key markers, negative examples, gray-area
guidance, and the classification procedure.

## Project structure

```
penelope/
├── SKILL.md              # SOC taxonomy, classification rules, LLM prompt
├── SPEC.md               # Full project specification
├── models.yaml           # Multi-model LLM configuration
├── .env.example          # API key template
├── requirements.txt      # Python dependencies
├── run.py                # CLI entry point
└── scripts/
    ├── config.py         # Config loader (.env + models.yaml)
    ├── models.py         # Pydantic schemas
    ├── extract.py        # Text extraction (.txt, .docx, .pdf)
    ├── soc_chunker.py    # Built-in chapter/heading chunker
    ├── soc_chonkie.py    # Chonkie library wrapper
    ├── analyze.py        # LLM analysis + response parsing
    └── export.py         # CSV/JSON export + summary
```

## Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint (OpenAI, Anthropic, Ollama, vLLM, etc.)

## License

TBD

## References

- Humphrey, Robert. *Stream of Consciousness in the Modern Novel*. University of
  California Press, 1954.
- Steinberg, Erwin R. *The Stream of Consciousness and Beyond in Ulysses*.
  University of Pittsburgh Press, 1973.
