# Research Software Specification Sheet

## 1. Project Overview

- **Project Name:** SOC Data — Stream of Consciousness Detection in Literary Texts
- **Principal Investigator(s):**
- **Institution:**
- **Start Date:**
- **Target Completion:**
- **Funding Source:**
- **License:**

### 1.1 Summary

This software is a text-analysis pipeline for identifying and classifying **stream of consciousness (SOC)** passages in literary texts. It ingests a folder of literary works (`.txt`, `.doc`, `.pdf`), segments them into meaningful chunks (chapters, sections, headings), and sends each chunk to an LLM for analysis. The LLM identifies **every instance** of SOC in the text and classifies it by type according to a taxonomy defined in [`SKILL.md`](SKILL.md). Results are aggregated and exported as CSV or JSON for downstream research use.

### 1.2 Goals & Objectives

- Detect **all** instances of stream of consciousness in a corpus of literary texts — exhaustive coverage, not sampling.
- Classify each SOC passage by type (interior monologue, free indirect discourse, sensory impression stream, etc.) per the definitions in `SKILL.md`.
- Produce structured, exportable results (CSV / JSON) suitable for quantitative literary analysis.
- Support configurable LLM backends via the OpenAI-compatible API (variable `base_url`, `model_name`, `api_key`).

---

## 2. Data

### 2.1 Data Sources

| Source | Format | Size (est.) | Access Method | Notes |
|---|---|---|---|---|
| Local folder of literary texts | `.txt`, `.doc`, `.pdf` | Variable | File system read | One file per work |

### 2.2 Data Schema / Structure

**Input:** A directory containing literary text files.

**Intermediate (chunks):**

| Field | Type | Description |
|---|---|---|
| `source_file` | string | Original filename |
| `chunk_id` | string | Unique identifier for the chunk (e.g., `work-title_ch03`) |
| `chunk_label` | string | Human-readable label (e.g., "Chapter 3: The Lighthouse") |
| `chunk_text` | string | Full text of the chunk |
| `chunk_index` | int | Ordinal position in the source work |

**Output (SOC instances):**

| Field | Type | Description |
|---|---|---|
| `source_file` | string | Original filename |
| `chunk_id` | string | Chunk where the SOC was found |
| `passage` | string | The SOC passage text |
| `soc_type` | string | Type of SOC (from `SKILL.md` taxonomy) |
| `explanation` | string | LLM's reasoning for the classification |
| `confidence` | string | Confidence level (high / medium / low) |
| `start_offset` | int | Character offset of passage start within the chunk (if available) |

### 2.3 Data Storage & Management

- **Storage location:** Local filesystem (project directory)
- **Backup strategy:** Git version control for code and config; raw texts managed by researcher
- **Data retention policy:** N/A — researcher-managed literary corpus
- **Sensitive / PII data?** No (published literary texts)

---

## 3. Functional Requirements

### 3.1 Core Features

1. **File ingestion** — Read a folder of `.txt`, `.doc`, and `.pdf` files and extract plain text.
2. **Chunking** — Segment each work into chunks by chapters, headings, or other structural markers. Fall back to fixed-size chunking with overlap if no structure is detected.
3. **LLM analysis** — Send each chunk to an LLM (via OpenAI-compatible API) with the SOC detection prompt derived from `SKILL.md`. The prompt must instruct the model to find **every** SOC instance, not just examples.
4. **Result parsing** — Parse the structured LLM response into the output schema above.
5. **Aggregation & export** — Combine results from all chunks across all works and export as **CSV** or **JSON**.
6. **Configuration** — Configurable `base_url`, `model_name`, and `api_key` for the LLM backend (via environment variables or config file).

### 3.2 Nice-to-Have Features

1. Progress bar / logging for long-running corpus processing.
2. Retry logic and rate-limit handling for LLM API calls.
3. Caching of LLM responses to avoid re-processing unchanged chunks.
4. Parallel / async chunk processing for throughput.
5. A summary report (counts by SOC type, by work, etc.).

### 3.3 User Workflows

- **Workflow 1 — Full corpus analysis:**
  1. User places literary text files in an `input/` folder.
  2. User sets LLM connection variables (`base_url`, `model_name`, `api_key`).
  3. User runs the pipeline (e.g., `python run.py --input input/ --output results/`).
  4. Software ingests files → chunks → LLM analysis → exports `results.csv` and/or `results.json`.

- **Workflow 2 — Single-text analysis:**
  1. User points the tool at a single file.
  2. Same pipeline, scoped to one work.

---

## 4. Technical Architecture

### 4.1 Language(s) & Frameworks

- **Python 3.10+**

### 4.2 Dependencies & Libraries

| Dependency | Purpose |
|---|---|
| `openai` | OpenAI-compatible Python client (generic — works with any `base_url`) |
| `python-docx` | Extract text from `.doc` / `.docx` files |
| `PyPDF2` or `pdfplumber` | Extract text from `.pdf` files |
| `pydantic` | Validate and structure LLM responses |
| `csv` / `json` (stdlib) | Export results |
| `pathlib` (stdlib) | File path handling |
| `argparse` or `click` | CLI interface |
| `python-dotenv` | Load config from `.env` file |
| `tqdm` | Progress bars (optional) |

### 4.3 Infrastructure

- **Deployment target:** Local machine (researcher's workstation)
- **OS compatibility:** Linux, macOS, Windows
- **Hardware requirements:** Minimal — LLM inference is remote via API
- **Containerization:** Optional Docker support

### 4.4 Architecture Diagram

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│  Input       │     │  Text        │     │  Chunking    │     │  LLM         │     │  Export    │
│  Folder      │────▶│  Extraction  │────▶│  (chapters/  │────▶│  Analysis    │────▶│  CSV/JSON  │
│  .txt/.doc/  │     │  (plain text)│     │   headings)  │     │  (OpenAI API)│     │            │
│  .pdf        │     └──────────────┘     └──────────────┘     └──────────────┘     └────────────┘
└─────────────┘                                                       │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  SKILL.md    │
                                                               │  (SOC defs   │
                                                               │  & prompt)   │
                                                               └──────────────┘
```

### 4.5 Configuration

The LLM client is initialized with three configurable values:

```python
from openai import OpenAI

client = OpenAI(
    base_url=BASE_URL,      # e.g., "https://api.openai.com/v1" or a local endpoint
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model=MODEL_NAME,       # e.g., "gpt-4o", "llama-3", etc.
    messages=[...],
)
```

These should be set via a `.env` file or environment variables:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=sk-...
```

---

## 5. User Interface

- **Interface type:** CLI (command-line interface)
- **Target audience:** Literary researchers and digital humanities scholars
- **Accessibility considerations:** Clear terminal output, structured log messages

---

## 6. Testing & Validation

- **Testing framework:** `pytest`
- **Test coverage target:** Core pipeline logic (ingestion, chunking, response parsing, export)
- **Validation approach:** Compare LLM-identified SOC passages against expert-annotated gold-standard texts; qualitative review of classification accuracy
- **Benchmark datasets:** A small curated set of literary passages with known SOC instances

---

## 7. Documentation & Reproducibility

- **User documentation:** README with setup and usage instructions
- **Developer documentation:** Inline docstrings; this spec
- **Citation information:** TBD
- **Reproducibility plan:** Pinned dependencies (`requirements.txt`), cached LLM responses, `SKILL.md` versioned in repo

---

## 8. Timeline & Milestones

| Milestone | Target Date | Description |
|---|---|---|
| v0.1 | | File ingestion + text extraction + chunking |
| v0.2 | | LLM integration + SOC detection with `SKILL.md` prompt |
| v0.3 | | Result parsing + CSV/JSON export |
| v1.0 | | End-to-end pipeline, tested on sample corpus |

---

## 9. Risks & Constraints

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM misses SOC passages (false negatives) | Medium | High | Prompt engineering; overlap in chunks; validation against gold standard |
| LLM hallucinates SOC where none exists (false positives) | Medium | Medium | Require passage quotation in response; human spot-checking |
| PDF text extraction is noisy | Medium | Medium | Use `pdfplumber` for layout-aware extraction; allow manual cleanup |
| API rate limits / cost on large corpora | Medium | Medium | Caching, batching, configurable concurrency |
| Chunk boundaries split an SOC passage | Medium | High | Overlapping chunks; post-processing to merge cross-boundary detections |

---

## 10. References

<!-- Related papers, prior work, similar tools, standards, etc. -->

-
