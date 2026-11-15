"""Penelope — passage-grounded teacher gold (Stage A of TEACHER_GOLD.md).

Re-labels every chunk from the source text with a single strong local teacher
(served by teacher_della.slurm via vLLM), emitting per-instance, verbatim-quote-
anchored, type-contrastive gold that a schema-echoing student cannot fake.

The teacher is shown the source in the SAME representation the student is trained
in: the system prompt is TRAINING_SYSTEM_PROMPT (taxonomy + schema) plus the
differentiation/grounding rules below. The user turn omits the `Source:` filename
so the teacher never sees a leak cue (POST-TRAINING §2).

Every returned instance passes a hard grounding gate — its `evidence` phrases and
its `passage` must appear VERBATIM (whitespace/case-normalized) in the chunk text.
Ungrounded output triggers a corrective retry; anything still ungrounded after the
retries is dropped rather than written.

Output is written in the silver-record schema (the same 15 fields silver.py emits)
so build_dataset.py and diagnose_gold.py ingest it unchanged. Progress is appended
to <output>.jsonl keyed by chunk_id for resume; the flat <output>.json array is
compiled at the end.

Usage
-----
    # smoke test (first 32 chunks), then inspect training_data/teacher.json
    python scripts/teacher_gold.py \\
        --input chunking_positives/ \\
        --output training_data/teacher.json \\
        --limit 32 --resume

    # full run (endpoint auto-discovered from teacher_endpoint.txt)
    python scripts/teacher_gold.py --input chunking_positives/ --resume

    # explicit endpoint / model
    python scripts/teacher_gold.py --input chunking_positives/ \\
        --base-url http://della-gpu:8000/v1 --model Qwen/Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

# Prepend project root so we can import scripts.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.models import Chunk, InferenceResponse, SocInstance, SocType
from scripts.prompts import TRAINING_SYSTEM_PROMPT
from scripts.soc_chunker import is_chunked_file, parse_chunked_dir, parse_chunked_file

# ---------------------------------------------------------------------------
# Teacher system prompt — TRAINING_SYSTEM_PROMPT + differentiation/grounding
# rules (TEACHER_GOLD.md §4.1). The rules are the text analog of the vision
# teacher's "differentiate per page, do not copy a coarse date onto every page".
# ---------------------------------------------------------------------------

_DIFFERENTIATION_RULES = """\

## Labeling rules (teacher gold)

You are creating training labels for a stream-of-consciousness classifier. You
are shown ONE chunk of a literary text. Identify every SoC passage and, for EACH,
classify it using the taxonomy above.

- GROUND every classification in the text. The `passage` must be copied VERBATIM
  from the chunk, and the `evidence` array must contain VERBATIM phrases copied
  exactly from that passage (matching words, punctuation, and any orthographic
  markers). Do not paraphrase the passage or the evidence.
- DIFFERENTIATE. A chunk usually contains more than one technique. Give two
  instances the same `soc_type` only when the text genuinely supports it. Do not
  apply one label to the whole chunk.
- ARGUE the type. In `explanation`, name the ONE alternative type a careful reader
  might choose instead, and say what in the text rules it out. A label with no
  contrast is not acceptable.
- ABSTAIN natively. If a span reads like conventional narrated psychology rather
  than unmediated consciousness, mark `is_soc: false` and say what it would need
  to qualify.
- Use `confidence: low` with a wide interpretive hedge rather than fabricating a
  crisp label you cannot defend from the text.

Return ONLY the JSON object matching the schema. One entry per distinct passage.
"""

TEACHER_SYSTEM_PROMPT: str = TRAINING_SYSTEM_PROMPT + _DIFFERENTIATION_RULES

_VALID_SOC_TYPES = {t.value for t in SocType}


# ---------------------------------------------------------------------------
# User-turn construction — same framing as silver.py, minus the `Source:` leak.
# ---------------------------------------------------------------------------

def _render_user_content(
    chunk_id: str,
    chunk_label: str,
    chunk_index: int,
    context_before: str,
    context_after: str,
    chunk_text: str,
) -> str:
    user_content = f"Chunk: {chunk_id} ({chunk_label})\nPosition: {chunk_index}\n"
    if context_before:
        user_content += f"\n[CONTEXT — end of previous chunk]\n{context_before}\n"
    user_content += f"\n[TEXT TO ANALYZE]\n{chunk_text}\n"
    if context_after:
        user_content += f"\n[CONTEXT — start of next chunk]\n{context_after}\n"
    user_content += (
        "\nAnalyze the [TEXT TO ANALYZE] section above. "
        "Identify and classify every SoC passage. "
        "Respond with JSON only."
    )
    return user_content


def _build_user_content(chunk: Chunk) -> str:
    return _render_user_content(
        chunk.chunk_id, chunk.chunk_label, chunk.chunk_index,
        chunk.context_before, chunk.context_after, chunk.chunk_text,
    )


# ---------------------------------------------------------------------------
# Token-budget fitting — trims context/text so system+user fits the server's
# --max-model-len minus the requested completion. Counting is delegated to
# _TokenCounter, which asks the live vLLM server's own /tokenize endpoint
# (the ground truth for what it will actually count — chat template, special
# tokens, and all) rather than approximating with a locally-loaded tokenizer.
# A local HF tokenizer loaded from the same cache is kept only as a fallback
# if /tokenize is ever unreachable. This runs BEFORE the request goes out, so
# oversized chunks never round-trip into a guaranteed 400 (which the old code
# retried 3x, each retry making the prompt bigger by appending the failed
# turn instead of smaller).
# ---------------------------------------------------------------------------

_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer(model: str):
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer

                logger.info("Loading local fallback tokenizer for %s...", model)
                _tokenizer = AutoTokenizer.from_pretrained(model)
    return _tokenizer


class _TokenCounter:
    """Counts prompt tokens for a chat `messages` list via the vLLM server's
    own /tokenize endpoint. Falls back to a local tokenizer render (chat
    template applied, then encoded) if that endpoint errors."""

    def __init__(self, base_url: str, model: str, http_client):
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        self._tokenize_url = url + "/tokenize"
        self._model = model
        self._http = http_client
        self._warned = False

    def __call__(self, messages: list[dict]) -> int:
        try:
            resp = self._http.post(
                self._tokenize_url,
                json={
                    "model": self._model,
                    "messages": messages,
                    "add_generation_prompt": True,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if "count" in data:
                return data["count"]
            if "tokens" in data:
                return len(data["tokens"])
            raise KeyError(f"no 'count'/'tokens' in /tokenize response: {data!r}")
        except Exception as exc:
            if not self._warned:
                logger.warning(
                    "  /tokenize endpoint unavailable (%s) — falling back to "
                    "local tokenizer estimate (less accurate)", exc,
                )
                self._warned = True
            tok = _get_tokenizer(self._model)
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return len(tok.encode(rendered))


def _fit_chunk_to_budget(
    count_fn, chunk: Chunk, *, max_model_len: int, max_tokens: int
) -> tuple[str, bool]:
    """Return (user_content, was_trimmed), fit against the server's own
    token count so system+user stays within max_model_len - max_tokens.

    Reserves extra headroom (beyond the hard budget) for a possible
    corrective retry turn — otherwise a chunk fit right up to the edge
    leaves zero room for even a short "that was invalid, retry" message,
    and _shrink_conversation_if_needed has nothing left to shrink."""
    margin = 320
    budget = max_model_len - max_tokens - margin

    def render(cb: str, ca: str, ct: str) -> str:
        return _render_user_content(
            chunk.chunk_id, chunk.chunk_label, chunk.chunk_index, cb, ca, ct,
        )

    def fits(user_content: str) -> bool:
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return count_fn(messages) <= budget

    cb, ca, ct = chunk.context_before, chunk.context_after, chunk.chunk_text
    user_content = render(cb, ca, ct)
    if fits(user_content):
        return user_content, False

    # 1. Drop trailing context first — cheapest, least informative loss.
    ca = ""
    user_content = render(cb, ca, ct)
    if fits(user_content):
        return user_content, True

    # 2. Drop leading context too.
    cb = ""
    user_content = render(cb, ca, ct)
    if fits(user_content):
        return user_content, True

    # 3. The chunk text itself is too long — binary search the longest
    # verbatim character prefix whose rendered prompt fits, each probe
    # verified against the live server (no token-boundary guessing needed).
    lo, hi = 0, len(ct)
    best = render("", "", "")
    for _ in range(16):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        candidate = render("", "", ct[:mid])
        if fits(candidate):
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


# ---------------------------------------------------------------------------
# JSON extraction — balanced-brace scan, tolerant of prose / fences / <think>.
# ---------------------------------------------------------------------------

def _extract_first_json(raw: str) -> tuple[dict | None, str]:
    """Return the first balanced-brace JSON object found AFTER the model's
    reasoning, or (None, err).

    vLLM's chat template seeds the opening `<think>` tag into the PROMPT (not
    the completion) for this hybrid-thinking model, so `raw` only ever
    contains a closing `</think>` with no matching open tag — a plain
    `<think>.*?</think>` strip is a no-op here. Splitting on the LAST
    `</think>` instead handles both that case and the (harmless) case where
    a paired tag is present. This matters beyond cosmetics: reasoning models
    often sketch a draft/example JSON mid-thought (sometimes trailing off
    with a literal "..."), and without this split the balanced-brace scanner
    below would happily parse that draft instead of the real answer.
    """
    if not raw:
        return None, "empty response"
    cleaned = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    cleaned = re.sub(r"```(?:json)?", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None, "no '{' found in response"

    depth = 0
    in_str = False
    esc = False
    out: list[str] = []
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch in "\n\r\t":
                # Raw control character inside a string literal — a common
                # LLM JSON-generation slip (unescaped newline/tab in a long
                # quoted field). Repair it in place rather than failing.
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
                continue
            out.append(ch)
            continue
        out.append(ch)
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = "".join(out)
                try:
                    return json.loads(candidate), ""
                except json.JSONDecodeError as exc:
                    return None, f"json decode: {exc}"
    return None, "unbalanced braces"


# ---------------------------------------------------------------------------
# Grounding gate — evidence + passage must appear verbatim in chunk_text.
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_verbatim(needle: str, haystack_norm: str) -> bool:
    needle = _normalise(needle)
    if len(needle) < 3:  # too short to be meaningful evidence
        return False
    return needle in haystack_norm


def _grounded_instances(
    resp: InferenceResponse, chunk_text: str
) -> tuple[list[SocInstance], list[str]]:
    """Split instances into grounded / ungrounded-reasons against chunk_text."""
    haystack = _normalise(chunk_text)
    kept: list[SocInstance] = []
    problems: list[str] = []

    for idx, inst in enumerate(resp.instances):
        if inst.soc_type not in _VALID_SOC_TYPES:
            problems.append(f"instance {idx}: soc_type '{inst.soc_type}' not in taxonomy")
            continue
        if not inst.is_soc:
            # Native abstention is legitimate; keep it (no verbatim requirement).
            kept.append(inst)
            continue
        if not _is_verbatim(inst.passage, haystack):
            problems.append(f"instance {idx}: passage not verbatim in chunk")
            continue
        ungrounded_ev = [e for e in inst.evidence if not _is_verbatim(e, haystack)]
        if ungrounded_ev:
            problems.append(
                f"instance {idx}: evidence not verbatim: {ungrounded_ev[:2]}"
            )
            continue
        kept.append(inst)

    return kept, problems


# ---------------------------------------------------------------------------
# Teacher call with corrective retries.
# ---------------------------------------------------------------------------

def _validate(obj: dict) -> tuple[InferenceResponse | None, str]:
    """Validate raw JSON into InferenceResponse, accepting both key spellings."""
    if "instances" not in obj and "soc_instances" in obj:
        obj = {**obj, "instances": obj["soc_instances"]}
    try:
        from pydantic import ValidationError

        return InferenceResponse.model_validate(obj), ""
    except ValidationError as exc:
        first = str(exc).splitlines()[0]
        return None, f"schema: {first}"


def _is_context_length_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "maximum context length" in msg or "context_length_exceeded" in msg


def _shrink_conversation_if_needed(
    count_fn, messages: list[dict], *, max_model_len: int, max_tokens: int
) -> list[dict]:
    """Corrective retries append an assistant echo + a follow-up turn each
    round; on a mid-sized chunk that's already near budget, two rounds of
    that growth alone can push a request over max_model_len even though the
    original chunk fit fine. Collapse back to system + original user + the
    latest correction the moment growth threatens the budget, instead of
    letting it fail deterministically on the next call."""
    if count_fn is None or len(messages) <= 2:
        return messages
    margin = 32
    budget = max_model_len - max_tokens - margin
    if count_fn(messages) <= budget:
        return messages
    system, user0, last_correction = messages[0], messages[1], messages[-1]
    return [system, user0, last_correction]


def teacher_relabel(
    client,
    model: str,
    chunk: Chunk,
    *,
    temperature: float,
    max_tokens: int,
    retries: int,
    count_fn=None,
    max_model_len: int = 8192,
    verbose: bool = False,
) -> tuple[list[SocInstance], str]:
    """Return (grounded/differentiated instances, raw model response) for one
    chunk. The raw response — reasoning trace and all — is returned verbatim
    alongside the parsed instances so it can be persisted in the output; this
    project's teacher-gold data is the reasoning trace, not just the labels."""
    trimmed = False
    if count_fn is not None:
        user_content, trimmed = _fit_chunk_to_budget(
            count_fn, chunk, max_model_len=max_model_len, max_tokens=max_tokens,
        )
    else:
        user_content = _build_user_content(chunk)
    if trimmed:
        logger.warning(
            "  %s (%d chars) exceeds the model's context budget — trimmed "
            "context/text before sending", chunk.chunk_id, len(chunk.chunk_text),
        )

    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_kept: list[SocInstance] = []
    last_raw = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if _is_context_length_error(exc):
                # Deterministic — retrying the same (or a growing) prompt can
                # only fail again. The pre-flight fit above should prevent
                # this in the normal case; if it still happens (tokenizer
                # skew, chat-template overhead), don't burn the retry budget.
                logger.warning(
                    "  %s: still over context length after pre-fit trim (%s) — "
                    "dropping chunk", chunk.chunk_id, exc,
                )
                return last_kept, last_raw
            logger.warning("  teacher call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(3.0)
            continue

        raw = response.choices[0].message.content or ""
        last_raw = raw  # always keep the most recent response, even if parsing fails entirely
        if verbose:
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            logger.info(
                "  %s (attempt %d) finish_reason=%s usage=%s raw response (%d chars):\n%s",
                chunk.chunk_id, attempt + 1, finish_reason, usage, len(raw), raw[:3000],
            )
        obj, err = _extract_first_json(raw)
        if obj is not None:
            resp, verr = _validate(obj)
            if resp is not None:
                kept, problems = _grounded_instances(resp, chunk.chunk_text)
                if not problems:
                    return kept, raw
                last_kept = kept  # remember grounded subset in case we run out
                last_raw = raw
                logger.info(
                    "  %s: %d/%d instance(s) ungrounded (attempt %d): %s",
                    chunk.chunk_id, len(problems), len(resp.instances),
                    attempt + 1, "; ".join(problems[:3]),
                )
                err = "; ".join(problems[:3])
            else:
                err = verr

        # Corrective follow-up that keeps the chunk in context.
        messages += [
            {"role": "assistant", "content": raw[:4000]},
            {
                "role": "user",
                "content": (
                    f"That was invalid ({err}). Return ONLY corrected JSON: each "
                    f"instance needs a soc_type from the taxonomy, an is_soc flag, a "
                    f"`passage` copied VERBATIM from the chunk, an `evidence` array "
                    f"whose strings appear VERBATIM in that passage, and an "
                    f"`explanation` that names and rules out one alternative type."
                ),
            },
        ]
        messages = _shrink_conversation_if_needed(
            count_fn, messages, max_model_len=max_model_len, max_tokens=max_tokens,
        )

    if last_kept:
        logger.info("  kept %d grounded instance(s) after retries", len(last_kept))
    return last_kept, last_raw


# ---------------------------------------------------------------------------
# Record conversion — silver schema (matches silver.py::_parse_instances).
# ---------------------------------------------------------------------------

def _instance_to_record(inst: SocInstance, chunk: Chunk, raw_response: str) -> dict:
    return {
        "model_label": "teacher",
        "source_file": chunk.source_file,
        "chunk_id": chunk.chunk_id,
        "chunk_label": chunk.chunk_label,
        "chunk_index": chunk.chunk_index,
        "is_soc": inst.is_soc,
        "passage": inst.passage,
        "soc_type": inst.soc_type,
        "secondary_devices": ", ".join(inst.secondary_devices),
        "affective_register": inst.affective_register,
        "narrator_position": inst.narrator_position,
        "character_pov": inst.character_pov,
        "explanation": inst.explanation,
        "evidence": ", ".join(inst.evidence),
        "confidence": inst.confidence,
        "notes": inst.notes,
        # Verbatim model output (reasoning trace + JSON) that produced this
        # instance — the reasoning trace itself is this project's deliverable,
        # not just the parsed labels.
        "raw_reasoning": raw_response,
    }


# ---------------------------------------------------------------------------
# Resumable checkpoint — <output>.jsonl keyed by chunk_id.
# ---------------------------------------------------------------------------

def _load_done(checkpoint: Path) -> set[str]:
    done: set[str] = set()
    if not checkpoint.exists():
        return done
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["chunk_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _append_checkpoint(
    checkpoint: Path, chunk_id: str, records: list[dict], raw_response: str = ""
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"chunk_id": chunk_id, "records": records, "raw_response": raw_response},
            ensure_ascii=False,
        ) + "\n")


def _compile_output(checkpoint: Path, output: Path) -> int:
    """Flatten the checkpoint jsonl into the silver-schema json array."""
    all_records: list[dict] = []
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            all_records.extend(json.loads(line).get("records", []))
        except json.JSONDecodeError:
            continue
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(all_records)


# ---------------------------------------------------------------------------
# Endpoint discovery.
# ---------------------------------------------------------------------------

def _discover_base_url(explicit: str | None, endpoint_file: Path) -> str:
    if explicit:
        return explicit
    if endpoint_file.exists():
        url = endpoint_file.read_text(encoding="utf-8").strip()
        if url:
            return url
    logger.error(
        "No --base-url given and %s is empty/missing. Start teacher_della.slurm "
        "or pass --base-url.", endpoint_file,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Passage-grounded teacher gold (TEACHER_GOLD.md Stage A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path("chunking_positives"),
                        help="Chunked file or dir with <chunk-N> markup (default: chunking_positives/)")
    parser.add_argument("--output", "-o", type=Path, default=Path("training_data/teacher.json"),
                        help="Output JSON array (default: training_data/teacher.json)")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3.6-27B",
                        help="Served model name (default: Qwen/Qwen3.6-27B)")
    parser.add_argument("--base-url", default=None,
                        help="vLLM endpoint (default: read teacher_endpoint.txt)")
    parser.add_argument("--endpoint-file", type=Path, default=Path("teacher_endpoint.txt"),
                        help="File published by teacher_della.slurm (default: teacher_endpoint.txt)")
    parser.add_argument("--api-key", default="EMPTY", help="API key (vLLM ignores it)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Low, for grounded discriminative labeling (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Must match the vLLM server's --max-model-len (default: 8192)")
    parser.add_argument("--retries", type=int, default=2, help="Corrective retries (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N pending chunks (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks already in the .jsonl checkpoint")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between chunks (default: 0.0; ignored if --concurrency > 1)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel in-flight requests to the vLLM server (default: 8)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log the model's full raw response for every attempt (debugging)")
    args = parser.parse_args()

    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai>=1.0.0")
        sys.exit(1)

    base_url = _discover_base_url(args.base_url, args.endpoint_file)
    logger.info("Teacher endpoint: %s  model: %s", base_url, args.model)
    client = OpenAI(base_url=base_url, api_key=args.api_key)
    count_fn = _TokenCounter(base_url, args.model, httpx.Client())

    # ── Load chunks ──────────────────────────────────────────────────
    if args.input.is_file():
        if not is_chunked_file(args.input):
            logger.error("%s has no <chunk-N> markup. Run chunk.py first.", args.input)
            sys.exit(1)
        chunks = parse_chunked_file(args.input)
    elif args.input.is_dir():
        chunks = parse_chunked_dir(args.input)
    else:
        logger.error("Input path not found: %s", args.input)
        sys.exit(1)

    if not chunks:
        logger.error("No chunks found in %s", args.input)
        sys.exit(1)

    checkpoint = args.output.with_suffix(args.output.suffix + ".jsonl")
    done = _load_done(checkpoint) if args.resume else set()

    pending = [c for c in chunks if c.chunk_id not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    logger.info(
        "Chunks: %d total, %d already done, %d to process%s",
        len(chunks), len(done), len(pending),
        f" (limited to {args.limit})" if args.limit is not None else "",
    )

    # ── Teacher pass ─────────────────────────────────────────────────
    checkpoint_lock = threading.Lock()
    done_count = 0

    def process(chunk: Chunk) -> tuple[Chunk, list[dict], str]:
        instances, raw = teacher_relabel(
            client, args.model, chunk,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            count_fn=count_fn,
            max_model_len=args.max_model_len,
            verbose=args.verbose,
        )
        # Persist abstentions too (is_soc: false), not just confirmed positives —
        # a native "this reads like narrated psychology, not unmediated
        # consciousness" call, with its `notes` explaining what's missing, is
        # itself evidence of where a passage sits relative to the taxonomy's
        # boundary. Downstream consumers should filter on the `is_soc` field.
        records = [_instance_to_record(inst, chunk, raw) for inst in instances]
        return chunk, records, raw

    if args.concurrency <= 1:
        for i, chunk in enumerate(pending, 1):
            logger.info("[%d/%d] %s (%d chars)", i, len(pending), chunk.chunk_id, len(chunk.chunk_text))
            _, records, raw = process(chunk)
            soc_types = [r["soc_type"] for r in records if r["is_soc"]]
            n_abstained = sum(1 for r in records if not r["is_soc"])
            logger.info(
                "  → %d confirmed, %d abstained: %s",
                len(soc_types), n_abstained, ", ".join(soc_types) or "(none)",
            )
            _append_checkpoint(checkpoint, chunk.chunk_id, records, raw)
            if args.delay and i < len(pending):
                time.sleep(args.delay)
    else:
        logger.info("Processing with %d concurrent requests...", args.concurrency)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process, chunk): chunk for chunk in pending}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    _, records, raw = future.result()
                except Exception:
                    logger.exception("  %s: unhandled error, skipping", chunk.chunk_id)
                    records, raw = [], ""
                done_count += 1
                soc_types = [r["soc_type"] for r in records if r["is_soc"]]
                n_abstained = sum(1 for r in records if not r["is_soc"])
                logger.info(
                    "[%d/%d] %s → %d confirmed, %d abstained: %s",
                    done_count, len(pending), chunk.chunk_id, len(soc_types),
                    n_abstained, ", ".join(soc_types) or "(none)",
                )
                with checkpoint_lock:
                    _append_checkpoint(checkpoint, chunk.chunk_id, records, raw)

    total = _compile_output(checkpoint, args.output)
    logger.info("Done. Compiled %d records into %s", total, args.output)


if __name__ == "__main__":
    main()
