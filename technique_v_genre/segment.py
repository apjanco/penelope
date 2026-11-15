"""Segment novels into passage-sized units for SoC scoring.

Implements HANDOFF_della.md Step 1, option 1 (paragraph-based windows,
target ~80-200 words). Two paragraph delimiters are used across this corpus
and must be auto-detected per file, not assumed uniform:

- Gutenberg-sourced texts (`pg<id>_*.txt`: Joyce, James, Richardson,
  Faulkner, Ford) have ordinary blank-line paragraph breaks and Gutenberg
  header/footer boilerplate that must be stripped first.
- Every Woolf text in positives/ (not just the "_cleaned"-suffixed ones —
  jacobs_room.txt, the_voyage_out.txt, the_years.txt, flush.txt, and
  between_the_acts.txt too) has been flattened to a single line with no
  newlines at all; paragraph breaks survive only as runs of tab characters.
  Splitting on blank lines alone silently treats an entire Woolf novel as
  one paragraph — checked directly against to_the_lighthouse_cleaned.txt
  (397,235 chars, 0 newlines) before writing this.

Output: one JSONL file per input, `technique_v_genre/passages/<stem>.jsonl`,
plus a combined `technique_v_genre/passages/all_passages.jsonl`. JSONL
rather than the handoff's suggested parquet — pyarrow isn't installed in
this environment and every other artifact in this project (silver_data/,
dataset/, the ambiguity index) is already JSONL; no reason to introduce a
new format for this one table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIVES_DIR = ROOT / "positives"
MANIFEST_PATH = ROOT / "technique_v_genre" / "corpus_manifest.json"
OUTPUT_DIR = ROOT / "technique_v_genre" / "passages"

GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?’”])\s+(?=[A-Z‘“])")

DEFAULT_MIN_WORDS = 80
DEFAULT_TARGET_WORDS = 140
DEFAULT_MAX_WORDS = 200


FRONT_MATTER_MARKERS = (
    "copyright",
    "all rights reserved",
    "manufactured in",
    "published in",
    "random house",
    "distributed in canada",
)


def _looks_like_front_matter(paragraph: str) -> bool:
    lower = paragraph.lower()
    if any(marker in lower for marker in FRONT_MATTER_MARKERS):
        return True
    letters = [c for c in paragraph if c.isalpha()]
    if len(letters) >= 8:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.5:
            return True
    # Plain-text title-page decoration: "=Title=", "_Author_", "[Illustration]",
    # short lines with no sentence-ending punctuation.
    stripped = paragraph.strip()
    if stripped.count("=") >= 2 or stripped.startswith("[") or "[Illustration]" in stripped:
        return True
    words = stripped.split()
    if len(words) <= 12 and not re.search(r"[.!?][\"')]?\s*$", stripped):
        return True
    return False


def strip_gutenberg_boilerplate(text: str) -> str:
    m = GUTENBERG_START_RE.search(text)
    if m:
        text = text[m.end():]
    m = GUTENBERG_END_RE.search(text)
    if m:
        text = text[: m.start()]
    text = text.strip()

    # The Gutenberg START marker ends before the *book's own* title page /
    # copyright block, which isn't part of Gutenberg's boilerplate and
    # survives the strip above. Skip leading paragraphs that look like
    # front matter (all-caps title typography, "Copyright", "Published
    # in...", etc.) rather than boilerplate-stripping again, since these
    # vary per publisher and aren't marked by a fixed delimiter. Uses the
    # same offset-tracking paragraph split as split_paragraphs() below
    # (not a fresh ad hoc regex) so the cut lands on a real paragraph
    # boundary rather than drifting mid-paragraph on irregular blank-line runs.
    paragraphs = split_paragraphs(text)
    cut_at = 0
    for p_start, p_end, p_text in paragraphs[:12]:
        if _looks_like_front_matter(p_text):
            cut_at = p_end
        else:
            break
    return text[cut_at:].strip()


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return (char_start, char_end, paragraph_text) offsets into `text`."""
    blank_line_breaks = len(re.findall(r"\n\s*\n", text))
    newline_count = text.count("\n")
    tab_run_count = len(re.findall(r"\t+", text))

    if blank_line_breaks >= 3:
        delimiter_re = re.compile(r"\n\s*\n+")
    elif newline_count < 5 and tab_run_count >= 3:
        # Flattened single-line file (every Woolf text in positives/):
        # tab runs are the only surviving paragraph marker.
        delimiter_re = re.compile(r"\t+")
    elif newline_count >= 3:
        delimiter_re = re.compile(r"\n+")
    else:
        # No detectable paragraph structure at all — treat whole text as
        # one unit; the word-count-based merge/split step below will still
        # cut it into passage-sized pieces at sentence boundaries.
        return [(0, len(text), text)]

    paragraphs = []
    pos = 0
    for m in delimiter_re.finditer(text):
        chunk = text[pos:m.start()]
        if chunk.strip():
            # locate actual content offsets within the raw slice
            lstripped = len(chunk) - len(chunk.lstrip())
            rstripped = len(chunk.rstrip())
            paragraphs.append((pos + lstripped, pos + rstripped, chunk.strip()))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        lstripped = len(tail) - len(tail.lstrip())
        rstripped = len(tail.rstrip())
        paragraphs.append((pos + lstripped, pos + rstripped, tail.strip()))
    return paragraphs


def hard_wrap(start: int, text: str, max_words: int) -> list[tuple[int, int, str]]:
    """Last-resort word-count wrap for a span with no usable sentence breaks
    (e.g. Faulkner's Benjy/Quentin sections, which run for hundreds of words
    without the punctuation SENTENCE_SPLIT_RE looks for)."""
    words = text.split()
    pieces = []
    for i in range(0, len(words), max_words):
        piece_text = " ".join(words[i : i + max_words])
        pieces.append((start, start + len(piece_text), piece_text))
    return pieces


def split_oversized_paragraph(
    start: int, end: int, text: str, max_words: int
) -> list[tuple[int, int, str]]:
    """Break one long paragraph into sentence-boundary pieces near max_words."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    pieces: list[tuple[int, int, str]] = []
    buf_words = 0
    buf_start = start
    buf_text_parts: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        idx = text.find(sent)  # relative to the paragraph slice `text`
        n_words = len(sent.split())
        if buf_words + n_words > max_words and buf_text_parts:
            piece_text = " ".join(buf_text_parts)
            pieces.append((buf_start, buf_start + len(piece_text), piece_text))
            buf_text_parts = []
            buf_words = 0
            buf_start = start + idx
        buf_text_parts.append(sent)
        buf_words += n_words
    if buf_text_parts:
        piece_text = " ".join(buf_text_parts)
        pieces.append((buf_start, buf_start + len(piece_text), piece_text))
    if not pieces:
        pieces = [(start, end, text)]
    # Sentence splitting can still leave an oversized piece (no usable
    # punctuation in that span) — fall back to a hard word-count wrap for
    # those specifically, rather than shipping a 700+ word "passage".
    final: list[tuple[int, int, str]] = []
    for p_start, p_end, p_text in pieces:
        if len(p_text.split()) > max_words * 1.5:
            final.extend(hard_wrap(p_start, p_text, max_words))
        else:
            final.append((p_start, p_end, p_text))
    return final


def merge_paragraphs_to_passages(
    paragraphs: list[tuple[int, int, str]],
    min_words: int,
    target_words: int,
    max_words: int,
) -> list[dict]:
    passages = []
    buf_parts: list[str] = []
    buf_start = None
    buf_end = None
    buf_words = 0

    def flush():
        nonlocal buf_parts, buf_start, buf_end, buf_words
        if buf_parts:
            text = " ".join(buf_parts)
            passages.append(
                {"char_start": buf_start, "char_end": buf_end, "text": text, "n_words": buf_words}
            )
        buf_parts, buf_start, buf_end, buf_words = [], None, None, 0

    for p_start, p_end, p_text in paragraphs:
        p_words = len(p_text.split())

        if p_words > max_words:
            # Oversized paragraph. If there's a too-small buffer sitting in
            # front of it (e.g. a bare "II" chapter marker), fold it into
            # this paragraph rather than flushing it as its own near-empty
            # passage; otherwise flush the buffer normally first.
            if buf_parts and buf_words < min_words:
                p_start = buf_start
                p_text = " ".join(buf_parts) + " " + p_text
                buf_parts, buf_start, buf_end, buf_words = [], None, None, 0
            else:
                flush()
            for piece_start, piece_end, piece_text in split_oversized_paragraph(
                p_start, p_end, p_text, max_words
            ):
                passages.append(
                    {
                        "char_start": piece_start,
                        "char_end": piece_end,
                        "text": piece_text,
                        "n_words": len(piece_text.split()),
                    }
                )
            continue

        if buf_start is None:
            buf_start = p_start
        buf_end = p_end
        buf_parts.append(p_text)
        buf_words += p_words

        if buf_words >= target_words:
            flush()

    if buf_parts and buf_words >= min_words:
        flush()
    elif buf_parts and passages:
        # Too-small tail: fold into the previous passage rather than drop it.
        prev = passages[-1]
        prev["text"] = prev["text"] + " " + " ".join(buf_parts)
        prev["char_end"] = buf_end
        prev["n_words"] = len(prev["text"].split())
    elif buf_parts:
        flush()

    return passages


def segment_file(path: Path, min_words: int, target_words: int, max_words: int) -> list[dict]:
    raw = path.read_text(errors="replace")
    text = strip_gutenberg_boilerplate(raw) if "PROJECT GUTENBERG" in raw[:2000].upper() else raw
    paragraphs = split_paragraphs(text)
    passages = merge_paragraphs_to_passages(paragraphs, min_words, target_words, max_words)
    return passages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", help="Specific source_file names from the manifest; default = all")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())["corpus"]
    if args.files:
        wanted = set(args.files)
        manifest = [r for r in manifest if r["source_file"] in wanted]
        missing = wanted - {r["source_file"] for r in manifest}
        if missing:
            sys.exit(f"Not in manifest: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_out = OUTPUT_DIR / "all_passages.jsonl"
    combined_rows = []

    for rec in manifest:
        path = POSITIVES_DIR / rec["source_file"]
        passages = segment_file(path, args.min_words, args.target_words, args.max_words)
        stem = Path(rec["source_file"]).stem
        out_path = OUTPUT_DIR / f"{stem}.jsonl"
        rows = []
        for i, p in enumerate(passages):
            row = {
                "passage_id": f"{stem}_{i:04d}",
                "source_file": rec["source_file"],
                "author": rec["author"],
                "title": rec["title"],
                "year": rec["year"],
                "role": rec["role"],
                "passage_index": i,
                "char_start": p["char_start"],
                "char_end": p["char_end"],
                "n_words": p["n_words"],
                "passage_text": p["text"],
            }
            rows.append(row)
        out_path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
        combined_rows.extend(rows)

        n = len(rows)
        wc = [r["n_words"] for r in rows]
        avg = sum(wc) / n if n else 0
        lo = min(wc) if wc else 0
        hi = max(wc) if wc else 0
        print(f"{rec['source_file']:<45} {n:>5} passages   avg {avg:>6.1f}w   range [{lo}, {hi}]")

    all_out.write_text("\n".join(json.dumps(r) for r in combined_rows) + "\n")
    print(f"\nTotal: {len(combined_rows)} passages across {len(manifest)} works -> {all_out}")


if __name__ == "__main__":
    main()
