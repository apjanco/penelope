"""Lightweight text chunker for SOC analysis — no external dependencies.

Splits literary texts into chunks using structural markers (chapter headings,
part dividers, etc.), falling back to sentence-boundary chunking with overlap
when no structure is detected.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from scripts.models import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading / chapter detection patterns
# ---------------------------------------------------------------------------

# Patterns that signal a chapter or section boundary.  Order matters —
# the first match wins, so more specific patterns come first.
HEADING_PATTERNS: list[re.Pattern[str]] = [
    # "CHAPTER I", "Chapter 1", "CHAPTER ONE", etc.
    re.compile(
        r"^\s*(CHAPTER|Chapter)\s+([IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*$",
        re.MULTILINE,
    ),
    # "PART I", "Part 1", "Part One"
    re.compile(
        r"^\s*(PART|Part)\s+([IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*$",
        re.MULTILINE,
    ),
    # "BOOK I", "Book 1"
    re.compile(
        r"^\s*(BOOK|Book)\s+([IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*$",
        re.MULTILINE,
    ),
    # "I.", "II.", "III." — Roman numeral section markers on their own line
    re.compile(r"^\s*[IVXLCDM]+\.\s*$", re.MULTILINE),
    # Lines that are ALL CAPS and at least 3 characters (likely a heading)
    re.compile(r"^[A-Z][A-Z\s]{2,}$", re.MULTILINE),
    # "--- " or "***" or "* * *" divider lines
    re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE),
    # "[Episode N]", "[Section N]" — bracketed labels
    re.compile(r"^\s*\[.+\]\s*$", re.MULTILINE),
]


def _find_split_points(text: str) -> list[tuple[int, str]]:
    """Find heading / divider positions in the text.

    Returns:
        Sorted list of (char_offset, label) tuples.
    """
    splits: list[tuple[int, str]] = []
    for pat in HEADING_PATTERNS:
        for m in pat.finditer(text):
            label = m.group(0).strip()
            splits.append((m.start(), label))
    # De-duplicate by position (different patterns may match the same line)
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for pos, label in sorted(splits):
        if pos not in seen:
            seen.add(pos)
            unique.append((pos, label))
    return unique


def _make_chunk_id(source_file: str, index: int, label: str) -> str:
    """Generate a stable chunk_id from file name + index."""
    stem = Path(source_file).stem
    # Normalize to ASCII-safe slug
    slug = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return f"{slug}_{index:03d}"


# ---------------------------------------------------------------------------
# Sentence-boundary fallback
# ---------------------------------------------------------------------------

# Rough sentence splitter — splits on .!? followed by whitespace and an uppercase letter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (approximate)."""
    return _SENTENCE_RE.split(text)


def _chunk_by_sentences(
    text: str,
    source_file: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Fall back to fixed-size sentence-boundary chunking with overlap."""
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + sent_len > chunk_size:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    source_file=source_file,
                    chunk_id=_make_chunk_id(source_file, idx, ""),
                    chunk_label=f"Chunk {idx + 1}",
                    chunk_text=chunk_text,
                    chunk_index=idx,
                )
            )
            idx += 1
            # Keep overlap — walk backward through sentences
            overlap_text = ""
            overlap_sents: list[str] = []
            for s in reversed(current):
                if len(overlap_text) + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_text = " ".join(overlap_sents)
            current = overlap_sents
            current_len = len(overlap_text)

        current.append(sent)
        current_len += sent_len

    # Final chunk
    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_id=_make_chunk_id(source_file, idx, ""),
                chunk_label=f"Chunk {idx + 1}",
                chunk_text=chunk_text,
                chunk_index=idx,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 20_000,
    overlap: int = 1_000,
) -> list[Chunk]:
    """Split a literary text into chunks for LLM analysis.

    Strategy:
      1. Look for chapter/heading markers and split there.
      2. If no markers are found, fall back to sentence-boundary chunking
         with the specified chunk_size (in characters) and overlap.
      3. Add overlap context between structural chunks.

    Args:
        text: Full plain text of the literary work.
        source_file: Original filename (used in metadata).
        chunk_size: Target chunk size in characters (fallback mode).
        overlap: Overlap size in characters between adjacent chunks.

    Returns:
        Ordered list of Chunk objects.
    """
    split_points = _find_split_points(text)

    if len(split_points) < 2:
        logger.info(
            "%s: no structural markers found, using sentence-boundary chunking",
            source_file,
        )
        return _chunk_by_sentences(text, source_file, chunk_size, overlap)

    logger.info(
        "%s: found %d structural markers, chunking by headings",
        source_file,
        len(split_points),
    )

    chunks: list[Chunk] = []
    for i, (start, label) in enumerate(split_points):
        end = split_points[i + 1][0] if i + 1 < len(split_points) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_id=_make_chunk_id(source_file, i, label),
                chunk_label=label,
                chunk_text=chunk_text,
                chunk_index=i,
            )
        )

    # Add overlap context between structural chunks
    _add_overlap_context(chunks, overlap)

    return chunks


def _add_overlap_context(chunks: list[Chunk], overlap: int) -> None:
    """Populate context_before / context_after on each chunk."""
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_text = chunks[i - 1].chunk_text
            chunk.context_before = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
        if i < len(chunks) - 1:
            next_text = chunks[i + 1].chunk_text
            chunk.context_after = next_text[:overlap] if len(next_text) > overlap else next_text


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Chunk a text file for SOC analysis")
    parser.add_argument("file", type=Path, help="Path to text file")
    parser.add_argument("--chunk-size", type=int, default=20_000, help="Target chunk size (chars)")
    parser.add_argument("--overlap", type=int, default=1_000, help="Overlap between chunks (chars)")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    text = args.file.read_text(encoding="utf-8")
    result = chunk_text(text, args.file.name, args.chunk_size, args.overlap)

    data = [c.model_dump() for c in result]
    output = json.dumps(data, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        logger.info("Wrote %d chunks to %s", len(result), args.output)
    else:
        print(output)
