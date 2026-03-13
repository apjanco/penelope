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

        # If adding this sentence would exceed the limit, flush the buffer
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

        # If a single sentence is bigger than chunk_size and buffer is empty,
        # accept it anyway (avoid infinite loop / lost content)
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
# Custom chunkers — file-specific splitting logic
# ---------------------------------------------------------------------------

# The 15 narrators in Faulkner's *As I Lay Dying*.  Each section heading
# is just the narrator's name on its own line, with blank lines around it.
_AILD_NARRATORS: set[str] = {
    "Darl", "Cora", "Jewel", "Dewey Dell", "Tull", "Anse",
    "Peabody", "Vardaman", "Cash", "Samson", "Addie",
    "Whitfield", "Armstid", "Moseley", "MacGowan",
}

# Matches a narrator name on its own line (with optional trailing whitespace).
# Handles single names (Darl), two-word names (Dewey Dell), and
# camelCase names (MacGowan).
_AILD_HEADING_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(n) for n in sorted(_AILD_NARRATORS, key=len, reverse=True)) + r")\s*$",
    re.MULTILINE,
)


def _chunk_as_i_lay_dying(
    text: str,
    source_file: str,
    chunk_size: int = 20_000,
    overlap: int = 1_000,
) -> list[Chunk]:
    """Custom chunker for Faulkner's *As I Lay Dying*.

    Splits on narrator headings (59 sections, 15 narrators).  Each chunk
    is labelled "Narrator (N)" where N is the occurrence count for that
    narrator, giving e.g. "Darl (1)", "Cora (1)", "Darl (2)", …

    Sections that exceed *chunk_size* are sub-chunked by sentences.
    """
    splits: list[tuple[int, str]] = []
    for m in _AILD_HEADING_RE.finditer(text):
        splits.append((m.start(), m.group(0).strip()))

    if len(splits) < 2:
        logger.warning(
            "%s: expected narrator headings but found %d — "
            "falling back to generic chunker",
            source_file, len(splits),
        )
        return _chunk_by_sentences(text, source_file, chunk_size, overlap)

    logger.info(
        "%s: found %d narrator sections (As I Lay Dying mode)",
        source_file, len(splits),
    )

    # Track per-narrator occurrence count for labelling
    narrator_counts: dict[str, int] = {}
    chunks: list[Chunk] = []

    for i, (start, narrator) in enumerate(splits):
        end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue

        # Build label: "Darl (3)" = third Darl section
        narrator_counts[narrator] = narrator_counts.get(narrator, 0) + 1
        label = f"{narrator} ({narrator_counts[narrator]})"

        # Sub-chunk oversized sections
        if len(section_text) > chunk_size:
            logger.info(
                "%s: section '%s' is %d chars (> %d), sub-chunking",
                source_file, label, len(section_text), chunk_size,
            )
            sub_chunks = _chunk_by_sentences(
                section_text, source_file, chunk_size, overlap,
            )
            for j, sc in enumerate(sub_chunks):
                sc.chunk_label = f"{label} [{j + 1}/{len(sub_chunks)}]"
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                Chunk(
                    source_file=source_file,
                    chunk_id=_make_chunk_id(source_file, i, label),
                    chunk_label=label,
                    chunk_text=section_text,
                    chunk_index=i,
                )
            )

    # Re-index sequentially
    for idx, chunk in enumerate(chunks):
        chunk.chunk_index = idx
        chunk.chunk_id = _make_chunk_id(source_file, idx, chunk.chunk_label)

    # Add overlap context
    _add_overlap_context(chunks, overlap)

    logger.info(
        "%s: %d chunks from %d narrator sections",
        source_file, len(chunks), len(splits),
    )
    return chunks


# Registry mapping filename patterns (lowercased stem substrings) to
# custom chunking functions.  The first match wins.
# Signature must match: (text, source_file, chunk_size, overlap) -> list[Chunk]
CUSTOM_CHUNKERS: dict[str, callable] = {
    "as_i_lay_dying": _chunk_as_i_lay_dying,
    "as i lay dying":  _chunk_as_i_lay_dying,
    "asilaydying":     _chunk_as_i_lay_dying,
}


def _get_custom_chunker(source_file: str):
    """Return a custom chunker for this file, or None."""
    stem = Path(source_file).stem.lower().replace("-", "_")
    for pattern, chunker in CUSTOM_CHUNKERS.items():
        if pattern in stem:
            return chunker
    return None


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
    # Check for a file-specific custom chunker first
    custom = _get_custom_chunker(source_file)
    if custom is not None:
        logger.info("%s: using custom chunker", source_file)
        return custom(text, source_file, chunk_size, overlap)

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
        section_text = text[start:end].strip()
        if not section_text:
            continue

        # If a structural section exceeds chunk_size, sub-chunk it
        if len(section_text) > chunk_size:
            logger.info(
                "%s: section '%s' is %d chars (> %d), sub-chunking by sentences",
                source_file, label, len(section_text), chunk_size,
            )
            sub_chunks = _chunk_by_sentences(
                section_text, source_file, chunk_size, overlap,
            )
            # Relabel sub-chunks to reflect the parent section
            for j, sc in enumerate(sub_chunks):
                sc.chunk_label = f"{label} ({j + 1}/{len(sub_chunks)})"
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                Chunk(
                    source_file=source_file,
                    chunk_id=_make_chunk_id(source_file, i, label),
                    chunk_label=label,
                    chunk_text=section_text,
                    chunk_index=i,
                )
            )

    # Re-index and re-id all chunks sequentially after sub-chunking
    for idx, chunk in enumerate(chunks):
        chunk.chunk_index = idx
        chunk.chunk_id = _make_chunk_id(source_file, idx, chunk.chunk_label)

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
# Chunk markup — write and read <chunk-N> annotated files
# ---------------------------------------------------------------------------

# Tag pattern:  <chunk-0 label="Chapter I">  ...  </chunk-0>
_CHUNK_OPEN_RE = re.compile(
    r'<chunk-(\d+)(?:\s+label="([^"]*)")?\s*>',
)
_CHUNK_CLOSE_RE = re.compile(r"</chunk-(\d+)>")


def write_chunked_file(chunks: list[Chunk], output_path: Path) -> None:
    """Write chunk-annotated text to a file.

    Format:
        <chunk-0 label="Chapter I">
        ...text of chunk 0...
        </chunk-0>

        <chunk-1 label="Chapter II">
        ...text of chunk 1...
        </chunk-1>

    The resulting file is plain text that can be opened in any editor.
    Users can move the tags, merge or split chunks, rename labels, etc.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for chunk in chunks:
        label_attr = f' label="{chunk.chunk_label}"' if chunk.chunk_label else ""
        lines.append(f"<chunk-{chunk.chunk_index}{label_attr}>")
        lines.append(chunk.chunk_text)
        lines.append(f"</chunk-{chunk.chunk_index}>")
        lines.append("")  # blank line between chunks for readability

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %d chunks to %s", len(chunks), output_path)


def parse_chunked_file(file_path: Path) -> list[Chunk]:
    """Read a chunk-annotated file and extract Chunk objects.

    Parses <chunk-N label="...">...</chunk-N> tags. Supports hand-edited
    files where users have moved boundaries, merged chunks, or changed labels.

    Args:
        file_path: Path to a chunked .txt file (from chunking/).

    Returns:
        Ordered list of Chunk objects extracted from the markup.
    """
    text = file_path.read_text(encoding="utf-8")
    source_file = file_path.name
    chunks: list[Chunk] = []

    # Walk through the text looking for chunk tags
    pos = 0
    while pos < len(text):
        open_match = _CHUNK_OPEN_RE.search(text, pos)
        if not open_match:
            break

        chunk_index = int(open_match.group(1))
        chunk_label = open_match.group(2) or ""

        # Content starts after the opening tag
        content_start = open_match.end()

        # Find the matching close tag
        close_match = _CHUNK_CLOSE_RE.search(text, content_start)
        if not close_match:
            logger.warning(
                "%s: unclosed <chunk-%d> tag at position %d — "
                "including all remaining text",
                file_path.name, chunk_index, open_match.start(),
            )
            chunk_text = text[content_start:].strip()
            pos = len(text)
        else:
            chunk_text = text[content_start:close_match.start()].strip()
            pos = close_match.end()

        if not chunk_text:
            logger.debug(
                "%s: chunk-%d is empty, skipping", file_path.name, chunk_index
            )
            continue

        chunk_id = _make_chunk_id(source_file, chunk_index, chunk_label)
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_id=chunk_id,
                chunk_label=chunk_label,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
            )
        )

    if not chunks:
        logger.warning(
            "%s: no <chunk-N> tags found — is this a chunked file?",
            file_path.name,
        )

    # Re-index sequentially (user may have renumbered or merged)
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunk_id = _make_chunk_id(source_file, i, chunk.chunk_label)

    # Add overlap context between chunks
    _add_overlap_context(chunks, overlap=1_000)

    logger.info(
        "%s: parsed %d chunk(s) from markup", file_path.name, len(chunks)
    )
    return chunks


def parse_chunked_dir(dir_path: Path) -> list[Chunk]:
    """Parse all chunked .txt files in a directory.

    Args:
        dir_path: Path to the chunking/ directory.

    Returns:
        Combined list of Chunk objects from all files, in file order.
    """
    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", dir_path)
        return []

    all_chunks: list[Chunk] = []
    for f in txt_files:
        chunks = parse_chunked_file(f)
        all_chunks.extend(chunks)

    logger.info(
        "Parsed %d total chunk(s) from %d file(s) in %s",
        len(all_chunks), len(txt_files), dir_path,
    )
    return all_chunks


def is_chunked_file(file_path: Path) -> bool:
    """Quick check whether a file contains <chunk-N> markup."""
    try:
        # Only read the first 2000 chars to check
        with open(file_path, encoding="utf-8") as f:
            head = f.read(2000)
        return bool(_CHUNK_OPEN_RE.search(head))
    except Exception:
        return False


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

