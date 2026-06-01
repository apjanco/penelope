#!/usr/bin/env python3
"""Download Project Gutenberg texts known for stream-of-consciousness technique.

Companion to download_gutenberg.py (which curates *negatives*).  This script
curates *positives* — texts where stream-of-consciousness, free indirect
discourse, or direct interior monologue is prominent — and downloads them to
positives/ for use with chunk.py → scripts/silver.py.

SoC TECHNIQUE NOTES
-------------------
Works are grouped by primary technique relevant to the taxonomy:
  • direct_interior_monologue   — unmediated first-person thought-stream
  • indirect_interior_monologue — free indirect discourse (FID), narrator close
  • reverie_fantasy             — digressive memory/dream passages
  • soliloquy                   — sustained self-address
  • hybrid / mixed              — multiple techniques present in same work

All works are in the public domain in the United States (published ≤ 1927,
or published abroad + 95+ years elapsed).  Check your own jurisdiction.

Usage
-----
    # Download all curated SoC texts to positives/
    python scripts/download_positives.py --output-dir positives/

    # Print the catalog without downloading
    python scripts/download_positives.py --list

    # Filter by author name (substring, case-insensitive)
    python scripts/download_positives.py --output-dir positives/ --author woolf

    # Download only specific Gutenberg IDs
    python scripts/download_positives.py --output-dir positives/ --ids 4300 5765

    # Slower rate (Gutenberg ToS: be respectful, ≥ 2 s recommended)
    python scripts/download_positives.py --output-dir positives/ --delay 3.0

After downloading, chunk and annotate:
    python chunk.py --input positives/ --output chunking_positives/
    python scripts/silver.py --input chunking_positives/ --output-dir training_data/
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated SoC positives
# Each tuple: (gutenberg_id, author, title, primary_soc_technique)
# Verify / browse at:  https://www.gutenberg.org/ebooks/<id>
# ---------------------------------------------------------------------------

CURATED_TEXTS: list[tuple[int, str, str, str]] = [
    # ── James Joyce ──────────────────────────────────────────────────────────
    # Defines direct interior monologue; most annotated SoC work in English
    (4300,  "James Joyce",      "Ulysses",
     "direct_interior_monologue + hybrid — Molly Bloom soliloquy, Proteus chapter"),
    (4217,  "James Joyce",      "A Portrait of the Artist as a Young Man",
     "indirect_interior_monologue — FID intensifies through Stephen's development"),
    (2814,  "James Joyce",      "Dubliners",
     "indirect_interior_monologue — epiphany moments, close-third interiority"),

    # ── Virginia Woolf ───────────────────────────────────────────────────────
    # Defines the modernist stream-of-consciousness novel
    (5765,  "Virginia Woolf",   "Mrs Dalloway",
     "indirect_interior_monologue + reverie_fantasy — landmark FID, time fluidity"),
    (690,   "Virginia Woolf",   "To the Lighthouse",
     "indirect_interior_monologue — 'Time Passes', close interior access"),
    (43524, "Virginia Woolf",   "The Waves",
     "direct_interior_monologue — six alternating interior soliloquies"),
    (5550,  "Virginia Woolf",   "Jacob's Room",
     "indirect_interior_monologue — experimental interiority, fragmented POV"),

    # ── Dorothy Richardson ───────────────────────────────────────────────────
    # Pilgrimage sequence: coined the phrase "stream of consciousness" in fiction
    (4517,  "Dorothy Richardson", "Pointed Roofs",
     "indirect_interior_monologue — vol. 1 of Pilgrimage; interior focalization"),
    (4518,  "Dorothy Richardson", "Backwater",
     "indirect_interior_monologue — vol. 2 of Pilgrimage"),
    (4519,  "Dorothy Richardson", "Honeycomb",
     "indirect_interior_monologue — vol. 3 of Pilgrimage"),

    # ── May Sinclair ─────────────────────────────────────────────────────────
    # Critic who named Richardson's technique; also practised it herself
    (34219, "May Sinclair",     "Mary Olivier: A Life",
     "indirect_interior_monologue — interior monologue across a life"),

    # ── Édouard Dujardin ─────────────────────────────────────────────────────
    # Widely credited as the first sustained SoC novel (1888); influenced Joyce
    (42984, "Edouard Dujardin", "We'll to the Woods No More",
     "direct_interior_monologue — proto-SoC; real-time unmediated thought-stream"),

    # ── Ford Madox Ford ──────────────────────────────────────────────────────
    # Unreliable first-person narrator; digressive memory/retrospective interiority
    (2775,  "Ford Madox Ford",  "The Good Soldier",
     "reverie_fantasy + indirect_interior_monologue — unreliable retrospective narrator"),

    # ── Henry James (late period) ────────────────────────────────────────────
    # Late James = dense FID; intense interior access to central consciousness.
    # NOTE: James appears in the EXCLUSION list for download_gutenberg.py (negatives).
    (432,   "Henry James",      "The Ambassadors",
     "indirect_interior_monologue — elaborated FID; Strether's 'central consciousness'"),
    (209,   "Henry James",      "The Wings of the Dove",
     "indirect_interior_monologue — late-James FID, intensive psychological interiority"),
    (1149,  "Henry James",      "The Turn of the Screw",
     "indirect_interior_monologue — unreliable narrator, interior obsession"),

    # ── Arthur Schnitzler ────────────────────────────────────────────────────
    # Austrian proto-SoC; 'Lieutenant Gustl' is the first German-language interior monologue
    (32220, "Arthur Schnitzler", "Lieutenant Gustl",
     "direct_interior_monologue — unbroken real-time interior monologue"),

    # ── Kate Chopin ──────────────────────────────────────────────────────────
    # The Awakening: FID and interior reverie, proto-modernist interiority
    (160,   "Kate Chopin",      "The Awakening",
     "indirect_interior_monologue + reverie_fantasy — FID, Edna's interior drift"),

    # ── Leo Tolstoy (translated) ──────────────────────────────────────────────
    # Anna Karenina's final chapters: sustained stream-of-consciousness monologue
    # (Constance Garnett translation, PD in US)
    (1399,  "Leo Tolstoy",      "Anna Karenina",
     "indirect_interior_monologue + reverie_fantasy — Anna's final interior monologue"),

    # ── Fyodor Dostoevsky (translated) ───────────────────────────────────────
    # Notes from Underground: obsessive self-interrogating first-person monologue
    (600,   "Fyodor Dostoevsky", "Notes from Underground",
     "soliloquy — proto-SoC; feverish self-address and internal contradiction"),

    # ── Gustave Flaubert (translated) ────────────────────────────────────────
    # Inventor of modern FID; Madame Bovary is a textbook FID example
    (2413,  "Gustave Flaubert", "Madame Bovary",
     "indirect_interior_monologue — pioneering FID; Emma's reverie and fantasy"),

    # ── Gertrude Stein ────────────────────────────────────────────────────────
    # Three Lives: repetitive present-tense style; early American SoC experiment
    (15408, "Gertrude Stein",   "Three Lives",
     "indirect_interior_monologue — repetitive interior style, present-tense consciousness"),
]

# ---------------------------------------------------------------------------
# URL patterns (tried in order)
# ---------------------------------------------------------------------------

URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

HEADERS = {
    "User-Agent": (
        "penelope-soc-research/1.0 "
        "(academic NLP research; contact: apjanco@princeton.edu)"
    )
}

# Gutenberg boilerplate markers
_START_RE = re.compile(
    r"\*{3}\s*START OF (THE |THIS )?PROJECT GUTENBERG EBOOK[^\n]*\n",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*{3}\s*END OF (THE |THIS )?PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Download helpers (mirrors download_gutenberg.py)
# ---------------------------------------------------------------------------


def _fetch_text(gutenberg_id: int, session: requests.Session) -> str | None:
    """Try each URL pattern; return raw text or None on failure."""
    for pattern in URL_PATTERNS:
        url = pattern.format(id=gutenberg_id)
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                logger.debug("Downloaded from %s", url)
                return resp.text
        except requests.RequestException as exc:
            logger.debug("Failed %s: %s", url, exc)
    return None


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_match = _START_RE.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = _END_RE.search(text)
    if end_match:
        text = text[: end_match.start()]

    return text.strip()


def _output_filename(gutenberg_id: int, author: str, title: str) -> str:
    """Generate a clean filename: pg04300_ulysses.txt"""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return f"pg{gutenberg_id:05d}_{slug}.txt"


def download_text(
    gutenberg_id: int,
    author: str,
    title: str,
    output_dir: Path,
    session: requests.Session,
    delay: float,
) -> bool:
    """Download one text, strip boilerplate, save to output_dir. Returns True on success."""
    filename = _output_filename(gutenberg_id, author, title)
    dest = output_dir / filename

    if dest.exists():
        logger.info("  skip (exists) %s", filename)
        return True

    logger.info("  fetching pg%d — %s, %s", gutenberg_id, author, title)
    raw = _fetch_text(gutenberg_id, session)
    if raw is None:
        logger.warning(
            "  FAILED pg%d (%s) — check https://www.gutenberg.org/ebooks/%d",
            gutenberg_id, title, gutenberg_id,
        )
        return False

    clean = _strip_gutenberg_boilerplate(raw)
    if len(clean.split()) < 1000:
        logger.warning(
            "  SKIP pg%d — stripped text too short (%d words); boilerplate stripping may have failed",
            gutenberg_id, len(clean.split()),
        )
        return False

    dest.write_text(clean, encoding="utf-8")
    logger.info("    saved %s (%d words)", filename, len(clean.split()))

    time.sleep(delay)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download curated Gutenberg texts as positive SoC training examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_positives.py --list\n"
            "  python scripts/download_positives.py --output-dir positives/\n"
            "  python scripts/download_positives.py --output-dir positives/ --author woolf\n"
            "  python scripts/download_positives.py --output-dir positives/ --ids 4300 5765\n"
            "\n"
            "After downloading:\n"
            "  python chunk.py --input positives/ --output chunking_positives/\n"
            "  python scripts/silver.py --input chunking_positives/ --output-dir training_data/\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("positives"),
        help="Directory to save .txt files (default: positives/)",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Download only these Gutenberg IDs (default: all curated texts)",
    )
    parser.add_argument(
        "--author",
        default=None,
        metavar="NAME",
        help="Download only texts whose author contains NAME (case-insensitive substring)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between downloads (default: 2.0; be respectful of Gutenberg)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the curated text list and exit without downloading",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\n{'ID':>7}  {'Author':<26}  {'Title':<40}  SoC technique")
        print("-" * 110)
        for gid, author, title, technique in sorted(CURATED_TEXTS, key=lambda x: x[1]):
            print(f"{gid:>7}  {author:<26}  {title:<40}  {technique}")
        print(f"\nTotal: {len(CURATED_TEXTS)} texts")
        return

    # ── Filter ───────────────────────────────────────────────────────
    texts = list(CURATED_TEXTS)

    if args.author:
        texts = [t for t in texts if args.author.lower() in t[1].lower()]
        if not texts:
            logger.error("No texts found matching author '%s'", args.author)
            sys.exit(1)

    if args.ids:
        id_set = set(args.ids)
        texts = [t for t in texts if t[0] in id_set]
        missing = id_set - {t[0] for t in texts}
        if missing:
            logger.warning(
                "IDs not in curated list (will still attempt): %s",
                ", ".join(str(i) for i in sorted(missing)),
            )
            for gid in sorted(missing):
                texts.append((gid, "unknown", f"pg{gid}", "unknown"))

    # ── Deduplicate by ID ─────────────────────────────────────────────
    seen: set[int] = set()
    deduped = []
    for entry in texts:
        if entry[0] not in seen:
            deduped.append(entry)
            seen.add(entry[0])
    texts = deduped

    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    logger.info("Downloading %d text(s) to %s/", len(texts), args.output_dir)

    n_ok = n_fail = 0
    for gid, author, title, _technique in texts:
        ok = download_text(gid, author, title, args.output_dir, session, args.delay)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(
        "Done: %d downloaded / already present, %d failed.  Output: %s/",
        n_ok, n_fail, args.output_dir,
    )
    if n_fail:
        logger.warning(
            "Some downloads failed.  Verify IDs at https://www.gutenberg.org/ebooks/<id> "
            "and re-run with --ids <correct_id> to add them manually."
        )
    if n_ok > 0:
        print(
            f"\nNext steps:\n"
            f"  python chunk.py --input {args.output_dir}/ --output chunking_positives/\n"
            f"  python scripts/silver.py --input chunking_positives/ --output-dir training_data/"
        )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
