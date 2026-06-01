#!/usr/bin/env python3
"""Download Project Gutenberg plain-text files for use as negative training examples.

INCLUSION / EXCLUSION RATIONALE
--------------------------------
The SoC taxonomy used by this project includes both direct interior monologue
AND indirect interior monologue (free indirect discourse, FID).  This means
authors who pioneered or heavily rely on FID — Jane Austen, George Eliot,
Henry James, Flaubert — are NOT safe negatives even though their prose is
conventionally "Victorian."

INCLUDED authors share a key property: their narration is predominantly
externalized (action, dialogue, description, first-person retrospective account)
with little to no rendering of thought-from-within.

EXCLUDED authors and reasons
  James Joyce                 — defines direct interior monologue
  Virginia Woolf              — defines stream-of-consciousness (all works)
  Dorothy Richardson          — Pilgrimage sequence, pioneered the form
  May Sinclair                — coined "stream of consciousness" for Richardson
  Katherine Mansfield         — short-story interior monologue
  Arthur Schnitzler           — interior monologue pioneer (Lieutenant Gustl)
  Sherwood Anderson           — associative interior narration
  Jane Austen                 — FID pioneer; ALL works carry indirect_interior_monologue risk
  George Eliot                — deep psychological interiority throughout
  Henry James                 — FID-heavy in ALL periods, intensifies late
  Gustave Flaubert (transl.)  — invented modern FID; Madame Bovary is a positive example
  Thomas Hardy (late)         — Jude, Return of the Native: omniscient_description risk
  Laurence Sterne             — Tristram Shandy is proto-stream-of-consciousness
  Samuel Richardson           — Clarissa: intense epistolary psychological interiority

INCLUDED authors and reasons
  Arthur Conan Doyle          — Holmes stories are observation-focused, almost no interiority
  Wilkie Collins              — epistolary/thriller; narrators report events, don't stream
  H.G. Wells (early SF)       — concept-driven; externalized third-person narration
  Jules Verne                 — action-adventure; almost no interior life
  H. Rider Haggard            — adventure narration
  Anthony Hope                — Ruritanian adventure; minimal interiority
  R.L. Stevenson (Treasure)   — adventure narration (NOT Jekyll, which is psychological)
  Charles Dickens             — theatrical/dramatic style; more external than Eliot/Austen
  Anthony Trollope            — cleaner externalized omniscient than Eliot
  Mark Twain                  — vernacular first-person is NOT modernist stream-of-consciousness
  Jerome K. Jerome            — comic travel; surface-level observation
  Walter Scott                — historical adventure
  Alexandre Dumas             — action-driven plots

Usage
-----
    # Download all curated texts
    python scripts/download_gutenberg.py --output-dir negatives/

    # Download specific IDs only
    python scripts/download_gutenberg.py --output-dir negatives/ --ids 98 766

    # Print the curated list without downloading
    python scripts/download_gutenberg.py --list

    # Slower rate to avoid Gutenberg throttling
    python scripts/download_gutenberg.py --output-dir negatives/ --delay 3.0
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
# Curated text list
# Each tuple: (gutenberg_id, author, title)
# Verify / browse IDs at: https://www.gutenberg.org/ebooks/<id>
# ---------------------------------------------------------------------------

CURATED_TEXTS: list[tuple[int, str, str]] = [
    # ── Arthur Conan Doyle ──────────────────────────────────────────────────
    # Holmes stories: externalized observation, almost zero interiority
    (244,   "Arthur Conan Doyle", "A Study in Scarlet"),
    (1661,  "Arthur Conan Doyle", "The Adventures of Sherlock Holmes"),
    (2097,  "Arthur Conan Doyle", "The Memoirs of Sherlock Holmes"),
    (2852,  "Arthur Conan Doyle", "The Hound of the Baskervilles"),
    (108,   "Arthur Conan Doyle", "The Return of Sherlock Holmes"),

    # ── Wilkie Collins ──────────────────────────────────────────────────────
    # Epistolary / multiple-narrator construction; reporters not streamers
    (583,   "Wilkie Collins", "The Woman in White"),
    (155,   "Wilkie Collins", "The Moonstone"),

    # ── H.G. Wells (early) ──────────────────────────────────────────────────
    # Scientific romance; externalized third-person omniscient narration
    (36,    "H.G. Wells", "The Time Machine"),
    (718,   "H.G. Wells", "The War of the Worlds"),
    (159,   "H.G. Wells", "The Island of Doctor Moreau"),
    (776,   "H.G. Wells", "When the Sleeper Wakes"),
    (1743,  "H.G. Wells", "The Invisible Man"),

    # ── Jules Verne ─────────────────────────────────────────────────────────
    # Pure action-adventure; virtually no interiority
    (164,   "Jules Verne", "Around the World in Eighty Days"),
    (103,   "Jules Verne", "Twenty Thousand Leagues Under the Sea"),
    (83,    "Jules Verne", "The Mysterious Island"),
    (3526,  "Jules Verne", "Michael Strogoff"),
    (4791,  "Jules Verne", "From the Earth to the Moon"),

    # ── H. Rider Haggard ────────────────────────────────────────────────────
    # Adventure narration; narrator records events not thoughts
    (2166,  "H. Rider Haggard", "She"),
    (2166,  "H. Rider Haggard", "King Solomon's Mines"),  # may share ID; try both

    # ── Anthony Hope ────────────────────────────────────────────────────────
    (95,    "Anthony Hope", "The Prisoner of Zenda"),
    (1951,  "Anthony Hope", "Rupert of Hentzau"),

    # ── R.L. Stevenson ──────────────────────────────────────────────────────
    # Treasure Island ONLY — adventure narration.
    # EXCLUDE Strange Case of Dr Jekyll: psychological frame/reverie risk.
    (120,   "R.L. Stevenson", "Treasure Island"),
    (858,   "R.L. Stevenson", "Kidnapped"),

    # ── Charles Dickens ─────────────────────────────────────────────────────
    # Theatrical/dramatic style; more externalized than Eliot or Austen.
    # Prefer omniscient-narrator novels over first-person ones.
    # Great Expectations / David Copperfield are retrospective first-person,
    # which is acceptable — retrospective memoir ≠ streaming present-thought.
    (98,    "Charles Dickens", "A Tale of Two Cities"),
    (730,   "Charles Dickens", "Oliver Twist"),
    (821,   "Charles Dickens", "Hard Times"),
    (588,   "Charles Dickens", "Dombey and Son"),
    (967,   "Charles Dickens", "Martin Chuzzlewit"),
    (1400,  "Charles Dickens", "Great Expectations"),
    (766,   "Charles Dickens", "David Copperfield"),

    # ── Anthony Trollope ────────────────────────────────────────────────────
    # Clear omniscient narration; less FID than Eliot
    (3256,  "Anthony Trollope", "The Warden"),
    (4181,  "Anthony Trollope", "Barchester Towers"),
    (9462,  "Anthony Trollope", "The Way We Live Now"),
    (5140,  "Anthony Trollope", "Doctor Thorne"),

    # ── Mark Twain ──────────────────────────────────────────────────────────
    # Vernacular first-person is very different from modernist stream-of-consciousness.
    # Huck's voice is colloquial/observational, not associative inner flow.
    (74,    "Mark Twain", "The Adventures of Tom Sawyer"),
    (76,    "Mark Twain", "Adventures of Huckleberry Finn"),
    (102,   "Mark Twain", "The Prince and the Pauper"),
    (245,   "Mark Twain", "Life on the Mississippi"),  # non-fiction, very safe

    # ── Jerome K. Jerome ────────────────────────────────────────────────────
    # Comic travel; surface observation, no interiority
    (308,   "Jerome K. Jerome", "Three Men in a Boat"),
    (1400,  "Jerome K. Jerome", "Three Men on the Bummel"),  # double-check ID

    # ── Walter Scott ────────────────────────────────────────────────────────
    # Historical adventure; externalized
    (82,    "Walter Scott", "Ivanhoe"),
    (84,    "Walter Scott", "Waverley"),

    # ── Alexandre Dumas ─────────────────────────────────────────────────────
    # Plot-driven action; no interiority
    (1184,  "Alexandre Dumas", "The Three Musketeers"),
    (1257,  "Alexandre Dumas", "Twenty Years After"),
    (2759,  "Alexandre Dumas", "The Count of Monte Cristo"),

    # ── Rudyard Kipling ─────────────────────────────────────────────────────
    # Short stories / adventure; externalized narration
    (2775,  "Rudyard Kipling", "Plain Tales from the Hills"),
    (2226,  "Rudyard Kipling", "The Jungle Book"),
    (236,   "Rudyard Kipling", "Kim"),  # note: Kim has some interiority, use with caution

    # ── Jack London ─────────────────────────────────────────────────────────
    # Naturalist adventure; animal narration is externalized
    (910,   "Jack London", "The Call of the Wild"),
    (1164,  "Jack London", "White Fang"),
    (1696,  "Jack London", "The Sea-Wolf"),

    # ── O. Henry ────────────────────────────────────────────────────────────
    # American short stories; ironic external narration
    (2776,  "O. Henry", "Cabbages and Kings"),

    # ── Anthony Trollope (extra) ─────────────────────────────────────────────
    # Adding more Trollope for range; he's one of the cleanest choices
    (4735,  "Anthony Trollope", "He Knew He Was Right"),

    # ── W.W. Jacobs ─────────────────────────────────────────────────────────
    # Comic/horror short stories; externalized
    (7234,  "W.W. Jacobs", "The Monkey's Paw and Other Stories"),

    # ── Non-fiction (very safe) ──────────────────────────────────────────────
    # Essays and journalism are the cleanest possible negatives:
    # argument-structure prose has no interior focalization by design.
    (1726,  "Jonathan Swift",    "Gulliver's Travels"),  # satirical frame narrative
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
# Download helpers
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
    """Generate a clean filename from metadata."""
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
        logger.warning("  FAILED pg%d (%s)", gutenberg_id, title)
        return False

    clean = _strip_gutenberg_boilerplate(raw)
    if len(clean.split()) < 1000:
        logger.warning("  SKIP pg%d — stripped text too short (%d words)", gutenberg_id, len(clean.split()))
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
        description="Download curated Gutenberg texts as negative SoC training examples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("negatives"),
        help="Directory to save .txt files (default: negatives/)",
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
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between downloads (default: 2.0; be respectful)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the curated text list and exit without downloading",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':>7}  {'Author':<28}  Title")
        print("-" * 70)
        for gid, author, title in sorted(set(CURATED_TEXTS)):
            print(f"{gid:>7}  {author:<28}  {title}")
        return

    # Resolve which texts to download
    if args.ids:
        id_set = set(args.ids)
        texts = [(gid, author, title) for gid, author, title in CURATED_TEXTS if gid in id_set]
        missing = id_set - {gid for gid, *_ in texts}
        if missing:
            logger.warning("IDs not in curated list (will still attempt): %s", missing)
            for gid in sorted(missing):
                texts.append((gid, "unknown", f"pg{gid}"))
    else:
        # Deduplicate by ID (keep first occurrence)
        seen: set[int] = set()
        texts = []
        for gid, author, title in CURATED_TEXTS:
            if gid not in seen:
                texts.append((gid, author, title))
                seen.add(gid)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    n_ok = n_fail = 0
    for gid, author, title in texts:
        ok = download_text(gid, author, title, args.output_dir, session, args.delay)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info(
        "Done: %d downloaded / already present, %d failed. Output: %s/",
        n_ok, n_fail, args.output_dir,
    )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
