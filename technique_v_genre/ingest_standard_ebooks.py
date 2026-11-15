"""Ingest Standard Ebooks titles into positives/ as plain text.

Why a second ingest path at all: the five works still missing from the
corpus (Faulkner 1926-29, the last two Parade's End volumes) are all US
public domain but none are on Gutenberg, and Princeton's online holdings
resolve either to Gutenberg passthroughs, audiobooks, or physical volumes
in ReCAP. Standard Ebooks has three of the five as hand-proofread text —
no OCR, and measurably so: its quotation marks balance exactly (279/279 on
The Last Post) where Internet Archive's scans of the same period run 3-5%
unbalanced. That matters more here than it would elsewhere, because the
level-1/level-2 boundary on Cohn's cline *is* the quotation mark.

What is kept and dropped. Standard Ebooks epubs carry three files that
describe the edition rather than the work — imprint, colophon and
uncopyright — and they are dropped, the same way scripts/download_gutenberg
strips Gutenberg's boilerplate. Left in, they would inject "Standard
Ebooks", "public domain" and transcription credits into scored passages.
Title page, half title, epigraph and dedication are kept, matching the
Gutenberg-sourced files in positives/, which retain their front matter
(and which verify_corpus.py's title_on_titlepage check relies on).

Spine order, not filename order: chapter-1-10 sorts before chapter-1-2
alphabetically, so the epub's declared spine is the only correct reading
order.

Provenance: every book has a public GitHub repo carrying the canonical
content.opf, which verify_corpus.py checks the way it checks Gutenberg RDF.
The epub's own dc:source records the upstream transcription — for The Last
Post, a Faded Page proofread text over a Google Books scan.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIVES_DIR = ROOT / "positives"

USER_AGENT = "penelope-corpus-ingest/1.0 (research; aj7878@princeton.edu)"
EPUB_URL = ("https://standardebooks.org/ebooks/{slug}/downloads/"
            "{flat}.epub?source=download")

# Files describing the Standard Ebooks edition rather than the work itself.
SE_BOILERPLATE = {"imprint", "colophon", "uncopyright"}

# (slug, author, title, first-publication year)
# Years are first publication, NOT the Standard Ebooks release date — the
# corpus's whole diachronic axis is publication year.
BOOKS = [
    ("william-faulkner/soldiers-pay",        "Faulkner", "Soldiers' Pay",        1926),
    ("ford-madox-ford/a-man-could-stand-up", "Ford",     "A Man Could Stand Up—", 1926),
    ("ford-madox-ford/the-last-post",        "Ford",     "The Last Post",         1928),
]


def fetch(url: str, *, retries: int = 5) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
            print(f"    attempt {i+1}/{retries} failed: {exc}")
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def epub_to_text(blob: bytes) -> tuple[str, dict[str, str]]:
    z = zipfile.ZipFile(BytesIO(blob))
    opf_name = next(n for n in z.namelist() if n.endswith(".opf"))
    opf = z.read(opf_name).decode("utf-8", "replace")

    meta: dict[str, str] = {}
    for tag in ("dc:title", "dc:creator", "dc:identifier", "dc:language"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", opf, re.S)
        if m:
            meta[tag.split(":")[1]] = html.unescape(m.group(1).strip())
    meta["source"] = "; ".join(
        html.unescape(s.strip())
        for s in re.findall(r"<dc:source[^>]*>(.*?)</dc:source>", opf, re.S)
    )

    hrefs = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf))
    for href, ident in re.findall(r'<item[^>]*href="([^"]+)"[^>]*id="([^"]+)"', opf):
        hrefs.setdefault(ident, href)
    spine = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)

    base = opf_name.rsplit("/", 1)[0] if "/" in opf_name else ""
    parts: list[str] = []
    for ref in spine:
        href = hrefs.get(ref)
        if not href:
            continue
        stem = href.rsplit("/", 1)[-1].removesuffix(".xhtml")
        if stem in SE_BOILERPLATE:
            continue
        path = f"{base}/{href}" if base else href
        if path not in z.namelist():
            continue
        s = z.read(path).decode("utf-8", "replace")
        s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
        # Block-level tags become paragraph breaks so segment.py can find
        # paragraph boundaries; everything else collapses to a space.
        s = re.sub(r"</(p|div|h[1-6]|blockquote|li|section)\s*>", "\n\n", s, flags=re.I)
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
        s = re.sub(r"<.*?>", " ", s, flags=re.S)
        s = html.unescape(s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        parts.append(s.strip())
    return "\n\n".join(p for p in parts if p) + "\n", meta


def output_name(slug: str) -> str:
    author, book = slug.split("/")
    return f"se_{author}_{book}.txt".replace("-", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    POSITIVES_DIR.mkdir(parents=True, exist_ok=True)
    for slug, author, title, year in BOOKS:
        dest = POSITIVES_DIR / output_name(slug)
        if dest.exists() and not args.force:
            print(f"  skip (exists) {dest.name}")
            continue
        flat = slug.replace("/", "_")
        print(f"  fetching {slug} — {author}, {title} ({year})")
        blob = fetch(EPUB_URL.format(slug=slug, flat=flat))
        text, meta = epub_to_text(blob)
        words = len(text.split())
        if words < 10_000:
            print(f"    REFUSING: only {words} words extracted — check the spine filter")
            continue
        dest.write_text(text, encoding="utf-8")
        print(f"    saved {dest.name} ({words:,} words)")
        print(f"    opf: {meta.get('title')!r} / {meta.get('creator')!r}")
        print(f"    upstream source: {meta.get('source','')[:96]}")
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
