"""Register the eight coauthor-supplied texts (2026-10-01).

These fill the gaps the coauthor asked for: Faulkner's experimental middle
period and the last three Pilgrimage volumes published in Richardson's
lifetime. They come from the coauthor directly, not from any public source,
so verify_corpus.py's provenance layer cannot reach them — each entry
carries a provenance_note instead, exactly as the nine Woolf texts and
As I Lay Dying do.

Quality is better than the Standard Ebooks ingest, measured the same way:
no-vowel tokens 0.07-0.12% (SE proofread sits at 0.26%), lone letters at or
below 0.02%, zero U+FFFD. These are clean digital texts, not scans.

Two things are fixed on the way in:

1. Four files carry a UTF-8 BOM, which would otherwise appear as a stray
   \\ufeff inside the first passage of each.

2. NOTHING is done to the quote convention here, deliberately, but it is
   recorded per entry. Richardson's new volumes use single quotes for
   dialogue (0.47-1.69 double quotes per 1k words) where her Gutenberg
   volumes 1-9 use double (12.5-40.0). That is a typesetting difference,
   not a style change — Pilgrimage is one continuous work — and it lands on
   the exact feature Cohn's L1/L2 boundary depends on, since level 2 is
   *framed* interior speech. Normalising silently would hide it; ignoring it
   would let a typographic switch masquerade as Richardson finally departing
   from her plateau in 1931. The quote_convention field lets the analysis
   test for the artefact instead. Faulkner's new texts need no such flag:
   they match his existing volumes at 51-66 double quotes per 1k, and the
   two low ones (Absalom 5.88, Light in August 25.75) are low for reasons
   internal to those books.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POS = ROOT / "positives"
MANIFEST = Path(__file__).resolve().parent / "corpus_manifest.json"

NOTE = (
    "Supplied by the project coauthor on 2026-10-01; not available from Gutenberg, "
    "Standard Ebooks, Faded Page or Internet Archive, so no external catalogue can "
    "verify it. Contents pinned by sha256 at ingest — this detects later drift, it "
    "does not confirm the original acquisition. Text quality measured at ingest: "
    "no-vowel tokens <=0.12%, lone letters <=0.02%, zero replacement characters, "
    "i.e. clean digital text rather than OCR."
)

# (file, author, title, first-publication year, role, quote_convention)
TEXTS = [
    ("mosquitoes_cleaned.txt",       "Faulkner",   "Mosquitoes",        1927, "core", "double"),
    ("sanctuary_cleaned.txt",        "Faulkner",   "Sanctuary",         1931, "core", "double"),
    ("light_in_august_cleaned.txt",  "Faulkner",   "Light in August",   1932, "core", "double"),
    ("pylon_cleaned.txt",            "Faulkner",   "Pylon",             1935, "core", "double"),
    ("absalom_absalom_cleaned.txt",  "Faulkner",   "Absalom, Absalom!", 1936, "core", "double"),
    ("dawns_left_hand_cleaned.txt",  "Richardson", "Dawn's Left Hand",  1931, "core", "single"),
    ("clear_horizon_cleaned.txt",    "Richardson", "Clear Horizon",     1935, "core", "single"),
    ("dimple_hill_cleaned.txt",      "Richardson", "Dimple Hill",       1938, "core", "single"),
]
VOLUME = {"dawns_left_hand_cleaned.txt": 10, "clear_horizon_cleaned.txt": 11,
          "dimple_hill_cleaned.txt": 12}


def main() -> None:
    stripped = []
    for sf, *_ in TEXTS:
        p = POS / sf
        raw = p.read_bytes()
        if raw[:3] == b"\xef\xbb\xbf":
            p.write_bytes(raw[3:]); stripped.append(sf); continue
        t = raw.decode("utf-8", "replace")
        if t.startswith("﻿"):
            p.write_text(t.lstrip("﻿"), encoding="utf-8"); stripped.append(sf)
    print(f"  stripped BOM from {len(stripped)} file(s): {', '.join(s[:20] for s in stripped)}")

    doc = json.loads(MANIFEST.read_text())
    have = {e["source_file"] for e in doc["corpus"]}
    added = 0
    for sf, author, title, year, role, quotes in TEXTS:
        if sf in have:
            print(f"  already in manifest: {sf}"); continue
        entry = {
            "source_file": sf, "author": author, "title": title, "year": year,
            "role": role,
            "word_count": len((POS / sf).read_text(encoding="utf-8").split()),
            "title_on_titlepage": False,
            "provenance_note": NOTE,
            "quote_convention": quotes,
        }
        if sf in VOLUME:
            entry["volume"] = VOLUME[sf]
        doc["corpus"].append(entry)
        added += 1
    doc["notes"] += (
        " 2026-10-01: added eight coauthor-supplied texts — Faulkner's Mosquitoes (1927), "
        "Sanctuary (1931), Light in August (1932), Pylon (1935) and Absalom, Absalom! (1936), "
        "and Pilgrimage volumes 10-12 (Dawn's Left Hand 1931, Clear Horizon 1935, Dimple Hill "
        "1938). No external catalogue holds these, so each carries a provenance_note. A new "
        "quote_convention field records that Richardson's volumes 10-12 use single quotes for "
        "dialogue where volumes 1-9 use double — a typesetting difference sitting directly on "
        "Cohn's L1/L2 boundary, which must be tested for rather than assumed harmless. "
        "SEVEN OF THESE EIGHT ARE IN US COPYRIGHT and have not been scored."
    )
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"  added {added} entries; manifest now {len(doc['corpus'])} works")


if __name__ == "__main__":
    main()
