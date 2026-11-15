"""Download the non-Woolf career-arc expansion texts into positives/.

The scored corpus is lopsided in a way that blocks the work-to-work
comparison the chapter wants to make: Woolf has 10 works over 26 years
(1915-1941), while every other author is a 1-3 year snapshot (Richardson
1915-17, Faulkner 1929-30, James 1902-03, Ford 1915 alone). Any claim
about how an author's mediation level *develops* is currently only
sayable about Woolf.

These are the public-domain texts that close that gap, verified
downloadable from Gutenberg on 2026-09-29. Reuses
scripts/download_gutenberg.py's download_text() rather than re-rolling a
fetcher, so the new files land with exactly the same boilerplate
stripping and naming as the existing positives/ (front matter such as
transcriber credits and introductions is retained, matching e.g.
pg03019_pointed_roofs.txt — segment.py is what handles the rest).

Richardson's Pilgrimage is the headline case: volumes 4-9 are all
available, taking her from 3 works/2 years to 9 works/12 years, which is
the only non-Woolf arc in the corpus long enough to be compared with
Woolf's on its own terms. Note the caveat for the writeup — Richardson
insisted Pilgrimage was one novel in chapters, not nine novels, so
"work-to-work" means something different for her than for Woolf.

Ford is the other acute gap (n=1 as a control). Parade's End vols 1-2 are
available; vols 3-4 (A Man Could Stand Up-, 1926; Last Post, 1928) are
NOT on Gutenberg, so the tetralogy stays incomplete either way.

Faulkner cannot be extended from Gutenberg: only The Sound and the Fury
is there. Soldiers' Pay (1926), Mosquitoes (1927) and Sartoris (1929) are
US public domain now but would have to come from another source.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

from scripts.download_gutenberg import download_text

# (gutenberg_id, author, title, year) — year is for the manifest, not the filename.
RICHARDSON_PILGRIMAGE = [
    (56447, "Richardson", "The Tunnel", 1919),
    (57395, "Richardson", "Interim", 1919),
    (56925, "Richardson", "Deadlock", 1921),
    (62967, "Richardson", "Revolving Lights", 1923),
    (64585, "Richardson", "The Trap", 1925),
    (77646, "Richardson", "Oberland", 1927),
]

FORD = [
    (64248, "Ford", "Some Do Not", 1924),
    (67622, "Ford", "No More Parades", 1925),
]

# Optional pre-Good Soldier Ford baseline, so his arc has a "before" as
# well as an "after". Not downloaded unless --with-early-ford.
FORD_EARLY = [
    (30188, "Ford", "The Fifth Queen", 1906),
    (71529, "Ford", "A Call", 1910),
]

# May Sinclair: the author who gave the field its term. She used "stream of
# consciousness" as a critical descriptor in 1918, reviewing Richardson's
# Pilgrimage in The Egoist, and then wrote in the mode herself. Her arc is
# picked to straddle that 1918 coinage rather than to sample her output
# evenly — Audrey Craven and The Helpmate are the conventional "before",
# The Tree of Heaven through Harriett Frean the "after". 1897-1922 is a
# 25-year span, the only non-Woolf arc in the corpus comparable in length
# to Woolf's 26.
#
# Publication years below are asserted from scholarship, NOT from
# Gutenberg: dcterms:issued in the RDF is the date PG posted the file
# (often the 2000s), not the date of publication. verify_corpus.py can
# therefore only bound these by the author's lifespan, not confirm them.
SINCLAIR = [
    (29766, "Sinclair", "Audrey Craven", 1897),
    (17867, "Sinclair", "The Helpmate", 1907),
    (28461, "Sinclair", "The Combined Maze", 1913),
    (11876, "Sinclair", "The Three Sisters", 1914),
    (13883, "Sinclair", "The Tree of Heaven", 1917),
    ( 9366, "Sinclair", "Mary Olivier: A Life", 1919),
    (13292, "Sinclair", "The Romantic", 1920),
    ( 9298, "Sinclair", "Life and Death of Harriett Frean", 1922),
]


# Henry James, trimmed to four career points (2026-09-29, second pass).
#
# Added after the first pass was challenged for leaving every male author a
# snapshot: of the four, only Joyce is genuinely blocked (Finnegans Wake is
# not US public domain until 2035). James was available all along and was
# parked as "just a control" — but Section 1.2 of the chapter makes
# Woolf-vs-Joyce its central contrast, and Joyce is precisely the author who
# cannot be given an arc. James is the nearest available stand-in for "the
# male modernist whose technique the tradition prioritised", which makes a
# developmental reading of him load-bearing rather than optional.
#
# Four points rather than his full bibliography: 1875-1904 is a 29-year
# span, longer than Woolf's 26, at roughly half the scoring cost. The
# Bostonians, The Tragic Muse and The Awkward Age would densify the
# "major phase" transition if the budget later allows.
#
# Portrait of a Lady is two Gutenberg volumes, so this is 5 files for 4
# works — the same split Wings of the Dove already has in the manifest.
JAMES = [
    (  176, "James", "Roderick Hudson",                1875),
    ( 2833, "James", "The Portrait of a Lady (vol. 1)", 1881),
    ( 2834, "James", "The Portrait of a Lady (vol. 2)", 1881),
    ( 7118, "James", "What Maisie Knew",               1897),
    ( 4264, "James", "The Golden Bowl",                1904),
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )
    texts = RICHARDSON_PILGRIMAGE + FORD + SINCLAIR + JAMES
    if "--with-early-ford" in sys.argv:
        texts = texts + FORD_EARLY

    out = ROOT / "positives"
    session = requests.Session()
    ok = 0
    for gid, author, title, _year in texts:
        if download_text(gid, author, title, out, session, delay=2.0):
            ok += 1
    print(f"\n{ok}/{len(texts)} texts present in {out}")


if __name__ == "__main__":
    main()
