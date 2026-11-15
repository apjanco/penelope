"""Assert every corpus text is the book the manifest claims it is.

Motivated by a failure this project has already hit: a Gutenberg ID is
transcribed or guessed wrong, the downloader happily saves *something*
under the intended filename, and the mistake only surfaces much later as
an inexplicable result — by which point it has been scored, analysed and
possibly written up. `positives/` also still shows the scar of a related
error (`pg05670_jacobs_room.txt`, a duplicate of `jacobs_room.txt`, is
carried in the manifest's `excluded` list).

The check has three layers, because the corpus has two kinds of file and
they admit different kinds of proof:

1. PINNING (offline, every file). sha256 + word count + a recorded
   opening excerpt must match what the manifest records. This does not
   prove a file is the right book; it proves the file has not changed
   since someone last confirmed it was. It is what makes the other two
   layers durable rather than a one-off inspection.

2. PROVENANCE (online, Gutenberg-sourced files only). The filename's
   `pg<id>` is resolved against Gutenberg's own RDF metadata and the
   returned title and author must match the manifest. This is the layer
   that actually catches "wrong ID, plausible filename". Volume numbers
   are checked separately and deliberately — a vol.1/vol.2 mix-up passes
   a title check trivially, and the corpus contains two multi-volume
   works where that could happen silently.

3. CORROBORATION (offline, files whose text carries a title page). The
   manifest's title must appear in the text's own front matter. Weaker
   than layer 2 but needs no network, and it is checked against the file
   itself rather than against a catalogue.

Four Woolf texts (The Voyage Out, Night and Day, Jacob's Room, To the
Lighthouse, Flush) open directly at "CHAPTER I" with no title page, and
none of the Woolf texts came from Gutenberg. For those, layers 2 and 3
are unavailable and the manifest says so explicitly per entry rather
than the checker guessing. Their provenance rests on layer 1 plus the
recorded `provenance_note` — an honest limit, not a silent pass.

Usage:
    python verify_corpus.py                 # offline layers (1 + 3)
    python verify_corpus.py --online        # all three layers
    python verify_corpus.py --bless         # record hashes for new entries
    pytest test_corpus_integrity.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIVES_DIR = ROOT / "positives"
MANIFEST_PATH = ROOT / "technique_v_genre" / "corpus_manifest.json"
RDF_CACHE = ROOT / "technique_v_genre" / "metadata_cache"

USER_AGENT = "penelope-corpus-verifier/1.0 (research; aj7878@princeton.edu)"
RDF_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.rdf"
# Standard Ebooks publishes every title's canonical content.opf to a public
# GitHub repo. That is the machine-readable external record for SE-sourced
# files, playing exactly the role Gutenberg's RDF plays for pg* files —
# cheap to fetch and stable, unlike SE's OPDS feeds, which are gated (401).
SE_OPF_URL = ("https://raw.githubusercontent.com/standardebooks/"
              "{repo}/master/src/epub/content.opf")

OPENING_CHARS = 120          # length of the recorded opening excerpt
TITLEPAGE_WINDOW = 3000      # chars of front matter searched for the title
PG_FILENAME_RE = re.compile(r"^pg(\d{4,5})_")

ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase, strip MARC subfield markers and punctuation, collapse space."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("$b", " ")  # MARC subfield delimiter, leaks into PG titles
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_core(title: str) -> str:
    """The title without subtitle or volume apparatus, normalised.

    'Some do not...: A novel'            -> 'some do not'
    'The tunnel : $b Pilgrimage, Vol. 4' -> 'the tunnel'
    'The Wings of the Dove (vol. 1)'     -> 'the wings of the dove'
    """
    t = re.sub(r"\(.*?\)", " ", title)          # parenthetical vol markers
    t = t.split(":")[0]                          # subtitle after a colon
    t = re.sub(r"\bvol(ume)?\b.*$", " ", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*$", "", t.strip())
    return normalize(t)


def extract_volume(title: str) -> int | None:
    """Volume number named in a title, arabic or roman. None if unmarked.

    Gutenberg is inconsistent here in a way that matters: The Wings of the
    Dove is 'Volume 1 of 2' for the first half and 'Volume II' for the
    second. Comparing the rendered strings would flag a correct pair as a
    mismatch, so both forms are parsed to an int before comparison.
    """
    n = normalize(title)
    m = re.search(r"\bvol(?:ume)?\s+(\d+)\b", n)
    if m:
        return int(m.group(1))
    m = re.search(r"\bvol(?:ume)?\s+([ivx]+)\b", n)
    if m and m.group(1) in ROMAN:
        return ROMAN[m.group(1)]
    return None


def surname(author_name: str) -> str:
    """Surname from either catalogue order or natural order.

    Gutenberg writes 'Richardson, Dorothy M. (Dorothy Miller)'; Standard
    Ebooks writes 'William Faulkner'. Comma present means the surname leads;
    otherwise it trails.
    """
    if "," in author_name:
        return normalize(author_name.split(",")[0]).split(" ")[0]
    parts = normalize(author_name).split(" ")
    return parts[-1] if parts else ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def opening_excerpt(text: str) -> str:
    return normalize(text[: OPENING_CHARS * 4])[:OPENING_CHARS]


# ---------------------------------------------------------------------------
# Gutenberg metadata
# ---------------------------------------------------------------------------

@dataclass
class PGMetadata:
    gutenberg_id: int
    title: str
    author: str
    birthdate: int | None
    deathdate: int | None


def fetch_rdf(gid: int, *, use_cache: bool = True, delay: float = 0.5) -> str:
    RDF_CACHE.mkdir(parents=True, exist_ok=True)
    cached = RDF_CACHE / f"pg{gid}.rdf"
    if use_cache and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(RDF_URL.format(id=gid), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    cached.write_text(body, encoding="utf-8")
    time.sleep(delay)  # be a good Gutenberg citizen
    return body


def one_tag(body: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", body, re.S)
    if not m:
        return None
    import html as _html
    return _html.unescape(m.group(1).strip())


def fetch_se_opf(repo: str, *, use_cache: bool = True, delay: float = 0.3) -> str:
    RDF_CACHE.mkdir(parents=True, exist_ok=True)
    cached = RDF_CACHE / f"se_{repo}.opf"
    if use_cache and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(SE_OPF_URL.format(repo=repo),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    cached.write_text(body, encoding="utf-8")
    time.sleep(delay)
    return body


def parse_rdf(gid: int, body: str) -> PGMetadata:
    def one(tag: str) -> str | None:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", body, re.S)
        return m.group(1).strip() if m else None

    title = one("dcterms:title")
    names = re.findall(r"<pgterms:name[^>]*>(.*?)</pgterms:name>", body, re.S)
    birth, death = one("pgterms:birthdate"), one("pgterms:deathdate")
    return PGMetadata(
        gutenberg_id=gid,
        title=title or "",
        author=names[0].strip() if names else "",
        birthdate=int(birth) if birth and birth.lstrip("-").isdigit() else None,
        deathdate=int(death) if death and death.lstrip("-").isdigit() else None,
    )


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

@dataclass
class Result:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: int = 0

    def fail(self, sf: str, msg: str) -> None:
        self.failures.append(f"{sf}: {msg}")

    def warn(self, sf: str, msg: str) -> None:
        self.warnings.append(f"{sf}: {msg}")


def verify(manifest: list[dict], *, online: bool = False, use_cache: bool = True) -> Result:
    res = Result()
    # Keyed by manifest position, not by source_file: keying on the filename
    # made a repeated entry invisible to its own duplicate check, since the
    # second occurrence compared equal to the first and was skipped.
    seen_sha: dict[str, tuple[int, str]] = {}
    seen_opening: dict[str, tuple[int, str]] = {}
    seen_source_file: dict[str, int] = {}

    for idx, entry in enumerate(manifest):
        sf = entry["source_file"]

        if sf in seen_source_file:
            res.fail(sf, f"listed twice in the manifest (entries {seen_source_file[sf]} and {idx})")
        seen_source_file.setdefault(sf, idx)

        path = POSITIVES_DIR / sf
        res.checked += 1

        # --- layer 1: pinning -------------------------------------------------
        if not path.exists():
            res.fail(sf, "file missing from positives/")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            res.fail(sf, "file is empty")
            continue

        words = len(text.split())
        if words != entry.get("word_count"):
            res.fail(sf, f"word_count {words} != manifest {entry.get('word_count')}")

        digest = sha256_of(path)
        if "sha256" not in entry:
            res.warn(sf, "no sha256 pinned in manifest — run with --bless")
        elif digest != entry["sha256"]:
            res.fail(sf, f"sha256 {digest[:16]}… != pinned {entry['sha256'][:16]}… (file changed)")

        opening = opening_excerpt(text)
        if "opening" in entry and opening != entry["opening"]:
            res.fail(sf, f"opening changed\n     pinned: {entry['opening'][:70]!r}\n     actual: {opening[:70]!r}")

        if digest in seen_sha and seen_sha[digest][0] != idx:
            prev = seen_sha[digest][1]
            res.fail(sf, f"byte-identical duplicate of {prev} "
                         f"(entries {seen_sha[digest][0]} and {idx})")
        seen_sha.setdefault(digest, (idx, sf))
        if opening in seen_opening and seen_opening[opening][0] != idx:
            res.fail(sf, f"opens identically to {seen_opening[opening][1]} — likely the same text")
        seen_opening.setdefault(opening, (idx, sf))

        # --- layer 3: corroboration from the text's own title page ------------
        core = title_core(entry["title"])
        head = normalize(text[:TITLEPAGE_WINDOW])
        has_titlepage = entry.get("title_on_titlepage")
        if has_titlepage is None:
            res.warn(sf, "manifest does not declare title_on_titlepage")
        elif has_titlepage:
            if core and core not in head:
                res.fail(sf, f"title {core!r} absent from first {TITLEPAGE_WINDOW} chars, "
                             f"but manifest claims a title page; head={head[:70]!r}")

        # Anything with no Gutenberg record cannot reach layer 2 at all, so
        # the manifest must at least say where it came from. Checked here
        # rather than folded into the title-page branch: a file can have a
        # perfectly good title page and still have untraceable provenance
        # (as_I_lay_dying.txt does).
        if (not entry.get("gutenberg_id") and not entry.get("standard_ebooks_id")
                and not entry.get("provenance_note")):
            res.fail(sf, "no gutenberg_id and no provenance_note — origin is unverifiable "
                         "and undocumented")

        # --- layer 2: provenance against Gutenberg ----------------------------
        m = PG_FILENAME_RE.match(sf)
        gid_from_name = int(m.group(1)) if m else None
        gid = entry.get("gutenberg_id")

        if gid_from_name is not None and gid is None:
            res.fail(sf, f"filename says pg{gid_from_name} but manifest has no gutenberg_id")
        elif gid is not None and gid_from_name is not None and gid != gid_from_name:
            res.fail(sf, f"filename pg{gid_from_name} contradicts manifest gutenberg_id {gid}")

        se_id = entry.get("standard_ebooks_id")
        if online and se_id:
            repo = se_id.replace("/", "_")
            try:
                opf = fetch_se_opf(repo, use_cache=use_cache)
            except (urllib.error.URLError, OSError) as exc:
                res.warn(sf, f"could not fetch Standard Ebooks metadata for {se_id}: {exc}")
            else:
                se_title = one_tag(opf, "dc:title")
                se_author = one_tag(opf, "dc:creator")
                se_ident = one_tag(opf, "dc:identifier")
                if not (core and se_title and (core in title_core(se_title)
                                               or title_core(se_title) in core)):
                    res.fail(sf, f"{se_id} is titled {se_title!r} at Standard Ebooks, "
                                 f"manifest claims {entry['title']!r} — WRONG BOOK")
                if se_author and surname(se_author) != normalize(entry["author"]):
                    res.fail(sf, f"{se_id} is by {se_author!r}, manifest says {entry['author']!r}")
                if se_ident and se_id not in se_ident:
                    res.fail(sf, f"{se_id} resolves to a different canonical id: {se_ident!r}")

        if not online or gid is None:
            continue

        try:
            meta = parse_rdf(gid, fetch_rdf(gid, use_cache=use_cache))
        except (urllib.error.URLError, OSError) as exc:
            res.warn(sf, f"could not fetch Gutenberg metadata for pg{gid}: {exc}")
            continue

        pg_core = title_core(meta.title)
        if not (core and pg_core and (core in pg_core or pg_core in core)):
            res.fail(sf, f"pg{gid} is titled {meta.title!r} on Gutenberg, "
                         f"manifest claims {entry['title']!r} — WRONG BOOK")

        # The explicit `volume` field wins. Previously this was
        # `extract_volume(title) or entry.get("volume")`, which let a title
        # reading "(vol. 1)" silently mask a `volume: 2` field — precisely
        # the swap this check exists to catch. A disagreement between the two
        # is now a fault in its own right, before Gutenberg is consulted.
        title_vol = extract_volume(entry["title"])
        field_vol = entry.get("volume")
        if title_vol is not None and field_vol is not None and title_vol != field_vol:
            res.fail(sf, f"manifest contradicts itself: title {entry['title']!r} says vol "
                         f"{title_vol}, volume field says {field_vol}")
        want_vol = field_vol if field_vol is not None else title_vol
        got_vol = extract_volume(meta.title)
        if want_vol is not None and got_vol is not None and want_vol != got_vol:
            res.fail(sf, f"volume mismatch: manifest says vol {want_vol}, "
                         f"pg{gid} is vol {got_vol} ({meta.title!r})")
        elif want_vol is not None and got_vol is None:
            res.warn(sf, f"manifest claims vol {want_vol}; pg{gid} title {meta.title!r} names no volume")

        if surname(meta.author) != normalize(entry["author"]):
            res.fail(sf, f"pg{gid} is by {meta.author!r}, manifest says {entry['author']!r}")

        year = entry["year"]
        if meta.birthdate and year < meta.birthdate + 15:
            res.fail(sf, f"year {year} precedes {meta.author}'s 15th birthday ({meta.birthdate})")
        if meta.deathdate and year > meta.deathdate + 5:
            res.fail(sf, f"year {year} is more than 5 years after {meta.author} died ({meta.deathdate})")

    return res


def bless(manifest: list[dict]) -> int:
    """Record sha256 / word_count / opening for entries that lack them."""
    n = 0
    for entry in manifest:
        path = POSITIVES_DIR / entry["source_file"]
        if not path.exists():
            print(f"  skip (missing) {entry['source_file']}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "sha256" in entry and "opening" in entry:
            continue
        entry["sha256"] = sha256_of(path)
        entry["opening"] = opening_excerpt(text)
        entry["word_count"] = len(text.split())
        print(f"  pinned {entry['source_file']} ({entry['word_count']:,} words)")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true", help="also verify against Gutenberg metadata")
    ap.add_argument("--no-cache", action="store_true", help="refetch RDF instead of using metadata_cache/")
    ap.add_argument("--bless", action="store_true", help="record hashes for entries that lack them")
    args = ap.parse_args()

    doc = json.loads(MANIFEST_PATH.read_text())
    manifest = doc["corpus"]

    if args.bless:
        n = bless(manifest)
        MANIFEST_PATH.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\npinned {n} entries -> {MANIFEST_PATH.name}")
        return 0

    res = verify(manifest, online=args.online, use_cache=not args.no_cache)

    for w in res.warnings:
        print(f"  WARN  {w}")
    for f in res.failures:
        print(f"  FAIL  {f}")

    layers = "pinning + corroboration" + (" + Gutenberg/Standard Ebooks provenance" if args.online else "")
    print(f"\n{res.checked} works checked ({layers})")
    print(f"{len(res.failures)} failures, {len(res.warnings)} warnings")
    if not args.online:
        print("note: run with --online to verify Gutenberg IDs resolve to the claimed books")
    return 1 if res.failures else 0


if __name__ == "__main__":
    sys.exit(main())
