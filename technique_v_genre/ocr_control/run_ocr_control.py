"""Does OCR'd text shift the mediation-level judgements?

The corpus is clean e-text throughout. Internet Archive scans would let it
grow (Faulkner's Mosquitoes 1927 and Sartoris 1929 exist nowhere else), but
they would be the only OCR'd members — a source difference that correlates
perfectly with author, sitting on top of the exact feature the primary
measurement uses. Cohn's cline separates level 1 (narrated monologue) from
level 2 (quoted monologue) by whether the interior speech is framed in
quotation marks, and OCR damages quotation marks specifically.

Faulkner's Soldiers' Pay exists in both forms, and not merely
coincidentally: Standard Ebooks names this very Internet Archive scan as its
upstream source, so the pair is one edition lineage — raw OCR against its
own proofread descendant. Same words, same edition, differing only in
transcription quality.

DESIGN (revised 2026-10-02 — paired, replacing the first draft's two
independently segmented samples). The proofread text is already a corpus
member: all 537 of its passages are scored by gpt-5.4 and qwen3-max at k=3
in scored/. So we sample 260 of those passages, locate the span of raw OCR
that transcribes the same words, and score only the OCR spans, with the
corpus protocol (same readers, same prompt, k=3). Each OCR judgement then
has an exact clean twin, and the readers' own run-to-run variation on the
clean text is the noise floor: OCR matters only if clean-vs-OCR disagreement
exceeds clean-vs-clean disagreement.

The OCR span is taken as the scan gives it — running heads, page numbers,
line-end hyphenation, dropped apostrophes and all — with only whitespace
collapsed, which is what our ingestion would do to a scanned text.

Prediction, stated before scoring (unchanged from the first draft): if OCR
is harmless, OCR-vs-clean disagreement sits at the run-to-run floor and the
level distributions match. If the quote damage matters, level-2 mass
migrates to level 1 or 0 in the OCR arm, concentrated in passages whose
quotation marks the scan lost, because framed interior speech stripped of
its frame reads as unframed.

Nothing here is written to positives/, passages/, scored/ or the manifest.

Usage:  python ocr_control/run_ocr_control.py            # build + report
        python ocr_control/run_ocr_control.py --score    # build + score + report
"""
from __future__ import annotations

import collections
import difflib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent

OCR_TEXT = HERE / "texts" / "ia_soldiers_pay_raw_ocr.txt"
CLEAN_PASSAGES = TV / "passages" / "se_william_faulkner_soldiers_pay.jsonl"
CLEAN_SCORED = TV / "scored" / "se_william_faulkner_soldiers_pay.jsonl"
PAIRED = HERE / "passages" / "ocr_paired.jsonl"
OCR_SCORED = HERE / "scored" / "ocr_paired.jsonl"

SAMPLE = 260
READERS = ["gpt-5.4", "qwen3-max"]
K = 3                 # same as the corpus, so run-to-run variance is comparable
NGRAM = 5
SEED = 20261002

WORD_RE = re.compile(r"[A-Za-z0-9]+")
QUOTES = "\"“”"


def tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group().lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def build_index(toks: list[tuple[str, int, int]]) -> dict[tuple, list[int]]:
    idx = collections.defaultdict(list)
    words = [t[0] for t in toks]
    for i in range(len(words) - NGRAM + 1):
        idx[tuple(words[i:i + NGRAM])].append(i)
    return idx


def anchor(words: list[str], idx, from_end: bool) -> int | None:
    """Position in the OCR word stream of the passage's first (or last)
    word, found via the nearest uniquely-occurring n-gram."""
    n = len(words)
    for off in range(0, min(40, n - NGRAM + 1)):
        i = n - NGRAM - off if from_end else off
        hits = idx.get(tuple(words[i:i + NGRAM]))
        if hits and len(hits) == 1:
            return hits[0] + (NGRAM - 1 + off if from_end else -off)
    return None


def build() -> None:
    if PAIRED.exists():
        print(f"  paired passages exist: {PAIRED.name}")
        return
    raw = OCR_TEXT.read_text()
    otoks = tokens(raw)
    idx = build_index(otoks)
    clean = [json.loads(l) for l in CLEAN_PASSAGES.read_text().splitlines() if l.strip()]
    body = clean[5:]                       # skip front matter
    rng = random.Random(SEED)
    order = rng.sample(range(len(body)), len(body))

    rows, failed = [], 0
    for j in order:
        if len(rows) == SAMPLE:
            break
        p = body[j]
        cw = [t[0] for t in tokens(p["passage_text"])]
        if len(cw) < 2 * NGRAM:
            failed += 1
            continue
        s, e = anchor(cw, idx, False), anchor(cw, idx, True)
        if s is None or e is None or e < s:
            failed += 1
            continue
        ratio = (e - s + 1) / len(cw)
        if not 0.85 <= ratio <= 1.15:      # hyphenation/headers move this a little, not 15%
            failed += 1
            continue
        # The anchors assume one OCR token per clean word, which line-end
        # splits ("con- temporaries") and misreads ("Tou" for "You") break.
        # Refine both edges with a local alignment over a window around the
        # anchored span: the first and last matched blocks fix the edges,
        # and clean words left unmatched beyond them are taken to be their
        # garbled OCR counterparts, one token each.
        w0 = max(0, s - 15)
        win = [t[0] for t in otoks[w0:min(len(otoks), e + 16)]]
        blocks = [b for b in difflib.SequenceMatcher(None, cw, win, autojunk=False)
                  .get_matching_blocks() if b.size]
        if not blocks:
            failed += 1
            continue
        fb, lb = blocks[0], blocks[-1]
        s = max(0, w0 + fb.b - fb.a)
        e = min(len(otoks) - 1, w0 + lb.b + lb.size - 1 + (len(cw) - lb.a - lb.size))
        # A last word split across lines ("good- bye", "some one") occupies
        # two OCR tokens; keep taking tokens while they still spell it out.
        joined = otoks[e][0]
        while (e + 1 < len(otoks) and joined != cw[-1]
               and cw[-1].startswith(joined + otoks[e + 1][0])):
            e += 1
            joined += otoks[e][0]
        # Include the punctuation hugging each edge — above all the opening
        # and closing quotation marks, which are what this control is about.
        a, b = otoks[s][1], otoks[e][2]
        a -= len(re.search(r"[^\sA-Za-z0-9]{0,3}$", raw[max(0, a - 3):a]).group())
        b += len(re.match(r"[^\sA-Za-z0-9]{0,3}", raw[b:]).group())
        ocr_text = re.sub(r"\s+", " ", raw[a:b]).strip()
        rows.append({
            "passage_id": "ocr_" + p["passage_id"],
            "clean_passage_id": p["passage_id"],
            "source_file": OCR_TEXT.name,
            "author": "Faulkner", "title": "Soldiers' Pay", "year": 1926,
            "role": "ocr_control",
            "n_words": len(ocr_text.split()),
            "word_ratio": round(ratio, 3),
            "clean_quotes": sum(p["passage_text"].count(q) for q in QUOTES),
            "ocr_quotes": sum(ocr_text.count(q) for q in QUOTES),
            "passage_text": ocr_text,
        })
    rows.sort(key=lambda r: r["clean_passage_id"])
    PAIRED.parent.mkdir(exist_ok=True)
    PAIRED.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  aligned {len(rows)} passages ({failed} skipped: no unique anchor "
          f"or span length off by >15%) -> {PAIRED.name}")


def score() -> None:
    OCR_SCORED.parent.mkdir(exist_ok=True)
    subprocess.run([
        "/home/aj7878/.conda/envs/penelope/bin/python", str(TV / "score.py"),
        "--input", str(PAIRED), "--readers", *READERS, "--k", str(K),
        "--concurrency", "8", "--output", str(OCR_SCORED),
    ], cwd=TV, check=False)


def load(path: Path, key) -> dict:
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or r.get("reader_id") not in READERS:
            continue
        lvl = r.get("mediation_level") if r.get("is_soc") else None
        out[key(r)][r["reader_id"]][int(r["run"])] = (bool(r.get("is_soc")), lvl)
    return out


def mode(vals):
    c = collections.Counter(v for v in vals if v is not None)
    return c.most_common(1)[0][0] if c else None


def report() -> None:
    pairs = [json.loads(l) for l in PAIRED.read_text().splitlines() if l.strip()]
    meta = {p["clean_passage_id"]: p for p in pairs}
    clean = load(CLEAN_SCORED, lambda r: r["passage_id"])
    ocr = load(OCR_SCORED, lambda r: r["passage_id"].removeprefix("ocr_"))

    qc, qo = sum(p["clean_quotes"] for p in pairs), sum(p["ocr_quotes"] for p in pairs)
    print("=" * 66)
    print(f"Soldiers' Pay: raw OCR vs proofread, {len(pairs)} paired passages")
    print("=" * 66)
    print(f"  quotation marks: clean {qc}, OCR {qo} ({100*(qo-qc)/max(qc,1):+.1f}%)")
    lost = {pid for pid, p in meta.items() if p["ocr_quotes"] < p["clean_quotes"]}
    print(f"  passages that lost >=1 quotation mark in OCR: {len(lost)}")

    for rd in READERS:
        done = [pid for pid in meta if len(ocr[pid][rd]) == K and len(clean[pid][rd]) == K]
        if not done:
            print(f"\n  {rd}: not scored yet")
            continue
        # noise floor: clean run i vs clean run j; treatment: clean run i vs OCR run i
        floor_lvl = floor_n = 0
        cross_lvl = cross_n = 0
        soc_c = soc_o = 0
        dist_c, dist_o = collections.Counter(), collections.Counter()
        shifts, shifts_lost = [], []
        for pid in done:
            c, o = clean[pid][rd], ocr[pid][rd]
            for i in range(K):
                for j in range(i + 1, K):
                    if c[i][1] is not None and c[j][1] is not None:
                        floor_n += 1; floor_lvl += c[i][1] == c[j][1]
                if c[i][1] is not None and o[i][1] is not None:
                    cross_n += 1; cross_lvl += c[i][1] == o[i][1]
                soc_c += c[i][0]; soc_o += o[i][0]
                if c[i][1] is not None: dist_c[c[i][1]] += 1
                if o[i][1] is not None: dist_o[o[i][1]] += 1
            mc, mo = mode([c[i][1] for i in range(K)]), mode([o[i][1] for i in range(K)])
            if mc is not None and mo is not None:
                shifts.append(mo - mc)
                if pid in lost:
                    shifts_lost.append(mo - mc)
        tc, to = sum(dist_c.values()) or 1, sum(dist_o.values()) or 1
        print(f"\n  {rd}  ({len(done)} passages, k={K})")
        print(f"    is_soc rate         clean {100*soc_c/(K*len(done)):5.1f}%   OCR {100*soc_o/(K*len(done)):5.1f}%")
        print(f"    level agreement     clean-vs-clean {100*floor_lvl/max(floor_n,1):5.1f}% (n={floor_n})"
              f"   clean-vs-OCR {100*cross_lvl/max(cross_n,1):5.1f}% (n={cross_n})")
        for lv in range(4):
            a, b = 100 * dist_c[lv] / tc, 100 * dist_o[lv] / to
            print(f"    L{lv}: clean {a:5.1f}%   OCR {b:5.1f}%   ({b-a:+5.1f}pp)")
        if shifts:
            up = sum(s > 0 for s in shifts); dn = sum(s < 0 for s in shifts)
            print(f"    modal level, OCR minus clean: mean {sum(shifts)/len(shifts):+.3f}; "
                  f"{up} up, {dn} down, {len(shifts)-up-dn} same")
        if shifts_lost:
            up = sum(s > 0 for s in shifts_lost); dn = sum(s < 0 for s in shifts_lost)
            print(f"    ...in quote-damaged passages only (n={len(shifts_lost)}): mean "
                  f"{sum(shifts_lost)/len(shifts_lost):+.3f}; {up} up, {dn} down")


if __name__ == "__main__":
    build()
    if "--score" in sys.argv:
        score()
    report()
