"""Annotator agreement: the human readers against every machine reader.

Reads human annotations from human_review/annotations/*.json (written by
ingest_annotations.py from each annotator's exported file) and machine
readings from machine_readings.json, and reports agreement on the primary
variable, Cohn's mediation level.

WHAT THE HEADLINE TEST IS. Not "the humans agree at kappa = x" on its own —
with 33 passages that number carries a 95% interval roughly +/-0.18 wide, so
it can separate substantial agreement from chance and not much more. The
claim worth making is comparative: the machine readers agree with a human
about as well as the two humans agree with each other. That is a PAIRED
comparison on the same passages, so passage difficulty cancels, and
bootstrapping the DIFFERENCE over passages gives a far tighter interval than
either kappa alone. bootstrap_difference() below is the test; the individual
kappas are reported as context for it.

WHY WEIGHTED KAPPA IS THE PRIMARY FORM. The cline is ordinal and the prompt
says so to every reader: levels 0 and 1 apart are near-agreement, 0 and 3 are
real disagreement. Unweighted kappa throws that away by scoring 0-vs-1 and
0-vs-3 as equally wrong. Linear weights are reported first; unweighted and
within-one-level rates are given alongside so nothing is hidden by the choice.

THE TWO BLOCKS ARE REPORTED SEPARATELY and never pooled into one headline.
Section A ("Time Passes") is a deliberately hard crux section whose labels
collapse to a near-binary L0/L1 problem. Section B is stratified across all
four levels. They measure different things.

AUDIT COLUMNS. Each response records the order it was made in and whether the
machine panel had been revealed for that passage beforehand. Agreement is
recomputed excluding any response made after a reveal, so the primary figures
cannot be contaminated by a reader who saw the machines first.
"""
from __future__ import annotations

import collections, glob, itertools, json, math, random, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHINES = ["gpt-5.4", "qwen3-max", "jev", "qwen3.6-local"]


# ---------------------------------------------------------------- agreement

def _kappa(pairs, weights="linear", levels=(0, 1, 2, 3)):
    """Cohen's kappa over (a, b) pairs. weights: 'linear', 'quadratic', None."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    if weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    ra = collections.Counter(a for a, _ in pairs)
    rb = collections.Counter(b for _, b in pairs)
    obs = sum(w[idx[a]][idx[b]] for a, b in pairs) / n
    exp = sum(w[i][j] * (ra.get(levels[i], 0) / n) * (rb.get(levels[j], 0) / n)
              for i in range(k) for j in range(k))
    if exp == 0:
        return 1.0 if obs == 0 else 0.0
    return 1 - obs / exp


def _alpha_ordinal(ratings, levels=(0, 1, 2, 3)):
    """Krippendorff's alpha, ordinal metric. ratings: {unit: {rater: value}}.

    Handles missing values natively, which matters here: a reader that finds
    no consciousness in a passage contributes no level for it.
    """
    units = [u for u, r in ratings.items()
             if len([v for v in r.values() if v is not None]) >= 2]
    if not units:
        return None
    coinc = collections.Counter()
    for u in units:
        vals = [v for v in ratings[u].values() if v is not None]
        m = len(vals)
        for a, b in itertools.permutations(vals, 2):
            coinc[(a, b)] += 1 / (m - 1)
    n = sum(coinc.values())
    marg = collections.Counter()
    for (a, b), c in coinc.items():
        marg[a] += c

    def delta2(c, k):
        lo, hi = min(c, k), max(c, k)
        g = sum(marg.get(v, 0) for v in levels if lo <= v <= hi)
        return (g - (marg.get(c, 0) + marg.get(k, 0)) / 2) ** 2

    do = sum(coinc[(c, k)] * delta2(c, k) for c in levels for k in levels) / n
    de = sum(marg.get(c, 0) * marg.get(k, 0) * delta2(c, k)
             for c in levels for k in levels) / (n * (n - 1))
    return 1 - do / de if de else None


def _icc2k(matrix):
    """ICC(2,k), two-way random effects, absolute agreement, k raters."""
    rows = [r for r in matrix if all(v is not None for v in r)]
    n, k = len(rows), len(rows[0]) if rows else 0
    if n < 2 or k < 2:
        return None
    grand = statistics.fmean(v for r in rows for v in r)
    rm = [statistics.fmean(r) for r in rows]
    cm = [statistics.fmean(r[j] for r in rows) for j in range(k)]
    ssr = k * sum((m - grand) ** 2 for m in rm)
    ssc = n * sum((m - grand) ** 2 for m in cm)
    sse = sum((rows[i][j] - rm[i] - cm[j] + grand) ** 2 for i in range(n) for j in range(k))
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (msc - mse) / n
    return (msr - mse) / denom if denom else None


def within_one(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs) if pairs else None


def raw(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def bootstrap_difference(units, a1, a2, machines, weights="linear", B=4000, seed=11):
    """CI on (human-human kappa) - (mean human-machine kappa), resampling PASSAGES.

    Paired: each bootstrap draw uses the same passages for both terms, so the
    passages' own difficulty cancels instead of adding noise to the contrast.
    """
    rng = random.Random(seed)
    def stat(sample):
        hh = _kappa([(a1[u], a2[u]) for u in sample], weights)
        hm = [_kappa([(h[u], m[u]) for u in sample], weights)
              for h in (a1, a2) for m in machines.values()]
        hm = [v for v in hm if v is not None]
        if hh is None or not hm:
            return None
        return hh - statistics.fmean(hm)
    point = stat(units)
    draws = []
    for _ in range(B):
        s = [rng.choice(units) for _ in units]
        v = stat(s)
        if v is not None:
            draws.append(v)
    draws.sort()
    if not draws:
        return point, None, None
    return point, draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]


# ---------------------------------------------------------------- loading

def load():
    key = json.loads((HERE / "passage_key.json").read_text())
    mach = json.loads((HERE / "machine_readings.json").read_text())
    humans = {}
    for f in sorted(glob.glob(str(HERE / "annotations" / "*.json"))):
        d = json.loads(Path(f).read_text())
        humans[d.get("label") or Path(f).stem] = d
    return key, mach, humans


def human_levels(doc, block=None, key=None, exclude_revealed=True):
    out = {}
    for code, r in (doc.get("responses") or {}).items():
        if block and key[code]["block"] != block:
            continue
        if exclude_revealed and r.get("saw_machine"):
            continue
        if r.get("is_soc") is False:
            out[code] = None
        elif isinstance(r.get("mediation_level"), int):
            out[code] = r["mediation_level"]
    return out


def machine_levels(mach, reader, codes):
    return {c: (mach[c]["readers"].get(reader) or {}).get("modal") for c in codes}


# ---------------------------------------------------------------- report

def report(block, key, mach, humans):
    codes = [c for c in key if key[c]["block"] == block]
    names = list(humans)
    hl = {n: human_levels(humans[n], block, key) for n in names}
    ml = {m: machine_levels(mach, m, codes) for m in MACHINES}
    ml = {m: v for m, v in ml.items() if any(x is not None for x in v.values())}

    title = "SECTION A — Time Passes" if block == "core" else "SECTION B — stratified cline sample"
    print(f"\n{'='*74}\n  {title}   ({len(codes)} passages)\n{'='*74}")

    graded = [c for c in codes
              if all(n in hl and hl[n].get(c) is not None for n in names)]
    print(f"  passages with a level from every human: {len(graded)}")
    if len(graded) < 5:
        print("  too few to report agreement yet.")
        return

    rows = [("pair", "kappa_w", "kappa", "exact", "within 1", "n")]
    allr = {**hl, **ml}
    for a, b in itertools.combinations(allr, 2):
        pairs = [(allr[a].get(c), allr[b].get(c)) for c in codes]
        kind = "H-H" if a in names and b in names else ("M-M" if a not in names and b not in names else "H-M")
        n = len([1 for x, y in pairs if x is not None and y is not None])
        rows.append((f"{kind}  {a} / {b}",
                     f"{_kappa(pairs,'linear'):.3f}" if n else "—",
                     f"{_kappa(pairs,None):.3f}" if n else "—",
                     f"{raw(pairs):.0%}" if n else "—",
                     f"{within_one(pairs):.0%}" if n else "—", str(n)))
    w = [max(len(r[i]) for r in rows) for i in range(6)]
    for i, r in enumerate(rows):
        print("   " + "  ".join(s.ljust(w[j]) if j == 0 else s.rjust(w[j]) for j, s in enumerate(r)))
        if i == 0:
            print("   " + "  ".join("-" * w[j] for j in range(6)))

    ratings = {c: {r: allr[r].get(c) for r in allr} for c in codes}
    hr = {c: {n: hl[n].get(c) for n in names} for c in codes}
    mr = {c: {m: ml[m].get(c) for m in ml} for c in codes}
    print(f"\n   Krippendorff alpha (ordinal)   all raters {_fmt(_alpha_ordinal(ratings))}"
          f"   humans {_fmt(_alpha_ordinal(hr))}   machines {_fmt(_alpha_ordinal(mr))}")
    mat = [[allr[r].get(c) for r in allr] for c in codes]
    print(f"   ICC(2,k) over all raters       {_fmt(_icc2k(mat))}")

    if len(names) == 2 and ml:
        a1, a2 = (hl[names[0]], hl[names[1]])
        pt, lo, hi = bootstrap_difference(graded, a1, a2, ml)
        print(f"\n   HEADLINE — (human-human) minus (mean human-machine), weighted kappa")
        print(f"     difference {pt:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   {'humans agree with each other MORE than machines agree with humans' if lo > 0 else ('no detectable gap — machines read like a third annotator' if hi > 0 else 'machines agree with humans MORE than the humans agree with each other')}")

    audit = collections.Counter()
    for n in names:
        for code, r in (humans[n].get("responses") or {}).items():
            if key[code]["block"] == block and (r.get("is_soc") is not None or r.get("mediation_level") is not None):
                audit[(r.get("order_mode", "?"), bool(r.get("saw_machine")))] += 1
    if audit:
        print("\n   audit — responses by order mode / machine panel seen first:")
        for (m, saw), n in sorted(audit.items()):
            print(f"     {m:10s} {'after reveal' if saw else 'blind      '}  {n}")


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def coupling(key, mach, humans):
    """Is mediation level a near-deterministic function of soc_type for humans
    as it is for the machine readers? If the humans decouple them, the coupling
    the machines show is a property of the models, not of the texts."""
    print(f"\n{'='*74}\n  soc_type -> mediation_level coupling\n{'='*74}")
    for name, doc in humans.items():
        tab = collections.defaultdict(collections.Counter)
        for code, r in (doc.get("responses") or {}).items():
            if r.get("soc_type") and isinstance(r.get("mediation_level"), int):
                tab[r["soc_type"]][r["mediation_level"]] += 1
        if not tab:
            continue
        det = statistics.fmean(max(c.values()) / sum(c.values()) for c in tab.values())
        print(f"   {name:14s} mean purity of level given type: {det:.1%}   ({len(tab)} types used)")
    for m in MACHINES:
        tab = collections.defaultdict(collections.Counter)
        for code in key:
            d = mach[code]["readers"].get(m)
            if d and d.get("modal_type") and d.get("modal") is not None:
                tab[d["modal_type"]][d["modal"]] += 1
        if tab:
            det = statistics.fmean(max(c.values()) / sum(c.values()) for c in tab.values())
            print(f"   {m:14s} mean purity of level given type: {det:.1%}   ({len(tab)} types used)")


def main():
    key, mach, humans = load()
    if not humans:
        print("No annotations yet. Run ingest_annotations.py once the readers have coded.")
        print(f"Machine readings are ready for {len(mach)} passages.")
        return
    print(f"Annotators: {', '.join(humans)}")
    for block in ("core", "cline"):
        report(block, key, mach, humans)
    coupling(key, mach, humans)


if __name__ == "__main__":
    main()
