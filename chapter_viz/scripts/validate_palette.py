"""Port of the dataviz skill's validate_palette.js — there is no node runtime
on Della, and the skill is explicit that CVD safety must be computed, not
eyeballed. Same OKLab math, same Machado CVD matrices, same thresholds.

Usage: python scripts/validate_palette.py "#2a78d6,#898781" --mode light
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h): h = h.strip().lstrip("#"); return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
def s2lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
def lin(h): return [s2lin(c) for c in hex2srgb(h)]
def rel_lum(h): r, g, b = lin(h); return 0.2126*r + 0.7152*g + 0.0722*b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h); M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0]*r + M[i][1]*g + M[i][2]*b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface")
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args()

    pal = [c if c.startswith("#") else "#"+c for c in
           (x.strip() for x in a.palette.split(",")) if c]
    surf = a.surface or SURFACE[a.mode]
    lo, hi = BAND[a.mode]
    ok = True
    print(f"  palette={pal}  mode={a.mode}  surface={surf}  pairs={a.pairs}")

    off = [(c, round(oklch(c)[0], 3)) for c in pal if not (lo <= oklch(c)[0] <= hi)]
    print(f"  [{'FAIL' if off else 'PASS'}] lightness band {lo}-{hi}" + (f"  offband={off}" if off else ""))
    ok &= not off

    lowc = [(c, round(oklch(c)[1], 3)) for c in pal if oklch(c)[1] < CHROMA_FLOOR]
    # A de-emphasis gray is intentionally below the chroma floor; flag, don't fail.
    print(f"  [{'NOTE' if lowc else 'PASS'}] chroma floor {CHROMA_FLOOR}" + (f"  low={lowc} (intentional for a neutral/gray slot)" if lowc else ""))

    idx = list(range(len(pal)))
    pairs = list(zip(idx, idx[1:])) if a.pairs == "adjacent" else list(itertools.combinations(idx, 2))
    if pairs:
        worst_cvd = min(min(delta_e(pal[i], pal[j], k) for k in ("protan", "deutan"))
                        for i, j in pairs)
        tri = min(delta_e(pal[i], pal[j], "tritan") for i, j in pairs)
        worst_norm = min(delta_e(pal[i], pal[j]) for i, j in pairs)
        v = "PASS" if worst_cvd >= CVD_TARGET else ("WARN" if worst_cvd >= CVD_FLOOR else "FAIL")
        print(f"  [{v}] worst protan/deutan dE = {worst_cvd:.1f} (target >={CVD_TARGET}, floor {CVD_FLOOR})")
        ok &= worst_cvd >= CVD_FLOOR
        print(f"  [{'PASS' if tri >= CVD_FLOOR else 'WARN'}] worst tritan dE = {tri:.1f}")
        v = "PASS" if worst_norm >= NORMAL_FLOOR else "FAIL"
        print(f"  [{v}] worst normal-vision dE = {worst_norm:.1f} (hard floor >={NORMAL_FLOOR})")
        ok &= worst_norm >= NORMAL_FLOOR

    low = [(c, round(contrast(c, surf), 2)) for c in pal if contrast(c, surf) < CONTRAST_MIN]
    print(f"  [{'WARN' if low else 'PASS'}] contrast vs surface >={CONTRAST_MIN}" + (f"  low={low} -> needs visible labels or table view" if low else ""))

    print(f"  => {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
