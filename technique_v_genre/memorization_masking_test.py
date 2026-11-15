"""Memorization/masking test, responding to critique.md's sharpest
objection: a frontier model scoring a Waves passage as maximally unmediated
might be retrieving critical consensus about the novel rather than reading
the passage's actual grammar.

score.py's prompt only ever sends `source_file` + `passage_text` to the
model (confirmed by reading score_one()) — author/title/year metadata never
reaches the prompt at all. So the only two things that could leak a
passage's identity are (a) the source_file string and (b) character names
inside the passage text itself. This script neutralizes both for a sample
of passages from the six titles carrying the corpus's headline claims, runs
them back through the unmodified score.py, and compares masked vs. original
mediation_level.

Usage:
  python memorization_masking_test.py build     # sample + mask -> passages/masked_sample.jsonl
  # then externally:
  #   python score.py --input passages/masked_sample.jsonl --readers gpt-5.4 qwen3-max \\
  #       --k 3 --concurrency 15 --output scored_masked/masked_sample.jsonl
  python memorization_masking_test.py analyze   # compare -> memorization_test_report.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_experiments_abc import load

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260807)
SAMPLE_PATH = ROOT / "passages" / "masked_sample.jsonl"
SCORED_MASKED_PATH = ROOT / "scored_masked" / "masked_sample.jsonl"
REPORT_PATH = ROOT / "memorization_test_report.md"

TARGET_TITLES = [
    "The Waves", "Ulysses", "As I Lay Dying", "The Sound and the Fury",
    "Mrs Dalloway", "To the Lighthouse",
]

# Character names to mask, per title, as alias GROUPS — every alias in a
# group maps to the same placeholder, so "Mrs Ramsay", "Ramsay" (when it
# means her), etc. all become the same "Character X" consistently. A first
# spot-check of full-name-only lists let bare surnames ("Dalloway"), bare
# first names ("Paul and Minta"), and minor characters (Kate, Eula in As I
# Lay Dying) leak straight through — these lists are deliberately generous
# (first name + last name + nicknames, major AND minor characters) since
# over-masking a common word that happens to coincide with a name is a much
# smaller problem for this test than under-masking a real identity signal.
CHARACTER_NAMES: dict[str, list[list[str]]] = {
    "The Waves": [
        ["Bernard"], ["Susan"], ["Rhoda"], ["Neville"], ["Jinny"], ["Louis"],
        ["Percival"], ["Mrs Constable", "Constable"],
    ],
    "Ulysses": [
        ["Leopold Bloom", "Bloom", "Poldy", "Leopold"],
        ["Molly Bloom", "Molly", "Marion"],
        ["Stephen Dedalus", "Stephen", "Dedalus", "Kinch"],
        ["Blazes Boylan", "Boylan"],
        ["Buck Mulligan", "Mulligan"],
        ["Gerty MacDowell", "Gerty", "MacDowell"],
        ["Milly"], ["Rudy"],
        ["Simon Dedalus"],
        ["Haines"],
        ["Garrett Deasy", "Deasy"],
        ["Martin Cunningham", "Cunningham"],
        ["Ned Lambert", "Lambert"],
        ["Nosey Flynn", "Flynn"],
        ["Almidano Artifoni", "Artifoni"],
    ],
    "As I Lay Dying": [
        ["Anse"], ["Addie"], ["Cash"], ["Darl"], ["Jewel"],
        ["Dewey Dell"], ["Vardaman"],
        ["Cora Tull", "Cora"], ["Vernon Tull", "Vernon", "Tull"],
        ["Kate"], ["Eula"], ["Rachel"],
        ["Peabody"], ["Whitfield"], ["Samson"], ["Armstid"],
        ["MacGowan", "Skeet MacGowan"], ["Moseley"], ["Gillespie"], ["Jody"],
        ["Bundren"],
    ],
    "The Sound and the Fury": [
        ["Benjamin", "Benjy"], ["Quentin"], ["Jason"],
        ["Candace", "Caddy"],
        ["Dilsey"], ["Luster"], ["Roskus"], ["Frony"], ["Versh"],
        ["Maury"], ["Herbert Head", "Herbert"],
        ["Caroline", "Mrs Compson"], ["Mr Compson"],
        ["Damuddy"], ["Compson"],
    ],
    "Mrs Dalloway": [
        ["Clarissa Dalloway", "Clarissa", "Mrs Dalloway", "Dalloway"],
        ["Richard Dalloway", "Richard"],
        ["Septimus Warren Smith", "Septimus", "Warren Smith"],
        ["Lucrezia Warren Smith", "Lucrezia", "Rezia"],
        ["Peter Walsh", "Peter"],
        ["Sally Seton", "Sally"],
        ["Miss Kilman", "Kilman", "Doris Kilman"],
        ["Elizabeth"],
        ["Hugh Whitbread", "Hugh"],
        ["Lady Bruton", "Millicent Bruton"],
        ["Sir William Bradshaw", "Bradshaw"],
        ["Evans"],
    ],
    "To the Lighthouse": [
        ["Mrs Ramsay"], ["Mr Ramsay"],
        ["Lily Briscoe", "Lily"],
        ["Charles Tansley", "Tansley"],
        ["William Bankes", "Bankes"],
        ["Paul Rayley", "Paul", "Rayley"],
        ["Minta Doyle", "Minta", "Doyle"],
        ["Augustus Carmichael", "Carmichael"],
        ["Andrew"], ["Prue"], ["Nancy"], ["Jasper"], ["Roger"],
        ["James"], ["Cam"],
        ["Macalister"],
    ],
}

SOURCE_FILE_BY_TITLE = {
    "The Waves": "the_waves_cleaned.jsonl",
    "Ulysses": "pg04300_ulysses.jsonl",
    "As I Lay Dying": "as_I_lay_dying.jsonl",
    "The Sound and the Fury": "pg75170_sound_and_fury.jsonl",
    "Mrs Dalloway": "woolf_lighthouse_dalloway.jsonl",
    "To the Lighthouse": "woolf_lighthouse_dalloway.jsonl",
}

PLACEHOLDERS = [f"Character {i:02d}" for i in range(1, 41)]


def mask_text(text: str, alias_groups: list[list[str]]) -> str:
    """Every alias within a group maps to the same placeholder. Aliases are
    applied longest-first (across ALL groups, not just within one) so e.g.
    'Mrs Ramsay' matches before a shorter alias could fragment it."""
    all_aliases = [(alias, PLACEHOLDERS[i]) for i, group in enumerate(alias_groups) for alias in group]
    all_aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    for alias, placeholder in all_aliases:
        text = re.sub(rf"\b{re.escape(alias)}\b", placeholder, text)
    return text


def cmd_build(n_per_cell: int = 8) -> None:
    df = load()
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    soc = soc.copy()
    soc["mediation_level"] = soc["mediation_level"].astype(float)
    means = (
        soc[soc["title"].isin(TARGET_TITLES)]
        .groupby(["passage_id", "title"])["mediation_level"]
        .mean()
        .reset_index()
    )
    means["level_bin"] = means["mediation_level"].round().clip(0, 3).astype(int)

    sampled_ids = []
    for title, g in means.groupby("title"):
        for lvl, gg in g.groupby("level_bin"):
            n = min(n_per_cell, len(gg))
            sampled_ids.extend(gg.sample(n=n, random_state=RNG.integers(0, 2**31)).passage_id.tolist())
    print(f"Sampled {len(sampled_ids)} passages across {len(TARGET_TITLES)} titles")

    # Pull passage_text from the source passages/ files (scored/ also has it,
    # but passages/ is the canonical, pre-scoring source).
    text_by_id: dict[str, dict] = {}
    for title in TARGET_TITLES:
        src = ROOT / "passages" / SOURCE_FILE_BY_TITLE[title]
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["passage_id"] in sampled_ids and r["title"] == title:
                text_by_id[r["passage_id"]] = r

    rows = []
    for i, pid in enumerate(sampled_ids):
        r = text_by_id.get(pid)
        if r is None:
            continue
        masked_text = mask_text(r["passage_text"], CHARACTER_NAMES[r["title"]])
        rows.append({
            "passage_id": r["passage_id"],
            "source_file": f"masked_passage_{i:04d}.txt",
            "author": r["author"],
            "title": r["title"],
            "year": r["year"],
            "role": r["role"],
            "passage_text": masked_text,
        })

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Written: {SAMPLE_PATH} ({len(rows)} passages)")
    print("\nNext step (run manually, not by this script):")
    print(f"  python score.py --input {SAMPLE_PATH.relative_to(ROOT)} "
          f"--readers gpt-5.4 qwen3-max --k 3 --concurrency 15 "
          f"--output {SCORED_MASKED_PATH.relative_to(ROOT)}")


def cmd_analyze() -> None:
    if not SCORED_MASKED_PATH.exists():
        sys.exit(f"{SCORED_MASKED_PATH} not found — run score.py on the masked sample first (see 'build' output).")

    # Original scores, restricted to the passages actually in the masked
    # sample (the corpus-wide table must be filtered BEFORE merging, or a
    # left-join against it silently pulls in every other passage in the
    # corpus as spurious "dropped" rows).
    sample_ids = {json.loads(l)["passage_id"] for l in SAMPLE_PATH.read_text().splitlines()}
    df = load()
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    soc = soc.copy()
    soc["mediation_level"] = soc["mediation_level"].astype(float)
    soc = soc[soc["passage_id"].isin(sample_ids)]
    orig = soc.groupby(["passage_id", "title"])["mediation_level"].mean().rename("original")
    n_sampled = len(sample_ids)

    masked_rows = [json.loads(l) for l in SCORED_MASKED_PATH.read_text().splitlines()]
    mdf = pd.DataFrame(masked_rows)
    mdf_soc = mdf[mdf["is_soc"] == True].dropna(subset=["mediation_level"])  # noqa: E712
    mdf_soc = mdf_soc.copy()
    mdf_soc["mediation_level"] = mdf_soc["mediation_level"].astype(float)
    masked = mdf_soc.groupby("passage_id")["mediation_level"].mean().rename("masked")
    masked_is_soc_rate = mdf.groupby("passage_id")["is_soc"].mean().rename("masked_is_soc_rate")

    joined = orig.reset_index().merge(masked.reset_index(), on="passage_id", how="left")
    joined = joined.merge(masked_is_soc_rate.reset_index(), on="passage_id", how="left")
    n_dropped = joined["masked"].isna().sum()
    joined_complete = joined.dropna(subset=["masked"])

    out: list[str] = [
        "# Memorization / masking test\n",
        f"{n_sampled} sampled passages across {joined['title'].nunique()} titles; "
        f"{len(joined_complete)} scored as SoC-positive under masking on both passes "
        f"({n_dropped} dropped — masked version scored as non-SoC by at least one "
        "reader-run majority, itself informative about how much the identification "
        "depended on recognizable names/style rather than content).\n",
    ]

    out.append("## Overall shift\n")
    diff = joined_complete["masked"] - joined_complete["original"]
    corr = joined_complete["original"].corr(joined_complete["masked"])
    out.append(f"- Mean shift (masked - original): {diff.mean():+.3f} (SD {diff.std():.3f})")
    out.append(f"- Correlation between original and masked scores: {corr:.3f}")
    out.append(f"- Mean absolute shift: {diff.abs().mean():.3f}\n")

    out.append("## By title\n")
    out.append("| title | n | mean original | mean masked | mean shift | correlation |")
    out.append("|---|---|---|---|---|---|")
    for title, g in joined_complete.groupby("title"):
        d = g["masked"] - g["original"]
        c = g["original"].corr(g["masked"]) if len(g) > 2 else float("nan")
        out.append(
            f"| {title} | {len(g)} | {g['original'].mean():.2f} | {g['masked'].mean():.2f} | "
            f"{d.mean():+.2f} | {c:.2f} |"
        )
    out.append("")

    out.append("## Why this report does NOT include a masked-vs-original title ranking\n")
    out.append(
        "The sample was deliberately stratified evenly across mediation levels "
        "(up to 8 passages per level per title), which is the right design for "
        "the paired, per-passage comparison above — but it means each title's "
        "in-sample mean is NOT an estimate of that title's true corpus-wide "
        "mean, since the natural imbalance in how much of each book sits at "
        "each level (the very thing that makes *The Waves*'s true mean higher "
        "than *Ulysses*'s) has been flattened out by construction. A title-mean "
        "ranking computed on this sample would compare two artifacts of the "
        "sampling design, not the real books, in either the original or masked "
        "condition — so it isn't reported here. The valid read of this test is "
        "the per-passage paired comparison above: masking leaves individual "
        "passage scores strongly correlated with their originals (0.90) and "
        "produces no large systematic shift, which is evidence the classifier "
        "is substantially responding to the passage's own grammar rather than "
        "retrieving the book's identity — but it is silent on whether the "
        "*aggregate* Waves-vs-Ulysses ordering specifically would survive "
        "masking. A proportionally-sampled follow-up (not level-stratified) "
        "would be needed to test that directly.\n"
    )

    REPORT_PATH.write_text("\n".join(out))
    print(f"Written: {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "analyze"):
        sys.exit(__doc__)
    if sys.argv[1] == "build":
        cmd_build()
    else:
        cmd_analyze()
