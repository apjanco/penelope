"""Static (print / Google Docs) figure: within-novel distribution of mediation levels,
Woolf's ten novels in publication order. Fills the chapter's
"[figure for intra-novel variation]" slot.

Three panels: pooled judgments, then GPT-5.4 and Qwen3-Max alone, because the text's
Jacob's Room and The Years arguments depend on the readers disagreeing.

Uses the same loading and tabulation as technique_v_genre/recompute_chapter_stats.py
(per-judgment shares of is_soc passages, vol. suffixes folded into one title).

Writes chapter_viz/figures/static/fig_woolf_within_novel.{png,pdf,svg}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV = ROOT.parent / "technique_v_genre"
OUT = ROOT / "figures" / "static"

SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}
READERS = {"gpt-5.4", "qwen3-max", "qwen3.6-local", "jev"}
LEVELS = [0, 1, 2, 3]
LEVEL_NAME = ["psycho-narration", "narrated monologue", "quoted monologue",
              "autonomous monologue"]
# ordinal blue ramp, steps 250 / 400 / 550 / 700 (lightest still clears 2:1 on white)
LEVEL_COL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
LABEL_INK = ["#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#ffffff"

for f in font_manager.findSystemFonts():
    if "SourceSans3" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Source Sans 3", "DejaVu Sans"], "font.size": 8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def load():
    rows = []
    files = sorted((TV / "scored").glob("*.jsonl")) + sorted((TV / "scored_jev").glob("*.jsonl"))
    for f in files:
        if f.name in SKIP:
            continue
        for line in f.open():
            r = json.loads(line)
            if r.get("reader_id") not in READERS or r.get("error") or r.get("author") != "Woolf":
                continue
            rows.append({k: r.get(k) for k in ("passage_id", "title", "year", "reader_id",
                                                "is_soc", "mediation_level")})
    df = pd.DataFrame(rows)
    df["title"] = df["title"].str.replace(r"\s*\(vol\. \d\)", "", regex=True)
    df = df[(df.is_soc == True) & df.mediation_level.notna()].copy()  # noqa: E712
    df["mediation_level"] = df.mediation_level.astype(float)
    return df


def shares(g):
    v = g.mediation_level
    p = v.value_counts(normalize=True).reindex(LEVELS, fill_value=0) * 100
    return pd.Series({"n": len(v), "mean": v.mean(), **{f"L{k}": p[k] for k in LEVELS}})


def main():
    df = load()
    order = (df.groupby("title").year.min().sort_values().index.tolist())
    years = df.groupby("title").year.min()
    # "Pooled" stays GPT + Qwen (+local), the basis of the chapter's quoted percentages;
    # Jev is shown as its own panel rather than folded in.
    panels = [("Pooled", df[df.reader_id != "jev"]), ("GPT-5.4", df[df.reader_id == "gpt-5.4"]),
              ("Qwen3-Max", df[df.reader_id == "qwen3-max"]), ("Jev", df[df.reader_id == "jev"])]
    tabs = [(name, d.groupby("title").apply(shares, include_groups=False).reindex(order))
            for name, d in panels]

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 4.4), sharey=True,
                             gridspec_kw={"wspace": 0.17})
    H = 0.72
    for ax, (name, t) in zip(axes, tabs):
        for i, title in enumerate(order):
            left = 0.0
            for k in LEVELS:
                w = t.loc[title, f"L{k}"]
                ax.barh(i, w, left=left, height=H, color=LEVEL_COL[k],
                        edgecolor=SURFACE, linewidth=0.8)
                if w >= 10:
                    ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center",
                            fontsize=6.2, color=LABEL_INK[k])
                left += w
        ax.set_xlim(0, 100)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_title(name, loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=4)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"])
        ax.tick_params(axis="x", colors=MUTED, labelsize=6.8, length=2, width=0.5)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.spines["bottom"].set_linewidth(0.6)

    pooled = tabs[0][1]
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([""] * len(order))
    for i, title in enumerate(order):
        axes[0].text(-3, i - 0.12, title, ha="right", va="center", fontsize=8, color=INK,
                     transform=axes[0].transData)
        axes[0].text(-3, i + 0.27, f"{int(years[title])} · mean {pooled.loc[title, 'mean']:.2f}",
                     ha="right", va="center", fontsize=6.4, color=MUTED,
                     transform=axes[0].transData)

    handles = [Patch(color=LEVEL_COL[k], label=f"{k}  {LEVEL_NAME[k]}") for k in LEVELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.585, -0.005), handlelength=1.2, columnspacing=1.4,
               labelcolor=INK2, title="Mediation level (Cohn), most to least mediated",
               title_fontsize=7, alignment="center")
    fig.subplots_adjust(left=0.2, right=0.975, top=0.865, bottom=0.2)
    fig.text((0.2 + 0.975) / 2, 0.12, "% of passages", ha="center", fontsize=7, color=MUTED)
    fig.text(0.2, 0.985, "Share of passages at each mediation level, Woolf's novels in "
             "publication order.\nPooled = GPT-5.4 and Qwen3-Max together; the mean beneath each title is pooled.",
             fontsize=7, color=MUTED, va="top")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_woolf_within_novel.{ext}", dpi=300, facecolor=SURFACE)
    for name, t in tabs:
        print(f"\n{name}\n", t.round(2).to_string())


if __name__ == "__main__":
    main()
