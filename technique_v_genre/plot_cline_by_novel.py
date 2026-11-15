"""One point per novel on Cohn's mediation cline, colored by author.

Mean position per novel uses the same passage-first aggregation as the
ranking table in woolf_variance_decomposition.md: collapse each passage's
readings (both models, all runs) to a passage mean, then average passages
within a title.

Usage: python plot_cline_by_novel.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pilot_experiments_abc import load

ROOT = Path(__file__).resolve().parent
LEVEL_NAMES = {
    0: "psycho-\nnarration",
    1: "narrated\nmonologue / FID",
    2: "quoted\nmonologue",
    3: "autonomous\nmonologue",
}


def novel_means(df):
    soc = df[df["is_soc"] == True].dropna(subset=["mediation_level"]).copy()  # noqa: E712
    soc["mediation_level"] = soc["mediation_level"].astype(float)
    passage = (
        soc.groupby(["passage_id", "author", "title"])["mediation_level"]
        .mean()
        .reset_index()
    )
    return (
        passage.groupby(["author", "title"])["mediation_level"]
        .mean()
        .reset_index()
        .rename(columns={"mediation_level": "cline"})
    )


def main():
    means = novel_means(load())

    # Order authors by their mean cline position (quiet -> radical).
    author_order = (
        means.groupby("author")["cline"].mean().sort_values().index.tolist()
    )
    y_of = {a: i for i, a in enumerate(author_order)}
    cmap = plt.get_cmap("tab10")
    color_of = {a: cmap(i % 10) for i, a in enumerate(author_order)}

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.6, len(author_order) - 0.4)

    # Draw points + author lines first; collect labels to place afterwards so
    # widths can be measured against a fixed axis.
    labels = []  # (author, title, x, y_author, text_obj)
    for author in author_order:
        y = y_of[author]
        sub = means[means["author"] == author].sort_values("cline")
        ax.plot(
            [sub["cline"].min(), sub["cline"].max()], [y, y],
            color=color_of[author], linewidth=1.5, alpha=0.5, zorder=2,
        )
        for _, row in sub.iterrows():
            ax.scatter(
                row["cline"], y, s=90, color=color_of[author],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            txt = ax.text(
                row["cline"], y, row["title"], fontsize=7, color="#333333",
                ha="center", va="bottom", zorder=4,
            )
            labels.append((author, row["cline"], y, txt))

    # Measure label widths in data units, then greedily stack overlapping
    # labels into vertical lanes (with leader lines) so nearly-coincident
    # titles like Richardson's three volumes never sit on top of each other.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    LANE_STEP, LANE_BASE, PAD = 0.16, 0.08, 0.02
    by_author: dict = {}
    for author, x, y, txt in labels:
        bb = txt.get_window_extent(renderer=renderer)
        (dx0, _), (dx1, _) = inv.transform((bb.x0, 0)), inv.transform((bb.x1, 0))
        half = abs(dx1 - dx0) / 2 + PAD
        by_author.setdefault(y, []).append((x, half, txt))

    for y, items in by_author.items():
        lane_right: list[float] = []  # rightmost occupied x per lane
        for x, half, txt in sorted(items, key=lambda t: t[0]):
            left, right = x - half, x + half
            lane = next((i for i, r in enumerate(lane_right) if left > r), len(lane_right))
            if lane == len(lane_right):
                lane_right.append(right)
            else:
                lane_right[lane] = right
            label_y = y + LANE_BASE + lane * LANE_STEP
            txt.set_position((x, label_y))
            if lane > 0:  # leader line from point up to a stacked label
                ax.plot([x, x], [y, label_y], color="#bbbbbb",
                        linewidth=0.5, zorder=1)

    ax.set_yticks(range(len(author_order)))
    ax.set_yticklabels(author_order)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([LEVEL_NAMES[i] for i in range(4)], fontsize=8)
    ax.set_xlabel("mean position on Cohn's mediation cline (0 = most mediated, 3 = least)")
    ax.set_title("Mediation cline by novel, one point per title")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = ROOT / "cline_by_novel.png"
    fig.savefig(out, dpi=200)
    print(f"Written: {out}  ({len(means)} novels, {means['author'].nunique()} authors)")


if __name__ == "__main__":
    main()
