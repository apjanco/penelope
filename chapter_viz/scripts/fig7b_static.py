"""Static (print / Google Docs) version of Figure 7b: each author's works joined in
publication order on the mean-mediation x within-novel-spread plane, against the faint
full 54-work cloud. Reader: jev, the only reader covering all 54 works.

Labels the first and last work and any work reached by a step of >= 0.8 in mean level
(the same "big step" threshold as Figure 5), so the two figures name the same novels.

Reads  chapter_viz/data/figure_payload.json  (scatter rows)
Writes chapter_viz/figures/static/fig7b_career_paths.{png,pdf,svg}
       chapter_viz/figures/static/fig7b_panels/fig7b_<author>.png  (one panel each)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "static"

READER, BIG, EMPH = "jev", 0.8, "Woolf"
ACC, OTHER = "#2a78d6", "#0b0b0b"
INK, INK2, MUTED, CLOUD = "#0b0b0b", "#52514e", "#898781", "#898781"
GRID, SURFACE = "#e1e0d9", "#ffffff"

for f in font_manager.findSystemFonts():
    if "SourceSans3" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Source Sans 3", "DejaVu Sans"], "font.size": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "svg.fonttype": "none", "pdf.fonttype": 42,
})


# Hand-placed labels where the default (beside the point) collides with a path or
# another label: (dx, dy) in points, ha, va. A leader line is drawn when offset.
LABEL_POS = {
    ("Woolf", "The Voyage Out"): (12, 24, "center", "bottom"),
    ("Woolf", "Flush"): (-4, -26, "center", "top"),
    ("Woolf", "Between the Acts"): (-22, 36, "center", "bottom"),
    ("Faulkner", "Sanctuary"): (-14, 20, "center", "bottom"),
    ("Faulkner", "Soldiers' Pay"): (-10, -22, "center", "top"),
    ("Faulkner", "Absalom, Absalom!"): (6, -1, "left", "center"),
    ("Sinclair", "Life and Death of Harriett Frean"): (4, -14, "left", "top"),
    ("Sinclair", "Audrey Craven"): (14, 18, "center", "bottom"),
    ("Richardson", "Pointed Roofs"): (-18, 22, "center", "bottom"),
    ("Richardson", "Dimple Hill"): (6, -6, "left", "top"),
    ("Joyce", "Dubliners"): (6, -5, "left", "top"),
    ("Ford", "The Last Post"): (-6, -6, "right", "top"),
}
# Same order as Figure 5 (largest single step first), so the two figures line up.
ORDER = ["Woolf", "Faulkner", "Joyce", "Sinclair", "Ford", "James", "Richardson"]


def short(t, n=24):
    return t if len(t) <= n else t[: n - 1] + "…"


XLABEL = "Mean mediation level of the novel (0–3)\n← more mediated          less mediated →"
XLABEL_SHORT = "Mean mediation level (0–3):  ← more mediated · less mediated →"
YLABEL = "Spread within the novel\n(sd of passage levels)"


def draw(ax, a, pts, lims, show_xlabel=True, show_ylabel=True, standalone=False):
    (x0, x1), (y0, y1) = lims
    ws = sorted((p for p in pts if p["a"] == a), key=lambda p: p["y"])
    acc = a == EMPH
    col = ACC if acc else OTHER

    for v in (0, 1, 2):
        ax.axvline(v, color=GRID, lw=0.6, zorder=0)
    for v in (0.6, 0.9, 1.2):
        ax.axhline(v, color=GRID, lw=0.6, zorder=0)
    ax.scatter([p["m"] for p in pts], [p["sd"] for p in pts], s=6, color=CLOUD,
               alpha=0.22, lw=0, zorder=1)

    xs, ys = [p["m"] for p in ws], [p["sd"] for p in ws]
    ax.plot(xs, ys, color=col, lw=1.3, alpha=0.95 if acc else 0.5,
            solid_joinstyle="round", zorder=2)
    # small arrowhead at each segment's midpoint shows direction of travel
    for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if abs(xb - xa) < 0.06 and abs(yb - ya) < 0.04:
            continue  # too short to carry an arrowhead legibly
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.annotate("", xy=(mx + (xb - xa) * 0.02, my + (yb - ya) * 0.02), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                    color=col, alpha=0.95 if acc else 0.6, lw=0),
                    zorder=2)
    for i, p in enumerate(ws):
        first, last = i == 0, i == len(ws) - 1
        ax.scatter(p["m"], p["sd"], s=26 if first or last else 13, zorder=3,
                   facecolor=SURFACE if first else col,
                   edgecolor=col if first else SURFACE,
                   linewidth=1.2 if first else 0.6,
                   alpha=1 if (first or last or acc) else 0.75)

    # label first, last, and works arrived at by a big step
    for i, p in enumerate(ws):
        big = i > 0 and abs(p["m"] - ws[i - 1]["m"]) >= BIG
        if not (i == 0 or i == len(ws) - 1 or big):
            continue
        right = p["m"] < (x0 + x1) / 2
        dx, dy, ha, va = LABEL_POS.get(
            (a, p["t"]), (5 if right else -5, 4, "left" if right else "right", "bottom"))
        leader = (a, p["t"]) in LABEL_POS and abs(dx) + abs(dy) > 18
        ax.annotate(f"{short(p['t'])} ({p['y']})", (p["m"], p["sd"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=6.3, color=INK2, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                    shrinkA=1, shrinkB=3) if leader else None)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0.6, 0.9, 1.2])
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(a, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=12)
    sub = f"{ws[0]['y']}–{ws[-1]['y']} · {len(ws)} works"
    if standalone:
        sub += ", one dot per novel, joined in publication order"
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.8, color=MUTED, va="bottom")
    if show_xlabel:
        ax.set_xlabel(XLABEL if standalone else XLABEL_SHORT, fontsize=6.8, color=INK2, labelpad=3, linespacing=1.4)
    if show_ylabel:
        ax.set_ylabel(YLABEL, fontsize=6.8, color=INK2, labelpad=3, linespacing=1.3)
    if standalone:  # the combined figure carries this in its legend cell
        key = [
            Line2D([], [], ls="", marker="o", ms=5, mfc=SURFACE, mec=col, mew=1.2,
                   label="first novel in the data"),
            Line2D([], [], ls="", marker="o", ms=5, mfc=col, mec=SURFACE, label="last novel"),
            Line2D([], [], ls="", marker="o", ms=3.5, mfc=CLOUD, mec="none", alpha=0.45,
                   label="all 54 works, 7 authors"),
        ]
        ax.legend(handles=key, loc="lower right", frameon=False, fontsize=6.2,
                  labelcolor=INK2, handletextpad=0.3, borderaxespad=0.2,
                  labelspacing=0.3)


def legend(ax):
    ax.axis("off")
    handles = [
        Line2D([], [], color=ACC, lw=1.3, marker="o", ms=4, mec=SURFACE, label="Woolf"),
        Line2D([], [], color=OTHER, lw=1.3, alpha=0.5, marker="o", ms=4, mec=SURFACE,
               label="Other authors"),
        Line2D([], [], ls="", marker="o", ms=5.5, mfc=SURFACE, mec=OTHER, mew=1.2,
               label="first work in the data"),
        Line2D([], [], ls="", marker="o", ms=5.5, mfc=OTHER, mec=SURFACE,
               label="last work"),
        Line2D([], [], ls="", marker="o", ms=3.5, mfc=CLOUD, mec="none", alpha=0.4,
               label="all 54 works, for context"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2,
              labelcolor=INK2, borderaxespad=0, handlelength=2.2)
    ax.text(0, 0.22, "One dot per novel; each author's novels are joined\n"
            "in publication order, arrows showing direction.\n"
            "Across: how mediated the novel is on average.\n"
            "Up: how much its passages vary in level.\n"
            "Labelled: first and last work, and any novel\n"
            "reached by a step of 0.8 or more. Axes shared.\n"
            "Reader: jev (all 54 works).",
            transform=ax.transAxes, fontsize=6.6, color=MUTED, va="top")


def main():
    pts = [p for p in json.load(open(ROOT / "data" / "figure_payload.json"))["scatter"]
           if p["r"] == READER]
    lims = ((0, max(p["m"] for p in pts) + 0.15),
            (min(p["sd"] for p in pts) - 0.05, max(p["sd"] for p in pts) + 0.05))
    authors = [a for a in ORDER if any(p["a"] == a for p in pts)]
    assert len(authors) == len({p["a"] for p in pts}), "author missing from ORDER"

    fig, axes = plt.subplots(4, 2, figsize=(6.5, 9.0))
    axes = axes.ravel()
    for i, (ax, a) in enumerate(zip(axes, authors)):
        draw(ax, a, pts, lims, show_xlabel=True, show_ylabel=i % 2 == 0)
    legend(axes[len(authors)])
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig7b_career_paths.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)

    pdir = OUT / "fig7b_panels"
    pdir.mkdir(exist_ok=True)
    for a in authors:
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        draw(ax, a, pts, lims, standalone=True)
        fig.tight_layout()
        fig.savefig(pdir / f"fig7b_{a.lower()}.png", dpi=300, facecolor=SURFACE)
        plt.close(fig)
    print("wrote", OUT / "fig7b_career_paths.png", "and", len(authors), "panels")


if __name__ == "__main__":
    main()
