"""Static (print / Google Docs) version of Figure 5: novel-to-novel movement over a career.

Unlike the interactive draft, each author's time axis runs from their first work to
their last work in the data, so the panel shows that author's own fingerprint rather
than their position on a shared 1875-1941 axis. The y-scale (signed step) stays shared
across panels so step sizes remain comparable between authors.

Reads  chapter_viz/data/work_steps.csv
Writes chapter_viz/figures/static/fig5_career_steps.{png,pdf,svg}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "static"

READER, BIG, EMPH = "jev", 0.8, "Woolf"
ACC, DE = "#2a78d6", "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASELINE, SURFACE = "#c3c2b7", "#ffffff"

for f in font_manager.findSystemFonts():
    if "SourceSans3" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Source Sans 3", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": BASELINE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.5, "xtick.major.width": 0.6, "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def short(t, n=26):
    return t if len(t) <= n else t[: n - 1] + "…"


def main():
    df = pd.read_csv(ROOT / "data" / "work_steps.csv")
    df = df[df.reader == READER].copy()
    norm = df.step.abs().median()  # corpus norm: median |step| over all transitions
    mx = df.step.abs().max() * 1.28
    authors = (df.groupby("author").step.apply(lambda s: s.abs().max())
               .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(4, 2, figsize=(6.5, 8.4))
    axes = axes.ravel()
    for ax, a in zip(axes, authors):
        ss = df[df.author == a].reset_index(drop=True)  # already chronological
        acc = a == EMPH
        col = ACC if acc else DE
        y0, y1 = ss.from_year.iloc[0], ss.to_year.iloc[-1]
        pad = max(0.8, 0.05 * (y1 - y0))
        lo, hi = y0 - pad, y1 + pad
        bw = 0.04 * (hi - lo)

        # Dodge bars that share a publication year (e.g. two-volume James novels)
        counts = ss.to_year.value_counts()
        xs, idx = [], {}
        for yr in ss.to_year:
            j = idx.get(yr, 0)
            idx[yr] = j + 1
            xs.append(yr + (j - (counts[yr] - 1) / 2) * bw * 1.15)

        ax.set_xlim(lo, hi)
        ax.set_ylim(-mx, mx)
        for v in (BIG, -BIG):
            ax.axhline(v, color=BASELINE, lw=0.7, ls=(0, (3, 3)), zorder=1)
        for v in (norm, -norm):
            ax.axhline(v, color=MUTED, lw=0.7, ls=(0, (1, 2.5)), zorder=1)
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=2)

        if len(ss) >= 4:  # 3-step centred rolling mean of |step|; too few steps -> no envelope
            roll = ss.step.abs().rolling(3, center=True, min_periods=1).mean()
            ax.fill_between(xs, roll, -roll, color=col, alpha=0.17 if acc else 0.12,
                            lw=0, zorder=2)
            for s in (1, -1):
                ax.plot(xs, s * roll, color=col, lw=1.1, alpha=0.85 if acc else 0.6,
                        solid_joinstyle="round", zorder=3)

        for x, (_, s) in zip(xs, ss.iterrows()):
            big = abs(s.step) >= BIG
            ax.bar(x, s.step if abs(s.step) > 0.02 else (0.02 if s.step >= 0 else -0.02),
                   width=bw, color=col, alpha=1 if big else (0.6 if acc else 0.45),
                   edgecolor=SURFACE, linewidth=0.6, zorder=4)
            if big:
                off = 0.07 * mx
                ax.text(x, s.step + (off if s.step > 0 else -off), short(s.to_title),
                        ha="center", va="bottom" if s.step > 0 else "top",
                        fontsize=6.8, color=INK2, zorder=5)

        # first work: no incoming step, so mark where the career starts
        ax.plot(y0, 0, marker="o", ms=4.2, mfc=SURFACE, mec=col, mew=1.1, zorder=5,
                clip_on=False)

        # ticks: first and last year always, a few round years between
        span = y1 - y0
        step = 2 if span <= 10 else 5 if span <= 25 else 10
        inner = [t for t in range(y0 - y0 % step + step, y1, step)
                 if y0 + 0.12 * span < t < y1 - 0.12 * span]
        ax.set_xticks([y0, *inner, y1])
        ax.set_xticklabels([str(int(t)) for t in [y0, *inner, y1]])
        ax.set_yticks([-BIG, 0, BIG])
        ax.set_yticklabels(["−0.8", "0", "+0.8"])
        ax.tick_params(axis="y", length=0, pad=2)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

        over = int((ss.step.abs() > norm).sum())
        ax.set_title(f"{a}", loc="left", fontsize=9.5, fontweight="bold", color=INK,
                     pad=17)
        ax.text(0, 1.035, f"Max step "
                f"{ss.step.abs().max():.2f} · {over}/{len(ss)} steps above norm",
                transform=ax.transAxes, fontsize=6.8, color=MUTED, va="bottom")

    # legend in the empty eighth cell
    lax = axes[len(authors)]
    lax.axis("off")
    handles = [
        Patch(color=ACC, label="Woolf"),
        Patch(color=DE, label="Other authors"),
        Patch(color=DE, alpha=0.45, label="step < 0.8 (lighter)"),
        Patch(color=DE, alpha=0.15, label="3-step rolling mean of |step|"),
        Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2.5)),
               label=f"corpus norm, median |step| = {norm:.3f}"),
        Line2D([], [], color=BASELINE, lw=0.9, ls=(0, (3, 3)), label="±0.8, \"big\" step"),
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=SURFACE, mec=DE, mew=1.1,
               label="first work in the data"),
    ]
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2,
               handlelength=2.2, labelcolor=INK2, borderaxespad=0)
    lax.text(0, -0.2, "Bars sit at the later novel's year; up = toward less-mediated\n"
             "narration. Each panel spans that author's first to last work;\n"
             "the vertical scale is shared. Reader: jev (all 54 works).",
             transform=lax.transAxes, fontsize=6.6, color=MUTED, va="top")

    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig5_career_steps.{ext}", dpi=300, facecolor=SURFACE)
    print("wrote", OUT / "fig5_career_steps.png", "| norm", round(norm, 4))


if __name__ == "__main__":
    main()
