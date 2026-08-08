# CHR 2027 lightning talk — LaTeX submission

This directory is the official ACH LaTeX template
(`https://anthology.ach.org/resources/ach-latex-template.zip`), unmodified
except for `paper.tex`, `bibliography.bib`, and the addition of `silence.sty`
(see below). Do not edit `anthology-ch.cls`.

- `paper.tex` — the lightning talk content (title, abstract, five short
  sections, ~750 words), matching `CHR2027_lightning_talk.md` at the project
  root.
- `bibliography.bib` — only the four sources the talk cites. Two entries
  (Long & So; Zhang) are marked `TODO` — the exact venue/year/pages could not
  be confirmed from anything in the project, so verify them against your
  reference manager before submitting.
- `silence.sty` — the template's class file requires this package; it isn't
  in the local TeX Live install, so a copy from CTAN is bundled here rather
  than installed system-wide.

## Compiling

The class file also requires `biblatex` + the `biber` backend, which are
**not installed** in this environment's TeX Live (a minimal Fedora-packaged
install; `tlmgr` refuses to modify it, even in user mode, without an
`ignore-warning` override — not something to force on a shared system
without asking first). Two options:

1. **Overleaf (recommended, and what CHR's CFP itself suggests):** open
   `https://www.overleaf.com/latex/templates/ach-proceedings-paper-template/znmzmfngrwpd`
   as a template, then replace its `paper.tex` and `bibliography.bib` with
   the versions in this folder. Overleaf's `biblatex`/`biber` toolchain is
   preinstalled, so it will compile immediately.
2. **A machine/cluster with a full TeX Live:** run `make` (uses the provided
   `Makefile`: two `xelatex` passes, `biber`, two more `xelatex` passes).
   Confirm `biber` and `biblatex` are present first (`tlmgr info biblatex`).

## Before submitting

- Fill in real author name(s), ORCID(s), and affiliation(s) in `paper.tex`
  (currently placeholders — safe to leave populated pre-submission since the
  class auto-renders "Anonymous Submission" in non-`final` mode).
- Resolve the two `TODO` bibliography entries.
- Confirm word count is ≤750 words (excluding references), per the CFP.
- Add the `final` class option only for the camera-ready version, after
  acceptance.
