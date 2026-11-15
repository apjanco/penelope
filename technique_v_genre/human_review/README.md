# Manual review — human readers against every machine reader

## What this measures

Every mediation figure in the chapter rests on machine readers. This folder
adds human readings of the same passages, under the same rubric, so the
machine numbers can be compared against something outside the models.

The headline test is not "the humans agree at kappa = x". At 33 passages that
figure carries a 95% interval about 0.36 wide. The claim worth making is
comparative, and paired:

> the machine readers agree with a human about as well as the two humans
> agree with each other

Both terms are computed on the same passages, so passage difficulty cancels,
and bootstrapping the *difference* gives a much tighter interval than either
kappa alone. `analyze_agreement.py` reports that difference with its CI as the
headline, the individual kappas as context.

**Report it as a bound, not as a hypothesis test.** Simulation on the measured
marginals puts power to detect a true gap of 0.10 kappa at 22% with two
annotators and 30% with three, over 63 passages. A design that cannot find a
gap must not report "no significant difference" as if it had looked. What it
*can* support is an equivalence bound — how large an apparent gap arises by
chance when there is really none:

| passages | annotators | claimable bound |
|---:|---:|---|
| 63 | 2 | gap no larger than 0.11 kappa |
| 63 | 3 | gap no larger than 0.08 kappa |
| 63 | 4 | gap no larger than 0.07 kappa |

Two to three is the large step; three to four buys 0.01. Three is the
recommendation, and the stronger reason is not precision: with exactly two
annotators, human-human agreement *is* the ceiling by definition, and there is
no way to detect an idiosyncratic annotator. That is the same tautology
`../calibration/evaluate_reader.py` already flags on the machine side, where
gpt-5.4 and qwen3-max define the consensus they are scored against. A third
reader breaks it, and makes a majority-vote gold label possible.

## The passage set — 63, in two blocks

**Section A — "Time Passes" (33).** The whole section, in text order. Every
machine reader has been run on it at k=10. Deliberately not a random sample:
it is the crux section, and the earlier k=10 analysis
(`../time_passes_findings.md`) found the typology strains there in a narrow,
specific way. Agreement measured here describes a hard case.

**Section B — stratified cline sample (30).** From the existing calibration
set, 8/8/7/7 across Cohn's four levels, at most two passages per author per
cell. Section A collapses to a near-binary problem — 57% L0, 40% L1, two L2s,
no L3 — so agreement there alone says nothing about the top of the cline.

The two blocks report separately and are never pooled into one headline.

## Blinding

Passages are served under opaque codes (P01–P63). The code to passage_id
mapping is in `passage_key.json`, which stays in the repo and is not built
into the page: an annotator who can see that a passage comes from *Ulysses*
has a prior on it before reading a word.

Machine readings are withheld until a passage has been coded, then revealed.
Each response records the order mode it was made in and whether the machine
panel had been opened for that passage first; `analyze_agreement.py` excludes
post-reveal responses from the primary figures.

Annotators are blind to each other because each works in their own browser and
exports their own file. Nothing is shared until `ingest_annotations.py` is run.

## Order modes

- **Reading order** (default) — Section A start to end, as the section runs.
- **Most contested** — sorted by contention, most contested first.
- **Shuffled** — seeded from the annotator's name, so fatigue does not fall on
  the same passages for every reader.

Contention is the Simpson index of the average per-reader distribution over
mediation levels: 0 when every qualifying reader on every run gives the same
level, rising both when readers disagree with each other and when one reader
is unstable across its own runs. Each reader gets one vote regardless of its
k. Only readers that passed the calibration gate count toward it — llama put
90 of 90 "Time Passes" passages at level 1 and would flatten the score toward
whatever it always says. It still appears in the reveal panel, labelled,
because a reader failing that way is worth seeing.

Sorting by contention makes the order itself a clue about machine behaviour.
That is intended — the second goal is to understand the machine annotations —
but it is recorded, and those responses can be analysed apart from the
reading-order pass.

## The instrument

`annotate.html` is a single self-contained static page: no server, no network,
no build step at view time. The passages, the machine readings and a build
stamp are injected into the file by `build_page.py`. That stamp is the
sha256 of each input plus the build time, it travels in every export, and
`ingest_annotations.py` refuses to pool a file coded against a different
passage set.

**Opening it.** Firefox will run it straight from disk. Chrome and Safari deny
local storage to `file://` pages, so a reload would lose the session — the
page detects this and says so. To avoid it entirely:

```
cd technique_v_genre/human_review
python -m http.server 8000
# open http://localhost:8000/annotate.html
```

For a remote annotator, the same file works from any static host (GitHub
Pages, a department web directory) or can simply be emailed — it is one file.

**Saving.** Work is kept in the browser as you go, but the browser is only a
convenience. The record is the exported file: the status line stays amber
until the current state has been exported, and the page warns before you
navigate away with unexported changes.

## Deploying it

`build_page.py` writes `_site/`, which is the whole deployable site:

```
_site/index.html     the page
_site/robots.txt     Disallow: /
_site/_headers       X-Robots-Tag: noindex, nofollow, noarchive
```

Drag `_site` onto Netlify, or point a git-based deploy at this folder as the
base directory and `netlify.toml` here does the rest. There is no build step.

**Upload `_site`, never this folder.** `passage_key.json` maps the opaque
codes back to novel and author and carries the machine-derived reference level
used to stratify section B. Publishing it would undo the blinding the codes
exist to provide.

**A Netlify site is public to anyone with the URL,** and free-tier Netlify has
no password protection — site passwords and Identity-based access are paid
features. Five of the 63 passages are still in US copyright: three from *The
Years* (1937), one from *Between the Acts* (1941), one from *The Waves* (1931,
public domain from January 2027). All are Woolf, all in section B, and each
runs 170–290 words. Short scholarly excerpts of that size, used for
non-commercial research and substituting for nothing, are a strong fair-use
case; the `noindex` headers and `robots.txt` keep them out of search and
archive crawlers. The remaining 58 passages are public domain. Decide
deliberately rather than by default, and if the posture matters, either put
the site behind a Netlify password on a paid plan or send the coauthor
`annotate.html` directly — it is one file and needs no host.

## Running it

```
python build_annotation_set.py     # -> annotation_set.json, passage_key.json
python machine_readings.py         # -> machine_readings.json  (re-run as readers land)
python build_page.py               # -> annotate.html
# ... annotators code, each exporting annotations_<name>.json ...
cp ~/Downloads/annotations_*.json exports/
python ingest_annotations.py       # validates build stamps -> annotations/*.json
python analyze_agreement.py
```

## What a small panel does not give

- **No majority-vote gold below three annotators.** Two raters cannot break a
  tie. With two, resolve disagreements in an adjudication pass afterwards,
  recorded separately from the blind round; for a humanities chapter that is
  better than majority vote anyway, since it yields reasoning worth quoting.
- **No intra-annotator figure** unless you make one. The machines have k=10; a
  human has k=1. Re-coding ~10 passages after a week gives the honest
  counterpart to machine k-run variance.

## Statistics

`analyze_agreement.py` reports linear-weighted Cohen's kappa as primary — the
cline is ordinal and the prompt tells every reader that 0-vs-1 is
near-agreement while 0-vs-3 is not — with unweighted kappa, exact agreement
and within-one-level rates alongside. Also Krippendorff's alpha (ordinal,
which handles the missing values a no-consciousness verdict produces) and
ICC(2,k).

All are checked against reference implementations: weighted and unweighted
kappa against `sklearn.metrics.cohen_kappa_score` across 18 comparisons, and
ICC(2,k) against the Shrout & Fleiss (1979) Table 1 worked example, which it
reproduces at 0.6201 against the published .62.
