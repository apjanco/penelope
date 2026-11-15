# Variance-decomposition robustness checks

Direct response to `critique.md`'s arithmetic objections: eta-squared recomputed as a variance-components ICC (corrects the group-count bias eta^2 has), with a passage-level bootstrap CI, leave-one-title-out sensitivity, within-title level distributions, within-title autocorrelation, and explicit composition robustness checks. See `woolf_variance_decomposition.md` for the write-up this supports.

## ICC (variance-components), bootstrap CI, and leave-one-out sensitivity

ICC replaces the raw eta^2 reported in the original note: eta^2 is upward-biased by title count (an author with more titles is structurally advantaged), which the ICC's unbalanced-groups correction accounts for. The bootstrap CI captures finite-sample uncertainty given the actual titles present; leave-one-out shows how much the figure depends on any single title.

| author | titles | ICC | 95% CI | ICC w/o highest-mean title | which title |
|---|---|---|---|---|---|
| Faulkner | 2 | 0.037 | [0.006, 0.085] | nan | The Sound and the Fury |
| James | 2 | 0.008 | [0.000, 0.022] | nan | The Ambassadors |
| Joyce | 3 | 0.358 | [0.322, 0.391] | 0.117 | Ulysses |
| Richardson * | 3 | 0.001 | [0.000, 0.022] | 0.000 | Honeycomb |
| Woolf | 10 | 0.561 | [0.534, 0.588] | 0.199 | The Waves |

\* **Richardson**: Pilgrimage's volumes are installments of one continuous work (one protagonist, one uninterrupted narrative), not independent books — near-zero between-title variance here is close to tautological, not a finding about a 'fixed style.'

Full leave-one-out table (ICC with each title removed in turn):

- **Joyce** (full ICC = 0.358):
    - without *Ulysses*: 0.117
    - without *Dubliners*: 0.297
    - without *A Portrait of the Artist as a Young Man*: 0.426
- **Richardson** (full ICC = 0.001):
    - without *Honeycomb*: 0.000
    - without *Backwater*: 0.003
    - without *Pointed Roofs*: 0.004
- **Woolf** (full ICC = 0.561):
    - without *The Waves*: 0.199
    - without *Flush*: 0.563
    - without *Night and Day*: 0.570
    - without *Mrs Dalloway*: 0.578
    - without *The Voyage Out*: 0.579
    - without *Jacob's Room*: 0.579
    - without *To the Lighthouse*: 0.580
    - without *Between the Acts*: 0.582
    - without *Orlando*: 0.583
    - without *The Years*: 0.612

## Composition robustness: Woolf without the biography/mock-biography texts

*Flush* (a dog's-eye-view biography) and *Orlando* (a mock-biography) are genre-heterogeneous relative to Woolf's other eight novels; dropping them tests whether the nine-novel 'flat band' finding is an artifact of including them.

- All 10 titles: ICC = 0.561
- Excluding *Flush* and *Orlando* (8 titles): ICC = 0.586
- Finding holds without them — not fragile in this direction, but the choice to include them should be stated, not silent.

## Within-title level distributions

Full proportion breakdown across mediation levels 0-3 per title, not just the mean — a 0.87 mean is compatible with prose that sits flatly at level 1 or that alternates violently between 0 and 2.

| author | title | mean | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| Faulkner | As I Lay Dying | 1.89 | 25% | 14% | 10% | 52% |
| Faulkner | The Sound and the Fury | 2.21 | 18% | 7% | 10% | 64% |
| Ford | The Good Soldier | 0.79 | 64% | 13% | 4% | 20% |
| James | The Ambassadors | 0.48 | 58% | 37% | 4% | 1% |
| James | The Wings of the Dove | 0.40 | 62% | 36% | 2% | 0% |
| Joyce | A Portrait of the Artist as a Young Man | 0.69 | 45% | 47% | 4% | 5% |
| Joyce | Dubliners | 0.38 | 67% | 30% | 2% | 1% |
| Joyce | Ulysses | 1.77 | 20% | 28% | 6% | 45% |
| Richardson | Backwater | 0.88 | 28% | 58% | 11% | 3% |
| Richardson | Honeycomb | 0.96 | 23% | 62% | 10% | 5% |
| Richardson | Pointed Roofs | 0.87 | 26% | 64% | 5% | 4% |
| Woolf | Between the Acts | 1.02 | 24% | 57% | 12% | 7% |
| Woolf | Flush | 0.37 | 65% | 33% | 2% | 0% |
| Woolf | Jacob's Room | 0.59 | 51% | 39% | 9% | 1% |
| Woolf | Mrs Dalloway | 0.89 | 19% | 74% | 7% | 1% |
| Woolf | Night and Day | 0.37 | 68% | 27% | 5% | 0% |
| Woolf | Orlando | 0.51 | 59% | 31% | 10% | 1% |
| Woolf | The Voyage Out | 0.39 | 69% | 24% | 7% | 1% |
| Woolf | The Waves | 2.50 | 4% | 9% | 20% | 67% |
| Woolf | The Years | 0.98 | 27% | 49% | 24% | 0% |
| Woolf | To the Lighthouse | 0.76 | 28% | 67% | 4% | 0% |

## Within-title autocorrelation (lag-1, along text order)

Positive values: the register moves in sustained runs (a passage's level predicts the next). Near zero: the register oscillates passage to passage roughly independently of what came before.

| author | title | lag-1 autocorrelation | n passages |
|---|---|---|---|
| Faulkner | As I Lay Dying | 0.327 | 307 |
| Faulkner | The Sound and the Fury | 0.418 | 421 |
| Ford | The Good Soldier | 0.330 | 429 |
| James | The Ambassadors | 0.151 | 981 |
| James | The Wings of the Dove | 0.022 | 1126 |
| Joyce | A Portrait of the Artist as a Young Man | 0.457 | 417 |
| Joyce | Dubliners | 0.163 | 273 |
| Joyce | Ulysses | 0.571 | 1387 |
| Richardson | Backwater | 0.237 | 301 |
| Richardson | Honeycomb | 0.151 | 291 |
| Richardson | Pointed Roofs | 0.259 | 320 |
| Woolf | Between the Acts | 0.366 | 230 |
| Woolf | Flush | 0.050 | 152 |
| Woolf | Jacob's Room | 0.051 | 277 |
| Woolf | Mrs Dalloway | 0.177 | 380 |
| Woolf | Night and Day | 0.111 | 850 |
| Woolf | Orlando | 0.325 | 401 |
| Woolf | The Voyage Out | 0.080 | 659 |
| Woolf | The Waves | 0.242 | 405 |
| Woolf | The Years | 0.164 | 655 |
| Woolf | To the Lighthouse | 0.315 | 453 |
