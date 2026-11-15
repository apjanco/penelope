# Memorization / masking test

177 sampled passages across 6 titles; 166 scored as SoC-positive under masking on both passes (11 dropped — masked version scored as non-SoC by at least one reader-run majority, itself informative about how much the identification depended on recognizable names/style rather than content).

## Overall shift

- Mean shift (masked - original): -0.042 (SD 0.472)
- Correlation between original and masked scores: 0.902
- Mean absolute shift: 0.253

## By title

| title | n | mean original | mean masked | mean shift | correlation |
|---|---|---|---|---|---|
| As I Lay Dying | 27 | 1.59 | 1.57 | -0.02 | 0.93 |
| Mrs Dalloway | 25 | 1.00 | 1.09 | +0.09 | 0.74 |
| The Sound and the Fury | 28 | 1.58 | 1.36 | -0.23 | 0.90 |
| The Waves | 32 | 1.60 | 1.48 | -0.12 | 0.94 |
| To the Lighthouse | 24 | 0.98 | 0.97 | -0.01 | 0.95 |
| Ulysses | 30 | 1.57 | 1.62 | +0.05 | 0.90 |

## Why this report does NOT include a masked-vs-original title ranking

The sample was deliberately stratified evenly across mediation levels (up to 8 passages per level per title), which is the right design for the paired, per-passage comparison above — but it means each title's in-sample mean is NOT an estimate of that title's true corpus-wide mean, since the natural imbalance in how much of each book sits at each level (the very thing that makes *The Waves*'s true mean higher than *Ulysses*'s) has been flattened out by construction. A title-mean ranking computed on this sample would compare two artifacts of the sampling design, not the real books, in either the original or masked condition — so it isn't reported here. The valid read of this test is the per-passage paired comparison above: masking leaves individual passage scores strongly correlated with their originals (0.90) and produces no large systematic shift, which is evidence the classifier is substantially responding to the passage's own grammar rather than retrieving the book's identity — but it is silent on whether the *aggregate* Waves-vs-Ulysses ordering specifically would survive masking. A proportionally-sampled follow-up (not level-stratified) would be needed to test that directly.
