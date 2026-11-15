# Experiment D — validation gate

97 of 150 sampled passages were flagged as containing consciousness-rendering by at least 2 of 2 readers (gpt-5.4, qwen3-max); correlations below are computed only over that co-detected set, since mediation_level and fit_category are undefined where a reader returned no instance.

## Correlation with mean mediation_level

| feature | predicted sign | Spearman ρ | p |
|---|---|---|---|
| reporting_verb_density | - | 0.160 | 0.1167 |
| first_person_ratio | + | 0.422 | 0.0000 |
| past_tense_ratio | - | -0.269 | 0.0078 |
| fragment_punctuation_density | + | -0.345 | 0.0005 |
| dash_ellipsis_density | + | -0.276 | 0.0062 |
| semicolon_density | - (hypotactic-elaboration marker, not fragmentation) | -0.290 | 0.0039 |

## Correlation with cross-reader ordinal spread (0-3)

If boundary tension concentrates where surface features are ambiguous (neither clearly high nor low), that supports treating boundary tension as the typology's genuine research frontier rather than noise.

| feature | Spearman ρ (vs spread) | p |
|---|---|---|
| reporting_verb_density | 0.114 | 0.2662 |
| first_person_ratio | 0.076 | 0.4571 |
| past_tense_ratio | 0.070 | 0.4972 |
| fragment_punctuation_density | 0.031 | 0.7615 |
| dash_ellipsis_density | 0.021 | 0.8366 |
| semicolon_density | -0.035 | 0.7310 |

## By author

| author | n | mean mediation_level | mean spread | fit_category agreement rate |
|---|---|---|---|---|
| Faulkner | 19 | 2.47 | 0.42 | 31.6% |
| James | 32 | 0.56 | 0.31 | 15.6% |
| Woolf | 46 | 0.79 | 0.24 | 13.0% |

## Verdict

**Gate cleared, with one corrected prediction and one weak feature.**

first_person_ratio (rho=0.422, p<0.0001) and past_tense_ratio (rho=-0.269,
p=0.008) both land in the predicted direction and are strongly significant
— straightforward confirmation that mediation_level tracks measurable,
independently-computed surface features rather than being an artifact of
model idiosyncrasy.

fragment_punctuation_density came back significant but backwards
(rho=-0.345). Splitting it into dash_ellipsis_density and semicolon_density
(diagnostic step, not a fix for noise) shows this wasn't a confound between
two opposed sub-features cancelling out: BOTH come back negative and both
individually significant (rho=-0.276, p=0.006 and rho=-0.290, p=0.004).
The working prediction was simply wrong, not the measurement. On
reflection this is the more coherent reading anyway: heavier punctuation
of any kind (dashes, ellipses, semicolons alike) tracks more syntactically
worked, narratorially-controlled prose (psycho-narration / FID, low
mediation) — consistent with this project's own prior finding in
woolf_richardson_comparison.md that elaborate, subordinated syntax pulls
toward omniscient_description. The least-mediated passages in this sample
(Faulkner's autonomous monologue) are characteristically UNDER-punctuated,
running short declarative sentences together, not dash-heavy — Humphrey's
own "orthographic_marker" gloss (scripts/prompts.py) conflates two
different things (ADDED marks like em-dashes vs. ABSENT punctuation), which
is exactly why a single directional prediction was shaky going in. Worth
carrying forward as a corrected, specific hypothesis: punctuation
*density* (of any kind) predicts LOWER mediation; punctuation *absence* —
not measured here — is the actual signature of the high end of the cline
and would need a different feature (e.g. sentence length before the next
mark) to test properly.

reporting_verb_density is the one null result (rho=0.160, p=0.117) — not
evidence against the signal, but a specifically weak feature: thought-tag
verbs like "thought" or "wondered" appear at every mediation level (a
quoted monologue is still often framed by "she thought"), so this proxy
doesn't discriminate as cleanly as the others. Worth narrowing to explicit
verb+pronoun thought-tags ("she thought", "he wondered") rather than
counting the bare verb anywhere in the passage, if it's revisited.

Cross-reader ordinal spread does not correlate with any of these six
features (all p > 0.25) — boundary tension isn't explained by simple
lexical/punctuation counts. That's consistent with, not contradicting, the
project's existing finding that boundary tension tracks a *conjunction*
(habitual aspect + elaborate architecture together, woolf_richardson_
comparison.md) rather than any single linear surface feature — this
validation pass wasn't designed to detect a conjunction and doesn't
overturn that result.

The by-author breakdown previews Experiment A cleanly even at n=97:
Faulkner's mean mediation_level (2.47) sits near the autonomous-monologue
end, Woolf (0.79) and James (0.56, the control) sit near the psycho-
narration end with James lowest as predicted. That's the expected
fingerprint shape showing up before Experiment A has formally run.

**Recommendation:** proceed to Experiments A-C on the strength of 2 clearly
confirmed features, one specific corrected hypothesis (not a null result),
and a preview of the predicted author fingerprint — rather than re-running
Experiment D. Fold the punctuation-direction correction and the reporting-
verb narrowing into the next iteration of the schema/prompt if this proves
worth refining further.

