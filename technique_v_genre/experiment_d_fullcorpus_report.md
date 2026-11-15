# Experiment D at corpus scale

Passages in the main corpus: 25,157. Co-detected as rendering consciousness by both gpt-5.4 and qwen3-max (modal is_soc over k=3): **16,953**. Features are `analyze_experiment_d.surface_features`, unchanged. Spearman ρ throughout; *within-author* = both variables demeaned by author first.

## Feature vs. level

| feature | predicted | target | pooled ρ | p | n | within-author ρ | p | n |
|---|---|---|---|---|---|---|---|---|
| reporting_verb_density | - | two-reader mean | -0.017 | 0.031 | 16,953 | -0.020 | 0.008 | 16,953 |
| reporting_verb_density | - | gpt-5.4 | -0.021 | 0.008 | 16,953 | -0.021 | 0.006 | 16,953 |
| reporting_verb_density | - | qwen3-max | -0.029 | <0.001 | 16,953 | -0.030 | <0.001 | 16,953 |
| reporting_verb_density | - | jev | -0.062 | <0.001 | 16,774 | -0.060 | <0.001 | 16,774 |
| first_person_ratio | + | two-reader mean | +0.298 | <0.001 | 16,911 | +0.289 | <0.001 | 16,911 |
| first_person_ratio | + | gpt-5.4 | +0.278 | <0.001 | 16,911 | +0.256 | <0.001 | 16,911 |
| first_person_ratio | + | qwen3-max | +0.301 | <0.001 | 16,911 | +0.260 | <0.001 | 16,911 |
| first_person_ratio | + | jev | +0.237 | <0.001 | 16,737 | +0.208 | <0.001 | 16,737 |
| past_tense_ratio | - | two-reader mean | -0.274 | <0.001 | 16,953 | -0.216 | <0.001 | 16,953 |
| past_tense_ratio | - | gpt-5.4 | -0.265 | <0.001 | 16,953 | -0.204 | <0.001 | 16,953 |
| past_tense_ratio | - | qwen3-max | -0.263 | <0.001 | 16,953 | -0.194 | <0.001 | 16,953 |
| past_tense_ratio | - | jev | -0.280 | <0.001 | 16,774 | -0.220 | <0.001 | 16,774 |
| dash_ellipsis_density | + (original prediction; reversed at n=97) | two-reader mean | +0.084 | <0.001 | 16,953 | +0.134 | <0.001 | 16,953 |
| dash_ellipsis_density | + (original prediction; reversed at n=97) | gpt-5.4 | +0.064 | <0.001 | 16,953 | +0.146 | <0.001 | 16,953 |
| dash_ellipsis_density | + (original prediction; reversed at n=97) | qwen3-max | +0.061 | <0.001 | 16,953 | +0.127 | <0.001 | 16,953 |
| dash_ellipsis_density | + (original prediction; reversed at n=97) | jev | +0.078 | <0.001 | 16,774 | +0.130 | <0.001 | 16,774 |
| semicolon_density | - | two-reader mean | -0.082 | <0.001 | 16,953 | +0.044 | <0.001 | 16,953 |
| semicolon_density | - | gpt-5.4 | -0.087 | <0.001 | 16,953 | +0.056 | <0.001 | 16,953 |
| semicolon_density | - | qwen3-max | -0.077 | <0.001 | 16,953 | +0.051 | <0.001 | 16,953 |
| semicolon_density | - | jev | -0.046 | <0.001 | 16,774 | +0.061 | <0.001 | 16,774 |

## Feature vs. cross-reader disagreement (|gpt-5.4 − qwen3-max|)

| feature | pooled ρ | p | n | within-author ρ | p | n |
|---|---|---|---|---|---|---|
| reporting_verb_density | +0.048 | <0.001 | 16,953 | +0.046 | <0.001 | 16,953 |
| first_person_ratio | +0.061 | <0.001 | 16,911 | +0.065 | <0.001 | 16,911 |
| past_tense_ratio | +0.000 | 0.954 | 16,953 | +0.007 | 0.373 | 16,953 |
| dash_ellipsis_density | +0.019 | 0.013 | 16,953 | +0.014 | 0.063 | 16,953 |
| semicolon_density | +0.025 | 0.001 | 16,953 | -0.031 | <0.001 | 16,953 |

## By author (two-reader mean level)

| author | n | mean level | mean spread | ρ first-person | ρ past-tense |
|---|---|---|---|---|---|
| Faulkner | 787 | 1.85 | 0.49 | +0.56 | -0.27 |
| Ford | 1,790 | 0.85 | 0.39 | +0.17 | -0.20 |
| James | 3,738 | 0.45 | 0.38 | +0.03 | -0.11 |
| Joyce | 1,443 | 1.50 | 0.40 | +0.56 | -0.46 |
| Richardson | 2,696 | 0.93 | 0.35 | +0.27 | -0.24 |
| Sinclair | 2,702 | 0.59 | 0.33 | +0.19 | -0.22 |
| Woolf | 3,797 | 0.87 | 0.38 | +0.32 | -0.15 |
