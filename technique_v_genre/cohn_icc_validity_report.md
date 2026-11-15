# Is the per-author ICC a measure of Cohn's scale?

Passages: 22,272 SOC-positive passages, 44 titles, 7 authors (pooled gpt-5.4 + qwen3-max + qwen3.6-local, per-passage mean; the chapter's ICC basis). Each table cell is that author's one-way ICC across their titles. **Woolf rank** = her position among the 7 authors (1 = highest). **ρ vs readers** = Spearman correlation, across the 7 authors, between that row's ICCs and the readers' ICCs (n = 7: descriptive, not a significance test).

## Reference: the readers' mediation level

| measure | Woolf | Joyce | Faulkner | Sinclair | Richardson | Ford | James | Woolf rank | ρ vs readers |
|---|---|---|---|---|---|---|---|---|---|
| **readers, pooled** | 0.561 | 0.358 | 0.219 | 0.185 | 0.029 | 0.025 | 0.012 | 1 |  |
| gpt-5.4 only | 0.498 | 0.373 | 0.161 | 0.132 | 0.039 | 0.041 | 0.010 | 1 | +0.96 |
| qwen3-max only | 0.544 | 0.353 | 0.440 | 0.226 | 0.014 | 0.026 | 0.020 | 1 | +0.86 |

## Do the features track level at all? (passage-level Spearman ρ with pooled level)

| feature | expected | ρ with level | within-author ρ |
|---|---|---|---|
| first_person_ratio | + | +0.182 | +0.205 |
| present_aux_ratio | + | +0.175 | +0.172 |
| psycho_verb_density | − | -0.041 | -0.025 |
| thought_tag_density | + | +0.141 | +0.136 |
| fid_question_exclaim | + | +0.122 | +0.092 |
| deictic_in_past_sentence | + | -0.036 | -0.020 |
| cohn_composite | + | +0.259 | +0.269 |
| mean_sentence_length | none | -0.189 | -0.138 |
| mean_word_length | none | -0.111 | -0.176 |
| mattr50 | none | -0.055 | -0.077 |
| comma_density | none | -0.124 | -0.080 |
| noun_ratio | none | +0.061 | -0.052 |
| adj_ratio | none | -0.041 | -0.067 |
| style_pc1 | none | -0.071 | +0.081 |
| style_pc2 | none | -0.242 | -0.210 |
| style_pc3 | none | +0.041 | +0.010 |

## Test 1 — convergent: ICC of Cohn-grounded features (no model labels)

| measure | Woolf | Joyce | Faulkner | Sinclair | Richardson | Ford | James | Woolf rank | ρ vs readers |
|---|---|---|---|---|---|---|---|---|---|
| first_person_ratio | 0.574 | 0.088 | 0.249 | 0.021 | 0.032 | 0.152 | 0.034 | 1 | +0.50 |
| present_aux_ratio | 0.459 | 0.318 | 0.214 | 0.015 | 0.038 | 0.053 | 0.016 | 1 | +0.75 |
| psycho_verb_density | 0.066 | 0.133 | 0.039 | 0.007 | 0.050 | 0.003 | 0.001 | 2 | +0.86 |
| thought_tag_density | 0.117 | 0.043 | 0.010 | 0.013 | 0.011 | 0.012 | 0.002 | 1 | +0.75 |
| fid_question_exclaim | 0.077 | 0.007 | 0.044 | 0.009 | 0.000 | 0.107 | 0.006 | 2 | +0.29 |
| deictic_in_past_sentence | 0.068 | 0.000 | 0.015 | 0.019 | 0.050 | 0.028 | 0.016 | 1 | +0.00 |
| **Cohn composite (theory-signed)** | 0.185 | 0.197 | 0.114 | 0.011 | 0.034 | 0.014 | 0.007 | 2 | +0.86 |

## Test 2 — discriminant: ICC of features outside Cohn's definitions

Style PCs are the first three principal components of 61 function-word frequencies (pronouns and auxiliaries excluded), explaining 4.4%, 3.4%, 2.3% of their variance.

| measure | Woolf | Joyce | Faulkner | Sinclair | Richardson | Ford | James | Woolf rank | ρ vs readers |
|---|---|---|---|---|---|---|---|---|---|
| mean_sentence_length | 0.146 | 0.011 | 0.047 | 0.136 | 0.035 | 0.067 | 0.079 | 1 | +0.04 |
| mean_word_length | 0.114 | 0.116 | 0.406 | 0.038 | 0.102 | 0.035 | 0.038 | 3 | +0.79 |
| mattr50 | 0.100 | 0.334 | 0.243 | 0.081 | 0.048 | 0.034 | 0.006 | 3 | +0.89 |
| comma_density | 0.107 | 0.072 | 0.279 | 0.118 | 0.204 | 0.037 | 0.291 | 5 | -0.32 |
| noun_ratio | 0.222 | 0.247 | 0.045 | 0.042 | 0.034 | 0.056 | 0.079 | 2 | +0.39 |
| adj_ratio | 0.045 | 0.016 | 0.239 | 0.027 | 0.007 | 0.011 | 0.029 | 2 | +0.39 |
| style_pc1 | 0.109 | 0.043 | 0.006 | 0.045 | 0.021 | 0.022 | 0.046 | 1 | +0.14 |
| style_pc2 | 0.286 | 0.027 | 0.185 | 0.199 | 0.137 | 0.004 | 0.016 | 1 | +0.71 |
| style_pc3 | 0.017 | 0.013 | 0.070 | 0.019 | 0.002 | 0.007 | 0.087 | 4 | -0.07 |
| **mean of non-Cohn ICCs** | 0.127 | 0.098 | 0.169 | 0.078 | 0.066 | 0.030 | 0.074 | 2 | +0.79 |
| mean of Cohn-feature ICCs (Test 1) | 0.227 | 0.098 | 0.095 | 0.014 | 0.030 | 0.059 | 0.012 | 1 | +0.86 |

## Test 3 — decomposition: the part of reader level that Cohn's grammar predicts

Leave-one-author-out prediction of pooled level from the 6 Cohn features. Out-of-author R²: linear (ridge) 0.081, gradient boosting 0.115. ICC is recomputed on the predicted values and on the residuals.

| measure | Woolf | Joyce | Faulkner | Sinclair | Richardson | Ford | James | Woolf rank | ρ vs readers |
|---|---|---|---|---|---|---|---|---|---|
| readers (observed) | 0.561 | 0.358 | 0.219 | 0.185 | 0.029 | 0.025 | 0.012 | 1 |  |
| pred_linear | 0.489 | 0.129 | 0.196 | 0.022 | 0.034 | 0.105 | 0.027 | 1 | +0.71 |
| resid_linear | 0.409 | 0.322 | 0.165 | 0.175 | 0.015 | 0.058 | 0.025 | 1 | +0.86 |
| pred_gb | 0.501 | 0.043 | 0.203 | 0.015 | 0.029 | 0.067 | 0.028 | 1 | +0.57 |
| resid_gb | 0.334 | 0.319 | 0.128 | 0.173 | 0.014 | 0.054 | 0.026 | 1 | +0.86 |

### Title means: observed vs Cohn-predicted (gradient boosting), within author

| author | titles | Spearman(observed, predicted title means) |
|---|---|---|
| Woolf | 10 | +0.76 |
| Joyce | 3 | +0.50 |
| Faulkner | 3 | +0.50 |
| Sinclair | 8 | +0.55 |
| Richardson | 9 | +0.43 |
| Ford | 5 | -0.10 |
| James | 6 | +0.26 |

Woolf title means (observed level, Cohn-predicted, residual):

| title | n | observed | predicted | residual |
|---|---|---|---|---|
| Night and Day | 850 | 0.35 | 0.66 | -0.31 |
| The Voyage Out | 659 | 0.36 | 0.67 | -0.31 |
| Flush | 152 | 0.36 | 0.58 | -0.22 |
| Orlando | 401 | 0.46 | 0.68 | -0.22 |
| Jacob's Room | 277 | 0.56 | 0.72 | -0.16 |
| To the Lighthouse | 453 | 0.79 | 0.62 | +0.17 |
| Mrs Dalloway | 380 | 0.87 | 0.67 | +0.20 |
| The Years | 655 | 0.91 | 0.74 | +0.18 |
| Between the Acts | 230 | 1.00 | 0.81 | +0.19 |
| The Waves | 405 | 2.49 | 1.59 | +0.90 |

### Woolf, 95% cluster-bootstrap CIs (passages resampled within title, 1,000 draws)

- level: 0.561 [0.534, 0.586]
- cohn_composite: 0.185 [0.168, 0.205]
- pred_gb: 0.501 [0.475, 0.531]
- resid_gb: 0.334 [0.308, 0.363]
- mean_sentence_length: 0.146 [0.131, 0.166]
- style_pc1: 0.109 [0.093, 0.130]

## Without each author's most extreme title (highest reader-level mean)

Woolf's chapter claim has two parts: *The Waves* departs, and her other nine novels still vary more than other authors' do. This separates which part each measure supports.

| author | titles | level | cohn_composite | first_person_ratio | present_aux_ratio | pred_gb | resid_gb | mean Cohn-feature ICC | mean non-Cohn ICC |
|---|---|---|---|---|---|---|---|---|---|
| Woolf | all | 0.561 | 0.185 | 0.574 | 0.459 | 0.501 | 0.334 | 0.227 | 0.127 |
| Woolf | without *The Waves* | 0.199 | 0.009 | 0.130 | 0.111 | 0.036 | 0.181 | 0.083 | 0.131 |
| Joyce | all | 0.358 | 0.197 | 0.088 | 0.318 | 0.043 | 0.319 | 0.098 | 0.098 |
| Joyce | without *Ulysses* | 0.117 | 0.016 | 0.013 | 0.070 | 0.011 | 0.153 | 0.017 | 0.048 |
| Faulkner | all | 0.219 | 0.114 | 0.249 | 0.214 | 0.203 | 0.128 | 0.095 | 0.169 |
| Faulkner | without *The Sound and the Fury* | 0.205 | 0.155 | 0.220 | 0.350 | 0.328 | 0.056 | 0.115 | 0.211 |
| Sinclair | all | 0.185 | 0.011 | 0.021 | 0.015 | 0.015 | 0.173 | 0.014 | 0.078 |
| Sinclair | without *Mary Olivier: A Life* | 0.104 | 0.013 | 0.025 | 0.019 | 0.017 | 0.090 | 0.014 | 0.066 |
| Richardson | all | 0.029 | 0.034 | 0.032 | 0.038 | 0.029 | 0.014 | 0.030 | 0.066 |
| Richardson | without *The Tunnel* | 0.025 | 0.040 | 0.022 | 0.041 | 0.026 | 0.015 | 0.034 | 0.061 |
| Ford | all | 0.025 | 0.014 | 0.152 | 0.053 | 0.067 | 0.054 | 0.059 | 0.030 |
| Ford | without *No More Parades* | 0.019 | 0.018 | 0.204 | 0.070 | 0.090 | 0.054 | 0.071 | 0.039 |
| James | all | 0.012 | 0.007 | 0.034 | 0.016 | 0.028 | 0.026 | 0.012 | 0.074 |
| James | without *The Ambassadors* | 0.011 | 0.004 | 0.042 | 0.018 | 0.034 | 0.033 | 0.015 | 0.087 |
