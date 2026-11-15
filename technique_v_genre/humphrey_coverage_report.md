# Humphrey/Steinberg taxonomy coverage

Run on the full corpus: 21 files, 10715 SoC-positive passages, 56937 total (passage, reader, run) rows where is_soc=true.

## Overall fit — row level

Every (passage, reader, run) judgment counted once:

| fit_category | proportion | raw n |
|---|---|---|
| technique_fit | 44.3% | 25245 |
| boundary_tension | 55.0% | 31321 |
| no_type_fits | 0.7% | 371 |

## Overall fit — passage-level, reader-mode comparison

For each passage, each reader's own k=3 runs are first collapsed to that reader's modal fit_category. Passages are then classified by whether the readers' modes agree:

| | proportion of passages | raw n passages |
|---|---|---|
| readers agree (consensus) | 27.8% | 2976 |
| readers stably disagree | 72.2% | 7739 |

Where readers agree, what they agree on:

| fit_category | proportion of consensus passages | raw n |
|---|---|---|
| technique_fit | 17.6% | 524 |
| boundary_tension | 80.7% | 2401 |
| no_type_fits | 1.7% | 51 |

Where readers stably disagree, which pairing:

| pairing | raw n passages |
|---|---|
| boundary_tension/technique_fit | 7680 |
| no_type_fits/technique_fit | 48 |
| boundary_tension/no_type_fits | 11 |

## soc_type distribution (row level)

Which of the twelve Humphrey/Steinberg slugs the corpus actually uses:

| soc_type | proportion | raw n |
|---|---|---|
| direct_interior_monologue | 12.5% | 7136 |
| indirect_interior_monologue | 35.2% | 20019 |
| omniscient_description | 36.5% | 20787 |
| soliloquy | 0.9% | 488 |
| free_association | 0.5% | 257 |
| space_montage | 0.0% | 5 |
| imagery | 1.1% | 650 |
| simulation_state_of_mind | 2.8% | 1569 |
| reverie_fantasy | 2.8% | 1582 |
| hybrid | 3.2% | 1847 |
| other_soc | 0.4% | 204 |

Never assigned in this corpus: orthographic_marker.

## Which named types fit cleanly, and which strain the taxonomy

fit_category rate by soc_type (row level) — where a given named type most often lands once assigned:

| soc_type | technique_fit | boundary_tension | no_type_fits | n |
|---|---|---|---|---|
| direct_interior_monologue | 66.0% | 34.0% | 0.0% | 7136 |
| indirect_interior_monologue | 57.1% | 42.9% | 0.0% | 20019 |
| omniscient_description | 40.3% | 59.6% | 0.0% | 20787 |
| soliloquy | 24.0% | 75.6% | 0.4% | 488 |
| free_association | 23.0% | 77.0% | 0.0% | 257 |
| space_montage | 0.0% | 100.0% | 0.0% | 5 |
| imagery | 0.5% | 99.5% | 0.0% | 650 |
| simulation_state_of_mind | 0.2% | 99.6% | 0.3% | 1569 |
| reverie_fantasy | 7.4% | 92.6% | 0.0% | 1582 |
| hybrid | 17.5% | 82.2% | 0.3% | 1847 |
| other_soc | 0.5% | 10.8% | 88.7% | 204 |

## fit_category by author (row level)

Whether the taxonomy strains more for some authors than others — connects to the genre-pole/technique-author distinction (Experiment A) and the diachronic frontier (Experiment C):

| author | technique_fit | boundary_tension | no_type_fits | n |
|---|---|---|---|---|
| Faulkner | 43.7% | 55.5% | 0.8% | 3539 |
| Ford | 49.1% | 50.1% | 0.9% | 2335 |
| James | 43.1% | 56.6% | 0.3% | 10602 |
| Joyce | 40.4% | 57.8% | 1.8% | 10386 |
| Richardson | 45.1% | 54.8% | 0.1% | 5118 |
| Woolf | 46.0% | 53.6% | 0.4% | 24957 |

## antimimetic flag (the stricter genre-pole criterion)

antimimetic=true is reserved for no_type_fits passages with an explicit textual reason (Richardson 2006 / Alber 2010's unnatural narratology), not merely low-confidence calls. Rate among no_type_fits rows specifically:

57.4% of no_type_fits rows (371 total) are also flagged antimimetic=true — the rest are no_type_fits without that stricter theoretical justification attached.
