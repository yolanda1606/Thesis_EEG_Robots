# Focused personalized binary Image search: weak-participant diagnostic

This is a read-only descriptive analysis of the completed focused-search CSVs and the existing no-ICA Image merged tables. It fits no models, performs no threshold selection, and changes no data, rankings, model outputs, or serialized pipelines.

## Definitions and limits

- Weak and robust are descriptive strata defined from the frozen participant-best outer-CV BA: weak `<0.55`; robust `>=0.65`.
- Sensitivity and specificity are aggregated held-out predictions of each participant-target's frozen best configuration. The completed run does not retain probabilities, so calibration and an optimal threshold cannot be determined here.
- Selected-feature stability is the mean pairwise Jaccard similarity of the five outer-fold selected feature sets for the frozen best configuration.
- LOW/HIGH separation uses the existing 120 `all_eeg` features only: pooled standardized mean differences, rank-biserial effects, and a standardized class-centroid-to-within-class-distance ratio. These are descriptive geometry measures, not fitted classifiers or inferential tests.
- Triage labels are intentionally conservative and do not identify exclusions, modify training, or establish causes.

## Weak versus robust strata

| target | stratum | n | median_best_ba | median_candidate_ba | median_fold_sd | median_minor_class_fraction | median_rank_biserial_max | median_centroid_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | weak (<0.55) | 11.000 | 0.516 | 0.467 | 0.096 | 0.349 | 0.280 | 0.141 |
| arousal | robust (>=0.65) | 5.000 | 0.679 | 0.564 | 0.097 | 0.353 | 0.441 | 0.200 |
| valence | weak (<0.55) | 10.000 | 0.525 | 0.463 | 0.086 | 0.444 | 0.287 | 0.131 |
| valence | robust (>=0.65) | 4.000 | 0.678 | 0.567 | 0.086 | 0.340 | 0.362 | 0.155 |

Weak cases generally show lower univariate and multivariate separation than robust references, but overlap remains substantial. Accordingly, low BA alone is not treated as evidence of a recording problem or of a threshold remedy.

## Focused weak cases

| participant | target | best_ba | median_candidate_ba | fold_ba_sd | fold_ba_range | minority_class_fraction | oof_sensitivity | oof_specificity | mean_pairwise_selected_feature_jaccard | maximum_absolute_rank_biserial_all_eeg | centroid_to_within_distance_ratio_all_eeg | diagnostic_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P02 | arousal | 0.497 | 0.438 | 0.068 | 0.187 | 0.390 | 0.705 | 0.282 | 0.215 | 0.248 | 0.151 | unclear |
| P30 | arousal | 0.500 | 0.467 | 0.000 | 0.000 | 0.367 | 1.000 | 0.000 | 0.258 | 0.215 | 0.105 | likely weak/no separable signal |
| P31 | arousal | 0.500 | 0.425 | 0.000 | 0.000 | 0.487 | 0.000 | 1.000 | 0.172 | 0.216 | 0.104 | likely weak/no separable signal |
| P10 | arousal | 0.502 | 0.474 | 0.135 | 0.362 | 0.467 | 0.578 | 0.429 | 0.283 | 0.272 | 0.129 | likely weak/no separable signal |
| P23 | arousal | 0.513 | 0.438 | 0.067 | 0.149 | 0.305 | 0.793 | 0.222 | 0.354 | 0.322 | 0.157 | unclear |
| P33 | arousal | 0.516 | 0.450 | 0.114 | 0.269 | 0.274 | 0.788 | 0.250 | 0.236 | 0.302 | 0.134 | possible classifier/threshold limitation |
| P04 | arousal | 0.516 | 0.468 | 0.141 | 0.337 | 0.382 | 0.603 | 0.436 | 0.175 | 0.307 | 0.141 | unclear |
| P20 | arousal | 0.522 | 0.474 | 0.100 | 0.256 | 0.312 | 0.656 | 0.379 | 0.184 | 0.280 | 0.146 | unclear |
| P15 | arousal | 0.522 | 0.464 | 0.095 | 0.261 | 0.205 | 0.730 | 0.304 | 0.343 | 0.307 | 0.153 | possible classifier/threshold limitation |
| P28 | arousal | 0.529 | 0.478 | 0.096 | 0.254 | 0.288 | 0.722 | 0.344 | 0.371 | 0.339 | 0.146 | possible classifier/threshold limitation |
| P29 | arousal | 0.543 | 0.496 | 0.096 | 0.223 | 0.349 | 0.803 | 0.289 | 0.399 | 0.275 | 0.131 | likely weak/no separable signal |
| P31 | valence | 0.492 | 0.435 | 0.019 | 0.042 | 0.487 | 0.885 | 0.086 | 0.186 | 0.247 | 0.109 | likely weak/no separable signal |
| P44 | valence | 0.519 | 0.465 | 0.048 | 0.130 | 0.294 | 0.750 | 0.286 | 0.447 | 0.284 | 0.152 | possible classifier/threshold limitation |
| P34 | valence | 0.521 | 0.463 | 0.105 | 0.258 | 0.417 | 0.438 | 0.612 | 0.345 | 0.350 | 0.105 | unclear |
| P33 | valence | 0.522 | 0.483 | 0.080 | 0.178 | 0.487 | 0.211 | 0.833 | 0.265 | 0.291 | 0.139 | unclear |
| P10 | valence | 0.524 | 0.457 | 0.052 | 0.136 | 0.417 | 0.729 | 0.320 | 0.472 | 0.278 | 0.145 | unclear |
| P11 | valence | 0.526 | 0.479 | 0.098 | 0.258 | 0.483 | 0.516 | 0.534 | 0.574 | 0.256 | 0.115 | likely weak/no separable signal |
| P24 | valence | 0.527 | 0.462 | 0.118 | 0.313 | 0.470 | 0.473 | 0.581 | 0.318 | 0.410 | 0.131 | unclear |
| P30 | valence | 0.536 | 0.463 | 0.075 | 0.208 | 0.492 | 0.492 | 0.576 | 0.547 | 0.246 | 0.105 | likely weak/no separable signal |
| P07 | valence | 0.541 | 0.468 | 0.092 | 0.206 | 0.207 | 0.924 | 0.167 | 1.000 | 0.319 | 0.147 | possible classifier/threshold limitation |
| P13 | valence | 0.548 | 0.455 | 0.131 | 0.343 | 0.326 | 0.547 | 0.548 | 0.580 | 0.363 | 0.131 | unclear |

## Robust references

| participant | target | best_ba | fold_ba_sd | minority_class_fraction | maximum_absolute_rank_biserial_all_eeg | centroid_to_within_distance_ratio_all_eeg |
| --- | --- | --- | --- | --- | --- | --- |
| P19 | arousal | 0.704 | 0.031 | 0.458 | 0.498 | 0.238 |
| P36 | arousal | 0.704 | 0.126 | 0.216 | 0.441 | 0.200 |
| P16 | arousal | 0.679 | 0.068 | 0.353 | 0.374 | 0.177 |
| P39 | arousal | 0.656 | 0.251 | 0.119 | 0.482 | 0.204 |
| P06 | arousal | 0.655 | 0.097 | 0.437 | 0.332 | 0.138 |
| P39 | valence | 0.704 | 0.086 | 0.280 | 0.385 | 0.149 |
| P02 | valence | 0.684 | 0.066 | 0.298 | 0.470 | 0.242 |
| P04 | valence | 0.672 | 0.086 | 0.382 | 0.322 | 0.160 |
| P29 | valence | 0.671 | 0.118 | 0.413 | 0.339 | 0.138 |

## Imbalance flags

Severe imbalance is defined before interpretation as a minority-class fraction below 0.30. Threshold optimization is only marked *plausible* when severe imbalance coincides with an absolute held-out sensitivity/specificity gap of at least 0.15; it remains a hypothesis requiring leakage-safe inner-CV evaluation.

| participant | target | low_count | high_count | minority_class_fraction | oof_sensitivity | oof_specificity | oof_absolute_sensitivity_specificity_gap | threshold_optimization_plausible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P07 | arousal | 100.000 | 17.000 | 0.145 | 0.353 | 0.780 | 0.427 | True |
| P12 | arousal | 4.000 | 116.000 | 0.033 | 0.922 | 0.250 | 0.672 | True |
| P15 | arousal | 23.000 | 89.000 | 0.205 | 0.730 | 0.304 | 0.426 | True |
| P17 | arousal | 32.000 | 87.000 | 0.269 | 0.782 | 0.469 | 0.313 | True |
| P27 | arousal | 21.000 | 68.000 | 0.236 | 0.838 | 0.333 | 0.505 | True |
| P28 | arousal | 32.000 | 79.000 | 0.288 | 0.722 | 0.344 | 0.378 | True |
| P32 | arousal | 16.000 | 104.000 | 0.133 | 0.837 | 0.438 | 0.399 | True |
| P33 | arousal | 32.000 | 85.000 | 0.274 | 0.788 | 0.250 | 0.538 | True |
| P34 | arousal | 33.000 | 82.000 | 0.287 | 0.744 | 0.545 | 0.198 | True |
| P36 | arousal | 22.000 | 80.000 | 0.216 | 0.688 | 0.727 | 0.040 | False |
| P37 | arousal | 18.000 | 89.000 | 0.168 | 0.820 | 0.333 | 0.487 | True |
| P39 | arousal | 104.000 | 14.000 | 0.119 | 0.643 | 0.712 | 0.069 | False |
| P46 | arousal | 33.000 | 85.000 | 0.280 | 0.835 | 0.333 | 0.502 | True |
| P02 | valence | 25.000 | 59.000 | 0.298 | 0.847 | 0.520 | 0.327 | True |
| P07 | valence | 24.000 | 92.000 | 0.207 | 0.924 | 0.167 | 0.757 | True |
| P21 | valence | 25.000 | 83.000 | 0.231 | 0.855 | 0.320 | 0.535 | True |
| P39 | valence | 33.000 | 85.000 | 0.280 | 0.894 | 0.515 | 0.379 | True |
| P44 | valence | 35.000 | 84.000 | 0.294 | 0.750 | 0.286 | 0.464 | True |

P03, P13, and P28 remain included. Their individual results appear in the CSV and should be compared in a later pre-specified QC sensitivity analysis rather than used for automatic exclusion.

The full 92-row, machine-readable diagnostic is `focused_personalized_weak_participant_diagnostic.csv` beside this report.
