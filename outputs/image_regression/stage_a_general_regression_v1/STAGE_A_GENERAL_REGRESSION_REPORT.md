# Stage A general regression

Cohort: P10, P11, P19, P22, P25, P32, P40, P42, P43, P45. Evaluation used outer leave-one-participant-out CV with group-safe inner tuning. Primary predictions are unconstrained.

## Valence

Best configuration: `multimodal` / `svr` / `5` features.
- RMSE: 1.766 ± 0.352 (range 1.308–2.271)
- MAE: 1.493
- R2: -0.030 (range -0.121–-0.000)
- Explained variance: -0.003
- Most frequent tuned hyperparameters: {"regressor__C": 0.1, "regressor__epsilon": 0.1, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (8/10 folds); {"regressor__C": 0.1, "regressor__epsilon": 0.1, "regressor__gamma": 0.1, "regressor__kernel": "rbf"} (1/10 folds); {"regressor__C": 1, "regressor__epsilon": 0.1, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (1/10 folds)

Best by modality:

| modality | regressor | feature_count_request | rmse_mean | rmse_std | mae_mean | r2_mean | explained_variance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | svr | 5 | 1.766 | 0.352 | 1.493 | -0.030 | -0.003 |
| face | svr | 5 | 1.767 | 0.352 | 1.495 | -0.032 | 0.001 |
| multimodal | svr | 5 | 1.766 | 0.352 | 1.493 | -0.030 | -0.003 |

Most stable selected features:

- eeg_se__C4 — selected in 90% of LOSO folds; mean f-regression score 10.687.
- eeg_bp_beta__PO8 — selected in 90% of LOSO folds; mean f-regression score 10.627.
- eeg_se_beta__C4 — selected in 80% of LOSO folds; mean f-regression score 9.789.
- video_enso_norm_mean — selected in 60% of LOSO folds; mean f-regression score 9.209.
- eeg_mf_hz__Cz — selected in 30% of LOSO folds; mean f-regression score 8.283.
- eeg_bp_alpha__C4 — selected in 30% of LOSO folds; mean f-regression score 8.193.
- eeg_mf_hz__PO8 — selected in 30% of LOSO folds; mean f-regression score 7.898.
- eeg_hc__Pz — selected in 20% of LOSO folds; mean f-regression score 7.708.
- eeg_se_delta__C3 — selected in 20% of LOSO folds; mean f-regression score 5.544.
- eeg_bp_beta__Pz — selected in 10% of LOSO folds; mean f-regression score 7.961.

## Arousal

Best configuration: `multimodal` / `svr` / `all` features.
- RMSE: 1.702 ± 0.338 (range 1.154–2.327)
- MAE: 1.474
- R2: -0.112 (range -0.427–-0.005)
- Explained variance: -0.000
- Most frequent tuned hyperparameters: {"regressor__C": 0.1, "regressor__epsilon": 0.1, "regressor__gamma": 0.1, "regressor__kernel": "rbf"} (4/10 folds); {"regressor__C": 10, "regressor__epsilon": 0.1, "regressor__gamma": 0.1, "regressor__kernel": "rbf"} (3/10 folds); {"regressor__C": 0.1, "regressor__epsilon": 0.25, "regressor__gamma": 0.1, "regressor__kernel": "rbf"} (2/10 folds); {"regressor__C": 1, "regressor__epsilon": 0.25, "regressor__gamma": 0.1, "regressor__kernel": "rbf"} (1/10 folds)

Best by modality:

| modality | regressor | feature_count_request | rmse_mean | rmse_std | mae_mean | r2_mean | explained_variance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | svr | 5 | 1.702 | 0.374 | 1.450 | -0.102 | 0.000 |
| face | svr | 5 | 1.728 | 0.337 | 1.481 | -0.156 | -0.007 |
| multimodal | svr | all | 1.702 | 0.338 | 1.474 | -0.112 | -0.000 |

## Interpretation

Across both targets, the best mean R2 values are negative. This means that, under held-out-participant evaluation, these models performed worse than the R2 test-fold baseline; negative values were retained rather than clipped.

Explained variance is close to zero for the best models. Together with negative R2, this suggests predictions capture little trial-level variation and also have participant-level calibration bias. The result does not support a strong claim of generalizable continuous-rating prediction in this Stage A cohort.

## Classification comparison

Classification and regression answer different questions: classification distinguishes LOW (<4) from HIGH (>=4), whereas regression predicts the original continuous rating. Balanced accuracy and RMSE/R2 are therefore not numerically comparable.

- Valence: best classification balanced accuracy 0.526 (eeg, knn, 20 features); best regression RMSE/R2 1.766 / -0.030 (multimodal, svr, 5 features).
- Arousal: best classification balanced accuracy 0.511 (multimodal, gnb, 5 features); best regression RMSE/R2 1.702 / -0.112 (multimodal, svr, all features).

## All configurations

| target | modality | regressor | feature_count_request | rmse_mean | rmse_std | mae_mean | r2_mean | explained_variance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | multimodal | svr | all | 1.702 | 0.338 | 1.474 | -0.112 | -0.000 |
| arousal | eeg | svr | 5 | 1.702 | 0.374 | 1.450 | -0.102 | 0.000 |
| arousal | multimodal | svr | 5 | 1.704 | 0.374 | 1.450 | -0.104 | 0.000 |
| arousal | eeg | svr | all | 1.704 | 0.337 | 1.475 | -0.116 | -0.001 |
| arousal | eeg | svr | 10 | 1.712 | 0.376 | 1.461 | -0.116 | -0.008 |
| arousal | multimodal | svr | 10 | 1.713 | 0.377 | 1.461 | -0.117 | -0.007 |
| arousal | eeg | ridge | 5 | 1.719 | 0.337 | 1.484 | -0.141 | -0.013 |
| arousal | multimodal | ridge | 5 | 1.722 | 0.336 | 1.484 | -0.145 | -0.013 |
| arousal | multimodal | svr | 20 | 1.727 | 0.349 | 1.482 | -0.151 | -0.040 |
| arousal | face | svr | 5 | 1.728 | 0.337 | 1.481 | -0.156 | -0.007 |
| arousal | eeg | svr | 20 | 1.730 | 0.351 | 1.478 | -0.155 | -0.039 |
| arousal | face | ridge | 5 | 1.735 | 0.317 | 1.491 | -0.167 | -0.006 |
| arousal | face | svr | 10 | 1.737 | 0.342 | 1.491 | -0.169 | -0.009 |
| arousal | face | svr | 20 | 1.737 | 0.342 | 1.491 | -0.169 | -0.009 |
| arousal | face | svr | all | 1.737 | 0.342 | 1.491 | -0.169 | -0.009 |
| arousal | eeg | ridge | 20 | 1.744 | 0.338 | 1.497 | -0.175 | -0.037 |
| arousal | eeg | ridge | 10 | 1.750 | 0.351 | 1.498 | -0.179 | -0.033 |
| arousal | multimodal | ridge | 10 | 1.751 | 0.351 | 1.497 | -0.180 | -0.031 |
| arousal | multimodal | ridge | 20 | 1.769 | 0.332 | 1.509 | -0.210 | -0.045 |
| arousal | face | ridge | 10 | 1.783 | 0.329 | 1.519 | -0.228 | -0.019 |
| arousal | face | ridge | 20 | 1.783 | 0.329 | 1.519 | -0.228 | -0.019 |
| arousal | face | ridge | all | 1.783 | 0.329 | 1.519 | -0.228 | -0.019 |
| arousal | eeg | knn | 5 | 1.786 | 0.314 | 1.522 | -0.241 | -0.108 |
| arousal | eeg | knn | 10 | 1.807 | 0.289 | 1.540 | -0.290 | -0.156 |
| arousal | multimodal | knn | 5 | 1.810 | 0.313 | 1.537 | -0.278 | -0.118 |
| arousal | multimodal | knn | 10 | 1.818 | 0.286 | 1.542 | -0.307 | -0.142 |
| arousal | eeg | knn | 20 | 1.826 | 0.298 | 1.546 | -0.310 | -0.172 |
| arousal | multimodal | knn | 20 | 1.832 | 0.297 | 1.550 | -0.317 | -0.151 |
| arousal | multimodal | knn | all | 1.845 | 0.283 | 1.566 | -0.356 | -0.139 |
| arousal | eeg | knn | all | 1.846 | 0.309 | 1.570 | -0.341 | -0.136 |
| arousal | face | knn | 5 | 1.859 | 0.314 | 1.584 | -0.372 | -0.121 |
| arousal | eeg | ridge | all | 1.881 | 0.352 | 1.590 | -0.370 | -0.148 |
| arousal | face | knn | 10 | 1.883 | 0.279 | 1.591 | -0.431 | -0.160 |
| arousal | face | knn | 20 | 1.883 | 0.279 | 1.591 | -0.431 | -0.160 |
| arousal | face | knn | all | 1.883 | 0.279 | 1.591 | -0.431 | -0.160 |
| arousal | multimodal | ridge | all | 1.940 | 0.392 | 1.626 | -0.440 | -0.150 |
| valence | multimodal | svr | 5 | 1.766 | 0.352 | 1.493 | -0.030 | -0.003 |
| valence | eeg | svr | 5 | 1.766 | 0.352 | 1.493 | -0.030 | -0.003 |
| valence | face | svr | 5 | 1.767 | 0.352 | 1.495 | -0.032 | 0.001 |
| valence | multimodal | svr | all | 1.770 | 0.352 | 1.494 | -0.035 | -0.000 |
| valence | eeg | svr | 10 | 1.771 | 0.356 | 1.495 | -0.035 | -0.004 |
| valence | eeg | svr | all | 1.773 | 0.349 | 1.496 | -0.039 | -0.002 |
| valence | multimodal | svr | 10 | 1.773 | 0.357 | 1.496 | -0.038 | -0.003 |
| valence | eeg | svr | 20 | 1.776 | 0.360 | 1.502 | -0.041 | -0.007 |
| valence | multimodal | ridge | 5 | 1.778 | 0.361 | 1.517 | -0.043 | -0.012 |
| valence | eeg | ridge | 5 | 1.778 | 0.359 | 1.512 | -0.043 | -0.016 |
| valence | multimodal | svr | 20 | 1.780 | 0.362 | 1.505 | -0.046 | -0.009 |
| valence | face | svr | 10 | 1.786 | 0.372 | 1.497 | -0.054 | -0.001 |
| valence | face | svr | 20 | 1.786 | 0.372 | 1.497 | -0.054 | -0.001 |
| valence | face | svr | all | 1.786 | 0.372 | 1.497 | -0.054 | -0.001 |
| valence | face | ridge | 5 | 1.790 | 0.360 | 1.525 | -0.058 | -0.008 |
| valence | multimodal | ridge | 10 | 1.809 | 0.382 | 1.530 | -0.080 | -0.026 |
| valence | face | ridge | 10 | 1.811 | 0.362 | 1.548 | -0.083 | -0.015 |
| valence | face | ridge | 20 | 1.811 | 0.362 | 1.548 | -0.083 | -0.015 |
| valence | face | ridge | all | 1.811 | 0.362 | 1.548 | -0.083 | -0.015 |
| valence | eeg | ridge | 10 | 1.813 | 0.388 | 1.528 | -0.084 | -0.030 |
| valence | multimodal | ridge | 20 | 1.822 | 0.393 | 1.536 | -0.095 | -0.031 |
| valence | eeg | ridge | 20 | 1.826 | 0.394 | 1.538 | -0.099 | -0.036 |
| valence | multimodal | knn | 5 | 1.840 | 0.341 | 1.567 | -0.124 | -0.096 |
| valence | eeg | knn | 5 | 1.843 | 0.344 | 1.557 | -0.128 | -0.088 |
| valence | eeg | knn | 20 | 1.863 | 0.368 | 1.570 | -0.153 | -0.117 |
| valence | eeg | knn | all | 1.866 | 0.359 | 1.581 | -0.153 | -0.101 |
| valence | multimodal | knn | 10 | 1.868 | 0.367 | 1.591 | -0.156 | -0.109 |
| valence | eeg | knn | 10 | 1.873 | 0.374 | 1.581 | -0.163 | -0.108 |
| valence | multimodal | knn | all | 1.874 | 0.358 | 1.585 | -0.163 | -0.110 |
| valence | multimodal | knn | 20 | 1.878 | 0.355 | 1.587 | -0.171 | -0.126 |
| valence | face | knn | 5 | 1.885 | 0.385 | 1.606 | -0.175 | -0.096 |
| valence | eeg | ridge | all | 1.911 | 0.440 | 1.603 | -0.198 | -0.083 |
| valence | face | knn | 10 | 1.915 | 0.389 | 1.607 | -0.215 | -0.112 |
| valence | face | knn | 20 | 1.915 | 0.389 | 1.607 | -0.215 | -0.112 |
| valence | face | knn | all | 1.915 | 0.389 | 1.607 | -0.215 | -0.112 |
| valence | multimodal | ridge | all | 1.925 | 0.462 | 1.614 | -0.216 | -0.097 |
