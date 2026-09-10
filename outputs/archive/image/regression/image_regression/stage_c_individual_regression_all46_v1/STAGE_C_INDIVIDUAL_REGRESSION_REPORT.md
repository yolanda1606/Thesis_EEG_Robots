# Stage C individual regression: all 46 participants

## Scope and evaluation

This report summarizes the completed Stage C individual regression run for 46 participants (P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46). It uses the already saved outer-fold results; no models were refit for this report.

Each participant-target configuration used 5-fold nested shuffled K-fold cross-validation. Hyperparameters were selected in each training split using `neg_root_mean_squared_error; sklearn negates RMSE because greater scores are better`. The reported outcomes are unconstrained continuous ratings.

For each participant and target, the table below selects the configuration with the lowest mean outer-fold RMSE across the 36 evaluated combinations (3 modalities x 3 regressors x 4 feature-count requests). Exact RMSE ties are resolved deterministically in the order EEG, face, multimodal; KNN, ridge, SVR; 5, 10, 20, all. The `RMSE ties` column records how many configurations shared that exact minimum.

## Participant-best performance

### Valence

Participant-best RMSE: mean 1.752; median 1.738; SD 0.326; range 0.740-2.455. MAE: mean 1.490; median 1.474; SD 0.324; range 0.553-2.235. R2: mean -0.052; median -0.028; SD 0.128; range -0.715-0.139. Positive mean R2: 12/46 participants.

### Arousal

Participant-best RMSE: mean 1.563; median 1.532; SD 0.371; range 0.849-2.526. MAE: mean 1.327; median 1.293; SD 0.375; range 0.735-2.359. R2: mean -0.042; median -0.024; SD 0.080; range -0.324-0.104. Positive mean R2: 12/46 participants.

| Participant | Target | Modality | Regressor | Features | RMSE | RMSE SD | MAE | R2 | Explained variance | RMSE ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | arousal | face | ridge | 5 | 1.398 | 0.181 | 1.184 | 0.077 | 0.111 | 1 |
| P01 | valence | face | svr | 5 | 1.491 | 0.226 | 1.303 | -0.058 | 0.007 | 1 |
| P02 | arousal | face | knn | 10 | 1.390 | 0.124 | 1.157 | -0.084 | 0.036 | 3 |
| P02 | valence | eeg | knn | all | 1.575 | 0.229 | 1.302 | -0.055 | 0.109 | 1 |
| P03 | arousal | multimodal | svr | all | 2.471 | 0.201 | 2.343 | -0.274 | 0.048 | 1 |
| P03 | valence | multimodal | knn | 20 | 2.185 | 0.259 | 1.897 | -0.715 | 0.094 | 1 |
| P04 | arousal | eeg | knn | all | 1.791 | 0.201 | 1.499 | 0.006 | 0.080 | 1 |
| P04 | valence | face | knn | 10 | 1.802 | 0.221 | 1.477 | 0.059 | 0.078 | 3 |
| P05 | arousal | eeg | svr | all | 1.565 | 0.179 | 1.340 | -0.093 | 0.000 | 1 |
| P05 | valence | multimodal | svr | all | 1.858 | 0.171 | 1.661 | -0.016 | 0.000 | 1 |
| P06 | arousal | face | ridge | 10 | 1.468 | 0.183 | 1.267 | -0.011 | 0.008 | 3 |
| P06 | valence | multimodal | ridge | 20 | 1.707 | 0.136 | 1.409 | -0.004 | 0.041 | 1 |
| P07 | arousal | face | ridge | 10 | 1.328 | 0.360 | 1.033 | -0.324 | 0.008 | 3 |
| P07 | valence | face | svr | 10 | 0.740 | 0.096 | 0.553 | 0.019 | 0.140 | 3 |
| P08 | arousal | multimodal | svr | all | 0.849 | 0.100 | 0.739 | -0.092 | 0.014 | 1 |
| P08 | valence | eeg | svr | all | 1.366 | 0.184 | 1.161 | -0.044 | 0.046 | 1 |
| P09 | arousal | eeg | svr | 5 | 1.540 | 0.195 | 1.238 | -0.096 | 0.014 | 2 |
| P09 | valence | face | knn | 5 | 1.971 | 0.087 | 1.709 | 0.025 | 0.045 | 1 |
| P10 | arousal | face | ridge | 5 | 1.524 | 0.043 | 1.376 | -0.015 | 0.011 | 1 |
| P10 | valence | face | ridge | 5 | 1.936 | 0.070 | 1.755 | -0.084 | -0.028 | 1 |
| P11 | arousal | face | svr | 10 | 1.905 | 0.057 | 1.652 | -0.079 | -0.000 | 3 |
| P11 | valence | face | ridge | 5 | 2.107 | 0.085 | 1.842 | 0.009 | 0.043 | 1 |
| P12 | arousal | eeg | svr | all | 1.046 | 0.140 | 0.858 | -0.024 | 0.028 | 1 |
| P12 | valence | face | ridge | 5 | 1.806 | 0.279 | 1.585 | -0.070 | 0.061 | 1 |
| P13 | arousal | eeg | knn | 5 | 1.069 | 0.261 | 0.903 | -0.045 | 0.017 | 1 |
| P13 | valence | eeg | svr | 5 | 1.285 | 0.088 | 0.994 | -0.076 | 0.005 | 1 |
| P14 | arousal | face | ridge | 10 | 2.526 | 0.195 | 2.359 | -0.140 | 0.000 | 3 |
| P14 | valence | multimodal | knn | all | 2.455 | 0.153 | 2.235 | -0.005 | 0.013 | 1 |
| P15 | arousal | face | ridge | 5 | 1.297 | 0.168 | 0.939 | -0.015 | 0.043 | 1 |
| P15 | valence | face | ridge | 5 | 1.691 | 0.266 | 1.340 | 0.022 | 0.032 | 1 |
| P16 | arousal | multimodal | knn | 5 | 1.610 | 0.179 | 1.352 | 0.029 | 0.080 | 1 |
| P16 | valence | eeg | svr | all | 1.700 | 0.136 | 1.493 | -0.025 | 0.000 | 1 |
| P17 | arousal | multimodal | svr | 5 | 1.260 | 0.204 | 1.033 | -0.067 | 0.015 | 1 |
| P17 | valence | eeg | ridge | 5 | 1.692 | 0.143 | 1.447 | 0.059 | 0.098 | 1 |
| P18 | arousal | face | ridge | 5 | 1.408 | 0.224 | 1.159 | -0.025 | 0.016 | 1 |
| P18 | valence | eeg | svr | all | 1.860 | 0.209 | 1.673 | -0.104 | -0.031 | 1 |
| P19 | arousal | eeg | ridge | 20 | 1.544 | 0.311 | 1.293 | 0.104 | 0.128 | 1 |
| P19 | valence | face | ridge | 5 | 1.567 | 0.270 | 1.248 | 0.009 | 0.047 | 1 |
| P20 | arousal | eeg | svr | 10 | 1.348 | 0.268 | 1.114 | 0.017 | 0.024 | 1 |
| P20 | valence | eeg | svr | 5 | 1.613 | 0.178 | 1.357 | -0.043 | -0.007 | 1 |
| P21 | arousal | eeg | ridge | all | 2.071 | 0.226 | 1.804 | 0.014 | 0.061 | 1 |
| P21 | valence | eeg | ridge | 20 | 1.769 | 0.154 | 1.397 | 0.074 | 0.106 | 1 |
| P22 | arousal | multimodal | knn | all | 2.105 | 0.298 | 1.782 | 0.024 | 0.086 | 1 |
| P22 | valence | face | knn | 10 | 2.174 | 0.307 | 1.891 | -0.056 | -0.020 | 3 |
| P23 | arousal | eeg | svr | all | 1.586 | 0.161 | 1.293 | -0.047 | 0.000 | 1 |
| P23 | valence | eeg | knn | all | 1.953 | 0.230 | 1.681 | -0.000 | 0.004 | 1 |
| P24 | arousal | face | ridge | 5 | 2.091 | 0.244 | 1.922 | 0.011 | 0.056 | 1 |
| P24 | valence | eeg | ridge | all | 2.444 | 0.184 | 2.206 | -0.114 | -0.014 | 1 |
| P25 | arousal | eeg | ridge | 5 | 1.629 | 0.098 | 1.449 | -0.073 | -0.043 | 2 |
| P25 | valence | multimodal | knn | 20 | 1.799 | 0.165 | 1.541 | -0.114 | -0.040 | 1 |
| P26 | arousal | eeg | svr | all | 2.059 | 0.239 | 1.854 | -0.023 | 0.000 | 1 |
| P26 | valence | eeg | knn | 5 | 2.183 | 0.252 | 1.886 | -0.080 | -0.041 | 2 |
| P27 | arousal | eeg | svr | 5 | 1.235 | 0.184 | 1.011 | -0.018 | 0.026 | 2 |
| P27 | valence | eeg | ridge | 10 | 1.525 | 0.215 | 1.272 | -0.010 | 0.065 | 1 |
| P28 | arousal | face | ridge | 5 | 1.466 | 0.299 | 1.118 | 0.021 | 0.048 | 1 |
| P28 | valence | face | ridge | 5 | 1.578 | 0.098 | 1.255 | -0.000 | 0.035 | 1 |
| P29 | arousal | multimodal | svr | all | 1.520 | 0.179 | 1.274 | -0.203 | -0.043 | 1 |
| P29 | valence | eeg | ridge | 5 | 1.997 | 0.279 | 1.741 | -0.185 | -0.052 | 2 |
| P30 | arousal | multimodal | svr | all | 1.390 | 0.151 | 1.198 | -0.020 | -0.000 | 1 |
| P30 | valence | face | knn | 5 | 1.854 | 0.214 | 1.579 | -0.015 | 0.004 | 1 |
| P31 | arousal | multimodal | svr | all | 1.641 | 0.218 | 1.478 | -0.007 | -0.000 | 1 |
| P31 | valence | multimodal | svr | all | 2.119 | 0.253 | 1.963 | -0.054 | -0.000 | 1 |
| P32 | arousal | face | svr | 10 | 0.955 | 0.181 | 0.735 | -0.082 | -0.002 | 3 |
| P32 | valence | eeg | svr | all | 1.498 | 0.107 | 1.200 | -0.065 | -0.007 | 1 |
| P33 | arousal | eeg | svr | all | 1.551 | 0.250 | 1.327 | -0.059 | 0.000 | 1 |
| P33 | valence | face | ridge | 10 | 1.513 | 0.248 | 1.275 | 0.139 | 0.226 | 3 |
| P34 | arousal | eeg | ridge | 5 | 1.681 | 0.158 | 1.384 | -0.166 | 0.048 | 1 |
| P34 | valence | eeg | svr | all | 1.700 | 0.158 | 1.471 | -0.320 | 0.000 | 1 |
| P35 | arousal | face | ridge | 5 | 1.595 | 0.082 | 1.347 | -0.011 | 0.020 | 1 |
| P35 | valence | face | knn | 5 | 1.985 | 0.129 | 1.699 | 0.012 | 0.056 | 1 |
| P36 | arousal | eeg | ridge | all | 1.452 | 0.127 | 1.109 | 0.011 | 0.082 | 1 |
| P36 | valence | face | ridge | 5 | 1.913 | 0.229 | 1.590 | -0.020 | 0.165 | 1 |
| P37 | arousal | face | svr | 5 | 1.146 | 0.318 | 0.851 | -0.022 | -0.001 | 1 |
| P37 | valence | face | knn | 10 | 1.673 | 0.264 | 1.350 | -0.012 | 0.042 | 3 |
| P38 | arousal | multimodal | svr | all | 1.459 | 0.156 | 1.305 | 0.034 | 0.078 | 1 |
| P38 | valence | face | ridge | 10 | 1.588 | 0.176 | 1.376 | -0.031 | 0.085 | 3 |
| P39 | arousal | face | svr | 10 | 1.229 | 0.167 | 0.979 | -0.026 | -0.010 | 3 |
| P39 | valence | face | ridge | 5 | 1.312 | 0.282 | 1.083 | -0.114 | -0.017 | 1 |
| P40 | arousal | face | knn | 10 | 1.982 | 0.205 | 1.673 | -0.076 | 0.061 | 3 |
| P40 | valence | multimodal | knn | 20 | 2.062 | 0.109 | 1.756 | -0.054 | 0.039 | 1 |
| P41 | arousal | face | ridge | 10 | 2.095 | 0.159 | 1.858 | -0.012 | 0.003 | 3 |
| P41 | valence | multimodal | svr | all | 1.971 | 0.101 | 1.662 | -0.251 | -0.006 | 1 |
| P42 | arousal | face | knn | 10 | 1.624 | 0.126 | 1.404 | -0.027 | 0.021 | 3 |
| P42 | valence | face | svr | 10 | 1.338 | 0.161 | 1.142 | 0.030 | 0.077 | 3 |
| P43 | arousal | eeg | svr | all | 1.217 | 0.064 | 1.021 | -0.048 | 0.052 | 1 |
| P43 | valence | eeg | svr | all | 1.307 | 0.185 | 0.997 | -0.068 | 0.000 | 1 |
| P44 | arousal | multimodal | ridge | 20 | 2.035 | 0.172 | 1.761 | 0.071 | 0.149 | 1 |
| P44 | valence | face | svr | 10 | 1.967 | 0.341 | 1.616 | 0.077 | 0.144 | 3 |
| P45 | arousal | face | svr | 10 | 1.603 | 0.178 | 1.362 | -0.026 | 0.016 | 3 |
| P45 | valence | multimodal | svr | 10 | 1.458 | 0.159 | 1.201 | -0.034 | 0.001 | 1 |
| P46 | arousal | face | svr | 5 | 1.154 | 0.204 | 0.912 | -0.039 | -0.005 | 1 |
| P46 | valence | eeg | svr | 5 | 1.516 | 0.127 | 1.269 | -0.026 | 0.015 | 1 |

## Model-selection patterns

- Valence — modality: eeg: 17; face: 20; multimodal: 9; regressor: knn: 13; ridge: 16; svr: 17; feature request: 5: 18; 10: 10; 20: 5; all: 13.
- Arousal — modality: eeg: 16; face: 20; multimodal: 10; regressor: knn: 7; ridge: 17; svr: 22; feature request: 5: 16; 10: 12; 20: 2; all: 16.

## Strongest participant-best results

### Valence

- P07: RMSE 0.740, MAE 0.553, R2 0.019; face/svr/10 features.
- P13: RMSE 1.285, MAE 0.994, R2 -0.076; eeg/svr/5 features.
- P43: RMSE 1.307, MAE 0.997, R2 -0.068; eeg/svr/all features.
- P39: RMSE 1.312, MAE 1.083, R2 -0.114; face/ridge/5 features.
- P42: RMSE 1.338, MAE 1.142, R2 0.030; face/svr/10 features.

### Arousal

- P08: RMSE 0.849, MAE 0.739, R2 -0.092; multimodal/svr/all features.
- P32: RMSE 0.955, MAE 0.735, R2 -0.082; face/svr/10 features.
- P12: RMSE 1.046, MAE 0.858, R2 -0.024; eeg/svr/all features.
- P13: RMSE 1.069, MAE 0.903, R2 -0.045; eeg/knn/5 features.
- P37: RMSE 1.146, MAE 0.851, R2 -0.022; face/svr/5 features.

## Interpretation and limitations

These results estimate within-participant predictive performance using held-out folds, so they are appropriate for exploratory participant-specific calibration. However, choosing the lowest-RMSE option from 36 configurations per participant-target is an optimistic selection procedure, not an independent performance estimate for a pre-specified deployed model.

R2 is retained without clipping: negative values mean the configuration performed worse than the fold's mean-rating baseline. Exact minimum-RMSE ties occurred for 24/92 participant-target selections. Modality and regressor counts therefore describe selected configurations, not proof that one modality or algorithm is intrinsically superior.

No hypothesis tests, permutation tests, or corrections for the configuration search are included. Results should be presented as exploratory unless evaluated on an independent held-out session or a pre-registered final-model protocol.
