# Stage A: General-model LOSO summary

Stage A used the 10 `Very clean` participants: P10, P11, P19, P22, P25, P32, P40, P42, P43, P45. Each outer fold held out one entire participant; tuning, scaling, and feature selection used the other nine only. Balanced-accuracy chance level is 0.50.

## Usable trials

EEG valence/arousal: 1,200 / 1,200; face valence/arousal: 1,200 / 1,200; multimodal valence/arousal: 1,200 / 1,200.

## Best mean LOSO balanced accuracy

### Valence

| Model | EEG | Face | Multimodal |
|---|---:|---:|---:|
| kNN | 0.526 (20) | 0.509 (5) | 0.499 (20) |
| SVM | 0.489 (20) | 0.507 (10) | 0.504 (10) |
| GNB | 0.492 (10) | 0.505 (5) | 0.489 (all) |

### Arousal

| Model | EEG | Face | Multimodal |
|---|---:|---:|---:|
| kNN | 0.508 (5) | 0.492 (5) | 0.509 (5) |
| SVM | 0.509 (10) | 0.502 (5) | 0.507 (10) |
| GNB | 0.501 (5) | 0.509 (5) | 0.511 (5) |

Parentheses identify feature-count setting; `all` means No feature selection.

## Winning cells

### Valence

- eeg/gnb, 10: mean 0.492, SD 0.051, median 0.505, range 0.391–0.559; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- face/gnb, 5: mean 0.505, SD 0.020, median 0.504, range 0.475–0.537; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- multimodal/gnb, all: mean 0.489, SD 0.035, median 0.493, range 0.412–0.532; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- eeg/knn, 20: mean 0.526, SD 0.048, median 0.541, range 0.444–0.574; frequent tuned parameters `{"classifier__n_neighbors": 5, "classifier__weights": "uniform"}`.
- face/knn, 5: mean 0.509, SD 0.042, median 0.510, range 0.449–0.577; frequent tuned parameters `{"classifier__n_neighbors": 11, "classifier__weights": "uniform"}`.
- multimodal/knn, 20: mean 0.499, SD 0.041, median 0.500, range 0.410–0.558; frequent tuned parameters `{"classifier__n_neighbors": 3, "classifier__weights": "uniform"}`.
- eeg/svm, 20: mean 0.489, SD 0.061, median 0.491, range 0.425–0.642; frequent tuned parameters `{"classifier__C": 1, "classifier__gamma": 0.1, "classifier__kernel": "rbf"}`.
- face/svm, 10: mean 0.507, SD 0.038, median 0.500, range 0.474–0.613; frequent tuned parameters `{"classifier__C": 10, "classifier__gamma": 0.01, "classifier__kernel": "rbf"}`.
- multimodal/svm, 10: mean 0.504, SD 0.021, median 0.500, range 0.477–0.561; frequent tuned parameters `{"classifier__C": 0.1, "classifier__gamma": "scale", "classifier__kernel": "rbf"}`.

### Arousal

- eeg/gnb, 5: mean 0.501, SD 0.042, median 0.501, range 0.438–0.590; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- face/gnb, 5: mean 0.509, SD 0.035, median 0.521, range 0.429–0.542; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- multimodal/gnb, 5: mean 0.511, SD 0.029, median 0.504, range 0.469–0.571; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- eeg/knn, 5: mean 0.508, SD 0.036, median 0.508, range 0.429–0.558; frequent tuned parameters `{"classifier__n_neighbors": 11, "classifier__weights": "uniform"}`.
- face/knn, 5: mean 0.492, SD 0.044, median 0.500, range 0.405–0.563; frequent tuned parameters `{"classifier__n_neighbors": 11, "classifier__weights": "distance"}`.
- multimodal/knn, 5: mean 0.509, SD 0.040, median 0.508, range 0.447–0.570; frequent tuned parameters `{"classifier__n_neighbors": 7, "classifier__weights": "uniform"}`.
- eeg/svm, 10: mean 0.509, SD 0.036, median 0.500, range 0.478–0.608; frequent tuned parameters `{"classifier__C": 1, "classifier__kernel": "linear"}`.
- face/svm, 5: mean 0.502, SD 0.018, median 0.500, range 0.457–0.526; frequent tuned parameters `{"classifier__C": 10, "classifier__gamma": "scale", "classifier__kernel": "rbf"}`.
- multimodal/svm, 10: mean 0.507, SD 0.028, median 0.500, range 0.469–0.556; frequent tuned parameters `{"classifier__C": 1, "classifier__gamma": 0.01, "classifier__kernel": "rbf"}`.

## Best configurations

- Overall valence: eeg/knn, 20: mean 0.526, SD 0.048, median 0.541, range 0.444–0.574; frequent tuned parameters `{"classifier__n_neighbors": 5, "classifier__weights": "uniform"}`.
- Overall arousal: multimodal/gnb, 5: mean 0.511, SD 0.029, median 0.504, range 0.469–0.571; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.
- EEG-only: eeg/knn, 20: mean 0.526, SD 0.048, median 0.541, range 0.444–0.574; frequent tuned parameters `{"classifier__n_neighbors": 5, "classifier__weights": "uniform"}`.
- Face-only: face/knn, 5: mean 0.509, SD 0.042, median 0.510, range 0.449–0.577; frequent tuned parameters `{"classifier__n_neighbors": 11, "classifier__weights": "uniform"}`.
- Multimodal: multimodal/gnb, 5: mean 0.511, SD 0.029, median 0.504, range 0.469–0.571; frequent tuned parameters `{"classifier__var_smoothing": 1e-11}`.

## Held-out participant performance

- Best valence: P10: 0.566, P11: 0.574, P19: 0.562, P22: 0.536, P25: 0.444, P32: 0.491, P40: 0.571, P42: 0.454, P43: 0.521, P45: 0.546
- Best arousal: P10: 0.512, P11: 0.500, P19: 0.484, P22: 0.507, P25: 0.501, P32: 0.531, P40: 0.571, P42: 0.534, P43: 0.469, P45: 0.500

## Feature-selection stability

### EEG valence
- eeg_se__Pz: 90% (9/10)
- eeg_se_delta__C3: 90% (9/10)
- eeg_bp_beta__C3: 80% (8/10)
- eeg_bp_delta__PO7: 80% (8/10)
- eeg_hm__Pz: 80% (8/10)
- eeg_mf_hz__C3: 70% (7/10)
- eeg_se_alpha__Pz: 70% (7/10)
- eeg_hm__Oz: 50% (5/10)
- eeg_bp_beta__Fz: 40% (4/10)
- eeg_hm__C3: 40% (4/10)

### EEG arousal
- eeg_hm__Fz: 80% (8/10)
- eeg_sd__Oz: 80% (8/10)
- eeg_hc__Pz: 70% (7/10)
- eeg_bp_theta__Oz: 60% (6/10)
- eeg_bp_theta__Fz: 40% (4/10)
- eeg_mf_hz__Pz: 40% (4/10)
- eeg_bp_delta__Oz: 20% (2/10)
- eeg_hm__Pz: 20% (2/10)
- eeg_bp_delta__Cz: 10% (1/10)
- eeg_bp_gamma__Fz: 10% (1/10)

### Face valence
- video_irisdo_norm_mean: 100% (10/10)
- video_enso_norm_std: 90% (9/10)
- video_mnso_norm_mean: 90% (9/10)
- video_eso_norm_mean: 80% (8/10)
- video_mwo_norm_std: 70% (7/10)
- video_enso_norm_mean: 30% (3/10)
- video_irisdo_norm_std: 30% (3/10)
- video_eso_norm_std: 10% (1/10)
- video_mnso_norm_std: 0% (0/10)
- video_mwo_norm_mean: 0% (0/10)

### Face arousal
- video_mnso_norm_std: 100% (10/10)
- video_enso_norm_std: 90% (9/10)
- video_irisdo_norm_mean: 90% (9/10)
- video_enso_norm_mean: 80% (8/10)
- video_eso_norm_std: 80% (8/10)
- video_irisdo_norm_std: 30% (3/10)
- video_eso_norm_mean: 10% (1/10)
- video_mnso_norm_mean: 10% (1/10)
- video_mwo_norm_mean: 10% (1/10)
- video_mwo_norm_std: 0% (0/10)

### Multimodal valence
- eeg_bp_alpha__C3: 100% (10/10)
- eeg_bp_alpha__C4: 100% (10/10)
- eeg_bp_alpha__Cz: 100% (10/10)
- eeg_bp_alpha__Fz: 100% (10/10)
- eeg_bp_alpha__Oz: 100% (10/10)
- eeg_bp_alpha__PO7: 100% (10/10)
- eeg_bp_alpha__PO8: 100% (10/10)
- eeg_bp_alpha__Pz: 100% (10/10)
- eeg_bp_beta__C3: 100% (10/10)
- eeg_bp_beta__C4: 100% (10/10)

### Multimodal arousal
- video_irisdo_norm_mean: 90% (9/10)
- eeg_hm__Fz: 80% (8/10)
- eeg_sd__Oz: 80% (8/10)
- eeg_bp_theta__Oz: 40% (4/10)
- eeg_hc__Pz: 40% (4/10)
- eeg_mf_hz__Pz: 40% (4/10)
- eeg_bp_delta__Oz: 20% (2/10)
- eeg_bp_theta__Fz: 20% (2/10)
- eeg_hm__Pz: 20% (2/10)
- eeg_bp_gamma__Fz: 10% (1/10)

## Interpretation

These are exploratory unseen-participant results, not significance tests. Values above 0.50 are descriptive performance above balanced-accuracy chance; held-out participant ranges show consistency. Multimodal is not assumed superior when its best mean is below a single-modality result.
