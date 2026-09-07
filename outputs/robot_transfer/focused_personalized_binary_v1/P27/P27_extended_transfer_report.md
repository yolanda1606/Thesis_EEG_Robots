# P27 final transfer extended descriptive report

All quantities are post-inference summaries of saved predictions. Robot ratings are retained on their original 1–7 scale and used only for descriptive HIGH/LOW comparison; P(HIGH) is not mapped to that scale.

## Task-level consensus and agreement

| Task     | Robot condition   | Target   |   Median P(HIGH) |   Mean P(HIGH) |   Consensus windows ≥0.5 | Verdict   |   Mean range |   Median range |   Mean pairwise |ΔP| |   Unanimous LOW |   Unanimous HIGH |   Unanimous |   Majority HIGH |   Self-report (1–7) | Rating class   | Verdict match   |
|:---------|:------------------|:---------|-----------------:|---------------:|-------------------------:|:----------|-------------:|---------------:|---------------------:|----------------:|-----------------:|------------:|----------------:|--------------------:|:---------------|:----------------|
| PnP      | SLOW              | arousal  |         0.663684 |       0.637261 |                 0.756098 | HIGH      |     0.432357 |       0.420701 |             0.288238 |       0.097561  |        0.195122  |    0.292683 |       0.560976  |                   7 | HIGH           | True            |
| SSAlone  | N/A               | arousal  |         0.403646 |       0.402719 |                 0.137931 | LOW       |     0.608206 |       0.521934 |             0.405471 |       0.413793  |        0         |    0.413793 |       0.0344828 |                   4 | HIGH           | False           |
| SSInt    | N/A               | arousal  |         0.523153 |       0.565852 |                 0.642105 | HIGH      |     0.47873  |       0.450663 |             0.319153 |       0.294737  |        0.105263  |    0.4      |       0.378947  |                   7 | HIGH           | True            |
| SSObs    | FAST              | arousal  |         0.769052 |       0.706346 |                 0.8      | HIGH      |     0.440251 |       0.427921 |             0.293501 |       0.1       |        0.383333  |    0.483333 |       0.65      |                   6 | HIGH           | True            |
| Sisyphus | N/A               | arousal  |         0.515156 |       0.503387 |                 0.535433 | HIGH      |     0.554861 |       0.508053 |             0.369908 |       0.15748   |        0.0708661 |    0.228346 |       0.362205  |                   7 | HIGH           | True            |
| St       | FAULTY            | arousal  |         0.74171  |       0.725083 |                 0.885714 | HIGH      |     0.44508  |       0.421527 |             0.29672  |       0.0714286 |        0.442857  |    0.514286 |       0.714286  |                   7 | HIGH           | True            |
| PnP      | SLOW              | valence  |         0.554054 |       0.51774  |                 0.707317 | HIGH      |     0.292391 |       0.269918 |             0.194927 |       0.219512  |        0.414634  |    0.634146 |       0.707317  |                   6 | HIGH           | True            |
| SSAlone  | N/A               | valence  |         0.494727 |       0.433168 |                 0.413793 | LOW       |     0.289753 |       0.342371 |             0.193169 |       0.413793  |        0.275862  |    0.689655 |       0.413793  |                   6 | HIGH           | False           |
| SSInt    | N/A               | valence  |         0.488628 |       0.459207 |                 0.484211 | LOW       |     0.323275 |       0.322912 |             0.215516 |       0.463158  |        0.273684  |    0.736842 |       0.484211  |                   5 | HIGH           | False           |
| SSObs    | FAST              | valence  |         0.532957 |       0.51211  |                 0.616667 | HIGH      |     0.328521 |       0.339559 |             0.219014 |       0.266667  |        0.366667  |    0.633333 |       0.616667  |                   6 | HIGH           | True            |
| Sisyphus | N/A               | valence  |         0.436885 |       0.365487 |                 0.425197 | LOW       |     0.276847 |       0.314506 |             0.184565 |       0.527559  |        0.204724  |    0.732283 |       0.425197  |                   7 | HIGH           | False           |
| St       | FAULTY            | valence  |         0.463563 |       0.444091 |                 0.342857 | LOW       |     0.380397 |       0.414732 |             0.253598 |       0.514286  |        0.1       |    0.614286 |       0.342857  |                   5 | HIGH           | False           |

## Historical versus final P27 transfer

This is a read-only comparison. It explains documented design and provenance differences; it does not seek to reproduce historical trajectories.

### Model selection and selected features

The historical run selected Stage-B representatives and then prepared/refit deployment models. The final run uses the frozen focused EEG-only top-3 table and does not select or refit transfer models.

- valence R1: historical `eeg svm all` → final `EEG logreg all_eeg (8 selected)`.
  - Historical selected features (120): All 120 frozen EEG features (full list retained in source manifest).
  - Final selected features (8): eeg_se__C3; eeg_se__Cz; eeg_hm__Oz; eeg_bp_delta__Cz; eeg_bp_delta__Oz; eeg_se_theta__Cz; eeg_se_alpha__C4; eeg_se_beta__Oz
- valence R2: historical `multimodal svm 10` → final `EEG logreg all_eeg (10 selected)`.
  - Historical selected features (10): eeg_se__C3; eeg_hm__C3; eeg_hm__Oz; eeg_hc__Cz; eeg_mf_hz__C3; eeg_bp_delta__C3; eeg_se_delta__C3; eeg_se_theta__Cz; eeg_se_alpha__C4; eeg_se_beta__Oz
  - Final selected features (10): eeg_se__C3; eeg_se__Cz; eeg_hm__Oz; eeg_hc__Cz; eeg_hc__Oz; eeg_bp_delta__Cz; eeg_bp_delta__Oz; eeg_se_theta__Cz; eeg_se_alpha__C4; eeg_se_beta__Oz
- valence R3: historical `eeg svm 20` → final `EEG gnb hjorth (8 selected)`.
  - Historical selected features (20): eeg_se__C3; eeg_se__Cz; eeg_hm__C3; eeg_hm__Oz; eeg_hc__C3; eeg_hc__Cz; eeg_hc__Oz; eeg_hc__Pz; eeg_mf_hz__C3; eeg_bp_delta__C3; eeg_bp_delta__Cz; eeg_bp_delta__Oz; eeg_se_delta__C3; eeg_se_delta__PO7; eeg_se_theta__Cz; eeg_se_theta__Fz; eeg_bp_alpha__C4; eeg_se_alpha__C4; eeg_se_beta__C3; eeg_se_beta__Oz
  - Final selected features (8): eeg_hm__C3; eeg_hm__Cz; eeg_hm__Oz; eeg_hm__Pz; eeg_hc__C4; eeg_hc__Cz; eeg_hc__Oz; eeg_hc__Pz
- arousal R1: historical `multimodal knn 5` → final `EEG gnb paper_compact (10 selected)`.
  - Historical selected features (5): eeg_hm__C4; eeg_bp_gamma__C4; eeg_bp_gamma__Oz; eeg_se_gamma__C4; video_eso_norm_std
  - Final selected features (10): eeg_sd__PO8; eeg_hm__C4; eeg_bp_beta__PO7; eeg_bp_beta__PO8; eeg_se_beta__C4; eeg_bp_gamma__C4; eeg_bp_gamma__Oz; eeg_bp_gamma__PO7; eeg_bp_gamma__PO8; eeg_bp_gamma__Pz
- arousal R2: historical `multimodal gnb 10` → final `EEG logreg paper_compact (10 selected)`.
  - Historical selected features (10): eeg_se__C3; eeg_se__C4; eeg_hm__C4; eeg_se_beta__C4; eeg_bp_gamma__C4; eeg_bp_gamma__Oz; eeg_se_gamma__C4; video_irisdo_norm_std; video_eso_norm_std; video_enso_norm_std
  - Final selected features (10): eeg_sd__Cz; eeg_hm__C4; eeg_mf_hz__C3; eeg_bp_beta__C3; eeg_bp_beta__C4; eeg_se_beta__C4; eeg_se_beta__Fz; eeg_bp_gamma__C3; eeg_bp_gamma__C4; eeg_se_gamma__C4
- arousal R3: historical `eeg knn 5` → final `EEG svm all_eeg (10 selected)`.
  - Historical selected features (5): eeg_se__C3; eeg_hm__C4; eeg_bp_gamma__C4; eeg_bp_gamma__Oz; eeg_se_gamma__C4
  - Final selected features (10): eeg_hm__C4; eeg_se_delta__Fz; eeg_bp_alpha__C4; eeg_se_alpha__PO8; eeg_bp_beta__PO7; eeg_bp_gamma__C4; eeg_bp_gamma__Oz; eeg_bp_gamma__PO7; eeg_bp_gamma__PO8; eeg_bp_gamma__Pz

### Robot input provenance and preprocessing

- Historical ICA input: `derived/P27/Robot_Experiment/runs/p27_robot_continuous/features/eeg_window_features.csv`; it also supplied `derived/P27/Robot_Experiment/runs/p27_robot_continuous/features/video_window_features.csv` for Multimodal models.
- Historical no-ICA input: `derived/P27/Robot_Experiment/runs/p27_robot_continuous_no_ica/features/eeg_window_features.csv`; it also supplied `derived/P27/Robot_Experiment/runs/p27_robot_continuous/features/video_window_features.csv` for Multimodal models.
- Final input: `/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/derived/P27/Robot_Experiment/runs/p27_robot_final_features/features/eeg_window_features.csv` only. It is the canonical final no-ICA Robot EEG feature table; no video features or Face/Multimodal models are used.
- The final output has 422 windows for every model. Historical Multimodal P27 rows had 371 windows per condition, while its EEG-only rows had 422, because Multimodal inference required matching video windows.

### SVM probability handling

- Historical P27 valence models were SVC artifacts serialized with `probability=True`, after the historical deployment-model preparation/refit workflow.
- Final P27 contains one SVM: arousal R3. Its original frozen artifact is `SVC(probability=False)` and retains its hard class. Its P(HIGH) comes from the separate Image-only cross-validated Platt-calibrated wrapper; Robot data and ratings did not participate in calibration.

These differences in model families, modalities, fixed selected features, source Image run, Robot input condition, available windows, and SVM probability construction make trajectory changes expected. They are not evidence that either result should be transformed to reproduce the other.
