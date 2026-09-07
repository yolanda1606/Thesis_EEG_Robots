# P19 Image-to-Robot EEG Inference Pilot

## Calibration

- **Valence**: EEG -> StandardScaler -> SelectKBest(k=5) -> gnb. 
  Internal Image-only CV selected `{"classifier__var_smoothing": 1e-11}` (BA 0.559).
  Image calibration trials: 120 (LOW=43, HIGH=77).
  Selected features: eeg_se__PO8, eeg_hc__PO8, eeg_mf_hz__PO8, eeg_mf_hz__Fz, eeg_bp_delta__Fz.
  Exact saved nested-CV ties: 4; EEG/GaussianNB/Top-5 tied exactly in mean nested-CV balanced accuracy with EEG/GaussianNB/Top-10 and multimodal counterparts. The single-modality rule removes multimodal variants; the smaller-feature-set rule selects Top-5. This also matches stage_b_best_per_participant.csv.
- **Arousal**: EEG -> StandardScaler -> SelectKBest(k=10) -> knn. 
  Internal Image-only CV selected `{"classifier__n_neighbors": 11, "classifier__weights": "distance"}` (BA 0.676).
  Image calibration trials: 120 (LOW=65, HIGH=55).
  Selected features: eeg_bp_beta__PO8, eeg_bp_beta__Fz, eeg_hc__Fz, eeg_hm__Fz, eeg_se__Fz, eeg_hc__PO8, eeg_mf_hz__Fz, eeg_hm__PO8, eeg_mf_hz__PO8, eeg_bp_beta__Pz.
  Exact saved nested-CV ties: 4; EEG/kNN/Top-10 is the pre-specified representative among exact mean-BA ties, which include EEG/SVM/Top-5 and multimodal variants. The single-modality rule removes multimodal variants; the requested representative retains EEG/kNN/Top-10. It did not numerically outperform the tied alternatives.

## Robot application

- Analysed 337 existing 2.0 s / 1.0 s-step EEG windows across five tasks.
- Shape Sorter Alone was intentionally excluded: no Status anchors, filename-fallback timing, BDF/header clock conflict, and no explicit usable task/video interval.
- `valence_high_score` and `arousal_high_score` are high-class probabilities: how strongly a window resembles the respective HIGH class learned from P19's Image Experiment. They are not continuous emotion levels.
- Robot windows have no continuous emotion ground truth. Post-task ratings are descriptive only and were not used for fitting, tuning, or inference.
- Initial P19 transfer-pilot limitation: the existing robot feature table has incomplete ICA provenance and its exact producing source version cannot currently be independently reconstructed. Robot preprocessing is not claimed to be proven perfectly equivalent to the Image preprocessing pipeline.
