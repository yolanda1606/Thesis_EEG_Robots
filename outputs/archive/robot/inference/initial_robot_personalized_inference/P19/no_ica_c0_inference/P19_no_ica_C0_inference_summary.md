# P19 Matched No-ICA Image-to-C0 Robot Inference

P19 Image no-ICA models were selected from the completed Image-only comparison, tuned/refit only on all usable P19 no-ICA Image trials, then frozen before application to the existing C0 Robot feature table. No Image/Robot feature extraction was rerun and no Robot data was used for fitting, scaling, selection, or tuning.

## Frozen no-ICA calibration

- Valence: EEG -> StandardScaler -> SelectKBest(k=10) -> gnb; final Image-only internal CV BA=0.541; parameters `{'classifier__var_smoothing': 1e-11}`; selected features: eeg_se__Fz, eeg_se__PO8, eeg_hm__PO8, eeg_hc__Fz, eeg_hc__PO8, eeg_mf_hz__Fz, eeg_mf_hz__PO8, eeg_bp_delta__Fz, eeg_bp_delta__PO8, eeg_se_gamma__PO7.
- Arousal: EEG -> StandardScaler -> SelectKBest(k=10) -> knn; final Image-only internal CV BA=0.649; parameters `{'classifier__n_neighbors': 5, 'classifier__weights': 'uniform'}`; selected features: eeg_se__Fz, eeg_se__PO8, eeg_hm__Fz, eeg_hm__PO8, eeg_hc__Fz, eeg_hc__PO8, eeg_mf_hz__Fz, eeg_bp_beta__Fz, eeg_bp_beta__PO8, eeg_bp_beta__Pz.

## C0 Robot application

| Target | Task | Median nearest Image distance | % beyond Image reference max | Mean HIGH score | % predicted HIGH |
|---|---|---:|---:|---:|---:|
| valence | Pick and Place | 2.058 | 16.7 | 0.664 | 69.0 |
| valence | Shape Sorter Interaction | 5.868 | 72.7 | 0.084 | 7.8 |
| valence | Shape Sorter Observation | 1.954 | 6.8 | 0.498 | 50.8 |
| valence | Sisyphus | 5.374 | 63.7 | 0.202 | 19.8 |
| valence | Stack | 2.757 | 25.0 | 0.331 | 32.4 |
| arousal | Pick and Place | 2.151 | 21.4 | 0.329 | 16.7 |
| arousal | Shape Sorter Interaction | 5.484 | 85.7 | 0.304 | 22.1 |
| arousal | Shape Sorter Observation | 2.203 | 11.9 | 0.403 | 35.6 |
| arousal | Sisyphus | 5.143 | 78.0 | 0.242 | 11.0 |
| arousal | Stack | 2.945 | 26.5 | 0.262 | 11.8 |

## Interpretation boundary

These are matched no-ICA Image-to-C0 Robot probabilities and feature-domain diagnostics, not Robot emotion ground truth or model confidence. They do not replace the canonical ICA-based transfer analysis.
