# P19 Feature Transferability and Feature-Domain Reliability

This is a read-only diagnostic of frozen P19 Image-calibrated models. It does not alter probabilities, fit a model, use robot ratings, establish confidence, or establish emotion validity.

## Methods

- Distribution overlap means intersection of robot and Image-class IQR intervals; it is a robust descriptive convention, not a test of equivalence.
- `TRANSFER_COMPATIBLE`, `MODERATELY_SHIFTED`, and `SEVERELY_SHIFTED` are exploratory multi-indicator labels. They use several z/range/overlap indicators and do not justify deleting a feature.
- Window labels describe feature-domain proximity to leave-self-out Image neighbours only: `within_reference`, `above_image_95pct`, and `beyond_image_reference`.

## Per-task feature transferability

### Valence
- Pick and Place — eeg_se__PO8: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_delta__Fz: SEVERELY_SHIFTED.
- Shape Sorter Observation — eeg_se__PO8: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_delta__Fz: SEVERELY_SHIFTED.
- Stack — eeg_se__PO8: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_delta__Fz: SEVERELY_SHIFTED.
- Sisyphus — eeg_se__PO8: MODERATELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: TRANSFER_COMPATIBLE, eeg_bp_delta__Fz: SEVERELY_SHIFTED.
- Shape Sorter Interaction — eeg_se__PO8: TRANSFER_COMPATIBLE, eeg_hc__PO8: TRANSFER_COMPATIBLE, eeg_mf_hz__Fz: TRANSFER_COMPATIBLE, eeg_mf_hz__PO8: TRANSFER_COMPATIBLE, eeg_bp_delta__Fz: MODERATELY_SHIFTED.

### Arousal
- Pick and Place — eeg_se__Fz: SEVERELY_SHIFTED, eeg_hm__Fz: SEVERELY_SHIFTED, eeg_hm__PO8: SEVERELY_SHIFTED, eeg_hc__Fz: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_beta__Fz: TRANSFER_COMPATIBLE, eeg_bp_beta__PO8: SEVERELY_SHIFTED, eeg_bp_beta__Pz: TRANSFER_COMPATIBLE.
- Shape Sorter Observation — eeg_se__Fz: SEVERELY_SHIFTED, eeg_hm__Fz: SEVERELY_SHIFTED, eeg_hm__PO8: SEVERELY_SHIFTED, eeg_hc__Fz: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_beta__Fz: TRANSFER_COMPATIBLE, eeg_bp_beta__PO8: MODERATELY_SHIFTED, eeg_bp_beta__Pz: TRANSFER_COMPATIBLE.
- Stack — eeg_se__Fz: SEVERELY_SHIFTED, eeg_hm__Fz: SEVERELY_SHIFTED, eeg_hm__PO8: SEVERELY_SHIFTED, eeg_hc__Fz: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: MODERATELY_SHIFTED, eeg_bp_beta__Fz: TRANSFER_COMPATIBLE, eeg_bp_beta__PO8: SEVERELY_SHIFTED, eeg_bp_beta__Pz: TRANSFER_COMPATIBLE.
- Sisyphus — eeg_se__Fz: SEVERELY_SHIFTED, eeg_hm__Fz: SEVERELY_SHIFTED, eeg_hm__PO8: TRANSFER_COMPATIBLE, eeg_hc__Fz: SEVERELY_SHIFTED, eeg_hc__PO8: SEVERELY_SHIFTED, eeg_mf_hz__Fz: MODERATELY_SHIFTED, eeg_mf_hz__PO8: TRANSFER_COMPATIBLE, eeg_bp_beta__Fz: TRANSFER_COMPATIBLE, eeg_bp_beta__PO8: SEVERELY_SHIFTED, eeg_bp_beta__Pz: TRANSFER_COMPATIBLE.
- Shape Sorter Interaction — eeg_se__Fz: MODERATELY_SHIFTED, eeg_hm__Fz: TRANSFER_COMPATIBLE, eeg_hm__PO8: MODERATELY_SHIFTED, eeg_hc__Fz: MODERATELY_SHIFTED, eeg_hc__PO8: TRANSFER_COMPATIBLE, eeg_mf_hz__Fz: TRANSFER_COMPATIBLE, eeg_mf_hz__PO8: TRANSFER_COMPATIBLE, eeg_bp_beta__Fz: TRANSFER_COMPATIBLE, eeg_bp_beta__PO8: SEVERELY_SHIFTED, eeg_bp_beta__Pz: TRANSFER_COMPATIBLE.

## Distance and likelihood drivers

- Pick and Place: arousal nearest-neighbour median distance is led by `eeg_hc__Fz` (51.0%; largest in 81.0% of windows). Valence likelihood is led by `eeg_bp_delta__Fz` (median |HIGH−LOW| log contribution 158; dominant in 100.0% of windows).
- Shape Sorter Observation: arousal nearest-neighbour median distance is led by `eeg_hc__Fz` (62.4%; largest in 84.7% of windows). Valence likelihood is led by `eeg_bp_delta__Fz` (median |HIGH−LOW| log contribution 172; dominant in 94.9% of windows).
- Stack: arousal nearest-neighbour median distance is led by `eeg_hc__Fz` (66.1%; largest in 86.8% of windows). Valence likelihood is led by `eeg_bp_delta__Fz` (median |HIGH−LOW| log contribution 334; dominant in 100.0% of windows).
- Sisyphus: arousal nearest-neighbour median distance is led by `eeg_hc__Fz` (55.6%; largest in 73.6% of windows). Valence likelihood is led by `eeg_bp_delta__Fz` (median |HIGH−LOW| log contribution 344; dominant in 100.0% of windows).
- Shape Sorter Interaction: arousal nearest-neighbour median distance is led by `eeg_bp_beta__PO8` (50.7%; largest in 79.2% of windows). Valence likelihood is led by `eeg_mf_hz__PO8` (median |HIGH−LOW| log contribution 1.18; dominant in 46.8% of windows).

## Window feature-domain reliability

| Target | within_reference | above_image_95pct | beyond_image_reference |
|---|---:|---:|---:|
| valence | 43 | 27 | 267 |
| arousal | 3 | 8 | 326 |

## Scaling-mismatch investigation

mixture/unclear: 100.0% of severely shifted target-feature pairs have a consistent median-offset direction across tasks, but task-level median |z| varies by 1.5x. This does not support a single global additive or multiplicative correction.
The extremely large Image-scaled z-scores require preprocessing/provenance investigation, but this analysis cannot distinguish a unit mismatch from context effects conclusively and applies no correction.

## Interpretation boundary

Shape Sorter Interaction may be closer for a subset of selected features, but arousal distance diagnostics still place most of its windows beyond the Image reference maximum. No feature subset, rescaling, or reduced model is recommended or implemented here.
