# P19 Experiment C: Preprocessing Harmonization

## Branch decision

`IMAGE_EQUIVALENT_ICA_NOT_REPRODUCIBLE_FOR_CONTINUOUS_ROBOT_DATA`

C0 is a reproducible harmonized preprocessing baseline without ICA: raw BDF -> Image channel mapping -> CAR -> Image-configured 1–40 Hz fourth-order Butterworth IIR -> task-local 500-sample/2.0-s windows at 1.0-s step -> direct Image feature extraction. It is not claimed to be perfectly Image-matched.

## Why C1 was not implemented

The Image code uses MNE FastICA, seed 42, rank-aware component count, fit once on concatenated unbaselined valid Image stimulus epochs after CAR/filtering. For each Image epoch it then identifies positive ACC X/Y/Z-correlated components relative to that epoch's component-correlation mean plus 2 SD and high-pass filters those source signals at 3 Hz before reconstruction. This fit corpus and trial-specific decision rule depend on Image stimulus epochs; applying it to continuous Robot tasks would require inventing a non-equivalent segmentation and component-selection rule.

## Representation comparison

| Representation | Target | Task | Median nearest Image distance | % beyond Image reference max | Median selected |z| | 95th selected |z| | Mean HIGH score | % predicted HIGH |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A_existing_robot_features | valence | Pick and Place | 19.015 | 97.6 | 3.706 | 122.161 | 0.000 | 0.0 |
| A_existing_robot_features | valence | Shape Sorter Observation | 20.089 | 93.2 | 3.577 | 83.933 | 0.001 | 0.0 |
| A_existing_robot_features | valence | Stack | 29.312 | 100.0 | 2.924 | 103.937 | 0.000 | 0.0 |
| A_existing_robot_features | valence | Sisyphus | 30.438 | 98.9 | 2.189 | 84.090 | 0.000 | 0.0 |
| A_existing_robot_features | valence | Shape Sorter Interaction | 1.199 | 16.9 | 1.082 | 8.366 | 0.622 | 66.2 |
| A_existing_robot_features | arousal | Pick and Place | 9.514 | 100.0 | 2.543 | 18.912 | 0.890 | 100.0 |
| A_existing_robot_features | arousal | Shape Sorter Observation | 12.758 | 94.9 | 2.610 | 16.389 | 0.888 | 98.3 |
| A_existing_robot_features | arousal | Stack | 14.189 | 100.0 | 2.538 | 18.656 | 0.874 | 100.0 |
| A_existing_robot_features | arousal | Sisyphus | 14.495 | 100.0 | 2.359 | 17.093 | 0.852 | 98.9 |
| A_existing_robot_features | arousal | Shape Sorter Interaction | 5.450 | 89.6 | 1.281 | 9.439 | 0.359 | 27.3 |
| C0_reproducible_harmonized_no_ICA | valence | Pick and Place | 0.788 | 14.3 | 0.805 | 10.523 | 0.654 | 71.4 |
| C0_reproducible_harmonized_no_ICA | valence | Shape Sorter Observation | 0.915 | 6.8 | 0.796 | 4.989 | 0.527 | 55.9 |
| C0_reproducible_harmonized_no_ICA | valence | Stack | 0.994 | 7.4 | 0.981 | 4.300 | 0.436 | 39.7 |
| C0_reproducible_harmonized_no_ICA | valence | Sisyphus | 1.143 | 8.8 | 1.179 | 9.473 | 0.361 | 35.2 |
| C0_reproducible_harmonized_no_ICA | valence | Shape Sorter Interaction | 1.116 | 10.4 | 1.378 | 5.073 | 0.230 | 22.1 |
| C0_reproducible_harmonized_no_ICA | arousal | Pick and Place | 2.307 | 23.8 | 0.705 | 10.529 | 0.324 | 19.0 |
| C0_reproducible_harmonized_no_ICA | arousal | Shape Sorter Observation | 2.238 | 20.3 | 0.862 | 4.731 | 0.432 | 35.6 |
| C0_reproducible_harmonized_no_ICA | arousal | Stack | 3.036 | 41.2 | 0.990 | 4.267 | 0.325 | 19.1 |
| C0_reproducible_harmonized_no_ICA | arousal | Sisyphus | 5.185 | 82.4 | 0.947 | 8.899 | 0.282 | 12.1 |
| C0_reproducible_harmonized_no_ICA | arousal | Shape Sorter Interaction | 5.416 | 89.6 | 1.128 | 8.689 | 0.370 | 28.6 |

## Reproducibility artifact

`P19_C0_harmonized_window_features.csv` is retained inside this Experiment C output only because the numeric comparison, frozen inference, and per-feature summary must be reproducible from the exact regenerated C0 representation. It is not a canonical derived Robot run.

## Interpretation boundary

Use domain geometry and feature distributions—not probability aesthetics—to assess whether C0 changes cross-context proximity. C1 is absent by scientific design; no ICA comparison is implied.
