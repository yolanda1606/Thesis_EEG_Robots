# ICA vs no-ICA: Full Individual Image-Classifier Accuracy Audit

## Scope and provenance

This audit includes only participant-specific (`individual`) binary Image classifiers. It excludes general and LOSO classifiers. The historical combined directory was inspected; it is a presentation-only aggregation and contains only `combined_best_per_participant.csv` plus figures and a summary, not a full result table, folds, or manifest. The fair ICA source is therefore the reconstruction of its two documented source runs: `stage_b_individual_v1` (28 participants) and `stage_c_individual_added18_v1` (18 participants).

The reconstructed ICA table and the no-ICA `results_summary.csv` each have 3,312 unique configurations: 46 participants × 2 targets × 3 modalities × 3 classifier families × 4 feature-count requests. Every configuration matches one-to-one on participant, target, modality, classifier, and `feature_count_request`.

## Comparability verification

| Item | ICA | no-ICA | Result |
|---|---|---|---|
| Participants | P01–P46 (28 + 18 source runs) | P01–P46 | Same 46 participants |
| Targets | valence, arousal | valence, arousal | Same |
| Modalities | EEG, face, multimodal | EEG, face, multimodal | Same |
| Classifier families | KNN, SVM, Gaussian NB | KNN, SVM, Gaussian NB | Same |
| Feature-count requests | all, 5, 10, 20 | all, 5, 10, 20 | Same |
| Labels | LOW < 4; HIGH ≥ 4 | LOW < 4; HIGH ≥ 4 | Same |
| CV | nested StratifiedKFold | nested StratifiedKFold | Same |
| Seed | 42 | 42 | Same |
| BA / accuracy means | mean across five outer-fold values in `train_classification.py` | Same | Same definition |

All manifest settings above are identical: True. In `train_classification.py`, `balanced_accuracy_mean` and `accuracy_mean` are the arithmetic means of the five corresponding outer-fold metrics; `balanced_accuracy_sd` is the sample SD of those five folds.

### Retention caveat

The available merged tables have identical row counts for all 46 participants (54–120 rows each); trigger multisets and non-missing valence/arousal-rating counts also agree for every participant (rows=True, triggers=True, labels=True). This supports matched trial identities at the final merged-table level. It does not prove that ICA had no intermediate AutoReject or preprocessing effect: the run manifests do not provide per-condition retained-epoch logs or a trial-by-trial preprocessing lineage. Interpret differences as a matched pipeline-condition comparison, not strict evidence that ICA alone caused them.

## A. Matched configuration comparison

Delta = no-ICA minus ICA. Positive values favor no-ICA. “Unchanged” is the descriptive ±0.02 band, not a statistical equivalence test.

### Valence — eeg

- Balanced accuracy: n=552; mean -0.003; median -0.004; SD 0.047; range -0.162 to +0.193; improved/unchanged/worsened 165/194/193 (29.9%/35.1%/35.0%).
- Ordinary accuracy: n=552; mean -0.003; median -0.004; SD 0.047; range -0.153 to +0.182; improved/unchanged/worsened 166/188/198 (30.1%/34.1%/35.9%).

### Valence — multimodal

- Balanced accuracy: n=552; mean -0.001; median +0.000; SD 0.045; range -0.168 to +0.133; improved/unchanged/worsened 161/235/156 (29.2%/42.6%/28.3%).
- Ordinary accuracy: n=552; mean -0.002; median +0.000; SD 0.044; range -0.164 to +0.124; improved/unchanged/worsened 152/234/166 (27.5%/42.4%/30.1%).

### Valence — face

- Balanced accuracy: n=552; mean +0.000; median +0.000; SD 0.000; range +0.000 to +0.000; improved/unchanged/worsened 0/552/0 (0.0%/100.0%/0.0%).
- Ordinary accuracy: n=552; mean +0.000; median +0.000; SD 0.000; range +0.000 to +0.000; improved/unchanged/worsened 0/552/0 (0.0%/100.0%/0.0%).

### Arousal — eeg

- Balanced accuracy: n=552; mean -0.000; median +0.000; SD 0.044; range -0.159 to +0.140; improved/unchanged/worsened 162/221/169 (29.3%/40.0%/30.6%).
- Ordinary accuracy: n=552; mean -0.001; median -0.000; SD 0.045; range -0.160 to +0.176; improved/unchanged/worsened 165/226/161 (29.9%/40.9%/29.2%).

### Arousal — multimodal

- Balanced accuracy: n=552; mean +0.001; median +0.000; SD 0.045; range -0.159 to +0.160; improved/unchanged/worsened 174/216/162 (31.5%/39.1%/29.3%).
- Ordinary accuracy: n=552; mean -0.000; median -0.002; SD 0.046; range -0.158 to +0.215; improved/unchanged/worsened 165/216/171 (29.9%/39.1%/31.0%).

### Arousal — face

- Balanced accuracy: n=552; mean +0.000; median +0.000; SD 0.000; range +0.000 to +0.000; improved/unchanged/worsened 0/552/0 (0.0%/100.0%/0.0%).
- Ordinary accuracy: n=552; mean +0.000; median +0.000; SD 0.000; range +0.000 to +0.000; improved/unchanged/worsened 0/552/0 (0.0%/100.0%/0.0%).

## B. Best-per-participant comparison

For each condition independently, the exact historical rule selects the largest `balanced_accuracy_mean` within each participant × target; exact BA ties are broken by alphabetical modality, alphabetical classifier, then feature-count order all, 5, 10, 20. This is descriptive model selection, not an independent deployment estimate.

### All modalities

| Participant | Target | ICA best modality | ICA best model | ICA BA | ICA accuracy | no-ICA best modality | no-ICA best model | no-ICA BA | no-ICA accuracy | BA delta | Accuracy delta |
|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| P01 | arousal | face | svm (all) | 0.607 | 0.651 | multimodal | knn (5) | 0.636 | 0.662 | +0.029 | +0.011 |
| P01 | valence | eeg | svm (5) | 0.583 | 0.605 | face | gnb (5) | 0.574 | 0.558 | -0.008 | -0.047 |
| P02 | arousal | multimodal | knn (all) | 0.583 | 0.634 | eeg | svm (all) | 0.526 | 0.560 | -0.057 | -0.074 |
| P02 | valence | eeg | gnb (5) | 0.647 | 0.682 | eeg | gnb (20) | 0.662 | 0.702 | +0.015 | +0.020 |
| P03 | arousal | face | gnb (all) | 0.575 | 0.593 | face | gnb (all) | 0.575 | 0.593 | +0.000 | +0.000 |
| P03 | valence | eeg | gnb (5) | 0.671 | 0.716 | eeg | gnb (5) | 0.656 | 0.696 | -0.014 | -0.020 |
| P04 | arousal | eeg | knn (20) | 0.607 | 0.613 | eeg | knn (all) | 0.611 | 0.640 | +0.004 | +0.027 |
| P04 | valence | eeg | svm (5) | 0.633 | 0.674 | face | svm (all) | 0.622 | 0.667 | -0.011 | -0.007 |
| P05 | arousal | eeg | svm (20) | 0.572 | 0.600 | eeg | svm (20) | 0.601 | 0.635 | +0.029 | +0.035 |
| P05 | valence | face | svm (all) | 0.603 | 0.617 | face | svm (all) | 0.603 | 0.617 | +0.000 | +0.000 |
| P06 | arousal | multimodal | svm (all) | 0.629 | 0.643 | multimodal | svm (10) | 0.678 | 0.680 | +0.049 | +0.037 |
| P06 | valence | face | gnb (all) | 0.634 | 0.633 | face | gnb (all) | 0.634 | 0.633 | +0.000 | +0.000 |
| P07 | arousal | multimodal | gnb (all) | 0.541 | 0.592 | face | gnb (5) | 0.539 | 0.842 | -0.002 | +0.250 |
| P07 | valence | eeg | gnb (10) | 0.537 | 0.546 | face | gnb (5) | 0.534 | 0.680 | -0.003 | +0.134 |
| P08 | arousal | eeg | svm (5) | 0.587 | 0.587 | eeg | gnb (all) | 0.560 | 0.555 | -0.027 | -0.033 |
| P08 | valence | eeg | knn (10) | 0.595 | 0.597 | eeg | knn (all) | 0.552 | 0.555 | -0.042 | -0.042 |
| P09 | arousal | eeg | gnb (10) | 0.625 | 0.722 | multimodal | gnb (all) | 0.660 | 0.743 | +0.035 | +0.021 |
| P09 | valence | face | knn (5) | 0.598 | 0.642 | face | knn (5) | 0.598 | 0.642 | +0.000 | +0.000 |
| P10 | arousal | face | knn (all) | 0.570 | 0.575 | face | knn (all) | 0.570 | 0.575 | +0.000 | +0.000 |
| P10 | valence | face | knn (all) | 0.571 | 0.583 | face | knn (all) | 0.571 | 0.583 | +0.000 | +0.000 |
| P11 | arousal | face | knn (all) | 0.582 | 0.608 | face | knn (all) | 0.582 | 0.608 | +0.000 | +0.000 |
| P11 | valence | face | gnb (5) | 0.616 | 0.608 | face | gnb (5) | 0.616 | 0.608 | +0.000 | +0.000 |
| P12 | arousal | eeg | gnb (all) | 0.500 | 0.967 | eeg | gnb (all) | 0.500 | 0.967 | +0.000 | +0.000 |
| P12 | valence | multimodal | knn (10) | 0.547 | 0.692 | multimodal | gnb (all) | 0.547 | 0.633 | +0.000 | -0.058 |
| P13 | arousal | eeg | svm (10) | 0.594 | 0.604 | multimodal | svm (20) | 0.658 | 0.663 | +0.064 | +0.059 |
| P13 | valence | face | gnb (all) | 0.595 | 0.581 | face | gnb (all) | 0.595 | 0.581 | +0.000 | +0.000 |
| P14 | arousal | multimodal | knn (5) | 0.592 | 0.604 | multimodal | svm (20) | 0.581 | 0.598 | -0.011 | -0.005 |
| P14 | valence | multimodal | gnb (all) | 0.598 | 0.586 | eeg | knn (20) | 0.604 | 0.608 | +0.006 | +0.022 |
| P15 | arousal | face | gnb (5) | 0.588 | 0.725 | multimodal | gnb (all) | 0.621 | 0.742 | +0.033 | +0.017 |
| P15 | valence | face | gnb (5) | 0.613 | 0.675 | multimodal | svm (20) | 0.620 | 0.678 | +0.007 | +0.003 |
| P16 | arousal | eeg | svm (20) | 0.674 | 0.701 | multimodal | knn (20) | 0.630 | 0.663 | -0.044 | -0.038 |
| P16 | valence | face | svm (5) | 0.587 | 0.600 | multimodal | svm (10) | 0.608 | 0.612 | +0.020 | +0.012 |
| P17 | arousal | eeg | gnb (all) | 0.582 | 0.543 | eeg | knn (10) | 0.619 | 0.756 | +0.037 | +0.213 |
| P17 | valence | face | svm (all) | 0.688 | 0.700 | face | svm (all) | 0.688 | 0.700 | +0.000 | +0.000 |
| P18 | arousal | face | svm (5) | 0.602 | 0.600 | face | svm (5) | 0.602 | 0.600 | +0.000 | +0.000 |
| P18 | valence | eeg | knn (20) | 0.601 | 0.604 | eeg | gnb (20) | 0.634 | 0.635 | +0.034 | +0.032 |
| P19 | arousal | eeg | knn (10) | 0.683 | 0.692 | eeg | knn (10) | 0.657 | 0.667 | -0.026 | -0.025 |
| P19 | valence | eeg | gnb (5) | 0.600 | 0.642 | multimodal | gnb (10) | 0.620 | 0.658 | +0.020 | +0.017 |
| P20 | arousal | eeg | gnb (10) | 0.594 | 0.640 | multimodal | knn (10) | 0.585 | 0.656 | -0.009 | +0.016 |
| P20 | valence | face | knn (all) | 0.606 | 0.625 | face | knn (all) | 0.606 | 0.625 | +0.000 | +0.000 |
| P21 | arousal | face | gnb (5) | 0.617 | 0.642 | face | gnb (5) | 0.617 | 0.642 | +0.000 | +0.000 |
| P21 | valence | multimodal | knn (10) | 0.582 | 0.784 | multimodal | gnb (20) | 0.584 | 0.703 | +0.001 | -0.081 |
| P22 | arousal | face | gnb (all) | 0.678 | 0.700 | face | gnb (all) | 0.678 | 0.700 | +0.000 | +0.000 |
| P22 | valence | face | svm (5) | 0.639 | 0.658 | face | svm (5) | 0.639 | 0.658 | +0.000 | +0.000 |
| P23 | arousal | face | knn (all) | 0.598 | 0.667 | face | knn (all) | 0.598 | 0.667 | +0.000 | +0.000 |
| P23 | valence | face | knn (all) | 0.584 | 0.583 | face | knn (all) | 0.584 | 0.583 | +0.000 | +0.000 |
| P24 | arousal | multimodal | knn (10) | 0.591 | 0.591 | eeg | knn (all) | 0.597 | 0.596 | +0.006 | +0.006 |
| P24 | valence | eeg | svm (10) | 0.631 | 0.632 | face | knn (5) | 0.557 | 0.558 | -0.074 | -0.074 |
| P25 | arousal | multimodal | knn (5) | 0.584 | 0.592 | eeg | knn (20) | 0.595 | 0.617 | +0.011 | +0.025 |
| P25 | valence | eeg | svm (all) | 0.624 | 0.625 | eeg | svm (all) | 0.596 | 0.600 | -0.028 | -0.025 |
| P26 | arousal | eeg | knn (5) | 0.549 | 0.600 | eeg | gnb (5) | 0.582 | 0.603 | +0.033 | +0.003 |
| P26 | valence | eeg | gnb (10) | 0.583 | 0.598 | eeg | knn (all) | 0.576 | 0.567 | -0.008 | -0.031 |
| P27 | arousal | multimodal | knn (5) | 0.586 | 0.758 | multimodal | svm (all) | 0.557 | 0.742 | -0.029 | -0.017 |
| P27 | valence | eeg | svm (all) | 0.595 | 0.639 | eeg | gnb (all) | 0.583 | 0.661 | -0.012 | +0.022 |
| P28 | arousal | face | gnb (5) | 0.596 | 0.634 | face | gnb (5) | 0.596 | 0.634 | +0.000 | +0.000 |
| P28 | valence | face | svm (all) | 0.570 | 0.642 | eeg | gnb (5) | 0.609 | 0.676 | +0.038 | +0.034 |
| P29 | arousal | eeg | svm (20) | 0.543 | 0.645 | eeg | knn (5) | 0.575 | 0.669 | +0.032 | +0.024 |
| P29 | valence | eeg | gnb (5) | 0.590 | 0.618 | eeg | svm (5) | 0.608 | 0.643 | +0.018 | +0.025 |
| P30 | arousal | face | gnb (all) | 0.585 | 0.617 | face | gnb (all) | 0.585 | 0.617 | +0.000 | +0.000 |
| P30 | valence | face | gnb (all) | 0.590 | 0.592 | face | gnb (all) | 0.590 | 0.592 | +0.000 | +0.000 |
| P31 | arousal | face | gnb (5) | 0.547 | 0.550 | face | gnb (5) | 0.547 | 0.550 | +0.000 | +0.000 |
| P31 | valence | face | knn (5) | 0.536 | 0.533 | face | knn (5) | 0.536 | 0.533 | +0.000 | +0.000 |
| P32 | arousal | face | svm (5) | 0.533 | 0.875 | eeg | gnb (5) | 0.538 | 0.783 | +0.005 | -0.092 |
| P32 | valence | eeg | gnb (20) | 0.598 | 0.600 | eeg | gnb (20) | 0.615 | 0.617 | +0.017 | +0.017 |
| P33 | arousal | face | svm (all) | 0.569 | 0.700 | face | svm (all) | 0.569 | 0.700 | +0.000 | +0.000 |
| P33 | valence | face | svm (5) | 0.658 | 0.658 | face | svm (5) | 0.658 | 0.658 | +0.000 | +0.000 |
| P34 | arousal | multimodal | svm (all) | 0.574 | 0.721 | eeg | gnb (5) | 0.566 | 0.670 | -0.008 | -0.051 |
| P34 | valence | eeg | svm (5) | 0.575 | 0.593 | multimodal | knn (5) | 0.538 | 0.557 | -0.036 | -0.037 |
| P35 | arousal | eeg | gnb (all) | 0.573 | 0.639 | multimodal | gnb (all) | 0.586 | 0.638 | +0.014 | -0.001 |
| P35 | valence | multimodal | svm (all) | 0.635 | 0.641 | eeg | knn (all) | 0.648 | 0.650 | +0.013 | +0.009 |
| P36 | arousal | eeg | knn (all) | 0.616 | 0.697 | eeg | knn (10) | 0.653 | 0.804 | +0.036 | +0.107 |
| P36 | valence | multimodal | knn (10) | 0.649 | 0.655 | multimodal | svm (5) | 0.714 | 0.717 | +0.065 | +0.062 |
| P37 | arousal | face | knn (all) | 0.624 | 0.867 | face | knn (all) | 0.624 | 0.867 | +0.000 | +0.000 |
| P37 | valence | multimodal | gnb (all) | 0.670 | 0.676 | multimodal | gnb (all) | 0.597 | 0.608 | -0.072 | -0.068 |
| P38 | arousal | multimodal | gnb (all) | 0.624 | 0.606 | multimodal | svm (all) | 0.603 | 0.625 | -0.020 | +0.019 |
| P38 | valence | eeg | svm (20) | 0.621 | 0.641 | multimodal | svm (20) | 0.543 | 0.583 | -0.078 | -0.058 |
| P39 | arousal | multimodal | gnb (all) | 0.533 | 0.708 | multimodal | svm (20) | 0.571 | 0.855 | +0.038 | +0.147 |
| P39 | valence | eeg | svm (5) | 0.596 | 0.735 | multimodal | svm (10) | 0.667 | 0.772 | +0.071 | +0.037 |
| P40 | arousal | face | svm (all) | 0.654 | 0.683 | face | svm (all) | 0.654 | 0.683 | +0.000 | +0.000 |
| P40 | valence | multimodal | svm (all) | 0.615 | 0.625 | multimodal | svm (all) | 0.643 | 0.650 | +0.028 | +0.025 |
| P41 | arousal | multimodal | knn (10) | 0.587 | 0.625 | eeg | knn (10) | 0.587 | 0.616 | +0.001 | -0.009 |
| P41 | valence | face | knn (5) | 0.588 | 0.600 | eeg | knn (all) | 0.601 | 0.624 | +0.013 | +0.024 |
| P42 | arousal | multimodal | knn (10) | 0.650 | 0.658 | face | svm (all) | 0.594 | 0.600 | -0.056 | -0.058 |
| P42 | valence | face | gnb (5) | 0.599 | 0.583 | face | gnb (5) | 0.599 | 0.583 | +0.000 | +0.000 |
| P43 | arousal | multimodal | svm (20) | 0.578 | 0.608 | multimodal | svm (all) | 0.578 | 0.600 | +0.000 | -0.008 |
| P43 | valence | multimodal | knn (10) | 0.580 | 0.633 | eeg | knn (10) | 0.546 | 0.583 | -0.035 | -0.050 |
| P44 | arousal | multimodal | knn (all) | 0.648 | 0.680 | face | svm (5) | 0.638 | 0.650 | -0.010 | -0.030 |
| P44 | valence | face | knn (all) | 0.595 | 0.708 | face | knn (all) | 0.595 | 0.708 | +0.000 | +0.000 |
| P45 | arousal | face | svm (all) | 0.575 | 0.583 | face | svm (all) | 0.575 | 0.583 | +0.000 | +0.000 |
| P45 | valence | face | svm (all) | 0.594 | 0.658 | face | svm (all) | 0.594 | 0.658 | +0.000 | +0.000 |
| P46 | arousal | eeg | knn (5) | 0.556 | 0.683 | eeg | gnb (5) | 0.601 | 0.703 | +0.045 | +0.020 |
| P46 | valence | eeg | knn (5) | 0.621 | 0.640 | face | gnb (5) | 0.601 | 0.600 | -0.020 | -0.040 |

### EEG-containing best models only

Here each condition is independently restricted to EEG and multimodal rows before applying the same BA-first selection rule.

| Participant | Target | ICA modality/model | ICA BA | ICA accuracy | no-ICA modality/model | no-ICA BA | no-ICA accuracy | BA delta | Accuracy delta |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| P01 | arousal | multimodal/gnb (20) | 0.594 | 0.618 | multimodal/knn (5) | 0.636 | 0.662 | +0.041 | +0.044 |
| P01 | valence | eeg/svm (5) | 0.583 | 0.605 | multimodal/svm (10) | 0.571 | 0.589 | -0.012 | -0.017 |
| P02 | arousal | multimodal/knn (all) | 0.583 | 0.634 | eeg/svm (all) | 0.526 | 0.560 | -0.057 | -0.074 |
| P02 | valence | eeg/gnb (5) | 0.647 | 0.682 | eeg/gnb (20) | 0.662 | 0.702 | +0.015 | +0.020 |
| P03 | arousal | eeg/gnb (5) | 0.565 | 0.582 | eeg/gnb (all) | 0.572 | 0.607 | +0.007 | +0.025 |
| P03 | valence | eeg/gnb (5) | 0.671 | 0.716 | eeg/gnb (5) | 0.656 | 0.696 | -0.014 | -0.020 |
| P04 | arousal | eeg/knn (20) | 0.607 | 0.613 | eeg/knn (all) | 0.611 | 0.640 | +0.004 | +0.027 |
| P04 | valence | eeg/svm (5) | 0.633 | 0.674 | multimodal/gnb (5) | 0.606 | 0.637 | -0.026 | -0.037 |
| P05 | arousal | eeg/svm (20) | 0.572 | 0.600 | eeg/svm (20) | 0.601 | 0.635 | +0.029 | +0.035 |
| P05 | valence | eeg/svm (10) | 0.556 | 0.563 | eeg/svm (20) | 0.549 | 0.556 | -0.007 | -0.007 |
| P06 | arousal | multimodal/svm (all) | 0.629 | 0.643 | multimodal/svm (10) | 0.678 | 0.680 | +0.049 | +0.037 |
| P06 | valence | eeg/gnb (10) | 0.615 | 0.678 | multimodal/gnb (10) | 0.612 | 0.664 | -0.003 | -0.014 |
| P07 | arousal | multimodal/gnb (all) | 0.541 | 0.592 | eeg/gnb (20) | 0.510 | 0.625 | -0.031 | +0.033 |
| P07 | valence | eeg/gnb (10) | 0.537 | 0.546 | multimodal/gnb (all) | 0.520 | 0.517 | -0.016 | -0.029 |
| P08 | arousal | eeg/svm (5) | 0.587 | 0.587 | eeg/gnb (all) | 0.560 | 0.555 | -0.027 | -0.033 |
| P08 | valence | eeg/knn (10) | 0.595 | 0.597 | eeg/knn (all) | 0.552 | 0.555 | -0.042 | -0.042 |
| P09 | arousal | eeg/gnb (10) | 0.625 | 0.722 | multimodal/gnb (all) | 0.660 | 0.743 | +0.035 | +0.021 |
| P09 | valence | multimodal/gnb (all) | 0.570 | 0.530 | multimodal/svm (all) | 0.562 | 0.570 | -0.009 | +0.040 |
| P10 | arousal | multimodal/svm (20) | 0.511 | 0.517 | multimodal/svm (all) | 0.547 | 0.558 | +0.036 | +0.042 |
| P10 | valence | eeg/knn (20) | 0.539 | 0.575 | eeg/knn (all) | 0.550 | 0.575 | +0.011 | +0.000 |
| P11 | arousal | eeg/gnb (5) | 0.580 | 0.600 | eeg/knn (20) | 0.558 | 0.592 | -0.022 | -0.008 |
| P11 | valence | multimodal/gnb (5) | 0.547 | 0.542 | multimodal/gnb (5) | 0.583 | 0.575 | +0.036 | +0.033 |
| P12 | arousal | eeg/gnb (all) | 0.500 | 0.967 | eeg/gnb (all) | 0.500 | 0.967 | +0.000 | +0.000 |
| P12 | valence | multimodal/knn (10) | 0.547 | 0.692 | multimodal/gnb (all) | 0.547 | 0.633 | +0.000 | -0.058 |
| P13 | arousal | eeg/svm (10) | 0.594 | 0.604 | multimodal/svm (20) | 0.658 | 0.663 | +0.064 | +0.059 |
| P13 | valence | multimodal/gnb (10) | 0.515 | 0.539 | multimodal/gnb (5) | 0.580 | 0.579 | +0.065 | +0.040 |
| P14 | arousal | multimodal/knn (5) | 0.592 | 0.604 | multimodal/svm (20) | 0.581 | 0.598 | -0.011 | -0.005 |
| P14 | valence | multimodal/gnb (all) | 0.598 | 0.586 | eeg/knn (20) | 0.604 | 0.608 | +0.006 | +0.022 |
| P15 | arousal | multimodal/knn (all) | 0.582 | 0.776 | multimodal/gnb (all) | 0.621 | 0.742 | +0.039 | -0.034 |
| P15 | valence | eeg/svm (20) | 0.601 | 0.664 | multimodal/svm (20) | 0.620 | 0.678 | +0.019 | +0.013 |
| P16 | arousal | eeg/svm (20) | 0.674 | 0.701 | multimodal/knn (20) | 0.630 | 0.663 | -0.044 | -0.038 |
| P16 | valence | multimodal/knn (10) | 0.548 | 0.555 | multimodal/svm (10) | 0.608 | 0.612 | +0.060 | +0.057 |
| P17 | arousal | eeg/gnb (all) | 0.582 | 0.543 | eeg/knn (10) | 0.619 | 0.756 | +0.037 | +0.213 |
| P17 | valence | multimodal/gnb (5) | 0.643 | 0.661 | eeg/gnb (5) | 0.641 | 0.664 | -0.002 | +0.003 |
| P18 | arousal | multimodal/gnb (20) | 0.588 | 0.595 | eeg/gnb (20) | 0.585 | 0.587 | -0.003 | -0.009 |
| P18 | valence | eeg/knn (20) | 0.601 | 0.604 | eeg/gnb (20) | 0.634 | 0.635 | +0.034 | +0.032 |
| P19 | arousal | eeg/knn (10) | 0.683 | 0.692 | eeg/knn (10) | 0.657 | 0.667 | -0.026 | -0.025 |
| P19 | valence | eeg/gnb (5) | 0.600 | 0.642 | multimodal/gnb (10) | 0.620 | 0.658 | +0.020 | +0.017 |
| P20 | arousal | eeg/gnb (10) | 0.594 | 0.640 | multimodal/knn (10) | 0.585 | 0.656 | -0.009 | +0.016 |
| P20 | valence | multimodal/svm (all) | 0.539 | 0.546 | eeg/knn (5) | 0.590 | 0.590 | +0.051 | +0.044 |
| P21 | arousal | eeg/knn (5) | 0.591 | 0.648 | multimodal/knn (5) | 0.585 | 0.619 | -0.006 | -0.028 |
| P21 | valence | multimodal/knn (10) | 0.582 | 0.784 | multimodal/gnb (20) | 0.584 | 0.703 | +0.001 | -0.081 |
| P22 | arousal | multimodal/knn (5) | 0.606 | 0.650 | multimodal/knn (10) | 0.614 | 0.675 | +0.008 | +0.025 |
| P22 | valence | multimodal/knn (5) | 0.611 | 0.633 | multimodal/svm (5) | 0.601 | 0.624 | -0.010 | -0.009 |
| P23 | arousal | multimodal/gnb (all) | 0.525 | 0.522 | multimodal/knn (5) | 0.500 | 0.629 | -0.025 | +0.107 |
| P23 | valence | multimodal/knn (all) | 0.582 | 0.607 | multimodal/gnb (5) | 0.552 | 0.568 | -0.030 | -0.039 |
| P24 | arousal | multimodal/knn (10) | 0.591 | 0.591 | eeg/knn (all) | 0.597 | 0.596 | +0.006 | +0.006 |
| P24 | valence | eeg/svm (10) | 0.631 | 0.632 | multimodal/svm (all) | 0.553 | 0.555 | -0.079 | -0.077 |
| P25 | arousal | multimodal/knn (5) | 0.584 | 0.592 | eeg/knn (20) | 0.595 | 0.617 | +0.011 | +0.025 |
| P25 | valence | eeg/svm (all) | 0.624 | 0.625 | eeg/svm (all) | 0.596 | 0.600 | -0.028 | -0.025 |
| P26 | arousal | eeg/knn (5) | 0.549 | 0.600 | eeg/gnb (5) | 0.582 | 0.603 | +0.033 | +0.003 |
| P26 | valence | eeg/gnb (10) | 0.583 | 0.598 | eeg/knn (all) | 0.576 | 0.567 | -0.008 | -0.031 |
| P27 | arousal | multimodal/knn (5) | 0.586 | 0.758 | multimodal/svm (all) | 0.557 | 0.742 | -0.029 | -0.017 |
| P27 | valence | eeg/svm (all) | 0.595 | 0.639 | eeg/gnb (all) | 0.583 | 0.661 | -0.012 | +0.022 |
| P28 | arousal | multimodal/gnb (all) | 0.541 | 0.627 | multimodal/knn (all) | 0.528 | 0.666 | -0.013 | +0.038 |
| P28 | valence | multimodal/knn (all) | 0.570 | 0.691 | eeg/gnb (5) | 0.609 | 0.676 | +0.039 | -0.015 |
| P29 | arousal | eeg/svm (20) | 0.543 | 0.645 | eeg/knn (5) | 0.575 | 0.669 | +0.032 | +0.024 |
| P29 | valence | eeg/gnb (5) | 0.590 | 0.618 | eeg/svm (5) | 0.608 | 0.643 | +0.018 | +0.025 |
| P30 | arousal | eeg/gnb (20) | 0.569 | 0.587 | eeg/svm (10) | 0.502 | 0.625 | -0.067 | +0.038 |
| P30 | valence | multimodal/knn (all) | 0.555 | 0.551 | eeg/knn (all) | 0.520 | 0.517 | -0.036 | -0.034 |
| P31 | arousal | eeg/knn (all) | 0.514 | 0.517 | eeg/knn (20) | 0.537 | 0.538 | +0.023 | +0.020 |
| P31 | valence | multimodal/knn (10) | 0.528 | 0.526 | multimodal/svm (20) | 0.518 | 0.521 | -0.010 | -0.005 |
| P32 | arousal | multimodal/svm (20) | 0.525 | 0.825 | eeg/gnb (5) | 0.538 | 0.783 | +0.013 | -0.042 |
| P32 | valence | eeg/gnb (20) | 0.598 | 0.600 | eeg/gnb (20) | 0.615 | 0.617 | +0.017 | +0.017 |
| P33 | arousal | eeg/knn (20) | 0.523 | 0.701 | multimodal/svm (20) | 0.538 | 0.658 | +0.015 | -0.043 |
| P33 | valence | multimodal/knn (5) | 0.623 | 0.624 | multimodal/knn (10) | 0.588 | 0.589 | -0.035 | -0.034 |
| P34 | arousal | multimodal/svm (all) | 0.574 | 0.721 | eeg/gnb (5) | 0.566 | 0.670 | -0.008 | -0.051 |
| P34 | valence | eeg/svm (5) | 0.575 | 0.593 | multimodal/knn (5) | 0.538 | 0.557 | -0.036 | -0.037 |
| P35 | arousal | eeg/gnb (all) | 0.573 | 0.639 | multimodal/gnb (all) | 0.586 | 0.638 | +0.014 | -0.001 |
| P35 | valence | multimodal/svm (all) | 0.635 | 0.641 | eeg/knn (all) | 0.648 | 0.650 | +0.013 | +0.009 |
| P36 | arousal | eeg/knn (all) | 0.616 | 0.697 | eeg/knn (10) | 0.653 | 0.804 | +0.036 | +0.107 |
| P36 | valence | multimodal/knn (10) | 0.649 | 0.655 | multimodal/svm (5) | 0.714 | 0.717 | +0.065 | +0.062 |
| P37 | arousal | eeg/gnb (5) | 0.550 | 0.797 | eeg/svm (5) | 0.553 | 0.823 | +0.003 | +0.026 |
| P37 | valence | multimodal/gnb (all) | 0.670 | 0.676 | multimodal/gnb (all) | 0.597 | 0.608 | -0.072 | -0.068 |
| P38 | arousal | multimodal/gnb (all) | 0.624 | 0.606 | multimodal/svm (all) | 0.603 | 0.625 | -0.020 | +0.019 |
| P38 | valence | eeg/svm (20) | 0.621 | 0.641 | multimodal/svm (20) | 0.543 | 0.583 | -0.078 | -0.058 |
| P39 | arousal | multimodal/gnb (all) | 0.533 | 0.708 | multimodal/svm (20) | 0.571 | 0.855 | +0.038 | +0.147 |
| P39 | valence | eeg/svm (5) | 0.596 | 0.735 | multimodal/svm (10) | 0.667 | 0.772 | +0.071 | +0.037 |
| P40 | arousal | eeg/knn (20) | 0.606 | 0.658 | multimodal/knn (20) | 0.602 | 0.658 | -0.004 | +0.000 |
| P40 | valence | multimodal/svm (all) | 0.615 | 0.625 | multimodal/svm (all) | 0.643 | 0.650 | +0.028 | +0.025 |
| P41 | arousal | multimodal/knn (10) | 0.587 | 0.625 | eeg/knn (10) | 0.587 | 0.616 | +0.001 | -0.009 |
| P41 | valence | eeg/gnb (10) | 0.574 | 0.581 | eeg/knn (all) | 0.601 | 0.624 | +0.027 | +0.043 |
| P42 | arousal | multimodal/knn (10) | 0.650 | 0.658 | eeg/gnb (5) | 0.574 | 0.567 | -0.076 | -0.092 |
| P42 | valence | multimodal/svm (20) | 0.568 | 0.575 | multimodal/svm (all) | 0.591 | 0.608 | +0.023 | +0.033 |
| P43 | arousal | multimodal/svm (20) | 0.578 | 0.608 | multimodal/svm (all) | 0.578 | 0.600 | +0.000 | -0.008 |
| P43 | valence | multimodal/knn (10) | 0.580 | 0.633 | eeg/knn (10) | 0.546 | 0.583 | -0.035 | -0.050 |
| P44 | arousal | multimodal/knn (all) | 0.648 | 0.680 | multimodal/svm (10) | 0.629 | 0.647 | -0.019 | -0.034 |
| P44 | valence | multimodal/knn (20) | 0.575 | 0.705 | multimodal/knn (20) | 0.532 | 0.680 | -0.043 | -0.026 |
| P45 | arousal | eeg/gnb (20) | 0.534 | 0.558 | multimodal/gnb (20) | 0.536 | 0.558 | +0.002 | +0.000 |
| P45 | valence | eeg/gnb (20) | 0.550 | 0.608 | eeg/gnb (20) | 0.536 | 0.600 | -0.014 | -0.008 |
| P46 | arousal | eeg/knn (5) | 0.556 | 0.683 | eeg/gnb (5) | 0.601 | 0.703 | +0.045 | +0.020 |
| P46 | valence | eeg/knn (5) | 0.621 | 0.640 | eeg/knn (10) | 0.585 | 0.593 | -0.036 | -0.047 |

## C. Participant-level effects (all-modality best models)

| Target and metric | no-ICA better (> +0.02) | Approximately unchanged (±0.02) | ICA better (< -0.02) |
|---|---:|---:|---:|
| Valence BA | 6 | 33 | 7 |
| Valence Ordinary accuracy | 10 | 23 | 13 |
| Arousal BA | 12 | 27 | 7 |
| Arousal Ordinary accuracy | 11 | 27 | 8 |

The same effect labels, deltas, and selected configurations are represented in the detailed CSV as selection flags on the matched configuration rows.

## D. Distribution of participant-best performance

| Selection | Target | Metric | ICA mean / median / SD / IQR / min–max | no-ICA mean / median / SD / IQR / min–max | Mean paired delta |
|---|---|---|---|---|---:|
| All modalities | Valence | Balanced accuracy | 0.605 / 0.598 / 0.033 / 0.036 / 0.536–0.688 | 0.604 / 0.601 / 0.040 / 0.038 / 0.534–0.714 | -0.002 |
| All modalities | Valence | Ordinary accuracy | 0.634 / 0.633 / 0.049 / 0.058 / 0.533–0.784 | 0.631 / 0.629 / 0.052 / 0.075 / 0.533–0.772 | -0.003 |
| All modalities | Arousal | Balanced accuracy | 0.592 / 0.587 / 0.039 / 0.041 / 0.500–0.683 | 0.597 / 0.595 / 0.040 / 0.046 / 0.500–0.678 | +0.004 |
| All modalities | Arousal | Ordinary accuracy | 0.657 / 0.639 / 0.083 / 0.094 / 0.543–0.967 | 0.670 / 0.653 / 0.089 / 0.096 / 0.550–0.967 | +0.013 |
| EEG-containing only | Valence | Balanced accuracy | 0.591 / 0.592 / 0.038 / 0.051 / 0.515–0.671 | 0.589 / 0.589 / 0.044 / 0.059 / 0.518–0.714 | -0.002 |
| EEG-containing only | Valence | Ordinary accuracy | 0.623 / 0.624 / 0.058 / 0.077 / 0.526–0.784 | 0.617 / 0.608 / 0.056 / 0.080 / 0.517–0.772 | -0.006 |
| EEG-containing only | Arousal | Balanced accuracy | 0.580 / 0.583 / 0.041 / 0.045 / 0.500–0.683 | 0.582 / 0.583 / 0.045 / 0.055 / 0.500–0.678 | +0.003 |
| EEG-containing only | Arousal | Ordinary accuracy | 0.646 / 0.631 / 0.084 / 0.093 / 0.517–0.967 | 0.660 / 0.643 / 0.086 / 0.069 / 0.538–0.967 | +0.013 |

## Interpretation

- Participant-best valence BA changed by -0.002 on average (no-ICA minus ICA; n=46).
- Participant-best arousal BA changed by +0.004 on average (no-ICA minus ICA; n=46).
- Across matched eeg-only configurations, BA changed by -0.002 and ordinary accuracy by -0.002 on average (n=1,104).
- Across matched multimodal-only configurations, BA changed by -0.000 and ordinary accuracy by -0.001 on average (n=1,104).
- Across matched face-only configurations, BA changed by +0.000 and ordinary accuracy by +0.000 on average (n=1,104).
- Across all-modality participant-best selections, no-ICA changed BA by +0.001 and ordinary accuracy by +0.005 on average (n=92 paired participant-target selections).
- Face-only matched configurations changed by BA +0.000 and accuracy +0.000. Face is a control: any non-zero change cannot be attributed to ICA acting on EEG features alone, and underscores that reruns/pipeline data may differ beyond ICA.
- Many per-configuration and selected-model differences are smaller than the corresponding five-fold BA SDs recorded in the summaries. Fold SD describes within-condition CV variability, not a paired uncertainty interval for the difference, so it is contextual rather than a formal test.
- With the independent participant-best selection used here, no-ICA preserves individual-classifier performance overall: the mean changes are +0.001 BA and +0.005 ordinary accuracy, both inside the predeclared descriptive ±0.02 band.
- Participants with ordinary-accuracy drops greater than 0.02 after removing ICA: P01 valence (-0.047), P02 arousal (-0.074), P03 valence (-0.020), P08 arousal (-0.033), P08 valence (-0.042), P12 valence (-0.058), P16 arousal (-0.038), P19 arousal (-0.025), P21 valence (-0.081), P24 valence (-0.074), P25 valence (-0.025), P26 valence (-0.031), P32 arousal (-0.092), P34 arousal (-0.051), P34 valence (-0.037), P37 valence (-0.068), P38 valence (-0.058), P42 arousal (-0.058), P43 valence (-0.050), P44 arousal (-0.030), P46 valence (-0.040).
- Participants with ordinary-accuracy improvements greater than 0.02 after removing ICA: P04 arousal (+0.027), P05 arousal (+0.035), P06 arousal (+0.037), P07 arousal (+0.250), P07 valence (+0.134), P09 arousal (+0.021), P13 arousal (+0.059), P14 valence (+0.022), P17 arousal (+0.213), P18 valence (+0.032), P25 arousal (+0.025), P27 valence (+0.022), P28 valence (+0.034), P29 arousal (+0.024), P29 valence (+0.025), P36 arousal (+0.107), P36 valence (+0.062), P39 arousal (+0.147), P39 valence (+0.037), P40 valence (+0.025), P41 valence (+0.024).
- Overall individual-classifier verdict: **EFFECTIVELY SIMILAR**. The full cohort is broadly consistent with, and therefore does not weaken, the current no-ICA preference from the Robot-transfer analysis because it finds no meaningful loss in individualized Image classification. It is not independent causal confirmation of that preference: the Robot-transfer conclusion should continue to rest on its own target, sample construction, and deployment conditions.

## Output contents

`ica_vs_no_ica_full_individual_accuracy_audit.csv` has 3,312 matched configuration rows. It retains both conditions’ BA means/SDs, ordinary accuracy means, deltas, and four flags identifying each condition’s independently selected all-modality and EEG-containing best configuration. No general/LOSO rows are included.
