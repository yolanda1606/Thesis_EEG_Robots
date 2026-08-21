# P19 Image Classification: ICA versus No-ICA

This is an EEG-only, P19-only comparison. Both representations use the unchanged validated individual nested-CV evaluator; no Robot data was used.

## Comparability audit

- Canonical retained EEG rows: 960; no-ICA retained EEG rows: 960.
- Canonical retained triggers: 120; no-ICA retained triggers: 120.
- EEG feature column names identical: True; count canonical/no-ICA: 15/15.
- Channel set identical: True; participant/trial keys identical: True.
- Target/rating rows identical: True; merged row count canonical/no-ICA: 120/120.
- Non-feature EEG metadata identical: True. Numeric EEG feature values are intentionally permitted to differ because ICA is disabled.
- Resolved preprocessing configuration differs only at ICA enabled state and installation-specific resource_root path.
- Video/face artifacts are absent from p19_no_ica by design; this wrapper evaluates EEG only.

## Best configurations

| Representation | Target | Classifier | Requested features | Mean nested-CV BA | SD | LOW recall | HIGH recall | Exact top tie |
|---|---|---|---|---:|---:|---:|---:|---|
| no_ICA | arousal | knn | 10 | 0.657 | 0.056 | 0.769 | 0.545 | False |
| no_ICA | valence | gnb | 10 | 0.602 | 0.090 | 0.464 | 0.739 | False |
| with_ICA | arousal | knn | 10 | 0.683 | 0.076 | 0.785 | 0.582 | True |
| with_ICA | arousal | svm | 5 | 0.683 | 0.073 | 0.785 | 0.582 | True |
| with_ICA | valence | gnb | 10 | 0.600 | 0.084 | 0.486 | 0.714 | True |
| with_ICA | valence | gnb | 5 | 0.600 | 0.099 | 0.461 | 0.739 | True |

## Matched-configuration BA changes

`delta_BA_no_ICA_minus_with_ICA` is descriptive. No formal significance test is performed; differences must be interpreted alongside outer-fold variability.

| Target | Classifier | Features | With ICA BA | No-ICA BA | Delta |
|---|---|---|---:|---:|---:|
| arousal | gnb | 10 | 0.631 | 0.654 | +0.023 |
| arousal | gnb | 20 | 0.640 | 0.624 | -0.015 |
| arousal | gnb | 5 | 0.662 | 0.645 | -0.017 |
| arousal | gnb | all | 0.615 | 0.606 | -0.009 |
| arousal | knn | 10 | 0.683 | 0.657 | -0.026 |
| arousal | knn | 20 | 0.603 | 0.582 | -0.022 |
| arousal | knn | 5 | 0.631 | 0.610 | -0.021 |
| arousal | knn | all | 0.590 | 0.562 | -0.027 |
| arousal | svm | 10 | 0.652 | 0.631 | -0.021 |
| arousal | svm | 20 | 0.633 | 0.556 | -0.077 |
| arousal | svm | 5 | 0.683 | 0.642 | -0.041 |
| arousal | svm | all | 0.546 | 0.529 | -0.017 |
| valence | gnb | 10 | 0.600 | 0.602 | +0.001 |
| valence | gnb | 20 | 0.554 | 0.558 | +0.004 |
| valence | gnb | 5 | 0.600 | 0.601 | +0.001 |
| valence | gnb | all | 0.546 | 0.536 | -0.011 |
| valence | knn | 10 | 0.569 | 0.552 | -0.017 |
| valence | knn | 20 | 0.580 | 0.539 | -0.041 |
| valence | knn | 5 | 0.545 | 0.590 | +0.045 |
| valence | knn | all | 0.512 | 0.500 | -0.012 |
| valence | svm | 10 | 0.545 | 0.565 | +0.019 |
| valence | svm | 20 | 0.548 | 0.477 | -0.071 |
| valence | svm | 5 | 0.518 | 0.537 | +0.019 |
| valence | svm | all | 0.444 | 0.443 | -0.002 |

## Interpretation boundary

A higher Robot-domain compatibility result from a separate experiment does not determine Image model selection here. No-ICA is not called better based on a small mean BA difference alone. Face and multimodal configurations are outside this comparison because no face/video artifacts were regenerated for `p19_no_ica`.
