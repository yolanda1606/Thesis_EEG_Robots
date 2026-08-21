# Stage B Individual Classification

## Cohort

The completed individual analysis includes 28 participants: P10, P11, P14, P15, P16, P17, P18, P19, P20, P22, P23, P24, P25, P26, P27, P30, P32, P33, P34, P38, P39, P40, P41, P42, P43, P44, P45, P46.

## Overall individualized performance

- Valence: mean 0.606; median 0.599; SD 0.026; range 0.571–0.688; ≥0.55: 28/28; ≥0.60: 13/28; ≥0.65: 2/28.
- Arousal: mean 0.597; median 0.586; SD 0.041; range 0.533–0.683; ≥0.55: 25/28; ≥0.60: 8/28; ≥0.65: 5/28.

## Best participants

### Valence

- P17: BA 0.688; face; svm; All.
- P33: BA 0.658; face; svm; Top 5.
- P22: BA 0.639; face; svm; Top 5.
- P24: BA 0.631; eeg; svm; Top 10.
- P25: BA 0.624; eeg; svm; All.

### Arousal

- P19: BA 0.683; eeg; knn; Top 10.
- P22: BA 0.678; face; gnb; All.
- P16: BA 0.674; eeg; svm; Top 20.
- P40: BA 0.654; face; svm; All.
- P42: BA 0.650; multimodal; knn; Top 10.

## Modality patterns

Exact ties are defined as configurations with exactly equal best mean BA. A participant is counted as `Tie` only when those exact winners span multiple modalities; otherwise the shared modality receives the count.
- Valence: eeg 9; face 14; multimodal 3; Tie 2.
- Arousal: eeg 4; face 11; multimodal 11; Tie 2.

These counts describe winning configurations and do not show that multimodal input is intrinsically better; multimodal models contain more predictors.

## Classifier patterns

The same exact-tie rule is used for classifier counts.
- Valence: knn 8; svm 12; gnb 8; Tie 0.
- Arousal: knn 12; svm 8; gnb 7; Tie 1.

## Feature-count patterns

- All: 24 participant-target winners.
- Top 5: 19 participant-target winners.
- Top 10: 8 participant-target winners.
- Top 20: 5 participant-target winners.

## Interpretation

These are exploratory nested-CV results. Selecting the best of many configurations per participant is useful for identifying calibration potential, but it is not an independent estimate of a pre-specified deployed model.

The completed Stage B subject-independent LOSO analysis was near chance at its best configuration: valence BA 0.513; arousal BA 0.523.
Participant-best individual values that are higher than these LOSO results are consistent with participant-specific calibration being more promising than subject-independent transfer. They are not final thesis conclusions, and no significance tests were performed here.
