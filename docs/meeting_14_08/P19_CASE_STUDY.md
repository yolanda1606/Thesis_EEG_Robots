# P19 Individual Emotion-Classification Case Study

## Why P19

P19 belongs to the **Very clean** QC group and had 120 usable trials for both targets and all three modalities. Its rating classes were: valence LOW/HIGH = 43/77; arousal LOW/HIGH = 65/55. These are moderately imbalanced, so raw accuracy alone would be misleading; balanced accuracy remains the primary metric.

## Best individual result

| Target | Modality | Classifier | Features | Balanced accuracy | SD | Range | Frequent hyperparameters |
|---|---|---|---|---:|---:|---:|---|
| Best overall tie: arousal | eeg | knn | 10 | 0.683 | 0.076 | 0.573–0.780 | `{"classifier__n_neighbors": 7, "classifier__weights": "distance"}` |
| Best valence: valence | eeg | gnb | 10 | 0.600 | 0.084 | 0.531–0.733 | `{"classifier__var_smoothing": 1e-11}` |
| Best arousal: arousal | eeg | knn | 10 | 0.683 | 0.076 | 0.573–0.780 | `{"classifier__n_neighbors": 7, "classifier__weights": "distance"}` |
| Best face: arousal | face | svm | 5 | 0.534 | 0.113 | 0.360–0.671 | `{"classifier__C": 10, "classifier__gamma": "scale", "classifier__kernel": "rbf"}` |


The highest P19 mean balanced accuracy was a four-way numerical tie at **0.683**: eeg knn 10; eeg svm 5; multimodal knn 10; multimodal svm 5. This report uses **arousal, EEG, kNN, Top 10** as the representative winner because it is the direct EEG-only result. The tied multimodal kNN result selected the same EEG-only feature set, so it did not demonstrate an added face-feature benefit.

The kNN model classifies a trial according to the LOW/HIGH ratings of its most similar training trials in the selected EEG-feature space. Each outer fold used 96 P19 trials for training/tuning and 24 untouched P19 trials for testing.

Fold balanced accuracies were 0.710, 0.696, 0.573, 0.780, 0.657; mean 0.683, SD 0.076, median 0.696, range 0.573–0.780. Fold confusion-matrix totals were TN=51, FP=14, FN=23, TP=32. Tuning selected: fold 1: {"classifier__n_neighbors": 7, "classifier__weights": "distance"}; fold 2: {"classifier__n_neighbors": 11, "classifier__weights": "distance"}; fold 3: {"classifier__n_neighbors": 7, "classifier__weights": "distance"}; fold 4: {"classifier__n_neighbors": 11, "classifier__weights": "uniform"}; fold 5: {"classifier__n_neighbors": 5, "classifier__weights": "uniform"}. The most frequent combination was `{"classifier__n_neighbors": 7, "classifier__weights": "distance"}`.

## Valence vs arousal

P19 was more predictable for **arousal**: best arousal 0.683 ± 0.076 (median 0.696; range 0.573–0.780) versus best valence 0.600 ± 0.084 (median 0.578; range 0.531–0.733). This is an exploratory within-person comparison, not a significance test.

## EEG vs face vs multimodal

The best EEG result was 0.683 ± 0.076 (median 0.696; range 0.573–0.780). The best face result was 0.534 ± 0.113 (median 0.535; range 0.360–0.671). The best multimodal result tied the EEG result, but its selected predictors were EEG-only; therefore multimodal input did **not** improve on EEG for P19’s top result.

## Most informative features

These features were consistently selected by the representative kNN classifier for P19; they are not established neural biomarkers.

- `eeg_bp_beta__PO8` — beta-band power at PO8; selected in 5/5 folds (100%), mean F score 18.69.
- `eeg_bp_beta__Fz` — beta-band power at Fz; selected in 5/5 folds (100%), mean F score 14.59.
- `eeg_hc__Fz` — Hjorth complexity at Fz; selected in 5/5 folds (100%), mean F score 14.02.
- `eeg_hm__Fz` — Hjorth mobility at Fz; selected in 5/5 folds (100%), mean F score 11.68.
- `eeg_se__Fz` — spectral entropy at Fz; selected in 5/5 folds (100%), mean F score 11.61.
- `eeg_hc__PO8` — Hjorth complexity at PO8; selected in 4/5 folds (80%), mean F score 11.07.
- `eeg_mf_hz__Fz` — median frequency at Fz; selected in 4/5 folds (80%), mean F score 10.11.
- `eeg_mf_hz__PO8` — median frequency at PO8; selected in 3/5 folds (60%), mean F score 11.83.
- `eeg_hm__PO8` — Hjorth mobility at PO8; selected in 3/5 folds (60%), mean F score 11.05.
- `eeg_mf_hz__Cz` — median frequency at Cz; selected in 2/5 folds (40%), mean F score 11.15.

Selected features were concentrated in frontal/central/posterior channels: Fz (25 selections), PO8 (16 selections), Oz (4 selections), Cz (2 selections), PO7 (2 selections), Pz (1 selections). Band-specific selections were dominated by beta (13), delta (2); other repeatedly selected features were Hjorth mobility/complexity, median frequency, and full-spectrum spectral entropy. Feature names and definitions come directly from the frozen feature extractor: band-power features integrate Welch PSD in the named band, while spectral-entropy features apply Shannon entropy to the relevant PSD values.

## Individual vs unseen-participant generalization

**Individual nested CV:** trained and tested only within P19, representative arousal EEG kNN Top 10 = 0.683.

**General LOSO, matched configuration:** trained on the other nine Very clean participants and tested entirely on P19, arousal EEG kNN Top 10 = 0.422; absolute difference = 0.262.

**General LOSO, best P19 arousal configuration:** multimodal knn 5 = 0.570. This best-vs-best comparison uses different configurations and is descriptive only.

The difference between personalized and unseen-participant performance is consistent with P19-specific affective patterns being more learnable after personal calibration. One participant cannot establish that conclusion statistically.

## Interpretation for the thesis

P19 is a useful case study because its individualized arousal prediction was above balanced-accuracy chance and reasonably stable across folds, whereas P19 was weaker when completely unseen by models trained on other people. This supports investigating personalized calibration alongside subject-independent approaches, without generalizing from P19 to the wider cohort.

## What I can say in the meeting

- "P19 was in the Very clean QC group and had 120 usable trials for both valence and arousal."
- "P19’s strongest individualized result was arousal classification from EEG features, with mean balanced accuracy 0.683."
- "The result was a tie with some multimodal fits because their winning feature sets contained EEG features only; face features did not add to that result."
- "The consistently selected features included beta-band power at PO8 and Fz, plus Hjorth mobility and complexity at Fz."
- "When P19 was held out entirely from a general model, the matched configuration fell to 0.422, close to or below balanced-accuracy chance."
- "This is compatible with participant-specific mappings that may benefit from personal calibration, but it is not a statistical conclusion from one participant."
