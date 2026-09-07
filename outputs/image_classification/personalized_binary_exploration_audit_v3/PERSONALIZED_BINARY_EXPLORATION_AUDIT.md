# Personalized binary Image classification: exploratory-search audit

Generated from completed outputs only. No models were fit for this report; raw `data/` was not read or changed.

## What the four folders are

| Folder | Cohort and design | Completed configuration-level rows |
|---|---|---:|
| `no_ica` | Baseline: P01–P46, EEG/face/multimodal, kNN/SVM/GNB, requests all/5/10/20 | 3,312 |
| `individual_optimized_top15` | Follow-up restricted to P06/P17/P22/P36/P40; adds LogReg and ExtraTrees, requests 3/5/8/10/15/20/30/all | 1,050 |
| `feature_family_top5` | Same five participants; EEG-only feature-family/channel-subset representations and kNN/SVM/LogReg/GNB | 3,400 |
| `label_ambiguity_top5` | Same five; standard labels vs excluding rating 4, with EEG/face/multimodal and kNN/SVM/LogReg/GNB | 1,680 |

Every row is an outer-CV mean balanced accuracy (BA), using shuffled stratified outer CV (up to 5 folds) and fold-local inner CV (up to 3 folds) for selection. The standard label is LOW `<4`, HIGH `>=4`; midpoint exclusion is LOW `<=3`, HIGH `>=5`, dropping rating 4. `no_ica` has fold-level predictions/metrics; the three follow-ups retain only configuration-level outer means and selected fold hyperparameters, not held-out predictions.

The searches are **not one common benchmark**: the top-five studies are selected from the 46-person baseline, expand the model/count grids, and in the feature-family study use EEG-only representations. Claims below are therefore within-experiment comparisons unless marked exploratory.

## Opportunity-adjusted evidence

Use `factor_summary.csv` for every requested count, median, mean, upper quartile, maximum, and BA success fraction. Counts differ for valid reasons: face has fewer feasible feature-count requests; family/subset representations have different available dimensions; and `paper_compact` consequently has 1,120 of 3,400 feature-family rows. Raw numbers of threshold exceedances must not be compared without their denominators.

### Baseline (46 participants; apples-to-apples within this grid)

| level | n_configurations | median_ba | mean_ba | upper_quartile_ba | maximum_ba | fraction_ba_ge_60 |
| --- | --- | --- | --- | --- | --- | --- |
| gnb | 552.000 | 0.521 | 0.521 | 0.561 | 0.662 | 0.101 |
| svm | 552.000 | 0.514 | 0.520 | 0.552 | 0.714 | 0.069 |
| knn | 552.000 | 0.513 | 0.512 | 0.543 | 0.648 | 0.029 |

Valence and arousal were both weak overall (baseline medians 0.515 and 0.508 respectively). GNB has the best typical baseline BA; kNN is consistently lower. Face is modestly strongest for both targets in this baseline, so it would be incorrect to claim an EEG-only family changed the multimodal baseline.

### Expanded five-participant search (apples-to-apples within the five participants)

| level | n_configurations | median_ba | mean_ba | upper_quartile_ba | maximum_ba | fraction_ba_ge_60 |
| --- | --- | --- | --- | --- | --- | --- |
| gnb | 105.000 | 0.581 | 0.569 | 0.603 | 0.670 | 0.276 |
| logreg | 105.000 | 0.552 | 0.567 | 0.625 | 0.708 | 0.362 |
| svm | 105.000 | 0.554 | 0.558 | 0.588 | 0.674 | 0.190 |
| extra_trees | 105.000 | 0.554 | 0.556 | 0.589 | 0.679 | 0.152 |
| knn | 105.000 | 0.534 | 0.537 | 0.565 | 0.681 | 0.086 |

LogReg is the most useful broad addition: paired with identical participant/target/modality/count settings, it beats kNN in 69.0% of pairs (mean ΔBA +0.024), ExtraTrees in 65.7% (+0.016), SVM in 58.1% (+0.009), and ties GNB overall (50.0%, +0.004). Its advantage is larger for valence versus kNN (+0.031) but present for both targets. This is robust relative to nearby *configurations*, not proof of a general population effect: all five participants were preselected for promising baseline maxima.

Feature count 10 is the strongest broadly supported count in this follow-up (especially arousal); count 8 is also competitive. Requests 3 and 5 are notably weak for arousal, while 20/30 add little typical performance. Face has the best typical valence BA but only five valid requests per classifier versus eight for EEG/multimodal.

### Feature-family experiment (EEG representations only)

| level | n_configurations | median_ba | mean_ba | upper_quartile_ba | maximum_ba | fraction_ba_ge_60 |
| --- | --- | --- | --- | --- | --- | --- |
| time_statistical | 60.000 | 0.560 | 0.558 | 0.588 | 0.628 | 0.217 |
| hjorth | 180.000 | 0.548 | 0.546 | 0.583 | 0.670 | 0.144 |
| bandpower_all | 140.000 | 0.544 | 0.543 | 0.571 | 0.655 | 0.107 |
| all_eeg | 140.000 | 0.541 | 0.542 | 0.573 | 0.670 | 0.129 |
| paper_compact | 560.000 | 0.536 | 0.534 | 0.568 | 0.666 | 0.091 |
| bandpower_beta_gamma | 240.000 | 0.528 | 0.522 | 0.553 | 0.644 | 0.033 |
| entropy_beta_gamma | 240.000 | 0.504 | 0.507 | 0.538 | 0.600 | 0.004 |
| entropy_all_bands | 140.000 | 0.491 | 0.491 | 0.541 | 0.616 | 0.007 |

For arousal, `all_eeg` has the best family-level typical BA (mean 0.546); for valence, time-statistical (only 60 rows) and Hjorth are strongest. `paper_compact` is middling for valence (mean 0.534) and weak for arousal (0.513), despite having far more opportunities than several families. That makes its isolated maxima particularly unpersuasive.

`paper_compact` exactly comprises 64 all-channel features: standard deviation; Hjorth mobility and complexity; median frequency; beta and gamma bandpower; beta and gamma spectral entropy, over Fz/C3/Cz/C4/Pz/PO7/Oz/PO8. Its declared subsets contain 32 features each (midline, posterior, frontal-central). It deliberately excludes broad-band spectral entropy and every unlisted feature type.

The central hypothesis is not supported globally. Across matched `paper_compact` cells, LogReg beats SVM in 55.7% (mean Δ +0.005), GNB in 52.1% (+0.004), and kNN in 63.2% (+0.016). Its stronger pockets are participant-specific (notably P17 valence/posterior and P36 arousal/frontal-central), while midline/paper_compact is consistently weak. All-channel and frontal-central `paper_compact` are therefore hypotheses worth a small confirmatory block, not a favored primary family.

### Label ambiguity (matched but changes the dataset)

| target | n_matched_configurations | mean_delta_ba | median_delta_ba | fraction_logreg_better |
| --- | --- | --- | --- | --- |
| arousal | 420.000 | -0.010 | -0.012 | 0.436 |
| valence | 420.000 | 0.016 | 0.013 | 0.607 |

Excluding midpoint trials worsens arousal in 56.4% of matched settings (mean Δ −0.010) and helps valence in 60.7% (mean Δ +0.016). This is exploratory because the evaluation set, number of trials, and class balance change simultaneously; it should not replace standard labels without a prespecified target-specific validation.

## Participants and apparent learnability

| participant | target | median_ba | maximum_ba | fraction_ba_ge_60 | optimized_median_ba | optimized_maximum_ba | optimized_fraction_ba_ge_60 | eeg_usable_trials | eeg_retention_percent | qc_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P06 | arousal | 0.563 | 0.678 | 0.222 | 0.564 | 0.679 | 0.257 | 115.000 | 95.833 | Review due to elevated EEG rejection |
| P22 | arousal | 0.575 | 0.678 | 0.111 | 0.566 | 0.678 | 0.105 | 120.000 | 100.000 | Very clean |
| P40 | arousal | 0.489 | 0.654 | 0.222 | 0.488 | 0.703 | 0.086 | 120.000 | 100.000 | Very clean |
| P36 | arousal | 0.553 | 0.653 | 0.139 | 0.579 | 0.695 | 0.295 | 102.000 | 85.000 | Review due to elevated EEG rejection |
| P17 | arousal | 0.527 | 0.619 | 0.028 | 0.546 | 0.632 | 0.076 | 118.000 | 98.333 | Clean with minor EEG loss |
| P36 | valence | 0.583 | 0.714 | 0.278 | 0.591 | 0.708 | 0.419 | 102.000 | 85.000 | Review due to elevated EEG rejection |
| P17 | valence | 0.592 | 0.688 | 0.389 | 0.588 | 0.678 | 0.352 | 118.000 | 98.333 | Clean with minor EEG loss |
| P40 | valence | 0.512 | 0.643 | 0.167 | 0.516 | 0.642 | 0.067 | 120.000 | 100.000 | Very clean |
| P22 | valence | 0.553 | 0.639 | 0.278 | 0.546 | 0.650 | 0.171 | 120.000 | 100.000 | Very clean |
| P06 | valence | 0.556 | 0.634 | 0.139 | 0.547 | 0.634 | 0.057 | 115.000 | 95.833 | Review due to elevated EEG rejection |

P36 (both targets) and P17 valence have broad above-chance-looking configuration distributions, not just one high maximum; P06 arousal and P22 arousal are moderately promising. P40 is the clearest “isolated maximum” case: maxima of 0.703/0.642 but optimized medians 0.501/0.515 and only 8.6%/6.7% of configurations at BA ≥0.60. It should not drive design decisions.

Across all 46 participants, many target-specific distributions stay near 0.50. This alone does **not** identify noisy EEG: several near-chance participants have 100% retention, complete ratings, and “Very clean” QC, while P36 performs well despite 85% retention. Class composition by target is retained in `fold_results.csv`; the report's participant table includes retained-trial/QC context, but no fold predictions exist for the later studies to establish a mechanistic cause.

## Hyperparameters: what can and cannot be concluded

The saved hyperparameters are the *inner-CV winner per outer fold*, not performance for every candidate grid point. They can support pruning candidates that never win, but cannot establish a causal outer-BA ranking for a hyperparameter. In the optimized study, GNB selected `var_smoothing=1e-11` in every fold; this is a reasonable reduction. For LogReg, selected C values 0.01–1 have slightly better descriptive outer distributions than 10–100, and `class_weight=balanced` is modestly better. For SVM, RBF gamma 1 and the low-C 0.1 region are unproductive in the saved selections; linear or RBF gamma 0.001/0.01 are more defensible. These are pruning hypotheses, not estimates from a separate hyperparameter experiment.

`selected_hyperparameter_summary.csv` supplies the selection count and the same descriptive BA measures for every selected value. The count is outer folds that selected the value, not an independent count of model fits.

## Proposed reduced next experiment

**KEEP**

- Standard labels as the primary analysis; nested CV, fold-local selection, seed 42, and BA as the selection metric.
- LogReg alongside GNB and SVM: LogReg is consistently better than kNN/ExtraTrees and comparable-to-slightly-better than SVM/GNB in matched expanded settings.
- Feature requests 8 and 10 (plus `all` only as a prespecified reference); GNB `var_smoothing=1e-11`; LogReg C `0.01, 0.1, 1` with and without balanced weights; SVM linear plus RBF gamma `0.001, 0.01`, C `1, 10`.
- EEG `all_eeg` and Hjorth as primary EEG family representations. Retain face/multimodal only if the goal remains operational multimodal prediction, because the strongest baseline typical result was face rather than EEG.

**DROP**

- kNN and ExtraTrees from the primary focused search; neither is competitive with LogReg in matched settings.
- Feature count 30, and 3/5 for arousal; they consume search budget without robust benefit.
- `entropy_beta_gamma` and its midline/posterior variants; they are consistently near chance despite many opportunities.
- `paper_compact` midline; posterior is also not a general candidate. Do not use the P17/P36 high cells as global defaults.
- SVM RBF gamma 1 and the broad redundant SVM/LogReg grids unless a prespecified sensitivity block requires them.

**UNCERTAIN**

- `paper_compact × LogReg`: include only a small, explicitly secondary all-channel/frontal-central confirmation block; it is not supported as the main route to higher typical BA.
- Excluding rating-4 trials for valence only: a target-specific secondary label-sensitivity analysis is warranted; arousal evidence argues against it.
- Participant-specific model spaces: promising P36/P17/P06/P22 targets merit confirmation, but were selected after exploration and need a guard against winner’s curse.

Smallest reasonable next test: use the five selected participants but prespecify a confirmatory split/repeated-CV protocol, evaluate standard labels, feature counts 8/10, classifiers LogReg/GNB/SVM, and three representations (`all_eeg`, Hjorth-all-channel, plus secondary `paper_compact` all-channel/frontal-central). Hold one untouched evaluation procedure/seed set or use repeated outer CV; report each participant-target rather than only pooled maxima. Add label-midpoint exclusion only as a valence sensitivity analysis. This is substantially smaller than the previous grids while directly testing the remaining LogReg and paper-compact hypotheses.

For genuine participant-level above-chance claims, run a later, separately budgeted permutation test that repeats the **entire nested pipeline** (including feature selection and inner tuning) with shuffled labels, then compare the observed outer-CV BA to that participant-target null distribution. Permuting only final predictions or tuning on unpermuted labels would be invalid.
