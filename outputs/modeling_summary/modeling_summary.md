# Image Experiment modeling performance summary

This read-only inventory uses saved result files as the source of truth. Classification is ranked by balanced accuracy (higher is better); regression is ranked by RMSE (lower is better). Midpoint-excluded and three-class diagnostics are retained in the inventory but excluded from standard binary best-model selection.

Recognized saved result files inspected: 23.

## A. Individual classification

Global maximum saved **standard binary** balanced accuracy by target and modality (historical standard-label runs included):

| target | modality | model | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | participant_if_individual | feature_count_or_family | experiment | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | eeg | SVM | 0.705 | balanced_accuracy | 0.707 | accuracy | P36 | paper_compact | feature-family diagnostic | outputs/image_classification/feature_family_top5/best_per_participant_target.csv |
| arousal | face | kNN | 0.703 | balanced_accuracy | 0.717 | accuracy | P40 | 10 | optimized classifier search | outputs/image_classification/individual_optimized_top15/best_per_participant_target.csv |
| arousal | multimodal | kNN | 0.683 | balanced_accuracy | 0.692 | accuracy | P19 | 10 | historical stage_a_individual_v1 | outputs/image_classification/stage_a_individual_v1/results_summary.csv |
| valence | eeg | GaussianNB | 0.671 | balanced_accuracy | 0.716 | accuracy | P03 | 5 | historical stage_c_individual_added18_v1 | outputs/image_classification/stage_c_individual_added18_v1/results_summary.csv |
| valence | face | SVM | 0.688 | balanced_accuracy | 0.700 | accuracy | P17 | 10 | baseline no-ICA | outputs/image_classification/no_ica/results_summary.csv |
| valence | multimodal | SVM | 0.714 | balanced_accuracy | 0.717 | accuracy | P36 | 5 | baseline no-ICA | outputs/image_classification/no_ica/results_summary.csv |

Highest participant BA and mean of each participant's best saved standard BA:

| target | modality | highest_participant_BA | mean_best_per_participant_BA | participants |
| --- | --- | --- | --- | --- |
| arousal | eeg | 0.705 | 0.584 | 46.000 |
| arousal | face | 0.703 | 0.560 | 46.000 |
| arousal | multimodal | 0.683 | 0.586 | 46.000 |
| valence | eeg | 0.671 | 0.598 | 46.000 |
| valence | face | 0.688 | 0.577 | 46.000 |
| valence | multimodal | 0.714 | 0.597 | 46.000 |

## B. General classification

| target | modality | model | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | participant_if_individual | feature_count_or_family | experiment | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | eeg | SVM | 0.506 | balanced_accuracy | 0.597 | accuracy |  | 20 | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| arousal | face | SVM | 0.507 | balanced_accuracy | 0.596 | accuracy |  | 5 | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| arousal | multimodal | GaussianNB | 0.504 | balanced_accuracy | 0.580 | accuracy |  | 20 | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| valence | eeg | kNN | 0.511 | balanced_accuracy | 0.527 | accuracy |  | 10 | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| valence | face | kNN | 0.516 | balanced_accuracy | 0.532 | accuracy |  | 5 | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| valence | multimodal | SVM | 0.511 | balanced_accuracy | 0.534 | accuracy |  | all | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |

## C. Individual regression

| target | modality | model | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | participant_if_individual | feature_count_or_family | experiment | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | eeg | SVR | 0.859 | RMSE | 0.749 | MAE | P08 | all | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| arousal | face | kNN | 0.869 | RMSE | 0.731 | MAE | P08 | 10 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| arousal | multimodal | SVR | 0.849 | RMSE | 0.739 | MAE | P08 | all | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| valence | eeg | SVR | 0.819 | RMSE | 0.541 | MAE | P07 | 5 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| valence | face | SVR | 0.740 | RMSE | 0.553 | MAE | P07 | 10 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| valence | multimodal | SVR | 0.793 | RMSE | 0.551 | MAE | P07 | 10 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |

## D. General regression

| target | modality | model | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | participant_if_individual | feature_count_or_family | experiment | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal | eeg | SVR | 1.702 | RMSE | 1.450 | MAE |  | 5 | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| arousal | face | SVR | 1.714 | RMSE | 1.477 | MAE |  | 5 | historical stage_c_general_regression_all46_v1 | outputs/image_regression/stage_c_general_regression_all46_v1/results_summary.csv |
| arousal | multimodal | Ridge | 1.695 | RMSE | 1.460 | MAE |  | 20 | historical stage_c_general_regression_all46_v1 | outputs/image_regression/stage_c_general_regression_all46_v1/results_summary.csv |
| valence | eeg | SVR | 1.766 | RMSE | 1.493 | MAE |  | 5 | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| valence | face | SVR | 1.767 | RMSE | 1.495 | MAE |  | 5 | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| valence | multimodal | SVR | 1.766 | RMSE | 1.493 | MAE |  | 5 | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |

## E. Neural-network regression/classification experiments

| task_type | model_scope | target | modality | model | input_representation | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | source_file | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | general LOSO | valence | eeg | CNN | raw EEG time-series | 1.837 | RMSE | 1.580 | MAE | outputs/cnn_all_participants_regression_general/valence/aggregate_summary.csv | mean outer-test RMSE; aggregate also records dummy-RMSE deltas |
| regression | general LOSO | arousal | eeg | CNN | raw EEG time-series | 1.756 | RMSE | 1.510 | MAE | outputs/cnn_all_participants_regression_general/arousal/aggregate_summary.csv | mean outer-test RMSE; aggregate also records dummy-RMSE deltas |
| regression | not recoverable | not recoverable | EEG | MLP | handcrafted EEG features |  | RMSE |  | MAE | no persistent output located | result not recoverable from saved outputs |
| regression | not recoverable | not recoverable | Face | MLP | face features |  | RMSE |  | MAE | no persistent output located | result not recoverable from saved outputs |
| regression | not recoverable | not recoverable | Multimodal | MLP | flat multimodal handcrafted features |  | RMSE |  | MAE | no persistent output located | result not recoverable from saved outputs |
| regression | not recoverable | not recoverable | EEG | Branched MLP | branched EEG + face handcrafted features |  | RMSE |  | MAE | no persistent output located | result not recoverable from saved outputs |

## F. Overall best results

| task_type | model_scope | target | modality | model | best_primary_metric | primary_metric_name | secondary_metric | secondary_metric_name | participant_if_individual | experiment | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | general LOSO | arousal | eeg | SVM | 0.506 | balanced_accuracy | 0.597 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | general LOSO | arousal | face | SVM | 0.507 | balanced_accuracy | 0.596 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | general LOSO | arousal | multimodal | GaussianNB | 0.504 | balanced_accuracy | 0.580 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | general LOSO | valence | eeg | kNN | 0.511 | balanced_accuracy | 0.527 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | general LOSO | valence | face | kNN | 0.516 | balanced_accuracy | 0.532 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | general LOSO | valence | multimodal | SVM | 0.511 | balanced_accuracy | 0.534 | accuracy |  | baseline no-ICA | outputs/image_classification/general_no_ica/results_summary.csv |
| classification | individual | arousal | eeg | SVM | 0.705 | balanced_accuracy | 0.707 | accuracy | P36 | feature-family diagnostic | outputs/image_classification/feature_family_top5/best_per_participant_target.csv |
| classification | individual | arousal | face | kNN | 0.703 | balanced_accuracy | 0.717 | accuracy | P40 | optimized classifier search | outputs/image_classification/individual_optimized_top15/best_per_participant_target.csv |
| classification | individual | arousal | multimodal | SVM | 0.679 | balanced_accuracy | 0.696 | accuracy | P06 | optimized classifier search | outputs/image_classification/individual_optimized_top15/best_per_participant_target.csv |
| classification | individual | valence | eeg | SVM | 0.670 | balanced_accuracy | 0.677 | accuracy | P36 | feature-family diagnostic | outputs/image_classification/feature_family_top5/best_per_participant_target.csv |
| classification | individual | valence | face | SVM | 0.688 | balanced_accuracy | 0.700 | accuracy | P17 | baseline no-ICA | outputs/image_classification/no_ica/results_summary.csv |
| classification | individual | valence | multimodal | SVM | 0.714 | balanced_accuracy | 0.717 | accuracy | P36 | baseline no-ICA | outputs/image_classification/no_ica/results_summary.csv |
| regression | general LOSO | arousal | eeg | SVR | 1.702 | RMSE | 1.450 | MAE |  | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| regression | general LOSO | arousal | face | SVR | 1.714 | RMSE | 1.477 | MAE |  | historical stage_c_general_regression_all46_v1 | outputs/image_regression/stage_c_general_regression_all46_v1/results_summary.csv |
| regression | general LOSO | arousal | multimodal | Ridge | 1.695 | RMSE | 1.460 | MAE |  | historical stage_c_general_regression_all46_v1 | outputs/image_regression/stage_c_general_regression_all46_v1/results_summary.csv |
| regression | general LOSO | valence | eeg | SVR | 1.766 | RMSE | 1.493 | MAE |  | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| regression | general LOSO | valence | face | SVR | 1.767 | RMSE | 1.495 | MAE |  | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| regression | general LOSO | valence | multimodal | SVR | 1.766 | RMSE | 1.493 | MAE |  | historical stage_a_general_regression_v1 | outputs/image_regression/stage_a_general_regression_v1/results_summary.csv |
| regression | individual | arousal | eeg | SVR | 0.859 | RMSE | 0.749 | MAE | P08 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| regression | individual | arousal | face | kNN | 0.869 | RMSE | 0.731 | MAE | P08 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| regression | individual | arousal | multimodal | SVR | 0.849 | RMSE | 0.739 | MAE | P08 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| regression | individual | valence | eeg | SVR | 0.819 | RMSE | 0.541 | MAE | P07 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| regression | individual | valence | face | SVR | 0.740 | RMSE | 0.553 | MAE | P07 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
| regression | individual | valence | multimodal | SVR | 0.793 | RMSE | 0.551 | MAE | P07 | historical stage_c_individual_regression_all46_v1 | outputs/image_regression/stage_c_individual_regression_all46_v1/results_summary.csv |
