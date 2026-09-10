# Frozen Robot-transfer context analysis

Read-only analysis of saved final cohort task-level outputs. Participant-level aggregation precedes every task-type or condition test; no overlapping windows were treated as independent observations.

## Task-type P(HIGH) summaries

| branch            | target   | task_type   | metric                |   n_participants |     mean |       sd |   median |       q1 |       q3 |      iqr |
|:------------------|:---------|:------------|:----------------------|-----------------:|---------:|---------:|---------:|---------:|---------:|---------:|
| eeg_only          | arousal  | Observation | task_consensus_p_high |               46 | 0.546743 | 0.250218 | 0.554296 | 0.456402 | 0.718666 | 0.262264 |
| eeg_only          | arousal  | Interaction | task_consensus_p_high |               44 | 0.536214 | 0.295997 | 0.537882 | 0.38178  | 0.770548 | 0.388768 |
| eeg_only          | arousal  | Alone       | task_consensus_p_high |               44 | 0.537508 | 0.342681 | 0.530126 | 0.272538 | 0.846094 | 0.573557 |
| eeg_only          | valence  | Observation | task_consensus_p_high |               46 | 0.553302 | 0.162755 | 0.517892 | 0.465597 | 0.64588  | 0.180283 |
| eeg_only          | valence  | Interaction | task_consensus_p_high |               44 | 0.541088 | 0.191915 | 0.501819 | 0.413684 | 0.648789 | 0.235105 |
| eeg_only          | valence  | Alone       | task_consensus_p_high |               44 | 0.552282 | 0.263727 | 0.53504  | 0.413692 | 0.722152 | 0.30846  |
| modality_agnostic | arousal  | Observation | task_consensus_p_high |               46 | 0.524842 | 0.256456 | 0.522412 | 0.410576 | 0.681624 | 0.271048 |
| modality_agnostic | arousal  | Interaction | task_consensus_p_high |               44 | 0.502252 | 0.276211 | 0.504378 | 0.35615  | 0.681995 | 0.325845 |
| modality_agnostic | arousal  | Alone       | task_consensus_p_high |               44 | 0.473565 | 0.316747 | 0.424324 | 0.253279 | 0.718632 | 0.465354 |
| modality_agnostic | valence  | Observation | task_consensus_p_high |               46 | 0.562102 | 0.197362 | 0.540994 | 0.442863 | 0.678125 | 0.235262 |
| modality_agnostic | valence  | Interaction | task_consensus_p_high |               44 | 0.545048 | 0.213804 | 0.475835 | 0.40284  | 0.688735 | 0.285895 |
| modality_agnostic | valence  | Alone       | task_consensus_p_high |               44 | 0.5402   | 0.265997 | 0.502914 | 0.398264 | 0.729419 | 0.331154 |

## Task-type Friedman tests: P(HIGH)

| branch            | target   | metric                |   paired_n |   friedman_chi_square |   p_value_raw | test_status   |   p_value_fdr_bh | fdr_significant_0_05   |
|:------------------|:---------|:----------------------|-----------:|----------------------:|--------------:|:--------------|-----------------:|:-----------------------|
| eeg_only          | arousal  | task_consensus_p_high |         42 |               1.71429 |     0.424373  | computed      |        0.636559  | False                  |
| eeg_only          | valence  | task_consensus_p_high |         42 |               4.90476 |     0.0860884 | computed      |        0.154959  | False                  |
| modality_agnostic | arousal  | task_consensus_p_high |         42 |               5.33333 |     0.0694835 | computed      |        0.0955329 | False                  |
| modality_agnostic | valence  | task_consensus_p_high |         42 |               2.28571 |     0.318907  | computed      |        0.531511  | False                  |

## Observation-condition paired tests: P(HIGH)

| branch            | target   | comparison   | left_condition   | right_condition   | metric                |   paired_n |   mean_delta |   median_delta |   positive_delta_count |   negative_delta_count |   zero_delta_count |   wilcoxon_statistic |   p_value_raw |   rank_biserial_correlation |   median_delta_ci_95_low |   median_delta_ci_95_high |   p_value_fdr_bh | fdr_significant_0_05   |
|:------------------|:---------|:-------------|:-----------------|:------------------|:----------------------|-----------:|-------------:|---------------:|-----------------------:|-----------------------:|-------------------:|---------------------:|--------------:|----------------------------:|-------------------------:|--------------------------:|-----------------:|:-----------------------|
| eeg_only          | arousal  | fault_effect | FAULTY_FAST      | CONTROL_FAST      | task_consensus_p_high |         46 | -0.0269634   |    1.75371e-10 |                     23 |                     22 |                  1 |                  474 |      0.62342  |                  -0.084058  |              -0.0197004  |               0.00989591  |         0.956858 | False                  |
| eeg_only          | arousal  | speed_effect | CONTROL_FAST     | CONTROL_SLOW      | task_consensus_p_high |         45 |  0.0151196   |   -0.000156228 |                     22 |                     23 |                  0 |                  452 |      0.466816 |                   0.12657   |              -0.00919972 |               0.0197163   |         0.875664 | False                  |
| eeg_only          | valence  | fault_effect | FAULTY_FAST      | CONTROL_FAST      | task_consensus_p_high |         46 | -0.017908    |   -0.0138259   |                     21 |                     25 |                  0 |                  444 |      0.297277 |                  -0.178538  |              -0.0421218  |               0.0214      |         0.778426 | False                  |
| eeg_only          | valence  | speed_effect | CONTROL_FAST     | CONTROL_SLOW      | task_consensus_p_high |         45 | -0.0216234   |   -0.00866803  |                     19 |                     26 |                  0 |                  449 |      0.44643  |                  -0.132367  |              -0.025142   |               0.0329393   |         0.669645 | False                  |
| modality_agnostic | arousal  | fault_effect | FAULTY_FAST      | CONTROL_FAST      | task_consensus_p_high |         46 | -0.000477296 |   -7.52119e-05 |                     23 |                     23 |                  0 |                  527 |      0.88814  |                   0.0249769 |              -0.0139141  |               0.0259344   |         1        | False                  |
| modality_agnostic | arousal  | speed_effect | CONTROL_FAST     | CONTROL_SLOW      | task_consensus_p_high |         45 | -0.0110089   |   -0.0106376   |                     17 |                     28 |                  0 |                  389 |      0.149778 |                  -0.248309  |              -0.0426181  |               6.48697e-09 |         0.374445 | False                  |
| modality_agnostic | valence  | fault_effect | FAULTY_FAST      | CONTROL_FAST      | task_consensus_p_high |         46 | -0.00869825  |   -0.000314501 |                     21 |                     25 |                  0 |                  456 |      0.362165 |                  -0.156337  |              -0.0355225  |               0.0104596   |         0.879597 | False                  |
| modality_agnostic | valence  | speed_effect | CONTROL_FAST     | CONTROL_SLOW      | task_consensus_p_high |         45 | -0.0302464   |   -0.0097678   |                     21 |                     24 |                  0 |                  416 |      0.256998 |                  -0.196135  |              -0.0335301  |               0.00997253  |         0.539733 | False                  |

## Ratings-only directional reference

Ratings-only results found FAULTY_FAST lower valence, higher arousal, and no clear CONTROL_FAST versus CONTROL_SLOW effect. Predictor-versus-rating direction is descriptive only; no merged inferential test was performed.

## Interpretation boundary

Friedman tests compare Observation, Interaction, and Alone without automated post-hoc tests. Wilcoxon tests are targeted FAULTY_FAST−CONTROL_FAST and CONTROL_FAST−CONTROL_SLOW comparisons. FDR correction is within each branch × target × metric family for task types and within each branch × target × contrast family for condition metrics.

