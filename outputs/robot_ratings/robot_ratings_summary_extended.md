# Extended Robot participant-ratings analysis

This ratings-only extension uses no EEG, model predictions, or transfer outputs.

## Task-level ratings

Unchanged from `task_level_summary.csv` and the original ratings-only summary.

## Task-type ratings

Unchanged from `task_type_summary.csv` and the original ratings-only summary.

## Fault effect

Unchanged from `condition_comparison.csv`; paired figures remain unchanged.

## Speed effect

Unchanged from `condition_comparison.csv`; paired figures remain unchanged.

## Valence-arousal space

Unchanged; see `valence_arousal_space.png`.

## Class imbalance

LOW is defined as rating <4 and HIGH as rating ≥4 only to describe later deployment-validation labels. This does not alter the original 1–7 ratings analysis.

| scope   | group   | target   |   n_total |   n_low |   n_high |   proportion_low |   proportion_high |   high_to_low_ratio |
|:--------|:--------|:---------|----------:|--------:|---------:|-----------------:|------------------:|--------------------:|
| overall | Overall | arousal  |       276 |      75 |      201 |         0.271739 |          0.728261 |             2.68    |
| overall | Overall | valence  |       276 |      39 |      237 |         0.141304 |          0.858696 |             6.07692 |

Most LOW-sparse task × target distributions:

| scope   | group    | target   |   n_total |   n_low |   n_high |   proportion_low |   proportion_high |   high_to_low_ratio |
|:--------|:---------|:---------|----------:|--------:|---------:|-----------------:|------------------:|--------------------:|
| task    | SSAlone  | valence  |        46 |       3 |       43 |        0.0652174 |          0.934783 |             14.3333 |
| task    | Sisyphus | arousal  |        46 |       4 |       42 |        0.0869565 |          0.913043 |             10.5    |
| task    | SSInt    | valence  |        46 |       4 |       42 |        0.0869565 |          0.913043 |             10.5    |
| task    | Sisyphus | valence  |        46 |       4 |       42 |        0.0869565 |          0.913043 |             10.5    |

Accuracy = (TP + TN) / N. Balanced Accuracy = 0.5 × (Sensitivity_HIGH + Specificity_LOW). When HIGH is much more common than LOW, accuracy can appear acceptable despite poor LOW performance; balanced accuracy weights the two classes equally.

## Exploratory demographic associations

Demographic tests are exploratory. Age uses Spearman correlation with deterministic bootstrap 95% CIs. Gender and prior experience use Mann–Whitney U with Cliff's delta. Benjamini–Hochberg FDR correction is applied separately across the 12 participant-level measures within each demographic family.

| demographic            | measure             |   n | effect_size_name                   |   effect_size |   p_value_raw |   p_value_fdr_bh | fdr_significant_0_05   |
|:-----------------------|:--------------------|----:|:-----------------------------------|--------------:|--------------:|-----------------:|:-----------------------|
| age                    | overall_valence     |  46 | spearman_rho                       |    -0.128542  |     0.394569  |         0.676403 | False                  |
| age                    | overall_arousal     |  46 | spearman_rho                       |    -0.170687  |     0.256735  |         0.676403 | False                  |
| age                    | observation_valence |  46 | spearman_rho                       |    -0.0428697 |     0.777265  |         0.807086 | False                  |
| age                    | observation_arousal |  46 | spearman_rho                       |    -0.136311  |     0.366372  |         0.676403 | False                  |
| age                    | interaction_valence |  46 | spearman_rho                       |    -0.0780118 |     0.606324  |         0.727589 | False                  |
| age                    | interaction_arousal |  46 | spearman_rho                       |    -0.0961411 |     0.525043  |         0.700058 | False                  |
| age                    | alone_valence       |  46 | spearman_rho                       |    -0.314694  |     0.0331658 |         0.39799  | False                  |
| age                    | alone_arousal       |  46 | spearman_rho                       |    -0.147478  |     0.328034  |         0.676403 | False                  |
| age                    | fault_valence_delta |  46 | spearman_rho                       |     0.103654  |     0.493017  |         0.700058 | False                  |
| age                    | fault_arousal_delta |  46 | spearman_rho                       |     0.25685   |     0.0848626 |         0.509176 | False                  |
| age                    | speed_valence_delta |  45 | spearman_rho                       |    -0.228929  |     0.130368  |         0.521471 | False                  |
| age                    | speed_arousal_delta |  45 | spearman_rho                       |    -0.0374421 |     0.807086  |         0.807086 | False                  |
| gender                 | overall_valence     |  46 | cliffs_delta_group_1_minus_group_2 |    -0.0666667 |     0.706891  |         0.833117 | False                  |
| gender                 | overall_arousal     |  46 | cliffs_delta_group_1_minus_group_2 |    -0.152381  |     0.382489  |         0.806138 | False                  |
| gender                 | observation_valence |  46 | cliffs_delta_group_1_minus_group_2 |     0.0380952 |     0.833117  |         0.833117 | False                  |
| gender                 | observation_arousal |  46 | cliffs_delta_group_1_minus_group_2 |     0.125714  |     0.470247  |         0.806138 | False                  |
| gender                 | interaction_valence |  46 | cliffs_delta_group_1_minus_group_2 |    -0.0590476 |     0.737063  |         0.833117 | False                  |
| gender                 | interaction_arousal |  46 | cliffs_delta_group_1_minus_group_2 |    -0.28381   |     0.0980999 |         0.3924   | False                  |
| gender                 | alone_valence       |  46 | cliffs_delta_group_1_minus_group_2 |    -0.127619  |     0.450681  |         0.806138 | False                  |
| gender                 | alone_arousal       |  46 | cliffs_delta_group_1_minus_group_2 |    -0.308571  |     0.0697    |         0.3924   | False                  |
| gender                 | fault_valence_delta |  46 | cliffs_delta_group_1_minus_group_2 |    -0.0685714 |     0.69429   |         0.833117 | False                  |
| gender                 | fault_arousal_delta |  46 | cliffs_delta_group_1_minus_group_2 |     0.04      |     0.823028  |         0.833117 | False                  |
| gender                 | speed_valence_delta |  45 | cliffs_delta_group_1_minus_group_2 |     0.178571  |     0.297427  |         0.806138 | False                  |
| gender                 | speed_arousal_delta |  45 | cliffs_delta_group_1_minus_group_2 |     0.299603  |     0.0833467 |         0.3924   | False                  |
| prior_robot_experience | overall_valence     |  46 | cliffs_delta_group_1_minus_group_2 |    -0.274102  |     0.112607  |         0.542035 | False                  |
| prior_robot_experience | overall_arousal     |  46 | cliffs_delta_group_1_minus_group_2 |    -0.221172  |     0.201431  |         0.571317 | False                  |
| prior_robot_experience | observation_valence |  46 | cliffs_delta_group_1_minus_group_2 |    -0.26276   |     0.12736   |         0.542035 | False                  |
| prior_robot_experience | observation_arousal |  46 | cliffs_delta_group_1_minus_group_2 |    -0.143667  |     0.406533  |         0.571317 | False                  |
| prior_robot_experience | interaction_valence |  46 | cliffs_delta_group_1_minus_group_2 |    -0.1569    |     0.360598  |         0.571317 | False                  |
| prior_robot_experience | interaction_arousal |  46 | cliffs_delta_group_1_minus_group_2 |    -0.122873  |     0.476098  |         0.571317 | False                  |
| prior_robot_experience | alone_valence       |  46 | cliffs_delta_group_1_minus_group_2 |    -0.189036  |     0.259686  |         0.571317 | False                  |
| prior_robot_experience | alone_arousal       |  46 | cliffs_delta_group_1_minus_group_2 |    -0.253308  |     0.135509  |         0.542035 | False                  |
| prior_robot_experience | fault_valence_delta |  46 | cliffs_delta_group_1_minus_group_2 |    -0.128544  |     0.453531  |         0.571317 | False                  |
| prior_robot_experience | fault_arousal_delta |  46 | cliffs_delta_group_1_minus_group_2 |    -0.162571  |     0.343678  |         0.571317 | False                  |
| prior_robot_experience | speed_valence_delta |  45 | cliffs_delta_group_1_minus_group_2 |    -0.0652174 |     0.70848   |         0.772887 | False                  |
| prior_robot_experience | speed_arousal_delta |  45 | cliffs_delta_group_1_minus_group_2 |     0.0494071 |     0.782159  |         0.782159 | False                  |

FDR-significant rows:

None.

## Data/linkage limitations

The demographic linkage is reconstructed from the ordered `#` field (# 1 → P01 through # 46 → P46), not from a complete original ID field. P01/P02 are directly confirmed; P14 is independently corroborated by the FAULTY_SLOW Stack exception; acquisition chronology supports the sequence. Rows 26–32 have scheduling/acquisition-date differences consistent with rescheduling. The demographic-sheet Group values for rows 27/28 conflict with the canonical YAMLs and were never used; all experimental group/task/condition metadata comes only from canonical participant YAML files.

