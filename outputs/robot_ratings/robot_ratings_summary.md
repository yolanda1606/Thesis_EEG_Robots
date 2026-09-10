# Robot participant-ratings analysis

This is a ratings-only, read-only analysis of the original 1–7 Robot valence and arousal self-reports. It does not use EEG, model predictions, or transfer outputs.

## Validation

- Raw ratings: 276 rows (46 participants × 6 tasks).
- All valence and arousal ratings are within the original 1–7 scale.
- Task type mapping: Observation = PnP/SSObs/St; Interaction = Sisyphus/SSInt; Alone = SSAlone.
- Actual observation-condition counts: FAULTY_FAST=46, CONTROL_FAST=46, CONTROL_SLOW=45, FAULTY_SLOW=1.
- P14 Stack is retained as FAULTY_SLOW in descriptive outputs and is excluded only from the CONTROL_FAST versus CONTROL_SLOW contrast.
- P20 has planned-versus-actual speed deviations; actual metadata is used throughout.

## Task-level ratings

| task                     | task_display   | target   |   n |    mean |      sd |   median |   q1 |   q3 |   iqr |   min |   max |   mean_ci_95_low |   mean_ci_95_high |
|:-------------------------|:---------------|:---------|----:|--------:|--------:|---------:|-----:|-----:|------:|------:|------:|-----------------:|------------------:|
| pick_place               | PnP            | valence  |  46 | 4.71739 | 1.65547 |      5   | 4    |    6 |  2    |     1 |     7 |          4.22578 |           5.20901 |
| pick_place               | PnP            | arousal  |  46 | 4.6087  | 1.94911 |      5   | 3.25 |    6 |  2.75 |     1 |     7 |          4.02988 |           5.18751 |
| shape_sorter_observation | SSObs          | valence  |  46 | 5.41304 | 1.42324 |      6   | 4.25 |    7 |  2.75 |     2 |     7 |          4.99039 |           5.83569 |
| shape_sorter_observation | SSObs          | arousal  |  46 | 4.95652 | 1.61873 |      5   | 4    |    6 |  2    |     1 |     7 |          4.47582 |           5.43723 |
| stack                    | St             | valence  |  46 | 4.80435 | 1.49992 |      5   | 4    |    6 |  2    |     2 |     7 |          4.35893 |           5.24977 |
| stack                    | St             | arousal  |  46 | 4.78261 | 1.53352 |      5   | 4    |    6 |  2    |     2 |     7 |          4.32721 |           5.23801 |
| sisyphus                 | Sisyphus       | valence  |  46 | 5.5     | 1.36219 |      6   | 5    |    7 |  2    |     2 |     7 |          5.09548 |           5.90452 |
| sisyphus                 | Sisyphus       | arousal  |  46 | 5.67391 | 1.26587 |      6   | 5    |    7 |  2    |     2 |     7 |          5.298   |           6.04983 |
| shape_sorter_interaction | SSInt          | valence  |  46 | 5.91304 | 1.26185 |      6   | 5    |    7 |  2    |     3 |     7 |          5.53832 |           6.28777 |
| shape_sorter_interaction | SSInt          | arousal  |  46 | 5.08696 | 1.82362 |      5.5 | 4    |    7 |  3    |     1 |     7 |          4.54541 |           5.62851 |
| shape_sorter_alone       | SSAlone        | valence  |  46 | 5.15217 | 1.33279 |      5   | 4    |    6 |  2    |     1 |     7 |          4.75638 |           5.54796 |
| shape_sorter_alone       | SSAlone        | arousal  |  46 | 2.93478 | 1.85475 |      2.5 | 1    |    4 |  3    |     1 |     7 |          2.38399 |           3.48558 |

## Participant-level task-type aggregates

| target   | task_type   |   n |    mean |      sd |   median |      q1 |      q3 |     iqr |     min |     max |   mean_ci_95_low |   mean_ci_95_high |   friedman_paired_n |   friedman_chi_square |   friedman_p_value |
|:---------|:------------|----:|--------:|--------:|---------:|--------:|--------:|--------:|--------:|--------:|-----------------:|------------------:|--------------------:|----------------------:|-------------------:|
| valence  | Observation |  46 | 4.97826 | 1.03376 |  5       | 4.33333 | 5.66667 | 1.33333 | 2.66667 | 6.66667 |          4.67127 |           5.28525 |                  46 |               13.5092 |        0.0011655   |
| valence  | Interaction |  46 | 5.70652 | 1.05186 |  6       | 5       | 6.5     | 1.5     | 3.5     | 7       |          5.39416 |           6.01888 |                  46 |               13.5092 |        0.0011655   |
| valence  | Alone       |  46 | 5.15217 | 1.33279 |  5       | 4       | 6       | 2       | 1       | 7       |          4.75638 |           5.54796 |                  46 |               13.5092 |        0.0011655   |
| arousal  | Observation |  46 | 4.78261 | 1.20946 |  4.66667 | 4.08333 | 5.58333 | 1.5     | 2.33333 | 7       |          4.42344 |           5.14178 |                  46 |               40.7931 |        1.38641e-09 |
| arousal  | Interaction |  46 | 5.38043 | 1.36294 |  5.5     | 5       | 6.375   | 1.375   | 2       | 7       |          4.97569 |           5.78518 |                  46 |               40.7931 |        1.38641e-09 |
| arousal  | Alone       |  46 | 2.93478 | 1.85475 |  2.5     | 1       | 4       | 3       | 1       | 7       |          2.38399 |           3.48558 |                  46 |               40.7931 |        1.38641e-09 |

## Repeated-measures omnibus tests

| target   |   paired_n |   friedman_chi_square |   friedman_p_value |
|:---------|-----------:|----------------------:|-------------------:|
| valence  |         46 |               13.5092 |        0.0011655   |
| arousal  |         46 |               40.7931 |        1.38641e-09 |

## Pre-specified actual-condition contrasts

| comparison   | target   | left_condition   | right_condition   |   paired_n |   mean_delta |   median_delta |   median_delta_ci_95_low |   median_delta_ci_95_high |   positive_delta_count |   negative_delta_count |   zero_delta_count |   wilcoxon_statistic |   wilcoxon_p_value |   rank_biserial_correlation |
|:-------------|:---------|:-----------------|:------------------|-----------:|-------------:|---------------:|-------------------------:|--------------------------:|-----------------------:|-----------------------:|-------------------:|---------------------:|-------------------:|----------------------------:|
| fault_effect | valence  | FAULTY_FAST      | CONTROL_FAST      |         46 |   -1.67391   |           -1.5 |                       -2 |                        -1 |                      2 |                     34 |                 10 |                 24   |        9.85066e-07 |                   -0.927928 |
| speed_effect | valence  | CONTROL_FAST     | CONTROL_SLOW      |         45 |    0.0888889 |            0   |                        0 |                         1 |                     17 |                     15 |                 13 |                236.5 |        0.590948    |                    0.104167 |
| fault_effect | arousal  | FAULTY_FAST      | CONTROL_FAST      |         46 |    0.804348  |            0   |                        0 |                         1 |                     22 |                     13 |                 11 |                154   |        0.00756093  |                    0.511111 |
| speed_effect | arousal  | CONTROL_FAST     | CONTROL_SLOW      |         45 |   -0.111111  |            0   |                       -1 |                         0 |                     16 |                     18 |                 11 |                263.5 |        0.555947    |                   -0.114286 |

## Interpretation boundary

The Wilcoxon tests are the two pre-specified paired contrasts separately for valence and arousal. The task-type results are Friedman omnibus tests; no unplanned post-hoc pairwise testing was performed. Effect size is rank-biserial correlation for non-zero paired differences. Paired-difference confidence intervals are deterministic percentile-bootstrap 95% intervals for the median difference.

## Metadata deviations recorded

| participant   | task       | planned_condition   | planned_speed   | actual_condition   | actual_speed   |
|:--------------|:-----------|:--------------------|:----------------|:-------------------|:---------------|
| P14           | stack      | CONTROL             | SLOW            | FAULTY             | SLOW           |
| P20           | pick_place | CONTROL             | FAST            | CONTROL            | SLOW           |
| P20           | stack      | CONTROL             | SLOW            | CONTROL            | FAST           |

