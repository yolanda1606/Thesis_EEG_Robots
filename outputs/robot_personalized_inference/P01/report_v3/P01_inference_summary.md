# P01 personalized Image-to-Robot inference

Face-only control maximum absolute ICA/no-ICA P(HIGH) difference: 0.

Predictions are frozen-model P(HIGH), not Robot emotion ground truth. Robot ratings are descriptive post-task comparisons only.

## Task-level descriptive consensus

| Task | Target | Condition | Median consensus P(HIGH) | Consensus verdict | Model agreement | Robot rating | Rating class | Match? | Shift warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pick_place | arousal | ICA | 0.5725788473604116 | HIGH | 0.0 | 5.0 | HIGH | True | ICA: R1 |
| pick_place | arousal | no-ICA | 0.5725788473604116 | HIGH | 0.0 | 5.0 | HIGH | True | no-ICA: R1 |
| shape_sorter_alone | arousal | ICA | 0.5555431693458013 | HIGH | 0.0 | 1.0 | LOW | False | ICA: R1 |
| shape_sorter_alone | arousal | no-ICA | 0.5555431693458013 | HIGH | 0.0 | 1.0 | LOW | False | no-ICA: R1 |
| shape_sorter_interaction | arousal | ICA | 0.5092632894912932 | HIGH | 0.013513513513513514 | 5.0 | HIGH | True | None |
| shape_sorter_interaction | arousal | no-ICA | 0.5092632894912932 | HIGH | 0.013513513513513514 | 5.0 | HIGH | True | None |
| shape_sorter_observation | arousal | ICA | 0.5212495674723514 | HIGH | 0.018518518518518517 | 6.0 | HIGH | True | ICA: R1 |
| shape_sorter_observation | arousal | no-ICA | 0.5212495674723514 | HIGH | 0.018518518518518517 | 6.0 | HIGH | True | no-ICA: R1 |
| sisyphus | arousal | ICA | 0.6158097525636586 | HIGH | 0.02127659574468085 | 5.0 | HIGH | True | None |
| sisyphus | arousal | no-ICA | 0.6158097525636586 | HIGH | 0.02127659574468085 | 5.0 | HIGH | True | None |
| stack | arousal | ICA | 0.4903536746193199 | LOW | 0.046875 | 5.0 | HIGH | False | ICA: R1 |
| stack | arousal | no-ICA | 0.4903536746193199 | LOW | 0.046875 | 5.0 | HIGH | False | no-ICA: R1 |
| pick_place | valence | ICA | 0.6473504250003228 | HIGH | 0.4883720930232558 | 5.0 | HIGH | True | ICA: R3 |
| pick_place | valence | no-ICA | 0.5302369129979838 | HIGH | 0.4883720930232558 | 5.0 | HIGH | True | no-ICA: R3 |
| shape_sorter_alone | valence | ICA | 0.8325102046080923 | HIGH | 0.4666666666666667 | 6.0 | HIGH | True | ICA: R3 |
| shape_sorter_alone | valence | no-ICA | 0.6321447069314525 | HIGH | 0.4666666666666667 | 6.0 | HIGH | True | no-ICA: R3 |
| shape_sorter_interaction | valence | ICA | 0.7886039719217215 | HIGH | 0.43373493975903615 | 3.0 | LOW | False | None |
| shape_sorter_interaction | valence | no-ICA | 0.6389413624040005 | HIGH | 0.40963855421686746 | 3.0 | LOW | False | None |
| shape_sorter_observation | valence | ICA | 0.9258183620580058 | HIGH | 0.5555555555555556 | 7.0 | HIGH | True | ICA: R3 |
| shape_sorter_observation | valence | no-ICA | 0.9258183620580058 | HIGH | 0.42857142857142855 | 7.0 | HIGH | True | no-ICA: R3 |
| sisyphus | valence | ICA | 0.988677734022281 | HIGH | 0.8181818181818182 | 5.0 | HIGH | True | None |
| sisyphus | valence | no-ICA | 0.988677734022281 | HIGH | 0.7386363636363636 | 5.0 | HIGH | True | None |
| stack | valence | ICA | 0.9247496171305987 | HIGH | 0.5571428571428572 | 5.0 | HIGH | True | ICA: R3 |
| stack | valence | no-ICA | 0.9247496171305987 | HIGH | 0.5714285714285714 | 5.0 | HIGH | True | no-ICA: R3 |

See numerical CSVs for task summaries, condition-specific shift warnings, ICA/no-ICA comparisons, and top-three agreement.
