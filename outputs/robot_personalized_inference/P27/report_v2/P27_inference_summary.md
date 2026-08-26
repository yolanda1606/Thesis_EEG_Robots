# P27 personalized Image-to-Robot inference

Face-only control maximum absolute ICA/no-ICA P(HIGH) difference: nan.

Predictions are frozen-model P(HIGH), not Robot emotion ground truth. Robot ratings are descriptive post-task comparisons only.

## Task-level descriptive consensus

| Task | Target | Condition | Median consensus P(HIGH) | Consensus verdict | Model agreement | Robot rating | Rating class | Match? | Shift warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pick_place | arousal | ICA | 0.8571428571428571 | HIGH | 0.9024390243902439 | 7.0 | HIGH | True | None |
| pick_place | arousal | no-ICA | 0.8571428571428571 | HIGH | 0.9024390243902439 | 7.0 | HIGH | True | None |
| shape_sorter_alone | arousal | ICA | 0.7142857142857143 | HIGH | 0.8275862068965517 | 4.0 | HIGH | True | None |
| shape_sorter_alone | arousal | no-ICA | 0.7142857142857143 | HIGH | 0.8275862068965517 | 4.0 | HIGH | True | None |
| shape_sorter_interaction | arousal | ICA | 0.7142857142857143 | HIGH | 0.7473684210526316 | 7.0 | HIGH | True | None |
| shape_sorter_interaction | arousal | no-ICA | 0.7142857142857143 | HIGH | 0.6947368421052632 | 7.0 | HIGH | True | None |
| shape_sorter_observation | arousal | ICA | 0.8571428571428571 | HIGH | 0.9333333333333333 | 6.0 | HIGH | True | None |
| shape_sorter_observation | arousal | no-ICA | 0.8571428571428571 | HIGH | 0.9333333333333333 | 6.0 | HIGH | True | None |
| sisyphus | arousal | ICA | 0.8571428571428571 | HIGH | 0.6929133858267716 | 7.0 | HIGH | True | None |
| sisyphus | arousal | no-ICA | 0.8571428571428571 | HIGH | 0.6929133858267716 | 7.0 | HIGH | True | None |
| stack | arousal | ICA | 0.8571428571428571 | HIGH | 0.9 | 7.0 | HIGH | True | None |
| stack | arousal | no-ICA | 0.8571428571428571 | HIGH | 0.9 | 7.0 | HIGH | True | None |
| pick_place | valence | ICA | 0.6807778783353082 | HIGH | 0.5365853658536586 | 6.0 | HIGH | True | ICA: R1 |
| pick_place | valence | no-ICA | 0.6807778783353404 | HIGH | 0.5365853658536586 | 6.0 | HIGH | True | no-ICA: R1 |
| shape_sorter_alone | valence | ICA | 0.7089907405509791 | HIGH | 0.5862068965517241 | 6.0 | HIGH | True | ICA: R1 |
| shape_sorter_alone | valence | no-ICA | 0.708990740550979 | HIGH | 0.5862068965517241 | 6.0 | HIGH | True | no-ICA: R1 |
| shape_sorter_interaction | valence | ICA | 0.6300159775027665 | HIGH | 0.4 | 5.0 | HIGH | True | ICA: R1 |
| shape_sorter_interaction | valence | no-ICA | 0.6788429242216241 | HIGH | 0.4842105263157895 | 5.0 | HIGH | True | no-ICA: R1 |
| shape_sorter_observation | valence | ICA | 0.6683809869432726 | HIGH | 0.5833333333333334 | 6.0 | HIGH | True | ICA: R1 |
| shape_sorter_observation | valence | no-ICA | 0.668380986943246 | HIGH | 0.5833333333333334 | 6.0 | HIGH | True | no-ICA: R1 |
| sisyphus | valence | ICA | 0.6970250117873119 | HIGH | 0.4251968503937008 | 7.0 | HIGH | True | ICA: R1 |
| sisyphus | valence | no-ICA | 0.6970250117873211 | HIGH | 0.4251968503937008 | 7.0 | HIGH | True | no-ICA: R1 |
| stack | valence | ICA | 0.6353742551596577 | HIGH | 0.42857142857142855 | 5.0 | HIGH | True | ICA: R1 |
| stack | valence | no-ICA | 0.6353742551596733 | HIGH | 0.42857142857142855 | 5.0 | HIGH | True | no-ICA: R1 |

See numerical CSVs for task summaries, condition-specific shift warnings, ICA/no-ICA comparisons, and top-three agreement.
