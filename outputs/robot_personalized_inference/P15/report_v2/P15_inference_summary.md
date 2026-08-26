# P15 personalized Image-to-Robot inference

Face-only control maximum absolute ICA/no-ICA P(HIGH) difference: 0.

Predictions are frozen-model P(HIGH), not Robot emotion ground truth. Robot ratings are descriptive post-task comparisons only.

## Task-level descriptive consensus

| Task | Target | Condition | Median consensus P(HIGH) | Consensus verdict | Model agreement | Robot rating | Rating class | Match? | Shift warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pick_place | arousal | ICA | 0.6 | HIGH | 0.25 | 6.0 | HIGH | True | ICA: R2 |
| pick_place | arousal | no-ICA | 0.6 | HIGH | 0.25 | 6.0 | HIGH | True | no-ICA: R2 |
| shape_sorter_alone | arousal | ICA | 0.8 | HIGH | 0.5833333333333334 | 4.0 | HIGH | True | ICA: R2 |
| shape_sorter_alone | arousal | no-ICA | 0.8 | HIGH | 0.5833333333333334 | 4.0 | HIGH | True | no-ICA: R2 |
| shape_sorter_interaction | arousal | ICA | 0.9614424023910964 | HIGH | 0.75 | 6.0 | HIGH | True | ICA: R2 |
| shape_sorter_interaction | arousal | no-ICA | 0.9614424023910964 | HIGH | 0.75 | 6.0 | HIGH | True | no-ICA: R2 |
| shape_sorter_observation | arousal | ICA | 0.9599224568776215 | HIGH | 0.6730769230769231 | 6.0 | HIGH | True | ICA: R2 |
| shape_sorter_observation | arousal | no-ICA | 0.9599224568776215 | HIGH | 0.6730769230769231 | 6.0 | HIGH | True | no-ICA: R2 |
| sisyphus | arousal | ICA | 0.6 | HIGH | 0.0 | 6.0 | HIGH | True | ICA: R2, R3 |
| sisyphus | arousal | no-ICA | 0.8757557002859646 | HIGH | 0.6526315789473685 | 6.0 | HIGH | True | no-ICA: R2 |
| stack | arousal | ICA | 0.9751474648256124 | HIGH | 0.7540983606557377 | 7.0 | HIGH | True | ICA: R2 |
| stack | arousal | no-ICA | 0.9751474648256124 | HIGH | 0.7540983606557377 | 7.0 | HIGH | True | no-ICA: R2 |
| pick_place | valence | ICA | 0.06330203976162325 | LOW | 0.32 | 5.0 | HIGH | False | None |
| pick_place | valence | no-ICA | 0.06330203976162325 | LOW | 0.32 | 5.0 | HIGH | False | None |
| shape_sorter_alone | valence | ICA | 0.03351897213106835 | LOW | 0.6521739130434783 | 5.0 | HIGH | False | None |
| shape_sorter_alone | valence | no-ICA | 0.03351897213106835 | LOW | 0.6521739130434783 | 5.0 | HIGH | False | None |
| shape_sorter_interaction | valence | ICA | 0.12563628412347824 | LOW | 0.35294117647058826 | 6.0 | HIGH | False | None |
| shape_sorter_interaction | valence | no-ICA | 0.12563628412347824 | LOW | 0.35294117647058826 | 6.0 | HIGH | False | None |
| shape_sorter_observation | valence | ICA | 0.19480462461497766 | LOW | 0.2777777777777778 | 6.0 | HIGH | False | None |
| shape_sorter_observation | valence | no-ICA | 0.19480462461497766 | LOW | 0.2777777777777778 | 6.0 | HIGH | False | None |
| sisyphus | valence | ICA | 0.5096730694542276 | HIGH | 0.27102803738317754 | 5.0 | HIGH | True | ICA: R2 |
| sisyphus | valence | no-ICA | 0.5096730694542276 | HIGH | 0.37383177570093457 | 5.0 | HIGH | True | None |
| stack | valence | ICA | 0.2869776275322607 | LOW | 0.42028985507246375 | 6.0 | HIGH | False | None |
| stack | valence | no-ICA | 0.2869776275322607 | LOW | 0.42028985507246375 | 6.0 | HIGH | False | None |

See numerical CSVs for task summaries, condition-specific shift warnings, ICA/no-ICA comparisons, and top-three agreement.
