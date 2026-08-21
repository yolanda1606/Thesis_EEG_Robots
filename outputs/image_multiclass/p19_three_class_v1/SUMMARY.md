# P19 three-class classification pilot

Exploratory, leakage-safe nested-CV analysis. LOW=ratings 1--2, NEUTRAL=3--5, HIGH=6--7. Three-class balanced-accuracy chance is approximately 0.333; it is not numerically comparable to binary chance BA of 0.500.

## Class counts

| Target | LOW | NEUTRAL | HIGH |
|---|---:|---:|---:|
| Valence | 22 | 75 | 23 |
| Arousal | 31 | 73 | 16 |

All classes met the pilot's minimum count requirement (three rows per class) for nested stratified CV after complete-feature filtering.

## Best models

### Valence

Best configuration: **face / svm / 10 features**. Mean multiclass BA: **0.431**; macro F1: **0.407**. Per-class recall (fold mean): LOW=0.360, NEUTRAL=0.773, HIGH=0.160.

Across out-of-fold predictions, the most frequent paired error was **NEUTRAL vs HIGH** (22 directional errors combined). This indicates whether NEUTRAL is mainly confused with an adjacent affect level.

### Arousal

Best configuration: **multimodal / gnb / 5 features**. Mean multiclass BA: **0.373**; macro F1: **0.365**. Per-class recall (fold mean): LOW=0.314, NEUTRAL=0.756, HIGH=0.050.

Across out-of-fold predictions, the most frequent paired error was **LOW vs NEUTRAL** (36 directional errors combined). This indicates whether NEUTRAL is mainly confused with an adjacent affect level.

## Context

Existing P19 binary reference results were approximately BA=0.600 for valence and BA=0.683 for arousal. These are descriptive references only: a lower three-class BA does not by itself mean the three-class model is worse, because the chance baseline changes from 0.500 to about 0.333.
