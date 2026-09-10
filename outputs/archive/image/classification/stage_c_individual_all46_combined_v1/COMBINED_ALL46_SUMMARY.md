# Combined Individual Classification: P01–P46

## Provenance

This directory combines completed summaries from `outputs/image_classification/stage_b_individual_v1` and `outputs/image_classification/stage_c_individual_added18_v1`. It is not a single training invocation and contains no retraining output.

The source manifests match exactly for individual mode, targets, modalities, classifiers, feature-count settings, nested StratifiedKFold CV logic, random seed (42), hyperparameter grids, and label definition. The source participant sets are disjoint and together cover P01–P46 exactly once.

## QC strata

- Strict QC-approved participants (30): P10, P11, P12, P14, P15, P16, P17, P18, P19, P20, P22, P23, P24, P25, P26, P27, P30, P31, P32, P33, P34, P38, P39, P40, P41, P42, P43, P44, P45, P46
- Technically usable but QC-caveat participants (16): P01, P02, P03, P04, P05, P06, P07, P08, P09, P13, P21, P28, P29, P35, P36, P37

## Participant-best balanced accuracy

- Valence: mean 0.605; median 0.598; SD 0.033; range 0.536–0.688; ≥0.55 43/46; ≥0.60 21/46; ≥0.65 4/46.
- Arousal: mean 0.592; median 0.587; SD 0.039; range 0.500–0.683; ≥0.55 39/46; ≥0.60 15/46; ≥0.65 5/46.

Participant-best figures select the highest completed nested-CV configuration for each participant and target. They are descriptive; selecting among many configurations is not an independent estimate of a pre-specified deployed model.
