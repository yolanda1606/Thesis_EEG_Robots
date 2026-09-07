# Old broad vs focused personalized classification: all modalities

This read-only comparison selects one best mean outer-CV BA per participant × target × modality in each search. It does not use configuration-row counts as evidence because the searches differ in candidate spaces.

## Eeg

| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Valence | 0.576 | 0.589 | +0.013 | 0.578 | 0.587 | +0.012 | 27/19/0 | 14 → 18 | 3 → 4 |
| Arousal | 0.568 | 0.585 | +0.016 | 0.566 | 0.584 | +0.011 | 33/13/0 | 10 → 16 | 4 → 5 |

## Face

| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Valence | 0.577 | 0.577 | +0.000 | 0.587 | 0.582 | +0.000 | 19/15/12 | 14 → 15 | 2 → 1 |
| Arousal | 0.559 | 0.563 | +0.004 | 0.566 | 0.567 | +0.000 | 20/14/12 | 7 → 12 | 2 → 1 |

## Multimodal

| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Valence | 0.581 | 0.575 | -0.006 | 0.583 | 0.582 | +0.000 | 17/19/10 | 16 → 16 | 2 → 4 |
| Arousal | 0.576 | 0.573 | -0.003 | 0.575 | 0.574 | -0.001 | 17/23/6 | 13 → 13 | 4 → 5 |

## Verdict

**EEG:** yes. The focused search improved participant-best mean and median BA for both targets (valence mean Δ +0.013; arousal mean Δ +0.016) and increased the BA ≥0.60 count from 14 to 18 for valence and from 10 to 16 for arousal.

**Face:** modest but uncertain improvement. Valence mean BA was effectively unchanged (+0.000) and its cohort median was lower; arousal mean BA increased by +0.004. The BA ≥0.60 count increased from 14 to 15 for valence and from 7 to 12 for arousal, but the BA ≥0.65 count fell to 1 and 1, respectively. This is not yet a strong broad cohort-level gain.

**Multimodal:** no general cohort-level improvement. Valence declined in both mean and median (mean Δ -0.006; median Δ +0.000); arousal mean and median also declined (-0.003; -0.001). Some threshold counts and the focused arousal maximum were higher, but these do not outweigh the typical participant-level results.

Overall, the focused search improved personalized classification for the EEG branch—the intended optimization target—and produced modest Face gains, but not a general Multimodal gain: mean ΔBA was positive in 4 of six target-modality combinations and negative in 2. This conclusion is based on participant-best comparisons, not incomparable configuration-row counts or a single maximum.

## Overall best personalized model regardless of modality

For each participant and target, this section selects the single highest mean outer-CV BA across EEG, Face, and Multimodal separately within the old and focused searches. Exact BA ties use the fixed order EEG, Face, Multimodal only to make metadata deterministic.

| Target | Old mean | Focused mean | Mean Δ | Old median | Focused median | Median Δ | Improved/worsened/tied | ≥0.60 old → focused | ≥0.65 old → focused | Max old → focused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Valence | 0.604 | 0.608 | +0.005 | 0.601 | 0.608 | +0.000 | 22/18/6 | 24 → 25 | 6 → 7 | 0.714 → 0.708 |
| Arousal | 0.597 | 0.609 | +0.012 | 0.595 | 0.608 | +0.005 | 29/13/4 | 19 → 27 | 7 → 8 | 0.678 → 0.704 |

Winning-modality counts:

| Target | Search | EEG | Face | Multimodal | Winner modality changed |
| --- | --- | ---: | ---: | ---: | ---: |
| Valence | Old | 14 | 21 | 11 |  |
| Valence | Focused | 17 | 18 | 11 | 25 |
| Arousal | Old | 16 | 17 | 13 |  |
| Arousal | Focused | 20 | 17 | 9 | 17 |

**Overall verdict:** Yes—after allowing all three modalities to compete per participant, the focused search changed mean participant-best BA by +0.005 for valence and +0.012 for arousal. The accompanying improved/worsened/tied and threshold counts above determine whether this is a cohort-level gain rather than a maximum-only result.
