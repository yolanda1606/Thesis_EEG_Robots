# Image Experiment Data Quality Summary

## Dataset processed so far

37 official final runs were found: P01, P02, P03, P08, P10, P11, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P29, P30, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46. There are 36 full 120-trial sessions and 1 approved incomplete session (P03). The merged tables contain 4,374 matched trial rows (36 × 120 trials plus 54 P03 incomplete-session rows); 4,236 EEG trials were retained and 137 rejected. P03 also has one matched row without an EEG epoch because 54 image events yielded 53 pre-cleaning epochs.

## Preprocessing pipeline

The frozen pipeline uses 8 EEG channels, common-average reference, and a 1–40 Hz fourth-order Butterworth IIR filter. It uses rank-aware ICA (post-CAR rank expected 7), with one stable ICA decomposition fitted on concatenated valid image epochs. ACC X/Y/Z correlation is evaluated per trial; flagged motion-related ICA source activity is high-pass filtered at 3 Hz, then EEG is reconstructed. Baseline correction (-0.5 to 0.0 s) follows ICA reconstruction. AutoReject then runs with epoch-wise channel interpolation disabled. Features use 0–2 s and delta 1–4, theta 4–8, alpha 8–12, beta 12–30, and gamma 30–40 Hz bands. This is an adapted ACC-guided ICA preprocessing strategy, not a claim of perfect reproduction of the reference paper.

## Cross-machine integrity check

**The copied `derived/` final runs are safe to use for subsequent analysis on this Linux PC.** All 37 final runs were opened successfully: merged, EEG-feature, face-feature, and QC tables parsed, and cleaned FIF epochs opened with MNE. Windows absolute paths remain in provenance metadata, but none was an active path needed to open the copied outputs. No zero-byte expected outputs were found.

## EEG quality

Participant EEG retention was mean 96.9%, median 97.5%, SD 3.4%, range 85.0–100.0%, and quartiles 96.7/100.0%. Rejected epochs per participant ranged from 0 to 18 (median 3.0). Across processed EEG epochs, retention was 96.9% (137 rejected).

## ICA and motion correction

Motion correction affected mean 28.5 and median 28.0 trials per participant (24.0% and 23.3%). Using the descriptive 1.5×IQR rule (> 35.4%), unusually high motion-correction rates occurred in no participants. The descriptive association with AutoReject rejection was Pearson r=0.50 (p=0.002) and Spearman ρ=0.46 (p=0.004); this does not establish causation.

## Facial feature quality

Weighted face detection was mean 99.9%, median 100.0%, range 98.7–100.0%. Participants below 99%: P21; below 95%: none.

## Self-rating completeness

There are 27 missing valence ratings and 12 missing arousal ratings. Participants affected: P01, P02. Ratings are available for 99.4% of merged rows (valence) and 99.7% (arousal).

## Usable sample sizes for modeling

- EEG-only: 4,236 trial rows
- Face-only: 4,373 trial rows
- EEG + face: 4,236 trial rows
- EEG + face + valence: 4,211 trial rows
- EEG + face + arousal: 4,225 trial rows

AutoReject-rejected EEG trials remain in the merged table but have missing EEG feature values. Therefore many NaN cells can arise from relatively few rejected trial rows. Missing EEG does not automatically remove a face-only observation. Missing ratings reduce supervised-learning sample size for that target, but do not invalidate the raw physiological recording.

## Participants requiring additional review

P01, P02, P03, P08, P21, P29, P35, P36, P37. These QC groupings are descriptive and are not scientific exclusion decisions; see `modeling_readiness.csv` for transparent reasons.

## Main takeaways

The copied outputs are readable and internally usable on Linux. EEG quality is generally high, face detection is high, and the principal data-completeness limitation is concentrated in the incomplete P03 session and the missing ratings recorded for P01/P02. Model training was not performed.

## Questions for Friday's discussion

1. What participant-level EEG rejection rate should trigger exclusion versus simply removing individual trials?
2. Should participants with unusually high ACC-based motion correction remain in the primary cohort if their post-correction EEG passes AutoReject?
3. Is the <4 versus ≥4 valence/arousal threshold scientifically appropriate for final classification, or should participant-normalized labels or another formulation be considered?
4. Should final evaluation use leave-one-subject-out/group-aware validation rather than random trial splitting?
5. How should participants with substantial missing self-ratings be used: modality-specific training, partial-target analysis, or exclusion from the corresponding supervised target?
