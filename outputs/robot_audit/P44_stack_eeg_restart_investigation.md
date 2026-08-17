# P44 Stack EEG restart investigation

Read-only investigation. No BDF files were concatenated or preprocessed, and no raw data, pipeline code, derived data, or participant configuration was modified.

## Recommendation: A

**All three `UnicornRecorder` BDFs are consecutive fragments of one Stack execution and should later be represented as one segmented EEG recording.**

The conclusion is supported by the unique, ordered Status-trigger progression across the fragments and its direct correspondence to the one continuous Stack vision log and robot-metrics task sequence. The recorder has missing intervals; these must remain gaps, not be interpolated.

## BDF fragment timing and Status evidence

`absolute header start/end` are MNE/BDF header values. The filename-derived start labels conflict materially with those header values for fragments 2–3; both are retained below. The header timestamps therefore must not be used alone to infer physical overlap or run identity.

|Order|Recorder BDF|Absolute header start|Header-derived end|Duration / samples / rate|Status codes (relative BDF time)|Header-clock trigger time range|
|---|---|---|---|---|---|---|
|1|`D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_35_13.bdf`|2026-07-24 14:35:26.000 UTC|14:35:38.608 UTC|12.608 s; 3,152; 250 Hz|1@5.544, 11@5.972, 12@10.592 s|1: 14:35:31.544; 12: 14:35:36.592 UTC|
|2|`D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_35_33.bdf`|2026-07-24 14:36:41.000 UTC|14:37:48.172 UTC|67.172 s; 16,793; 250 Hz|16@1.644, 17@5.700, 18@8.404, 19@10.048, 21@12.308, 22@16.816, 23@19.520, 24@21.556, 25@23.364, 26@28.284, 27@32.340, 28@35.052, 29@36.660, 31@38.944, 32@43.444, 33@46.140, 34@48.092, 35@49.892, 36@54.840, 37@58.768, 38@61.428, 39@63.060, 41@65.308 s|16: 14:36:42.644; 41: 14:37:46.308 UTC|
|3|`D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_36_49.bdf`|2026-07-24 14:37:16.000 UTC|14:37:42.372 UTC|26.372 s; 6,593; 250 Hz|45@0.400, 46@5.284, 47@9.328, 48@12.056, 49@13.696, 90@16.004, 99@23.204 s|45: 14:37:16.400; 99: 14:37:39.204 UTC|

### Fragment order, gaps, and apparent overlap

- Filename labels order the Recorder fragments as 14:35:13, 14:35:33, and 14:36:49. They are also consistent with the vision-log trigger progression.
- Under the BDF header timestamps, fragment 1 ends 62.392 s before fragment 2 starts; fragments 2 and 3 overlap by 32.172 s. This is an internal timestamp inconsistency, not evidence of duplicate task execution: fragment 3’s codes begin where fragment 2’s codes stop in the Stack timeline.
- Under the filename labels plus file durations, the nominal recorder-off gaps are approximately 7.392 s (fragment 1 → 2) and 8.828 s (fragment 2 → 3). These nominal gaps are compatible with the experimenter’s note that the EEG device stopped during Stack.
- On the independent vision-task clock, BDF-covered trigger regions are separated by 11.405 s from trigger 12 to 16 and 10.971 s from trigger 41 to 45. The omitted task events fall within those intervals: 13–15 and 42–44 respectively. This is the practical EEG coverage gap for any later timeline representation.
- No Status code is duplicated across the three Recorder fragments. The sequence is sensible and continuous at the task level: `1,11,12` → `16…41` → `45…99`.

## Behavioural/task evidence

|Artifact|Timing/evidence|
|---|---|
|Video `vision_video_14-35-17_STACK_CONTROL_SLOW.avi`|3,487 frames, 30 fps, 116.233 s, 640×480. Filename start label 14:35:17.|
|Vision log `vision_log_14-35-17_STACK_CONTROL_SLOW.csv`|3,487 rows; wall-clock 14:35:18.424–14:37:15.339; `Experiment_Time` 0.445–117.360 s; Status/trigger sequence runs once from 1 through 99.|
|Robot metrics `robot_metrics_14-37-12_STACK_CONTROL_SLOW.csv`|98,670 rows; relative `Experiment_Time` 0.000–99.028 s. It contains one `START` (1), complete Cube 1–4 task progression, `RETURNING_HOME` (90), and `END` (99). The filename time is near the vision-log 99 event (14:37:12.904), so it appears to reflect file completion rather than a documented metrics start time.|

The vision log and robot metrics both show one normal four-cube Stack completion. Their event codes provide the external timeline that associates fragment 1 with the early task, fragment 2 with the middle task, and fragment 3 with the final Cube 4/return-home/end phase. Nothing in their sequence indicates a reset or a second Stack attempt.

## Existing YAML state

`processing/multimodal_robot/configs/participants/P44.yaml` currently lists all three Recorder BDFs (and their RawDataRecorder companions) as candidates, with `file: null`, `review_required: true`, and `multimodal_usable: false` for Stack. That is appropriately conservative before this investigation; it was not changed.

## Future YAML representation (proposal only; do not apply in this investigation)

Represent the selected EEG as an ordered segment list, preserving each raw file and its discontinuity rather than creating a combined BDF:

```yaml
eeg:
  segments:
    - file: D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_35_13.bdf
      order: 1
      status_codes: [1, 11, 12]
    - file: D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_35_33.bdf
      order: 2
      status_codes: [16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41]
    - file: D:\Thesis\data\P44_2026-07-24\Stack\EEG\UnicornRecorder_24_07_2026_14_36_49.bdf
      order: 3
      status_codes: [45, 46, 47, 48, 49, 90, 99]
  usable: true
  review_required: false
  discontinuities_preserved: true
  notes: Recorder-off gaps exist between segments; never interpolate EEG across them.
```

Future processing may apply identical frozen preprocessing independently per raw segment and join only the resulting feature/timeline representation by known task time and Status codes. It must retain the two recorder-off gaps and label the corresponding missing EEG intervals.
