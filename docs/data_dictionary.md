# Data dictionary

## General rules

- `participant_id` uses a pseudonymous identifier such as `P01`.
- Raw source files remain under `data/` and are never changed by analysis work.
- Derived tables, figures, reports, models, and documentation must be stored outside participant folders in `data/`.
- A missing value is not the same as a value of zero or a negative result.
- A condition that is not applicable is not the same as an unknown condition.

## Participant table

One row per participant.

| Field | Meaning | Allowed values or format |
|---|---|---|
| `participant_id` | Pseudonymous participant identifier | `P##` |
| `collection_date` | Date of data collection | `YYYY-MM-DD` |
| `assigned_group` | Group planned before recording | `A`, `B`, `C`, `unknown` |
| `protocol_version` | Protocol version, if documented | text or `unknown` |
| `participant_status` | Current data status | `active`, `complete`, `excluded`, `unknown` |
| `notes_reference` | Reference to a non-identifying study note | text or missing |

## Recording manifest

One row per recorded run. This table is the file-level inventory and may contain more than one row for the same participant and task.

| Field | Meaning |
|---|---|
| `participant_id` | Participant identifier |
| `task` | Recorded task name |
| `run_id` | Unique identifier for this recording run |
| `recording_start_time` | Best available recording start time |
| `source_eeg_file` | Raw EEG file path |
| `source_video_file` | Raw video file path |
| `source_vision_log_file` | Raw vision-log file path |
| `source_robot_metrics_file` | Raw robot-metrics file path, where applicable |
| `run_selected` | Whether this is the approved run for analysis |
| `selection_reason` | Reason for selecting or not selecting the run |
| `recording_qc_status` | `pass`, `review`, `fail`, or `not_checked` |
| `qc_notes` | Short explanation of a QC result |

## Robot condition-level analysis table

One row per participant, selected run, and robot task.

| Field | Meaning |
|---|---|
| `participant_id` | Participant identifier |
| `task` | `pick_and_place`, `shape_sorter_observation`, `stack`, `sisyphus`, `shape_sorter_interaction`, or `shape_sorter_alone` |
| `run_id` | Selected recording-run identifier |
| `task_order` | Order of this task within the robot session |
| `observation_position` | `first`, `second`, or `third` for observation tasks; otherwise `not_applicable` |
| `assigned_group` | Planned group |
| `planned_behavior_condition` | `control`, `faulty`, or `unknown` |
| `planned_speed_condition` | `fast`, `slow`, or `unknown` |
| `executed_behavior_condition` | `control`, `faulty`, or `unknown` |
| `executed_speed_condition` | `fast`, `slow`, or `unknown` |
| `planned_condition` | See valid condition values below |
| `executed_condition` | See valid condition values below |
| `protocol_compliance` | `compliant`, `mismatch`, or `not_assessable` |
| `evidence_level` | Strength of evidence for executed condition |
| `valence_rating` | Condition-level participant rating, 1 to 7 |
| `arousal_rating` | Condition-level participant rating, 1 to 7 |
| `rating_qc_status` | `pass`, `review`, `fail`, or `missing` |
| `recording_qc_status` | QC status for the selected run |

## OASIS trial-level analysis table

One row per participant-image trial.

| Field | Meaning |
|---|---|
| `participant_id` | Participant identifier |
| `session_id` | Experiment session identifier |
| `segment_id` | EEG segment identifier, if recording was restarted |
| `trial_index` | Order of the image trial |
| `stim_id` | Image file or stimulus identifier |
| `stimulus_category` | Intended category: `HAHV`, `HALV`, `LAHV`, or `LALV` |
| `trigger_code` | Image trigger code |
| `image_onset_time` | Best available image-onset time |
| `valence_rating` | Observed trial-level rating, 1 to 7 |
| `valence_rt_s` | Valence response time in seconds |
| `arousal_rating` | Observed trial-level rating, 1 to 7 |
| `arousal_rt_s` | Arousal response time in seconds |
| `rating_match_status` | Whether the rating row and EEG event were matched |
| `trial_qc_status` | `pass`, `review`, `fail`, or `missing` |

## EEG feature table

One row per selected EEG epoch or time window.

| Field | Meaning |
|---|---|
| `participant_id` | Participant identifier |
| `run_id` | Recording-run identifier |
| `task` | Task name |
| `analysis_level` | `oasis_trial`, `robot_condition`, or `robot_event_window` |
| `epoch_id` | Unique epoch or window identifier |
| `event_code` | Trigger or event code used to define the epoch |
| `window_start_s` | Window start relative to the event |
| `window_end_s` | Window end relative to the event |
| `preprocessing_version` | Version or dated description of preprocessing settings |
| `eeg_qc_status` | `pass`, `review`, or `fail` |
| `feature_*` | EEG feature columns, named by channel and measure |

## Facial-feature table

One row per video frame or approved time window.

| Field | Meaning |
|---|---|
| `participant_id` | Participant identifier |
| `run_id` | Recording-run identifier |
| `task` | Task name |
| `frame_index` | Video-frame number, when frame-level |
| `timestamp_s` | Time from video start |
| `event_code` | Trigger matched to the frame or window, if available |
| `face_detected` | `0` or `1` |
| `detection_confidence` | Detector confidence, if the method provides it |
| `facial_qc_status` | `pass`, `review`, or `fail` |
| `feature_*` | Facial landmark, geometry, pose, or other approved facial-feature columns |

## Valid condition values

| Field | Allowed values |
|---|---|
| `behavior_condition` | `control`, `faulty`, `unknown` |
| `speed_condition` | `fast`, `slow`, `unknown` |
| `planned_condition` | `control_fast`, `control_slow`, `faulty_fast`, `faulty_slow`, `unknown` |
| `executed_condition` | `control_fast`, `control_slow`, `faulty_fast`, `faulty_slow`, `unknown` |
| `protocol_compliance` | `compliant`, `mismatch`, `not_assessable` |
| `evidence_level` | `confirmed`, `strongly_supported`, `probable`, `unknown` |

## Missing values, run selection, and QC

- Use blank or `NA` for a missing numeric measurement.
- Use `unknown` for a categorical value that cannot be determined.
- Use `not_applicable` when a field does not apply to that task.
- Never replace a missing OASIS participant rating with the intended stimulus category.
- `run_selected` is `1` for the approved run and `0` for all other runs.
- `selection_reason` records why a run was selected or retained as unselected.
- QC fields describe data quality only. They must not silently change planned or executed condition labels.
