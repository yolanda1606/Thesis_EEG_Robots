# Data health checks

## Purpose

This checklist is for reviewing present and future participant data without changing raw files. It should be applied to every participant folder found in `data/`, rather than to a fixed participant list.

The checks produce an inventory or review record outside raw participant directories. They do not preprocess data or run statistical or machine-learning analyses.

## 1. Participant completeness

For every participant folder that follows the project naming pattern:

- confirm the participant identifier and collection date can be read from the folder name;
- list available task folders;
- record whether image, observation, interaction, and individual-task recordings are present;
- distinguish missing files from intentionally non-applicable files;
- record incomplete, interrupted, or repeated runs without deleting them.

## 2. EEG files and channels

For every EEG recording:

- record BDF and companion CSV file names, sizes, and timestamps;
- inspect header information without loading the full dataset where possible;
- confirm expected channel names, including eight EEG channels and the `Status` channel;
- record sampling frequency, recording duration, and number of segments;
- flag unreadable files, missing channels, unexpected channel names, or unusual duration;
- keep restarted recordings as separate segments until event matching is reviewed.

## 3. Event markers

### OASIS image experiment

- check for fixation, image, valence, and arousal markers;
- check image trigger ranges and compare available image events with rating triggers;
- flag missing, duplicate, or out-of-order markers;
- record the number of matched trials without replacing missing ratings.

### Robot tasks

- check task start and end markers;
- check that robot metrics, video logs, and EEG events can be associated with the same run;
- for observation tasks, check task-specific fault markers when the executed condition is faulty;
- record missing markers as a QC issue rather than changing the condition label.

## 4. Ratings

### OASIS ratings

- confirm required columns: stimulus ID, category, trigger, valence, and arousal;
- identify duplicate exports and choose an authoritative export by documented completeness rules;
- check rating range is 1 to 7;
- record missing valence and arousal separately;
- do not impute missing ratings from stimulus category.

### Robot ratings

- check that there is one rating row per participant and robot task;
- check that valence and arousal are in the expected 1 to 7 range;
- flag duplicate participant-task rows, missing rows, or unclear task names;
- link the rating to the selected recording run through the manifest.

## 5. Video and vision logs

- confirm that each selected recording has a video and a companion vision log where expected;
- record frame rate, frame count, duration, and resolution;
- check that frame counts are continuous and elapsed time is monotonic;
- check that trigger values appear where expected;
- record face-detection coverage and detector failures when facial features are used;
- preserve raw video and original vision logs unchanged.

## 6. Robot metrics

- check that metrics files have start and end records;
- confirm that filename profile, trigger sequence, and waypoint stages agree;
- confirm control or faulty behaviour from task-specific evidence where available;
- confirm fast or slow profile from recorded profile and duration pattern;
- flag profiles not planned by the protocol, such as Faulty/Slow Stack;
- do not classify a recording from duration alone when profile or trigger evidence exists.

## 7. Synchronization

- list the EEG, video, vision-log, and robot-metrics files associated with each run;
- compare file timestamps and recording starts;
- identify shared trigger codes across systems;
- record any known offset, drift, restart, or pairing uncertainty;
- mark synchronization as `pass`, `review`, or `fail` for the intended analysis level.

## 8. Repeated runs

- identify more than one recording for the same participant and task;
- retain every raw run in the recording manifest;
- do not merge or delete repeated runs;
- require `run_selected` and `selection_reason` before producing one-row-per-task analysis data;
- record whether all repeated runs have the same executed condition or conflicting conditions.

## 9. Participant inclusion and exclusion

Keep these questions separate:

1. Is the raw recording present?
2. Does the recording pass technical quality checks?
3. Does the executed condition match the planned group assignment?
4. Is the participant included in a particular analysis dataset?

A protocol mismatch does not automatically mean that data are technically invalid. The main robot analysis uses executed conditions. The protocol-compliant sensitivity analysis uses participants whose executed observation-task matrix matches their assigned matrix.

## 10. Current observation-task items requiring attention

- P14 has an unplanned Faulty/Slow Stack recording.
- P20, P24, P25, P27, and P28 have confirmed planned/executed assignment mismatches.
- P04 has two compliant Pick-and-Place Control/Slow runs and needs an approved run-selection decision.

These are current audit findings. Future participant checks must discover new cases from the manifest rather than assume this list is complete.
