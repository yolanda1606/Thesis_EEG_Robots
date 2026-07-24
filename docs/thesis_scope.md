# Thesis scope

## Working title

**Affective responses during emotional image viewing and human-robot interaction: EEG, facial behaviour, and self-reported ratings**

## Study objective

This thesis examines emotional and affective responses during two parts of an experiment:

1. An OASIS image experiment, where participants viewed emotional images and rated their valence and arousal after each image.
2. Robot tasks, where participants completed or observed robot activities while EEG and participant video were recorded. After each robot condition, participants provided one valence rating and one arousal rating.

The current dataset contains 44 participants. More participants may be collected later. Participant counts in reports must therefore state the date of the report and be recalculated when new data are added.

## Main research questions

1. During the OASIS image experiment, how are trial-level valence and arousal ratings related to EEG features and facial features?
2. Within each robot observation task, do the executed robot behaviour condition and speed condition relate to condition-level valence, arousal, EEG features, or facial features?

## Secondary research questions

- Do EEG-derived and facial-derived measures show similar patterns in the OASIS experiment?
- Can selected EEG or facial features distinguish between executed robot conditions within a task?
- What participant-level patterns are visible in individual case studies?

## Main analyses

- OASIS analyses use one row per participant-image trial and the participant's observed valence and arousal ratings.
- Robot observation analyses use one selected run per participant and task, classified by the condition that was actually executed.
- The main robot-condition dataset uses executed conditions.
- A sensitivity analysis uses the protocol-compliant subset.

## Labels and their level of detail

### OASIS image experiment

Each image trial has its own image identifier, trigger code, valence rating, and arousal rating. These are **trial-level labels**.

The intended OASIS category (`HAHV`, `HALV`, `LAHV`, or `LALV`) describes the stimulus selection. It must not replace a missing participant rating.

### Robot experiment

Each participant has one valence rating and one arousal rating for each robot task. These are **condition-level ratings**.

Robot event markers describe events such as task start, movement stages, faults, drops, and task end. They can support time-based segmentation, but they do not provide event-level valence or arousal labels.

## Observation-task design

The three observation tasks were always performed in this order:

1. Pick and Place
2. Shape Sorter Observation
3. Stack

For the current 44 participants, 38 have an executed observation-task matrix that matches their assigned group matrix. The participants with confirmed mismatches are P14, P20, P24, P25, P27, and P28.

P14's Stack recording is an unplanned Faulty/Slow condition with one participant. It is retained as a documented special case but is not part of the planned contrasts.

P04 has two compliant Pick-and-Place Control/Slow recordings. A documented run-selection rule is needed before creating a one-row-per-task analysis table.

## Scope limitations

- Robot ratings are condition-level, not event-level.
- Faulty behaviour was normally recorded only at fast speed.
- The Faulty/Slow Stack condition occurs once only, for P14.
- Task identity is fully linked to observation position because task order was fixed.
- Differences between the three observation tasks cannot be separated from possible fatigue, habituation, or time effects.
- Robot condition labels describe robot behaviour and speed; they are not emotion labels.
- EEG has eight configured EEG channels and no dedicated EOG channels.
- Any facial analysis must state clearly which facial measure is being studied. Facial landmarks or geometry features are not direct emotion labels by themselves.

## Optional analyses

- Comparison of EEG and facial measures during OASIS trials.
- Combined EEG and facial analyses after synchronization and quality checks are approved.
- Exploratory participant-level case studies.
- Descriptive reporting of special or repeated recordings.

## Decisions still needed

- Final thesis title.
- Final primary outcomes and statistical models.
- The approved P04 run-selection rule.
- Quality rules for including EEG, video, and ratings in each analysis.
- How future participants will be handled if the protocol, group assignment, or task order changes.
