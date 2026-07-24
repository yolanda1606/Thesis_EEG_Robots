# Analysis decisions

## Status of this document

This document records current agreed rules and open decisions. It does not authorize preprocessing, machine learning, or changes to raw data.

## Raw-data protection

- Files under `data/` are treated as immutable source material.
- Do not modify, rename, move, or delete raw EEG, videos, ratings, vision logs, robot metrics, or participant records.
- Store derived data, reports, figures, models, and documentation outside raw participant directories.
- Keep every source recording, including interrupted and repeated runs.

## Executed versus planned condition policy

The main robot observation analyses use the condition that was actually executed.

- `planned_condition` records the assigned group matrix.
- `executed_condition` is determined from runtime profile evidence, task-specific triggers, and robot metrics.
- `protocol_compliance` records whether the executed observation-task matrix matches the assigned matrix.

Planned and executed conditions must both be retained. They answer different questions.

## Mismatch-participant policy

The confirmed mismatch participants are P14, P20, P24, P25, P27, and P28.

- They remain eligible for the full executed-condition dataset when their selected recordings pass quality checks.
- They are excluded from the protocol-compliant sensitivity dataset.
- A mismatch is not, by itself, a technical data-quality failure.

The current protocol-compliant sensitivity dataset contains 38 participants, subject to the approved P04 run-selection rule and later quality decisions.

## P14 Faulty/Slow Stack policy

P14 has the only observed Faulty/Slow Stack condition.

- Retain this recording in the raw inventory and full executed-condition dataset if it passes quality checks.
- Report it separately as an unplanned special case.
- Exclude it from planned Faulty/Fast versus Control/Fast contrasts.
- Exclude it from planned Control/Fast versus Control/Slow contrasts.
- Do not estimate a Faulty/Slow effect from one participant.

## P04 run-selection policy

P04 has two Pick-and-Place Control/Slow recordings. Both are condition-compliant.

Proposed selection rule:

1. Prefer the run whose video/log start is closest to the paired EEG BDF start.
2. If more than one candidate remains, prefer the earliest complete run with expected start/end markers and task-event sequence.
3. Record the selected run and the reason in the recording manifest.
4. Retain the other run in the manifest as unselected.

This rule requires researcher approval before a selected analysis row is created.

## Label-resolution policy

### OASIS experiment

- Use observed valence and arousal ratings at the image-trial level.
- Keep trigger, image identity, stimulus category, and response-time information.
- Do not replace missing ratings with intended stimulus categories.

### Robot experiment

- Use one valence rating and one arousal rating per participant and condition.
- Treat these as condition-level ratings.
- Use robot events for task segmentation or exploratory time windows only.
- Do not treat robot event markers as event-level affect labels.

## Current analysis priorities

1. Maintain a recording manifest that links raw files, repeated runs, and selected runs.
2. Complete data-health and synchronization review before preprocessing.
3. Define EEG, video, and rating quality criteria.
4. Build derived analysis tables outside `data/`.
5. Use participant-aware validation for any future predictive modelling.

## Fixed-order limitation

Observation tasks were always completed in this order:

1. Pick and Place
2. Shape Sorter Observation
3. Stack

Task identity is therefore fully confounded with observation position. Cross-task differences cannot be separated from fatigue, habituation, or general time effects.

## Decisions still requiring approval

- Final primary outcomes and statistical models.
- P04 run selection.
- Technical quality thresholds for EEG, video, ratings, and synchronization.
- Whether non-observation robot tasks are primary, secondary, or descriptive only.
- Whether mismatch participants are included in the main executed-condition analysis by default after QC.
- How future participants will be documented if the protocol, task order, or group matrix changes.
- The facial-analysis method and the meaning of its outcome measures.
