# Thesis EEG–Robot Interaction Workspace

This repository supports a study combining portable EEG, participant video,
emotional-image ratings, observation of robot tasks, and human–robot interaction.
It contains live acquisition code and exploratory/validated analysis code. Treat
participant data as sensitive research data.

## Safety-critical boundaries

- The primary acquisition launcher is `bash_files/launch_experiment.sh`.
- Do not edit acquisition scripts during active data collection without a staged
  bench test and explicit approval.
- Do not run experiment or robot executables as a diagnostic check.
- Never modify files under `data/`; analysis should treat them as immutable.
- Never commit raw EEG, ratings, participant video, participant identifiers, or
  generated participant-level outputs.

## Study design

Each participant completes three sections.

### 1. Emotional baseline

OASIS images are presented while EEG and participant video are recorded. After
each image, the participant reports valence and arousal on seven-point scales.

The image task uses four intended stimulus categories:

- HAHV: high arousal, high valence
- HALV: high arousal, low valence
- LAHV: low arousal, high valence
- LALV: low arousal, low valence

These intended categories are stimulus metadata. They must not replace a missing
participant rating.

### 2. Robot observation

- Pick and Place
- Stack
- Shape Sorter Observation

### 3. Interaction and individual tasks

- Sisyphus Interaction
- Shape Sorter Interaction
- Shape Sorter Alone

The observation tasks vary fault state and speed according to the study's group
assignment. Preserve `FAULTY`/`CONTROL` and `FAST`/`SLOW` metadata through every
processing stage.

## Repository map

```text
bash_files/                 Experiment launcher; acquisition-critical
camera/                     RealSense video and frame-level emotion logging
eeg_emotion_experiment/     PsychoPy OASIS presentation and ratings
robot_experiments/          Franka C++ robot tasks and robot metrics
data/                       Sensitive acquisition data; read-only for analysis
processing/emotional_baseline/
                            Reproducible image-experiment analysis pipeline
Playground/Image_Experiment/
                            Legacy exploratory scripts retained for reference
eeg_venv/                   Required environment for EEG analysis commands
```

## Data collection flow

The launcher asks for a participant ID and task. It starts
`camera/vision_node.py`, which records RealSense RGB video and a frame-level CSV.
For robot tasks, the launcher then starts the selected compiled C++ executable.
For the image task, it activates `eeg_emotion_experiment/venv` and starts
`emotion_eeg_exp.py`.

Robot and image-task events are sent as UDP datagrams to:

- the EEG recording computer at `10.0.0.2:1000`, and
- the local video logger at `127.0.0.1:5005`.

The EEG recorder produces Unicorn BDF/CSV exports. The PsychoPy task produces
ratings CSV and PSYDAT exports. The vision node produces AVI video and a CSV with
frame timestamps, elapsed time, triggers, and inferred facial emotion. Robot
programs write timestamped metrics CSV files.

If the EEG recorder stops during the image task, restart it and preserve both BDF
segments. They represent consecutive segments of the same session; analysis must
match events in each segment rather than assuming one BDF per participant.

## Running an experiment

Only trained operators with the hardware prepared should run acquisition.
Starting the launcher may activate the camera and move the Franka robot.

From the repository root:

```bash
cd /home/sysgen/Projects/Yolanda/Thesis_EEG_Robots
bash bash_files/launch_experiment.sh
```

The operator then:

1. Enters the participant ID using the established pseudonymous format.
2. Selects one of the seven tasks.
3. For observation tasks, enters the protocol-approved fault and speed settings.
4. Confirms that EEG trigger reception and video recording are active.
5. Monitors the task and uses Ctrl+C only when an emergency or the documented
   human-only-task stop procedure requires it.
6. Verifies that EEG, video, ratings, and robot outputs were saved in the same
   participant/date directory before continuing.

Do not use this command to test the repository. It is the live experiment entry
point. Hardware/network setup details currently remain in `Set_up.txt` and should
be consolidated into a controlled operator protocol after data collection.

## Analysis environment

All Python analysis commands must use `eeg_venv`:

```bash
cd /home/sysgen/Projects/Yolanda/Thesis_EEG_Robots
source eeg_venv/bin/activate
which python
python --version
python -c "import sys; print(sys.executable)"
```

Expected executable:

```text
/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/eeg_venv/bin/python
```

Do not install or change dependencies during data collection without recording
and reviewing the change.

## Emotional-baseline analysis

The maintained image-only workflow is documented in
`processing/emotional_baseline/README.md`. Its order is:

1. Build a participant/session/segment inventory.
2. Review ratings and EEG-event completeness.
3. Match behavioral trials to EEG triggers.
4. Preprocess approved BDF segments.
5. Extract epoch-level features.
6. Join features to observed valence/arousal ratings.
7. Train and evaluate with participant-grouped validation.

Generated analysis artifacts go under
`processing/emotional_baseline/outputs/`, never beside raw recordings.

## Current data-quality rules

- Select the most complete valid PsychoPy CSV when duplicate exports exist.
- Preserve duplicate source files; record which export is authoritative.
- Keep interrupted EEG BDF segments separate until event matching is complete.
- Exclude a trial only from the target whose rating is missing.
- Do not impute missing ratings from HAHV/HALV/LAHV/LALV.
- Split training and evaluation by participant, never by random epochs.
- Fit scaling, feature selection, and tuning inside training folds only.

## Version control and privacy

The root `.gitignore` excludes `data/`, virtual environments, videos, and build
artifacts. This does not remove sensitive files that may already exist in Git
history. Review tracked participant-derived exemplars before publishing or
sharing the repository.

