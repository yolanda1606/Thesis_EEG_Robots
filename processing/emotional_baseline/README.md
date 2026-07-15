# Emotional Baseline Processing

This workspace contains the reproducible analysis pipeline for the OASIS image
experiment. It is deliberately isolated from experiment acquisition code.

## Safety boundary

- Inputs under `../../data/` are treated as read-only.
- Nothing here launches PsychoPy, the camera, EEG recorder, or robot.
- Generated files belong under `outputs/`, which is ignored by Git.
- The legacy scripts in `../../Playground/Image_Experiment/` remain unchanged.

## Layout

```text
configs/                  Channel, event, and preprocessing definitions
src/emotional_baseline/   Reusable Python implementation
scripts/                  Command-line entry points
tests/                    Fast tests using synthetic metadata
outputs/                  Generated manifests, QC, features, and models
```

## Environment

Run every Python command from the project root in `eeg_venv`:

```bash
cd /home/sysgen/Projects/Yolanda/Thesis_EEG_Robots
source eeg_venv/bin/activate
which python
python --version
python -c "import sys; print(sys.executable)"
```

The executable must be inside `Thesis_EEG_Robots/eeg_venv/`.

For commands below, make the package importable without installing it:

```bash
export PYTHONPATH="$PWD/processing/emotional_baseline/src"
```

## Recommended workflow

### 1. Build the inventory

```bash
python processing/emotional_baseline/scripts/build_inventory.py
```

This scans filenames, BDF headers, events, and rating completeness. It writes
only aggregate and pseudonymous metadata to
`processing/emotional_baseline/outputs/inventory/`.

### 2. Review QC

Do not preprocess a block until its inventory status and EEG/rating pairing have
been reviewed. Interrupted recordings remain separate segments of one session.
The pipeline combines epochs after event matching rather than blindly joining
files by sorted filename position.

### 3. Preprocess

```bash
python processing/emotional_baseline/scripts/run_preprocessing.py --participant P01_YYYY-MM-DD
```

This command reads the inventory, loads approved BDF segments, assigns channel
types, renames EEG channels, filters, rereferences, and epochs image events. It
writes FIF epochs and QC JSON under `outputs/epochs/`.

The initial configuration is intentionally conservative. Notch filtering, bad
channel decisions, ICA, and AutoReject must be justified and validated before
being enabled for the definitive analysis.

### 4. Extract features

```bash
python processing/emotional_baseline/scripts/extract_features.py
```

Feature rows retain participant, session, segment, trigger, and target metadata.
Missing participant ratings remain missing; stimulus quadrant is never used to
impute a behavioral response.

### 5. Train and evaluate

```bash
python processing/emotional_baseline/scripts/attach_ratings.py
python processing/emotional_baseline/scripts/trainer.py --target valence
python processing/emotional_baseline/scripts/eval.py --target valence
```

Training and evaluation are grouped by participant. Scaling, feature selection,
and hyperparameter selection must be fitted using training participants only.

## Current dataset cautions

- Most ratings CSVs are duplicate PsychoPy exports.
- For P08, the primary CSV has complete ratings and the `_1` CSV is a logical
  duplicate with empty rating columns.
- Some EEG sessions contain multiple BDF segments because recording restarted.
- Partial EEG segments can still contribute valid, event-matched trials.
- Missing ratings must be excluded target-by-target, never replaced with the
  intended OASIS category.
