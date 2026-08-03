# Reproducibility guide

## Repository and archive layout

Clone this repository and place raw acquisition data outside Git. On the
current machine, the external archive uses sibling directories named `data`
and `derived`. On another computer, choose equivalent locations and point the
pipeline to them through participant YAML files and command-line arguments;
do not edit source code to add a machine-specific path.

The repository deliberately excludes raw `data/`, complete `derived/` runs,
virtual environments, caches, build products, licensed stimuli, and the
downloaded landmark-model file. Derived manifests and logs can contain
participant labels and absolute paths, so the complete derived archive is not
stored in Git.

## Environment

This work was run with Python 3.10.12. Create an environment compatible with
that version, then install the recorded environment packages:

```bash
python -m venv eeg_venv
source eeg_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

For the existing workspace, activate the supplied environment instead:

```bash
cd /path/to/Thesis_EEG_Robots
source eeg_venv/bin/activate
```

`requirements-lock.txt` records the full environment used here, including ROS
packages. It is an environment snapshot rather than a minimal dependency set.

## Inputs, resources, and configuration

Keep raw recordings in the location referenced by the participant YAML files.
Use the layered configuration files under
`processing/multimodal_image/configs/`:

1. `pipeline/eeg_video_defaults.yaml` for shared settings.
2. `experiments/image_experiment.yaml` for experiment settings.
3. `participants/<ID>.yaml` for a participant-specific recording layout.

The face-landmark model must be named `face_landmarker.task` and placed in a
directory passed as `--resource-root`, normally
`processing/multimodal_image/models`. The exact file used in this workspace
has SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The repository does not record an authoritative model URL or licence, so do
not redistribute it until that provenance is confirmed. Obtain the same model
from an official source approved by the project owner, then verify its
checksum before use.

Licensed stimuli are expected under
`eeg_emotion_experiment/stimuli/`, arranged by the four condition directories
`HAHV`, `HALV`, `LAHV`, and `LALV`, using the filenames selected for the
experiment. They are excluded from Git. Their redistribution status and an
authoritative acquisition source are not recorded in this repository; obtain
them through the applicable licence or project owner. Create a checksum
manifest only from a lawfully obtained local copy if distribution control
permits it.

## Running the image pipeline

From the repository root, first inspect a configuration without writing data:

```bash
python -u processing/multimodal_image/run_image_pipeline.py \
  --config processing/multimodal_image/configs/experiments/image_experiment.yaml \
  --participant-config processing/multimodal_image/configs/participants/P01.yaml \
  --resource-root processing/multimodal_image/models \
  --output-root /path/to/derived \
  --run-name p01_dry_run_example --dry-run --verbose
```

Use a new descriptive `--run-name` for every run; the pipeline refuses to
overwrite an existing run directory. For a focused input/synchronization
check, replace `--dry-run` with `--health-check-only --verbose`. For full
processing, use the full multimodal command in
`docs/run_image_pipeline_commands.md`; it documents EEG preprocessing, video
processing, feature extraction, merging, validation, and crop-preview modes.

Before every processing run, record the revision and environment:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

## External archive verification

The current-machine archive example is
`/media/sysgen/ADATA SD600Q/Thesis/{data,derived}`. To verify a copied derived
archive without following symlinks:

```bash
find derived -type f -printf '.' | wc -c
find '/media/sysgen/ADATA SD600Q/Thesis/derived' -type f -printf '.' | wc -c
du -sh -- derived
du -sh -- '/media/sysgen/ADATA SD600Q/Thesis/derived'
rsync -a --dry-run --checksum --safe-links --itemize-changes \
  -- 'derived/' '/media/sysgen/ADATA SD600Q/Thesis/derived/'
```

On filesystems that do not support Unix permissions, rsync may report
permission-only differences even when file content checksums match.
