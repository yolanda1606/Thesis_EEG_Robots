# P05 EEG and P27/P28 group-source investigation

Read-only targeted investigation. No preprocessing was run; no raw data, pipeline code, participant YAML, audit output, or `derived/` file was modified.

## Part 1 — P05 Shape Sorter Alone EEG

### Direct BDF result

- Exact file: `D:\Thesis\data\P05_2026-05-29\Shape Sorter Alone\EGG\UnicornRecorder_29_05_2026_10_24_08.bdf`
- Opens successfully with MNE BDF reading; no corruption/read error observed.
- Sampling rate: 250 Hz.
- Samples: 8,934.
- Duration: 35.736 s.
- Channels: 18 — `EEG 1`–`EEG 8`, `ACC X/Y/Z`, `GYR X/Y/Z`, `CNT`, `VALID`, `DT`, and `Status`.
- BDF header measurement time: 2026-05-29 10:24:44 UTC.
- Decoded Status events: none.
- Unique Status codes: none.
- Status trigger timestamps: none.

### Consistency with the task

The filename start label (`10_24_08`) and 35.736-s duration encompass the Shape Sorter Alone vision-log interval (2026-05-29 10:24:22.762–10:24:41.401). This is consistent with the BDF belonging to P05 Shape Sorter Alone. As seen in other Unicorn files, the BDF header measurement time is not a reliable wall-clock proxy for the filename/video timeline here: it is later than the vision-log end.

There is no technical reason to reject the BDF as continuous EEG: it opens normally and has the expected Unicorn channel layout. Its limitation is that the `Status` channel contains no detectable events, so it cannot support BDF-trigger-based task alignment. In addition, the associated raw video reports zero frames, while the vision log has 541 rows and no triggers; this constrains multimodal video use but does not invalidate the BDF as an EEG file.

**P05 Shape Sorter Alone EEG usable: yes, for continuous EEG; no Status-trigger alignment is available.**

### Why the audit missed it

The audit searched the task directory’s literal `EEG` subdirectory for `UnicornRecorder_*.bdf`. P05 stores this acquisition under the misspelled `EGG` subdirectory. Therefore the prior `EEG UnicornRecorder BDF` missing flag was a directory-discovery false negative, not a missing BDF.

### Recommended later correction

- Select the BDF above in `processing/multimodal_robot/configs/participants/P05.yaml` under `tasks.shape_sorter_alone.eeg`.
- Set an explicit reason such as `valid UnicornRecorder BDF found under raw EGG directory; Status events absent` and set `status_triggers_available: false`.
- Reassess task-level multimodal eligibility separately because the selected video has zero frames; do not treat BDF presence alone as proof of EEG–video alignment.
- When the robot audit is next regenerated, make directory discovery robust to both `EEG` and legacy/misspelled `EGG` directories (or discover Unicorn BDFs recursively under a task directory while retaining the discovered folder name). Do not silently move or rename raw folders.

## Part 2 — provenance of P27/P28 planned-group values

### Originating source

The planned-group values are direct values from the raw participant assignment inventory, not inferred from runtime filenames:

|Participant|Source file and physical row|Source field|Value currently recorded|
|---|---|---|---|
|P27|`D:\Thesis\data\Thesis Organization - Participant Info.csv`, line 29 (`# = 27`)|`Group`|`A`|
|P28|`D:\Thesis\data\Thesis Organization - Participant Info.csv`, line 30 (`# = 28`)|`Group`|`C`|

The audit loaded that CSV using its second physical row as headers and mapped `#` to `Pxx` and `Group` to the audit `planned_group`. It then applied the fixed planned A/B/C observation matrix. Thus P27’s planned task matrix was A (`faulty/fast`, `control/slow`, `control/fast`) and P28’s was C (`control/slow`, `control/fast`, `faulty/fast`).

The raw source is the direct source of the present planned assignments. The runtime filenames show the opposite complete three-task configurations (P27 matches C; P28 matches A), but that alone establishes executed/analytic grouping, not whether the contemporaneous assignment-inventory row was originally entered incorrectly or the participant was run under a different configuration. The available files cannot distinguish those two historical explanations.

### Downstream propagation found

|Location|Current propagation|
|---|---|
|`derived/robot_audit/robot_dataset_audit.csv`|`planned_group` is A for P27 and C for P28 on each observation-task row; planned fields and deviations were calculated from those values.|
|`derived/robot_audit/robot_dataset_audit.md`, lines 37–42|Lists the resulting P27/P28 planned-versus-actual mismatch rows.|
|`processing/multimodal_robot/configs/participants/P27.yaml`|All three observation `planned.group` fields are A, with A-matrix planned condition/speed and mismatch deviations.|
|`processing/multimodal_robot/configs/participants/P28.yaml`|All three observation `planned.group` fields are C, with C-matrix planned condition/speed and mismatch deviations.|
|`docs/analysis_decisions.md` line 26; `docs/data_health_checks.md` line 111; `docs/thesis_scope.md` line 56|Name P27/P28 as planned/executed mismatch participants, but do not themselves declare the A/C row values.|

No versioned robot-audit generation script or separate hard-coded P27/P28 group mapping exists in the repository. The audit and YAMLs were generated static outputs from the raw inventory plus the planned matrix. Consequently, changing the raw source assignment would **not** automatically refresh existing CSV/Markdown/YAML outputs; they require a deliberate later regeneration/update. Image participant YAMLs contain no robot group field and are unaffected.

## Recommended correction plan (do not apply in this investigation)

### P05

1. Preserve raw `EGG` folder unchanged.
2. Update P05 Shape Sorter Alone YAML to select `...\EGG\UnicornRecorder_29_05_2026_10_24_08.bdf`, with absent Status events explicitly recorded.
3. Refresh the robot audit row after improving future directory discovery to include `EGG`; keep the separate zero-frame-video issue visible.

### P27/P28

1. First make an explicit governance decision: either correct the raw inventory `Group` field (`#27: A → C`, `#28: C → A`) as an erroneous assignment record, or retain it as historical planned assignment and add a separate analysis/executed-group field. The present direct evidence supports the latter distinction but cannot prove the original inventory was erroneous.
2. If the decided source correction is `P27 = C` and `P28 = A`, correct the `Group` field in `Thesis Organization - Participant Info.csv` under an explicitly authorized raw-data correction process; do not edit it casually.
3. Then deliberately regenerate/update `robot_dataset_audit.csv`, `robot_dataset_audit.md`, and the observation `planned`/`actual`/`protocol_deviation` fields in P27/P28 robot YAMLs. Runtime-evidenced `actual` condition/speed should remain unchanged; only the planned baseline and deviation labels would change.
4. Update the three documentation pages above so they no longer label P27/P28 as planned/executed mismatches if the source assignment correction is accepted.
