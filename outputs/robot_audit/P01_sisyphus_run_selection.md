# P01 Sisyphus repeated-run selection investigation

Read-only investigation. No raw data, preprocessing code, participant configuration, epochs, or derived data were modified.

## Conclusion

**Run 2 is the recommended valid behavioural/video run, but it cannot currently be selected for multimodal processing because no matching EEG BDF is present.**

Run 1 is the only attempt represented in the available `UnicornRecorder` BDF, but its robot sequence ends during the second cycle after collision-spike events and before the expected second grasp/lift/place/release sequence. Thus Run 1 should be retained as a rejected/aborted attempt, not silently substituted for Run 2.

Accordingly, the selection is **Run 2 for task execution**, with the multimodal record **unresolved/unusable until a Run-2 EEG recording can be located**. Do not combine Run-1 EEG with Run-2 video/log data.

## Artifact sets inspected

### Run 1

- Video: `D:\Thesis\data\P01_2026-05-26\Sisyphus\vision_video_15-38-57_SISYPHUS_INTERACTION.avi`
  - 1,369 frames at 30 fps; 45.633 s; 640 × 480; first frame readable.
- Vision log: `D:\Thesis\data\P01_2026-05-26\Sisyphus\vision_log_15-38-57_SISYPHUS_INTERACTION.csv`
  - 1,369 rows; `Timestamp` 2026-05-26 15:38:58.180 through 15:39:44.198; `Experiment_Time` 0.486–46.504 s.
  - Non-zero trigger sequence: `1, 11, 12, 50, 13, 50, 20, 14, 15, 50, 50, 16, 50, 21, 17, 11, 12, 13, 50, 50, 50, 50`.
  - All 1,369 emotion values are `no face`; this is a vision-detection outcome, not evidence of task interruption.
- Robot metrics: `D:\Thesis\data\P01_2026-05-26\Sisyphus\robot_metrics_15-39-44_SISYPHUS_INTERACTION.csv`
  - 39,014 rows; `Experiment_Time` 0.000–39.131 s.
  - Explicit `START` (trigger 1) and `END` (99) rows are present.
  - First cycle reaches `GRASP` (20), `LIFTING` (14), `PLACE` (16), `RELEASE` (21), and `CLEARING` (17).
  - A second cycle starts (`WAIT` 11, `PRE_PICK` 12, `PICK` 13) but has four subsequent `HRI_COLLISION_SPIKE` events (50) and then `END` (99), with no second `GRASP`, `LIFTING`, `PLACE`, `RELEASE`, or `CLEARING`.
  - This is direct evidence of an incomplete second cycle/early termination, despite an `END` marker.

### Run 2

- Video: `D:\Thesis\data\P01_2026-05-26\Sisyphus\vision_video_15-40-43_SISYPHUS_INTERACTION.avi`
  - 1,510 frames at 30 fps; 50.333 s; 640 × 480; first frame readable.
- Vision log: `D:\Thesis\data\P01_2026-05-26\Sisyphus\vision_log_15-40-43_SISYPHUS_INTERACTION.csv`
  - 1,510 rows; `Timestamp` 2026-05-26 15:40:43.967 through 15:41:34.686; `Experiment_Time` 0.485–51.204 s.
  - Non-zero trigger sequence contains two completed movement cycles: the second includes `GRASP` (20), `LIFTING` (14), `PRE_PLACE` (15), `PLACE` (16), `RELEASE` (21), and `CLEARING` (17), followed by `WAIT` (11).
  - All 1,510 emotion values are `no face`.
- Robot metrics: `D:\Thesis\data\P01_2026-05-26\Sisyphus\robot_metrics_15-41-35_SISYPHUS_INTERACTION.csv`
  - 39,611 rows; `Experiment_Time` 0.000–39.742 s.
  - Explicit `START` (1) and `END` (99) rows are present.
  - Both cycles reach the expected grasp/lift/place/release/clearing stages. The second cycle reaches `RELEASE`/`CLEARING` at 36.058 s and then `END` at 39.742 s.
  - The vision log and metrics have the same ordered task-event progression. Their small duration difference reflects capture/log boundaries, not an observed abort.

## EEG evidence and run association

The only task BDF used by the pipeline convention is:

- `D:\Thesis\data\P01_2026-05-26\Sisyphus\EEG\UnicornRecorder_26_05_2026_15_38_51.bdf`
  - 250 Hz; 22,253 samples; 89.012 s; 18 channels including `Status`.
  - One continuous BDF file, not two EEG fragments.
  - Status events span BDF time 7.072–52.840 s.

Its Status event sequence agrees with **Run 1**, in order and timing. The Run-1 vision-log triggers from 1 through the final recorded 50 align to BDF triggers with an approximately constant BDF-minus-log offset of about 6.11 s (for example: trigger 1, 7.072 vs 0.945 s; trigger 13, 16.156 vs 10.054 s; trigger 17, 34.648 vs 28.528 s). The BDF then records `END` (99) at 52.840 s, shortly after the log stops.

The BDF does **not** contain Run 2's distinctive second-cycle sequence (`20, 14, 15, 16, 21, 17`) after its second `PICK`; it instead contains the Run-1 collision-spike sequence. Its 89.012-s acquisition duration also cannot span the approximately 106-s separation between the two vision-log starts if those wall-clock timestamps are used. Therefore the available BDF represents Run 1 only; it is not a continuous EEG record containing both attempts.

The companion `UnicornRawDataRecorder_26_05_2026_15_38_51.bdf` and CSV are acquisition companions, not evidence of a separate Run-2 session under the established pipeline convention.

## Exact files to retain/use

For the selected behavioural/video attempt (Run 2):

- `vision_video_15-40-43_SISYPHUS_INTERACTION.avi`
- `vision_log_15-40-43_SISYPHUS_INTERACTION.csv`
- `robot_metrics_15-41-35_SISYPHUS_INTERACTION.csv`

Do **not** pair these with `UnicornRecorder_26_05_2026_15_38_51.bdf`. There is no currently identified EEG file to use for Run 2.

For traceability of the rejected attempt (Run 1), retain its video/log/metrics and link it to the BDF above; do not delete or overwrite any raw artifact.

## EEG handling recommendation

- Do not epoch or preprocess this BDF for the selected Run 2.
- If a Run-2 BDF is later located, process it as a separate acquisition segment/run with its own event-to-vision alignment.
- If no Run-2 EEG exists, exclude P01 Sisyphus from EEG–video multimodal analyses. Run 1 could only be analysed separately as an explicitly rejected/incomplete attempt, never as Run-2 EEG.

## Proposed preservation metadata

Store these fields in a later, authorized run-selection manifest/configuration layer (not in raw data):

- `participant_id: P01`
- `task: Sisyphus`
- `run_id` and `run_role` (`run_1_rejected`, `run_2_selected_behavioral`)
- `selection_status` (`selected_behavioral_only` for Run 2; `rejected_incomplete` for Run 1)
- `rejection_reason` (Run 1: second cycle lacks grasp/lift/place/release/clearing after collision spikes)
- exact `video_file`, `vision_log_file`, `robot_metrics_file`, and `eeg_file` paths
- `eeg_run_association` (`Run 1` / `missing_for_Run_2`)
- `event_sequence_summary` and `event_time_reference`
- `sync_offset_s` / `alignment_method` for Run 1 only
- `modality_eligibility` (`multimodal_no` for selected Run 2 until EEG is located)
- `raw_preservation_note` stating that the rejected attempt remains retained and unmodified.
