# Robot Dataset Audit ? corrected metadata view

This is a regenerated audit view under `outputs/`; the prior audit under `derived/` was left unchanged. No raw data or preprocessing output was modified.

## Applied metadata corrections

- P05 Shape Sorter Alone: selected `EGG/UnicornRecorder_29_05_2026_10_24_08.bdf`; valid continuous EEG, 250 Hz, 8,934 samples, 35.736 s, no Status events. The prior missing-EEG flag was caused by literal `EEG`-folder discovery.
- P27 canonical/planned group: C. Its runtime filenames match the full C matrix; all three observation protocol-deviation fields are `none`.
- P28 canonical/planned group: A. Its runtime filenames match the full A matrix; all three observation protocol-deviation fields are `none`.
- The raw Participant Info CSV remains unmodified; P27/P28 YAMLs record the source-entry discrepancy explicitly.

## Remaining genuine planned-versus-actual mismatches

Three task-level mismatch rows remain, based on direct runtime filenames:

| Participant | Task | Planned | Actual | Runtime evidence |
|---|---|---|---|---|
| P14 | Stack | control/slow | faulty/slow | `robot_metrics_10-49-30_STACK_FAULTY_SLOW.csv` |
| P20 | Pick & Place | control/fast | control/slow | `robot_metrics_12-26-58_PnP_CONTROL_SLOW.csv` |
| P20 | Stack | control/slow | control/fast | `robot_metrics_12-32-29_STACK_CONTROL_FAST.csv` |

P14 and P20 cannot be mapped to one standard A/B/C configuration across all three observation tasks. P14 matches B for Pick & Place and Shape Sorter Observation but has nonstandard faulty/slow Stack. P20 has a mixed configuration: C-like Pick & Place, B-like Shape Sorter Observation, and a nonstandard control/fast Stack relative to B.

The companion CSV preserves all 276 participant-task audit rows, with corrected P05/P27/P28 metadata.
