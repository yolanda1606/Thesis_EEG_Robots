# P19 robot alignment-failure investigation

This is a read-only audit of the completed P19 alignment outputs and raw timing
metadata. It does not alter feature definitions, windows, crops, raw files, or
the production alignment threshold.

## Method

For trigger tasks, the current constant-offset model uses the median of
`vision_time - eeg_time`. Residuals below are each pair's offset minus that
median. Anchors were checked by event identity, occurrence order, EEG order,
vision-log order, and available robot-metrics phase information. In both tasks
each code occurs once, so occurrence is `1`; no pairing was made solely by
dictionary lookup of a repeated code.

For the robust sensitivity analysis only, the documented generic candidate rule
is: reject an anchor when `abs(residual - median residual) >
max(50 ms, 3 * 1.4826 * MAD)`. This is not applied in production.

## Pick and Place

Current median offset: -6.088 s. There are 11 matched anchors.

| Code | Occurrence | EEG s | Vision/log s | Residual ms |
|---:|---:|---:|---:|---:|
| 1 | 1 | 6.976 | 0.882 | -6 |
| 11 | 1 | 7.452 | 1.349 | -15 |
| 12 | 1 | 10.584 | 4.613 | **117** |
| 13 | 1 | 16.088 | 9.986 | -14 |
| 14 | 1 | 20.464 | 14.388 | 12 |
| 15 | 1 | 24.844 | 18.756 | 0 |
| 16 | 1 | 29.976 | 23.892 | 4 |
| 20 | 1 | 13.720 | 7.618 | -14 |
| 21 | 1 | 27.976 | 21.891 | 3 |
| 30 | 1 | 33.108 | 27.027 | 7 |
| 99 | 1 | 39.992 | 33.896 | -8 |

Absolute-residual distribution: median 8 ms; 90th percentile 15 ms; 95th
percentile 66 ms. Ten of 11 anchors are within 20 ms, 50 ms, and 100 ms. The
maximum is code 12 at occurrence 1. It is isolated: its offset is -5.971 s,
whereas the remaining offsets cluster around -6.091 s. Robot metrics label
code 12 as `CartesianPose / PICK`; no duplicated or reordered code was found.
The anomaly is therefore a timing discrepancy for this anchor, not evidence of
a systematic clock mismatch.

The generic MAD rule has a 50-ms floor, retains 10/11 anchors, and rejects only
code 12. The retained constant-offset model is -6.091 s, RMSE 9.16 ms, and
maximum residual 15 ms.

## Stack

Current median offset: -4.227 s. There are 40 matched anchors.

| Code | Occurrence | EEG s | Vision/log s | Residual ms |
|---:|---:|---:|---:|---:|
| 1 | 1 | 5.192 | 1.200 | **235** |
| 11 | 1 | 5.628 | 1.381 | -20 |
| 12 | 1 | 7.636 | 3.416 | 7 |
| 13 | 1 | 8.840 | 4.616 | 3 |
| 14 | 1 | 10.896 | 6.650 | -19 |
| 15 | 1 | 11.700 | 7.484 | 11 |
| 16 | 1 | 13.908 | 9.685 | 4 |
| 17 | 1 | 15.712 | 11.486 | 1 |
| 18 | 1 | 16.920 | 12.686 | -7 |
| 19 | 1 | 18.568 | 14.354 | 13 |
| 21 | 1 | 19.576 | 15.354 | 5 |
| 22 | 1 | 21.580 | 17.355 | 2 |
| 23 | 1 | 22.784 | 18.556 | -1 |
| 25 | 1 | 27.088 | 22.858 | -3 |
| 26 | 1 | 29.296 | 25.059 | -10 |
| 27 | 1 | 31.104 | 26.859 | -18 |
| 28 | 1 | 32.312 | 28.093 | 8 |
| 29 | 1 | 33.964 | 29.727 | -10 |
| 31 | 1 | 34.968 | 30.728 | -13 |
| 32 | 1 | 36.976 | 32.728 | -21 |
| 33 | 1 | 38.180 | 33.962 | 9 |
| 34 | 1 | 40.108 | 35.863 | -18 |
| 35 | 1 | 40.912 | 36.697 | 12 |
| 36 | 1 | 43.120 | 38.898 | 5 |
| 37 | 1 | 44.924 | 40.699 | 2 |
| 38 | 1 | 46.132 | 41.899 | -6 |
| 39 | 1 | 47.796 | 43.566 | -3 |
| 41 | 1 | 48.800 | 44.567 | -6 |
| 42 | 1 | 50.808 | 46.568 | -13 |
| 43 | 1 | 52.012 | 47.768 | -17 |
| 44 | 1 | 53.952 | 49.736 | 11 |
| 45 | 1 | 54.756 | 50.536 | 7 |
| 46 | 1 | 58.624 | 54.404 | 7 |
| 47 | 1 | 60.432 | 56.205 | 0 |
| 48 | 1 | 61.636 | 57.406 | -3 |
| 49 | 1 | 62.480 | 58.240 | -13 |
| 80 | 1 | 24.680 | 20.457 | 4 |
| 81 | 1 | 56.964 | 52.737 | 0 |
| 90 | 1 | 63.484 | 59.240 | -17 |
| 99 | 1 | 66.692 | 62.475 | 10 |

Absolute-residual distribution: median 7.5 ms; 90th percentile 18.1 ms; 95th
percentile 20.05 ms. Thirty-eight of 40 anchors are within 20 ms; 39 are
within 50 ms and 100 ms. The maximum is code 1 at occurrence 1. It is the
`Initialization / START` anchor in robot metrics. No duplicate/reordered code
was found; this first startup landmark is the sole isolated anomaly.

The generic MAD rule again has a 50-ms floor, retains 39/40 anchors, and
rejects only code 1. The retained constant-offset model is -4.227 s, RMSE
10.54 ms, and maximum residual 21 ms.

## Shape Sorter Alone

No Status events are present, which is expected for this task. Explicit timing
evidence is:

| Source | Start | End | Duration |
|---|---|---|---:|
| EEG BDF header (`meas_date`, UTC) | 2026-06-11 16:10:36+00:00 | 16:11:04.672+00:00 | 28.672 s |
| Vision log wall clock (naive/local) | 2026-06-11 16:10:18.664 | 16:10:31.367 | 13.188 s |
| Video filename | `vision_video_16-10-18...avi` | — | AVI frame metadata reports 0 frames; vision-log elapsed time is used |
| EEG filename | `UnicornRecorder_..._16_10_07.bdf` | — | filename timestamp is 16:10:07 |

The current implementation interprets naive vision timestamps as Europe/Berlin
(UTC+02:00) and BDF `meas_date` as UTC. It therefore calculates an offset of
+7217.336 s and zero overlap. Even if both header clocks are treated as the
same local clock, the BDF header starts 4.633 s after the logged video ends,
so the header values still produce zero overlap. This is a header/clock-source
conflict, not a missing-trigger failure.

The filenames and vision-log start are internally consistent: the video starts
about 11.664 s after the EEG filename time and ends about 24.367 s after it,
which would lie within the 28.672-s EEG recording. That is promising but is
not sufficient to silently replace BDF header timing.

## Conclusion and generic proposal

Both trigger tasks are defensibly usable *if* a documented robust-anchor stage
is approved: fit the offset/linear candidate on ordered identity/phase matched
anchors, reject isolated MAD-rule outliers, refit, and retain a full audit of
all rejected anchors. This identifies one generic outlier in each task rather
than making participant-specific exclusions. No drift model is warranted after
outlier removal; residuals are already about 9--11 ms under a constant offset.

For trigger-free tasks, use a generic precedence rule: (1) matched triggers;
(2) corroborated absolute BDF/header and vision-log timestamps; (3) when those
conflict, a validated recording-start timestamp parsed from both source
filenames plus monotonic vision-log elapsed time. The third method must record
its evidence and confidence and require an overlap check; it should not be
used automatically until approved.

No re-extraction was performed.

## Approved alignment rerun

After approval, the generic robust-anchor rule and filename fallback were
implemented and P19 alone was re-extracted. All tasks passed the resulting
alignment QC. Pick and Place rejected code 12 occurrence 1; Stack rejected
code 1 occurrence 1. Both exclusions were generated by the same MAD rule.

| Task | Method | Original / retained / rejected anchors | RMSE ms | Max ms | Overlap s | EEG / video / multimodal windows |
|---|---|---:|---:|---:|---:|---:|
| Pick and Place | status constant offset | 11 / 10 / 1 | 9.16 | 15.0 | 33.400 | 42 / 42 / 36 |
| Shape Sorter Observation | status constant offset | 35 / 35 / 0 | 8.59 | 18.0 | 50.867 | 59 / 59 / 53 |
| Stack | status constant offset | 40 / 39 / 1 | 10.54 | 21.0 | 62.500 | 68 / 68 / 65 |
| Sisyphus | status constant offset | 42 / 42 / 0 | 11.24 | 20.0 | 77.516 | 91 / 91 / 81 |
| Shape Sorter Interaction | status constant offset | 26 / 26 / 0 | 10.90 | 19.5 | 70.367 | 77 / 77 / 73 |
| Shape Sorter Alone | filename fallback | 0 / 0 / 0 | N/A | N/A | 13.188 | 27 / 12 / 12 |

Shape Sorter Alone used EEG filename start `2026-06-11T16:10:07` and video
filename start `2026-06-11T16:10:18`, yielding offset -11.0 s and validated
EEG-timeline overlap `[11.0, 24.188)` s. BDF/header timing was retained in
machine-readable QC and explicitly rejected because it conflicts with these
source filenames.

Final freeze validation confirms that filename-fallback residual statistics are
null (not zero). Its complete video windows map as `[0,2) -> [11,13)`,
`[1,3) -> [12,14)`, through `[11,13) -> [22,24)` on the EEG timeline.
