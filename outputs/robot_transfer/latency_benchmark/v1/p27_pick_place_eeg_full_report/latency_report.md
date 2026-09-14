# P27 Pick & Place EEG-only latency report

## Scope and compatibility

This report summarizes the completed offline replay benchmark for participant P27, task Pick & Place, EEG-only mode. Replay compatibility **passed**: tolerance `1e-09`, maximum absolute numerical difference `1.818989403545856e-12`. Latency conclusions below are therefore based on a compatibility-validated replay.

The benchmark contains 60 valid frozen-model inference calls across 10 chronological 2-s windows (1-s step): six participant-specific frozen models per window, comprising valence ranks 1–3 and arousal ranks 1–3.

Timed per-call computation includes EEG window extraction, EEG feature extraction, feature-vector assembly, the original frozen-pipeline hard prediction, fitted transforms where separable, and frozen-model P(HIGH) inference. It excludes BDF loading, model loading/hashing, configuration loading, alignment fitting, video opening, MediaPipe initialization, output writing, acquisition, Bluetooth, RealSense driver latency, and live synchronization.

## Main latency results

| Metric | Result |
|---|---:|
| Mean total compute time | 10.28 ms |
| SD | 1.79 ms |
| Median | 10.71 ms |
| p95 | 13.13 ms |
| Minimum / maximum | 6.81 / 13.47 ms |
| Calls below 1000 ms | 100.0% (60/60) |
| Median / 1000 ms | 0.0107 |
| p95 / 1000 ms | 0.0131 |

The median computation consumes 1.07% of the 1-s update interval; p95 consumes 1.31%. All measured calls were below the 1000-ms interval. On this offline computational scope, the frozen participant-specific predictors are **computationally compatible with a 1 s prediction update interval**.

## Stage profile and outliers

The largest median component is **EEG features** (5.05 ms). EEG feature extraction is the principal recurring cost; model-specific assembly, transform, hard prediction, and P(HIGH) inference are smaller. The calibrated SVM has no separately observed transform/scaling stage because its fitted wrapper encapsulates that computation; its complete probability call is timed as one operation.

The maximum observed total was 13.47 ms at `pick_place_s1_2.000` for arousal rank 3 (svm); this remains far below 1000 ms. The sequence plot shows bounded variation over the ten replay windows, without a spike that approaches the update interval. With only ten windows per model, this is a descriptive stability check rather than a broad tail-latency guarantee.

## Continuous preprocessing

Continuous CAR plus 1–40 Hz filtering required 7.94 ms for the 42.036-s recording. Across 10 valid windows, its arithmetic amortization is 0.79 ms/window. This is an **amortized continuous preprocessing cost**, not measured true online per-window latency. The canonical preprocessing uses zero-phase continuous filtering and is not an online causal implementation.

## Cautious conclusion

For this completed P27 Pick & Place offline replay, frozen EEG-only P(HIGH) computation is computationally feasible within the 1-s update budget. This does **not** establish complete real-time system latency: acquisition, wireless communication, camera/driver delays, synchronization, and causal live-preprocessing behavior were outside the benchmark.
