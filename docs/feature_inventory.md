# P01 Image Experiment feature inventory

## Scope and data representation

This inventory describes the active completed run:

`derived/P01/Image_Experiment/runs/p01_image_initial_v2/`

EEG features are extracted per retained trial and per EEG channel in
`eeg/features/eeg_epoch_features.csv`. The merged trial table stores the same
EEG values in wide form, with names such as `eeg_sd__C3`.

Channels retained for P01 are: `C3`, `C4`, `Cz`, `Fz`, `Oz`, `PO7`, `PO8`, and
`Pz`. There are 119 retained EEG trials; trigger 326 has no EEG features after
AutoReject.

Facial measures are computed frame-by-frame within the two-second image window,
then summarized per trial in `video/features/video_trial_features.csv`.

## EEG feature families

For every family below, the source EEG table uses the base column name shown.
The merged table contains one exact column for each retained channel, using the
suffixes `__C3`, `__C4`, `__Cz`, `__Fz`, `__Oz`, `__PO7`, `__PO8`, and `__Pz`.
For example, `eeg_sd` becomes `eeg_sd__C3` through `eeg_sd__Pz`.

| Source column / merged-column family | Meaning | Main characteristic | Calculation and interpretation |
|---|---|---|---|
| `eeg_sd` / `eeg_sd__{channel}` | Standard deviation of the EEG signal within the epoch. | Signal amplitude / variability | Larger values indicate greater variation in the recorded EEG voltage over the epoch. It is affected by neural activity, remaining noise, and signal scale; it is not frequency-specific. |
| `eeg_se` / `eeg_se__{channel}` | Shannon entropy of the full Welch power spectral density (PSD). | Frequency-content complexity | The PSD values are converted to relative non-negative values and entropy is calculated as `-sum(p * log2(p))`. Larger values mean power is spread more evenly across frequencies; lower values mean power is concentrated in fewer frequencies. |
| `eeg_hm` / `eeg_hm__{channel}` | Hjorth mobility. | Temporal complexity / frequency-related activity | Calculated as `sqrt(var(diff(signal)) / var(signal))`. It summarizes how rapidly the signal changes from sample to sample. |
| `eeg_hc` / `eeg_hc__{channel}` | Hjorth complexity. | Temporal complexity | Calculated as the mobility of the first difference divided by the mobility of the original signal. Larger values indicate a more complex waveform relative to a simple oscillation. |
| `eeg_mf_hz` / `eeg_mf_hz__{channel}` | Median frequency in hertz. | Frequency content | The frequency at which cumulative trapezoid-integrated PSD reaches half of total spectral power. Higher values indicate relatively more high-frequency power. |
| `eeg_bp_delta` / `eeg_bp_delta__{channel}` | Delta-band power. | Band activity | Welch PSD integrated from 0.5 to 4 Hz. The numerical unit follows the EEG signal unit squared per hertz integrated over frequency. |
| `eeg_se_delta` / `eeg_se_delta__{channel}` | Delta-band spectral entropy. | Band complexity | Shannon entropy of PSD values in the 0.5–4 Hz band. It describes how concentrated or spread the delta-band spectrum is. |
| `eeg_bp_theta` / `eeg_bp_theta__{channel}` | Theta-band power. | Band activity | Welch PSD integrated from 4 to 8 Hz. |
| `eeg_se_theta` / `eeg_se_theta__{channel}` | Theta-band spectral entropy. | Band complexity | Shannon entropy of PSD values in the 4–8 Hz band. |
| `eeg_bp_alpha` / `eeg_bp_alpha__{channel}` | Alpha-band power. | Band activity | Welch PSD integrated from 8 to 12 Hz. |
| `eeg_se_alpha` / `eeg_se_alpha__{channel}` | Alpha-band spectral entropy. | Band complexity | Shannon entropy of PSD values in the 8–12 Hz band. |
| `eeg_bp_beta` / `eeg_bp_beta__{channel}` | Beta-band power. | Band activity | Welch PSD integrated from 12 to 30 Hz. |
| `eeg_se_beta` / `eeg_se_beta__{channel}` | Beta-band spectral entropy. | Band complexity | Shannon entropy of PSD values in the 12–30 Hz band. |
| `eeg_bp_gamma` / `eeg_bp_gamma__{channel}` | Gamma-band power. | Band activity | Welch PSD integrated from 30 to 40 Hz. This is the pipeline’s defined gamma range, not a broader conventional gamma range. |
| `eeg_se_gamma` / `eeg_se_gamma__{channel}` | Gamma-band spectral entropy. | Band complexity | Shannon entropy of PSD values in the 30–40 Hz band. |

All 15 EEG families are valid scientific candidate predictors. Their
interpretation must remain cautious: EEG band power and entropy are not direct
measurements of emotion, and amplitude-related values can be influenced by
residual artifacts and recording conditions.

`epoch_index` is not an EEG feature. It is an export-order field and must not
be modeled.

## Facial measures: exact implementation

The feature code uses only the x/y coordinates of the detected MediaPipe face
landmarks. It converts normalized coordinates into pixel coordinates in the
approved cropped video frame:

`xy = landmarks[:, :2] * [frame_width, frame_height]`

All five measures are normalized by the Euclidean distance between:

- left eye outer corner: landmark 33
- right eye outer corner: landmark 263

This inter-outer-eye distance is the denominator for every facial measure.
Therefore, the resulting values are dimensionless ratios and are intended to
reduce the effect of face size in the image.

| Code name | Exact construction in the implementation | Interpretation |
|---|---|---|
| `IRISDO` (`irisdo_norm`) | Mean of the Euclidean left eyelid opening, distance(159, 145), and right eyelid opening, distance(386, 374), divided by outer-eye distance(33, 263). | Normalized eyelid opening. The code does not use iris landmarks for this quantity despite its name. |
| `ESO` (`eso_norm`) | Euclidean distance between left iris center 468 and right iris center 473, divided by outer-eye distance(33, 263). | Normalized distance between iris centers. |
| `ENSO` (`enso_norm`) | Euclidean distance between the midpoint of iris centers 468 and 473 and subnasale approximation 2, divided by outer-eye distance(33, 263). | Normalized eye-region-to-subnasale distance. |
| `MNSO` (`mnso_norm`) | Euclidean distance between upper-lip center 0 and subnasale approximation 2, divided by outer-eye distance(33, 263). | Normalized upper-lip-to-subnasale distance. |
| `MWO` (`mwo_norm`) | Euclidean distance between mouth-left 61 and mouth-right 291, divided by outer-eye distance(33, 263). | Normalized mouth width. |

The abbreviations are retained as pipeline feature names. Their definitions
above describe exactly what the code calculates; no additional anatomical
meaning is assumed beyond those calculations.

## Trial-level facial predictor columns

The following 10 columns are valid facial candidate predictors. Each is
calculated across detected frames only within a trial’s two-second image window.

| Column | Meaning |
|---|---|
| `video_irisdo_norm_mean` | Trial mean normalized eyelid opening. |
| `video_irisdo_norm_std` | Trial standard deviation of normalized eyelid opening. |
| `video_eso_norm_mean` | Trial mean normalized iris-center separation. |
| `video_eso_norm_std` | Trial standard deviation of normalized iris-center separation. |
| `video_enso_norm_mean` | Trial mean normalized iris-midpoint-to-subnasale distance. |
| `video_enso_norm_std` | Trial standard deviation of normalized iris-midpoint-to-subnasale distance. |
| `video_mnso_norm_mean` | Trial mean normalized upper-lip-to-subnasale distance. |
| `video_mnso_norm_std` | Trial standard deviation of normalized upper-lip-to-subnasale distance. |
| `video_mwo_norm_mean` | Trial mean normalized mouth width. |
| `video_mwo_norm_std` | Trial standard deviation of normalized mouth width. |

These measures describe facial geometry and within-window movement variability,
not validated emotion labels. They can be affected by pose, camera view,
landmark error, and expression changes unrelated to the presented image.

## Column roles and modeling eligibility

### Valid predictors

- EEG: all 120 channel-specific columns from the 15 EEG families above
  (15 families × 8 channels).
- Face: the 10 trial-level facial predictor columns listed above.

Future modeling will retain all eight EEG channels and focus on EEG feature
selection rather than electrode-selection analysis.

### Identifiers

- `trigger`
- `stim_id`

These fields connect records but must not be inputs.

### Participant ratings and response-side variables

- `valence_rating`
- `arousal_rating`
- `valence_rt`
- `arousal_rt`

The first two are the participant-specific prediction targets. Rating reaction
times are recorded after stimulus presentation and are response-side variables,
not physiological or facial predictors; they must be excluded.

### Stimulus metadata

- `category`

`category` labels the experiment’s intended affect grouping. It is metadata,
not a valid predictor.

### Quality-control and export metadata

- `epoch_index__C3` through `epoch_index__Pz`
- `video_frame_count`
- `face_detected_frames`
- `face_detection_rate`

`epoch_index__*` records EEG export order. The video fields describe sampling
and face-detection quality. Face-detection rate is 1.0 for all 120 P01 trials,
so it is constant and cannot provide useful variation here.

### Leakage-risk columns and values

Do not model:

- `trigger`, because trigger ranges encode the experiment condition;
- `category`, because it encodes the intended high/low affect grouping;
- `stim_id`, because item identity can memorize stimulus-specific ratings;
- `epoch_index__*`, because they encode recording/export order rather than
  physiology;
- participant ratings and reaction times;
- OASIS valence/arousal means, which are normative references and must not be
  mixed with participant targets;
- synchronization variables or face-quality fields as emotion predictors.

## Scaling and future validation

Scaling is required for kNN, SVM, and SVR because the features have different
units and numerical ranges: EEG power, entropy, Hjorth measures, frequency,
and normalized facial distances are not directly comparable. A `StandardScaler`
will be fitted on each training fold only and included inside the scikit-learn
pipeline.

For later analyses:

- Regression will use repeated KFold.
- Classification will use repeated stratified KFold when class counts allow it.
- Scaling and any feature selection will be fitted only inside each training
  fold. No electrode-selection analysis is planned at this stage.
