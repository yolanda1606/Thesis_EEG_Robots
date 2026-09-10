# P19 Image-to-Robot Transfer Validity Audit

## Verdict: `TRANSFER_SHOWS_SUBSTANTIAL_DOMAIN_SHIFT`

- 60.0% of selected feature/model rows have robot 95th-percentile |z| above 3.
- 66.7% have more than 25% of robot windows outside the Image observed min/max range.
- 90.0% of task/model summaries have more than half of windows beyond the Image leave-self-out nearest-distance 95th percentile.

This is a diagnostic comparison against the frozen P19 Image calibration domain. It does not infer emotion states, alter predictions, or establish clinical/scientific validity.

## Calibration and data scope

- Frozen models, scalers, selectors, hyperparameters, robot features, and primary inference outputs were read without modification.
- Five Status-aligned robot tasks / 337 windows were audited. Shape Sorter Alone remains excluded.
- Mean/SD z diagnostics are descriptive; feature distributions may be non-Gaussian, so threshold counts are not invalidity rules.

## Strongest selected-feature shifts

- valence/eeg_bp_delta__Fz: robot mean shift 1409.30 Image SD; 81.9% outside Image min/max; robot 95th |z|=109.47.
- arousal/eeg_hc__Fz: robot mean shift 11.48 Image SD; 79.8% outside Image min/max; robot 95th |z|=21.63.
- arousal/eeg_se__Fz: robot mean shift -7.29 Image SD; 80.1% outside Image min/max; robot 95th |z|=11.62.
- valence/eeg_hc__PO8: robot mean shift 3.71 Image SD; 34.1% outside Image min/max; robot 95th |z|=10.23.
- arousal/eeg_hc__PO8: robot mean shift 3.71 Image SD; 34.1% outside Image min/max; robot 95th |z|=10.23.
- arousal/eeg_bp_beta__PO8: robot mean shift 4.25 Image SD; 36.8% outside Image min/max; robot 95th |z|=9.59.
- valence/eeg_se__PO8: robot mean shift -2.22 Image SD; 40.9% outside Image min/max; robot 95th |z|=7.05.
- arousal/eeg_hm__Fz: robot mean shift -3.78 Image SD; 74.2% outside Image min/max; robot 95th |z|=5.71.

## Distance geometry

- Valence Image leave-self-out nearest distance: median 0.645, 95th percentile 1.310, maximum 2.390.
  - Pick and Place: median robot nearest distance 19.015; 100.0% above Image 95th percentile; 97.6% above Image maximum.
  - Shape Sorter Observation: median robot nearest distance 20.089; 98.3% above Image 95th percentile; 93.2% above Image maximum.
  - Stack: median robot nearest distance 29.312; 100.0% above Image 95th percentile; 100.0% above Image maximum.
  - Sisyphus: median robot nearest distance 30.438; 100.0% above Image 95th percentile; 98.9% above Image maximum.
  - Shape Sorter Interaction: median robot nearest distance 1.199; 45.5% above Image 95th percentile; 16.9% above Image maximum.
- Arousal Image leave-self-out nearest distance: median 1.371, 95th percentile 2.485, maximum 3.320.
  - Pick and Place: median robot nearest distance 9.514; 100.0% above Image 95th percentile; 100.0% above Image maximum.
  - Shape Sorter Observation: median robot nearest distance 12.758; 98.3% above Image 95th percentile; 94.9% above Image maximum.
  - Stack: median robot nearest distance 14.189; 100.0% above Image 95th percentile; 100.0% above Image maximum.
  - Sisyphus: median robot nearest distance 14.495; 100.0% above Image 95th percentile; 100.0% above Image maximum.
  - Shape Sorter Interaction: median robot nearest distance 5.450; 97.4% above Image 95th percentile; 89.6% above Image maximum.

## Arousal kNN geometry

- Among 205 robot windows with arousal HIGH probability >=0.90, 99.5% are beyond the Image nearest-distance 95th percentile.
- The neighbour audit records exact 11-neighbour distances, labels, and distance-weighted vote fractions for every robot window; the reconstructed weighted fraction exactly matches the frozen classifier probability.

## Valence GaussianNB behavior

- Pick and Place: 100.0% scores <0.05; most common log-likelihood driver `eeg_bp_delta__Fz` (100.0% of windows), so one feature commonly dominates.
- Shape Sorter Observation: 100.0% scores <0.05; most common log-likelihood driver `eeg_bp_delta__Fz` (94.9% of windows), so one feature commonly dominates.
- Stack: 100.0% scores <0.05; most common log-likelihood driver `eeg_bp_delta__Fz` (100.0% of windows), so one feature commonly dominates.
- Sisyphus: 100.0% scores <0.05; most common log-likelihood driver `eeg_bp_delta__Fz` (100.0% of windows), so one feature commonly dominates.
- Shape Sorter Interaction: 24.7% scores <0.05; most common log-likelihood driver `eeg_mf_hz__PO8` (46.8% of windows), so likelihood separation is distributed across features/windows.

## Arousal kNN geometry by task

- Pick and Place: mean distance-weighted HIGH neighbour vote 0.890; 36 windows score >=0.90, of which 100.0% exceed the Image nearest-distance 95th percentile.
- Shape Sorter Observation: mean distance-weighted HIGH neighbour vote 0.888; 53 windows score >=0.90, of which 98.1% exceed the Image nearest-distance 95th percentile.
- Stack: mean distance-weighted HIGH neighbour vote 0.874; 55 windows score >=0.90, of which 100.0% exceed the Image nearest-distance 95th percentile.
- Sisyphus: mean distance-weighted HIGH neighbour vote 0.852; 59 windows score >=0.90, of which 100.0% exceed the Image nearest-distance 95th percentile.
- Shape Sorter Interaction: mean distance-weighted HIGH neighbour vote 0.359; 2 windows score >=0.90, of which 100.0% exceed the Image nearest-distance 95th percentile.

## Shape Sorter Interaction reversal

- valence/eeg_bp_delta__Fz: Interaction robot mean is 1103.54 Image SD from the Image mean; task-level 95th |z|=9.89.
- arousal/eeg_bp_beta__PO8: Interaction robot mean is 5.95 Image SD from the Image mean; task-level 95th |z|=9.91.
- arousal/eeg_hc__Fz: Interaction robot mean is 3.21 Image SD from the Image mean; task-level 95th |z|=6.64.
- arousal/eeg_se__Fz: Interaction robot mean is -1.91 Image SD from the Image mean; task-level 95th |z|=6.68.
- arousal/eeg_hm__PO8: Interaction robot mean is 1.33 Image SD from the Image mean; task-level 95th |z|=3.13.
- arousal/eeg_hm__Fz: Interaction robot mean is -1.29 Image SD from the Image mean; task-level 95th |z|=3.40.
- valence/eeg_mf_hz__Fz: Interaction robot mean is -1.15 Image SD from the Image mean; task-level 95th |z|=2.26.
- arousal/eeg_mf_hz__Fz: Interaction robot mean is -1.15 Image SD from the Image mean; task-level 95th |z|=2.26.

## Post-task ratings: descriptive only

| Task | Rating valence | Mean valence HIGH score | Rating arousal | Mean arousal HIGH score |
|---|---:|---:|---:|---:|
| Pick and Place | 5 | 0.000 | 5 | 0.890 |
| Shape Sorter Observation | 5 | 0.001 | 5 | 0.888 |
| Stack | 4 | 0.000 | 4 | 0.874 |
| Sisyphus | 4 | 0.000 | 5 | 0.852 |
| Shape Sorter Interaction | 6 | 0.622 | 6 | 0.359 |

No formal correlation or correctness claim is made from five post-task ratings: each rating summarizes a whole task, while the classifier outputs Image-calibrated window probabilities.

## Temporal score behavior

- Valence / Pick and Place: first=0.000, last=0.000, median=0.000, IQR=0.000, <0.05=100.0%, >0.95=0.0%.
- Valence / Shape Sorter Observation: first=0.000, last=0.000, median=0.000, IQR=0.000, <0.05=100.0%, >0.95=0.0%.
- Valence / Stack: first=0.000, last=0.000, median=0.000, IQR=0.000, <0.05=100.0%, >0.95=0.0%.
- Valence / Sisyphus: first=0.000, last=0.000, median=0.000, IQR=0.000, <0.05=100.0%, >0.95=0.0%.
- Valence / Shape Sorter Interaction: first=0.000, last=0.000, median=0.860, IQR=0.748, <0.05=24.7%, >0.95=26.0%.
- Arousal / Pick and Place: first=0.549, last=0.911, median=0.914, IQR=0.003, <0.05=0.0%, >0.95=0.0%.
- Arousal / Shape Sorter Observation: first=0.640, last=0.913, median=0.914, IQR=0.002, <0.05=0.0%, >0.95=0.0%.
- Arousal / Stack: first=0.554, last=0.826, median=0.913, IQR=0.001, <0.05=0.0%, >0.95=0.0%.
- Arousal / Sisyphus: first=0.555, last=0.559, median=0.912, IQR=0.089, <0.05=0.0%, >0.95=0.0%.
- Arousal / Shape Sorter Interaction: first=0.549, last=0.545, median=0.264, IQR=0.356, <0.05=0.0%, >0.95=0.0%.

## Scientific limitation

Initial P19 transfer-pilot limitation: the existing robot feature table has incomplete ICA provenance and its exact producing source version cannot currently be independently reconstructed. Robot preprocessing is not claimed to be proven perfectly equivalent to the Image preprocessing pipeline.
The audit result should guide further preprocessing/model investigation before strong direct Image-to-Robot interpretations.
