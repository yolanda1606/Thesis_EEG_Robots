# P19 individual continuous-rating regression

Command: `python -u processing/multimodal_image/modeling/train_regression.py --mode individual --target all --modality all --models knn,svr,ridge --feature-counts all,5,10,20 --participants P19 --run-name p19_individual_regression_v1`.

Primary predictions are unconstrained. Inner tuning used `neg_root_mean_squared_error`; sklearn negates this score for its greater-is-better convention.

## Valence

### Best overall

- Model/features: `ridge`, `5`.
- RMSE: 1.567 ± 0.270; range 1.250–1.947.
- MAE: 1.248; R2: 0.009; explained variance: 0.047.
- R2 range: -0.101–0.065.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### Best EEG

- Model/features: `svr`, `10`.
- RMSE: 1.600 ± 0.333; range 1.241–2.129.
- MAE: 1.246; R2: -0.025; explained variance: -0.005.
- R2 range: -0.117–0.027.
- Most frequent tuned hyperparameters: {"regressor__C": 0.1, "regressor__epsilon": 0.1, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (4/5 folds); {"regressor__C": 1, "regressor__epsilon": 0.25, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (1/5 folds).

### Best face

- Model/features: `ridge`, `5`.
- RMSE: 1.567 ± 0.270; range 1.250–1.947.
- MAE: 1.248; R2: 0.009; explained variance: 0.047.
- R2 range: -0.101–0.065.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### Best multimodal

- Model/features: `svr`, `10`.
- RMSE: 1.599 ± 0.338; range 1.235–2.129.
- MAE: 1.242; R2: -0.023; explained variance: -0.001.
- R2 range: -0.117–0.036.
- Most frequent tuned hyperparameters: {"regressor__C": 0.1, "regressor__epsilon": 0.1, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (4/5 folds); {"regressor__C": 1, "regressor__epsilon": 0.25, "regressor__gamma": 0.01, "regressor__kernel": "rbf"} (1/5 folds).

### All configurations

| modality | regressor | feature_count_request | rmse_mean | rmse_std | mae_mean | r2_mean | explained_variance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| face | ridge | 5 | 1.567 | 0.270 | 1.248 | 0.009 | 0.047 |
| face | ridge | all | 1.582 | 0.267 | 1.266 | -0.012 | 0.026 |
| face | ridge | 10 | 1.582 | 0.267 | 1.266 | -0.012 | 0.026 |
| face | ridge | 20 | 1.582 | 0.267 | 1.266 | -0.012 | 0.026 |
| face | svr | 5 | 1.597 | 0.306 | 1.254 | -0.025 | 0.001 |
| multimodal | svr | 10 | 1.599 | 0.338 | 1.242 | -0.023 | -0.001 |
| eeg | svr | 10 | 1.600 | 0.333 | 1.246 | -0.025 | -0.005 |
| multimodal | svr | 20 | 1.603 | 0.353 | 1.247 | -0.026 | -0.003 |
| eeg | svr | 20 | 1.603 | 0.329 | 1.248 | -0.030 | -0.012 |
| eeg | svr | all | 1.605 | 0.295 | 1.247 | -0.037 | -0.002 |
| multimodal | svr | all | 1.605 | 0.294 | 1.246 | -0.037 | -0.001 |
| eeg | svr | 5 | 1.612 | 0.350 | 1.257 | -0.038 | -0.023 |
| face | svr | 10 | 1.631 | 0.252 | 1.290 | -0.084 | -0.045 |
| face | svr | all | 1.631 | 0.252 | 1.290 | -0.084 | -0.045 |
| face | svr | 20 | 1.631 | 0.252 | 1.290 | -0.084 | -0.045 |
| multimodal | svr | 5 | 1.636 | 0.352 | 1.288 | -0.070 | -0.053 |
| multimodal | ridge | 10 | 1.646 | 0.352 | 1.301 | -0.083 | -0.037 |
| eeg | knn | 20 | 1.647 | 0.342 | 1.337 | -0.085 | -0.010 |
| eeg | ridge | 10 | 1.652 | 0.345 | 1.322 | -0.093 | -0.056 |
| eeg | knn | all | 1.654 | 0.267 | 1.320 | -0.109 | -0.054 |
| face | knn | 5 | 1.655 | 0.227 | 1.300 | -0.124 | -0.075 |
| eeg | ridge | 5 | 1.657 | 0.353 | 1.322 | -0.097 | -0.063 |
| multimodal | knn | all | 1.666 | 0.242 | 1.314 | -0.133 | -0.059 |
| eeg | knn | 10 | 1.668 | 0.293 | 1.356 | -0.122 | -0.078 |
| face | knn | 10 | 1.670 | 0.206 | 1.314 | -0.154 | -0.093 |
| face | knn | 20 | 1.670 | 0.206 | 1.314 | -0.154 | -0.093 |
| face | knn | all | 1.670 | 0.206 | 1.314 | -0.154 | -0.093 |
| multimodal | ridge | 5 | 1.672 | 0.341 | 1.330 | -0.120 | -0.082 |
| multimodal | knn | 20 | 1.674 | 0.334 | 1.319 | -0.124 | -0.045 |
| multimodal | ridge | 20 | 1.677 | 0.377 | 1.324 | -0.122 | -0.067 |
| eeg | ridge | 20 | 1.688 | 0.370 | 1.348 | -0.139 | -0.096 |
| multimodal | knn | 10 | 1.713 | 0.267 | 1.362 | -0.191 | -0.138 |
| multimodal | knn | 5 | 1.723 | 0.352 | 1.417 | -0.189 | -0.149 |
| multimodal | ridge | all | 1.776 | 0.342 | 1.466 | -0.286 | -0.215 |
| eeg | ridge | all | 1.785 | 0.369 | 1.484 | -0.298 | -0.237 |
| eeg | knn | 5 | 1.880 | 0.261 | 1.545 | -0.476 | -0.428 |

## Arousal

### Best overall

- Model/features: `ridge`, `20`.
- RMSE: 1.544 ± 0.311; range 1.251–1.944.
- MAE: 1.293; R2: 0.104; explained variance: 0.128.
- R2 range: 0.014–0.182.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### Best EEG

- Model/features: `ridge`, `20`.
- RMSE: 1.544 ± 0.311; range 1.251–1.944.
- MAE: 1.293; R2: 0.104; explained variance: 0.128.
- R2 range: 0.014–0.182.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### Best face

- Model/features: `ridge`, `5`.
- RMSE: 1.669 ± 0.301; range 1.328–1.955.
- MAE: 1.438; R2: -0.054; explained variance: -0.040.
- R2 range: -0.237–0.013.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### Best multimodal

- Model/features: `ridge`, `10`.
- RMSE: 1.556 ± 0.292; range 1.255–1.922.
- MAE: 1.309; R2: 0.088; explained variance: 0.109.
- R2 range: 0.036–0.127.
- Most frequent tuned hyperparameters: {"regressor__alpha": 100} (5/5 folds).

### All configurations

| modality | regressor | feature_count_request | rmse_mean | rmse_std | mae_mean | r2_mean | explained_variance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eeg | ridge | 20 | 1.544 | 0.311 | 1.293 | 0.104 | 0.128 |
| eeg | ridge | 10 | 1.556 | 0.292 | 1.309 | 0.088 | 0.109 |
| multimodal | ridge | 10 | 1.556 | 0.292 | 1.309 | 0.088 | 0.109 |
| eeg | ridge | 5 | 1.558 | 0.298 | 1.310 | 0.086 | 0.108 |
| multimodal | ridge | 5 | 1.558 | 0.298 | 1.310 | 0.086 | 0.108 |
| multimodal | ridge | 20 | 1.567 | 0.309 | 1.319 | 0.075 | 0.099 |
| multimodal | knn | 5 | 1.570 | 0.324 | 1.297 | 0.075 | 0.097 |
| eeg | knn | 5 | 1.570 | 0.324 | 1.297 | 0.075 | 0.097 |
| multimodal | knn | 10 | 1.584 | 0.361 | 1.313 | 0.064 | 0.076 |
| eeg | knn | 10 | 1.590 | 0.359 | 1.309 | 0.057 | 0.067 |
| eeg | svr | 5 | 1.594 | 0.310 | 1.342 | 0.045 | 0.071 |
| multimodal | svr | 5 | 1.594 | 0.310 | 1.345 | 0.045 | 0.075 |
| eeg | svr | 10 | 1.603 | 0.301 | 1.344 | 0.032 | 0.071 |
| multimodal | svr | 10 | 1.603 | 0.301 | 1.340 | 0.032 | 0.067 |
| eeg | svr | 20 | 1.613 | 0.267 | 1.359 | 0.010 | 0.038 |
| multimodal | svr | 20 | 1.616 | 0.266 | 1.361 | 0.006 | 0.035 |
| eeg | knn | all | 1.628 | 0.302 | 1.351 | -0.002 | 0.018 |
| eeg | knn | 20 | 1.632 | 0.315 | 1.352 | -0.002 | 0.018 |
| eeg | svr | all | 1.634 | 0.302 | 1.370 | -0.007 | 0.027 |
| multimodal | knn | 20 | 1.636 | 0.313 | 1.369 | -0.008 | 0.011 |
| face | ridge | 5 | 1.669 | 0.301 | 1.438 | -0.054 | -0.040 |
| face | ridge | 20 | 1.672 | 0.305 | 1.445 | -0.058 | -0.045 |
| face | ridge | all | 1.672 | 0.305 | 1.445 | -0.058 | -0.045 |
| face | ridge | 10 | 1.672 | 0.305 | 1.445 | -0.058 | -0.045 |
| face | svr | 5 | 1.686 | 0.296 | 1.416 | -0.075 | -0.021 |
| multimodal | svr | all | 1.688 | 0.291 | 1.439 | -0.080 | -0.053 |
| face | svr | all | 1.697 | 0.289 | 1.428 | -0.090 | -0.040 |
| face | svr | 10 | 1.697 | 0.289 | 1.428 | -0.090 | -0.040 |
| face | svr | 20 | 1.697 | 0.289 | 1.428 | -0.090 | -0.040 |
| multimodal | knn | all | 1.707 | 0.218 | 1.419 | -0.120 | -0.106 |
| face | knn | 10 | 1.748 | 0.247 | 1.471 | -0.170 | -0.159 |
| face | knn | all | 1.748 | 0.247 | 1.471 | -0.170 | -0.159 |
| face | knn | 20 | 1.748 | 0.247 | 1.471 | -0.170 | -0.159 |
| face | knn | 5 | 1.750 | 0.251 | 1.495 | -0.174 | -0.158 |
| eeg | ridge | all | 1.755 | 0.321 | 1.483 | -0.163 | -0.136 |
| multimodal | ridge | all | 1.793 | 0.320 | 1.524 | -0.217 | -0.188 |

## Best P19 regression configuration

Best overall: arousal, eeg, ridge, 20 features (mean RMSE 1.544).

### Fold metrics

| fold | rmse | mae | r2 | explained_variance |
| --- | --- | --- | --- | --- |
| 1.000 | 1.944 | 1.644 | 0.014 | 0.019 |
| 2.000 | 1.485 | 1.281 | 0.087 | 0.115 |
| 3.000 | 1.259 | 1.075 | 0.182 | 0.227 |
| 4.000 | 1.780 | 1.500 | 0.135 | 0.139 |
| 5.000 | 1.251 | 0.966 | 0.104 | 0.140 |

### Selected features

| Feature | Interpretation | Frequency | Mean score |
| --- | --- | ---: | ---: |
| eeg_hc__PO8 | Hjorth complexity at PO8 | 1.00 | 14.476 |
| eeg_mf_hz__PO8 | mean frequency (Hz) at PO8 | 1.00 | 11.673 |
| eeg_hm__PO8 | Hjorth mobility at PO8 | 1.00 | 10.740 |
| eeg_hm__Cz | Hjorth mobility at Cz | 1.00 | 9.974 |
| eeg_se__PO8 | spectral entropy at PO8 | 1.00 | 9.776 |
| eeg_hc__Fz | Hjorth complexity at Fz | 1.00 | 9.255 |
| eeg_hc__Cz | Hjorth complexity at Cz | 1.00 | 8.749 |
| eeg_bp_beta__PO8 | band power in beta band at PO8 | 1.00 | 8.669 |
| eeg_mf_hz__Cz | mean frequency (Hz) at Cz | 1.00 | 8.490 |
| eeg_bp_beta__Pz | band power in beta band at Pz | 1.00 | 8.413 |
| eeg_bp_beta__Fz | band power in beta band at Fz | 1.00 | 7.660 |
| eeg_bp_beta__PO7 | band power in beta band at PO7 | 1.00 | 7.221 |
| eeg_bp_beta__C4 | band power in beta band at C4 | 0.80 | 7.210 |
| eeg_bp_beta__C3 | band power in beta band at C3 | 0.80 | 7.187 |
| eeg_hm__Oz | Hjorth mobility at Oz | 0.60 | 7.475 |
| eeg_mf_hz__Fz | mean frequency (Hz) at Fz | 0.60 | 6.348 |
| eeg_bp_delta__PO8 | band power in delta band at PO8 | 0.60 | 6.213 |
| eeg_mf_hz__Pz | mean frequency (Hz) at Pz | 0.60 | 5.816 |
| eeg_bp_delta__Cz | band power in delta band at Cz | 0.60 | 5.747 |
| eeg_hc__C3 | Hjorth complexity at C3 | 0.60 | 5.254 |
| eeg_se__Fz | spectral entropy at Fz | 0.40 | 5.862 |
| eeg_hc__Oz | Hjorth complexity at Oz | 0.40 | 5.857 |
| eeg_hm__Fz | Hjorth mobility at Fz | 0.40 | 5.282 |
| eeg_bp_delta__C3 | band power in delta band at C3 | 0.40 | 4.244 |
| eeg_se__Cz | spectral entropy at Cz | 0.20 | 5.163 |
| eeg_se__Oz | spectral entropy at Oz | 0.20 | 5.021 |
| eeg_bp_beta__Cz | band power in beta band at Cz | 0.20 | 4.914 |
| eeg_hc__PO7 | Hjorth complexity at PO7 | 0.20 | 4.603 |
| eeg_sd__C4 | standard deviation at C4 | 0.20 | 4.599 |
| eeg_hm__PO7 | Hjorth mobility at PO7 | 0.20 | 3.893 |
| eeg_bp_delta__Oz | band power in delta band at Oz | 0.00 | 4.487 |
| eeg_hm__Pz | Hjorth mobility at Pz | 0.00 | 4.132 |
| eeg_bp_alpha__C4 | band power in alpha band at C4 | 0.00 | 3.961 |
| eeg_mf_hz__Oz | mean frequency (Hz) at Oz | 0.00 | 3.567 |
| eeg_bp_delta__Fz | band power in delta band at Fz | 0.00 | 3.562 |
| eeg_hc__Pz | Hjorth complexity at Pz | 0.00 | 3.487 |
| eeg_mf_hz__PO7 | mean frequency (Hz) at PO7 | 0.00 | 3.432 |
| eeg_bp_gamma__PO7 | band power in gamma band at PO7 | 0.00 | 2.952 |
| eeg_hc__C4 | Hjorth complexity at C4 | 0.00 | 2.942 |
| eeg_mf_hz__C3 | mean frequency (Hz) at C3 | 0.00 | 2.904 |
| eeg_se_gamma__Oz | spectral entropy in gamma band at Oz | 0.00 | 2.766 |
| eeg_se__Pz | spectral entropy at Pz | 0.00 | 2.707 |
| eeg_se_alpha__C4 | spectral entropy in alpha band at C4 | 0.00 | 2.650 |
| eeg_bp_beta__Oz | band power in beta band at Oz | 0.00 | 2.649 |
| eeg_bp_alpha__PO7 | band power in alpha band at PO7 | 0.00 | 2.646 |
| eeg_bp_alpha__C3 | band power in alpha band at C3 | 0.00 | 2.483 |
| eeg_se_gamma__Pz | spectral entropy in gamma band at Pz | 0.00 | 2.443 |
| eeg_se_alpha__Fz | spectral entropy in alpha band at Fz | 0.00 | 2.414 |
| eeg_bp_theta__C4 | band power in theta band at C4 | 0.00 | 2.385 |
| eeg_se_gamma__C4 | spectral entropy in gamma band at C4 | 0.00 | 2.120 |
| eeg_se_beta__Oz | spectral entropy in beta band at Oz | 0.00 | 1.975 |
| eeg_se_beta__C3 | spectral entropy in beta band at C3 | 0.00 | 1.946 |
| eeg_sd__Oz | standard deviation at Oz | 0.00 | 1.937 |
| eeg_sd__C3 | standard deviation at C3 | 0.00 | 1.916 |
| eeg_se_alpha__C3 | spectral entropy in alpha band at C3 | 0.00 | 1.816 |
| eeg_bp_alpha__Oz | band power in alpha band at Oz | 0.00 | 1.712 |
| eeg_se__C4 | spectral entropy at C4 | 0.00 | 1.633 |
| eeg_se_theta__Cz | spectral entropy in theta band at Cz | 0.00 | 1.623 |
| eeg_bp_alpha__PO8 | band power in alpha band at PO8 | 0.00 | 1.599 |
| eeg_bp_theta__C3 | band power in theta band at C3 | 0.00 | 1.571 |
| eeg_bp_gamma__Oz | band power in gamma band at Oz | 0.00 | 1.539 |
| eeg_se_gamma__Fz | spectral entropy in gamma band at Fz | 0.00 | 1.465 |
| eeg_sd__PO7 | standard deviation at PO7 | 0.00 | 1.454 |
| eeg_bp_theta__PO7 | band power in theta band at PO7 | 0.00 | 1.332 |
| eeg_bp_theta__Fz | band power in theta band at Fz | 0.00 | 1.225 |
| eeg_se_delta__Cz | spectral entropy in delta band at Cz | 0.00 | 1.220 |
| eeg_se_theta__Fz | spectral entropy in theta band at Fz | 0.00 | 1.190 |
| eeg_bp_gamma__Cz | band power in gamma band at Cz | 0.00 | 1.157 |
| eeg_se_delta__C3 | spectral entropy in delta band at C3 | 0.00 | 1.137 |
| eeg_bp_gamma__C3 | band power in gamma band at C3 | 0.00 | 1.109 |
| eeg_bp_alpha__Cz | band power in alpha band at Cz | 0.00 | 1.088 |
| eeg_sd__PO8 | standard deviation at PO8 | 0.00 | 1.078 |
| eeg_bp_gamma__C4 | band power in gamma band at C4 | 0.00 | 1.076 |
| eeg_se_beta__Pz | spectral entropy in beta band at Pz | 0.00 | 1.071 |
| eeg_se_beta__Fz | spectral entropy in beta band at Fz | 0.00 | 1.045 |
| eeg_se_alpha__PO8 | spectral entropy in alpha band at PO8 | 0.00 | 0.972 |
| eeg_bp_delta__PO7 | band power in delta band at PO7 | 0.00 | 0.917 |
| eeg_hm__C4 | Hjorth mobility at C4 | 0.00 | 0.852 |
| eeg_bp_gamma__Pz | band power in gamma band at Pz | 0.00 | 0.842 |
| eeg_se_beta__PO8 | spectral entropy in beta band at PO8 | 0.00 | 0.830 |
| eeg_se_theta__PO8 | spectral entropy in theta band at PO8 | 0.00 | 0.761 |
| eeg_se_alpha__Oz | spectral entropy in alpha band at Oz | 0.00 | 0.741 |
| eeg_bp_theta__PO8 | band power in theta band at PO8 | 0.00 | 0.733 |
| eeg_se__PO7 | spectral entropy at PO7 | 0.00 | 0.718 |
| eeg_bp_delta__Pz | band power in delta band at Pz | 0.00 | 0.713 |
| eeg_se_alpha__PO7 | spectral entropy in alpha band at PO7 | 0.00 | 0.670 |
| eeg_mf_hz__C4 | mean frequency (Hz) at C4 | 0.00 | 0.660 |
| eeg_se_gamma__Cz | spectral entropy in gamma band at Cz | 0.00 | 0.655 |
| eeg_se_delta__C4 | spectral entropy in delta band at C4 | 0.00 | 0.589 |
| eeg_se_delta__Pz | spectral entropy in delta band at Pz | 0.00 | 0.583 |
| eeg_bp_alpha__Fz | band power in alpha band at Fz | 0.00 | 0.574 |
| eeg_se_theta__PO7 | spectral entropy in theta band at PO7 | 0.00 | 0.526 |
| eeg_se_alpha__Pz | spectral entropy in alpha band at Pz | 0.00 | 0.496 |
| eeg_se_gamma__PO7 | spectral entropy in gamma band at PO7 | 0.00 | 0.473 |
| eeg_se_gamma__C3 | spectral entropy in gamma band at C3 | 0.00 | 0.433 |
| eeg_bp_alpha__Pz | band power in alpha band at Pz | 0.00 | 0.425 |
| eeg_bp_gamma__Fz | band power in gamma band at Fz | 0.00 | 0.419 |
| eeg_se_delta__PO8 | spectral entropy in delta band at PO8 | 0.00 | 0.412 |
| eeg_bp_gamma__PO8 | band power in gamma band at PO8 | 0.00 | 0.357 |
| eeg_se_theta__C3 | spectral entropy in theta band at C3 | 0.00 | 0.348 |
| eeg_se_gamma__PO8 | spectral entropy in gamma band at PO8 | 0.00 | 0.340 |
| eeg_se_theta__C4 | spectral entropy in theta band at C4 | 0.00 | 0.337 |
| eeg_se_beta__PO7 | spectral entropy in beta band at PO7 | 0.00 | 0.329 |
| eeg_se_delta__PO7 | spectral entropy in delta band at PO7 | 0.00 | 0.309 |
| eeg_se_theta__Oz | spectral entropy in theta band at Oz | 0.00 | 0.301 |
| eeg_se__C3 | spectral entropy at C3 | 0.00 | 0.276 |
| eeg_hm__C3 | Hjorth mobility at C3 | 0.00 | 0.241 |
| eeg_sd__Fz | standard deviation at Fz | 0.00 | 0.210 |
| eeg_se_theta__Pz | spectral entropy in theta band at Pz | 0.00 | 0.205 |
| eeg_se_beta__C4 | spectral entropy in beta band at C4 | 0.00 | 0.204 |
| eeg_bp_theta__Oz | band power in theta band at Oz | 0.00 | 0.202 |
| eeg_bp_delta__C4 | band power in delta band at C4 | 0.00 | 0.200 |
| eeg_sd__Pz | standard deviation at Pz | 0.00 | 0.189 |
| eeg_se_delta__Fz | spectral entropy in delta band at Fz | 0.00 | 0.152 |
| eeg_se_delta__Oz | spectral entropy in delta band at Oz | 0.00 | 0.138 |
| eeg_bp_theta__Pz | band power in theta band at Pz | 0.00 | 0.119 |
| eeg_se_beta__Cz | spectral entropy in beta band at Cz | 0.00 | 0.110 |
| eeg_bp_theta__Cz | band power in theta band at Cz | 0.00 | 0.076 |
| eeg_se_alpha__Cz | spectral entropy in alpha band at Cz | 0.00 | 0.058 |
| eeg_sd__Cz | standard deviation at Cz | 0.00 | 0.024 |

- True-rating mean ± SD: 3.617 ± 1.661.
- Predicted-rating mean ± SD: 3.607 ± 0.649.

## Classification vs Regression

Classification asks: can we distinguish LOW (<4) vs HIGH (>=4)?

Regression asks: can we predict the original continuous rating?

Balanced accuracy and RMSE/R2 assess different tasks and are not numerically comparable.

- Valence best classification balanced accuracy: 0.600 (eeg, gnb, 10 features).
- Valence best regression RMSE/R2: 1.567 / 0.009 (face, ridge, 5 features).
  - Same modality: False; same model family: False; shared selected features: none.

- Arousal best classification balanced accuracy: 0.683 (eeg, knn, 10 features).
- Arousal best regression RMSE/R2: 1.544 / 0.104 (eeg, ridge, 20 features).
  - Same modality: True; same model family: False; shared selected features: eeg_bp_beta__C3, eeg_bp_beta__C4, eeg_bp_beta__Cz, eeg_bp_beta__Fz, eeg_bp_beta__PO7, eeg_bp_beta__PO8, eeg_bp_beta__Pz, eeg_bp_delta__C3, eeg_bp_delta__Cz, eeg_hc__C3, eeg_hc__Cz, eeg_hc__Fz, eeg_hc__Oz, eeg_hc__PO7, eeg_hc__PO8, eeg_hm__Cz, eeg_hm__Fz, eeg_hm__Oz, eeg_hm__PO7, eeg_hm__PO8, eeg_mf_hz__Cz, eeg_mf_hz__Fz, eeg_mf_hz__PO8, eeg_sd__C4, eeg_se__Cz, eeg_se__Fz, eeg_se__Oz, eeg_se__PO8.
