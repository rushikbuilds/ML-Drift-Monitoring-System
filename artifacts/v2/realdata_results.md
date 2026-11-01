# Real-data study (leave-one-dataset-out)

| dataset | windows | windows with event | mean baseline accuracy |
|---|---:|---:|---:|
| electricity | 5 | 5 | 0.725 |
| airlines | 40 | 28 | 0.622 |
| covertype | 40 | 27 | 0.798 |
| bank_marketing | 5 | 3 | 0.913 |

## Label delay L = 0 (pooled over held-out datasets)

| method | mean AUROC | recall | early-warning rate | FA/100 clean |
|---|---:|---:|---:|---:|
| stat_only | 0.57 | 0.40 [0.29, 0.51] | 0.24 [0.14, 0.35] | 28.63 |
| conf_only | 0.51 | 0.11 [0.05, 0.19] | 0.05 [0.00, 0.11] | 3.67 |
| fixed_score | 0.70 | 0.43 [0.32, 0.56] | 0.29 [0.17, 0.40] | 21.31 |
| slope_forecast | 0.63 | 0.41 [0.29, 0.54] | 0.24 [0.14, 0.35] | 5.85 |
| delayed_label_only | 0.77 | 0.97 [0.92, 1.00] | 0.25 [0.14, 0.37] | 5.39 |
| cal_signals_lodo | 0.44 | 0.79 [0.70, 0.89] | 0.59 [0.48, 0.71] | 82.47 |
| cal_full_lodo | 0.69 | 0.89 [0.81, 0.95] | 0.37 [0.25, 0.49] | 34.14 |
| cal_signals_synthetic | 0.57 | 0.24 [0.13, 0.35] | 0.22 [0.13, 0.33] | 30.31 |
| cal_full_synthetic | 0.71 | 0.14 [0.06, 0.24] | 0.06 [0.02, 0.13] | 6.89 |

AUROC by held-out dataset:

| method | electricity | airlines | covertype | bank_marketing |
|---|---:|---:|---:|---:|
| stat_only | 0.47 | 0.77 | 0.46 | 0.57 |
| conf_only | 0.34 | 0.56 | 0.41 | 0.75 |
| fixed_score | 0.70 | 0.66 | 0.45 | 0.97 |
| slope_forecast | 0.59 | 0.59 | 0.52 | 0.83 |
| delayed_label_only | 0.46 | 0.87 | 0.86 | 0.90 |
| cal_signals_lodo | 0.53 | 0.34 | 0.46 | 0.44 |
| cal_full_lodo | 0.44 | 0.80 | 0.64 | 0.88 |
| cal_signals_synthetic | 0.67 | 0.36 | 0.53 | 0.71 |
| cal_full_synthetic | 0.61 | 0.59 | 0.74 | 0.88 |

## Label delay L = 2 (pooled over held-out datasets)

| method | mean AUROC | recall | early-warning rate | FA/100 clean |
|---|---:|---:|---:|---:|
| stat_only | 0.57 | 0.40 [0.27, 0.52] | 0.24 [0.14, 0.35] | 28.63 |
| conf_only | 0.51 | 0.11 [0.05, 0.19] | 0.05 [0.00, 0.11] | 3.67 |
| fixed_score | 0.70 | 0.43 [0.30, 0.56] | 0.29 [0.17, 0.40] | 21.31 |
| slope_forecast | 0.63 | 0.41 [0.30, 0.54] | 0.24 [0.13, 0.35] | 5.85 |
| delayed_label_only | 0.79 | 0.97 [0.92, 1.00] | 0.14 [0.06, 0.24] | 4.75 |
| cal_signals_lodo | 0.44 | 0.79 [0.70, 0.89] | 0.59 [0.46, 0.70] | 82.47 |
| cal_full_lodo | 0.70 | 0.89 [0.81, 0.95] | 0.43 [0.30, 0.56] | 46.49 |
| cal_signals_synthetic | 0.57 | 0.24 [0.14, 0.35] | 0.22 [0.13, 0.33] | 30.31 |
| cal_full_synthetic | 0.71 | 0.24 [0.14, 0.35] | 0.13 [0.05, 0.22] | 10.73 |

AUROC by held-out dataset:

| method | electricity | airlines | covertype | bank_marketing |
|---|---:|---:|---:|---:|
| stat_only | 0.47 | 0.77 | 0.46 | 0.57 |
| conf_only | 0.34 | 0.56 | 0.41 | 0.75 |
| fixed_score | 0.70 | 0.66 | 0.45 | 0.97 |
| slope_forecast | 0.59 | 0.59 | 0.52 | 0.83 |
| delayed_label_only | 0.61 | 0.84 | 0.84 | 0.87 |
| cal_signals_lodo | 0.53 | 0.34 | 0.46 | 0.44 |
| cal_full_lodo | 0.60 | 0.73 | 0.62 | 0.84 |
| cal_signals_synthetic | 0.67 | 0.36 | 0.53 | 0.71 |
| cal_full_synthetic | 0.69 | 0.54 | 0.72 | 0.86 |

## Label delay L = 5 (pooled over held-out datasets)

| method | mean AUROC | recall | early-warning rate | FA/100 clean |
|---|---:|---:|---:|---:|
| stat_only | 0.57 | 0.40 [0.29, 0.52] | 0.24 [0.14, 0.35] | 28.63 |
| conf_only | 0.51 | 0.11 [0.05, 0.19] | 0.05 [0.00, 0.11] | 3.67 |
| fixed_score | 0.70 | 0.43 [0.30, 0.56] | 0.29 [0.17, 0.40] | 21.31 |
| slope_forecast | 0.63 | 0.41 [0.30, 0.54] | 0.24 [0.14, 0.35] | 5.85 |
| delayed_label_only | 0.70 | 0.73 [0.62, 0.84] | 0.11 [0.05, 0.19] | 4.55 |
| cal_signals_lodo | 0.44 | 0.79 [0.68, 0.89] | 0.59 [0.48, 0.71] | 82.47 |
| cal_full_lodo | 0.58 | 0.86 [0.76, 0.94] | 0.56 [0.43, 0.67] | 74.61 |
| cal_signals_synthetic | 0.57 | 0.24 [0.13, 0.35] | 0.22 [0.13, 0.33] | 30.31 |
| cal_full_synthetic | 0.63 | 0.22 [0.13, 0.33] | 0.14 [0.06, 0.24] | 11.05 |

AUROC by held-out dataset:

| method | electricity | airlines | covertype | bank_marketing |
|---|---:|---:|---:|---:|
| stat_only | 0.47 | 0.77 | 0.46 | 0.57 |
| conf_only | 0.34 | 0.56 | 0.41 | 0.75 |
| fixed_score | 0.70 | 0.66 | 0.45 | 0.97 |
| slope_forecast | 0.59 | 0.59 | 0.52 | 0.83 |
| delayed_label_only | 0.61 | 0.72 | 0.74 | 0.75 |
| cal_signals_lodo | 0.53 | 0.34 | 0.46 | 0.44 |
| cal_full_lodo | 0.56 | 0.52 | 0.56 | 0.66 |
| cal_signals_synthetic | 0.67 | 0.36 | 0.53 | 0.71 |
| cal_full_synthetic | 0.68 | 0.42 | 0.67 | 0.76 |