# Final locked evaluation: delay-anchored estimator

| held-out stream | windows | with event |
|---|---:|---:|
| covpokelec | 40 | 33 |
| hyperplane | 40 | 14 |
| led | 40 | 0 |
| poker_hand | 40 | 3 |
| sea | 40 | 1 |

## L = 0: Development (leave-one-dataset-out, 8 real streams)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | electricity | airlines | covertype | bank_marketing | run_or_walk | vehicle_sensit | kddcup99 | ldpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_last | 9.64 | 9.8 | 4.4 | 9.5 | 5.7 | 4.9 | 3.1 | 5.9 | 33.8 |
| stale_mean3 | 9.85 | 9.5 | 4.2 | 6.9 | 7.2 | 7.7 | 2.5 | 7.1 | 33.8 |
| cbpe_raw | 17.91 | 21.4 | 10.7 | 9.8 | 16.0 | 26.8 | 2.5 | 20.1 | 36.1 |
| da_cbpe | 10.14 | 11.2 | 4.7 | 5.8 | 7.2 | 7.3 | 2.5 | 8.0 | 34.5 |
| anchored | 9.11 | 9.9 | 4.1 | 5.5 | 5.8 | 5.5 | 2.5 | 6.6 | 33.0 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.70 | 0.37 [0.27, 0.47] | 0.12 | 0.13 | 7.64 |
| delayed_label_only | 0.72 | 1.00 [1.00, 1.00] | 0.29 | 0.29 | 6.72 |
| da_cbpe_p2 | 0.75 | 0.89 [0.82, 0.95] | 0.83 | 0.23 | 5.52 |
| anchored | 0.76 | 0.70 [0.60, 0.79] | 1.66 | 0.07 | 7.12 |
| anchored_p2 | 0.74 | 0.91 [0.85, 0.97] | 1.30 | 0.14 | 6.87 |

## L = 0: Held-out streams (5, unseen during design)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | hyperplane | sea | covpokelec | poker_hand | led |
|---|---:|---:|---:|---:|---:|---:|
| stale_last | 4.66 | 2.1 | 2.4 | 13.6 | 3.5 | 1.8 |
| stale_mean3 | 3.95 | 1.8 | 2.0 | 11.6 | 2.9 | 1.5 |
| cbpe_raw | 4.87 | 2.7 | 1.9 | 15.6 | 2.7 | 1.4 |
| da_cbpe | 3.76 | 1.8 | 2.0 | 10.9 | 2.8 | 1.4 |
| anchored | 3.84 | 2.0 | 2.1 | 10.5 | 3.0 | 1.6 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.60 | 0.02 [0.00, 0.06] | -5.00 | 0.02 | 0.58 |
| delayed_label_only | 0.74 | 1.00 [1.00, 1.00] | -0.80 | 0.53 | 3.07 |
| da_cbpe_p2 | 0.73 | 0.96 [0.90, 1.00] | 0.00 | 0.35 | 4.56 |
| anchored | 0.72 | 0.33 [0.22, 0.47] | 0.76 | 0.08 | 0.21 |
| anchored_p2 | 0.75 | 0.96 [0.90, 1.00] | -0.47 | 0.45 | 3.76 |

## L = 2: Development (leave-one-dataset-out, 8 real streams)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | electricity | airlines | covertype | bank_marketing | run_or_walk | vehicle_sensit | kddcup99 | ldpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_last | 11.72 | 10.8 | 4.8 | 6.8 | 8.0 | 8.3 | 2.8 | 8.2 | 44.1 |
| stale_mean3 | 11.08 | 10.4 | 4.5 | 6.7 | 9.4 | 10.3 | 2.4 | 8.6 | 36.2 |
| cbpe_raw | 17.91 | 21.4 | 10.7 | 9.8 | 16.0 | 26.8 | 2.5 | 20.1 | 36.1 |
| da_cbpe | 11.31 | 12.3 | 5.2 | 5.7 | 9.2 | 9.0 | 2.4 | 9.1 | 37.6 |
| anchored | 10.26 | 11.1 | 4.6 | 5.5 | 7.6 | 7.6 | 2.5 | 7.9 | 35.3 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.70 | 0.37 [0.27, 0.47] | 0.12 | 0.13 | 7.64 |
| delayed_label_only | 0.70 | 1.00 [1.00, 1.00] | 1.00 | 0.20 | 7.67 |
| da_cbpe_p2 | 0.74 | 0.85 [0.77, 0.91] | 0.99 | 0.25 | 5.56 |
| anchored | 0.76 | 0.78 [0.69, 0.86] | 1.85 | 0.08 | 6.26 |
| anchored_p2 | 0.76 | 0.92 [0.86, 0.98] | 1.76 | 0.12 | 4.87 |

## L = 2: Held-out streams (5, unseen during design)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | hyperplane | sea | covpokelec | poker_hand | led |
|---|---:|---:|---:|---:|---:|---:|
| stale_last | 4.62 | 2.1 | 2.4 | 13.3 | 3.4 | 1.8 |
| stale_mean3 | 3.95 | 1.8 | 2.0 | 11.6 | 2.9 | 1.5 |
| cbpe_raw | 4.87 | 2.7 | 1.9 | 15.6 | 2.7 | 1.4 |
| da_cbpe | 3.83 | 1.8 | 2.0 | 11.2 | 2.8 | 1.4 |
| anchored | 3.93 | 2.0 | 2.2 | 10.8 | 3.0 | 1.7 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.60 | 0.02 [0.00, 0.06] | -5.00 | 0.02 | 0.58 |
| delayed_label_only | 0.73 | 1.00 [1.00, 1.00] | -0.47 | 0.43 | 2.50 |
| da_cbpe_p2 | 0.73 | 0.96 [0.90, 1.00] | 0.65 | 0.35 | 4.77 |
| anchored | 0.71 | 0.39 [0.25, 0.53] | 1.60 | 0.06 | 0.39 |
| anchored_p2 | 0.73 | 0.92 [0.84, 0.98] | 0.28 | 0.35 | 2.21 |

## L = 5: Development (leave-one-dataset-out, 8 real streams)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | electricity | airlines | covertype | bank_marketing | run_or_walk | vehicle_sensit | kddcup99 | ldpa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_last | 14.15 | 12.6 | 6.0 | 8.5 | 13.9 | 14.5 | 3.0 | 11.3 | 43.3 |
| stale_mean3 | 13.84 | 11.6 | 5.9 | 7.9 | 15.2 | 15.6 | 2.4 | 11.8 | 40.2 |
| cbpe_raw | 17.91 | 21.4 | 10.7 | 9.8 | 16.0 | 26.8 | 2.5 | 20.1 | 36.1 |
| da_cbpe | 13.57 | 14.4 | 6.9 | 7.0 | 12.7 | 13.5 | 2.4 | 13.1 | 38.5 |
| anchored | 12.05 | 10.1 | 6.3 | 6.4 | 10.3 | 12.1 | 2.5 | 10.1 | 38.7 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.70 | 0.37 [0.27, 0.47] | 0.12 | 0.13 | 7.64 |
| delayed_label_only | 0.68 | 0.79 [0.70, 0.87] | 2.47 | 0.13 | 4.25 |
| da_cbpe_p2 | 0.68 | 0.55 [0.45, 0.65] | 0.54 | 0.20 | 6.07 |
| anchored | 0.68 | 0.56 [0.45, 0.66] | 2.04 | 0.10 | 3.89 |
| anchored_p2 | 0.66 | 0.48 [0.38, 0.59] | 1.68 | 0.08 | 4.07 |

## L = 5: Held-out streams (5, unseen during design)

Nowcast MAE in accuracy points x100 (macro over datasets):

| estimator | macro MAE | hyperplane | sea | covpokelec | poker_hand | led |
|---|---:|---:|---:|---:|---:|---:|
| stale_last | 4.74 | 2.2 | 2.3 | 14.2 | 3.3 | 1.7 |
| stale_mean3 | 4.26 | 2.0 | 1.9 | 13.1 | 2.9 | 1.4 |
| cbpe_raw | 4.87 | 2.7 | 1.9 | 15.6 | 2.7 | 1.4 |
| da_cbpe | 4.11 | 2.0 | 2.0 | 12.4 | 2.8 | 1.3 |
| anchored | 4.18 | 2.1 | 2.1 | 12.1 | 3.0 | 1.6 |

Alerting (thresholds from development data only):

| method | macro AUROC | recall | mean time-to-alert (batches after onset) | early-warning | FA/100 clean |
|---|---:|---:|---:|---:|---:|
| fixed_score | 0.60 | 0.02 [0.00, 0.06] | -5.00 | 0.02 | 0.58 |
| delayed_label_only | 0.64 | 0.90 [0.82, 0.98] | 0.89 | 0.27 | 3.65 |
| da_cbpe_p2 | 0.64 | 0.59 [0.45, 0.73] | 0.50 | 0.20 | 4.27 |
| anchored | 0.63 | 0.20 [0.10, 0.31] | 2.30 | 0.04 | 0.54 |
| anchored_p2 | 0.64 | 0.55 [0.41, 0.67] | 0.71 | 0.22 | 1.65 |