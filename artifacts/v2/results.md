# v2 results (held-out test seeds)

Alert budget: 5.0 false alerts / 100 clean batches, thresholds selected on validation seeds.

Test event prevalence: clean: 2/50, harmless_shift: 1/50, harmful_gradual: 50/50, harmful_abrupt: 50/50, concept_gradual: 50/50

## Label delay L = 0

| method | recall | early-warning rate | FA/100 (no-event eps) | AUROC (state) | AUROC (harmful) | Brier |
|---|---:|---:|---:|---:|---:|---:|
| stat_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.05 | 0.55 | 0.73 | n/a |
| conf_only | 0.44 [0.35, 0.51] | 0.27 [0.20, 0.34] | 6.50 | 0.50 | 0.51 | n/a |
| fixed_score | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 | 0.56 | 0.73 | n/a |
| slope_forecast | 0.33 [0.26, 0.41] | 0.03 [0.01, 0.07] | 5.37 | 0.57 | 0.71 | n/a |
| delayed_label_only | 1.00 [1.00, 1.00] | 0.50 [0.42, 0.58] | 4.76 | 0.92 | 0.91 | n/a |
| cal_signals_static | 0.73 [0.66, 0.80] | 0.42 [0.35, 0.50] | 2.27 | 0.83 | 0.75 | 0.161 |
| cal_full_static | 1.00 [1.00, 1.00] | 0.61 [0.53, 0.69] | 3.89 | 0.95 | 0.92 | 0.080 |
| cal_full_online | 1.00 [1.00, 1.00] | 0.61 [0.53, 0.69] | 3.52 | 0.95 | 0.92 | 0.080 |

Per-scenario recall / early-warning rate / mean lead:

| method | harmful_gradual | harmful_abrupt | concept_gradual | FA/100 clean | FA/100 harmless |
|---|---:|---:|---:|---:|---:|
| stat_only | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.10 |
| conf_only | 0.44 / 0.24 / 3.4 | 0.38 / 0.18 / 2.6 | 0.46 / 0.36 / 3.0 | 6.92 | 6.07 |
| fixed_score | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.05 |
| slope_forecast | 0.00 / 0.00 / n/a | 1.00 / 0.10 / 2.4 | 0.00 / 0.00 / n/a | 0.00 | 10.60 |
| delayed_label_only | 1.00 / 0.72 / 3.2 | 1.00 / 0.16 / 2.5 | 1.00 / 0.62 / 2.8 | 5.06 | 4.98 |
| cal_signals_static | 1.00 / 0.98 / 4.3 | 1.00 / 0.18 / 3.7 | 0.20 / 0.10 / 4.6 | 4.51 | 0.56 |
| cal_full_static | 1.00 / 0.98 / 4.0 | 1.00 / 0.28 / 2.7 | 1.00 / 0.58 / 3.0 | 7.03 | 1.78 |
| cal_full_online | 1.00 / 0.98 / 4.0 | 1.00 / 0.28 / 2.9 | 1.00 / 0.60 / 3.0 | 6.26 | 1.83 |

## Label delay L = 2

| method | recall | early-warning rate | FA/100 (no-event eps) | AUROC (state) | AUROC (harmful) | Brier |
|---|---:|---:|---:|---:|---:|---:|
| stat_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.05 | 0.55 | 0.73 | n/a |
| conf_only | 0.44 [0.36, 0.52] | 0.27 [0.20, 0.34] | 6.50 | 0.50 | 0.51 | n/a |
| fixed_score | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 | 0.56 | 0.73 | n/a |
| slope_forecast | 0.33 [0.26, 0.41] | 0.03 [0.01, 0.07] | 5.37 | 0.57 | 0.71 | n/a |
| delayed_label_only | 1.00 [1.00, 1.00] | 0.42 [0.35, 0.50] | 5.26 | 0.90 | 0.90 | n/a |
| cal_signals_static | 0.73 [0.65, 0.80] | 0.42 [0.34, 0.50] | 2.27 | 0.83 | 0.75 | 0.161 |
| cal_full_static | 1.00 [1.00, 1.00] | 0.56 [0.48, 0.64] | 3.94 | 0.94 | 0.92 | 0.086 |
| cal_full_online | 1.00 [1.00, 1.00] | 0.57 [0.49, 0.65] | 3.91 | 0.94 | 0.91 | 0.086 |

Per-scenario recall / early-warning rate / mean lead:

| method | harmful_gradual | harmful_abrupt | concept_gradual | FA/100 clean | FA/100 harmless |
|---|---:|---:|---:|---:|---:|
| stat_only | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.10 |
| conf_only | 0.44 / 0.24 / 3.4 | 0.38 / 0.18 / 2.6 | 0.46 / 0.36 / 3.0 | 6.92 | 6.07 |
| fixed_score | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.05 |
| slope_forecast | 0.00 / 0.00 / n/a | 1.00 / 0.10 / 2.4 | 0.00 / 0.00 / n/a | 0.00 | 10.60 |
| delayed_label_only | 1.00 / 0.62 / 3.2 | 1.00 / 0.16 / 3.4 | 1.00 / 0.50 / 2.4 | 5.52 | 5.40 |
| cal_signals_static | 1.00 / 0.98 / 4.3 | 1.00 / 0.18 / 3.7 | 0.20 / 0.10 / 4.6 | 4.51 | 0.56 |
| cal_full_static | 1.00 / 1.00 / 3.8 | 1.00 / 0.24 / 2.5 | 1.00 / 0.48 / 2.5 | 6.00 | 2.24 |
| cal_full_online | 1.00 / 1.00 / 3.8 | 1.00 / 0.26 / 2.7 | 1.00 / 0.48 / 2.5 | 6.03 | 2.24 |

## Label delay L = 5

| method | recall | early-warning rate | FA/100 (no-event eps) | AUROC (state) | AUROC (harmful) | Brier |
|---|---:|---:|---:|---:|---:|---:|
| stat_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.05 | 0.55 | 0.73 | n/a |
| conf_only | 0.44 [0.36, 0.52] | 0.27 [0.20, 0.34] | 6.50 | 0.50 | 0.51 | n/a |
| fixed_score | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 | 0.56 | 0.73 | n/a |
| slope_forecast | 0.33 [0.26, 0.41] | 0.03 [0.01, 0.07] | 5.37 | 0.57 | 0.71 | n/a |
| delayed_label_only | 0.92 [0.87, 0.95] | 0.31 [0.24, 0.38] | 5.76 | 0.86 | 0.87 | n/a |
| cal_signals_static | 0.73 [0.65, 0.80] | 0.42 [0.34, 0.49] | 2.27 | 0.83 | 0.75 | 0.161 |
| cal_full_static | 0.97 [0.95, 0.99] | 0.44 [0.36, 0.52] | 3.73 | 0.92 | 0.89 | 0.102 |
| cal_full_online | 0.97 [0.95, 0.99] | 0.44 [0.35, 0.52] | 3.62 | 0.92 | 0.89 | 0.101 |

Per-scenario recall / early-warning rate / mean lead:

| method | harmful_gradual | harmful_abrupt | concept_gradual | FA/100 clean | FA/100 harmless |
|---|---:|---:|---:|---:|---:|
| stat_only | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.10 |
| conf_only | 0.44 / 0.24 / 3.4 | 0.38 / 0.18 / 2.6 | 0.46 / 0.36 / 3.0 | 6.92 | 6.07 |
| fixed_score | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 / 0.00 / n/a | 0.00 | 0.05 |
| slope_forecast | 0.00 / 0.00 / n/a | 1.00 / 0.10 / 2.4 | 0.00 / 0.00 / n/a | 0.00 | 10.60 |
| delayed_label_only | 0.98 / 0.56 / 3.2 | 0.80 / 0.10 / 2.8 | 0.98 / 0.26 / 3.1 | 5.93 | 5.96 |
| cal_signals_static | 1.00 / 0.98 / 4.3 | 1.00 / 0.18 / 3.7 | 0.20 / 0.10 / 4.6 | 4.51 | 0.56 |
| cal_full_static | 1.00 / 0.90 / 4.0 | 1.00 / 0.20 / 2.6 | 0.96 / 0.22 / 3.2 | 6.31 | 1.55 |
| cal_full_online | 1.00 / 0.90 / 4.1 | 1.00 / 0.20 / 2.6 | 0.96 / 0.22 / 3.2 | 6.01 | 1.66 |
