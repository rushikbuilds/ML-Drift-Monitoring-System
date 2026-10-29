# Final Project Scope

## Project title

**Label-Delay-Aware Early Warning for ML Performance Degradation**

## Research question

For a fixed binary tabular classifier, can recent input-shift and prediction-confidence signals estimate the probability of a sustained accuracy drop within the next five batches, before the labels for recent predictions have arrived? How does delayed feedback affect warning lead time and false alarms?

This is the project's final scope. The earlier DEWS code is a starting prototype. The protocol, scenario replay, baseline scoring, and calibrator are implemented and the default held-out experiment has run. Its results do not support an early-warning improvement claim; see the [evaluation report](evaluation_results.md). A new model should be evaluated only with a predeclared protocol and untouched seeds.

## Problem boundary

The project monitors one reference-trained classifier. It does not retrain or replace that model. At each batch it can observe input features, model outputs, and monitoring signals. Ground-truth labels are hidden until a configurable delay expires.

Use the same fixed classifier specification for the controlled and chronological tabular runs: `StandardScaler` followed by `LogisticRegression(max_iter=500, random_state=42)`. Fit preprocessing and the classifier on reference/training rows only.

An input distribution change is not the same as a performance failure. A shift can be harmless, and a change in the relationship between inputs and targets can be invisible to an input-only monitor. The project will measure these cases separately and will not present an alert as proof that the model has degraded.

## Main contribution

Implement a small **delayed-feedback performance-risk calibrator**. Given recent statistical feature-shift signals, prediction-confidence summaries, and their short-term trends, it estimates:

> `P(sustained accuracy drop within the next 5 batches | information available now)`

Use a lightweight probability model, such as calibrated logistic regression. It may update only when the outcomes used for its update have matured. For every prediction at time `t`, the predictor must not use labels or other information from after `t` except when constructing the target during offline evaluation.

For a risk example at batch `t`, the target is whether an event starts within the next `H = 5` batches. An event is confirmed by `K = 2` consecutive low-accuracy batches, while its onset is recorded as the first batch in that pair. The target can only mature after the horizon closes, the latest possible event has been confirmed, and its labels have arrived: `t + H + (K - 1) + L` (here, `t + 6 + L`). The online calibrator may train on it only at or after that batch. This delayed target maturity is separate from the arrival time of each batch's labels (`batch + L`).

The contribution is the concrete risk-estimation and delayed-feedback evaluation for this project. Do not claim the broader ideas of uncertainty monitoring, drift detection, or model-performance estimation are new. Prediction-uncertainty-based early drift detection has prior work ([PU-index/PUDD, AAAI 2025](https://arxiv.org/abs/2412.11158)); confidence-based and direct-loss performance estimation are also established ([NannyML documentation](https://nannyml.readthedocs.io/en/v0.10.3/how_it_works/performance_estimation.html)). Complete a focused related-work comparison before making any “first” or field-wide novelty claim.

## Degradation event definition

Use the same outcome definition in calibration, validation, and final evaluation:

- Establish baseline accuracy on five clean batches before the controlled intervention.
- A harmful event begins at the first of two consecutive batches whose accuracy is at least five percentage points below that baseline.
- The predictor's horizon is the next five batches.
- Simulate labels arriving after `L = 0`, `2`, or `5` batches. A label can affect a risk prediction only after it has matured.

These values are the initial protocol and must be fixed before the held-out evaluation. If a dataset cannot supply enough clean and post-intervention batches, report that limitation instead of quietly changing the event definition.

## Data and scenarios

Use two complementary settings:

1. **Controlled synthetic episodes — primary evaluation.** Generate repeatable clean, harmless-shift, gradual-harmful, and abrupt-harmful episodes with known onset and outcome labels. Change one mechanism at a time where possible. Include a harmless input shift and a harmful shift that causes the fixed classifier to cross the event definition.
2. **Chronological Electricity stream — reality check.** Preserve time order and report observed performance. The currently checked-in split does not meet the primary degradation criterion; treat it as a no-confirmed-degradation example unless a new, predeclared split does meet the criterion. Optional controlled interventions on the real features must be labeled as injected, not natural drift.

Use fixed, disjoint episode seed groups: seeds `0–29` for training, `30–39` for validation, and `40–49` for held-out testing. Every scenario generated from a given seed uses that seed's same split. This gives ten held-out test episodes per scenario. Do not split batches from one episode across partitions or update the monitored classifier during evaluation.

## Signals and baselines

### Main monitor inputs

- Statistical input-shift score(s), including the existing feature-level statistics where useful.
- Prediction confidence summaries and confidence-distribution change.
- Recent rolling means and slopes of these signals.

Do not treat the existing scikit-learn `embedding_features` output as learned embeddings: its adapter returns raw input columns. A genuine model embedding may be included only as a separately named ablation when the model actually exposes one.

### Required comparisons

- Statistical-only monitor.
- Confidence-only monitor.
- Existing fixed weighted drift score and threshold.
- Simple recent-slope forecast of the fixed score.
- Proposed delayed-feedback risk calibrator.

Keep the classifier, data split, batch boundaries, event labels, and alert budget matched across methods. One optional performance-estimation package may be added only if its assumptions fit the chosen model and dataset. Do not turn the study into a broad vendor comparison.

## Evaluation protocol

Fit the initial risk calibrator on training episodes and select thresholds on validation episodes to target no more than **five alerts per 100 clean batches**. Before final testing, freeze the model specification, initial fit, online-update rule, and thresholds. During held-out replay, the predeclared online rule may incorporate that episode's labels only after their protocol maturity time; it must not tune thresholds or update from another held-out episode. Reset to the same training-fitted calibrator for each episode and repeat with the defined label delays.

Report:

- Harmful-event recall at the selected false-alert budget.
- False alerts per 100 clean batches and separately warnings during harmless shift.
- Warning lead when an alert precedes the event; detection lag when it follows; missed events when no alert matches.
- Event-level precision/recall and calibration quality (Brier score plus a reliability plot).
- Runtime per batch as a secondary implementation measure.

This method predicts event risk within a fixed horizon; it does not output a point estimate of event time. Report warning lead and detection lag instead of time-to-event prediction error.

Report averages and variability across held-out episodes/seeds. If an episode has no measured harmful event, degradation lead time is not applicable; do not record it as zero false alarms or as a successful forecast. Recent work emphasizes controlled drift interventions, timing-aware criteria, and held-out parameter selection, so the evaluation protocol alone is not the project's novelty claim ([KDD 2026 benchmark framework](https://arxiv.org/abs/2606.07789)).

### Success and fallback criteria

The proposed method is supported only if it improves harmful-event recall or warning lead over the fixed-score baseline at the same held-out false-alert budget, across multiple episodes. If it does not, report the negative result plainly and keep the monitor as a baseline prototype; do not tune on the test set or imply that a drift alarm predicts failure.

## Demonstration requirements

The final demonstration should take under five minutes and show:

1. A clean, harmless-shift, or harmful-gradual scenario.
2. A frozen model processing batches while the monitor displays its signals and estimated risk.
3. Labels arriving only after the selected delay.
4. The measured accuracy event, its onset, the risk estimate, alert threshold, and warning timing on one timeline.
5. A matched comparison between the fixed-score baseline and the proposed method.

The demo must include a harmful episode and a harmless-shift case. Show the measured outcome as observed, including missed warnings; do not manufacture a successful warning. Clearly label every injected intervention. Do not present retraining or automated remediation as part of the final system.

## Out of scope

- Kafka, Celery, Redis, distributed scaling, and production throughput.
- PostgreSQL/TimescaleDB, MLflow, migrations, registries, and production failover.
- Automated retraining, webhooks, ticket creation, and closed-loop deployment.
- Multiple dashboard stacks or a general-purpose monitoring product.
- Broad claims about detecting arbitrary concept drift or guaranteed model failure.

Keep existing code that helps deliver the scoped experiment. Leave unrelated infrastructure dormant unless it blocks the local demo; do not spend the project schedule deleting it for its own sake.

## Implementation order

1. Correct event and warning metrics; make “no event” explicitly not applicable.
2. Separate clean, harmless, gradual-harmful, and abrupt-harmful scenario generation, with fixed seeds and known event times.
3. Record one consistent per-batch data format, including signal history, predictions, delayed label availability, and outcome events.
4. Implement matched baselines and the delayed-feedback calibrator. Check for information leakage.
5. Run the seed/episode and label-delay evaluation; freeze results and document limitations.
6. Provide one reliable local demo and rewrite the README/report around this scope.

The detailed file-by-file sequence, data-flow rules, phase gates, and run outputs are in the [implementation plan](implementation_plan.md).

## Should implementation start from scratch?

**No.** Reuse the tabular model adapter, statistical/confidence detector components, batch-processing loop, SQLite storage, and generated data as scaffolding. Rework the scenario generator and evaluation protocol; remove the unhelpful parts from the supported workflow. The current prototype does not need a wholesale rewrite, but the research experiment should be built as a clean, small path through it rather than carrying the production-stack claims forward.
