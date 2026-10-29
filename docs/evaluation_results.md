# Held-Out Evaluation Results

## Run

The default experiment completed on 120 held-out episodes: ten test seeds, four scenarios, and label delays of 0, 2, and 5 batches. It used the frozen protocol and classifier in [`final_scope.md`](final_scope.md), selected all alert thresholds on validation seeds, and reset the training-fitted calibrator between episodes. Per-episode ledgers and full aggregates are in `artifacts/early_warning/aggregate_results.json` (generated output, not checked into source control).

The evaluation used 20 confirmed harmful events per delay: ten gradual and ten abrupt. All 20 test episodes in each harmful scenario had a confirmed event. Results below are unchanged across the three delays for event recall and timing.

## Main result

The proposed risk calibrator did **not** satisfy the project's success criterion. It matched 0 of 20 harmful events at every label delay and issued no alerts in the harmful test episodes. The fixed weighted score matched 9 of 20 events (recall 0.45): 8 of 10 abrupt events and 1 of 10 gradual events. Those matches came after event onset; mean detection lag was 1.56 batches, and neither method produced a pre-event warning on the held-out harmful episodes.

| Held-out metric | Risk calibrator | Fixed weighted score |
|---|---:|---:|
| Harmful events matched, per delay | 0 / 20 | 9 / 20 |
| Harmful-event recall | 0.00 | 0.45 |
| Pre-event warnings | 0 | 0 |
| Mean lag among matched events | N/A | 1.56 batches |
| False alerts per 100 stable clean batches in harmless-shift episodes | 0.00 | 12.33 ± 9.20 |

The calibrator suppresses the harmless-shift alerts in this run, but it also fails to alert on harmful changes. That trade-off does not support the early-warning claim. Its threshold was selected only on validation episodes; do not lower it based on this test result.

Per-scenario event recall is mean ± population SD across the ten episodes. The fixed score matched 8/10 abrupt events (0.80 ± 0.40; mean lag 1.13 ± 0.93 batches among the eight detections) and 1/10 gradual events (0.10 ± 0.30; lag 5 batches for the one detection). The risk calibrator matched 0/10 in both. In clean episodes it averaged 1.33 ± 2.21 false alerts per 100 batches at delay 2; across delays the mean ranged from 1.33 to 1.67. The selected validation thresholds met the five-per-100 clean-batch budget; the fixed score exceeded it on the held-out harmless-shift episodes.

The pooled held-out Brier score for harmful episodes was approximately 0.135 at every delay. The harmful-gradual reliability curve at delay 2 falls above the ideal diagonal in its populated nonzero bins, so the probabilities underestimate the observed event frequency in those bins. The reliability plot is at `artifacts/early_warning/reliability_harmful_gradual_delay-2.png`.

## Demonstration and implementation checks

The paired demo completed on held-out seed 40 at delay 2. The harmful-gradual event began at batch 10: the risk calibrator issued zero alerts and the fixed-score baseline issued five alerts, all too late to count as warnings. In the harmless-shift episode there was no confirmed accuracy event: the risk calibrator issued zero alerts and the fixed-score baseline issued seven. The ledgers, summary, and episode plots are under `artifacts/early_warning/demo/`. This is a reproducible demonstration of the current negative result, not a successful warning demonstration.

The scenario generator also completed a separate sweep of 200 episodes (50 seeds by four scenarios) without a generation failure. After removing the unused legacy forecast and integration work from the scoped replay, measured runtime averaged 0.079 seconds per batch across test groups (group means 0.072–0.084 seconds) on this machine. Treat this as an implementation observation, not a throughput benchmark.

## Verification and limits

- `pytest -q --ignore=tests/test_api.py`: **54 passed**, with six statsmodels numerical warnings from the legacy Electricity weight-calibration test.
- The API `TestClient` checks did not complete: a health request hangs inside Starlette/AnyIO's blocking portal, including with a minimal standalone FastAPI app. Trying AnyIO 4.4.0 and restoring 4.15.1 made no difference. Directly calling the health endpoint succeeds. Treat API HTTP-level behavior as unverified in this environment.
- `git diff --check` passed after the review edits.
- No fresh chronological Electricity run was made because no local Electricity input file is present. The checked-in comparison remains a no-confirmed-degradation example and cannot support the early-warning claim.

## Claim-to-evidence summary

| Claim | Status |
|---|---|
| Live features exclude future labels and event outcomes | Supported by the replay path and leakage/maturity tests |
| The replay respects configured label delays and target maturity | Supported by focused tests and per-episode ledgers |
| The calibrator improves held-out harmful-event detection over the fixed score | **Not supported; it detected 0/20 events at all three delays** |
| Input shift alone indicates harmful degradation | Rejected by the harmless-shift episodes and the project's problem definition |
| The project is production-ready or has production throughput | Not evaluated; out of scope |

Keep the negative result in the report. Any follow-up model change needs a new predeclared experiment with untouched seeds; do not tune against these test episodes.
