# Viva Questions and Answers

Short, honest answers. Sources are in brackets.

## About the results

**Q: Does your system actually give early warnings?**
Yes, for drift, with a caveat about harm.
- The drift watch tier warned before the accuracy drop on 87% of harmful gradual drifts in 6 unseen datasets, a median of 6 batches early, within the false-alert budget. [`docs/v7_verdict.md`]
- It also fires on almost every harmless drift, so on its own it means "go check", not "the model is failing".

**Q: So did the project succeed or fail?**
Both, and we state it per hypothesis.
- **v6** (original design on held-out data): FAILURE.
- **v7:** PARTIAL SUCCESS.
- **v8:** PARTIAL SUCCESS.

Each later version fixed a diagnosed cause of the previous failure and was confirmed on datasets never used before.

**Q: Why should we trust a "partial success" you reached after a failure? Isn't that tuning until it works?**
Three safeguards:
1. Each version froze its design, thresholds and pass criteria in code before downloading its confirmation data.
2. The confirmation datasets were new every time: v7 used 6 never-touched datasets, v8 another 5.
3. The verdict is computed by a script that refuses to run if the frozen files changed.

The old held-out data was only used post hoc, and labelled as such.

**Q: What is AUROC 0.78 in plain words?**
If you take a random harmful drift and a random harmless one, the harm tier ranks the harmful one as more dangerous 78% of the time. Drift statistics alone manage 59%, close to a coin flip.

**Q: Why not just wait for the labels?**
- Labels can arrive days or weeks late. In our runs, labels confirmed the drop a median of several batches after our warnings (20 batches in the demo example).
- Waiting is still the gold standard for confirming harm, which is why the system has a third tier where labels confirm or clear a warning.
- In the v6 comparison, waiting for labels was as good as or better than our learned system. We report that too.

**Q: Why was the AUROC target 0.80?**
It is the conventional threshold for "good" discrimination (a common grading calls 0.7–0.8 acceptable and 0.8–0.9 excellent). We fixed it in code before seeing any data and kept it across experiments so results stay comparable. It was not derived from a cost analysis; a target based on how many false alarms an engineer will tolerate would be the better design next time.

**Q: 0.78 is so close to 0.80. Why not count it as a pass?**
- The rule was the point estimate, fixed in advance, so it is a miss.
- The 95% interval is 0.70–0.86 and includes 0.80, so the data cannot tell our result apart from the target.
- We then tried to clear it properly in a new experiment (v9): nonlinear estimators reached only 0.70 on development data, and ensembles of all estimators at most 0.76, both below the simple method's 0.77. So we kept the simple method. [`docs/v9_protocol.md`]

**Q: Why do 34% of harmless drifts still raise a harm warning?**
- Mostly accumulated background false alarms, not misjudged harm. On unseen data, streams with no shift at all raised a harm warning in 40% of episodes and harmless shifts in 34%, against 88% for harmful shifts. The harm tier treats harmless shifts like no shift at all.
- The harm estimate is noisy from batch to batch (about 0.8 accuracy points). Over about 33 monitored batches, a 2.8-per-100 false-alert rate gives about a 60% chance of at least one alert.
- Our 10% target was inconsistent with our own 5-per-100 per-batch budget: it would need about 0.3 per 100. That was a design error on our side.

**Q: Did you try to reduce those alarms?**
Yes, in v10: stricter thresholds, smoothing, CUSUM change detection, stream-normalized scores and gating by the drift watch. The best rule cut harmless alarms to 12% on six fresh datasets, but it then warned before the labels on only 30% of harmful shifts (we required at least 50%), so it was not adopted. Lowering alarms by raising the bar also removes the early warnings the system exists to give. [`docs/v10_protocol.md`]

## About the method

**Q: What is "stream normalization"?**
Each signal is measured in robust standard deviations from that stream's own first 8 batches after deployment. A threshold like "4 standard deviations above normal" then means the same thing on every dataset. Without it, false alerts on unseen data ranged from 1.2 to 44.6 per 100 batches; with it they stayed under 5. [`docs/ablations.md` A]

**Q: How does the harm tier estimate accuracy without labels?**
1. A small classifier learns to tell the labeled reference rows from the current batch.
2. Its odds reweight the reference rows to look like the current batch.
3. The model's known correctness on those reweighted rows estimates the batch's accuracy.

If data moves into regions where the model was often wrong, the estimate drops.

**Q: Why can't you detect concept drift?**
If the inputs look the same but what the labels mean changes, nothing observable without labels changes. That is a limit of any label-free method. The system relies on delayed labels for those cases.

**Q: Why does it fail on natural streams?**
- On the natural streams the median accuracy event started at batch 8, the end of the burn-in.
- Accuracy already varied about 12% within the burn-in.
- So the "normal" baseline was not normal. Normalization needs a stable period after deployment. [`docs/ablations.md` F]

**Q: Why only logistic regression?**
It was the pre-registered primary model. Gradient boosting was run in v6 as a robustness check (similar picture). v7 and v8 confirmations used logistic regression only, to keep the confirmations clean and the runtime manageable. Extending to other models is future work.

**Q: Why are 80% of your shifts harmless? Is the benchmark broken?**
No. We measured it in every data group and checked that shift strength doesn't change it (the Day 2 gate failed at every strength). Most realistic input changes simply don't move data into regions where the model is wrong. That is itself a key finding: drift does not equal degradation.

## About the engineering

**Q: What happened to Kafka, Grafana and the original production stack?**
We removed it. The final scope is the monitoring method and its evaluation. A distributed stack added complexity without contributing evidence.

**Q: Can someone else reproduce your results?**
Yes. `pip install -r requirements-lock.txt` then `python scripts/reproduce.py`. In a clean environment, regenerated episodes match the saved ones to within 4×10⁻¹¹ and all three verdicts reproduce exactly. [`docs/reproducibility.md`]

**Q: Were there any deviations from your plans?**
Yes, all recorded in the protocol documents with dates:
- a class-balance rule clarified;
- block sampling made exhaustive;
- the Day 2 gate failed and you chose to continue;
- MiniBooNE shuffled because its rows were sorted by class;
- one dataset dropped after repeated download failures.

None changed a frozen choice after its data was seen.

## About what's next

**Q: How would you improve it?**
1. Better harm estimation, such as density-ratio models beyond logistic regression, or uncertainty from model ensembles.
2. A burn-in that adapts when the deployment period is itself unstable.
3. Testing on more model types and on real deployment logs with naturally delayed labels.
