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
