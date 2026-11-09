# Three-Minute Demo Script

**Before the session:**

```bash
.venv\Scripts\activate
streamlit run streamlit_app.py
```

Open http://localhost:8501 and the **Live monitor** page. Set **Data** to *Unseen data (v8 confirmation)* and **Label delay** to 5. All three episodes below are from credit_default, a dataset the system never saw during design.

---

## 1. A harmful drift, caught early (about 75 seconds)

**Select:** Dataset *credit_default* → Episode type *Gradual shift* → Episode *Seed 9624: gradual shift, harmful*.

1. **Drag the batch slider to 14.**
   - *Say:* "Monitoring started 14 batches ago. The status is Healthy."
2. **Drag to 16.** The status turns to **Harm warning**.
   - *Say:* "The input distribution began drifting at batch 13. At batch 16 the harm tier estimates, without any new labels, that the model is moving into regions where it makes more mistakes."
3. **Drag to 30.**
   - *Point at the Accuracy chart:* "Accuracy actually drops here, at batch 30: 14 batches after our warning. The green line is what the delayed labels show, and they still look fine."
4. **Drag to 36.** The status turns to **Degradation confirmed**.
   - *Say:* "Only now could the labels confirm the drop, 20 batches after the warning."
5. **Scroll to "Why the drift watch score is…".**
   - *Say:* "This shows which signals moved, so an engineer knows where to look."

## 2. A harmless drift (about 45 seconds)

**Select:** Episode type *Gradual shift* → Episode *Seed 9614: gradual shift, harmless*.

1. **Drag the slider to the end.**
   - *Say:* "The inputs drift here too: the drift-watch line crosses its threshold at batch 17. But the harm tier stays below its threshold. The accuracy line shows the model was fine."
   - **Honest caveat, say it out loud:** "This is a favourable example. Across all unseen data, 34% of harmless drifts still raise a harm warning. It's a big improvement over 97% without the harm tier, but it's not solved."

## 3. The limitation: label changes (about 30 seconds)

**Select:** Episode type *Label change* → Episode *Seed 9655: label change, accuracy event*.

1. **Drag the slider to the end.**
   - *Say:* "Here the inputs stay the same but what the labels mean changes. No label-free signal can see that. Neither tier fires; only the delayed labels confirm the drop at batch 28. That's a fundamental limit, and our verdicts report it."

## 4. Close on the verdicts (about 15 seconds)

**Open the Verdicts page.**
- *Say:* "Every number you saw comes from one-time runs with hypotheses frozen in advance. v8 is a partial success: it passes 2 of 4 hypotheses, and the harm separation of 0.78 just misses our 0.80 target."

---

**If something goes wrong:** the screenshots in the slides show the same episodes. Every episode is regenerated from its seed, so the numbers are identical on any machine with the saved results.
