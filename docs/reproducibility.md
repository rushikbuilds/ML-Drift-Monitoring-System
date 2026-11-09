# Reproducibility

## Environment

| File | Purpose |
|---|---|
| `requirements.txt` | Core dependencies (experiments, tests, dashboard), with supported version ranges |
| `requirements-lock.txt` | Exact versions verified on 2026-11-09 to reproduce the v6, v7 and v8 verdicts |
| `requirements-legacy.txt` | Optional legacy components (FastAPI service, MLflow, PostgreSQL). Use a separate environment: FastAPI 0.111 needs an older Starlette than Streamlit does |

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Environment history (what produced which result)

| Period | Results | NumPy | SciPy | pandas | scikit-learn |
|---|---|---|---|---|---|
| until 2026-11-01 | v1–v5 | 1.26.4 | 1.13.1 | 2.2.2 | 1.7.2 |
| from 2026-11-03 | **v6, v7, v8 (frozen verdicts)** | 2.5.4 | 1.13.1 | 2.2.2 | 1.7.2 |

NumPy was upgraded to 2.5.4 in the global Python environment on 2026-11-03, outside this project. SciPy 1.13.1 does not officially support NumPy 2.5 and warns about it. All frozen verdicts were produced in that environment.

## Verification in a clean, supported environment (2026-11-09)

In a fresh `.venv` with `requirements-lock.txt` (NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, scikit-learn 1.7.2):

- **Unit tests:** `pytest -q --ignore=tests/test_api.py` gives 84 passed.
- **Episode regeneration:** 10 v8 confirmation episodes were rebuilt from their seeds and compared with the saved ones. Outcome categories and event onsets are identical. The largest numerical difference in any signal is 4×10⁻¹¹ (accuracies are exactly equal).
- **Verdicts:** `scripts/make_verdict.py`, `scripts/run_v7.py confirm` and `scripts/run_v8.py confirm` reproduce the v6, v7 and v8 verdicts exactly. No committed verdict file changed.

**Saved data files need NumPy 2.** The `.pkl` files under `artifacts/` were written with NumPy 2 and cannot be loaded with NumPy 1.x. The old `numpy==1.26.4` pin was therefore removed.

## One command

```bash
python scripts/reproduce.py --dry-run   # plan, and what already exists
python scripts/reproduce.py             # verify mode
```

**Verify mode:**
1. Keeps the committed frozen thresholds and calibrators.
2. Downloads and rebuilds anything missing, running extraction as 6 parallel shards.
3. Re-runs every confirmation and the ablations.
4. Fails if any verdict differs from the state before the run.

`--refreeze` recomputes the development choices as well, which overwrites the frozen files; the confirmation scripts then refuse to run until the new fingerprints are recorded in the protocols. That is intentional.

**Runtime from scratch on a 16-core laptop:** about 2 hours, dominated by signal extraction (gradient-boosting episodes are slowest).

**Prerequisite for H5 only:** the natural-stream window lists come from the v2–v4 caches (`artifacts/v2/real_aux_signals.pkl`, `new_signals.pkl`). On a new machine, run the v2–v4 scripts listed in the README first.

## Determinism

- Every episode is generated from a recorded seed (`dews/v6_config.py`, `dews/v7_config.py`, `dews/v8_config.py`).
- All models use fixed `random_state`.
- Bootstrap intervals use fixed generator seeds.
- The only manual data step, shuffling MiniBooNE (rows stored sorted by class), is now part of `scripts/run_v7.py download`, with the same seed and permutation that was used and recorded.
