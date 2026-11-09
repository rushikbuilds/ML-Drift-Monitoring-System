from pathlib import Path

import numpy as np
import pytest

from dashboard import app_data
from dews import v7_config as v7cfg
from tests.test_v6_eval import _episode

ROOT = Path(__file__).resolve().parents[1]
HAVE_RESULTS = (ROOT / "artifacts" / "v8" / "v8_verdict.json").exists() and any((ROOT / "artifacts" / "v8").glob("signals_v8_confirmation_*.pkl"))


def _with_harm(ep):
    return dict(ep, iw_accuracy=ep["accuracy"].copy(), knn_accuracy=ep["accuracy"].copy(), meta={})


@pytest.mark.skipif(not (ROOT / "artifacts" / "v7" / "frozen_v7.json").exists(), reason="frozen thresholds not present")
def test_replay_statuses_follow_tiers():
    frame = app_data.replay(_with_harm(_episode()), delay=5)
    assert (frame["status"][: v7cfg.BURN_IN_BATCHES] == "calibrating").all()
    assert set(frame["status"]) <= set(app_data.STATUS)
    assert frame["status"].iloc[-1] == "confirmed"  # the toy episode has a sustained drop at batch 20


def test_visible_accuracy_respects_label_delay():
    frame = app_data.replay(_with_harm(_episode()), delay=5) if (ROOT / "artifacts" / "v7" / "frozen_v7.json").exists() else None
    if frame is None:
        pytest.skip("frozen thresholds not present")
    seen = app_data.visible_accuracy(frame, 20)
    assert seen.iloc[15].item() == pytest.approx(frame["true_accuracy"].iloc[15])
    assert np.isnan(seen.iloc[16])


@pytest.mark.skipif(not HAVE_RESULTS, reason="experiment artifacts not present")
def test_all_pages_render_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    for page in ("app_pages/live_monitor.py", "app_pages/verdicts.py", "app_pages/ablations.py"):
        at.switch_page(page).run()
        assert not at.exception, page
