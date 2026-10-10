"""scripts/push_eval.py: the checks made before a push (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import push_eval  # noqa: E402

RTX = {"metadata": {"kaggle": {"accelerator": "nvidiaRtxPro6000"}}}


def test_a_built_notebook_on_the_rtx_with_a_short_title_passes():
    assert push_eval.preflight({"title": "arc3 dprime r14a05 harness4 percept draft full"}, RTX) == ""
    assert push_eval.preflight({"title": "x" * push_eval.MAX_TITLE}, RTX) == ""


def test_a_title_over_50_characters_is_refused_before_kaggle_answers_400():
    problem = push_eval.preflight({"title": "arc3 dprime r14a05 harness4 percept draft untried full"}, RTX)
    assert "54 characters" in problem and "--slug" in problem


def test_a_notebook_off_the_rtx_is_refused():
    assert "accelerator" in push_eval.preflight({"title": "ok title"}, {"metadata": {"kaggle": {"accelerator": "gpu"}}})
    assert "accelerator" in push_eval.preflight({"title": "ok title"}, {})
