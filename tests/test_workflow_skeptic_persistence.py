# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-iteration skeptic-review persistence: a skeptic veto must be auditable
from the SAVED iteration artifacts, not only from the final workflow result.

Covers ``save_signal_results(..., skeptic_reviews=...)`` (additive payload key,
legacy key-set guarantee when ``save_evolution`` is False, copy semantics) and
the workflow loop passing the per-iteration gate reviews to both per-iteration
save paths. Mirrors the fake-LLM harness from ``tests/test_workflow_advanced.py``
(no live LLM, no NAT runtime)."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import signal_discovery_workflow.signal_discovery_optimization_workflow as workflow
from signal_discovery_workflow.signal_code_generator import _python_function_name
from signal_discovery_workflow.signal_discovery_optimization_workflow import (
    _build_lineage,
    save_signal_results,
)

LEGACY_SAVE_KEYS = {
    "timestamp",
    "iteration",
    "selected_signal",
    "signal_description",
    "signal_code",
    "evaluation_metrics",
}


# ---------------------------------------------------------------------------
# Fakes and harness (mirrored from tests/test_workflow_advanced.py)
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Fake ainvoke: replies consumed in call order (str or Exception)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        index = self.calls
        self.calls += 1
        reply = self.replies[index] if index < len(self.replies) else self.replies[-1]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def _signal(name, formula="Rank(TS_Return(Close, 10))"):
    return {
        "name": name,
        "formula": formula,
        "meaning": f"{name} signal",
        "category": "momentum",
        "data_fields_used": ["Close"],
        "operators_used": ["Rank", "TS_Return"],
        "lookback_periods": [10],
    }


def _signal_reply(*signals):
    return "```json\n" + json.dumps(list(signals)) + "\n```"


def _config(**overrides):
    defaults = dict(
        signal_generator_llm="signal",
        code_generator_llm="codegen",
        optimization_advisor_llm="advisor",
        ic_threshold=0.02,
        p_value_threshold=0.05,
        max_iterations=1,
        num_signals=2,
        forward_periods=5,
        save_results=True,
    )
    defaults.update(overrides)
    return workflow.SignalOptimizerConfig(**defaults)


async def _fake_codegen(llm, signal_json, operators, errors_out=None):
    """Build a self-contained module with one function per candidate name."""
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', signal_json)
    blocks = ["import pandas as pd", "import numpy as np", ""]
    for index, name in enumerate(names):
        blocks.append(f"def {_python_function_name(name, index)}(Close):")
        blocks.append("    return Close.pct_change(5).rank(axis=1)")
    return "\n".join(blocks)


def _stock_data(rows=30, cols=5):
    index = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.DataFrame(
        np.arange(float(rows * cols)).reshape(rows, cols), index=index, columns=[f"S{i}" for i in range(cols)]
    )
    return {"Close": close}


def _setup_workflow(monkeypatch, tmp_path, config, llms, ic_results=None):
    """Monkeypatch the evaluation stack; return (run_optimization, state)."""
    data = _stock_data()
    monkeypatch.setattr(workflow, "load_stock_data", lambda: data)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "generate_signal_code", _fake_codegen)

    state = SimpleNamespace(evaluated_codes=[], mean_ics=[])
    ic_queue = [dict(item) for item in (ic_results or [])]

    def fake_execute(code, stock_data, selection_periods=5):
        state.evaluated_codes.append(code)
        return data["Close"], "signal_alpha"

    def fake_rank_ic(signal_values, forward_returns):
        result = (
            dict(ic_queue.pop(0))
            if ic_queue
            else {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20, "ic_std": 0.01, "positive_ic_ratio": 0.6}
        )
        state.mean_ics.append(result.get("mean_ic"))
        return result

    monkeypatch.setattr(workflow, "execute_signal_code", fake_execute)
    monkeypatch.setattr(workflow, "compute_forward_returns", lambda close, periods=5: close.copy())
    monkeypatch.setattr(workflow, "compute_rank_ic", fake_rank_ic)

    class _Builder:
        async def get_llm(self, llm_name, wrapper_type=None):
            if llm_name not in llms:
                raise KeyError(f"no fake llm configured for '{llm_name}'")
            return llms[llm_name]

    async def _drive():
        async with workflow.signal_optimizer_function(config, _Builder()) as info:
            return info.single_fn

    return asyncio.run(_drive()), state


def _run(single_fn, request="momentum signals"):
    return json.loads(asyncio.run(single_fn(request)))


# ---------------------------------------------------------------------------
# save_signal_results: skeptic_reviews payload contract
# ---------------------------------------------------------------------------


def _review(name="Leaky Future", verdict="reject", vetoed=True, reasons=None):
    return {
        "name": name,
        "verdict": verdict,
        "issues": [],
        "static_risks": reasons or [],
        "vetoed": vetoed,
        "reasons": reasons or [],
    }


@pytest.fixture
def output_dir(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setattr(workflow, "OUTPUT_DIR", out)
    return out


class TestSaveSignalResultsSkepticReviews:
    def test_reviews_persisted_alongside_evolution_keys(self, output_dir):
        reviews = [_review(reasons=["lookahead: shift(-5) pulls future values"])]
        path = save_signal_results(
            '[{"name": "Leaky Future"}]',
            "code",
            {"mean_ic": 0.05, "p_value": 0.01, "selected_signal": "signal_leaky"},
            1,
            run_id="runabc123",
            lineage=_build_lineage(1, "runabc123", None, None, False),
            request="momentum signals",
            save_evolution=True,
            skeptic_reviews=reviews,
        )
        data = json.loads(Path(path).read_text())
        assert data["skeptic_reviews"] == reviews
        # The other new-format keys are still present.
        assert data["run_id"] == "runabc123"
        assert data["lineage"]["source"] == "seed"
        assert data["request"] == "momentum signals"

    def test_save_evolution_false_keeps_exactly_legacy_keys_even_with_reviews(self, output_dir):
        path = save_signal_results(
            '[{"name": "M"}]',
            "code",
            {"mean_ic": 0.05, "selected_signal": "signal_m"},
            2,
            run_id="runabc123",
            lineage={"parent_signal": None, "source": "seed"},
            request="momentum signals",
            save_evolution=False,
            run_id_for_filename="runabc123",
            skeptic_reviews=[_review()],
        )
        data = json.loads(Path(path).read_text())
        assert set(data) == LEGACY_SAVE_KEYS  # exact set: no skeptic_reviews key

    @pytest.mark.parametrize("empty", [None, []])
    def test_empty_or_none_reviews_add_no_key(self, output_dir, empty):
        path = save_signal_results(
            "[]",
            "code",
            {"mean_ic": 0.05, "selected_signal": "signal_m"},
            3,
            run_id="runabc123",
            lineage=_build_lineage(3, "runabc123", None, None, False),
            request="r",
            save_evolution=True,
            skeptic_reviews=empty,
        )
        data = json.loads(Path(path).read_text())
        assert "skeptic_reviews" not in data

    def test_saved_reviews_unaffected_by_caller_mutation(self, output_dir):
        reviews = [_review(reasons=["lookahead: shift(-5)"])]
        path = save_signal_results(
            "[]",
            "code",
            {"mean_ic": 0.05, "selected_signal": "signal_m"},
            1,
            run_id="runabc123",
            lineage=_build_lineage(1, "runabc123", None, None, False),
            request="r",
            save_evolution=True,
            skeptic_reviews=reviews,
        )
        before = json.loads(Path(path).read_text())["skeptic_reviews"]
        # Mutate the caller's list structurally and per-dict after the call.
        reviews.append(_review(name="Second", verdict="approve", vetoed=False))
        reviews[0]["vetoed"] = False
        reviews[0]["verdict"] = "approve"
        reviews.clear()
        after = json.loads(Path(path).read_text())["skeptic_reviews"]
        assert after == before
        assert after[0]["verdict"] == "reject"
        assert after[0]["vetoed"] is True


# ---------------------------------------------------------------------------
# Workflow level: vetoes recoverable from the saved per-iteration artifacts
# ---------------------------------------------------------------------------


class TestWorkflowSkepticPersistence:
    def test_vetoed_candidate_is_auditable_from_iteration_artifact(
        self, monkeypatch, tmp_path
    ):
        config = _config(max_iterations=2, num_signals=2, skeptic_review=True)
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(
                    _signal("Leaky Future", "Rank(Close.shift(-5))"),
                    _signal("Weak Clean", "Rank(TS_Return(Close, 10))"),
                ),
                _signal_reply(_signal("Strong Clean", "Rank(TS_Return(Close, 12))")),
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
            # Iteration 1 evaluates too weak to accept, so the run continues
            # into iteration 2 and iteration 2's result becomes the winner.
            ic_results=[{"mean_ic": 0.001, "p_value": 0.4, "num_periods": 20}],
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        assert result["iteration"] == 2
        # Iteration 1 was backtested (the clean survivor) and saved per-iteration.
        assert len(state.evaluated_codes) == 2
        output_dir = tmp_path / "output"
        iter1_files = sorted(output_dir.glob("signal_*_iter1.json"))
        assert len(iter1_files) == 1
        saved = json.loads(iter1_files[0].read_text())

        reviews = saved["skeptic_reviews"]
        leaky = [review for review in reviews if review["name"] == "Leaky Future"]
        assert len(leaky) == 1
        assert leaky[0]["verdict"] == "reject"
        assert leaky[0]["vetoed"] is True
        assert any(reason.startswith("lookahead:") for reason in leaky[0]["reasons"])
        # The surviving candidate's approve verdict is persisted too.
        weak = [review for review in reviews if review["name"] == "Weak Clean"]
        assert len(weak) == 1
        assert weak[0]["verdict"] == "approve"
        assert weak[0]["vetoed"] is False
        # The vetoed candidate is NOT part of the persisted (filtered) signal set.
        assert "Leaky Future" not in saved["signal_description"]
        assert "signal_leaky_future" not in saved["signal_code"]
        assert "Weak Clean" in saved["signal_description"]

    def test_all_vetoed_iteration_persists_reviews_with_error_payload(
        self, monkeypatch, tmp_path
    ):
        config = _config(max_iterations=2, num_signals=2, skeptic_review=True)
        leaky_reply = _signal_reply(
            _signal("Bad One", "Rank(Close.shift(-3))"),
            _signal("Bad Two", "Sub(Close.pct_change(-1), 0)"),
        )
        signal_llm = _ScriptedLLM([leaky_reply, leaky_reply])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        assert result["status"] == "failed"
        assert state.evaluated_codes == []  # the backtest never ran
        output_dir = tmp_path / "output"
        iter1_files = sorted(output_dir.glob("signal_*_iter1.json"))
        assert len(iter1_files) == 1
        saved = json.loads(iter1_files[0].read_text())

        assert saved["evaluation_metrics"]["error"] == "skeptic review rejected all candidates"
        assert saved["evaluation_metrics"]["mean_ic"] is None
        reviews = saved["skeptic_reviews"]
        assert {review["name"] for review in reviews} == {"Bad One", "Bad Two"}
        assert all(review["vetoed"] is True for review in reviews)
        assert all(review["verdict"] == "reject" for review in reviews)
        assert any(
            reason.startswith("lookahead:")
            for review in reviews
            for reason in review["reasons"]
        )
        # Iteration 2 was fully vetoed as well and is auditable the same way.
        iter2_files = sorted(output_dir.glob("signal_*_iter2.json"))
        assert len(iter2_files) == 1
        assert "skeptic_reviews" in json.loads(iter2_files[0].read_text())

    def test_skeptic_off_leaves_saved_payload_without_reviews_key(
        self, monkeypatch, tmp_path
    ):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        saved = json.loads(Path(result["saved_path"]).read_text())
        assert "skeptic_reviews" not in saved
