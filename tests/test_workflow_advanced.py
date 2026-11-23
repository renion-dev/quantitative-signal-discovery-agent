# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Part-2 workflow features: skeptic gate, genetic
crossover, and out-of-sample validation — all driven with fake LLMs and a
monkeypatched evaluation stack (no live LLM, no NAT runtime)."""

import asyncio
import json
import logging
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import signal_discovery_workflow.quant_metrics as quant_metrics
import signal_discovery_workflow.signal_discovery_optimization_workflow as workflow
from signal_discovery_workflow.signal_code_generator import _python_function_name

LEGACY_PAYLOAD_KEYS = (
    "status",
    "headline",
    "request",
    "iteration",
    "total_iterations",
    "selected_signal",
    "thresholds",
    "metrics",
    "signals",
    "saved_path",
)


# ---------------------------------------------------------------------------
# Fakes and harness
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


def _crossover_reply():
    crossover = _signal("CrossBred", "Div(TS_Return(Close, 10), TS_Std(Close, 10))")
    return "Here is the bred signal:\n```json\n" + json.dumps(crossover) + "\n```"


def _verdict_reply(verdict, issues=None, rationale="because"):
    return (
        "```json\n"
        + json.dumps({"verdict": verdict, "issues": issues or [], "rationale": rationale})
        + "\n```"
    )


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


def _setup_workflow(monkeypatch, tmp_path, config, llms, ic_results=None):
    """Monkeypatch the evaluation stack; return (run_optimization, state)."""
    data = _stock_data()
    monkeypatch.setattr(workflow, "load_stock_data", lambda: data)
    monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(workflow, "generate_signal_code", _fake_codegen)

    state = SimpleNamespace(evaluated_codes=[], mean_ics=[], cross_validate_calls=[])
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
        # register_function exposes the workflow as an asynccontextmanager
        # whose entered value is the FunctionInfo wrapping run_optimization.
        async with workflow.signal_optimizer_function(config, _Builder()) as info:
            return info.single_fn

    return asyncio.run(_drive()), state


def _stock_data(rows=30, cols=5):
    index = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.DataFrame(
        np.arange(float(rows * cols)).reshape(rows, cols), index=index, columns=[f"S{i}" for i in range(cols)]
    )
    return {"Close": close}


def _run(single_fn, request="momentum signals"):
    return json.loads(asyncio.run(single_fn(request)))


# ---------------------------------------------------------------------------
# 2A: skeptic gate
# ---------------------------------------------------------------------------


class TestSkepticGate:
    def test_blocking_candidate_vetoed_clean_candidate_survives(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=2, skeptic_review=True)
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(
                    _signal("Momentum Clean", "Rank(TS_Return(Close, 10))"),
                    _signal("Leaky Future", "Rank(Close.shift(-5))"),
                )
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        # The vetoed candidate was never backtested: the evaluated module
        # contains only the survivor's generated function.
        assert len(state.evaluated_codes) == 1
        assert "signal_leaky_future" not in state.evaluated_codes[0]
        assert "signal_momentum_clean" in state.evaluated_codes[0]

        assert result["skeptic"]["reviewed"] == 2
        assert result["skeptic"]["vetoed"] == 1
        rejection = result["skeptic"]["rejections"][0]
        assert rejection["name"] == "Leaky Future"
        assert rejection["reasons"] and rejection["reasons"][0].startswith("lookahead:")
        summary = {item["name"]: item for item in result["signals"]}
        assert "Leaky Future" not in summary
        assert summary["Momentum Clean"]["skeptic"]["verdict"] == "approve"
        assert summary["Momentum Clean"]["skeptic"]["static_risks"] == []

    def test_all_candidates_vetoed_does_not_crash_and_feeds_feedback(self, monkeypatch, tmp_path):
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
        assert "rejected by the skeptic review" in result["last_feedback"]
        assert "Bad One" in result["last_feedback"]
        assert result["skeptic"]["vetoed"] == 2

    def test_skeptic_off_leaves_payload_unchanged(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        assert "skeptic" not in result
        assert all("skeptic" not in item for item in result["signals"])
        assert len(state.evaluated_codes) == 1

    def test_llm_verdict_reject_vetoes_and_approve_keeps(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=2, skeptic_review=True, skeptic_llm="skeptic")
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(
                    _signal("Approved One", "Rank(TS_Return(Close, 10))"),
                    _signal("Rejected Two", "Rank(TS_Return(Close, 20))"),
                )
            ]
        )
        skeptic_llm = _ScriptedLLM(
            [_verdict_reply("approve", rationale="clean"), _verdict_reply("reject", issues=["too similar"])]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {
                "signal": signal_llm,
                "codegen": _ScriptedLLM([]),
                "advisor": _ScriptedLLM(["- try x"]),
                "skeptic": skeptic_llm,
            },
        )

        result = _run(single_fn)

        assert skeptic_llm.calls == 2  # every candidate was reviewed by the LLM
        assert result["status"] == "accepted"
        assert result["skeptic"]["vetoed"] == 1
        rejection = result["skeptic"]["rejections"][0]
        assert rejection["name"] == "Rejected Two"
        assert rejection["reasons"] == ["too similar"]
        assert "signal_rejected_two" not in state.evaluated_codes[0]

    def test_skeptic_llm_failure_fails_open(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1, skeptic_review=True, skeptic_llm="skeptic")
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        skeptic_llm = _ScriptedLLM([RuntimeError("skeptic down")])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {
                "signal": signal_llm,
                "codegen": _ScriptedLLM([]),
                "advisor": _ScriptedLLM(["- try x"]),
                "skeptic": skeptic_llm,
            },
        )

        result = _run(single_fn)

        # The broken skeptic LLM must not veto or crash the run.
        assert result["status"] == "accepted"
        assert result["skeptic"]["vetoed"] == 0
        assert result["skeptic"]["reviewed"] == 1


# ---------------------------------------------------------------------------
# 2B: genetic crossover
# ---------------------------------------------------------------------------


class TestGeneticCrossover:
    def test_crossover_merged_into_next_iteration_with_lineage(self, monkeypatch, tmp_path):
        config = _config(max_iterations=2, num_signals=2, genetic_crossover=True)
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(_signal("Alpha"), _signal("Gamma")),
                _crossover_reply(),
                _signal_reply(_signal("Beta")),
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
            ic_results=[{"mean_ic": 0.01, "p_value": 0.01, "num_periods": 20}, {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20}],
        )

        result = _run(single_fn)

        # Iteration 1: two parents evaluated, crossover prompt built and sent.
        assert "genetic crossover" in signal_llm.prompts[1].lower()
        assert "Rank(TS_Return(Close, 10))" in signal_llm.prompts[1]  # parent formula verbatim
        assert signal_llm.calls == 3  # gen(1) + crossover + gen(2)

        # Iteration 2 accepted with the merged batch; lineage reflects it.
        assert result["status"] == "accepted"
        assert result["iteration"] == 2
        assert result["lineage"]["source"] == "genetic_crossover"
        names = {item["name"] for item in result["signals"]}
        assert {"Beta", "CrossBred"} <= names
        saved = json.loads(Path(result["saved_path"]).read_text())
        assert saved["lineage"]["source"] == "genetic_crossover"
        assert len(state.evaluated_codes) == 2

    def test_crossover_llm_failure_leaves_run_working(self, monkeypatch, tmp_path):
        config = _config(max_iterations=2, num_signals=2, genetic_crossover=True)
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(_signal("Alpha"), _signal("Gamma")),
                RuntimeError("crossover llm down"),
                _signal_reply(_signal("Beta")),
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
            ic_results=[{"mean_ic": 0.01, "p_value": 0.01, "num_periods": 20}, {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20}],
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        assert result["lineage"]["source"] == "optimization_feedback"
        assert "CrossBred" not in {item["name"] for item in result["signals"]}

    def test_crossover_disabled_never_calls_llm_for_crossover(self, monkeypatch, tmp_path):
        config = _config(max_iterations=2, num_signals=2)
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(_signal("Alpha"), _signal("Gamma")),
                _signal_reply(_signal("Beta")),
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
            ic_results=[{"mean_ic": 0.01, "p_value": 0.01, "num_periods": 20}, {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20}],
        )

        result = _run(single_fn)

        assert result["status"] == "accepted"
        assert signal_llm.calls == 2  # generation calls only
        assert all("genetic crossover" not in prompt.lower() for prompt in signal_llm.prompts)


# ---------------------------------------------------------------------------
# 2C: out-of-sample gate
# ---------------------------------------------------------------------------


def _oos_result(is_overfit=True, test_mean_ic=0.005):
    return {
        "train_mean_ic": 0.05,
        "train_p_value": 0.001,
        "test_mean_ic": test_mean_ic,
        "test_p_value": 0.4,
        "ic_decay": test_mean_ic - 0.05,
        "is_overfit": is_overfit,
        "train_periods": 70,
        "test_periods": 30,
        "split_date": "2024-02-15",
    }


class TestOOSGate:
    def test_cross_validate_called_with_configured_ratio_and_overfit_marked(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1, oos_validation=True, oos_train_ratio=0.6)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        def fake_cross_validate(signal_values, forward_returns, train_ratio=0.7):
            state.cross_validate_calls.append(train_ratio)
            return _oos_result(is_overfit=True)

        monkeypatch.setattr(quant_metrics, "cross_validate_signal", fake_cross_validate)

        result = _run(single_fn)

        assert state.cross_validate_calls == [0.6]
        assert result["status"] == "accepted"  # oos_reject_overfit is False
        assert result["overfit_risk"] is True
        assert result["oos_validation"]["is_overfit"] is True
        saved = json.loads(Path(result["saved_path"]).read_text())
        assert saved["evaluation_metrics"]["overfit_risk"] is True
        assert saved["evaluation_metrics"]["oos_validation"]["train_mean_ic"] == 0.05

    def test_oos_reject_overfit_blocks_acceptance_and_feeds_feedback(self, monkeypatch, tmp_path):
        config = _config(
            max_iterations=2, num_signals=1, oos_validation=True, oos_reject_overfit=True
        )
        signal_llm = _ScriptedLLM(
            [
                _signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))")),
                _signal_reply(_signal("Clean Two", "Rank(TS_Return(Close, 12))")),
            ]
        )
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
            ic_results=[
                {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20},
                {"mean_ic": 0.05, "p_value": 0.01, "num_periods": 20},
            ],
        )

        def fake_cross_validate(signal_values, forward_returns, train_ratio=0.7):
            state.cross_validate_calls.append(train_ratio)
            return _oos_result(is_overfit=True)

        monkeypatch.setattr(quant_metrics, "cross_validate_signal", fake_cross_validate)

        result = _run(single_fn)

        assert len(state.cross_validate_calls) == 2
        assert result["status"] == "best_effort"  # never reported as accepted
        assert result["overfit_risk"] is True
        assert "OUT-OF-SAMPLE CHECK FAILED" in result["last_feedback"]

    def test_not_overfit_signal_is_accepted_without_rejection(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1, oos_validation=True, oos_reject_overfit=True)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        def fake_cross_validate(signal_values, forward_returns, train_ratio=0.7):
            state.cross_validate_calls.append(train_ratio)
            return _oos_result(is_overfit=False, test_mean_ic=0.04)

        monkeypatch.setattr(quant_metrics, "cross_validate_signal", fake_cross_validate)

        result = _run(single_fn)

        assert result["status"] == "accepted"
        assert result["overfit_risk"] is False

    def test_quant_metrics_import_failure_skips_oos_check(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1, oos_validation=True)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.quant_metrics", None)

        result = _run(single_fn)

        assert state.cross_validate_calls == []
        assert result["status"] == "accepted"
        assert "overfit_risk" not in result
        assert "oos_validation" not in result
        saved = json.loads(Path(result["saved_path"]).read_text())
        assert "oos_validation" not in saved["evaluation_metrics"]


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestLegacyPayloadShape:
    def test_accepted_payload_keeps_every_legacy_key(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        for key in LEGACY_PAYLOAD_KEYS:
            assert key in result
        assert result["metrics"]["mean_ic"] == 0.05


# ---------------------------------------------------------------------------
# Portfolio metrics (always computed; additive merge into evaluation metrics)
# ---------------------------------------------------------------------------


class TestPortfolioMetrics:
    def test_saved_evaluation_metrics_contain_portfolio_metrics(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        saved = json.loads(Path(result["saved_path"]).read_text())
        metrics = saved["evaluation_metrics"]
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "turnover"):
            assert key in metrics, key
            assert isinstance(metrics[key], (int, float))
            assert math.isfinite(metrics[key]), f"{key}={metrics[key]}"

    def test_rank_ic_num_periods_is_not_clobbered(self, monkeypatch, tmp_path):
        # The fake rank-IC reports 20 usable IC dates; the portfolio side would
        # report its own (larger) period count. The merge rule must keep 20.
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        saved = json.loads(Path(result["saved_path"]).read_text())
        metrics = saved["evaluation_metrics"]
        assert metrics["num_periods"] == 20  # rank-IC meaning preserved
        assert metrics["num_periods"] != metrics.get("total_return")  # sanity: merge happened
        assert "sharpe_ratio" in metrics  # portfolio metrics were still merged
        assert result["metrics"]["num_periods"] == 20

    def test_legacy_keys_still_present_alongside_portfolio_keys(self, monkeypatch, tmp_path):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )

        result = _run(single_fn)

        for key in LEGACY_PAYLOAD_KEYS:
            assert key in result
        saved = json.loads(Path(result["saved_path"]).read_text())
        for key in ("timestamp", "iteration", "selected_signal", "signal_description", "signal_code"):
            assert key in saved
        assert saved["evaluation_metrics"]["mean_ic"] == 0.05
        assert "sharpe_ratio" in saved["evaluation_metrics"]

    def test_quant_metrics_import_failure_logs_warning_and_continues(
        self, monkeypatch, tmp_path, caplog
    ):
        config = _config(max_iterations=1, num_signals=1)
        signal_llm = _ScriptedLLM([_signal_reply(_signal("Clean", "Rank(TS_Return(Close, 10))"))])
        single_fn, state = _setup_workflow(
            monkeypatch, tmp_path, config,
            {"signal": signal_llm, "codegen": _ScriptedLLM([]), "advisor": _ScriptedLLM(["- try x"])},
        )
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.quant_metrics", None)

        with caplog.at_level(logging.WARNING):
            result = _run(single_fn)

        assert result["status"] == "accepted"
        saved = json.loads(Path(result["saved_path"]).read_text())
        metrics = saved["evaluation_metrics"]
        assert metrics["mean_ic"] == 0.05  # rank-IC metrics still present
        assert "num_periods" in metrics and metrics["num_periods"] == 20
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "turnover"):
            assert key not in metrics
        assert "Portfolio metrics unavailable" in caplog.text
