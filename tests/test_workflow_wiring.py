# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow wiring: config keys, run_id/lineage persistence, and
parallel hypothesis generation. No live LLM or NAT runtime is required."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import signal_discovery_workflow.signal_discovery_optimization_workflow as workflow
from signal_discovery_workflow.signal_discovery_optimization_workflow import (
    _build_lineage,
    _format_workflow_result,
    generate_signals_concurrently,
    save_signal_results,
)

CONFIG_PATHS = [
    Path(__file__).parent.parent / "configs" / "config-optimization.yml",
    Path(__file__).parent.parent / "configs" / "config-optimization-no-telemetry.yml",
]

NEW_CONFIG_KEYS = {
    "parallel_generation": False,
    "max_concurrency": 4,
    "genetic_crossover": False,
    "genetic_parents": 2,
    "crossover_fallback_enabled": False,
    "skeptic_review": False,
    "directional_ic_enabled": False,
    "oos_validation": False,
    "oos_train_ratio": 0.7,
    "oos_reject_overfit": False,
    "save_evolution": True,
}

EXISTING_WORKFLOW_KEYS = [
    "_type",
    "signal_generator_llm",
    "code_generator_llm",
    "optimization_advisor_llm",
    "ic_threshold",
    "p_value_threshold",
    "max_iterations",
    "num_signals",
    "forward_periods",
    "save_results",
]

LEGACY_SAVE_KEYS = {
    "timestamp",
    "iteration",
    "selected_signal",
    "signal_description",
    "signal_code",
    "evaluation_metrics",
}


def _make_reply(index: int) -> str:
    """A one-signal fenced JSON reply, uniquely named per request slot."""
    signal = {
        "name": f"sig{index}",
        "formula": "Rank(TS_Return(Close, 10))",
        "meaning": f"momentum variant {index}",
        "category": "momentum",
        "data_fields_used": ["Close"],
        "operators_used": ["Rank", "TS_Return"],
        "lookback_periods": [10],
    }
    return "```json\n" + json.dumps([signal]) + "\n```"


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _InFlightTracker:
    def __init__(self):
        self.current = 0
        self.max_seen = 0

    def enter(self):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)

    def exit(self):
        self.current -= 1


class _FakeLLM:
    """Async fake mirroring the LangChain ``ainvoke`` interface used by the workflow."""

    def __init__(self, replies, delay_for=None, tracker: _InFlightTracker | None = None):
        self._replies = list(replies)
        self._delay_for = delay_for or (lambda index: 0.0)
        self._tracker = tracker
        self.calls = 0

    async def ainvoke(self, messages):
        index = self.calls
        self.calls += 1
        if self._tracker is not None:
            self._tracker.enter()
        try:
            await asyncio.sleep(self._delay_for(index))
            reply = self._replies[index]
            if isinstance(reply, Exception):
                raise reply
            return _FakeResponse(reply)
        finally:
            if self._tracker is not None:
                self._tracker.exit()


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


class TestWorkflowConfigs:
    @pytest.fixture
    def configs(self):
        return {path.name: yaml.safe_load(path.read_text()) for path in CONFIG_PATHS}

    def test_both_configs_parse_with_workflow_section(self, configs):
        assert set(configs) == {"config-optimization.yml", "config-optimization-no-telemetry.yml"}
        for name, cfg in configs.items():
            assert isinstance(cfg, dict), name
            assert cfg["workflow"]["_type"] == "signal_optimizer", name

    def test_new_keys_present_with_documented_defaults(self, configs):
        for name, cfg in configs.items():
            for key, expected in NEW_CONFIG_KEYS.items():
                assert cfg["workflow"][key] == expected, f"{name}: {key}"

    def test_configs_agree_on_new_keys(self, configs):
        values = [tuple(cfg["workflow"][key] for key in NEW_CONFIG_KEYS) for cfg in configs.values()]
        assert values[0] == values[1]

    def test_existing_keys_still_present(self, configs):
        for name, cfg in configs.items():
            for key in EXISTING_WORKFLOW_KEYS:
                assert key in cfg["workflow"], f"{name}: missing {key}"
            for llm_name in ("signal_generator", "code_generator", "optimization_advisor"):
                assert llm_name in cfg["llms"], f"{name}: missing llm {llm_name}"


# ---------------------------------------------------------------------------
# _build_lineage
# ---------------------------------------------------------------------------


class TestBuildLineage:
    def test_iteration_one_is_seed_with_null_parents(self):
        assert _build_lineage(1, "run123", None, None, False) == {
            "parent_signal": None,
            "parent_run_id": None,
            "source": "seed",
        }

    def test_iteration_one_with_seed_feedback_is_resumed(self):
        lineage = _build_lineage(1, "run123", "- try momentum", None, False)
        assert lineage["source"] == "resumed"
        assert lineage["parent_signal"] is None
        assert lineage["parent_run_id"] is None

    def test_later_iteration_uses_previous_selected_signal(self):
        assert _build_lineage(2, "run123", None, "signal_momentum", False) == {
            "parent_signal": "signal_momentum",
            "parent_run_id": "run123",
            "source": "optimization_feedback",
        }

    def test_later_iteration_parent_can_be_none(self):
        lineage = _build_lineage(3, "run123", None, None, False)
        assert lineage["parent_signal"] is None
        assert lineage["parent_run_id"] == "run123"
        assert lineage["source"] == "optimization_feedback"

    def test_crossover_flag_sets_genetic_crossover_source(self):
        lineage = _build_lineage(2, "run123", None, "signal_x", True)
        assert lineage["source"] == "genetic_crossover"
        assert lineage["parent_signal"] == "signal_x"
        assert lineage["parent_run_id"] == "run123"

    def test_source_always_within_allowed_set(self):
        allowed = {"seed", "resumed", "optimization_feedback", "genetic_crossover"}
        for iteration in (1, 2, 5):
            for seed in (None, "fb"):
                for parent in (None, "sig"):
                    for crossover in (False, True):
                        lineage = _build_lineage(iteration, "r", seed, parent, crossover)
                        assert lineage["source"] in allowed


# ---------------------------------------------------------------------------
# save_signal_results
# ---------------------------------------------------------------------------


class TestSaveSignalResults:
    @pytest.fixture
    def output_dir(self, monkeypatch, tmp_path):
        out = tmp_path / "output"
        monkeypatch.setattr(workflow, "OUTPUT_DIR", out)
        return out

    @staticmethod
    def _ic_results():
        return {"mean_ic": 0.05, "p_value": 0.01, "selected_signal": "signal_momentum"}

    def test_evolution_payload_and_filename(self, output_dir):
        lineage = {"parent_signal": None, "parent_run_id": None, "source": "seed"}
        path = save_signal_results(
            '[{"name": "M"}]',
            "code",
            self._ic_results(),
            1,
            run_id="runabc123",
            lineage=lineage,
            request="momentum signals",
            save_evolution=True,
        )
        assert "runabc123" in Path(path).name
        assert "_iter1" in Path(path).name
        data = json.loads(Path(path).read_text())
        assert data["run_id"] == "runabc123"
        assert data["lineage"] == lineage
        assert data["request"] == "momentum signals"
        for key in LEGACY_SAVE_KEYS:
            assert key in data

    def test_no_evolution_payload_is_exactly_legacy_keys(self, output_dir):
        path = save_signal_results(
            '[{"name": "M"}]',
            "code",
            self._ic_results(),
            2,
            run_id="runabc123",
            save_evolution=False,
            run_id_for_filename="runabc123",
        )
        assert "runabc123" in Path(path).name
        data = json.loads(Path(path).read_text())
        assert set(data) == LEGACY_SAVE_KEYS

    def test_legacy_call_keeps_legacy_filename_and_keys(self, output_dir):
        path = save_signal_results("[]", "code", {}, 3)
        assert re.fullmatch(r"signal_\d{8}_\d{6}_iter3\.json", Path(path).name)
        data = json.loads(Path(path).read_text())
        assert set(data) == LEGACY_SAVE_KEYS

    def test_run_id_identical_across_iterations_of_one_run(self, output_dir):
        p1 = save_signal_results(
            "[]", "c1", self._ic_results(), 1,
            run_id="runshared", lineage=_build_lineage(1, "runshared", None, None, False),
            request="r", save_evolution=True,
        )
        p2 = save_signal_results(
            "[]", "c2", self._ic_results(), 2,
            run_id="runshared", lineage=_build_lineage(2, "runshared", None, "signal_m", False),
            request="r", save_evolution=True,
        )
        d1 = json.loads(Path(p1).read_text())
        d2 = json.loads(Path(p2).read_text())
        assert d1["run_id"] == d2["run_id"] == "runshared"
        assert d1["lineage"]["source"] == "seed"
        assert d2["lineage"]["source"] == "optimization_feedback"
        assert Path(p1).name == "signal_runshared_iter1.json"
        assert Path(p2).name == "signal_runshared_iter2.json"


# ---------------------------------------------------------------------------
# generate_signals_concurrently
# ---------------------------------------------------------------------------


class TestGenerateSignalsConcurrently:
    def test_results_preserve_request_order_despite_reversed_completion(self):
        num = 5
        llm = _FakeLLM(
            [_make_reply(i) for i in range(num)],
            delay_for=lambda index: (num - index) * 0.02,  # last request finishes first
        )
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", num, [], None, max_concurrency=4))
        data = json.loads(out)
        assert [item["name"] for item in data] == [f"sig{i}" for i in range(num)]
        assert llm.calls == num

    def test_failing_call_does_not_abort_batch(self):
        replies = [_make_reply(0), RuntimeError("boom"), _make_reply(2), _make_reply(3)]
        llm = _FakeLLM(replies, delay_for=lambda index: 0.01 * (index % 2))
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", 4, [], None, max_concurrency=2))
        data = json.loads(out)
        assert [item["name"] for item in data] == ["sig0", "sig2", "sig3"]

    def test_unparseable_reply_is_skipped(self):
        replies = [_make_reply(0), "no structured output here", _make_reply(2)]
        llm = _FakeLLM(replies)
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", 3, [], None, max_concurrency=3))
        assert [item["name"] for item in json.loads(out)] == ["sig0", "sig2"]

    def test_respects_max_concurrency(self):
        tracker = _InFlightTracker()
        llm = _FakeLLM([_make_reply(i) for i in range(6)], delay_for=lambda index: 0.05, tracker=tracker)
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", 6, [], None, max_concurrency=2))
        assert len(json.loads(out)) == 6
        assert tracker.max_seen <= 2

    def test_all_failures_return_empty_array_without_raising(self):
        llm = _FakeLLM([RuntimeError("x")] * 3)
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", 3, [], None, max_concurrency=2))
        assert json.loads(out) == []

    def test_zero_signals_returns_empty_array(self):
        llm = _FakeLLM([])
        out = asyncio.run(generate_signals_concurrently(llm, "momentum", 0, [], None))
        assert json.loads(out) == []
        assert llm.calls == 0


# ---------------------------------------------------------------------------
# Regression: module import, config tolerance, result-payload shape
# ---------------------------------------------------------------------------


class TestWorkflowRegression:
    def test_workflow_module_imports_and_registers(self):
        assert hasattr(workflow, "signal_optimizer_function")
        assert hasattr(workflow, "SignalOptimizerConfig")
        assert hasattr(workflow, "save_signal_results")

    def test_config_data_model_defaults_match_yaml(self):
        cfg = workflow.SignalOptimizerConfig(llm_name="test-llm")
        for key, expected in NEW_CONFIG_KEYS.items():
            assert getattr(cfg, key) == expected, key

    def test_format_result_adds_run_id_and_lineage_only_when_supplied(self):
        base_kwargs = dict(
            status="accepted",
            request="momentum signals",
            iteration=1,
            total_iterations=3,
            signal_json='[{"name": "N", "formula": "F", "meaning": "M"}]',
            ic_results={"mean_ic": 0.1},
            saved_path=None,
            config=SimpleNamespace(ic_threshold=0.02, p_value_threshold=0.05),
        )
        legacy_payload = json.loads(_format_workflow_result(**base_kwargs))
        assert "run_id" not in legacy_payload
        assert "lineage" not in legacy_payload
        for key in (
            "status", "headline", "request", "iteration", "total_iterations",
            "selected_signal", "thresholds", "metrics", "signals", "saved_path",
        ):
            assert key in legacy_payload

        lineage = {"parent_signal": None, "parent_run_id": None, "source": "seed"}
        payload = json.loads(
            _format_workflow_result(**base_kwargs, run_id="runxyz789", lineage=lineage)
        )
        assert payload["run_id"] == "runxyz789"
        assert payload["lineage"] == lineage
        for key in ("status", "headline", "request", "iteration", "total_iterations",
                    "selected_signal", "thresholds", "metrics", "signals", "saved_path"):
            assert key in payload
