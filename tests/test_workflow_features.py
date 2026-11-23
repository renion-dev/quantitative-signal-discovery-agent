# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the genetic crossover and skeptic helper modules.

Both modules are pure (no LLM, no NAT runtime), so everything here runs
without network access or configured models.
"""

from signal_discovery_workflow import signal_genetic, signal_skeptic
from signal_discovery_workflow.signal_generator import load_calculator_operators

# ---------------------------------------------------------------------------
# signal_genetic.select_parents
# ---------------------------------------------------------------------------


class TestSelectParents:
    def test_picks_top_k_by_absolute_ic(self):
        signals = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        metrics = [{"mean_ic": 0.10}, {"mean_ic": -0.30}, {"mean_ic": 0.20}]
        assert signal_genetic.select_parents(signals, metrics) == [{"name": "b"}, {"name": "c"}]

    def test_negative_ic_outranks_positive(self):
        signals = [{"name": "a"}, {"name": "b"}]
        metrics = [{"mean_ic": 0.30}, {"mean_ic": -0.50}]
        assert signal_genetic.select_parents(signals, metrics) == [{"name": "b"}, {"name": "a"}]

    def test_ties_prefer_earlier_signal(self):
        signals = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        metrics = [{"mean_ic": 0.20}, {"mean_ic": -0.20}, {"mean_ic": 0.10}]
        assert signal_genetic.select_parents(signals, metrics) == [{"name": "a"}, {"name": "b"}]

    def test_ignores_missing_and_invalid_ic(self):
        signals = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}, {"name": "e"}, {"name": "f"}]
        metrics = [
            {"mean_ic": None},
            {"mean_ic": 0.40},
            {},
            {"mean_ic": "not-a-number"},
            {"mean_ic": float("nan")},
            {"mean_ic": 0.10},
        ]
        assert signal_genetic.select_parents(signals, metrics) == [{"name": "b"}, {"name": "f"}]

    def test_returns_empty_with_fewer_than_two_valid_candidates(self):
        assert signal_genetic.select_parents([{"name": "a"}], [{"mean_ic": 0.5}]) == []
        assert signal_genetic.select_parents([{"name": "a"}, {"name": "b"}], [{"mean_ic": 0.5}, {}]) == []
        assert signal_genetic.select_parents([], []) == []

    def test_k_controls_parent_count(self):
        signals = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        metrics = [{"mean_ic": 0.10}, {"mean_ic": -0.30}, {"mean_ic": 0.20}]
        assert signal_genetic.select_parents(signals, metrics, k=1) == [{"name": "b"}]
        assert signal_genetic.select_parents(signals, metrics, k=5) == [
            {"name": "b"},
            {"name": "c"},
            {"name": "a"},
        ]

    def test_returns_empty_for_non_positive_k(self):
        signals = [{"name": "a"}, {"name": "b"}]
        metrics = [{"mean_ic": 0.1}, {"mean_ic": 0.2}]
        assert signal_genetic.select_parents(signals, metrics, k=0) == []


# ---------------------------------------------------------------------------
# signal_genetic.build_crossover_prompt
# ---------------------------------------------------------------------------


class TestBuildCrossoverPrompt:
    def test_includes_both_parent_formulas_verbatim(self):
        parent_a = {"name": "Momentum", "formula": "Rank(TS_Return(Close, 10))", "meaning": "10d momentum"}
        parent_b = {"name": "Volume", "formula": "Div(Volume, TS_Mean(Volume, 20))", "meaning": "volume surge"}
        prompt = signal_genetic.build_crossover_prompt(parent_a, parent_b)
        assert "Rank(TS_Return(Close, 10))" in prompt
        assert "Div(Volume, TS_Mean(Volume, 20))" in prompt

    def test_requests_exactly_one_combined_hypothesis(self):
        prompt = signal_genetic.build_crossover_prompt({"formula": "A"}, {"formula": "B"})
        assert "EXACTLY ONE" in prompt
        assert "single" in prompt.lower()

    def test_embeds_generator_json_schema(self):
        prompt = signal_genetic.build_crossover_prompt({"formula": "A"}, {"formula": "B"})
        for field in ("name", "formula", "meaning", "category", "data_fields_used"):
            assert field in prompt

    def test_lists_operator_catalogue_when_given_as_list(self):
        prompt = signal_genetic.build_crossover_prompt(
            {"formula": "A"}, {"formula": "B"}, operator_catalog=load_calculator_operators()
        )
        assert "TS_Return" in prompt

    def test_passes_string_catalogue_through(self):
        prompt = signal_genetic.build_crossover_prompt(
            {"formula": "A"}, {"formula": "B"}, operator_catalog="- MyOp(x)\n"
        )
        assert "- MyOp(x)" in prompt

    def test_includes_feedback_when_given(self):
        prompt = signal_genetic.build_crossover_prompt(
            {"formula": "A"}, {"formula": "B"}, feedback="- prefer shorter lookbacks"
        )
        assert "- prefer shorter lookbacks" in prompt


# ---------------------------------------------------------------------------
# signal_genetic.parse_crossover_response
# ---------------------------------------------------------------------------


class TestParseCrossoverResponse:
    def test_parses_fenced_json_object(self):
        text = (
            "Reasoning first.\n"
            '```json\n{"name": "Crossover", "formula": "Add(A, B)", "meaning": "combined"}\n```\n'
        )
        assert signal_genetic.parse_crossover_response(text) == {
            "name": "Crossover",
            "formula": "Add(A, B)",
            "meaning": "combined",
        }

    def test_parses_fenced_json_array(self):
        text = '```json\n[{"name": "A", "formula": "f1", "meaning": "m1"}, {"name": "B"}]\n```'
        assert signal_genetic.parse_crossover_response(text) == {"name": "A", "formula": "f1", "meaning": "m1"}

    def test_parses_plain_json_object(self):
        text = '{"name": "X", "formula": "Rank(Close)", "meaning": "level"}'
        assert signal_genetic.parse_crossover_response(text)["name"] == "X"

    def test_parses_plain_json_array(self):
        text = '[{"name": "X", "formula": "Rank(Close)", "meaning": "level"}]'
        assert signal_genetic.parse_crossover_response(text)["formula"] == "Rank(Close)"

    def test_parses_json_embedded_in_prose(self):
        text = (
            "Here is the crossover hypothesis I propose:\n"
            '{"name": "Blend", "formula": "Sub(Rank(TS_Return(Close, 10)), Rank(Div(Volume, TS_Mean(Volume, 20))))", '
            '"meaning": "momentum minus volume surge"}\n'
            "Let me know if you want variants."
        )
        parsed = signal_genetic.parse_crossover_response(text)
        assert parsed is not None
        assert parsed["name"] == "Blend"

    def test_returns_none_for_garbage(self):
        assert signal_genetic.parse_crossover_response("no structured output here") is None

    def test_returns_none_for_empty_or_none(self):
        assert signal_genetic.parse_crossover_response("") is None
        assert signal_genetic.parse_crossover_response("   \n  ") is None
        assert signal_genetic.parse_crossover_response(None) is None  # type: ignore[arg-type]

    def test_returns_none_when_formula_missing(self):
        assert signal_genetic.parse_crossover_response('{"name": "X", "meaning": "no formula"}') is None


# ---------------------------------------------------------------------------
# signal_skeptic.detect_static_risks
# ---------------------------------------------------------------------------


class TestDetectStaticRisks:
    def test_flags_forward_looking_shift(self):
        risks = signal_skeptic.detect_static_risks(
            {"name": "Leaky", "formula": "Rank(Close.shift(-5))", "data_fields_used": ["Close"]}
        )
        assert any(r.startswith("lookahead:") for r in risks)

    def test_flags_negative_pct_change(self):
        risks = signal_skeptic.detect_static_risks({"name": "Leaky", "formula": "Sub(Close.pct_change(-1), 0)"})
        assert any(r.startswith("lookahead:") for r in risks)

    def test_flags_future_identifier(self):
        risks = signal_skeptic.detect_static_risks(
            {"name": "Leaky", "formula": "Rank(TS_Return(future_close, 10))"}
        )
        assert any(r.startswith("lookahead:") for r in risks)

    def test_flags_same_bar_lookahead(self):
        risks = signal_skeptic.detect_static_risks({"name": "SameBar", "formula": "Rank(Close)"})
        assert any(r.startswith("lookahead:") for r in risks)

    def test_flags_absurd_lookback(self):
        risks = signal_skeptic.detect_static_risks(
            {
                "name": "Slow",
                "formula": "Rank(TS_Mean(Close, 500))",
                "data_fields_used": ["Close"],
                "lookback_periods": [500],
            }
        )
        assert any(r.startswith("lookback:") for r in risks)

    def test_flags_lookback_below_minimum(self):
        risks = signal_skeptic.detect_static_risks({"name": "Fast", "formula": "Rank(TS_Return(Close, 1))"})
        assert any("below the minimum" in r for r in risks)

    def test_flags_unbounded_while_loop(self):
        code = "def signal_x(Close):\n    while True:\n        Close = Close * 1.0\n    return Close\n"
        risks = signal_skeptic.detect_static_risks({"name": "Spin", "formula": "Rank(Close)"}, code=code)
        assert any(r.startswith("loop:") for r in risks)

    def test_flags_huge_range_loop(self):
        code = "def signal_x(Close):\n    for i in range(20000):\n        pass\n    return Close\n"
        risks = signal_skeptic.detect_static_risks({"name": "Loop", "formula": "Rank(Close)"}, code=code)
        assert any(r.startswith("loop:") for r in risks)

    def test_does_not_flag_reasonable_range_loop(self):
        code = "def signal_x(Close):\n    for i in range(50):\n        pass\n    return Close\n"
        risks = signal_skeptic.detect_static_risks({"name": "Ok", "formula": "Rank(TS_Mean(Close, 5))"}, code=code)
        assert not any(r.startswith("loop:") for r in risks)

    def test_flags_non_ohlcv_declared_field(self):
        risks = signal_skeptic.detect_static_risks(
            {
                "name": "Sentiment",
                "formula": "Rank(TS_Return(Close, 10))",
                "data_fields_used": ["InsiderSentiment"],
            }
        )
        assert any(r.startswith("field:") for r in risks)

    def test_flags_future_only_field_as_leakage(self):
        risks = signal_skeptic.detect_static_risks(
            {"name": "Leak", "formula": "Rank(forward_return_5d)", "data_fields_used": ["forward_return"]}
        )
        assert any(r.startswith("leakage:") for r in risks)

    def test_flags_constant_formula_as_degenerate(self):
        risks = signal_skeptic.detect_static_risks({"name": "Const", "formula": "0.5"})
        assert any(r.startswith("degenerate:") for r in risks)

    def test_flags_hype_in_meaning(self):
        risks = signal_skeptic.detect_static_risks(
            {
                "name": "Hype",
                "formula": "Rank(TS_Return(Close, 10))",
                "meaning": "This signal is guaranteed to predict returns with 100% accuracy",
            }
        )
        assert any(r.startswith("suspicious:") for r in risks)

    def test_clean_momentum_formula_has_no_risks(self):
        risks = signal_skeptic.detect_static_risks(
            {
                "name": "Ten Day Rank Momentum",
                "formula": "Rank(TS_Return(Close, 10))",
                "meaning": "Ranks the 10-day close-price momentum",
                "category": "momentum",
                "data_fields_used": ["Close"],
                "lookback_periods": [10],
            }
        )
        assert risks == []

    def test_non_dict_description_returns_empty(self):
        assert signal_skeptic.detect_static_risks(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# signal_skeptic.has_blocking_risk
# ---------------------------------------------------------------------------


class TestHasBlockingRisk:
    def test_true_for_blocking_prefixes(self):
        for risk in ("lookahead: x", "leakage: y", "degenerate: z"):
            assert signal_skeptic.has_blocking_risk([risk]) is True

    def test_true_when_blocking_risk_mixed_with_advisory(self):
        assert signal_skeptic.has_blocking_risk(["lookback: long", "leakage: target"]) is True

    def test_false_for_advisory_only(self):
        risks = ["lookback: long", "loop: while True", "field: extra", "formula: odd", "suspicious: hype"]
        assert signal_skeptic.has_blocking_risk(risks) is False

    def test_false_for_empty_or_garbage(self):
        assert signal_skeptic.has_blocking_risk([]) is False
        assert signal_skeptic.has_blocking_risk(None) is False  # type: ignore[arg-type]
        assert signal_skeptic.has_blocking_risk([42]) is False  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# signal_skeptic.build_skeptic_prompt
# ---------------------------------------------------------------------------


class TestBuildSkepticPrompt:
    def test_mentions_verdicts_and_schema(self):
        prompt = signal_skeptic.build_skeptic_prompt({"name": "S", "formula": "Rank(Close)"})
        for token in ("approve", "revise", "reject", "verdict", "issues", "rationale"):
            assert token in prompt

    def test_embeds_description_code_and_risks(self):
        prompt = signal_skeptic.build_skeptic_prompt(
            {"name": "S", "formula": "Rank(Close)"},
            code="def signal_s(Close):\n    return Close\n",
            static_risks=["lookahead: same-bar prices"],
        )
        assert "Rank(Close)" in prompt
        assert "def signal_s(Close):" in prompt
        assert "lookahead: same-bar prices" in prompt

    def test_works_without_code_and_risks(self):
        prompt = signal_skeptic.build_skeptic_prompt({"name": "S", "formula": "Rank(Close)"})
        assert "not yet generated" in prompt
        assert "no classic pitfalls detected" in prompt


class TestLoadSkepticTemplate:
    def test_loads_shipped_template(self):
        template = signal_skeptic.load_skeptic_template()
        assert template.get("allowed_verdicts") == ["approve", "revise", "reject"]
        assert "verdict" in template.get("review_template", {})


# ---------------------------------------------------------------------------
# signal_skeptic.parse_skeptic_review
# ---------------------------------------------------------------------------


class TestParseSkepticReview:
    def test_parses_approve_json(self):
        review = signal_skeptic.parse_skeptic_review(
            '{"verdict": "approve", "issues": [], "rationale": "no lookahead detected"}'
        )
        assert review == {"verdict": "approve", "issues": [], "rationale": "no lookahead detected"}

    def test_parses_revise_json_with_alias(self):
        review = signal_skeptic.parse_skeptic_review(
            '{"verdict": "needs_revision", "issues": ["lookback too short"], "rationale": "extend to 20 days"}'
        )
        assert review["verdict"] == "revise"
        assert review["issues"] == ["lookback too short"]
        assert review["rationale"] == "extend to 20 days"

    def test_parses_reject_json_with_alias(self):
        review = signal_skeptic.parse_skeptic_review(
            '{"verdict": "rejected", "issues": ["uses future data"], "rationale": "lookahead bias"}'
        )
        assert review["verdict"] == "reject"

    def test_parses_fenced_json(self):
        text = 'Analysis...\n```json\n{"verdict": "reject", "issues": ["shift(-1)"], "rationale": "forward-looking"}\n```'
        review = signal_skeptic.parse_skeptic_review(text)
        assert review["verdict"] == "reject"
        assert "shift(-1)" in review["issues"]

    def test_heuristic_fallback_reject(self):
        review = signal_skeptic.parse_skeptic_review("This signal uses future prices. I reject it outright.")
        assert review["verdict"] == "reject"
        assert review["rationale"]

    def test_heuristic_fallback_revise(self):
        review = signal_skeptic.parse_skeptic_review("Promising, but please revise the lookback.")
        assert review["verdict"] == "revise"

    def test_garbage_defaults_to_approve(self):
        review = signal_skeptic.parse_skeptic_review("...lgtm...")
        assert review == {"verdict": "approve", "issues": [], "rationale": ""}

    def test_empty_and_none_inputs(self):
        default = {"verdict": "approve", "issues": [], "rationale": ""}
        assert signal_skeptic.parse_skeptic_review("") == default
        assert signal_skeptic.parse_skeptic_review(None) == default

    def test_never_raises_on_weird_input(self):
        for weird in (123, ["verdict"], {"verdict": 5}, "```json\n{broken\n```"):
            review = signal_skeptic.parse_skeptic_review(weird)  # type: ignore[arg-type]
            assert set(review) == {"verdict", "issues", "rationale"}
            assert review["verdict"] in ("approve", "revise", "reject")


# ---------------------------------------------------------------------------
# signal_skeptic.extract_function_source
# ---------------------------------------------------------------------------


class TestExtractFunctionSource:
    MODULE = (
        "import pandas as pd\n"
        "\n"
        "def signal_alpha(Close):\n"
        '    """alpha"""\n'
        "    return Close\n"
        "\n"
        "def signal_beta(Close):\n"
        "    return Close * 2\n"
    )

    def test_extracts_first_function(self):
        src = signal_skeptic.extract_function_source(self.MODULE, "signal_alpha")
        assert src is not None
        assert src.startswith("def signal_alpha(Close):")
        assert '"""alpha"""' in src
        assert "def signal_beta" not in src

    def test_extracts_second_function(self):
        src = signal_skeptic.extract_function_source(self.MODULE, "signal_beta")
        assert src is not None
        assert src.startswith("def signal_beta(Close):")
        assert "signal_alpha" not in src

    def test_returns_none_for_missing_function(self):
        assert signal_skeptic.extract_function_source(self.MODULE, "signal_gamma") is None
        assert signal_skeptic.extract_function_source("", "signal_alpha") is None
        assert signal_skeptic.extract_function_source(None, "signal_alpha") is None  # type: ignore[arg-type]
