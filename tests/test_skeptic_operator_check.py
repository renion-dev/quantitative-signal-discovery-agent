# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skeptic's blocking unknown-operator check (``operator:`` category).

A formula calling an operator that is absent from ``template/calculator.json``
cannot execute (NameError at backtest time), so the finding must be blocking.
These tests also pin the false-positive controls: Python builtins, module
references and non-operator-looking helper names are never reported, and an
unreadable catalogue degrades silently instead of raising.
"""

from pathlib import Path

import pytest

from signal_discovery_workflow import signal_skeptic


def _detect(formula: str) -> list[str]:
    return signal_skeptic.detect_static_risks({"name": "T", "formula": formula, "data_fields_used": ["Close"]})


class TestUnknownOperatorDetection:
    def test_flags_hallucinated_ts_operator_and_blocks(self):
        risks = _detect("Rank(TS_Volume(Close, 5))")
        operator_risks = [r for r in risks if r.startswith("operator:")]
        assert operator_risks, risks
        assert any("TS_Volume" in r for r in operator_risks)
        assert signal_skeptic.has_blocking_risk(risks) is True

    def test_reports_each_unknown_operator_once(self):
        risks = _detect("Add(TS_Volume(Close, 5), TS_Volatility(Volume, 5))")
        operator_risks = [r for r in risks if r.startswith("operator:")]
        assert any("TS_Volume" in r for r in operator_risks)
        assert any("TS_Volatility" in r for r in operator_risks)
        assert len(operator_risks) == 2

    def test_valid_catalogue_operators_not_flagged(self):
        assert _detect("Rank(TS_Return(Close, 20))") == []
        assert _detect("Div(TS_Return(Close, 5), Decay_Linear(Volume, 5))") == []

    def test_non_operator_helper_not_flagged(self):
        risks = _detect("my_helper(Close)")
        assert not any(r.startswith("operator:") for r in risks)

    def test_python_builtin_call_not_flagged(self):
        assert _detect("abs(TS_Return(Close, 5))") == []

    def test_dotted_module_and_method_calls_not_flagged(self):
        risks = _detect("Add(np.log(Close), my_helper(Close))")
        assert not any(r.startswith("operator:") for r in risks)

    def test_category_prefix_and_message_name_offender(self):
        risks = _detect("EMA_Return(Close, 10)")
        operator_risks = [r for r in risks if r.startswith("operator:")]
        assert operator_risks, risks
        catalogue = signal_skeptic._load_operator_catalogue()
        assert catalogue is not None
        assert f"not in the {len(catalogue.valid)}-operator catalogue" in operator_risks[0]
        assert "EMA_Return" in operator_risks[0]

    def test_missing_catalogue_degrades_silently(self, monkeypatch):
        monkeypatch.setattr(signal_skeptic, "_load_operator_catalogue", lambda: None)
        risks = _detect("Rank(TS_Volume(Close, 5))")
        assert isinstance(risks, list)
        assert not any(r.startswith("operator:") for r in risks)

    def test_unreadable_catalogue_path_degrades_silently(self, monkeypatch):
        monkeypatch.setattr(signal_skeptic, "OPERATOR_CATALOGUE_PATH", Path("/nonexistent/calculator.json"))
        first = _detect("Rank(TS_Volume(Close, 5))")
        second = _detect("Rank(TS_Volume(Close, 5))")  # repeated: cached-failure path must not raise either
        assert isinstance(first, list) and isinstance(second, list)
        assert not any(r.startswith("operator:") for r in first + second)


class TestExistingCategoriesUnaffected:
    def test_negative_shift_still_blocking_lookahead(self):
        risks = signal_skeptic.detect_static_risks(
            {"name": "Leaky", "formula": "Rank(Close.shift(-5))", "data_fields_used": ["Close"]}
        )
        assert any(r.startswith("lookahead:") for r in risks)
        assert signal_skeptic.has_blocking_risk(risks) is True

    def test_advisory_categories_still_non_blocking(self):
        assert signal_skeptic.has_blocking_risk(["lookback: long", "field: extra", "suspicious: hype"]) is False


class TestOperatorCaseNearMiss:
    """A catalogue operator spelled with a different letter case is NOT a hallucination.

    The candidate is valid (the operator exists), so the finding must be advisory
    (``operator_case:``) and must never veto the candidate. Genuine hallucinations
    (an operator absent from the catalogue under ANY case) stay blocking.
    """

    @pytest.mark.parametrize(
        "formula",
        [
            "Rank(TS_ZScore(Close, 20))",
            "Rank(Ts_Return(Close, 20))",
            "Rank(Ts_Mean(Close, 20))",
            "Rank(Ts_Std(Close, 20))",
            "Rank(Ts_Median(Close, 20))",
        ],
    )
    def test_case_only_near_miss_is_advisory_not_blocking(self, formula):
        risks = _detect(formula)
        assert not any(r.startswith("operator:") for r in risks), risks
        assert signal_skeptic.has_blocking_risk(risks) is False, risks
        advisory = [r for r in risks if r.startswith("operator_case:")]
        assert advisory, risks
        assert "differs in case" in advisory[0]

    def test_case_only_near_miss_names_both_spellings(self):
        risks = _detect("Rank(TS_ZScore(Close, 20))")
        advisory = [r for r in risks if r.startswith("operator_case:")][0]
        assert "TS_ZScore" in advisory
        assert "TS_Zscore" in advisory

    @pytest.mark.parametrize(
        "formula",
        [
            "Rank(TS_Volume(Close, 5))",
            "Rank(TS_Volatility(Close, 5))",
            "Rank(TS_MeanReversion(Close, 5))",
            "Rank(EMA_Return(Close, 5))",
        ],
    )
    def test_genuine_hallucinations_stay_blocking(self, formula):
        risks = _detect(formula)
        assert any(r.startswith("operator:") for r in risks), risks
        assert signal_skeptic.has_blocking_risk(risks) is True, risks

    def test_exact_catalogue_name_produces_no_advisory_noise(self):
        risks = _detect("Rank(TS_Zscore(Close, 20))")
        assert not any(r.startswith("operator_case:") for r in risks), risks
        assert not any(r.startswith("operator:") for r in risks), risks

    def test_bare_family_that_is_not_a_catalogue_operator_stays_blocking(self):
        # ``Std`` is only an operator-LIKE name; the catalogue only has ``TS_Std``.
        risks = _detect("Std(Close, 20)")
        assert any(r.startswith("operator:") for r in risks), risks
        assert signal_skeptic.has_blocking_risk(risks) is True, risks

    def test_module_and_builtin_references_are_not_flagged(self):
        risks = _detect("Rank(np.mean(Close))")
        assert not any(r.startswith("operator:") for r in risks), risks
        assert not any(r.startswith("operator_case:") for r in risks), risks

    def test_leakage_category_unaffected_by_case_normalisation(self):
        risks = signal_skeptic.detect_static_risks(
            {"name": "Leaky", "formula": "Rank(TS_Return(forward_return_5, 20))",
             "data_fields_used": ["Close"]}
        )
        assert any(r.startswith("leakage:") for r in risks), risks
        assert signal_skeptic.has_blocking_risk(risks) is True, risks

    def test_blocking_prefix_set_is_unchanged(self):
        assert signal_skeptic.BLOCKING_RISK_PREFIXES == (
            "lookahead:", "leakage:", "degenerate:", "operator:",
        )

    def test_case_category_is_advisory_by_contract(self):
        assert "operator_case:" in signal_skeptic.NON_BLOCKING_RISK_PREFIXES
        assert "operator_case:" not in signal_skeptic.BLOCKING_RISK_PREFIXES
        assert signal_skeptic.has_blocking_risk(["operator_case: spelling"]) is False

    def test_prompt_does_not_present_case_findings_as_hard_violations(self):
        risks = _detect("Rank(TS_ZScore(Close, 20))")
        prompt = signal_skeptic.build_skeptic_prompt(
            {"name": "T", "formula": "Rank(TS_ZScore(Close, 20))"}, code=None, static_risks=risks
        )
        assert "operator_case:" in prompt
        assert "NOT a hard violation" in prompt
