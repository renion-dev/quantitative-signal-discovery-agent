#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for skeptic gate functionality.
# Tests detect_static_risks for case-insensitive operator matching, unknown operators,
# lookahead detection, and BLOCKING_RISK_PREFIXES constant.

"""Verification of skeptic gate: case-insensitive operator matching, unknown operators, lookahead detection."""

import sys
from pathlib import Path

# Robust import handling
try:
    from signal_discovery_workflow.signal_skeptic import (
        BLOCKING_RISK_PREFIXES,
        _load_operator_catalogue,
        detect_static_risks,
        has_blocking_risk,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from signal_discovery_workflow.signal_skeptic import (
        BLOCKING_RISK_PREFIXES,
        _load_operator_catalogue,
        detect_static_risks,
        has_blocking_risk,
    )


def test_operator_case_insensitive():
    """Test that operator case mismatch produces operator_case: advisory (not blocking)."""
    # A formula calling an operator with different letter case
    signal_desc = {
        "formula": "TS_ZScore(Close, 20)",  # Should be TS_Zscore in catalogue
        "data_fields_used": ["Close"],
        "lookback_periods": [20],
    }

    risks = detect_static_risks(signal_desc)

    # Check for operator_case warning
    operator_case_found = any("operator_case:" in r for r in risks)

    if not operator_case_found:
        print(f"FAIL: operator_case warning should be found for case mismatch, got: {risks}")
        return False

    # Check it's NOT blocking
    if has_blocking_risk(risks):
        print(f"FAIL: operator_case should be advisory, not blocking. Risks: {risks}")
        return False

    print(f"PASS: Case mismatch {signal_desc['formula']} produces operator_case advisory (not blocking)")
    return True


def test_unknown_operator_not_blocking():
    """Test that unknown operator produces blocking operator: finding."""
    signal_desc = {
        "formula": "TS_UnknownOp(Close, 20)",  # TS_ prefix looks like operator, but unknown
        "data_fields_used": ["Close"],
        "lookback_periods": [20],
    }

    risks = detect_static_risks(signal_desc)

    # Check for operator: blocking finding
    blocking_operator_found = any(r.startswith("operator:") for r in risks)

    if not blocking_operator_found:
        print(f"FAIL: Unknown operator should produce blocking 'operator:' finding, got: {risks}")
        return False

    if not has_blocking_risk(risks):
        print("FAIL: Unknown operator should be a blocking risk")
        return False

    print(f"PASS: Unknown operator {signal_desc['formula']} produces blocking operator: finding")
    return True


def test_lookahead_shift_negative():
    """Test that shift(-5) produces blocking lookahead: finding."""
    signal_desc = {
        "formula": "Close.shift(-5)",  # Negative shift = lookahead
        "data_fields_used": ["Close"],
        "lookback_periods": [1],
    }

    risks = detect_static_risks(signal_desc)

    # Check for lookahead: blocking finding
    lookahead_found = any("lookahead:" in r for r in risks)
    blocking_found = has_blocking_risk(risks)

    if not lookahead_found:
        print(f"FAIL: shift(-5) should produce lookahead: finding, got: {risks}")
        return False

    if not blocking_found:
        print("FAIL: lookahead should be blocking")
        return False

    print("PASS: shift(-5) produces blocking lookahead: finding")
    return True


def test_blocking_risk_prefixes_constant():
    """Test that BLOCKING_RISK_PREFIXES equals exactly ('lookahead:', 'leakage:', 'degenerate:', 'operator:')."""
    expected_prefixes = ("lookahead:", "leakage:", "degenerate:", "operator:")

    if BLOCKING_RISK_PREFIXES != expected_prefixes:
        print(f"FAIL: BLOCKING_RISK_PREFIXES = {BLOCKING_RISK_PREFIXES}, expected {expected_prefixes}")
        return False

    # Verify via has_blocking_risk
    for prefix in expected_prefixes:
        # Create a mock risk with this prefix
        test_risk = f"{prefix}test issue"
        if not has_blocking_risk([test_risk]):
            print(f"FAIL: Risk with prefix '{prefix}' should be blocking")
            return False

    print(f"PASS: BLOCKING_RISK_PREFIXES = {BLOCKING_RISK_PREFIXES}")
    return True


def test_leakage_detection():
    """Test that leakage (forward returns etc.) is detected as blocking."""
    signal_desc = {
        "formula": "forward_return",  # Future-only field
        "data_fields_used": ["forward_return"],
        "lookback_periods": [20],
    }

    risks = detect_static_risks(signal_desc)

    leakage_found = any("leakage:" in r for r in risks)

    if not leakage_found:
        print(f"FAIL: forward_return should trigger leakage:, got: {risks}")
        return False

    if not has_blocking_risk(risks):
        print("FAIL: leakage should be blocking")
        return False

    print("PASS: leakage (forward_return) detected as blocking")
    return True


def test_valid_operator_passes():
    """Test that valid operators in catalogue don't trigger any risk."""
    signal_desc = {
        "formula": "TS_Corr(Close, Volume, 20)",  # Valid operator
        "data_fields_used": ["Close", "Volume"],
        "lookback_periods": [20],
    }

    risks = detect_static_risks(signal_desc)

    if risks:
        print(f"FAIL: Valid operator TS_Corr should not produce risks, got: {risks}")
        return False

    print("PASS: Valid operator TS_Corr passes without risks")
    return True


def test_lead_function_lookahead():
    """Test that lead() function is detected as lookahead."""
    signal_desc = {
        "formula": "lead(Close)",
        "data_fields_used": ["Close"],
        "lookback_periods": [1],
    }

    risks = detect_static_risks(signal_desc)

    lookahead_found = any("lookahead:" in r for r in risks)

    if not lookahead_found:
        print(f"FAIL: lead() should produce lookahead: finding, got: {risks}")
        return False

    print("PASS: lead() detected as lookahead")
    return True


def test_degenerate_constant_formula():
    """Test that constant formula is detected as degenerate."""
    signal_desc = {
        "formula": "1 + 2 + 3",  # No operators or data fields
        "data_fields_used": [],  # Empty fields
        "lookback_periods": [],
    }

    risks = detect_static_risks(signal_desc)

    degenerate_found = any("degenerate:" in r for r in risks)

    if not degenerate_found:
        print(f"FAIL: Constant formula should produce degenerate: finding, got: {risks}")
        return False

    if not has_blocking_risk(risks):
        print("FAIL: degenerate should be blocking")
        return False

    print("PASS: Constant formula detected as degenerate (blocking)")
    return True


def test_operator_catalogue_loads():
    """Test that operator catalogue loads and has expected operators."""
    catalogue = _load_operator_catalogue()

    if catalogue is None:
        print("FAIL: Operator catalogue should load")
        return False

    valid, _like_names, _like_prefixes = catalogue

    if not valid:
        print("FAIL: Operator catalogue should have valid operators")
        return False

    # Check some expected operators exist
    expected_ops = ["TS_Corr", "TS_Std", "EMA", "Delay", "Add", "Sub", "Mul", "Div", "Rank", "TS_Zscore"]
    for op in expected_ops:
        if op not in valid:
            print(f"FAIL: Expected operator {op} not in catalogue")
            return False

    print(f"PASS: Operator catalogue loads with {len(valid)} operators")
    return True


def main():
    """Run all skeptic gate verification tests."""
    print("=" * 60)
    print("Skeptic Gate Verification Tests")
    print("=" * 60)

    results = [
        test_operator_case_insensitive(),
        test_unknown_operator_not_blocking(),
        test_lookahead_shift_negative(),
        test_blocking_risk_prefixes_constant(),
        test_leakage_detection(),
        test_valid_operator_passes(),
        test_lead_function_lookahead(),
        test_degenerate_constant_formula(),
        test_operator_catalogue_loads(),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All skeptic gate verification tests passed")
        return 0
    else:
        print("FAIL: Some skeptic gate verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
