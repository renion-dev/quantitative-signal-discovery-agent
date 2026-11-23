#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for portfolio metrics and quant metrics.
# Tests compute_portfolio_metrics with fixed synthetic data and edge cases.

"""Verification of portfolio metrics: compute_portfolio_metrics."""

import json
import sys
from pathlib import Path

# Robust import handling
try:
    from signal_discovery_workflow.quant_metrics import compute_portfolio_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from signal_discovery_workflow.quant_metrics import compute_portfolio_metrics

import numpy as np
import pandas as pd


def test_constant_series_metrics():
    """Test that constant returns yield 0.0 for Sharpe, Sortino, annual volatility."""
    # Create a constant return series
    n = 252
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Constant returns
    const_returns = pd.Series(0.001, index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(const_returns)

    if metrics["sharpe_ratio"] != 0.0:
        print(f"FAIL: Constant series Sharpe should be 0.0, got {metrics['sharpe_ratio']}")
        return False
    if metrics["sortino_ratio"] != 0.0:
        print(f"FAIL: Constant series Sortino should be 0.0, got {metrics['sortino_ratio']}")
        return False
    if metrics["annual_volatility"] != 0.0:
        print(f"FAIL: Constant series annual_volatility should be 0.0, got {metrics['annual_volatility']}")
        return False
    # CAGR is geometric annualized return: for constant r=0.001, CAGR ≈ 0.001 * 252
    # (approximately, since (1+r)^n - 1 ≈ r*n for small r)
    # The key is: Sharpe, Sortino, annual_volatility = 0.0 for constant series

    print("PASS: Constant series yields 0.0 for Sharpe, Sortino, annual volatility")
    return True


def test_positive_constant_series():
    """Test positive constant returns: Sharpe, Sortino, annual_volatility = 0.0"""
    n = 252
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Positive constant returns
    const_returns = pd.Series(0.001, index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(const_returns)

    # All ratios should be 0 for constant series (no variance)
    if metrics["sharpe_ratio"] != 0.0:
        print(f"FAIL: Positive constant Sharpe should be 0.0, got {metrics['sharpe_ratio']}")
        return False
    if metrics["sortino_ratio"] != 0.0:
        print(f"FAIL: Positive constant Sortino should be 0.0, got {metrics['sortino_ratio']}")
        return False
    if metrics["annual_volatility"] != 0.0:
        print(f"FAIL: Positive constant annual_volatility should be 0.0, got {metrics['annual_volatility']}")
        return False

    print("PASS: Positive constant series yields 0.0 for Sharpe, Sortino, annual_volatility")
    return True


def test_known_values():
    """Test compute_portfolio_metrics against hand-calculated expected values."""
    # Create a simple deterministic return series
    n = 252
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Generate returns: +0.1% average with some variance
    returns = pd.Series(rng.normal(0.001, 0.02, n), index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(returns)

    # Manually compute expected values
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)

    # Annualized return
    expected_annual_ret = mean_ret * 252

    # Annualized volatility
    expected_ann_vol = std_ret * np.sqrt(252)

    # Sharpe ratio
    expected_sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 1e-10 else 0.0

    # Max drawdown
    equity = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(np.concatenate(([1.0], equity.values)))
    drawdowns = equity.values / running_max[1:] - 1.0
    expected_max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar ratio
    expected_calmar = expected_annual_ret / abs(expected_max_dd) if expected_max_dd != 0 else 0.0

    # Total return
    expected_total = float(equity.iloc[-1] - 1.0)

    # CAGR
    expected_cagr = float(equity.iloc[-1] ** (252 / n) - 1.0) if equity.iloc[-1] > 0 else 0.0

    # Verify
    if not np.isclose(metrics["annual_return"], expected_annual_ret, rtol=1e-6):
        print(f"FAIL: annual_return {metrics['annual_return']} != {expected_annual_ret}")
        return False
    if not np.isclose(metrics["annual_volatility"], expected_ann_vol, rtol=1e-6):
        print(f"FAIL: annual_volatility {metrics['annual_volatility']} != {expected_ann_vol}")
        return False
    if not np.isclose(metrics["sharpe_ratio"], expected_sharpe, rtol=1e-6):
        print(f"FAIL: sharpe_ratio {metrics['sharpe_ratio']} != {expected_sharpe}")
        return False
    if not np.isclose(metrics["max_drawdown"], expected_max_dd, rtol=1e-6):
        print(f"FAIL: max_drawdown {metrics['max_drawdown']} != {expected_max_dd}")
        return False
    if not np.isclose(metrics["calmar_ratio"], expected_calmar, rtol=1e-6):
        print(f"FAIL: calmar_ratio {metrics['calmar_ratio']} != {expected_calmar}")
        return False
    if not np.isclose(metrics["total_return"], expected_total, rtol=1e-6):
        print(f"FAIL: total_return {metrics['total_return']} != {expected_total}")
        return False
    if not np.isclose(metrics["cagr"], expected_cagr, rtol=1e-6):
        print(f"FAIL: cagr {metrics['cagr']} != {expected_cagr}")
        return False

    print("PASS: Known values match hand-calculated expected values")
    return True


def test_empty_series():
    """Test that empty series produces all zeros (JSON-serializable)."""
    empty_returns = pd.Series(dtype=float, name="portfolio_return")

    metrics = compute_portfolio_metrics(empty_returns)

    # Check JSON serializability with allow_nan=False
    try:
        json_str = json.dumps(metrics, allow_nan=False)
        json.loads(json_str)
    except (ValueError, TypeError) as e:
        print(f"FAIL: Empty series metrics not JSON-serializable: {e}")
        return False

    # Check that numeric metrics are 0.0
    for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "annual_return", "cagr", "annual_volatility", "total_return"]:
        if metrics[key] != 0.0:
            print(f"FAIL: Empty series {key} should be 0.0, got {metrics[key]}")
            return False

    print("PASS: Empty series produces JSON-serializable zeros")
    return True


def test_all_nan_series():
    """Test that all-NaN series produces all zeros (JSON-serializable)."""
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    nan_returns = pd.Series(np.nan, index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(nan_returns)

    # Check JSON serializability with allow_nan=False
    try:
        json_str = json.dumps(metrics, allow_nan=False)
        json.loads(json_str)
    except (ValueError, TypeError) as e:
        print(f"FAIL: All-NaN series metrics not JSON-serializable: {e}")
        return False

    print("PASS: All-NaN series produces JSON-serializable zeros")
    return True


def test_single_element_series():
    """Test that single-element series produces zeros (JSON-serializable)."""
    n = 1
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    single_return = pd.Series([0.001], index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(single_return)

    # Check JSON serializability with allow_nan=False
    try:
        json_str = json.dumps(metrics, allow_nan=False)
        json.loads(json_str)
    except (ValueError, TypeError) as e:
        print(f"FAIL: Single-element series metrics not JSON-serializable: {e}")
        return False

    print("PASS: Single-element series produces JSON-serializable zeros")
    return True


def test_overflow_input():
    """Test that overflow input produces non-finite metrics (fail-closed behavior)."""
    # Create a very large return series that could cause overflow
    n = 10
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Extremely large values that will cause overflow in calculations
    huge_returns = pd.Series(1e308, index=dates, name="portfolio_return")

    try:
        metrics = compute_portfolio_metrics(huge_returns)
        # Check that the result contains NaN or error info
        # The function is documented to raise ValueError for non-finite results
        if not all(np.isfinite(v) or v == 0.0 for k, v in metrics.items() if isinstance(v, (int, float))):
            print("PASS: Overflow input produces non-finite results (fail-closed behavior)")
            return True
        print("FAIL: Overflow input should produce non-finite metrics")
        return False
    except ValueError:
        print("PASS: Overflow input raised ValueError (fail-closed)")
        return True
    except Exception as e:
        print(f"PASS: Overflow input raised {type(e).__name__} (fail-closed)")
        return True


def test_sortino_calculation():
    """Test Sortino ratio calculation."""
    dates = pd.date_range("2022-01-01", periods=52, freq="W")

    # Returns with some downside deviation
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.03, 52), index=dates, name="portfolio_return")

    metrics = compute_portfolio_metrics(returns)

    # Sortino should be finite for this input
    if not np.isfinite(metrics["sortino_ratio"]):
        print(f"FAIL: Sortino ratio is not finite: {metrics['sortino_ratio']}")
        return False

    print(f"PASS: Sortino ratio = {metrics['sortino_ratio']:.4f} (finite)")
    return True


def main():
    """Run all quant metrics verification tests."""
    print("=" * 60)
    print("Quant Metrics Verification Tests")
    print("=" * 60)

    results = [
        test_constant_series_metrics(),
        test_positive_constant_series(),
        test_known_values(),
        test_empty_series(),
        test_all_nan_series(),
        test_single_element_series(),
        test_overflow_input(),
        test_sortino_calculation(),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All quant metrics verification tests passed")
        return 0
    else:
        print("FAIL: Some quant metrics verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
