# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantitative portfolio metrics and signal cross-validation."""

import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow.quant_metrics import (
    build_portfolio_weights,
    compute_portfolio_metrics,
    compute_turnover,
    cross_validate_signal,
    portfolio_returns_from_weights,
)


@pytest.fixture
def random_signal_and_returns():
    """Independent signal and return matrices with valid cross-sections."""
    rng = np.random.default_rng(15)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    tickers = [f"T{i:02d}" for i in range(20)]
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    signal = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    return signal, returns


class TestBuildPortfolioWeights:
    def test_known_bucket_membership_and_zero_net_exposure(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        signal = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [-5.0, -1.0, 0.0, 1.0, 5.0],
            ],
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )

        weights = build_portfolio_weights(signal, top_quantile=0.2, bottom_quantile=0.2)

        assert weights.loc[dates[0], "E"] == pytest.approx(0.5)
        assert weights.loc[dates[0], "A"] == pytest.approx(-0.5)
        assert weights.loc[dates[0], ["B", "C", "D"]].eq(0.0).all()
        assert weights.loc[dates[1], "E"] == pytest.approx(0.5)
        assert weights.loc[dates[1], "A"] == pytest.approx(-0.5)
        assert weights.sum(axis=1).tolist() == pytest.approx([0.0, 0.0])

    def test_nan_signals_are_ignored(self):
        signal = pd.DataFrame(
            [[1.0, np.nan, 3.0, 4.0, 5.0]],
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
            columns=["A", "B", "C", "D", "E"],
        )

        weights = build_portfolio_weights(signal, top_quantile=0.2, bottom_quantile=0.2)

        assert weights.loc[weights.index[0], "E"] == pytest.approx(0.5)
        assert weights.loc[weights.index[0], "A"] == pytest.approx(-0.5)
        assert weights.loc[weights.index[0], "B"] == pytest.approx(0.0)
        assert weights.sum(axis=1).iloc[0] == pytest.approx(0.0)

    def test_tied_signals_still_produce_disjoint_balanced_buckets(self):
        signal = pd.DataFrame(
            [np.ones(10)],
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
            columns=list("ABCDEFGHIJ"),
        )

        weights = build_portfolio_weights(signal, top_quantile=0.2, bottom_quantile=0.2)

        assert (weights.iloc[0] > 0).sum() == 2
        assert (weights.iloc[0] < 0).sum() == 2
        assert weights.sum(axis=1).iloc[0] == pytest.approx(0.0)

    def test_too_few_valid_names_leaves_row_uninvested(self):
        signal = pd.DataFrame(
            [[1.0, np.nan, np.nan, np.nan, np.nan]],
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
            columns=["A", "B", "C", "D", "E"],
        )

        weights = build_portfolio_weights(signal)

        assert weights.eq(0.0).all(axis=None)
        assert weights.sum(axis=1).iloc[0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("top_quantile", "bottom_quantile"),
        [(0.6, 0.4), (1.0, 0.0), (0.0, 0.2), (-0.1, 0.2)],
    )
    def test_invalid_quantiles_raise(self, top_quantile, bottom_quantile):
        signal = pd.DataFrame([[1.0, 2.0]])
        with pytest.raises(ValueError):
            build_portfolio_weights(signal, top_quantile, bottom_quantile)


class TestPortfolioReturns:
    def test_hand_computed_weighted_returns(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        weights = pd.DataFrame(
            [[0.5, -0.5], [0.5, -0.5]],
            index=dates,
            columns=["A", "B"],
        )
        forward_returns = pd.DataFrame(
            [[0.10, -0.04], [-0.02, 0.06]],
            index=dates,
            columns=["A", "B"],
        )

        result = portfolio_returns_from_weights(weights, forward_returns)

        expected = pd.Series([0.07, -0.04], index=dates)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_intersection_alignment_and_uninvested_date(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        weights = pd.DataFrame(
            [[0.5, -0.5], [0.0, 0.0]],
            index=dates,
            columns=["A", "B"],
        )
        forward_returns = pd.DataFrame(
            [[0.10, -0.04]],
            index=dates[:1],
            columns=["A", "B"],
        )

        result = portfolio_returns_from_weights(weights, forward_returns)

        assert result.index.tolist() == [dates[0]]
        assert result.iloc[0] == pytest.approx(0.07)

    def test_missing_held_return_produces_nan_not_partial_return(self):
        date = pd.Timestamp("2024-01-01")
        weights = pd.DataFrame([[0.5, -0.5]], index=[date], columns=["A", "B"])
        forward_returns = pd.DataFrame([[0.10, np.nan]], index=[date], columns=["A", "B"])

        result = portfolio_returns_from_weights(weights, forward_returns)

        assert np.isnan(result.iloc[0])


class TestPortfolioMetrics:
    def test_perfect_predictive_signal_has_positive_sharpe_and_return(self, random_signal_and_returns):
        _, returns = random_signal_and_returns
        signal = returns.copy()

        weights = build_portfolio_weights(signal)
        portfolio_returns = portfolio_returns_from_weights(weights, returns)
        metrics = compute_portfolio_metrics(portfolio_returns)

        assert set(metrics) == {
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "annual_return",
            "cagr",
            "annual_volatility",
            "total_return",
            "num_periods",
            "best_day",
            "worst_day",
        }
        assert metrics["sharpe_ratio"] > 0
        assert metrics["annual_return"] > 0
        assert metrics["num_periods"] == len(returns)

    def test_random_signal_has_sharpe_near_zero(self, random_signal_and_returns):
        signal, returns = random_signal_and_returns

        weights = build_portfolio_weights(signal)
        portfolio_returns = portfolio_returns_from_weights(weights, returns)
        metrics = compute_portfolio_metrics(portfolio_returns)

        assert abs(metrics["sharpe_ratio"]) < 0.5

    def test_sortino_uses_full_sample_textbook_downside_deviation(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02])

        metrics = compute_portfolio_metrics(returns, periods_per_year=4)

        # Hand calculation over all four observations:
        # sqrt((0^2 + 0.01^2 + 0^2 + 0.02^2) / 4) = 0.0111803398875.
        expected_downside_deviation = math.sqrt((0.0**2 + (-0.01) ** 2 + 0.0**2 + (-0.02) ** 2) / 4)
        expected_sortino = (((0.02 - 0.01 + 0.03 - 0.02) / 4) / expected_downside_deviation) * math.sqrt(4)
        assert expected_downside_deviation == pytest.approx(0.011180339887498949, abs=1e-15)
        assert expected_sortino == pytest.approx(0.8944271909999159, abs=1e-15)
        assert metrics["sortino_ratio"] == pytest.approx(expected_sortino, abs=1e-9)

    def test_cagr_for_one_year_of_constant_returns_equals_periodic_return(self):
        returns = pd.Series([0.01] * 252)

        # With one annual observation per year, geometric annualization leaves
        # the constant periodic return unchanged.
        metrics = compute_portfolio_metrics(returns, periods_per_year=1)

        expected_cagr = (1.01**252) ** (1 / 252) - 1.0
        assert metrics["cagr"] == pytest.approx(expected_cagr, abs=1e-12)
        assert metrics["cagr"] == pytest.approx(0.01, abs=1e-12)

        # With 252 daily observations per year, the same specified formula
        # compounds all 252 periods into one annual growth factor.
        daily_metrics = compute_portfolio_metrics(returns, periods_per_year=252)
        expected_daily_cagr = (1.01**252) ** (252 / 252) - 1.0
        assert daily_metrics["cagr"] == pytest.approx(expected_daily_cagr, abs=1e-12)
        assert daily_metrics["cagr"] == pytest.approx(11.274002099240246, abs=1e-12)

    def test_cagr_differs_from_arithmetic_annual_return_after_large_drawdown(self):
        returns = pd.Series([-0.50, *([0.01] * 251)])

        metrics = compute_portfolio_metrics(returns, periods_per_year=252)

        expected_final_equity = 0.50 * (1.01**251)
        expected_cagr = expected_final_equity ** (252 / 252) - 1.0
        expected_arithmetic_annual_return = returns.mean() * 252
        assert metrics["cagr"] == pytest.approx(expected_cagr, abs=1e-12)
        assert metrics["annual_return"] == pytest.approx(expected_arithmetic_annual_return, abs=1e-12)
        assert abs(metrics["cagr"] - metrics["annual_return"]) > 1.0

    def test_drawdown_and_calmar_are_hand_checkable(self):
        returns = pd.Series(
            [0.10, -0.20, 0.05],
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )

        metrics = compute_portfolio_metrics(returns, periods_per_year=252)

        # Equity is 1.10, 0.88, 0.924; its running peak is 1.10.
        assert metrics["max_drawdown"] == pytest.approx(-0.20)
        assert metrics["max_drawdown"] <= 0
        assert metrics["annual_return"] == pytest.approx(returns.mean() * 252)
        assert metrics["calmar_ratio"] == pytest.approx(metrics["annual_return"] / abs(metrics["max_drawdown"]))
        assert metrics["total_return"] == pytest.approx(-0.076)
        assert metrics["best_day"] == pytest.approx(0.10)
        assert metrics["worst_day"] == pytest.approx(-0.20)

    def test_constant_positive_series_is_treated_as_riskless(self):
        metrics = compute_portfolio_metrics(pd.Series([0.01] * 100))

        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["annual_volatility"] == 0.0
        json.dumps(metrics, allow_nan=False)

    def test_constant_negative_series_is_treated_as_riskless(self):
        metrics = compute_portfolio_metrics(pd.Series([-0.01] * 100))

        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["annual_volatility"] == 0.0
        json.dumps(metrics, allow_nan=False)

    def test_constant_series_at_risk_free_target_is_treated_as_riskless(self):
        per_period_target = 0.04 / 252

        metrics = compute_portfolio_metrics(
            pd.Series([per_period_target] * 100),
            risk_free_rate=0.04,
        )

        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["annual_volatility"] == 0.0
        assert all(math.isfinite(value) for key, value in metrics.items() if key != "num_periods")
        json.dumps(metrics, allow_nan=False)

    def test_genuinely_small_volatility_is_not_clamped(self):
        values = [0.009999, 0.010001] * 50
        returns = pd.Series(values)

        metrics = compute_portfolio_metrics(returns)

        expected_mean = sum(values) / len(values)
        expected_std = math.sqrt(sum((value - expected_mean) ** 2 for value in values) / (len(values) - 1))
        expected_sharpe = (expected_mean / expected_std) * math.sqrt(252)
        assert math.isfinite(metrics["sharpe_ratio"])
        assert metrics["sharpe_ratio"] != 0.0
        assert metrics["annual_volatility"] > 0.0
        assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-12)

    def test_normal_volatility_metrics_are_unchanged(self):
        returns = pd.Series([0.06, 0.05, 0.04, -0.05])

        metrics = compute_portfolio_metrics(returns)

        assert metrics == {
            "sharpe_ratio": 7.833494518006402,
            "sortino_ratio": 15.874507866387543,
            "max_drawdown": -0.050000000000000044,
            "calmar_ratio": 125.99999999999989,
            "annual_return": 6.3,
            "cagr": 396.08443342462664,
            "annual_volatility": 0.804238770515324,
            "total_return": 0.09964400000000029,
            "num_periods": 4,
            "best_day": 0.06,
            "worst_day": -0.05,
        }

    def test_constant_series_uses_finite_zero_risk_metrics(self):
        metrics = compute_portfolio_metrics(pd.Series([0.01] * 5))

        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["annual_volatility"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["calmar_ratio"] == 0.0
        assert metrics["annual_return"] == pytest.approx(0.01 * 252)
        assert metrics["num_periods"] == 5
        json.dumps(metrics, allow_nan=False)

    def test_cagr_is_zero_when_cumulative_equity_is_non_positive(self):
        metrics = compute_portfolio_metrics(pd.Series([-1.0]), periods_per_year=252)

        assert metrics["cagr"] == 0.0
        assert math.isfinite(metrics["cagr"])

    def test_one_period_uses_finite_zero_risk_metrics(self):
        metrics = compute_portfolio_metrics(pd.Series([0.01]))

        assert metrics["num_periods"] == 1
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["annual_volatility"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["calmar_ratio"] == 0.0
        assert all(math.isfinite(value) for key, value in metrics.items() if key != "num_periods")


class TestTurnover:
    def test_static_weights_have_zero_turnover(self):
        weights = pd.DataFrame(
            [[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]],
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
            columns=["A", "B"],
        )

        assert compute_turnover(weights, periods_per_year=1) == pytest.approx(0.0)

    def test_fully_rebalanced_weights_have_positive_turnover(self):
        weights = pd.DataFrame(
            [[0.5, -0.5], [-0.5, 0.5]],
            index=pd.date_range("2024-01-01", periods=2, freq="B"),
            columns=["A", "B"],
        )

        turnover = compute_turnover(weights, periods_per_year=1)

        assert turnover > 0
        # Full absolute-change convention: |0.5 - (-0.5)| for each of two names.
        assert turnover == pytest.approx(2.0)

    def test_insufficient_data_returns_zero(self):
        weights = pd.DataFrame([[0.5, -0.5]], columns=["A", "B"])
        assert compute_turnover(weights) == 0.0


class TestCrossValidation:
    def test_split_sizes_sum_to_aligned_date_count(self, random_signal_and_returns):
        signal, returns = random_signal_and_returns

        result = cross_validate_signal(signal, returns, train_ratio=0.6)

        assert set(result) == {
            "train_mean_ic",
            "train_p_value",
            "test_mean_ic",
            "test_p_value",
            "ic_decay",
            "is_overfit",
            "train_periods",
            "test_periods",
            "split_date",
        }
        aligned_dates = signal.index.intersection(returns.index)
        assert result["train_periods"] + result["test_periods"] == len(aligned_dates)
        assert result["train_periods"] == int(len(aligned_dates) * 0.6)
        assert result["split_date"] == aligned_dates[int(len(aligned_dates) * 0.6)].isoformat()

    @pytest.mark.parametrize("train_ratio", [0.0, 1.0, -0.2, 1.2])
    def test_invalid_train_ratio_raises(self, random_signal_and_returns, train_ratio):
        signal, returns = random_signal_and_returns
        with pytest.raises(ValueError):
            cross_validate_signal(signal, returns, train_ratio=train_ratio)

    def test_first_half_only_signal_is_flagged_as_overfit(self):
        rng = np.random.default_rng(321)
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        tickers = [f"T{i:02d}" for i in range(20)]
        returns = pd.DataFrame(
            rng.normal(size=(80, 20)),
            index=dates,
            columns=tickers,
        )
        signal = returns.copy()
        signal.iloc[40:] = rng.normal(size=(40, 20))

        result = cross_validate_signal(signal, returns, train_ratio=0.5)

        assert result["train_periods"] == 40
        assert result["test_periods"] == 40
        assert result["is_overfit"] is True

    def test_genuinely_predictive_signal_is_not_overfit(self):
        rng = np.random.default_rng(654)
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        tickers = [f"T{i:02d}" for i in range(20)]
        returns = pd.DataFrame(
            rng.normal(size=(80, 20)),
            index=dates,
            columns=tickers,
        )
        signal = returns.copy()

        result = cross_validate_signal(signal, returns, train_ratio=0.5)

        assert result["train_mean_ic"] == pytest.approx(1.0)
        assert result["test_mean_ic"] == pytest.approx(1.0)
        assert result["ic_decay"] == pytest.approx(0.0)
        assert result["is_overfit"] is False

    def test_standalone_fallback_uses_cross_sectional_ic(self, monkeypatch):
        rng = np.random.default_rng(77)
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        tickers = [f"T{i:02d}" for i in range(12)]
        returns = pd.DataFrame(
            rng.normal(size=(60, 12)),
            index=dates,
            columns=tickers,
        )
        signal = returns.copy()
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.signal_evaluator", None)

        result = cross_validate_signal(signal, returns, train_ratio=0.5)

        assert result["train_mean_ic"] == pytest.approx(1.0)
        assert result["test_mean_ic"] == pytest.approx(1.0)


class TestEdgeCases:
    def test_empty_inputs_return_documented_defaults(self):
        empty = pd.DataFrame()

        weights = build_portfolio_weights(empty)
        portfolio_returns = portfolio_returns_from_weights(weights, empty)
        metrics = compute_portfolio_metrics(pd.Series(dtype=float))
        validation = cross_validate_signal(empty, empty)

        assert weights.empty
        assert portfolio_returns.empty
        assert metrics["num_periods"] == 0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["cagr"] == 0.0
        assert validation["train_periods"] == 0
        assert validation["test_periods"] == 0
        assert validation["train_mean_ic"] is None
        assert validation["split_date"] is None
        assert validation["is_overfit"] is True

    def test_single_date_does_not_raise(self):
        date = pd.Timestamp("2024-01-01")
        tickers = [f"T{i:02d}" for i in range(12)]
        signal = pd.DataFrame([np.arange(12, dtype=float)], index=[date], columns=tickers)
        returns = pd.DataFrame([np.arange(12, dtype=float) / 100], index=[date], columns=tickers)

        weights = build_portfolio_weights(signal)
        portfolio_returns = portfolio_returns_from_weights(weights, returns)
        metrics = compute_portfolio_metrics(portfolio_returns)
        validation = cross_validate_signal(signal, returns)

        assert len(portfolio_returns) == 1
        assert math.isfinite(metrics["sharpe_ratio"])
        assert validation["train_periods"] == 0
        assert validation["test_periods"] == 1
        assert validation["is_overfit"] is True

    def test_all_nan_signal_does_not_raise(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        tickers = [f"T{i:02d}" for i in range(12)]
        signal = pd.DataFrame(np.nan, index=dates, columns=tickers)
        returns = pd.DataFrame(
            np.arange(48, dtype=float).reshape(4, 12) / 100,
            index=dates,
            columns=tickers,
        )

        weights = build_portfolio_weights(signal)
        portfolio_returns = portfolio_returns_from_weights(weights, returns)
        metrics = compute_portfolio_metrics(portfolio_returns)
        validation = cross_validate_signal(signal, returns, train_ratio=0.5)

        assert weights.eq(0.0).all(axis=None)
        assert portfolio_returns.isna().all()
        assert metrics["num_periods"] == 0
        assert metrics["sharpe_ratio"] == 0.0
        assert validation["train_mean_ic"] is None
        assert validation["test_mean_ic"] is None
        assert validation["ic_decay"] is None
        assert validation["is_overfit"] is True
