# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for signal_evaluator pure utility functions."""

import json
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

import signal_discovery_workflow.signal_evaluator as se
from signal_discovery_workflow.signal_evaluator import (
    _forward_returns_cache,
    clear_forward_returns_cache,
    clear_operator_cache,
    compute_directional_ic,
    compute_forward_returns,
    compute_rank_ic,
    compute_signal_turnover,
    evaluate_signal,
    execute_signal_code,
    extract_code_from_response,
)

# A benign signal module used to compare the legacy and sandboxed code paths.
BENIGN_SIGNAL_CODE = (
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "def signal_momentum(Close: pd.DataFrame) -> pd.DataFrame:\n"
    "    return Close.pct_change(5)\n"
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stock_dates():
    return pd.date_range("2020-01-01", periods=100, freq="B")


@pytest.fixture
def stock_tickers():
    return [f"S{i:02d}" for i in range(20)]


@pytest.fixture
def close_df(stock_dates, stock_tickers):
    """Realistic-looking close prices (random walk, always positive)."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=(len(stock_dates), len(stock_tickers)))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=stock_dates, columns=stock_tickers)


@pytest.fixture
def stock_data(close_df, stock_dates, stock_tickers):
    rng = np.random.default_rng(1)
    volume = pd.DataFrame(
        rng.integers(100_000, 1_000_000, size=close_df.shape).astype(float),
        index=stock_dates,
        columns=stock_tickers,
    )
    return {
        "Open": close_df * 0.999,
        "Close": close_df,
        "High": close_df * 1.005,
        "Low": close_df * 0.995,
        "Volume": volume,
    }


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:
    def test_shape_preserved(self, close_df):
        result = compute_forward_returns(close_df, periods=5)
        assert result.shape == close_df.shape

    def test_last_periods_are_nan(self, close_df):
        periods = 5
        result = compute_forward_returns(close_df, periods=periods)
        assert result.iloc[-periods:].isna().all().all()

    def test_early_values_not_nan(self, close_df):
        result = compute_forward_returns(close_df, periods=5)
        assert result.iloc[:-5].notna().all().all()

    def test_return_calculation(self, stock_dates, stock_tickers):
        """5-day forward return = price[t+5]/price[t] - 1."""
        prices = pd.DataFrame(
            [[100.0, 200.0], [110.0, 180.0], [120.0, 190.0],
             [105.0, 210.0], [115.0, 195.0], [130.0, 220.0]],
            index=pd.date_range("2020-01-01", periods=6, freq="B"),
            columns=["A", "B"],
        )
        result = compute_forward_returns(prices, periods=1)
        expected_row0 = prices.iloc[1] / prices.iloc[0] - 1
        pd.testing.assert_series_equal(result.iloc[0], expected_row0, check_names=False)

    def test_periods_one(self, close_df):
        result = compute_forward_returns(close_df, periods=1)
        assert result.iloc[-1].isna().all()
        assert result.iloc[0].notna().all()


# ---------------------------------------------------------------------------
# compute_rank_ic
# ---------------------------------------------------------------------------

class TestComputeRankIC:
    def _make_aligned(self, n_dates=60, n_stocks=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        tickers = [f"S{i:02d}" for i in range(n_stocks)]
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        return dates, tickers, returns

    def test_returns_all_expected_keys(self, close_df):
        fwd = compute_forward_returns(close_df, periods=5)
        result = compute_rank_ic(close_df, fwd)
        for key in ("mean_ic", "ic_std", "ic_ir", "t_stat", "p_value", "num_periods"):
            assert key in result

    def test_perfect_positive_ic(self):
        """Signal = forward returns → IC close to 1."""
        _, _, returns = self._make_aligned()
        result = compute_rank_ic(returns, returns)
        assert result["mean_ic"] is not None
        assert result["mean_ic"] > 0.9

    def test_perfect_negative_ic(self):
        """Signal = −forward returns → IC close to −1."""
        _, _, returns = self._make_aligned()
        result = compute_rank_ic(-returns, returns)
        assert result["mean_ic"] is not None
        assert result["mean_ic"] < -0.9

    def test_random_signal_ic_near_zero(self):
        """Uncorrelated signal and returns: mean IC should be near 0."""
        dates, tickers, returns = self._make_aligned(n_dates=120, seed=99)
        rng = np.random.default_rng(7)
        signal = pd.DataFrame(rng.standard_normal(returns.shape), index=dates, columns=tickers)
        result = compute_rank_ic(signal, returns)
        assert result["mean_ic"] is not None
        assert abs(result["mean_ic"]) < 0.15

    def test_zero_t_stat_has_p_value(self):
        """A zero t-statistic is valid and should produce p-value 1.0."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        tickers = [f"S{i:02d}" for i in range(20)]
        row = np.arange(20, dtype=float)
        returns = pd.DataFrame([row, row], index=dates, columns=tickers)
        signal = pd.DataFrame([row, -row], index=dates, columns=tickers)

        result = compute_rank_ic(signal, returns)

        assert result["mean_ic"] == pytest.approx(0.0)
        assert result["t_stat"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_no_common_dates(self):
        """Disjoint date ranges → None metrics, 0 periods."""
        dates_a = pd.date_range("2020-01-01", periods=30, freq="B")
        dates_b = pd.date_range("2022-01-01", periods=30, freq="B")
        tickers = [f"S{i}" for i in range(20)]
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(rng.standard_normal((30, 20)), index=dates_a, columns=tickers)
        returns = pd.DataFrame(rng.standard_normal((30, 20)), index=dates_b, columns=tickers)
        result = compute_rank_ic(signal, returns)
        assert result["num_periods"] == 0
        assert result["mean_ic"] is None

    def test_too_few_stocks_skipped(self):
        """Fewer than 10 stocks per date → no valid IC periods."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        tickers = [f"S{i}" for i in range(5)]  # Only 5 stocks
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(rng.standard_normal((30, 5)), index=dates, columns=tickers)
        returns = pd.DataFrame(rng.standard_normal((30, 5)), index=dates, columns=tickers)
        result = compute_rank_ic(signal, returns)
        assert result["num_periods"] == 0

    def test_constant_signal_skipped(self):
        """Constant signal values have zero variance → skipped each date."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        tickers = [f"S{i}" for i in range(20)]
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(1.0, index=dates, columns=tickers)  # All constant
        returns = pd.DataFrame(rng.standard_normal((30, 20)), index=dates, columns=tickers)
        result = compute_rank_ic(signal, returns)
        assert result["num_periods"] == 0

    def test_positive_ic_ratio(self):
        """IC for a positively predictive signal should mostly be positive."""
        _, _, returns = self._make_aligned()
        result = compute_rank_ic(returns, returns)
        assert result.get("positive_ic_ratio", 0) > 0.9

    def test_ic_percentiles_present(self):
        _, _, returns = self._make_aligned()
        result = compute_rank_ic(returns, returns)
        percentiles = result.get("ic_percentiles", {})
        assert set(percentiles.keys()) == {"5th", "25th", "50th", "75th", "95th"}

    def test_partial_stock_overlap(self):
        """Only a subset of stocks shared between signal and returns."""
        dates = pd.date_range("2020-01-01", periods=40, freq="B")
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(
            rng.standard_normal((40, 25)),
            index=dates,
            columns=[f"S{i}" for i in range(25)],
        )
        returns = pd.DataFrame(
            rng.standard_normal((40, 25)),
            index=dates,
            columns=[f"S{i}" for i in range(5, 30)],  # Overlap on S05–S24 (20 stocks)
        )
        result = compute_rank_ic(signal, returns)
        assert result["num_periods"] > 0


# ---------------------------------------------------------------------------
# compute_directional_ic
# ---------------------------------------------------------------------------


class TestComputeDirectionalIC:
    _LEGACY_RANK_IC_KEYS = {
        "mean_ic",
        "ic_std",
        "ic_ir",
        "t_stat",
        "p_value",
        "num_periods",
        "positive_ic_ratio",
        "ic_percentiles",
    }

    @staticmethod
    def _make_returns(n_dates=100, n_stocks=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        tickers = [f"S{i:02d}" for i in range(n_stocks)]
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        return returns

    @staticmethod
    def _correlated_signal(returns, direction=1, noise_scale=0.2, seed=1):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(returns.shape) * noise_scale
        return returns * direction + noise

    def test_positive_signal_detects_positive_orientation(self):
        returns = self._make_returns()
        signal = self._correlated_signal(returns, direction=1)

        payload = compute_directional_ic(
            signal,
            returns,
            ic_threshold=0.03,
            p_value_threshold=0.05,
        )

        assert payload["detected_sign"] == 1
        assert payload["decision"] == "positive"
        assert payload["is_significant"] is True
        assert payload["should_invert"] is False
        assert payload["train_mean_ic"] > 0.9

    def test_negative_signal_detects_wrong_orientation(self):
        returns = self._make_returns()
        signal = self._correlated_signal(returns, direction=-1)

        payload = compute_directional_ic(
            signal,
            returns,
            ic_threshold=0.03,
            p_value_threshold=0.05,
        )

        assert payload["detected_sign"] == -1
        assert payload["decision"] == "negative"
        assert payload["is_significant"] is True
        assert payload["should_invert"] is True
        assert payload["train_mean_ic"] < -0.9

    def test_sufficient_negative_is_decision_survives_short_oos(self, caplog):
        returns = self._make_returns(n_dates=50, seed=11)
        signal = self._correlated_signal(returns, direction=-1, seed=12)

        with caplog.at_level(logging.WARNING):
            payload = compute_directional_ic(
                signal,
                returns,
                train_ratio=0.7,
                ic_threshold=0.03,
                p_value_threshold=0.05,
            )

        assert payload["train_num_periods"] == 35
        assert payload["test_num_periods"] == 15
        assert payload["decision"] == "negative"
        assert payload["decision_reason"] is None
        assert payload["detected_sign"] == -1
        assert payload["is_significant"] is True
        assert payload["should_invert"] is True
        assert payload["oos_reason"] == "insufficient_out_of_sample_periods"
        assert payload["oriented_test_mean_ic"] is None
        assert payload["oriented_test_p_value"] is None
        assert "insufficient out-of-sample periods" in caplog.text

    def test_sufficient_negative_is_decision_survives_zero_usable_oos(self, caplog):
        returns = self._make_returns(n_dates=50, seed=13)
        # Make every OOS row constant, so no OOS date is usable for rank IC.
        returns.iloc[35:] = 1.0
        signal = self._correlated_signal(returns, direction=-1, seed=14)

        with caplog.at_level(logging.WARNING):
            payload = compute_directional_ic(
                signal,
                returns,
                train_ratio=0.7,
                ic_threshold=0.03,
                p_value_threshold=0.05,
            )

        assert payload["train_num_periods"] == 35
        assert payload["test_num_periods"] == 0
        assert payload["decision"] == "negative"
        assert payload["decision_reason"] is None
        assert payload["detected_sign"] == -1
        assert payload["is_significant"] is True
        assert payload["should_invert"] is True
        assert payload["oos_reason"] == "insufficient_out_of_sample_periods"
        assert payload["oriented_test_mean_ic"] is None
        assert "insufficient out-of-sample periods" in caplog.text

    def test_inverse_mean_is_negated_and_absolute_ic_is_unchanged(self):
        returns = self._make_returns(seed=4)

        payload = compute_directional_ic(returns, returns)

        assert payload["inverse_mean_ic"] == -payload["direct_mean_ic"]
        assert payload["inverse_train_mean_ic"] == -payload["train_mean_ic"]
        assert payload["inverse_test_mean_ic"] == -payload["test_mean_ic"]
        assert abs(payload["inverse_mean_ic"]) == abs(payload["direct_mean_ic"])
        assert payload["absolute_ic"] == abs(payload["direct_mean_ic"])

    def test_oos_opposite_sign_does_not_flip_is_decision(self):
        returns = self._make_returns(n_dates=100, seed=8)
        signal = self._correlated_signal(returns, direction=-1, seed=18)
        # Replace the OOS rows with a positively oriented signal.  IS remains
        # negative while OOS is positive; the IS decision must not flip.
        oos_signal = self._correlated_signal(returns, direction=1, seed=28)
        signal.iloc[70:] = oos_signal.iloc[70:]

        payload = compute_directional_ic(
            signal,
            returns,
            train_ratio=0.7,
            ic_threshold=0.03,
            p_value_threshold=0.05,
        )

        assert payload["train_mean_ic"] < -0.9
        assert payload["test_mean_ic"] > 0.9
        assert payload["detected_sign"] == -1
        assert payload["should_invert"] is True
        # The IS-derived sign (-1) is applied to OOS, so the oriented OOS IC
        # is negative even though the direct OOS IC is positive.
        assert payload["oriented_test_mean_ic"] < -0.9

    def test_no_decision_payload_is_json_safe_and_does_not_request_inversion(self):
        returns = self._make_returns(n_dates=10, seed=9)

        payload = compute_directional_ic(returns, returns)

        assert payload["decision"] == "none"
        assert payload["decision_reason"] == "insufficient_in_sample_periods"
        assert payload["detected_sign"] == 0
        assert payload["is_significant"] is False
        assert payload["oos_reason"] == "insufficient_out_of_sample_periods"
        assert payload["should_invert"] is False
        # ``allow_nan=False`` proves that no NaN/Infinity escaped the payload.
        encoded = json.dumps(payload, allow_nan=False)
        assert json.loads(encoded) == payload

    def test_reason_variants_for_insufficient_is_and_oos_periods(self, caplog):
        returns = self._make_returns(n_dates=10, seed=17)
        signal = self._correlated_signal(returns, direction=1, seed=18)

        with caplog.at_level(logging.WARNING):
            payload = compute_directional_ic(signal, returns, train_ratio=0.7)

        assert payload["train_num_periods"] == 7
        assert payload["test_num_periods"] == 3
        assert payload["decision_reason"] == "insufficient_in_sample_periods"
        assert payload["oos_reason"] == "insufficient_out_of_sample_periods"
        assert payload["error"] is None
        assert "insufficient in-sample periods" in caplog.text
        assert "insufficient out-of-sample periods" in caplog.text

    def test_reason_variants_for_unavailable_is_and_oos_ic(self, monkeypatch, caplog):
        returns = self._make_returns(n_dates=100, seed=19)
        signal = self._correlated_signal(returns, direction=1, seed=20)
        # A non-finite correlation leaves the usable-date count intact while
        # making the aggregate IS/OOS IC unavailable.
        monkeypatch.setattr(se.stats, "spearmanr", lambda *args, **kwargs: (np.inf, None))

        with caplog.at_level(logging.WARNING):
            payload = compute_directional_ic(signal, returns, train_ratio=0.7)

        assert payload["train_num_periods"] == 70
        assert payload["test_num_periods"] == 30
        assert payload["train_mean_ic"] is None
        assert payload["test_mean_ic"] is None
        assert payload["decision_reason"] == "unavailable_in_sample_ic"
        assert payload["oos_reason"] == "unavailable_out_of_sample_ic"
        assert payload["error"] is None
        assert "in-sample mean IC is unavailable" in caplog.text
        assert "OOS mean IC is unavailable" in caplog.text

    def test_reason_variants_for_zero_is_ic_and_no_is_decision(self, caplog):
        returns = self._make_returns(n_dates=100, seed=21)
        returns.iloc[:, :] = np.arange(1.0, 21.0)
        # This permutation has Spearman correlation exactly 0.0 against ranks
        # 1..20, so every usable IS date has zero IC.
        zero_ic_permutation = np.array(
            [19, 10, 14, 4, 9, 8, 6, 13, 3, 20, 1, 18, 12, 2, 17, 5, 15, 16, 7, 11],
            dtype=float,
        )
        signal = pd.DataFrame(
            np.tile(zero_ic_permutation, (len(returns), 1)),
            index=returns.index,
            columns=returns.columns,
        )

        with caplog.at_level(logging.WARNING):
            payload = compute_directional_ic(signal, returns, train_ratio=0.7)

        assert payload["train_mean_ic"] == 0.0
        assert payload["decision_reason"] == "zero_in_sample_ic"
        assert payload["oos_reason"] == "no_in_sample_decision"
        assert payload["error"] is None
        assert "in-sample mean IC is zero" in caplog.text
        assert "no in-sample decision" in caplog.text

    def test_populated_real_payload_contains_only_plain_json_scalars(self):
        returns = self._make_returns(n_dates=100, seed=15)
        signal = self._correlated_signal(returns, direction=-1, seed=16)

        payload = compute_directional_ic(
            signal,
            returns,
            ic_threshold=0.03,
            p_value_threshold=0.05,
        )

        assert payload["train_mean_ic"] is not None
        assert payload["train_p_value"] is not None
        assert payload["test_mean_ic"] is not None
        assert payload["oriented_test_mean_ic"] is not None

        def leaves(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from leaves(child)
            else:
                yield value

        allowed_scalar_types = {str, int, float, bool, type(None)}
        assert all(type(leaf) in allowed_scalar_types for leaf in leaves(payload))
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload

    def test_compute_rank_ic_legacy_key_set_is_exact(self):
        returns = self._make_returns(n_dates=40, seed=10)

        result = compute_rank_ic(returns, returns)

        assert set(result) == self._LEGACY_RANK_IC_KEYS


# ---------------------------------------------------------------------------
# extract_code_from_response
# ---------------------------------------------------------------------------

class TestExtractCodeFromResponse:
    def test_python_fenced_block(self):
        response = "Some text\n```python\nx = 1\n```\nmore text"
        assert extract_code_from_response(response) == "x = 1\n"

    def test_generic_fenced_block(self):
        response = "```\ny = 2\n```"
        assert extract_code_from_response(response) == "y = 2\n"

    def test_python_block_takes_precedence(self):
        response = "```\nraw\n```\n```python\npython_code\n```"
        assert extract_code_from_response(response) == "python_code\n"

    def test_multiple_python_blocks_joined(self):
        response = "```python\npart1\n```\n```python\npart2\n```"
        result = extract_code_from_response(response)
        assert "part1" in result
        assert "part2" in result

    def test_raw_code_returned_as_is(self):
        raw = "def f():\n    return 1"
        assert extract_code_from_response(raw) == raw

    def test_empty_string(self):
        assert extract_code_from_response("") == ""


# ---------------------------------------------------------------------------
# execute_signal_code
# ---------------------------------------------------------------------------

class TestExecuteSignalCode:
    def test_basic_momentum_signal(self, stock_data):
        code = (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "def signal_momentum(Close: pd.DataFrame) -> pd.DataFrame:\n"
            "    return Close.pct_change(5)\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is not None
        df, name = result
        assert isinstance(df, pd.DataFrame)
        assert name == "signal_momentum"

    def test_returns_dataframe_not_series(self, stock_data):
        code = (
            "import pandas as pd\n\n"
            "def signal_mean(Close: pd.DataFrame) -> pd.DataFrame:\n"
            "    return Close.pct_change(3)\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is not None
        df, _ = result
        assert isinstance(df, pd.DataFrame)

    def test_no_signal_functions_returns_none(self, stock_data):
        """Code with only imports and no callable signals → None."""
        code = "import pandas as pd\nimport numpy as np\n"
        result = execute_signal_code(code, stock_data)
        assert result is None

    def test_syntax_error_returns_none(self, stock_data):
        code = "def signal_broken(Close):\n    return Close.pct_change(5\n"
        result = execute_signal_code(code, stock_data)
        assert result is None

    def test_runtime_error_returns_none(self, stock_data):
        """Signal function that raises at call time → None."""
        code = (
            "def signal_crash(Close):\n"
            "    raise ValueError('intentional')\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is None

    def test_multiple_signals_returns_best(self, stock_data):
        """When two signals are provided, the one with higher |IC| is selected."""
        code = (
            "import pandas as pd\n\n"
            "def signal_alpha(Close):\n"
            "    return Close.pct_change(5)\n\n"
            "def signal_beta(Close):\n"
            "    return Close.pct_change(10)\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is not None
        _, name = result
        assert name in ("signal_alpha", "signal_beta")

    def test_selection_periods_controls_best_signal(self, stock_data):
        code = (
            "import pandas as pd\n\n"
            "def signal_one_day(Close):\n"
            "    return Close.shift(-1) / Close - 1\n\n"
            "def signal_five_day(Close):\n"
            "    return Close.shift(-5) / Close - 1\n"
        )

        one_day = execute_signal_code(code, stock_data, selection_periods=1)
        five_day = execute_signal_code(code, stock_data, selection_periods=5)

        assert one_day is not None
        assert five_day is not None
        assert one_day[1] == "signal_one_day"
        assert five_day[1] == "signal_five_day"

    def test_operator_names_not_treated_as_signals(self, stock_data):
        """Known operator names like TS_Return should not be returned as the signal."""
        code = (
            "import pandas as pd\n\n"
            "def TS_Return(x, d):\n"
            "    return x.pct_change(d)\n\n"
            "def signal_real(Close):\n"
            "    return TS_Return(Close, 5)\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is not None
        _, name = result
        assert name == "signal_real"

    def test_uses_volume_data(self, stock_data):
        """Signal using Volume parameter should be called with volume data."""
        code = (
            "import pandas as pd\n\n"
            "def signal_volume_change(Volume):\n"
            "    return Volume.pct_change(5)\n"
        )
        result = execute_signal_code(code, stock_data)
        assert result is not None
        df, name = result
        assert name == "signal_volume_change"
        assert df.shape == stock_data["Volume"].shape


# ---------------------------------------------------------------------------
# forward-returns cache
# ---------------------------------------------------------------------------

class TestForwardReturnsCache:
    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        """Isolate each test from cache state left by other tests."""
        clear_forward_returns_cache()
        yield
        clear_forward_returns_cache()

    def test_same_data_reuses_single_entry(self, close_df):
        """Two calls with identical data → exactly one cache entry."""
        first = compute_forward_returns(close_df, periods=5)
        second = compute_forward_returns(close_df, periods=5)
        assert len(_forward_returns_cache) == 1
        pd.testing.assert_frame_equal(first, second)

    def test_hit_avoids_recomputation(self, close_df, monkeypatch):
        """On a cache hit the frame is not recomputed (no shift call)."""
        calls = {"n": 0}
        orig_shift = pd.DataFrame.shift

        def counting_shift(self, *args, **kwargs):
            calls["n"] += 1
            return orig_shift(self, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "shift", counting_shift)

        compute_forward_returns(close_df, periods=3)
        assert calls["n"] == 1  # miss → computed once

        compute_forward_returns(close_df, periods=3)
        assert calls["n"] == 1  # hit → no recomputation

    def test_cache_hit_values_identical_to_direct_math(self, close_df):
        first = compute_forward_returns(close_df, periods=7)
        expected = close_df.shift(-7) / close_df - 1
        pd.testing.assert_frame_equal(first, expected)

    def test_different_periods_miss(self, close_df):
        compute_forward_returns(close_df, periods=5)
        compute_forward_returns(close_df, periods=10)
        assert len(_forward_returns_cache) == 2

    def test_different_data_does_not_reuse_entry(self, close_df):
        """Same labels/shape but changed values → distinct cache entry."""
        fwd_a = compute_forward_returns(close_df, periods=5)
        close_b = close_df.copy()
        close_b.iloc[0, 0] = close_b.iloc[0, 0] * 1.01
        fwd_b = compute_forward_returns(close_b, periods=5)

        assert len(_forward_returns_cache) == 2
        # The changed price alters row 0's forward return.
        assert not np.isclose(fwd_a.iloc[0, 0], fwd_b.iloc[0, 0])
        # Original entry is intact and still served for the original data.
        pd.testing.assert_frame_equal(fwd_a, compute_forward_returns(close_df, periods=5))

    def test_clear_resets_cache(self, close_df):
        compute_forward_returns(close_df, periods=5)
        assert len(_forward_returns_cache) == 1
        clear_forward_returns_cache()
        assert len(_forward_returns_cache) == 0
        # Next call recomputes and repopulates the cache.
        compute_forward_returns(close_df, periods=5)
        assert len(_forward_returns_cache) == 1


# ---------------------------------------------------------------------------
# compute_signal_turnover
# ---------------------------------------------------------------------------

class TestSignalTurnover:
    def test_constant_signal_zero_turnover(self, stock_dates, stock_tickers):
        signal = pd.DataFrame(1.0, index=stock_dates, columns=stock_tickers)
        assert compute_signal_turnover(signal) == 0.0

    def test_unchanging_signal_zero_turnover(self, stock_dates, stock_tickers):
        """Same (non-constant) cross-section repeated every day → 0 turnover."""
        rng = np.random.default_rng(3)
        row = rng.standard_normal(len(stock_tickers))
        signal = pd.DataFrame(
            np.tile(row, (len(stock_dates), 1)), index=stock_dates, columns=stock_tickers
        )
        assert compute_signal_turnover(signal) == pytest.approx(0.0)

    def test_flipping_signal_positive_turnover(self, stock_dates, stock_tickers):
        """Signal that swaps the top two stocks daily → strictly positive turnover."""
        n = len(stock_dates)
        data = np.zeros((n, len(stock_tickers)))
        data[:, 0] = np.tile([1.0, -1.0], n // 2 + 1)[:n]
        data[:, 1] = np.tile([-1.0, 1.0], n // 2 + 1)[:n]
        signal = pd.DataFrame(data, index=stock_dates, columns=stock_tickers)
        assert compute_signal_turnover(signal) > 0

    def test_annualization(self):
        """Two-stock flipping signal: per-period turnover = 0.5 → 0.5 * periods_per_year."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        vals = [[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]
        signal = pd.DataFrame(vals, index=idx, columns=["A", "B"])
        assert compute_signal_turnover(signal, periods_per_year=252) == pytest.approx(0.5 * 252)
        assert compute_signal_turnover(signal, periods_per_year=12) == pytest.approx(0.5 * 12)
        assert compute_signal_turnover(signal) == pytest.approx(0.5 * 252)  # default

    def test_empty_input_zero(self):
        assert compute_signal_turnover(pd.DataFrame()) == 0.0

    def test_single_row_insufficient_data(self):
        signal = pd.DataFrame([[1.0, 2.0]], index=pd.date_range("2020-01-01", periods=1), columns=["A", "B"])
        assert compute_signal_turnover(signal) == 0.0


# ---------------------------------------------------------------------------
# evaluate_signal
# ---------------------------------------------------------------------------

class TestEvaluateSignal:
    def _make_inputs(self, n_dates=60, n_stocks=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        tickers = [f"S{i:02d}" for i in range(n_stocks)]
        signal = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        return signal, returns

    def test_contains_rank_ic_keys_and_turnover(self):
        signal, returns = self._make_inputs()
        result = evaluate_signal(signal, returns)
        for key in ("mean_ic", "ic_std", "ic_ir", "t_stat", "p_value", "num_periods", "turnover"):
            assert key in result
        assert isinstance(result["turnover"], float)
        assert result["turnover"] >= 0.0

    def test_rank_ic_part_matches_compute_rank_ic(self):
        signal, returns = self._make_inputs(seed=5)
        result = evaluate_signal(signal, returns)
        direct = compute_rank_ic(signal, returns)
        for key, value in direct.items():
            assert result[key] == value

    def test_missing_quant_metrics_does_not_raise(self, monkeypatch):
        """quant_metrics import failure → portfolio keys omitted, no exception."""
        signal, returns = self._make_inputs()
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.quant_metrics", None)
        result = evaluate_signal(signal, returns)  # must not raise
        assert "mean_ic" in result
        assert "turnover" in result

    def test_portfolio_metrics_merged_from_quant_metrics(self, monkeypatch):
        """A functional quant_metrics module has its metrics merged into the result."""
        signal, returns = self._make_inputs()

        fake = types.ModuleType("signal_discovery_workflow.quant_metrics")
        # Signature mirrors the real quant_metrics.build_portfolio_weights
        # (no forward_returns parameter).
        fake.build_portfolio_weights = lambda sv, top_quantile=0.2, bottom_quantile=0.2: sv
        fake.portfolio_returns_from_weights = lambda w, fr: pd.Series(0.001, index=w.index)
        fake.compute_portfolio_metrics = lambda r, periods_per_year=252, risk_free_rate=0.0: {
            "sharpe": 1.23,
            "annual_return": 0.1,
        }
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.quant_metrics", fake)

        result = evaluate_signal(signal, returns)
        assert result["sharpe"] == 1.23
        assert result["annual_return"] == 0.1
        assert "turnover" in result

    def test_broken_quant_metrics_does_not_raise(self, monkeypatch, caplog):
        """A partial/buggy quant_metrics must never break the core evaluation."""
        signal, returns = self._make_inputs()

        fake = types.ModuleType("signal_discovery_workflow.quant_metrics")

        def boom(*args, **kwargs):
            raise RuntimeError("half-implemented")

        fake.build_portfolio_weights = boom
        fake.portfolio_returns_from_weights = boom
        fake.compute_portfolio_metrics = boom
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.quant_metrics", fake)

        with caplog.at_level(logging.WARNING, logger="signal_discovery_workflow.signal_evaluator"):
            result = evaluate_signal(signal, returns)  # must not raise

        assert "mean_ic" in result
        assert "turnover" in result
        assert "sharpe" not in result
        # Integration failures must be visible at WARNING, not silent DEBUG.
        warning_messages = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("portfolio" in r.getMessage().lower() for r in warning_messages)

    def test_real_quant_metrics_produces_portfolio_keys(self, close_df):
        """Regression: the quant_metrics integration must actually produce metrics.

        Guards against a silent contract mismatch (e.g. calling
        ``build_portfolio_weights`` with an argument it does not accept), which
        previously degraded to rank-IC + turnover only with no failing test.
        """
        signal = close_df.pct_change(5, fill_method=None)  # simple momentum signal
        forward_returns = compute_forward_returns(close_df, periods=5)

        result = evaluate_signal(signal, forward_returns)

        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown"):
            assert key in result, f"portfolio metric {key!r} missing — integration broken?"
            assert isinstance(result[key], float)
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]!r}"

        # Rank-IC statistics must never be clobbered by portfolio keys.
        assert result["num_periods"] == compute_rank_ic(signal, forward_returns)["num_periods"]


# ---------------------------------------------------------------------------
# sandbox integration (off by default)
# ---------------------------------------------------------------------------

class TestSandboxIntegration:
    def test_default_is_off(self):
        """No env var → legacy path (module flag default False)."""
        assert se.SANDBOX_ENABLED is False
        assert se._sandbox_enabled() is False

    def test_sandbox_off_legacy_path_executes(self, stock_data):
        """Default OFF: the legacy path still executes and returns a result."""
        result = execute_signal_code(BENIGN_SIGNAL_CODE, stock_data)
        assert result is not None
        df, name = result
        assert name == "signal_momentum"
        assert isinstance(df, pd.DataFrame)

    def test_sandbox_on_benign_matches_legacy(self, stock_data, monkeypatch):
        """Sandbox ON: same contract as the legacy path for benign code."""
        legacy = execute_signal_code(BENIGN_SIGNAL_CODE, stock_data)
        assert legacy is not None

        monkeypatch.setattr(se, "SANDBOX_ENABLED", True)
        sandboxed = execute_signal_code(BENIGN_SIGNAL_CODE, stock_data)

        assert sandboxed is not None
        df, name = sandboxed
        assert name == legacy[1]
        pd.testing.assert_frame_equal(df, legacy[0])

    def test_sandbox_on_malicious_returns_none_no_side_effect(self, stock_data, monkeypatch, tmp_path):
        """Sandbox ON: malicious code → None, no filesystem side effect."""
        marker = tmp_path / "pwned.txt"
        malicious = (
            "import os\n"
            f"os.system('touch {marker}')\n"
            "import pandas as pd\n\n"
            "def signal_evil(Close):\n"
            "    return Close.pct_change(5)\n"
        )
        monkeypatch.setattr(se, "SANDBOX_ENABLED", True)
        result = execute_signal_code(malicious, stock_data)
        assert result is None
        assert not marker.exists()

    def test_env_override_enables_and_disables(self, monkeypatch):
        """SIGNAL_SANDBOX_ENABLED=1 → enabled; =0 / unset → disabled."""
        monkeypatch.setenv("SIGNAL_SANDBOX_ENABLED", "1")
        assert se._sandbox_enabled() is True
        monkeypatch.setenv("SIGNAL_SANDBOX_ENABLED", "TRUE")  # case-insensitive
        assert se._sandbox_enabled() is True
        monkeypatch.setenv("SIGNAL_SANDBOX_ENABLED", "0")
        assert se._sandbox_enabled() is False
        monkeypatch.setenv("SIGNAL_SANDBOX_ENABLED", "false")
        assert se._sandbox_enabled() is False
        monkeypatch.delenv("SIGNAL_SANDBOX_ENABLED")
        assert se._sandbox_enabled() is False  # module flag default

    def test_sandbox_module_missing_falls_back_to_legacy(self, stock_data, monkeypatch):
        """Sandbox enabled but module missing → WARNING + legacy path, no crash."""
        monkeypatch.setattr(se, "SANDBOX_ENABLED", True)
        monkeypatch.setitem(sys.modules, "signal_discovery_workflow.sandbox", None)
        result = execute_signal_code(BENIGN_SIGNAL_CODE, stock_data)  # must not raise
        assert result is not None
        assert result[1] == "signal_momentum"


# ---------------------------------------------------------------------------
# GPU backend integration (off by default)
# ---------------------------------------------------------------------------

class TestGpuBackendIntegration:
    _LEGACY_KEYS = {
        "mean_ic", "ic_std", "ic_ir", "t_stat", "p_value",
        "num_periods", "positive_ic_ratio", "ic_percentiles",
    }

    def _make_inputs(self, n_dates=60, n_stocks=20, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        tickers = [f"S{i:02d}" for i in range(n_stocks)]
        signal = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)), index=dates, columns=tickers)
        return signal, returns

    def test_gpu_off_no_backend_key(self):
        """GPU OFF (default): result has NO "backend" key — legacy shape intact."""
        signal, returns = self._make_inputs()
        result = compute_rank_ic(signal, returns)
        assert "backend" not in result
        assert set(result.keys()) == self._LEGACY_KEYS

    def test_gpu_off_env_zero_no_backend_key(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_USE_GPU", "0")
        signal, returns = self._make_inputs()
        result = compute_rank_ic(signal, returns)
        assert "backend" not in result

    def test_gpu_on_backend_key_and_matches_cpu(self, monkeypatch):
        """GPU ON: extra "backend" key; stats match the CPU reference within 1e-9."""
        signal, returns = self._make_inputs()
        cpu = compute_rank_ic(signal, returns)

        monkeypatch.setenv("SIGNAL_USE_GPU", "1")
        gpu = compute_rank_ic(signal, returns)

        assert "backend" in gpu
        assert gpu["mean_ic"] == pytest.approx(cpu["mean_ic"], abs=1e-9)
        assert gpu["ic_std"] == pytest.approx(cpu["ic_std"], abs=1e-9)
        assert gpu["num_periods"] == cpu["num_periods"]

    def test_gpu_on_module_flag(self, monkeypatch):
        signal, returns = self._make_inputs()
        monkeypatch.setattr(se, "USE_GPU_BACKEND", True)
        result = compute_rank_ic(signal, returns)
        assert "backend" in result

    def test_gpu_backend_failure_falls_back_to_cpu(self, monkeypatch):
        """A raising GPU backend → WARNING + CPU fallback, still a valid result."""
        import signal_discovery_workflow.gpu_backend as gpu_mod

        def boom(*args, **kwargs):
            raise RuntimeError("gpu exploded")

        monkeypatch.setenv("SIGNAL_USE_GPU", "1")
        monkeypatch.setattr(gpu_mod, "compute_rank_ic_gpu", boom)
        signal, returns = self._make_inputs()

        result = compute_rank_ic(signal, returns)  # must not raise
        assert "mean_ic" in result
        assert "backend" not in result  # legacy CPU dict, no GPU key

    def test_gpu_backend_env_selector(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_GPU_BACKEND", "torch")
        assert se._gpu_backend_name() == "torch"
        monkeypatch.setenv("SIGNAL_GPU_BACKEND", "AUTO")
        assert se._gpu_backend_name() == "auto"
        monkeypatch.delenv("SIGNAL_GPU_BACKEND")
        assert se._gpu_backend_name() == "auto"  # module flag default


# ---------------------------------------------------------------------------
# operator cache integration (off by default)
# ---------------------------------------------------------------------------

# The signal returns 1.0 everywhere when its globals hold a wrapped operator,
# 0.0 when it sees the original function — an observable window into the
# namespace that execute_signal_code does not otherwise expose.
_PROBE_OPERATOR_CODE = (
    "import pandas as pd\n\n"
    "def TS_Return(x, d):\n"
    "    return x.pct_change(d)\n\n"
    "def signal_probe(Close):\n"
    "    wrapped = getattr(TS_Return, '_operator_cache_wrapped', False)\n"
    "    return Close * 0.0 + (1.0 if wrapped else 0.0)\n"
)


class TestOperatorCacheIntegration:
    def test_flag_off_namespace_untouched(self, stock_data):
        """Flag OFF (default): original operator objects, no cache activity."""
        clear_operator_cache()
        result = execute_signal_code(_PROBE_OPERATOR_CODE, stock_data)
        assert result is not None
        df, _ = result
        # 0.0 everywhere → the signal function saw the ORIGINAL operator.
        assert float(df.to_numpy().max()) == 0.0
        stats = se._OPERATOR_CACHE.stats()
        assert stats["entries"] == 0
        assert stats["misses"] == 0

    def test_flag_on_wraps_operator_in_namespace(self, stock_data, monkeypatch):
        """Flag ON: the signal function's globals hold the memoized wrapper."""
        monkeypatch.setattr(se, "USE_OPERATOR_CACHE", True)
        clear_operator_cache()
        result = execute_signal_code(_PROBE_OPERATOR_CODE, stock_data)
        assert result is not None
        df, _ = result
        # 1.0 everywhere → the in-place wrapping is visible to the signal.
        assert float(df.to_numpy().max()) == 1.0

    def test_flag_on_cache_hits_across_executions(self, stock_data, monkeypatch):
        """Flag ON: same operator call in a second execution → cache HIT."""
        code = (
            "import pandas as pd\n\n"
            "def TS_Return(x, d):\n"
            "    return x.pct_change(d)\n\n"
            "def signal_a(Close):\n"
            "    return TS_Return(Close, 5)\n"
        )
        clear_operator_cache()
        monkeypatch.setattr(se, "USE_OPERATOR_CACHE", True)

        first = execute_signal_code(code, stock_data)
        assert first is not None
        stats_after_first = se._OPERATOR_CACHE.stats()
        assert stats_after_first["misses"] >= 1
        assert stats_after_first["hits"] == 0

        second = execute_signal_code(code, stock_data)
        assert second is not None
        stats_after_second = se._OPERATOR_CACHE.stats()
        # The concrete proof the cached computation was reused:
        assert stats_after_second["hits"] >= 1
        assert stats_after_second["hits"] > stats_after_first["hits"]
        # Same signal selected, identical values.
        assert second[1] == first[1]
        pd.testing.assert_frame_equal(second[0], first[0])

    def test_different_implementations_same_name_no_shared_entry(self, stock_data, monkeypatch):
        """Regression: same operator NAME, different implementations → no reuse."""
        code_a = (
            "import pandas as pd\n\n"
            "def TS_Return(x, d):\n"
            "    return x * 2.0\n\n"
            "def signal_a(Close):\n"
            "    return TS_Return(Close, 1)\n"
        )
        code_b = (
            "import pandas as pd\n\n"
            "def TS_Return(x, d):\n"
            "    return x + 1.0\n\n"
            "def signal_b(Close):\n"
            "    return TS_Return(Close, 1)\n"
        )
        clear_operator_cache()
        monkeypatch.setattr(se, "USE_OPERATOR_CACHE", True)

        result_a = execute_signal_code(code_a, stock_data)
        result_b = execute_signal_code(code_b, stock_data)
        assert result_a is not None and result_b is not None

        stats = se._OPERATOR_CACHE.stats()
        assert stats["hits"] == 0  # implementations differ → never share an entry
        assert stats["misses"] >= 2
        # And the two modules genuinely computed different things.
        assert result_a[1] != result_b[1]
        assert not result_a[0].equals(result_b[0])

    def test_clear_operator_cache_resets_stats(self):
        clear_operator_cache()
        se._OPERATOR_CACHE.put(("some", "key"), object())
        assert se._OPERATOR_CACHE.stats()["entries"] == 1
        clear_operator_cache()
        stats = se._OPERATOR_CACHE.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entries"] == 0

    def test_sandbox_path_does_not_touch_operator_cache(self, stock_data, monkeypatch):
        """Sandbox enabled → cache not consulted, no wrappers, call succeeds."""
        clear_operator_cache()
        monkeypatch.setenv("SIGNAL_SANDBOX_ENABLED", "1")
        monkeypatch.setattr(se, "USE_OPERATOR_CACHE", True)
        stats_before = se._OPERATOR_CACHE.stats()

        result = execute_signal_code(BENIGN_SIGNAL_CODE, stock_data)

        assert result is not None
        assert se._OPERATOR_CACHE.stats() == stats_before

    def test_env_flag_and_max_entries_overrides(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_USE_OPERATOR_CACHE", "1")
        assert se._operator_cache_enabled() is True
        monkeypatch.setenv("SIGNAL_USE_OPERATOR_CACHE", "0")
        assert se._operator_cache_enabled() is False
        monkeypatch.delenv("SIGNAL_USE_OPERATOR_CACHE")
        assert se._operator_cache_enabled() is False  # module flag default

        monkeypatch.setenv("SIGNAL_OPERATOR_CACHE_MAX_ENTRIES", "7")
        assert se._operator_cache_max_entries() == 7
        monkeypatch.setenv("SIGNAL_OPERATOR_CACHE_MAX_ENTRIES", "bogus")
        assert se._operator_cache_max_entries() == se.OPERATOR_CACHE_MAX_ENTRIES
        monkeypatch.setenv("SIGNAL_OPERATOR_CACHE_MAX_ENTRIES", "-3")
        assert se._operator_cache_max_entries() == se.OPERATOR_CACHE_MAX_ENTRIES
        monkeypatch.delenv("SIGNAL_OPERATOR_CACHE_MAX_ENTRIES")
        assert se._operator_cache_max_entries() == se.OPERATOR_CACHE_MAX_ENTRIES
