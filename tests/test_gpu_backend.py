# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU rank-IC backend (gpu_backend).

The device engine is additionally validated on CPU-only machines by running the
shared engine through the NumPy adapter and comparing against scipy.spearmanr.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from signal_discovery_workflow.gpu_backend import (
    _ArrayOps,
    _rank_ic_engine,
    compute_forward_returns_gpu,
    compute_rank_ic_batch,
    compute_rank_ic_gpu,
    get_backend,
    gpu_info,
    is_gpu_available,
)
from signal_discovery_workflow.signal_evaluator import compute_rank_ic

METRIC_KEYS = {
    "mean_ic",
    "ic_std",
    "ic_ir",
    "t_stat",
    "p_value",
    "num_periods",
    "positive_ic_ratio",
    "ic_percentiles",
}

GPU_INFO_ALLOWED_KEYS = {
    "backend",
    "device_name",
    "cuda_version",
    "cudnn_version",
    "memory_total_mb",
    "memory_free_mb",
    "available",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_pair(n_dates=80, n_stocks=40, nan_frac=0.15, seed=7):
    """Signal / forward-return pair with realistic NaN holes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_stocks)]
    signal = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=tickers)
    returns = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=tickers)
    signal = signal.mask(rng.random((n_dates, n_stocks)) < nan_frac)
    returns = returns.mask(rng.random((n_dates, n_stocks)) < nan_frac)
    return signal, returns


def _assert_metrics_close(got, expected, tol=1e-6):
    """Same key set and values within tol (num_periods exact, None respected)."""
    assert set(got) == set(expected) | {"backend"}
    assert "error" not in got
    assert got["backend"] in {"cpu", "cupy", "torch"}
    for key, value in expected.items():
        if key == "ic_percentiles":
            for pkey, pvalue in value.items():
                assert got[key][pkey] == pytest.approx(pvalue, abs=tol), f"ic_percentiles[{pkey}]"
        elif key == "num_periods":
            assert got[key] == value
        elif value is None:
            assert got[key] is None, key
        else:
            assert got[key] == pytest.approx(value, abs=tol), key


# ---------------------------------------------------------------------------
# Backend selection / detection
# ---------------------------------------------------------------------------
def test_get_backend_reports_valid_backend_and_consistency():
    backend = get_backend()
    assert backend in {"cpu", "cupy", "torch"}
    assert is_gpu_available() == (backend != "cpu")
    assert get_backend("auto") == backend
    assert get_backend("cpu") == "cpu"
    assert is_gpu_available("cpu") is False
    assert isinstance(is_gpu_available(), bool)
    with pytest.raises(ValueError):
        get_backend("tpu")
    with pytest.raises(ValueError):
        is_gpu_available("nonsense")


def test_get_backend_cpu_without_gpu_libraries():
    has_cupy = has_torch = False
    try:
        import cupy  # noqa: F401

        has_cupy = True
    except (ImportError, OSError):
        pass
    try:
        import torch  # noqa: F401

        has_torch = True
    except (ImportError, OSError):
        pass

    if not (has_cupy or has_torch):
        assert get_backend() == "cpu"
        assert is_gpu_available() is False
        assert get_backend("cupy") == "cpu"
        assert get_backend("torch") == "cpu"
        assert is_gpu_available("cupy") is False
        assert is_gpu_available("torch") is False


def test_gpu_info_shape_and_no_secret_keys():
    info = gpu_info()
    assert set(info) <= GPU_INFO_ALLOWED_KEYS
    assert isinstance(info["available"], bool)
    assert info["backend"] in {"cpu", "cupy", "torch"}
    assert info["available"] == (info["backend"] != "cpu")
    if not info["available"]:
        assert set(info) <= {"backend", "available"}


# ---------------------------------------------------------------------------
# compute_rank_ic_gpu vs the CPU reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [7, 123])
def test_compute_rank_ic_gpu_matches_cpu_random_matrix(seed):
    signal, returns = _random_pair(seed=seed)
    reference = compute_rank_ic(signal, returns)
    result = compute_rank_ic_gpu(signal, returns)
    _assert_metrics_close(result, reference)


def test_compute_rank_ic_gpu_perfect_and_inverse_correlation():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=30)
    tickers = [f"T{i:02d}" for i in range(25)]
    returns = pd.DataFrame(rng.normal(size=(30, 25)), index=dates, columns=tickers)
    signal = returns.copy()  # identical values -> every cross-sectional Spearman == 1

    reference = compute_rank_ic(signal, returns)
    result = compute_rank_ic_gpu(signal, returns)
    _assert_metrics_close(result, reference)
    assert reference["mean_ic"] == pytest.approx(1.0, abs=1e-9)
    assert result["mean_ic"] == pytest.approx(1.0, abs=1e-6)
    assert result["positive_ic_ratio"] == pytest.approx(1.0)

    inverted = compute_rank_ic_gpu(-signal, returns)
    assert inverted["mean_ic"] == pytest.approx(-1.0, abs=1e-6)
    assert inverted["num_periods"] == reference["num_periods"]


def test_compute_rank_ic_gpu_backend_cpu_matches_reference_exactly():
    signal, returns = _random_pair(seed=11)
    reference = compute_rank_ic(signal, returns)
    result = compute_rank_ic_gpu(signal, returns, backend="cpu")
    assert result["backend"] == "cpu"
    for key, value in reference.items():
        assert result[key] == value, key


# ---------------------------------------------------------------------------
# compute_forward_returns_gpu
# ---------------------------------------------------------------------------
def test_compute_forward_returns_gpu_matches_pandas():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-01", periods=40)
    tickers = [f"T{i:02d}" for i in range(15)]
    step_returns = rng.normal(0.001, 0.02, size=(40, 15))
    close = pd.DataFrame(100 * np.exp(np.cumsum(step_returns, axis=0)), index=dates, columns=tickers)
    close.iloc[3, 2] = np.nan
    close.iloc[10, 7] = np.nan

    expected = close.shift(-5) / close - 1
    for backend in ("cpu", "auto"):
        result = compute_forward_returns_gpu(close, periods=5, backend=backend)
        pd.testing.assert_frame_equal(result, expected)

    # periods beyond the data -> all NaN, same shape
    beyond = compute_forward_returns_gpu(close, periods=50, backend="cpu")
    assert beyond.shape == close.shape
    assert beyond.isna().all().all()


def test_compute_forward_returns_gpu_edge_cases():
    rng = np.random.default_rng(6)
    close = pd.DataFrame(
        rng.uniform(50.0, 150.0, size=(12, 6)),
        index=pd.bdate_range("2024-01-01", periods=12),
        columns=[f"T{i}" for i in range(6)],
    )
    # periods=0 degenerates to the pandas expression
    pd.testing.assert_frame_equal(
        compute_forward_returns_gpu(close, periods=0, backend="cpu"),
        close.shift(0) / close - 1,
    )
    # empty frame
    empty = pd.DataFrame()
    result = compute_forward_returns_gpu(empty, periods=5, backend="cpu")
    assert result.empty
    assert result.index.equals(empty.index)
    assert result.columns.equals(empty.columns)


# ---------------------------------------------------------------------------
# compute_rank_ic_batch
# ---------------------------------------------------------------------------
def test_compute_rank_ic_batch_returns_one_result_per_signal():
    signal_a, returns = _random_pair(seed=21)
    signal_b, _ = _random_pair(seed=22)
    signal_c = returns + 0.5 * signal_a
    signals = {"alpha_a": signal_a, "alpha_b": signal_b, "alpha_c": signal_c}

    batch = compute_rank_ic_batch(signals, returns)
    assert set(batch) == set(signals)

    key_sets = set()
    for name, metrics in batch.items():
        reference = compute_rank_ic(signals[name], returns)
        _assert_metrics_close(metrics, reference)
        key_sets.add(tuple(sorted(metrics)))
    assert len(key_sets) == 1  # every signal reports the same key set


# ---------------------------------------------------------------------------
# Device engine validated on CPU through the NumPy adapter
# ---------------------------------------------------------------------------
def test_rank_ic_engine_matches_scipy_reference():
    """The shared engine (also used by the CuPy/PyTorch paths) vs a scipy loop."""
    rng = np.random.default_rng(99)
    n_dates, n_stocks = 50, 30
    signal = rng.normal(size=(n_dates, n_stocks))
    returns = rng.normal(size=(n_dates, n_stocks))

    # Mix in integer-coded values so average-rank tie handling is exercised.
    tie_values = rng.integers(0, 5, size=(n_dates, n_stocks)).astype(float)
    mixed = rng.random((n_dates, n_stocks)) < 0.5
    signal = np.where(mixed, signal, tie_values)

    signal[rng.random((n_dates, n_stocks)) < 0.1] = np.nan
    returns[rng.random((n_dates, n_stocks)) < 0.1] = np.nan

    signal[4, :] = 2.0  # constant signal row -> skipped
    returns[7, :] = 1.0  # constant returns row -> skipped
    signal[11, :] = np.nan  # < 10 common valid names -> skipped
    signal[11, :3] = [1.0, 2.0, 3.0]

    (got,) = _rank_ic_engine(signal[None, :, :], returns, _ArrayOps(np))

    expected = []
    for d in range(n_dates):
        signal_row = pd.Series(signal[d]).dropna()
        returns_row = pd.Series(returns[d]).dropna()
        common = signal_row.index.intersection(returns_row.index)
        if len(common) < 10:
            continue
        signal_vals = signal_row[common].to_numpy()
        returns_vals = returns_row[common].to_numpy()
        if np.std(signal_vals) < 1e-10 or np.std(returns_vals) < 1e-10:
            continue
        rho, _ = stats.spearmanr(signal_vals, returns_vals)
        if not np.isnan(rho):
            expected.append(rho)

    assert len(expected) > 0
    assert len(got) == len(expected)
    np.testing.assert_allclose(got, np.asarray(expected), atol=1e-9)
    # The skipped dates must be exactly the ones excluded above.
    assert len(got) == n_dates - 3


def test_rank_ic_engine_batches_multiple_signals_exactly():
    rng = np.random.default_rng(31)
    n_dates, n_stocks = 25, 20
    returns = rng.normal(size=(n_dates, n_stocks))
    returns[5, :14] = np.nan
    signals = [rng.normal(size=(n_dates, n_stocks)), rng.normal(size=(n_dates, n_stocks)), returns.copy()]
    signals[1][:, 0] = np.nan

    ops = _ArrayOps(np)
    batched = _rank_ic_engine(np.stack(signals), returns, ops)
    assert len(batched) == 3
    for i, signal in enumerate(signals):
        (single,) = _rank_ic_engine(signal[None, :, :], returns, ops)
        np.testing.assert_array_equal(batched[i], single)


# ---------------------------------------------------------------------------
# Edge cases: empty / degenerate inputs must mirror the CPU implementation
# ---------------------------------------------------------------------------
def test_compute_rank_ic_gpu_empty_dataframes():
    empty = pd.DataFrame()
    reference = compute_rank_ic(empty, empty)
    for backend in ("cpu", "auto"):
        result = compute_rank_ic_gpu(empty, empty, backend=backend)
        assert result["mean_ic"] is None
        assert result["num_periods"] == 0
        assert result["backend"] in {"cpu", "cupy", "torch"}
        assert reference["mean_ic"] is None
        assert reference["num_periods"] == 0


def test_compute_rank_ic_gpu_single_date_few_names():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=1)
    tickers = [f"T{i}" for i in range(5)]  # fewer than 10 names -> no periods
    signal = pd.DataFrame(rng.normal(size=(1, 5)), index=dates, columns=tickers)
    returns = pd.DataFrame(rng.normal(size=(1, 5)), index=dates, columns=tickers)

    reference = compute_rank_ic(signal, returns)
    for backend in ("cpu", "auto"):
        result = compute_rank_ic_gpu(signal, returns, backend=backend)
        assert result["num_periods"] == 0
        assert result["mean_ic"] is None
        assert reference["num_periods"] == 0
        assert reference["mean_ic"] is None


def test_compute_rank_ic_gpu_single_date_with_enough_names():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-01-01", periods=1)
    tickers = [f"T{i:02d}" for i in range(30)]
    signal = pd.DataFrame(rng.normal(size=(1, 30)), index=dates, columns=tickers)
    returns = pd.DataFrame(rng.normal(size=(1, 30)), index=dates, columns=tickers)

    reference = compute_rank_ic(signal, returns)
    assert reference["num_periods"] == 1
    result = compute_rank_ic_gpu(signal, returns)
    _assert_metrics_close(result, reference)
    assert result["num_periods"] == 1
    assert result["ic_ir"] is None  # std of a single IC is 0
    assert result["t_stat"] is None
    assert result["p_value"] is None


def test_compute_rank_ic_gpu_all_nan_signal():
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = [f"T{i:02d}" for i in range(12)]
    signal = pd.DataFrame(np.nan, index=dates, columns=tickers)
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(size=(20, 12)), index=dates, columns=tickers)

    reference = compute_rank_ic(signal, returns)
    result = compute_rank_ic_gpu(signal, returns)
    assert result["num_periods"] == 0
    assert result["mean_ic"] is None
    assert reference["num_periods"] == 0
    assert reference["mean_ic"] is None


def test_compute_rank_ic_gpu_too_few_stocks():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = [f"T{i}" for i in range(8)]  # always below the 10-name threshold
    signal = pd.DataFrame(rng.normal(size=(20, 8)), index=dates, columns=tickers)
    returns = pd.DataFrame(rng.normal(size=(20, 8)), index=dates, columns=tickers)

    reference = compute_rank_ic(signal, returns)
    result = compute_rank_ic_gpu(signal, returns)
    assert result["num_periods"] == 0
    assert result["mean_ic"] is None
    assert reference["num_periods"] == 0
