# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-accelerated Rank IC backend with a fully automatic, dependency-free CPU fallback.

This module reproduces the cross-sectional rank-IC statistics of
``signal_evaluator.compute_rank_ic`` on accelerators (NVIDIA CUDA via CuPy, or
PyTorch) while staying optional at install time: when no GPU stack is
importable, every function transparently falls back to the existing CPU
implementation, so the project keeps working unchanged on CPU-only machines.

Optional accelerator installs (never hard dependencies)::

    pip install cupy-cuda12x        # CUDA 12.x
    pip install cupy-cuda13x        # CUDA 13.x — for NVIDIA Blackwell GPUs and
                                    # newer CUDA 13 toolkits use the CUDA 13 wheels
    pip install torch --index-url https://download.pytorch.org/whl/cu124

Backend selection (``backend="auto"``) prefers CuPy when it is importable and a
CUDA device is present, then PyTorch when it is importable with CUDA available,
and otherwise resolves to ``"cpu"``. Explicit ``backend="cupy"`` /
``backend="torch"`` requests degrade to ``"cpu"`` when the requested stack is
unusable; ``backend="cpu"`` always runs the reference implementation.

The device kernel computes, per date and fully vectorised across dates (and
across signals in :func:`compute_rank_ic_batch`):

1. a joint finite mask over signal values and forward returns: entries that are
   NaN, ``+inf`` or ``-inf`` are treated as invalid on the device path (the CPU
   reference drops only NaN — see the known-divergence note below);
2. average ranks (scipy ``rankdata`` semantics, ties included): invalid entries
   are replaced with ``+inf`` before sorting, so every valid entry receives
   exactly the rank it would have within the valid subset alone;
3. a masked Pearson correlation of the two rank vectors, which is the
   cross-sectional Spearman rank IC.

Dates with fewer than 10 common valid names, dates where either side has a
cross-sectional std below 1e-10 (constant rows) and non-finite correlations are
skipped, mirroring the CPU implementation's rules for such dates. The resulting
per-date IC vector is aggregated on the host with the same NumPy/SciPy formulas
(mean, std, ICIR, t-stat, two-sided Student-t p-value with
``df = num_periods - 1``, positive ratio and percentiles).

Known divergence from the CPU reference
---------------------------------------

For inputs whose non-finite entries are NaN only, the device path matches
``signal_evaluator.compute_rank_ic`` up to floating-point noise (mean IC
difference < 1e-6 on identical inputs). The paths **diverge for ±inf inputs**:
the device engine masks every non-finite entry (``isfinite``), while the CPU
reference follows ``dropna()`` semantics, which keep ±inf and let
``scipy.stats.spearmanr`` rank them. Observable consequence: ``num_periods`` can
differ, and with it ``mean_ic`` — e.g. one date holding 9 finite + 3 ``+inf``
signal values (all other dates finite/NaN) yields ``num_periods = 24`` on the
device path versus ``num_periods = 25`` on the CPU path, with a different
``mean_ic``. This masking difference is intentional and documented here, not
"fixed". NaN-only inputs behave identically on both paths.

Verification note: the device code path was verified through the NumPy adapter
of the shared engine — the exact code the CuPy/PyTorch wrappers execute —
because no NVIDIA GPU was available in the verification environment; the real
CUDA execution path was NOT exercised.
"""

import logging
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "compute_forward_returns_gpu",
    "compute_rank_ic_batch",
    "compute_rank_ic_gpu",
    "get_backend",
    "gpu_info",
    "is_gpu_available",
]

_VALID_BACKENDS = ("auto", "cupy", "torch", "cpu")

# Cross-sectional std below which a date row is treated as constant and skipped
# (mirrors the threshold in signal_evaluator.compute_rank_ic).
_MIN_CROSS_SECTIONAL_STD = 1e-10

# Minimum number of common valid names per date required to compute an IC.
_MIN_NAMES_PER_DATE = 10

# Cached backend probes: module objects and usability flags.
_BACKEND_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lazy, optional backend detection
# ---------------------------------------------------------------------------
def _validate_backend_arg(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")


def _load_cupy() -> Any:
    """Return the cupy module or None. Imported lazily; never a hard dependency."""
    if "cupy_module" not in _BACKEND_CACHE:
        try:
            import cupy as cp
        except (ImportError, OSError):
            cp = None
        _BACKEND_CACHE["cupy_module"] = cp
    return _BACKEND_CACHE["cupy_module"]


def _load_torch() -> Any:
    """Return the torch module or None. Imported lazily; never a hard dependency."""
    if "torch_module" not in _BACKEND_CACHE:
        try:
            import torch as th
        except (ImportError, OSError):
            th = None
        _BACKEND_CACHE["torch_module"] = th
    return _BACKEND_CACHE["torch_module"]


def _cupy_usable() -> bool:
    """True when cupy is importable AND at least one CUDA device is present."""
    if "cupy_ok" not in _BACKEND_CACHE:
        usable = False
        cupy = _load_cupy()
        if cupy is not None:
            try:
                usable = int(cupy.cuda.runtime.getDeviceCount()) > 0
            except Exception:  # noqa: BLE001 — any CUDA/driver failure means unusable
                usable = False
        _BACKEND_CACHE["cupy_ok"] = usable
    return bool(_BACKEND_CACHE["cupy_ok"])


def _torch_usable() -> bool:
    """True when torch is importable AND a CUDA device is available."""
    if "torch_ok" not in _BACKEND_CACHE:
        usable = False
        torch = _load_torch()
        if torch is not None:
            try:
                usable = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
            except Exception:  # noqa: BLE001 — any CUDA/driver failure means unusable
                usable = False
        _BACKEND_CACHE["torch_ok"] = usable
    return bool(_BACKEND_CACHE["torch_ok"])


def is_gpu_available(backend: str = "auto") -> bool:
    """Whether a usable accelerator backend is importable AND a device is present.

    Args:
        backend: One of ``"auto"``, ``"cupy"``, ``"torch"``, ``"cpu"``.
            ``"cpu"`` always returns False.

    Returns:
        True only when the selected backend resolves to a working GPU stack.
    """
    _validate_backend_arg(backend)
    if backend == "cpu":
        return False
    return get_backend(backend) != "cpu"


def get_backend(backend: str = "auto") -> str:
    """Resolve a backend request to an actually usable backend name.

    Args:
        backend: ``"auto"`` picks cupy if importable with a device present, else
            torch if importable with CUDA available, else ``"cpu"``. An explicit
            ``"cupy"``/``"torch"`` request degrades to ``"cpu"`` when unusable.

    Returns:
        One of ``"cupy"``, ``"torch"``, ``"cpu"``.
    """
    _validate_backend_arg(backend)
    if backend == "cpu":
        return "cpu"
    if backend in ("auto", "cupy") and _cupy_usable():
        return "cupy"
    if backend in ("auto", "torch") and _torch_usable():
        return "torch"
    if backend in ("cupy", "torch"):
        logger.debug("Requested backend %r is unavailable; falling back to CPU", backend)
    return "cpu"


# ---------------------------------------------------------------------------
# Minimal array-op adapter: one engine implementation for numpy/cupy/torch
# ---------------------------------------------------------------------------
class _ArrayOps:
    """Tiny adapter unifying the row/array operations the IC engine needs across
    NumPy, CuPy and PyTorch.

    Running the engine on plain NumPy (``_ArrayOps(np)``) lets its math be
    validated on CPU-only machines; the CuPy/PyTorch adapters run the identical
    code on the device.
    """

    def __init__(self, lib: Any, device: str | None = None) -> None:
        self.lib = lib
        self.name = lib.__name__  # "numpy" | "cupy" | "torch"
        self.is_torch = self.name == "torch"
        if self.is_torch and device is None:
            device = "cuda"
        self.device = device
        self.f64 = lib.float64
        self.i64 = lib.int64
        self.boolean = lib.bool if self.is_torch else lib.bool_
        # 0-dim tensors are accepted by torch.where for any operand kind.
        if self.is_torch:
            self.inf = lib.tensor(math.inf, dtype=self.f64, device=self.device)
            self.nan = lib.tensor(math.nan, dtype=self.f64, device=self.device)
        else:
            self.inf = lib.inf
            self.nan = lib.nan

    def asarray(self, a: Any, dtype: Any = None) -> Any:
        dtype = self.f64 if dtype is None else dtype
        if self.is_torch:
            return self.lib.as_tensor(np.ascontiguousarray(np.asarray(a)), dtype=dtype, device=self.device)
        return self.lib.asarray(a, dtype=dtype)

    def full(self, shape: tuple, value: Any, dtype: Any) -> Any:
        if self.is_torch:
            return self.lib.full(shape, value, dtype=dtype, device=self.device)
        return self.lib.full(shape, value, dtype=dtype)

    def ones(self, shape: tuple, dtype: Any) -> Any:
        if self.is_torch:
            return self.lib.ones(shape, dtype=dtype, device=self.device)
        return self.lib.ones(shape, dtype=dtype)

    def arange(self, n: int, dtype: Any) -> Any:
        if self.is_torch:
            return self.lib.arange(n, dtype=dtype, device=self.device)
        return self.lib.arange(n, dtype=dtype)

    def concatenate(self, arrays: list, axis: int = 1) -> Any:
        if self.is_torch:
            return self.lib.cat(list(arrays), dim=axis)
        return self.lib.concatenate(list(arrays), axis=axis)

    def cumsum(self, a: Any, axis: int = 1) -> Any:
        if self.is_torch:
            return self.lib.cumsum(a, dim=axis)
        return self.lib.cumsum(a, axis=axis)

    def sum(self, a: Any, axis: int = 1) -> Any:
        if self.is_torch:
            return a.sum(dim=axis)
        return a.sum(axis=axis)

    def argsort(self, a: Any) -> Any:
        if self.is_torch:
            return self.lib.argsort(a, dim=-1)
        return self.lib.argsort(a, axis=-1)

    def take_along(self, a: Any, idx: Any) -> Any:
        if self.is_torch:
            return self.lib.take_along_dim(a, idx, dim=-1)
        return self.lib.take_along_axis(a, idx, axis=-1)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return self.lib.where(cond, a, b)

    def isfinite(self, a: Any) -> Any:
        return self.lib.isfinite(a)

    def sqrt(self, a: Any) -> Any:
        return self.lib.sqrt(a)

    def broadcast(self, a: Any, shape: tuple) -> Any:
        return self.lib.broadcast_to(a, shape)

    def reshape(self, a: Any, shape: tuple) -> Any:
        return a.reshape(shape)

    def astype(self, a: Any, dtype: Any) -> Any:
        if self.is_torch:
            return a.to(dtype)
        return a.astype(dtype)

    def to_host(self, a: Any) -> np.ndarray:
        if self.is_torch:
            return a.detach().cpu().numpy()
        if self.name == "cupy":
            return self.lib.asnumpy(a)
        return np.asarray(a)


def _rankdata_average_rows(a: Any, ops: _ArrayOps) -> Any:
    """Average ranks (1-based; ties share the mean rank) along axis 1 of a 2-D array.

    Matches ``scipy.stats.rankdata(..., method="average")`` row by row, fully
    vectorised across rows. Rows must be finite (the caller replaces invalid
    entries with ``+inf``); those entries sort above every valid value and their
    ranks are masked out downstream.

    Algorithm per row: sort, mark group boundaries, assign 1-based dense group
    ids, then look up each group's first (start) and one-past-last (end) sorted
    positions from a flat per-(row, group) table. The 1-based average rank of a
    group spanning sorted positions [start, end) is ``(start + end + 1) / 2``.
    """
    n_rows, n_cols = a.shape
    order = ops.argsort(a)  # (N, S) — any stable tie order gives the same averages
    sorted_vals = ops.take_along(a, order)

    boundary = sorted_vals[:, 1:] != sorted_vals[:, :-1]  # (N, S-1)
    true_col = ops.ones((n_rows, 1), ops.boolean)
    is_group_start = ops.concatenate([true_col, boundary], axis=1)
    is_group_end = ops.concatenate([boundary, true_col], axis=1)
    group_id = ops.astype(ops.cumsum(is_group_start, axis=1), ops.i64)  # 1-based, (N, S)

    positions = ops.broadcast(ops.arange(n_cols, ops.i64)[None, :], (n_rows, n_cols))
    rows = ops.broadcast(ops.arange(n_rows, ops.i64)[:, None], (n_rows, n_cols))

    # Flatten; one table slot per (row, group) pair — each is written exactly once.
    slot = rows.reshape(-1) * n_cols + (group_id.reshape(-1) - 1)
    pos_flat = positions.reshape(-1)
    start_flag = is_group_start.reshape(-1)
    end_flag = is_group_end.reshape(-1)

    starts_table = ops.full((n_rows * n_cols,), -1, ops.i64)
    starts_table[slot[start_flag]] = pos_flat[start_flag]
    ends_table = ops.full((n_rows * n_cols,), -1, ops.i64)
    ends_table[slot[end_flag]] = pos_flat[end_flag] + 1  # one past the group's last element

    ranks_sorted = ops.astype(starts_table[slot] + ends_table[slot] + 1, ops.f64) * 0.5
    ranks_sorted = ranks_sorted.reshape(n_rows, n_cols)

    inverse = ops.argsort(order)  # argsort(argsort(x)) is the inverse permutation
    return ops.take_along(ranks_sorted, inverse)


def _rank_ic_engine(signal_grids: Any, returns_grid: Any, ops: _ArrayOps) -> list[np.ndarray]:
    """Vectorised cross-sectional rank IC for K signal grids against one returns grid.

    Args:
        signal_grids: Device array ``(K, D, S)`` of signal values (NaN allowed).
        returns_grid: Device array ``(D, S)`` of forward returns (NaN allowed).
        ops: Backend adapter.

    Returns:
        One host NumPy 1-D array per signal, holding the per-date ICs of the
        dates that pass the CPU implementation's filters (for finite inputs;
        ±inf entries are masked — see the module docstring's known-divergence
        note), in date order.
    """
    k, d, s = signal_grids.shape
    if returns_grid.shape != (d, s):
        raise ValueError(f"returns grid shape {returns_grid.shape} does not match signals {(d, s)}")
    if k == 0 or d == 0 or s == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(k)]

    returns_3d = ops.broadcast(returns_grid[None, :, :], (k, d, s))
    mask = ops.isfinite(signal_grids) & ops.isfinite(returns_3d)  # (K, D, S)

    # Replace invalid entries with +inf BEFORE ranking so that, within each row,
    # all valid entries receive exactly the ranks of the valid subset.
    x2 = ops.reshape(ops.where(mask, signal_grids, ops.inf), (k * d, s))
    y2 = ops.reshape(ops.where(mask, returns_3d, ops.inf), (k * d, s))
    m2 = ops.reshape(mask, (k * d, s))

    rank_x = _rankdata_average_rows(x2, ops)
    rank_y = _rankdata_average_rows(y2, ops)

    # Per-row masked statistics. Denominators are guarded so degenerate rows
    # produce NaN/False flags instead of warnings; they are filtered below.
    n = ops.sum(m2, axis=1)  # int, (N,)
    n_f = ops.astype(n, ops.f64)
    safe_n = ops.where(n_f > 0, n_f, 1.0)

    x_finite = ops.where(m2, x2, 0.0)
    y_finite = ops.where(m2, y2, 0.0)
    mean_x = ops.sum(x_finite, axis=1) / safe_n
    mean_y = ops.sum(y_finite, axis=1) / safe_n
    dev_x = ops.where(m2, x_finite - mean_x[:, None], 0.0)
    dev_y = ops.where(m2, y_finite - mean_y[:, None], 0.0)
    std_x = ops.sqrt(ops.sum(dev_x * dev_x, axis=1) / safe_n)
    std_y = ops.sqrt(ops.sum(dev_y * dev_y, axis=1) / safe_n)

    rx_valid = ops.where(m2, rank_x, 0.0)
    ry_valid = ops.where(m2, rank_y, 0.0)
    mean_rx = ops.sum(rx_valid, axis=1) / safe_n
    mean_ry = ops.sum(ry_valid, axis=1) / safe_n
    dx = ops.where(m2, rx_valid - mean_rx[:, None], 0.0)
    dy = ops.where(m2, ry_valid - mean_ry[:, None], 0.0)
    cov = ops.sum(dx * dy, axis=1)
    var_x = ops.sum(dx * dx, axis=1)
    var_y = ops.sum(dy * dy, axis=1)
    denom = var_x * var_y
    safe_denom = ops.where(denom > 0, denom, 1.0)
    corr = ops.where(denom > 0, cov / ops.sqrt(safe_denom), math.nan)

    keep = (
        (n >= _MIN_NAMES_PER_DATE)
        & (std_x >= _MIN_CROSS_SECTIONAL_STD)
        & (std_y >= _MIN_CROSS_SECTIONAL_STD)
        & ops.isfinite(corr)
    )

    corr_host = ops.to_host(corr)
    keep_host = ops.to_host(keep)

    results: list[np.ndarray] = []
    for i in range(k):
        lo, hi = i * d, (i + 1) * d
        results.append(np.asarray(corr_host[lo:hi][keep_host[lo:hi]], dtype=np.float64))
    return results


# ---------------------------------------------------------------------------
# Aggregation and CPU reference (identical formulas to signal_evaluator)
# ---------------------------------------------------------------------------
def _aggregate_ics(ic_array: np.ndarray) -> dict[str, Any]:
    """Aggregate per-period ICs with the exact formulas of compute_rank_ic."""
    ic_array = np.asarray(ic_array, dtype=np.float64)
    mean_ic = float(np.mean(ic_array))
    ic_std = float(np.std(ic_array))
    num_periods = int(len(ic_array))

    ic_ir = mean_ic / ic_std if ic_std > 0 else None
    t_stat = mean_ic / (ic_std / np.sqrt(num_periods)) if ic_std > 0 else None
    p_value = (
        float(2 * (1 - stats.t.cdf(abs(t_stat), df=num_periods - 1))) if t_stat is not None else None
    )

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "num_periods": num_periods,
        "positive_ic_ratio": float(np.mean(ic_array > 0)),
        "ic_percentiles": {
            "5th": float(np.percentile(ic_array, 5)),
            "25th": float(np.percentile(ic_array, 25)),
            "50th": float(np.percentile(ic_array, 50)),
            "75th": float(np.percentile(ic_array, 75)),
            "95th": float(np.percentile(ic_array, 95)),
        },
    }


def _empty_result(backend_name: str, error_message: str) -> dict[str, Any]:
    return {
        "mean_ic": None,
        "ic_std": None,
        "ic_ir": None,
        "t_stat": None,
        "p_value": None,
        "num_periods": 0,
        "error": error_message,
        "backend": backend_name,
    }


def _import_cpu_compute_rank_ic() -> Any:
    """Return signal_evaluator.compute_rank_ic, or a local equivalent if unavailable."""
    try:
        from signal_discovery_workflow.signal_evaluator import compute_rank_ic
    except ImportError:
        try:
            from .signal_evaluator import compute_rank_ic  # type: ignore[no-redef]
        except ImportError:
            return _compute_rank_ic_local
    return compute_rank_ic


def _compute_rank_ic_local(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict[str, Any]:
    """Standalone CPU rank IC mirroring signal_evaluator.compute_rank_ic.

    Only used when signal_evaluator itself cannot be imported.
    """
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return _empty_result("cpu", "No common dates or stocks between signal and returns")

    signal_aligned = signal_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ic_series: list[float] = []
        for date in common_dates:
            signal_row = signal_aligned.loc[date].dropna()
            returns_row = returns_aligned.loc[date].dropna()
            common = signal_row.index.intersection(returns_row.index)
            if len(common) < _MIN_NAMES_PER_DATE:
                continue
            signal_vals = signal_row[common].values
            return_vals = returns_row[common].values
            if np.std(signal_vals) < _MIN_CROSS_SECTIONAL_STD or np.std(return_vals) < _MIN_CROSS_SECTIONAL_STD:
                continue
            try:
                correlation, _ = stats.spearmanr(signal_vals, return_vals)
                if not np.isnan(correlation):
                    ic_series.append(correlation)
            except Exception:  # noqa: BLE001 — mirror the reference per-row tolerance
                continue

    if len(ic_series) == 0:
        return _empty_result("cpu", "Could not compute IC for any period")
    return _aggregate_ics(np.array(ic_series))


def _rank_ic_cpu(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict[str, Any]:
    """CPU path: reuse signal_evaluator.compute_rank_ic (identical for finite
    inputs; see the module docstring's known-divergence note)."""
    return dict(_import_cpu_compute_rank_ic()(signal_values, forward_returns))


def _wrap_cpu_result(cpu_result: dict[str, Any], backend_name: str) -> dict[str, Any]:
    result = dict(cpu_result)
    result["backend"] = backend_name
    return result


def _load_backend_module(backend_name: str) -> Any:
    if backend_name == "cupy":
        return _load_cupy()
    if backend_name == "torch":
        return _load_torch()
    raise ValueError(f"unsupported device backend: {backend_name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rank_ic_gpu(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    backend: str = "auto",
) -> dict[str, Any]:
    """Rank IC statistics, computed on a GPU when available, else on CPU.

    Returns the same keys as ``signal_evaluator.compute_rank_ic``
    (``mean_ic``, ``ic_std``, ``ic_ir``, ``t_stat``, ``p_value``,
    ``num_periods``, ``positive_ic_ratio``, ``ic_percentiles`` — plus ``error``
    on the empty cases) and one extra key ``backend`` naming the backend that
    was actually used. For finite inputs, dates with < 10 common valid names,
    constant rows (cross-sectional std < 1e-10) and non-finite correlations are
    skipped exactly like the CPU implementation; ±inf inputs diverge — see the
    module docstring's known-divergence note.

    Args:
        signal_values: Signal values (rows=dates, cols=tickers).
        forward_returns: Forward returns (rows=dates, cols=tickers).
        backend: ``"auto"``, ``"cupy"``, ``"torch"`` or ``"cpu"``.

    Returns:
        Metrics dictionary including ``"backend"``.
    """
    resolved = get_backend(backend)
    if resolved == "cpu":
        return _wrap_cpu_result(_rank_ic_cpu(signal_values, forward_returns), "cpu")

    try:
        return _rank_ic_on_device(signal_values, forward_returns, resolved)
    except Exception as exc:  # noqa: BLE001 — automatic, always-available fallback
        logger.warning("GPU rank IC failed on backend %s (%s); falling back to CPU", resolved, exc)
        return _wrap_cpu_result(_rank_ic_cpu(signal_values, forward_returns), "cpu")


def _rank_ic_on_device(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    backend_name: str,
) -> dict[str, Any]:
    library = _load_backend_module(backend_name)
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return _empty_result(backend_name, "No common dates or stocks between signal and returns")

    signal_aligned = signal_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    ops = _ArrayOps(library)
    signal_grid = ops.asarray(np.asarray(signal_aligned.to_numpy(dtype="float64"))[None, :, :])
    returns_grid = ops.asarray(np.asarray(returns_aligned.to_numpy(dtype="float64")))

    (ic_array,) = _rank_ic_engine(signal_grid, returns_grid, ops)
    if ic_array.size == 0:
        return _empty_result(backend_name, "Could not compute IC for any period")

    result = _aggregate_ics(ic_array)
    result["backend"] = backend_name
    return result


def compute_forward_returns_gpu(
    close: pd.DataFrame,
    periods: int = 5,
    backend: str = "auto",
) -> pd.DataFrame:
    """Forward returns ``close.shift(-periods) / close - 1``, on device when available.

    Args:
        close: Closing prices (rows=dates, cols=tickers).
        periods: Number of forward periods. Values < 1 are delegated to pandas.
        backend: ``"auto"``, ``"cupy"``, ``"torch"`` or ``"cpu"``.

    Returns:
        Forward-returns DataFrame with the same index and columns as ``close``.
    """
    resolved = get_backend(backend)
    if resolved == "cpu" or periods < 1:
        # pandas handles every edge case (periods <= 0, empty frames) itself.
        return close.shift(-periods) / close - 1
    try:
        return _forward_returns_on_device(close, periods, resolved)
    except Exception as exc:  # noqa: BLE001 — automatic, always-available fallback
        logger.warning("GPU forward returns failed on backend %s (%s); falling back to CPU", resolved, exc)
        return close.shift(-periods) / close - 1


def _forward_returns_on_device(close: pd.DataFrame, periods: int, backend_name: str) -> pd.DataFrame:
    library = _load_backend_module(backend_name)
    ops = _ArrayOps(library)
    values = ops.asarray(np.asarray(close.to_numpy(dtype="float64")))
    n_dates = values.shape[0]

    shifted = ops.full(values.shape, math.nan, ops.f64)
    if periods < n_dates:
        shifted[:-periods] = values[periods:]  # row i receives the price at i + periods

    forward_returns = ops.to_host(shifted / values - 1)
    return pd.DataFrame(forward_returns, index=close.index, columns=close.columns)


def compute_rank_ic_batch(
    signal_dict: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    backend: str = "auto",
) -> dict[str, dict[str, Any]]:
    """Evaluate many signals at once; on GPU the rank/correlation work is batched.

    Every signal is aligned to the forward-returns grid (dates/tickers missing
    from a signal become NaN and are masked), so all signals are ranked and
    correlated in a single vectorised pass — no Python loop over signals when a
    device is present. Results match per-signal ``compute_rank_ic`` calls for
    finite/NaN inputs; ±inf inputs diverge (see the module docstring's
    known-divergence note).

    Args:
        signal_dict: Mapping of signal name to signal DataFrame.
        forward_returns: Forward returns (rows=dates, cols=tickers).
        backend: ``"auto"``, ``"cupy"``, ``"torch"`` or ``"cpu"``.

    Returns:
        Mapping of signal name to its metrics dictionary (each with ``"backend"``).
    """
    if not signal_dict:
        return {}

    resolved = get_backend(backend)
    if resolved == "cpu":
        return {name: _wrap_cpu_result(_rank_ic_cpu(df, forward_returns), "cpu") for name, df in signal_dict.items()}

    try:
        return _rank_ic_batch_on_device(signal_dict, forward_returns, resolved)
    except Exception as exc:  # noqa: BLE001 — automatic, always-available fallback
        logger.warning("GPU batch rank IC failed on backend %s (%s); falling back to CPU", resolved, exc)
        return {name: _wrap_cpu_result(_rank_ic_cpu(df, forward_returns), "cpu") for name, df in signal_dict.items()}


def _rank_ic_batch_on_device(
    signal_dict: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    backend_name: str,
) -> dict[str, dict[str, Any]]:
    library = _load_backend_module(backend_name)
    grid_dates = forward_returns.index
    grid_stocks = forward_returns.columns
    if len(grid_dates) == 0 or len(grid_stocks) == 0:
        return {
            name: _empty_result(backend_name, "No common dates or stocks between signal and returns")
            for name in signal_dict
        }

    ops = _ArrayOps(library)
    # Align every signal to the returns grid: entries missing from a signal are
    # NaN and get masked, which reproduces the per-signal intersection semantics.
    grids = [
        np.asarray(signal.reindex(index=grid_dates, columns=grid_stocks).to_numpy(dtype="float64"))
        for signal in signal_dict.values()
    ]
    signal_stacked = ops.asarray(np.stack(grids, axis=0))
    returns_grid = ops.asarray(np.asarray(forward_returns.to_numpy(dtype="float64")))

    ic_arrays = _rank_ic_engine(signal_stacked, returns_grid, ops)

    results: dict[str, dict[str, Any]] = {}
    for name, ic_array in zip(signal_dict, ic_arrays):
        if ic_array.size == 0:
            results[name] = _empty_result(backend_name, "Could not compute IC for any period")
        else:
            metrics = _aggregate_ics(ic_array)
            metrics["backend"] = backend_name
            results[name] = metrics
    return results


def _format_cuda_version(version: int) -> str:
    return f"{version // 1000}.{(version % 1000) // 10}"


def _format_cudnn_version(version: int) -> str:
    return f"{version // 1000}.{(version % 1000) // 100}.{version % 100}"


def gpu_info() -> dict[str, Any]:
    """Non-sensitive description of the accelerator in use.

    Returns a dict with keys drawn from ``backend``, ``device_name``,
    ``cuda_version``, ``cudnn_version``, ``memory_total_mb``,
    ``memory_free_mb`` and ``available``. Never exposes secrets or environment
    variables; probing failures simply omit the corresponding key.
    """
    backend = get_backend("auto")
    info: dict[str, Any] = {"backend": backend, "available": backend != "cpu"}

    if backend == "cupy":
        cupy = _load_cupy()
        try:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            info["device_name"] = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
        except Exception:  # noqa: BLE001 — degrade gracefully per field
            pass
        try:
            free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
            info["memory_total_mb"] = int(total_bytes // (1024 * 1024))
            info["memory_free_mb"] = int(free_bytes // (1024 * 1024))
        except Exception:  # noqa: BLE001
            pass
        try:
            info["cuda_version"] = _format_cuda_version(int(cupy.cuda.runtime.runtimeGetVersion()))
        except Exception:  # noqa: BLE001
            pass
        try:
            info["cudnn_version"] = _format_cudnn_version(int(cupy.cuda.cudnn.getVersion()))
        except Exception:  # noqa: BLE001
            pass
    elif backend == "torch":
        torch = _load_torch()
        try:
            info["device_name"] = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001
            pass
        try:
            if torch.version.cuda:
                info["cuda_version"] = str(torch.version.cuda)
        except Exception:  # noqa: BLE001
            pass
        try:
            if torch.backends.cudnn.is_available() and torch.backends.cudnn.version():
                info["cudnn_version"] = _format_cudnn_version(int(torch.backends.cudnn.version()))
        except Exception:  # noqa: BLE001
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["memory_total_mb"] = int(total_bytes // (1024 * 1024))
            info["memory_free_mb"] = int(free_bytes // (1024 * 1024))
        except Exception:  # noqa: BLE001
            pass

    return info
