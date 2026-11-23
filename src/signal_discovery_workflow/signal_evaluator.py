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

"""
Signal Evaluator: Rank IC computation, signal-code execution, and helpers.

The rank IC measures the Spearman correlation between signal values and forward
stock returns — a standard quant-research signal-quality metric.

Forward-returns caching
-----------------------
Forward returns are cached at module level (``_forward_returns_cache``) so that
many candidate signals evaluated against the same close prices reuse a single
precomputed frame instead of recomputing ``close.shift(-periods) / close - 1``
for every candidate. Cache keys combine the horizon, index/column labels,
shape, and a cheap *probabilistic* content fingerprint (``_close_fingerprint``)
— not a cryptographic hash of every value. The fingerprint samples ~64 values,
so two different datasets colliding on the same key is possible in principle
but negligible in practice; the trade-off keeps key construction O(rows + cols)
instead of hashing the full array. Use ``clear_forward_returns_cache()`` to
reset the cache in long-running processes.

Operator-result caching
-----------------------
When enabled (``USE_OPERATOR_CACHE`` or env ``SIGNAL_USE_OPERATOR_CACHE``),
the operator callables of each candidate module are wrapped IN PLACE with
``operator_cache.memoize_operator`` after signal-function detection, so
identical operator calls reuse one cached result — across candidates too,
since the cache key fingerprints the operator implementation, not object
identity (different implementations sharing a name never share an entry).
Cached values are handed out as shallow copies (``copy_on_return=True``).
The cache is cleared whenever market data is (re)loaded
(``load_stock_data`` / ``clear_operator_cache``).

LIMITATION: the sandboxed execution path is NOT cached — the sandbox runs
generated code in its own restricted namespace and does not share the
evaluator's operator cache.
"""

import inspect
import json
import logging
import math
import os
import re
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from signal_discovery_workflow.operator_cache import OperatorCache, memoize_operator

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "sp500"
CALCULATOR_PATH = Path(__file__).parent / "template" / "calculator.json"

# --- Optional integrations (both OFF by default: legacy behaviour unchanged) ---

#: Route ``execute_signal_code`` through the tested sandbox (``sandbox.py``).
#: Default OFF: when disabled, the legacy in-process ``exec`` path runs exactly
#: as before. Override at call time via env var ``SIGNAL_SANDBOX_ENABLED``.
SANDBOX_ENABLED: bool = False

#: Delegate ``compute_rank_ic`` to ``gpu_backend.compute_rank_ic_gpu``.
#: Default OFF: the returned dict keeps its exact legacy shape (no ``"backend"``
#: key). Override at call time via env var ``SIGNAL_USE_GPU``.
USE_GPU_BACKEND: bool = False

#: Backend selector passed to ``compute_rank_ic_gpu`` when GPU delegation is
#: enabled (``"auto"``, ``"cupy"``, ``"torch"`` or ``"cpu"``). Override at call
#: time via env var ``SIGNAL_GPU_BACKEND``.
GPU_BACKEND: str = "auto"

#: Guards against infinite recursion: ``gpu_backend``'s CPU path imports this
#: module's ``compute_rank_ic``, so the nested call must not delegate again.
#: Evaluation is single-threaded here; under concurrent calls this can only
#: skip GPU acceleration (never produce wrong results).
_GPU_DELEGATION_ACTIVE = False

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env-var override at call time (unset → ``default``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _sandbox_enabled() -> bool:
    """Whether sandboxed signal execution is enabled (env override wins)."""
    return _env_flag("SIGNAL_SANDBOX_ENABLED", SANDBOX_ENABLED)


def _gpu_enabled() -> bool:
    """Whether GPU rank-IC delegation is enabled (env override wins)."""
    return _env_flag("SIGNAL_USE_GPU", USE_GPU_BACKEND)


def _gpu_backend_name() -> str:
    """Backend selector for ``compute_rank_ic_gpu`` (env override wins)."""
    return os.environ.get("SIGNAL_GPU_BACKEND", GPU_BACKEND).strip().lower() or GPU_BACKEND


#: Memoize operator computations across candidate signal modules
#: (``operator_cache``). Default OFF: with the flag off the candidate namespace
#: is not touched at all. Override at call time via
#: ``SIGNAL_USE_OPERATOR_CACHE``.
USE_OPERATOR_CACHE: bool = False

#: Maximum number of entries in the shared operator cache. Env override
#: ``SIGNAL_OPERATOR_CACHE_MAX_ENTRIES`` (int; a bad value falls back to the
#: default).
OPERATOR_CACHE_MAX_ENTRIES: int = 512


def _operator_cache_enabled() -> bool:
    """Whether operator-result caching is enabled (env override wins)."""
    return _env_flag("SIGNAL_USE_OPERATOR_CACHE", USE_OPERATOR_CACHE)


def _operator_cache_max_entries() -> int:
    """Max entries for the operator cache (env override; bad value → default)."""
    raw = os.environ.get("SIGNAL_OPERATOR_CACHE_MAX_ENTRIES")
    if raw is None:
        return OPERATOR_CACHE_MAX_ENTRIES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"Invalid SIGNAL_OPERATOR_CACHE_MAX_ENTRIES={raw!r}; "
            f"using default {OPERATOR_CACHE_MAX_ENTRIES}"
        )
        return OPERATOR_CACHE_MAX_ENTRIES
    if value <= 0:
        logger.warning(
            f"Invalid SIGNAL_OPERATOR_CACHE_MAX_ENTRIES={raw!r}; "
            f"using default {OPERATOR_CACHE_MAX_ENTRIES}"
        )
        return OPERATOR_CACHE_MAX_ENTRIES
    return value


#: Shared operator-result cache for all candidate signal modules of a run.
_OPERATOR_CACHE = OperatorCache(max_entries=_operator_cache_max_entries())


def clear_operator_cache() -> None:
    """Reset the shared operator cache (drops entries and hit/miss counters)."""
    _OPERATOR_CACHE.clear()


def _install_wrapped_operators(namespace: dict[str, Any]) -> None:
    """
    Wrap operator callables in ``namespace`` IN PLACE with memoized versions.

    The signal functions' ``__globals__`` IS this namespace dict, so replacing
    the operator entries in place makes the already-created signal functions
    call the memoized wrappers. (``operator_cache.install_cached_operators``
    returns a NEW dict and therefore cannot be used here.) Only names present
    in ``OPERATOR_NAMES`` that are callables are wrapped — signal functions,
    helpers, builtins and modules are left untouched. Wrappers use
    ``copy_on_return=True`` so a cached result is handed out as a shallow
    copy; the implementation fingerprint inside the cache key keeps different
    operators that share a name from ever sharing an entry.
    """
    for name, obj in list(namespace.items()):
        if name not in OPERATOR_NAMES or not callable(obj):
            continue
        if getattr(obj, "_operator_cache_wrapped", False):
            continue  # already wrapped (idempotency guard from operator_cache)
        namespace[name] = memoize_operator(obj, _OPERATOR_CACHE, copy_on_return=True)


def _load_all_operators() -> dict[str, callable]:
    """Load and compile all operator functions from calculator.json into a callable dict."""
    if not CALCULATOR_PATH.exists():
        return {}
    with open(CALCULATOR_PATH, "r") as f:
        operators = json.load(f)
    op_namespace = {"pd": pd, "np": np}
    for op in operators:
        try:
            exec(op["code"], op_namespace)
        except Exception as e:
            logger.warning(f"Failed to compile operator {op.get('name')}: {e}")
    return {
        name: obj
        for name, obj in op_namespace.items()
        if callable(obj) and name not in ("pd", "np")
    }


_OPERATOR_FUNCTIONS = _load_all_operators()


def get_operator_arities() -> dict[str, tuple[int, int]]:
    """Return ``{operator_name: (min_required_args, max_args)}`` for every loaded operator.

    ``max_args`` is ``-1`` if the operator accepts ``*args``.
    Used by the code generator to validate formulas before execution.
    """
    arities: dict[str, tuple[int, int]] = {}
    for name, fn in _OPERATOR_FUNCTIONS.items():
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        required = sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        has_varargs = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        max_args = -1 if has_varargs else len([
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ])
        arities[name] = (required, max_args)
    return arities


def load_stock_data() -> dict[str, pd.DataFrame]:
    """Load all available stock price-volume data from CSV files."""
    data = {}
    data_files = ["Open", "Close", "High", "Low", "Volume"]

    for field in data_files:
        file_path = DATA_DIR / f"{field}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data[field] = df
                logger.info(f"Loaded {field}.csv with shape {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {field}.csv: {e}")
        else:
            logger.warning(f"Data file not found: {file_path}")

    # Market data (re)loaded: reset the operator cache to bound memory. The
    # content fingerprints in the cache keys already prevent stale hits.
    _OPERATOR_CACHE.clear()
    return data


# Module-level forward-returns cache. Keyed by (periods, labels, shape,
# content fingerprint); values are the computed forward-return frames.
_forward_returns_cache: dict[tuple, pd.DataFrame] = {}


def _close_fingerprint(close: pd.DataFrame) -> float:
    """
    Cheap probabilistic fingerprint of a frame's content.

    Samples ~64 values evenly spaced through the flattened value array (all
    values for small frames) and returns their sum. This is a cheap
    *probabilistic* fingerprint, not a cryptographic hash: two different
    datasets could theoretically produce the same fingerprint, though the
    probability is negligible for real price data. The trade-off is
    deliberate — key construction stays O(rows + cols) sampling instead of a
    full O(n) hash of every value.
    """
    values = np.nan_to_num(close.to_numpy()).ravel()
    stride = max(1, values.size // 64)
    return float(values[::stride].sum())


def _forward_returns_cache_key(close: pd.DataFrame, periods: int) -> tuple:
    """
    Deterministic, cheap cache key for a forward-returns computation.

    Combines the horizon, the frame's index/column labels, its shape, and the
    content fingerprint. The fingerprint guards against two different datasets
    with identical labels/shape silently sharing a cache entry. Full values are
    never hashed, so key construction is O(rows + cols).
    """
    return (
        periods,
        close.shape,
        tuple(close.index.tolist()),
        tuple(close.columns.tolist()),
        _close_fingerprint(close),
    )


def _get_cached_forward_returns(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Compute (or fetch from cache) forward returns ``close.shift(-periods) / close - 1``.

    On a cache hit the stored frame is returned as-is; on a miss the frame is
    computed, stored, and returned. Callers must not mutate the returned frame,
    since it may be shared across calls.
    """
    key = _forward_returns_cache_key(close, periods)
    cached = _forward_returns_cache.get(key)
    if cached is not None:
        logger.debug(f"Forward-returns cache HIT (periods={periods})")
        return cached
    logger.debug(f"Forward-returns cache MISS (periods={periods})")
    forward_returns = close.shift(-periods) / close - 1
    _forward_returns_cache[key] = forward_returns
    return forward_returns


def clear_forward_returns_cache() -> None:
    """Reset the module-level forward-returns cache.

    Needed by tests and by long-running processes that process many distinct
    datasets and must not accumulate cached frames indefinitely.
    """
    _forward_returns_cache.clear()


def compute_forward_returns(close: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """
    Compute forward returns for the next N periods.

    Results are served from a module-level cache keyed on the horizon, index,
    columns, shape, and a cheap content fingerprint, so repeated calls with
    the same data reuse one precomputed frame.

    Args:
        close: DataFrame of closing prices (rows=dates, cols=stocks).
        periods: Number of forward periods for return calculation.

    Returns:
        DataFrame of forward returns with the same shape as input. Callers
        must not mutate the returned frame — it may be shared with other
        callers; use ``clear_forward_returns_cache()`` to reset the cache.
    """
    return _get_cached_forward_returns(close, periods)


def compute_signal_turnover(
    signal_values: pd.DataFrame,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualized cross-sectional percentile-rank turnover of a signal.

    For each date, every stock's signal value is ranked cross-sectionally into
    [0, 1] (percentile rank). Per-date turnover is the mean absolute change of
    those percentile ranks versus the previous date, averaged across stocks:

        turnover[t] = mean_i( |rank[t, i] - rank[t - 1, i]| )

    The returned value is the mean per-period turnover annualized by
    ``periods_per_year``:

        annualized_turnover = mean_t( turnover[t] ) * periods_per_year

    Args:
        signal_values: DataFrame of signal values (rows=dates, cols=stocks).
        periods_per_year: Number of trading periods per year used to annualize
            (default 252 for daily data).

    Returns:
        Annualized turnover as a float. Returns 0.0 for empty input or when
        fewer than two usable dates exist (turnover requires a previous date).
    """
    if not isinstance(signal_values, pd.DataFrame):
        return 0.0
    if signal_values.empty or len(signal_values) < 2:
        return 0.0

    pct_ranks = signal_values.rank(axis=1, pct=True)
    rank_changes = pct_ranks.diff().abs()
    # The first date has no previous rank (NaN diff) and is dropped by mean().
    per_period_turnover = rank_changes.mean(axis=1).dropna()
    if per_period_turnover.empty:
        return 0.0

    return float(per_period_turnover.mean() * periods_per_year)


def compute_rank_ic(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute rank IC (Information Coefficient) between signal values and forward returns.

    The rank IC is the Spearman correlation between signal ranks and return ranks
    computed cross-sectionally for each date, then aggregated.

    Args:
        signal_values: DataFrame of signal values (rows=dates, cols=stocks).
        forward_returns: DataFrame of forward returns (rows=dates, cols=stocks).

    Returns:
        Dictionary containing IC statistics.

    GPU delegation:
        When enabled (``USE_GPU_BACKEND`` or env ``SIGNAL_USE_GPU``), the
        computation is delegated to ``gpu_backend.compute_rank_ic_gpu`` and the
        result additionally carries a ``"backend"`` key. When disabled (the
        default) the returned dict is exactly the legacy CPU result — no
        ``"backend"`` key, byte-identical values. Any GPU failure (including a
        missing/broken backend) is logged at WARNING and falls back to the CPU
        implementation below.
    """
    global _GPU_DELEGATION_ACTIVE
    if _gpu_enabled() and not _GPU_DELEGATION_ACTIVE:
        # gpu_backend's CPU path imports this module's compute_rank_ic, so the
        # delegation guard above prevents infinite recursion when it calls
        # back into us. Lazily imported: gpu_backend imports this module, and
        # an eager import here would create a circular import.
        _GPU_DELEGATION_ACTIVE = True
        try:
            from signal_discovery_workflow.gpu_backend import compute_rank_ic_gpu

            return compute_rank_ic_gpu(
                signal_values, forward_returns, backend=_gpu_backend_name()
            )
        except Exception as e:
            logger.warning(f"GPU rank-IC failed ({type(e).__name__}); falling back to CPU: {e}")
        finally:
            _GPU_DELEGATION_ACTIVE = False

    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return {
            "mean_ic": None,
            "ic_std": None,
            "ic_ir": None,
            "t_stat": None,
            "p_value": None,
            "num_periods": 0,
            "error": "No common dates or stocks between signal and returns",
        }

    signal_aligned = signal_values.loc[common_dates, common_stocks]
    returns_aligned = forward_returns.loc[common_dates, common_stocks]

    # Suppress numpy "invalid value encountered" RuntimeWarnings from std/mean
    # calculations on edge-case rows. The dropna/length checks already handle
    # the actual data validity logic correctly.
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        ic_series = []
        for date in common_dates:
            signal_row = signal_aligned.loc[date].dropna()
            returns_row = returns_aligned.loc[date].dropna()

            common = signal_row.index.intersection(returns_row.index)
            if len(common) < 10:
                continue

            signal_vals = signal_row[common].values
            return_vals = returns_row[common].values

            # Spearman correlation requires non-constant inputs.
            if np.std(signal_vals) < 1e-10 or np.std(return_vals) < 1e-10:
                continue

            try:
                correlation, _ = stats.spearmanr(signal_vals, return_vals)
                if not np.isnan(correlation):
                    ic_series.append(correlation)
            except Exception:
                continue

        if len(ic_series) == 0:
            return {
                "mean_ic": None,
                "ic_std": None,
                "ic_ir": None,
                "t_stat": None,
                "p_value": None,
                "num_periods": 0,
                "error": "Could not compute IC for any period",
            }

        ic_array = np.array(ic_series)
        mean_ic = float(np.mean(ic_array))
        ic_std = float(np.std(ic_array))
        num_periods = len(ic_series)

        ic_ir = mean_ic / ic_std if ic_std > 0 else None
        t_stat = mean_ic / (ic_std / np.sqrt(num_periods)) if ic_std > 0 else None
        p_value = (
            float(2 * (1 - stats.t.cdf(abs(t_stat), df=num_periods - 1)))
            if t_stat is not None
            else None
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


#: Minimum usable IC dates required for the IS sign decision and,
#: independently, for OOS reporting.  This matches the minimum sample used by
#: :func:`quant_metrics.cross_validate_signal`.
DIRECTIONAL_IC_MIN_PERIODS = 30


def _finite_float(value: Any) -> float | None:
    """Return a plain finite ``float`` for ``value``, or ``None``."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _plain_period_count(value: Any) -> int:
    """Return a non-negative plain ``int`` period count."""
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def _directional_ic_payload() -> dict[str, Any]:
    """Return the stable, JSON-safe no-decision directional-IC payload."""
    return {
        "direct_mean_ic": None,
        "inverse_mean_ic": None,
        "absolute_ic": None,
        "train_mean_ic": None,
        "train_p_value": None,
        "train_num_periods": 0,
        "inverse_train_mean_ic": None,
        "test_mean_ic": None,
        "test_p_value": None,
        "test_num_periods": 0,
        "inverse_test_mean_ic": None,
        "oriented_test_mean_ic": None,
        "oriented_test_p_value": None,
        "detected_sign": 0,
        "decision": "none",
        "decision_reason": None,
        "is_significant": False,
        "should_invert": False,
        "oos_reason": None,
        "split_date": None,
        "error": None,
    }


def compute_directional_ic(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_ratio: float = 0.7,
    ic_threshold: float = 0.03,
    p_value_threshold: float = 0.05,
    min_periods: int = DIRECTIONAL_IC_MIN_PERIODS,
) -> dict[str, Any]:
    """Compute sign-aware directional IC with an in-sample-only sign decision.

    The direct and inverse directions are evaluated by reusing
    :func:`compute_rank_ic` with the original signal and with ``-signal``.
    Spearman's correlation obeys ``rho(-x, y) == -rho(x, y)`` exactly, so the
    inverse direction can never have a larger absolute IC.  Its purpose is to
    identify a statistically significant signal whose orientation is wrong:
    the advisor can then recommend multiplying that signal by ``-1``.

    Dates are sorted chronologically and split exactly as in
    :func:`quant_metrics.cross_validate_signal`: the first
    ``floor(len(dates) * train_ratio)`` dates are the in-sample (IS) split and
    the remainder are the out-of-sample (OOS) split.  The orientation is
    decided from the IS mean IC only.  The selected sign is then applied to
    the OOS split; an opposing OOS result can never change the IS decision.

    Args:
        signal_values: Signal matrix (rows=dates, columns=stocks).
        forward_returns: Forward-return matrix with matching labels.
        train_ratio: Fraction of aligned dates assigned to the IS split.
        ic_threshold: Minimum absolute IS IC required for significance.
        p_value_threshold: Maximum IS p-value required for significance.
        min_periods: Minimum usable IS IC dates required for the sign
            decision.  The same minimum independently gates OOS reporting;
            OOS availability never gates the IS decision.

    Returns:
        A JSON-safe dictionary. Numeric values are plain ``float``/``int``,
        unavailable values are ``None``, and no NaN or infinity is emitted.
        The complete key set is:

        - ``direct_mean_ic`` / ``inverse_mean_ic``: full-sample direct and
          inverse mean IC (``float`` or ``None``).
        - ``absolute_ic``: absolute full-sample direct mean IC.
        - ``train_mean_ic`` / ``train_p_value`` / ``train_num_periods``:
          direct IS mean IC, p-value, and usable IC-date count.
        - ``inverse_train_mean_ic``: inverse IS mean IC.
        - ``test_mean_ic`` / ``test_p_value`` / ``test_num_periods``:
          direct OOS mean IC, p-value, and usable IC-date count.
        - ``inverse_test_mean_ic``: inverse OOS mean IC.
        - ``oriented_test_mean_ic`` / ``oriented_test_p_value``: OOS values
          after applying the already-fixed IS sign; both are ``None`` when OOS
          reporting is insufficient or no IS sign exists.
        - ``detected_sign``: ``-1`` for negative IS orientation, ``1`` for
          positive, and ``0`` when no IS decision exists.
        - ``decision``: ``"negative"``, ``"positive"``, or ``"none"``.
        - ``decision_reason``: ``None`` when an IS sign decision exists;
          otherwise ``"insufficient_in_sample_periods"``,
          ``"unavailable_in_sample_ic"``, ``"zero_in_sample_ic"``, or
          ``"directional_ic_computation_failed"``.  The last value means the
          directional-IC computation failed and the workflow continued with a
          safe no-decision fallback.
        - ``is_significant``: whether the IS mean IC and p-value satisfy the
          configured thresholds.
        - ``should_invert``: true only for a significant negative IS
          orientation.
        - ``oos_reason``: ``None`` when oriented OOS values are reported;
          otherwise ``"insufficient_out_of_sample_periods"``,
          ``"unavailable_out_of_sample_ic"``,
          ``"no_in_sample_decision"``, or
          ``"directional_ic_computation_failed"``.  The last value means the
          directional-IC computation failed and the workflow continued with a
          safe fallback that reports no oriented OOS values.
        - ``split_date``: first OOS date in ISO format, or ``None``.
        - ``error``: actual invalid-input/computation error text, or ``None``.
          Insufficient sample sizes are reported through the reason fields,
          not through this key.
    """
    payload = _directional_ic_payload()

    try:
        ratio = float(train_ratio)
    except (TypeError, ValueError):
        ratio = math.nan
    if not math.isfinite(ratio) or ratio <= 0 or ratio >= 1:
        payload["error"] = "train_ratio must be between 0 and 1"
        logger.warning("Directional IC skipped: %s", payload["error"])
        return payload

    try:
        minimum_periods = int(min_periods)
    except (TypeError, ValueError):
        minimum_periods = 0
    if minimum_periods < 1:
        payload["error"] = "min_periods must be a positive integer"
        logger.warning("Directional IC skipped: %s", payload["error"])
        return payload

    if not isinstance(signal_values, pd.DataFrame) or not isinstance(forward_returns, pd.DataFrame):
        payload["error"] = "signal_values and forward_returns must be DataFrames"
        return payload

    common_dates = signal_values.index.intersection(forward_returns.index).sort_values()
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        payload["error"] = "No common dates or stocks between signal and returns"
        return payload

    train_count = int(len(common_dates) * ratio)
    train_dates = common_dates[:train_count]
    test_dates = common_dates[train_count:]

    aligned_signal = signal_values.loc[common_dates, common_stocks]
    aligned_returns = forward_returns.loc[common_dates, common_stocks]
    train_signal = aligned_signal.loc[train_dates]
    train_returns = aligned_returns.loc[train_dates]
    test_signal = aligned_signal.loc[test_dates]
    test_returns = aligned_returns.loc[test_dates]

    def rank_ic_snapshot(values: pd.DataFrame, returns: pd.DataFrame) -> dict[str, Any]:
        """Call the legacy rank-IC function without leaking its key set."""
        try:
            result = compute_rank_ic(values, returns)
        except Exception as exc:
            logger.warning("Directional rank-IC computation failed: %s", exc)
            return {}
        return result if isinstance(result, dict) else {}

    direct_result = rank_ic_snapshot(aligned_signal, aligned_returns)
    try:
        inverse_signal = -aligned_signal
    except Exception as exc:
        payload["error"] = f"Could not invert signal values: {exc}"
        logger.warning("Directional IC skipped: %s", payload["error"])
        return payload

    inverse_result = rank_ic_snapshot(inverse_signal, aligned_returns)
    train_result = rank_ic_snapshot(train_signal, train_returns)
    inverse_train_result = rank_ic_snapshot(-train_signal, train_returns)
    test_result = rank_ic_snapshot(test_signal, test_returns)
    inverse_test_result = rank_ic_snapshot(-test_signal, test_returns)

    direct_mean_ic = _finite_float(direct_result.get("mean_ic"))
    inverse_mean_ic = _finite_float(inverse_result.get("mean_ic"))
    train_mean_ic = _finite_float(train_result.get("mean_ic"))
    train_p_value = _finite_float(train_result.get("p_value"))
    train_num_periods = _plain_period_count(train_result.get("num_periods"))
    inverse_train_mean_ic = _finite_float(inverse_train_result.get("mean_ic"))
    test_mean_ic = _finite_float(test_result.get("mean_ic"))
    test_p_value = _finite_float(test_result.get("p_value"))
    test_num_periods = _plain_period_count(test_result.get("num_periods"))
    inverse_test_mean_ic = _finite_float(inverse_test_result.get("mean_ic"))

    payload.update(
        {
            "direct_mean_ic": direct_mean_ic,
            "inverse_mean_ic": inverse_mean_ic,
            "absolute_ic": abs(direct_mean_ic) if direct_mean_ic is not None else None,
            "train_mean_ic": train_mean_ic,
            "train_p_value": train_p_value,
            "train_num_periods": train_num_periods,
            "inverse_train_mean_ic": inverse_train_mean_ic,
            "test_mean_ic": test_mean_ic,
            "test_p_value": test_p_value,
            "test_num_periods": test_num_periods,
            "inverse_test_mean_ic": inverse_test_mean_ic,
        }
    )

    # The IS sign is the ONLY input to the orientation decision.  OOS sample
    # availability and values are deliberately excluded from this condition:
    # they may gate OOS reporting below, but can never create, suppress, or
    # change the IS-derived decision.
    decision_reason = None
    if train_num_periods < minimum_periods:
        detected_sign = 0
        decision = "none"
        decision_reason = "insufficient_in_sample_periods"
        logger.warning(
            "Directional IC decision suppressed: insufficient in-sample periods "
            f"({train_num_periods} < {minimum_periods})"
        )
    elif train_mean_ic is None:
        detected_sign = 0
        decision = "none"
        decision_reason = "unavailable_in_sample_ic"
        logger.warning("Directional IC decision suppressed: in-sample mean IC is unavailable")
    elif train_mean_ic < 0:
        detected_sign = -1
        decision = "negative"
    elif train_mean_ic > 0:
        detected_sign = 1
        decision = "positive"
    else:
        detected_sign = 0
        decision = "none"
        decision_reason = "zero_in_sample_ic"
        logger.warning("Directional IC decision suppressed: in-sample mean IC is zero")

    ic_limit = _finite_float(ic_threshold)
    p_limit = _finite_float(p_value_threshold)
    is_significant = bool(
        detected_sign != 0
        and train_mean_ic is not None
        and train_p_value is not None
        and ic_limit is not None
        and p_limit is not None
        and ic_limit >= 0
        and 0 <= p_limit <= 1
        and abs(train_mean_ic) >= ic_limit
        and train_p_value <= p_limit
    )

    # OOS sufficiency gates only OOS reporting/application.  The IS decision
    # above remains valid even when this branch suppresses oriented OOS values.
    oriented_test_mean_ic = None
    oriented_test_p_value = None
    oos_reason = None
    if test_num_periods < minimum_periods:
        oos_reason = "insufficient_out_of_sample_periods"
        logger.warning(
            "Directional IC out-of-sample reporting suppressed: insufficient "
            f"out-of-sample periods ({test_num_periods} < {minimum_periods})"
        )
    elif test_mean_ic is None:
        oos_reason = "unavailable_out_of_sample_ic"
        logger.warning("Directional IC out-of-sample reporting suppressed: OOS mean IC is unavailable")
    elif detected_sign == 0:
        oos_reason = "no_in_sample_decision"
        logger.warning("Directional IC out-of-sample reporting suppressed: no in-sample decision")
    elif detected_sign == 1:
        oriented_test_mean_ic = test_mean_ic
        oriented_test_p_value = test_p_value
    else:
        oriented_test_mean_ic = _finite_float(inverse_test_result.get("mean_ic"))
        oriented_test_p_value = _finite_float(inverse_test_result.get("p_value"))

    if len(test_dates) > 0:
        try:
            split_date = pd.Timestamp(test_dates[0]).isoformat()
        except (TypeError, ValueError):
            split_date = str(test_dates[0])
    else:
        split_date = None

    payload.update(
        {
            "oriented_test_mean_ic": oriented_test_mean_ic,
            "oriented_test_p_value": oriented_test_p_value,
            "detected_sign": int(detected_sign),
            "decision": decision,
            "decision_reason": decision_reason,
            "is_significant": bool(is_significant),
            "should_invert": bool(detected_sign == -1 and is_significant),
            "oos_reason": oos_reason,
            "split_date": split_date,
        }
    )
    return payload


def evaluate_signal(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Full evaluation of a signal: rank-IC statistics, turnover, and portfolio metrics.

    Combines:
      - the full rank-IC statistics from :func:`compute_rank_ic`;
      - annualized turnover from :func:`compute_signal_turnover` (key
        ``"turnover"``);
      - portfolio metrics from the sibling ``quant_metrics`` module, when that
        module is importable and functional. The import happens lazily inside
        this function: on ImportError the portfolio keys are simply omitted —
        evaluation never hard-depends on the module. Even when the module
        exists, the calls are wrapped defensively: a partial or buggy
        implementation can never break the core evaluation, but any such
        integration failure is logged at WARNING so it cannot pass unnoticed.
        Portfolio keys are additive only: rank-IC keys (including
        ``num_periods``, which counts IC periods) are never overwritten by
        same-named portfolio metrics.

    Args:
        signal_values: DataFrame of signal values (rows=dates, cols=stocks).
        forward_returns: DataFrame of forward returns (rows=dates, cols=stocks).
        periods_per_year: Periods per year used to annualize turnover and
            portfolio metrics (default 252 for daily data).

    Returns:
        Dict with all rank-IC statistics, ``"turnover"``, and (when
        ``quant_metrics`` is available) its portfolio metric keys such as
        ``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``.
    """
    result: dict[str, Any] = dict(compute_rank_ic(signal_values, forward_returns))
    result["turnover"] = compute_signal_turnover(
        signal_values, periods_per_year=periods_per_year
    )

    try:
        from signal_discovery_workflow.quant_metrics import (  # noqa: PLC0415
            build_portfolio_weights,
            compute_portfolio_metrics,
            portfolio_returns_from_weights,
        )
    except ImportError as e:
        logger.debug(f"quant_metrics not available; skipping portfolio metrics: {e}")
    else:
        try:
            weights = build_portfolio_weights(
                signal_values, top_quantile=0.2, bottom_quantile=0.2
            )
            portfolio_returns = portfolio_returns_from_weights(weights, forward_returns)
            portfolio_metrics = compute_portfolio_metrics(
                portfolio_returns, periods_per_year=periods_per_year
            )
            if isinstance(portfolio_metrics, dict):
                # Additive merge: never clobber rank-IC statistics with
                # same-named portfolio keys (e.g. "num_periods").
                result.update(
                    {k: v for k, v in portfolio_metrics.items() if k not in result}
                )
        except Exception as e:
            logger.warning(f"Portfolio metrics unavailable; evaluate_signal returned partial results: {e}")

    return result


def extract_code_from_response(code_response: str) -> str:
    """Extract Python code from markdown code blocks, falling back to raw text."""
    code_blocks = re.findall(r"```python\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    code_blocks = re.findall(r"```\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    return code_response


# Operator function names — derived from the actual operators loaded from
# calculator.json so this set always stays in sync with available operators.
OPERATOR_NAMES = set(_OPERATOR_FUNCTIONS.keys())

# Standard data field names. Order matters: when a signal function uses a
# generic parameter name (e.g. `x`, `data`, `prices`), parameters are filled
# positionally from this list.
STANDARD_FIELDS = ["Close", "Volume", "High", "Low", "Open"]


def _is_dataframe_param(p: inspect.Parameter) -> bool:
    """Whether this parameter expects a DataFrame to be passed in."""
    ann = p.annotation
    if ann is pd.DataFrame:
        return True
    if isinstance(ann, str) and "DataFrame" in ann:
        return True
    if ann is inspect.Parameter.empty:
        # No type hint — treat as DataFrame only if there's no scalar default.
        if p.default is inspect.Parameter.empty:
            return True
        if p.default is None or isinstance(p.default, pd.DataFrame):
            return True
        return False
    return False


def _resolve_signal_args(
    sig: inspect.Signature,
    stock_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Map signal function parameters to stock data DataFrames.

    Only DataFrame parameters are filled. Numeric/string parameters
    (lookback windows, thresholds, etc.) are left to use their defaults so
    they don't get a DataFrame mistakenly passed in.

    DataFrame parameter resolution:
      1. Exact name match (e.g. `Close` -> Close).
      2. Case-insensitive substring match in either direction
         (e.g. `closing_price` -> Close, `vol` -> Volume).
      3. Positional fallback for unmatched params, drawing from
         STANDARD_FIELDS in order, skipping fields already assigned.
    """
    available_fields = [f for f in STANDARD_FIELDS if f in stock_data]
    df_params = [name for name, p in sig.parameters.items() if _is_dataframe_param(p)]

    kwargs: dict[str, pd.DataFrame] = {}
    used_fields: set[str] = set()
    unmatched: list[str] = []

    for param in df_params:
        match: str | None = None
        p_lower = param.lower()
        for field in available_fields:
            f_lower = field.lower()
            if p_lower == f_lower or f_lower in p_lower or p_lower in f_lower:
                match = field
                break

        if match and match not in used_fields:
            kwargs[param] = stock_data[match]
            used_fields.add(match)
        else:
            unmatched.append(param)

    fallback_pool = [f for f in available_fields if f not in used_fields]
    for param in unmatched:
        if not fallback_pool:
            break
        kwargs[param] = stock_data[fallback_pool.pop(0)]

    return kwargs


_SMART_UNICODE = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }
)


def _detect_helpers(candidates: list[tuple[str, Any]]) -> set[str]:
    """
    Among a set of candidate signal functions, return the names of those that
    are *called by* another candidate (i.e. helpers, not signals themselves).
    """
    helpers: set[str] = set()
    names = {name for name, _ in candidates}
    for _, fn in candidates:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for name in names:
            if name != fn.__name__ and re.search(rf"\b{re.escape(name)}\s*\(", src):
                helpers.add(name)
    return helpers


def execute_signal_code(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int = 5,
) -> tuple[pd.DataFrame, str] | None:
    """
    Execute self-contained signal code and call its signal function(s).

    The ``code`` string is expected to be a complete, runnable Python module
    that defines its own imports, operator functions, and signal functions.
    No pre-seeded globals are injected: signal functions receive their input
    DataFrames as arguments, and operators come from definitions in the
    module itself. This makes the saved ``signal_code`` portable: it can be
    copy-pasted into any Python session and run as-is.

    ``selection_periods`` controls the forward-return horizon used to choose
    the best signal when multiple signal functions are present.

    Sandboxing:
        When enabled (``SANDBOX_ENABLED`` or env ``SIGNAL_SANDBOX_ENABLED``),
        execution is delegated to ``sandbox.execute_signal_code_sandboxed``
        (same contract, no side effects from untrusted code). Any sandbox
        failure is logged at WARNING and yields None; a missing sandbox module
        falls back to the legacy path below. When disabled (the default), the
        legacy in-process path runs exactly as before.

    Returns the (signal_values, selected_signal_name) tuple of the highest-IC
    signal among those defined in ``code``, or None on failure.
    """
    if _sandbox_enabled():
        try:
            from signal_discovery_workflow.sandbox import execute_signal_code_sandboxed
        except ImportError as e:
            logger.warning(f"sandbox module unavailable ({type(e).__name__}); falling back to legacy execution: {e}")
        else:
            try:
                logger.info(
                    "Executing signal code in sandbox "
                    "(allow_subprocess=False: in-process restrictions active, subprocess isolation off)"
                )
                return execute_signal_code_sandboxed(
                    code, stock_data, selection_periods=selection_periods
                )
            except Exception as e:
                logger.warning(f"Sandboxed signal execution failed ({type(e).__name__}); returning None")
                return None

    namespace: dict[str, Any] = {}

    try:
        # Smart-quote/dash normalization — some LLMs emit U+2018/U+2019 etc.
        # that would otherwise raise SyntaxError.
        exec(code.translate(_SMART_UNICODE), namespace)

        # Candidate signals: user-defined functions that aren't operators.
        # `__code__` filters out modules/builtins; OPERATOR_NAMES excludes
        # operator implementations inlined at the top of the module.
        candidates = [
            (name, obj)
            for name, obj in namespace.items()
            if not name.startswith("_")
            and callable(obj)
            and hasattr(obj, "__code__")
            and name not in OPERATOR_NAMES
        ]

        if not candidates:
            logger.warning("No signal functions found in the code")
            return None

        helpers = _detect_helpers(candidates)
        signal_functions = [(n, f) for n, f in candidates if n not in helpers] or candidates

        logger.info(
            f"Found {len(signal_functions)} signal function(s): "
            f"{[f[0] for f in signal_functions]}"
            + (f" (skipping helpers: {sorted(helpers)})" if helpers else "")
        )

        if _operator_cache_enabled():
            # Wrap operator callables IN PLACE so the signal functions'
            # __globals__ (this namespace) resolve to the memoized wrappers.
            # Reached by the legacy path only: the sandbox delegation above
            # returns before this and runs code in its own restricted
            # namespace, which does not share this cache.
            _install_wrapped_operators(namespace)

        best_result = None
        best_ic = None
        best_name = None

        # Numpy / pandas emit RuntimeWarnings when rolling-window operators
        # (TS_Std, TS_Var, TS_Skew, etc.) hit windows that contain NaN values.
        # The result is correctly NaN; the warnings are noise.
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            for func_name, signal_func in signal_functions:
                try:
                    sig = inspect.signature(signal_func)
                    kwargs = _resolve_signal_args(sig, stock_data)
                    df_param_count = sum(
                        1 for p in sig.parameters.values() if _is_dataframe_param(p)
                    )

                    if len(kwargs) == df_param_count:
                        result = signal_func(**kwargs)
                    elif len(sig.parameters) == 0:
                        result = signal_func()
                    else:
                        logger.warning(
                            f"Cannot determine args for {func_name} "
                            f"(params={list(sig.parameters)}), skipping"
                        )
                        continue

                    if isinstance(result, pd.Series):
                        result = result.to_frame()

                    if isinstance(result, pd.DataFrame):
                        # Cached so multiple candidate signals in one module
                        # reuse the same forward-return frame.
                        forward_ret = _get_cached_forward_returns(
                            stock_data["Close"], selection_periods
                        )
                        valid_dates = result.dropna(how="all").index
                        sample_ics = []

                        for date in valid_dates:
                            if date not in forward_ret.index:
                                continue
                            fr = result.loc[date].dropna()
                            rr = forward_ret.loc[date].dropna()
                            common = fr.index.intersection(rr.index)
                            if len(common) < 10:
                                continue
                            f_vals = fr[common].values
                            r_vals = rr[common].values
                            if np.std(f_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                                continue
                            try:
                                corr, _ = stats.spearmanr(f_vals, r_vals)
                                if not np.isnan(corr):
                                    sample_ics.append(corr)
                            except Exception:
                                pass

                        if sample_ics:
                            mean_ic = abs(np.mean(sample_ics))
                            logger.info(f"  {func_name}: |IC| = {mean_ic:.4f}")
                            if best_ic is None or mean_ic > best_ic:
                                best_ic = mean_ic
                                best_result = result
                                best_name = func_name
                        elif best_result is None:
                            best_result = result
                            best_name = func_name

                except Exception as e:
                    logger.warning(f"Error executing {func_name}: {e}")
                    continue

        if best_result is not None:
            ic_str = f"{best_ic:.4f}" if best_ic is not None else "N/A"
            logger.info(f"Selected best signal: {best_name} with |IC| = {ic_str}")
            return best_result, best_name

        return None

    except SyntaxError as e:
        # Show the offending source line(s) so we can diagnose what the LLM
        # produced (truncated string, smart quote we missed, etc.).
        lineno = getattr(e, "lineno", None) or 0
        lines = code.splitlines()
        start, end = max(0, lineno - 3), min(len(lines), lineno + 2)
        snippet = "\n".join(
            f"  {i + 1:4d} {'>>' if i + 1 == lineno else '  '} {line}"
            for i, line in enumerate(lines[start:end], start=start)
        )
        logger.error(f"SyntaxError in generated signal code: {e}\n{snippet}")
        return None
    except Exception as e:
        logger.error(f"Error executing signal code: {e}")
        logger.debug(traceback.format_exc())
        return None
