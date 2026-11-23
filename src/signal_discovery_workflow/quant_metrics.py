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

"""Quantitative portfolio metrics and out-of-sample signal validation.

The functions in this module operate on date-by-stock pandas DataFrames.  They
intentionally keep portfolio construction separate from return calculation so
that the long/short convention is explicit and testable.
"""

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def build_portfolio_weights(
    signal_values: pd.DataFrame,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> pd.DataFrame:
    """Build an equal-weight, dollar-neutral long/short portfolio.

    For each date, the strongest signals form a long bucket with total weight
    ``+0.5`` and the weakest signals form a short bucket with total weight
    ``-0.5``.  Signals equal to ``NaN`` are excluded.  Bucket sizes are the
    quantiles rounded up to at least one name; a date is left uninvested (all
    zero weights) when those two buckets cannot be filled with distinct names.
    Ties are resolved deterministically by original column order.

    Args:
        signal_values: Signal matrix with dates on rows and stocks on columns.
        top_quantile: Fraction of valid names assigned to the long bucket.
        bottom_quantile: Fraction of valid names assigned to the short bucket.

    Returns:
        A DataFrame of weights with the same shape and labels as
        ``signal_values``.

    Raises:
        ValueError: If either quantile is outside ``(0, 1)`` or the two
            quantiles sum to one or more.
    """
    for name, quantile in (("top_quantile", top_quantile), ("bottom_quantile", bottom_quantile)):
        if not math.isfinite(quantile) or quantile <= 0 or quantile >= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if top_quantile + bottom_quantile >= 1:
        raise ValueError("top_quantile + bottom_quantile must be less than 1")

    weights = pd.DataFrame(0.0, index=signal_values.index, columns=signal_values.columns)

    for position, (_, signal_row) in enumerate(signal_values.iterrows()):
        valid_signal = signal_row.dropna()
        valid_count = len(valid_signal)
        if valid_count == 0:
            continue

        top_count = max(1, math.ceil(round(valid_count * top_quantile, 12)))
        bottom_count = max(1, math.ceil(round(valid_count * bottom_quantile, 12)))
        if top_count + bottom_count > valid_count:
            continue

        # A single stable ordering makes the two buckets disjoint even when
        # signal values tie (separate nlargest/nsmallest calls can otherwise
        # select the same tied names).
        ordered_signal = valid_signal.sort_values(ascending=False, kind="mergesort")
        top_names = ordered_signal.index[:top_count]
        bottom_names = ordered_signal.index[-bottom_count:]

        weights.iloc[position, weights.columns.get_indexer(top_names)] = 0.5 / top_count
        weights.iloc[position, weights.columns.get_indexer(bottom_names)] = -0.5 / bottom_count

    return weights


def portfolio_returns_from_weights(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Calculate portfolio returns from fixed weights and forward returns.

    Dates and stocks are aligned on their intersections.  A date has a valid
    portfolio return only when it has at least one non-zero position and every
    held position has a finite forward return.  Dates without a valid return
    are represented by ``NaN`` rather than by a partial portfolio return.

    Args:
        weights: Portfolio weights with dates on rows and stocks on columns.
        forward_returns: Forward returns with dates on rows and stocks on columns.

    Returns:
        A float Series of portfolio returns indexed by aligned date.
    """
    common_dates = weights.index.intersection(forward_returns.index)
    common_stocks = weights.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.Series(dtype=float, name="portfolio_return")

    aligned_weights = weights.loc[common_dates, common_stocks].astype(float)
    aligned_returns = forward_returns.loc[common_dates, common_stocks].astype(float)
    aligned_weights = aligned_weights.replace([np.inf, -np.inf], np.nan)
    aligned_returns = aligned_returns.replace([np.inf, -np.inf], np.nan)

    active_positions = aligned_weights.notna() & aligned_weights.ne(0.0)
    has_positions = active_positions.any(axis=1)
    missing_active_return = (active_positions & aligned_returns.isna()).any(axis=1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1, min_count=1)

    portfolio_returns.loc[~has_positions] = np.nan
    portfolio_returns.loc[missing_active_return] = np.nan
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def compute_turnover(weights: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Compute annualized two-way portfolio turnover.

    The convention used here is the *full absolute-weight-change* convention:
    for each consecutive date pair, turnover is ``sum(abs(w_t - w_(t-1)))``
    across all stocks.  The mean pairwise turnover is then multiplied by
    ``periods_per_year``.  For a dollar-neutral, gross-one book the
    pre-annualization turnover lies in ``[0.0, 2.0]`` per period, so the
    annualized value lies in ``[0.0, 2.0 * periods_per_year]``.  A ``+0.5``
    long flipping to a ``-0.5`` short contributes ``1.0`` to one pair's
    turnover.  Pairs containing missing weights are skipped rather than
    treated as zero changes.

    Args:
        weights: Portfolio weights with dates on rows and stocks on columns.
        periods_per_year: Number of periods in one year.

    Returns:
        Annualized turnover, or ``0.0`` when there are fewer than two complete
        consecutive weight rows.
    """
    if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    if len(weights) < 2:
        return 0.0

    numeric_weights = weights.astype(float).replace([np.inf, -np.inf], np.nan)
    absolute_changes = numeric_weights.diff().abs()
    complete_pairs = absolute_changes.notna().all(axis=1)
    if not complete_pairs.any():
        return 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pairwise_turnover = absolute_changes.loc[complete_pairs].sum(axis=1)
        annualized_turnover = float(pairwise_turnover.mean() * periods_per_year)

    return annualized_turnover if math.isfinite(annualized_turnover) else 0.0


def compute_portfolio_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Compute portfolio performance and risk metrics.

    ``annual_return`` is the arithmetic annualization
    ``mean(returns) * periods_per_year``.  ``cagr`` is the geometric
    annualization ``prod(1 + returns) ** (periods_per_year / N) - 1``.  CAGR
    is reported as ``0.0`` for an empty series or when cumulative equity is
    non-positive.  Annual volatility is the sample standard deviation
    (``ddof=1``) multiplied by ``sqrt(periods_per_year)`` when the series has
    real dispersion.

    The dispersion tolerance is
    ``max(1e-12, abs(mean(returns)) * 1e-9)``.  A series whose dispersion is
    below that tolerance is treated as riskless: Sharpe, Sortino, and annual
    volatility are reported as ``0.0``.  This also prevents floating-point
    residue in nominally constant series from producing meaningless ratios.

    Sharpe ratio is
    ``(mean(returns) - risk_free_rate / periods_per_year) /
    std(returns, ddof=1) * sqrt(periods_per_year)``.

    Sortino ratio uses the per-period target
    ``risk_free_rate / periods_per_year`` and textbook downside deviation:
    ``sqrt(mean(min(0, returns - target) ** 2))`` over the full sample.
    Sortino is ``(mean(returns) - target) / downside_deviation *
    sqrt(periods_per_year)``.  The series must have real dispersion and its
    downside deviation must exceed the same tolerance.  If no downside
    deviation is observed, the ratio is undefined and is reported as the
    JSON-safe finite value ``0.0``.

    Maximum drawdown is the minimum of ``equity / running_max(equity) - 1``,
    where equity is the cumulative product of ``1 + returns``.  Initial
    capital ``1.0`` is included in the running maximum, so a loss on the
    first observation is included in the drawdown.  Drawdown is reported as a
    non-positive number.  Calmar ratio is
    ``annual_return / abs(max_drawdown)``.

    Missing and non-finite return observations are excluded.  Undefined ratios
    and sample-risk statistics are defined as ``0.0``.  For an empty series,
    every numeric metric is ``0.0`` (except ``num_periods``, which is the
    valid observation count).  ``best_day`` and ``worst_day`` are raw,
    unannualized period returns.

    Args:
        returns: Period portfolio returns.
        periods_per_year: Number of periods in one year.
        risk_free_rate: Annual risk-free rate expressed as a decimal.

    Returns:
        Dictionary containing the portfolio metrics.
    """
    if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    numeric_returns = pd.to_numeric(pd.Series(returns), errors="coerce")
    numeric_returns = numeric_returns[np.isfinite(numeric_returns)]
    values = numeric_returns.to_numpy(dtype=float)
    num_periods = len(values)

    metrics: dict[str, Any] = {
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "annual_return": 0.0,
        "cagr": 0.0,
        "annual_volatility": 0.0,
        "total_return": 0.0,
        "num_periods": num_periods,
        "best_day": 0.0,
        "worst_day": 0.0,
    }
    if num_periods == 0:
        return metrics

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        mean_return = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1)) if num_periods > 1 else 0.0
        per_period_target = risk_free_rate / periods_per_year
        excess_mean = mean_return - per_period_target
        vol_tolerance = max(1e-12, abs(mean_return) * 1e-9)
        has_real_variance = sample_std > vol_tolerance and math.isfinite(sample_std)

        sharpe_ratio = 0.0
        if has_real_variance:
            sharpe_ratio = float((excess_mean / sample_std) * math.sqrt(periods_per_year))

        downside = np.minimum(values - per_period_target, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        downside_tolerance = max(1e-12, abs(mean_return) * 1e-9)
        sortino_ratio = 0.0
        if has_real_variance and downside_deviation > downside_tolerance and math.isfinite(downside_deviation):
            sortino_ratio = float((mean_return - per_period_target) / downside_deviation * math.sqrt(periods_per_year))

        growth_factors = 1.0 + values
        equity = np.cumprod(growth_factors)
        running_peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
        drawdowns = equity / running_peak - 1.0
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

        final_equity = float(equity[-1])
        total_return = final_equity - 1.0
        annual_return = float(mean_return * periods_per_year)
        if final_equity > 0.0 and math.isfinite(final_equity):
            cagr = float(final_equity ** (periods_per_year / num_periods) - 1.0)
        else:
            cagr = 0.0
        annual_volatility = float(sample_std * math.sqrt(periods_per_year)) if has_real_variance else 0.0
        calmar_ratio = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    if not all(
        math.isfinite(value)
        for value in (
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            annual_return,
            cagr,
            annual_volatility,
            total_return,
        )
    ):
        raise ValueError("returns produced a non-finite portfolio metric")

    metrics.update(
        {
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": calmar_ratio,
            "annual_return": annual_return,
            "cagr": float(cagr),
            "annual_volatility": annual_volatility,
            "total_return": float(total_return),
            "best_day": float(np.max(values)),
            "worst_day": float(np.min(values)),
        }
    )
    return metrics


def _compute_rank_ic_fallback(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict[str, Any]:
    """Compute cross-sectional rank IC without importing ``signal_evaluator``."""
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return {"mean_ic": None, "p_value": None, "num_periods": 0}

    aligned_signal = signal_values.loc[common_dates, common_stocks]
    aligned_returns = forward_returns.loc[common_dates, common_stocks]
    ic_values: list[float] = []

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for date in common_dates:
            signal_row = aligned_signal.loc[date].dropna()
            returns_row = aligned_returns.loc[date].dropna()
            common_names = signal_row.index.intersection(returns_row.index)
            if len(common_names) < 10:
                continue

            signal_values_row = signal_row.loc[common_names].to_numpy(dtype=float)
            returns_values_row = returns_row.loc[common_names].to_numpy(dtype=float)
            if (
                np.std(signal_values_row) < 1e-10
                or np.std(returns_values_row) < 1e-10
                or not np.isfinite(signal_values_row).all()
                or not np.isfinite(returns_values_row).all()
            ):
                continue

            correlation, _ = stats.spearmanr(signal_values_row, returns_values_row)
            if np.isfinite(correlation):
                ic_values.append(float(correlation))

    if not ic_values:
        return {"mean_ic": None, "p_value": None, "num_periods": 0}

    ic_array = np.asarray(ic_values, dtype=float)
    mean_ic = float(np.mean(ic_array))
    ic_std = float(np.std(ic_array, ddof=0))
    num_periods = len(ic_array)
    if ic_std > 0:
        t_stat = mean_ic / (ic_std / math.sqrt(num_periods))
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=num_periods - 1)))
    else:
        p_value = None

    return {"mean_ic": mean_ic, "p_value": p_value, "num_periods": num_periods}


def cross_validate_signal(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Evaluate a signal chronologically in-sample and out-of-sample.

    Dates are sorted chronologically and split at ``floor(len(dates) *
    train_ratio)``.  ``train_periods`` and ``test_periods`` report the number
    of aligned dates assigned to each split and therefore sum to the number
    of aligned dates.  Rank IC is computed separately and cross-sectionally
    within each date, then averaged across usable dates.

    A signal is classified as overfit when any of these rules applies:

    1. train and test mean IC have opposite signs;
    2. ``abs(test_mean_ic) < 0.5 * abs(train_mean_ic)``; or
    3. either split has fewer than 30 usable IC dates.

    The 30-date rule uses usable IC dates internally, which can be fewer than
    the dates reported by ``train_periods`` and ``test_periods`` when rows have
    missing values, fewer than 10 shared names, or zero variance.  Missing IC
    or p-values are represented by ``None``; a missing IC also makes the
    result overfit.  ``split_date`` is the first out-of-sample date in ISO
    format.

    Args:
        signal_values: Signal matrix with dates on rows and stocks on columns.
        forward_returns: Forward-return matrix with dates on rows and stocks on columns.
        train_ratio: Fraction of aligned dates assigned to the training split.

    Returns:
        Dictionary containing in-sample, out-of-sample, and overfit diagnostics.

    Raises:
        ValueError: If ``train_ratio`` is not strictly between zero and one.
    """
    if not math.isfinite(train_ratio) or train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")

    common_dates = signal_values.index.intersection(forward_returns.index).sort_values()
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    train_count = int(len(common_dates) * train_ratio)
    train_dates = common_dates[:train_count]
    test_dates = common_dates[train_count:]

    try:
        from signal_discovery_workflow.signal_evaluator import compute_rank_ic
    except ImportError:
        compute_rank_ic = _compute_rank_ic_fallback

    aligned_signal = signal_values.loc[common_dates, common_stocks]
    aligned_returns = forward_returns.loc[common_dates, common_stocks]
    train_signal = aligned_signal.loc[train_dates]
    train_returns = aligned_returns.loc[train_dates]
    test_signal = aligned_signal.loc[test_dates]
    test_returns = aligned_returns.loc[test_dates]

    train_result = compute_rank_ic(train_signal, train_returns)
    test_result = compute_rank_ic(test_signal, test_returns)

    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        return numeric_value if math.isfinite(numeric_value) else None

    train_mean_ic = _optional_float(train_result.get("mean_ic"))
    test_mean_ic = _optional_float(test_result.get("mean_ic"))
    train_p_value = _optional_float(train_result.get("p_value"))
    test_p_value = _optional_float(test_result.get("p_value"))
    train_usable_periods = int(train_result.get("num_periods", 0) or 0)
    test_usable_periods = int(test_result.get("num_periods", 0) or 0)

    ic_decay = float(test_mean_ic - train_mean_ic) if train_mean_ic is not None and test_mean_ic is not None else None
    opposite_signs = train_mean_ic is not None and test_mean_ic is not None and train_mean_ic * test_mean_ic < 0
    ic_halved = train_mean_ic is not None and test_mean_ic is not None and abs(test_mean_ic) < 0.5 * abs(train_mean_ic)
    too_little_data = train_usable_periods < 30 or test_usable_periods < 30
    missing_ic = train_mean_ic is None or test_mean_ic is None
    is_overfit = bool(opposite_signs or ic_halved or too_little_data or missing_ic)

    split_date = None
    if len(test_dates) > 0:
        try:
            split_date = pd.Timestamp(test_dates[0]).isoformat()
        except (TypeError, ValueError):
            split_date = str(test_dates[0])

    return {
        "train_mean_ic": train_mean_ic,
        "train_p_value": train_p_value,
        "test_mean_ic": test_mean_ic,
        "test_p_value": test_p_value,
        "ic_decay": ic_decay,
        "is_overfit": is_overfit,
        "train_periods": len(train_dates),
        "test_periods": len(test_dates),
        "split_date": split_date,
    }
