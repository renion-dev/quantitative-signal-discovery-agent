# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pure configuration and sizing logic for the dashboard paper-trading account."""

from __future__ import annotations

import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_OPEN_POSITIONS = 3
DEFAULT_STARTING_BALANCE = 10000.0
CONFIG_ENV_VAR = "PAPER_TRADING_CONFIG"
REPO_ROOT = Path(__file__).resolve().parents[2]

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "max_open_positions",
        "starting_balance",
        "max_position_size_pct",
        "max_position_size_amount",
    }
)


class PaperConfigError(ValueError):
    """Raised when the paper-trading configuration cannot be read or validated."""


@dataclass(frozen=True)
class PaperConfig:
    """Validated paper-trading limits and the source metadata used to load them."""

    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    starting_balance: float = DEFAULT_STARTING_BALANCE
    max_position_size_pct: float | None = None
    max_position_size_amount: float | None = None
    path: Path | None = None
    file_missing: bool = False
    warnings: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()


def resolve_paper_config_path(path: str | Path | None = None) -> Path:
    """Resolve the paper config path, anchoring relative paths to the repository root.

    ``PAPER_TRADING_CONFIG`` overrides the default config location when ``path`` is not supplied.
    """

    if path is None:
        override = os.environ.get(CONFIG_ENV_VAR, "").strip()
        candidate = Path(override).expanduser() if override else Path("configs/paper-trading.yml")
    else:
        candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    """Parse the small, flat YAML subset used by the paper config.

    This fallback is intentionally conservative. It supports comments, nulls, booleans, and numeric
    scalars, while rejecting nested YAML structures and malformed ``key: value`` lines.
    """

    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise PaperConfigError(f"line {line_number}: expected 'key: value'")
        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip()
        if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
            key = key[1:-1]
        if not key:
            raise PaperConfigError(f"line {line_number}: empty key")

        value_text = _strip_yaml_comment(raw_value.strip()).strip()
        if not value_text:
            value: Any = ""
        elif value_text.lower() in {"null", "~"}:
            value = None
        elif value_text.lower() == "true":
            value = True
        elif value_text.lower() == "false":
            value = False
        elif re.fullmatch(r"[+-]?\d+", value_text):
            value = int(value_text)
        else:
            try:
                value = float(value_text)
            except ValueError:
                value = value_text
        values[key] = value
    return values


def _strip_yaml_comment(value: str) -> str:
    """Remove a YAML-style inline comment while respecting quoted scalars."""

    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def _parse_yaml_document(text: str) -> dict[str, Any]:
    """Parse YAML with PyYAML when available, otherwise use the flat-file fallback."""

    try:
        import yaml
    except ImportError:
        return _parse_flat_yaml(text)

    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        raise PaperConfigError(f"invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PaperConfigError("the YAML document must contain a top-level mapping")
    return data


def _validate_non_negative_integer(value: Any, key: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PaperConfigError(f"{key} must be a non-negative integer")
    return value


def _validate_positive_number(value: Any, key: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperConfigError(f"{key} must be a number greater than zero")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise PaperConfigError(f"{key} must be a finite number greater than zero")
    return number


def load_paper_config(path: str | Path | None = None) -> PaperConfig:
    """Load and validate the paper-trading configuration.

    A missing file is not an error: defaults are returned with ``file_missing`` set to ``True`` so
    the dashboard can warn once. Invalid YAML, malformed values, or unknown data types raise a
    clear :class:`PaperConfigError`. Unknown keys produce a warning without preventing startup.
    """

    resolved_path = resolve_paper_config_path(path)
    if not resolved_path.exists():
        return PaperConfig(path=resolved_path, file_missing=True)

    try:
        text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PaperConfigError(f"cannot read {resolved_path}: {exc}") from exc

    try:
        raw_config = _parse_yaml_document(text)
    except PaperConfigError as exc:
        raise PaperConfigError(f"invalid paper-trading config {resolved_path}: {exc}") from exc

    unknown_keys = tuple(sorted(str(key) for key in set(raw_config) - _ALLOWED_CONFIG_KEYS))
    warning_messages: list[str] = []
    if unknown_keys:
        message = f"unknown paper-trading config key(s): {', '.join(unknown_keys)}"
        warning_messages.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    max_open_positions = _validate_non_negative_integer(
        raw_config.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS),
        "max_open_positions",
        resolved_path,
    )
    starting_balance = _validate_positive_number(
        raw_config.get("starting_balance", DEFAULT_STARTING_BALANCE),
        "starting_balance",
        resolved_path,
    )

    raw_pct = raw_config.get("max_position_size_pct")
    max_position_size_pct = (
        None
        if raw_pct is None
        else _validate_positive_number(raw_pct, "max_position_size_pct", resolved_path)
    )

    raw_amount = raw_config.get("max_position_size_amount")
    max_position_size_amount = (
        None
        if raw_amount is None
        else _validate_positive_number(raw_amount, "max_position_size_amount", resolved_path)
    )

    return PaperConfig(
        max_open_positions=max_open_positions,
        starting_balance=starting_balance,
        max_position_size_pct=max_position_size_pct,
        max_position_size_amount=max_position_size_amount,
        path=resolved_path,
        warnings=tuple(warning_messages),
        unknown_keys=unknown_keys,
    )


def _finite_number(value: Any) -> float | None:
    """Return a finite number, ignoring booleans and non-numeric junk."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def free_balance(config: PaperConfig, paper_state: Mapping[str, Any] | None) -> float:
    """Calculate the paper account's free balance.

    The exact formula is ``starting_balance + realized_pnl - open_notional``. Realized PnL is the
    cumulative PnL of closed trades and open notional is the sum of ``abs(qty * entry)`` across
    open positions. Missing and malformed state values are ignored.
    """

    realized_pnl = 0.0
    if isinstance(paper_state, Mapping):
        realized_pnl = _finite_number(paper_state.get("cumulative_closed_pnl", 0.0)) or 0.0

    open_notional = 0.0
    if isinstance(paper_state, Mapping):
        positions = paper_state.get("positions", [])
        if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
            for position in positions:
                if not isinstance(position, Mapping):
                    continue
                quantity = _finite_number(position.get("qty"))
                entry = _finite_number(position.get("entry"))
                if quantity is None or entry is None:
                    continue
                notional = abs(quantity * entry)
                if math.isfinite(notional):
                    open_notional += notional

    return config.starting_balance + realized_pnl - open_notional


def position_notional_cap(config: PaperConfig, free_balance: float) -> float | None:
    """Resolve the effective per-position notional cap.

    Percentage and fixed-amount limits are both supported. When both exist, the lower cap wins.
    A percentage cap uses the supplied free balance and is clamped at zero.
    """

    pct_cap: float | None = None
    if config.max_position_size_pct is not None:
        balance = _finite_number(free_balance)
        balance = 0.0 if balance is None else balance
        pct_cap = max(0.0, balance * config.max_position_size_pct / 100.0)

    amount_cap = config.max_position_size_amount
    if pct_cap is None and amount_cap is None:
        return None
    if pct_cap is None:
        return float(amount_cap)
    if amount_cap is None:
        return pct_cap
    return min(pct_cap, float(amount_cap))


def _round_quantity_down(quantity: float) -> float:
    """Round a positive quantity down to eight decimal places."""

    return math.floor(quantity * 100_000_000) / 100_000_000


def size_position_qty(
    base_qty: float,
    price: float,
    cap: float | None,
) -> tuple[float, str | None]:
    """Size a position without letting a notional cap increase the legacy base quantity.

    Args:
        base_qty: Legacy position quantity, currently one unit.
        price: Market price of one unit.
        cap: Maximum permitted position notional, or ``None`` for no cap.

    Returns:
        The allowed quantity and an optional skip reason.
    """

    base = _finite_number(base_qty)
    if base is None or base <= 0:
        return 0.0, "invalid_base_quantity"

    market_price = _finite_number(price)
    if market_price is None or market_price <= 0:
        return 0.0, "price_unavailable"

    if cap is None:
        quantity = _round_quantity_down(base)
        if quantity <= 0:
            return 0.0, "position_notional_below_minimum"
        return quantity, None

    notional_cap = _finite_number(cap)
    if notional_cap is None or notional_cap <= 0:
        return 0.0, "position_notional_below_minimum"

    raw_quantity = min(base, notional_cap / market_price)
    quantity = _round_quantity_down(raw_quantity)
    if quantity <= 0:
        return 0.0, "position_notional_below_minimum"

    # Guard against a floating-point boundary making the realised notional marginally exceed the
    # configured cap. Moving down by one floating-point step preserves the eight-decimal floor.
    while quantity * market_price > notional_cap:
        reduced = math.nextafter(quantity, 0.0)
        if reduced >= quantity:
            break
        quantity = reduced
    if quantity <= 0 or quantity * market_price > notional_cap:
        return 0.0, "position_notional_below_minimum"
    return quantity, None


def _desired_candidates(desired: Mapping[str, Any] | Iterable[Any]) -> list[tuple[str, str]]:
    """Normalize the dashboard's ordered desired-candidate mapping into candidate tuples."""

    if isinstance(desired, Mapping):
        raw_candidates: Iterable[Any] = desired.items()
    else:
        raw_candidates = desired

    candidates: list[tuple[str, str]] = []
    for candidate in raw_candidates:
        if isinstance(candidate, Mapping):
            symbol = candidate.get("symbol")
            side = candidate.get("side")
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)) and len(candidate) >= 2:
            symbol, side = candidate[0], candidate[1]
        else:
            continue
        candidates.append((str(symbol), str(side).lower()))
    return candidates


def select_open_candidates(
    desired: Mapping[str, Any] | Iterable[Any],
    open_positions: Sequence[Mapping[str, Any]],
    prices: Mapping[str, float],
    config: PaperConfig,
    *,
    paper_state: Mapping[str, Any] | None = None,
    base_qty: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Select and size new paper positions while enforcing the hard position-count limit.

    Candidates are processed in their supplied order. A same-side duplicate consumes no slot, and
    a missing price or a rejected size consumes no slot. When a percentage limit is configured,
    the cap is recalculated from the free balance before each candidate, so opening one position
    reduces the allowance for the next one.

    Returns:
        A pair containing selected candidates and skipped candidates with skip reasons.
    """

    positions = list(open_positions)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    occupied_sides = {
        (str(position.get("symbol")), str(position.get("side", "")).lower())
        for position in positions
        if isinstance(position, Mapping)
    }

    realized_pnl = 0.0
    if isinstance(paper_state, Mapping):
        realized_pnl = _finite_number(paper_state.get("cumulative_closed_pnl", 0.0)) or 0.0

    state_positions = list(positions)
    for symbol, side in _desired_candidates(desired):
        if (symbol, side) in occupied_sides:
            skipped.append({"symbol": symbol, "side": side, "reason": "already_open_same_side"})
            continue

        price = _finite_number(prices.get(symbol)) if isinstance(prices, Mapping) else None
        if price is None or price <= 0:
            skipped.append({"symbol": symbol, "side": side, "reason": "price_unavailable"})
            continue

        if len(positions) + len(selected) >= config.max_open_positions:
            skipped.append({"symbol": symbol, "side": side, "reason": "position_limit_reached"})
            continue

        sizing_state = {
            "cumulative_closed_pnl": realized_pnl,
            "positions": state_positions,
        }
        current_free_balance = free_balance(config, sizing_state)
        cap = position_notional_cap(config, current_free_balance)
        quantity, sizing_reason = size_position_qty(base_qty, price, cap)
        if sizing_reason is not None:
            skipped.append({"symbol": symbol, "side": side, "reason": sizing_reason})
            continue

        notional = quantity * price
        selected.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": quantity,
                "notional": notional,
                "sizing_cap": cap,
            }
        )
        occupied_sides.add((symbol, side))
        state_positions.append({"symbol": symbol, "side": side, "entry": price, "qty": quantity})

    return selected, skipped
