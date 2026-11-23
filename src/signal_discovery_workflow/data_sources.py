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
Pluggable multi-asset data layer for the signal discovery workflow.

This module broadens the data layer beyond the original S&P 500 yfinance
downloader while producing the SAME on-disk layout the evaluator already
consumes via ``signal_evaluator.load_stock_data()``:

    <output_dir>/
        Open.csv      # index = dates (first column, header "Date"), columns = tickers
        Close.csv
        High.csv
        Low.csv
        Volume.csv
        manifest.json (optional, written by write_dataset_manifest)

Because the layout is unchanged, any dataset produced here (crypto, forex,
sectors, ...) can be dropped into the evaluator's data directory and used
with zero changes to the evaluator.

Offline testing / injection
---------------------------
Every network-touching function accepts an injectable ``fetcher`` callable:

- ``fetch_ohlcv``        fetcher contract: ``(symbols, start, end, interval) -> dict[str, pd.DataFrame]``
                          (a yfinance-style wide MultiIndex DataFrame is also accepted)
- ``fetch_fundamentals`` fetcher contract: ``(symbols) -> dict[str, pd.DataFrame]``
- ``fetch_sentiment``    fetcher contract: ``(symbols, window_days) -> dict[str, pd.DataFrame]``

Tests and offline runs inject plain Python callables that return canned
DataFrames; the default implementations use yfinance (OHLCV, fundamentals)
and a no-op sentiment stub that performs NO network calls and requires NO
API keys (it degrades to empty frames with a warning until a real provider
fetcher is injected).

Network requirements: every preset in ``DATASETS`` declares
``requires_network`` for its *default* fetcher. Nothing in this module
contacts the network unless a default fetcher is actually invoked.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from signal_discovery_workflow.download_data import SP500_TICKERS

logger = logging.getLogger(__name__)

# Canonical column names the evaluator consumes.
OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
FUNDAMENTAL_FIELDS = [
    "marketCap",
    "trailingPE",
    "priceToBook",
    "profitMargins",
    "returnOnEquity",
    "totalRevenue",
    "enterpriseValue",
]
SENTIMENT_FIELDS = ["news_sentiment_score", "article_count"]

# Case-insensitive aliases accepted for OHLCV columns (e.g. lowercase frames
# from Binance-style fetchers are renamed to the canonical names).
_FIELD_ALIASES = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
_KNOWN_FIELD_NAMES = set(OHLCV_FIELDS) | set(FUNDAMENTAL_FIELDS) | set(SENTIMENT_FIELDS)

NASDAQ100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'COST',
    'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'LIN', 'QCOM', 'TXN', 'INTU',
    'AMGN', 'CMCSA', 'AMAT', 'HON', 'BKNG', 'ISRG', 'ADSK', 'GILD', 'PANW', 'ADP',
    'VRTX', 'MU', 'LRCX', 'ADI', 'REGN', 'SBUX', 'MELI', 'CTAS', 'MDLZ', 'KLAC',
    'CDNS', 'CSX', 'SNPS', 'MRVL', 'ORLY', 'ASML', 'CRWD', 'ABNB', 'MAR', 'WDAY',
    'PYPL', 'MNST', 'FTNT', 'ROP', 'PCAR', 'NXPI', 'DASH', 'ROST', 'DDOG', 'ODFL',
    'CPRT', 'KDP', 'AEP', 'EA', 'EXC', 'PAYX', 'FAST', 'TEAM', 'CCEP', 'XEL',
    'VRSK', 'PDD', 'TTD', 'KHC', 'CSGP', 'CHTR', 'CTSH', 'IDXX', 'DXCM', 'LULU',
    'GFS', 'BKR', 'BIIB', 'ON', 'SMCI', 'ANSS', 'MCHP', 'CDW', 'WBD', 'MDB',
    'AXON', 'ZS', 'ARM', 'GEHC', 'MSTR', 'MRNA', 'INTC', 'EBAY', 'TLN',
]

SECTOR_ETFS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

FOREX_MAJOR_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]

CRYPTO_SPOT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD"]

CRYPTO_FUTURES_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
]

FUNDAMENTAL_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "XOM"]

SENTIMENT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

DATASETS: dict[str, dict[str, Any]] = {
    "sp500": {
        "name": "sp500",
        "asset_class": "equity",
        "description": "S&P 500 equity universe (baseline dataset; reuses download_data.py tickers).",
        "fields": list(OHLCV_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(SP500_TICKERS),
    },
    "nasdaq100": {
        "name": "nasdaq100",
        "asset_class": "equity",
        "description": "Nasdaq-100 large-cap equity universe (tech-heavy).",
        "fields": list(OHLCV_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(dict.fromkeys(NASDAQ100_TICKERS)),
    },
    "sectors": {
        "name": "sectors",
        "asset_class": "etf",
        "description": "US sector SPDR ETFs (one per GICS sector).",
        "fields": list(OHLCV_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(dict.fromkeys(SECTOR_ETFS)),
    },
    "forex_major": {
        "name": "forex_major",
        "asset_class": "forex",
        "description": "Major USD currency pairs from Yahoo Finance (e.g. EURUSD=X).",
        "fields": list(OHLCV_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(FOREX_MAJOR_SYMBOLS),
    },
    "crypto_spot": {
        "name": "crypto_spot",
        "asset_class": "crypto",
        "description": "Spot crypto pairs quoted in USD from Yahoo Finance (e.g. BTC-USD).",
        "fields": list(OHLCV_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(CRYPTO_SPOT_SYMBOLS),
    },
    "crypto_futures": {
        "name": "crypto_futures",
        "asset_class": "crypto_perp",
        "description": (
            "Binance USDT-M perpetual futures (e.g. BTCUSDT). Downloaded with the existing "
            "download_binance_futures.py script, which covers this dataset (network required)."
        ),
        "fields": list(OHLCV_FIELDS),
        "source": "binance",
        "requires_network": True,
        "default_symbols": list(CRYPTO_FUTURES_SYMBOLS),
    },
    "fundamentals": {
        "name": "fundamentals",
        "asset_class": "fundamentals",
        "description": "Company fundamentals snapshot from Yahoo Finance (non-OHLCV, point-in-time).",
        "fields": list(FUNDAMENTAL_FIELDS),
        "source": "yfinance",
        "requires_network": True,
        "default_symbols": list(FUNDAMENTAL_SYMBOLS),
    },
    "sentiment": {
        "name": "sentiment",
        "asset_class": "sentiment",
        "description": (
            "News sentiment per symbol from a pluggable provider. The default fetcher performs no "
            "network calls, needs no API key, and returns empty frames; inject a custom fetcher "
            "(news RSS, GDELT, licensed provider) to enable it."
        ),
        "fields": list(SENTIMENT_FIELDS),
        "source": "news",
        "requires_network": True,
        "default_symbols": list(SENTIMENT_SYMBOLS),
    },
}

_DATASETS_BY_KEY = {name.strip().lower(): meta for name, meta in DATASETS.items()}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

def get_dataset(name: str) -> dict[str, Any]:
    """Return the dataset preset registered under ``name``.

    Lookup is case-insensitive. Raises ``KeyError`` with the list of
    available dataset names for unknown names.
    """
    key = str(name).strip().lower()
    if key in _DATASETS_BY_KEY:
        return _DATASETS_BY_KEY[key]
    available = ", ".join(sorted(DATASETS))
    raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}")


def list_datasets() -> list[str]:
    """Return the names of all registered dataset presets."""
    return list(DATASETS)


def normalize_symbol(symbol: str, dataset: str) -> str:
    """Normalize ``symbol`` into the canonical form used by ``dataset``.

    Examples:
        normalize_symbol("EUR/USD", "forex_major")  -> "EURUSD=X"
        normalize_symbol("BTC", "crypto_spot")      -> "BTC-USD"
        normalize_symbol("BTC-USDT", "crypto_futures") -> "BTCUSDT"
        normalize_symbol(" aapl ", "sp500")         -> "AAPL"

    Raises ``KeyError`` for unknown datasets and ``ValueError`` for symbols
    that cannot be mapped to the dataset's canonical form.
    """
    meta = get_dataset(dataset)
    name = meta["name"]
    asset_class = meta["asset_class"]
    if name == "forex_major" or asset_class == "forex":
        return _normalize_forex_symbol(symbol)
    if name == "crypto_futures" or asset_class == "crypto_perp":
        return _normalize_perp_symbol(symbol)
    if name == "crypto_spot" or asset_class == "crypto":
        return _normalize_crypto_spot_symbol(symbol)
    return symbol.strip().upper()


def _normalize_forex_symbol(symbol: str) -> str:
    """Map forex forms like ``EUR/USD``, ``eurusd``, ``EURUSD=X`` to ``EURUSD=X``."""
    cleaned = (
        symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    )
    if cleaned.endswith("=X"):
        cleaned = cleaned[:-2]
    if len(cleaned) == 6 and cleaned.isalpha():
        return f"{cleaned}=X"
    raise ValueError(
        f"Cannot normalize {symbol!r} as a forex pair; expected e.g. 'EUR/USD' or 'EURUSD'."
    )


def _normalize_crypto_spot_symbol(symbol: str) -> str:
    """Map crypto forms like ``BTC``, ``btc-usd``, ``ETH/USD`` to ``BTC-USD`` style."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if cleaned.endswith("USD") and len(cleaned) > 3:
        cleaned = cleaned[:-3]
    if not cleaned:
        raise ValueError(f"Cannot normalize {symbol!r} as a crypto spot symbol.")
    return f"{cleaned}-USD"


def _normalize_perp_symbol(symbol: str) -> str:
    """Map perpetual futures forms like ``BTC-USDT``/``btcusdt`` to ``BTCUSDT``."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if not cleaned:
        raise ValueError(f"Cannot normalize {symbol!r} as a perpetual futures symbol.")
    return cleaned


# ---------------------------------------------------------------------------
# Shared frame normalization helpers
# ---------------------------------------------------------------------------

def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame index to a tz-naive DatetimeIndex when it is date-like.

    Non-date-like indexes (e.g. integer snapshot indexes) are left untouched.
    """
    if len(df.index) == 0 or pd.api.types.is_numeric_dtype(df.index):
        return df
    try:
        parsed = pd.to_datetime(df.index, errors="coerce")
    except (TypeError, ValueError):
        return df
    if not isinstance(parsed, pd.DatetimeIndex) or parsed.isna().all():
        return df
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    df.index = parsed
    return df


def _deduplicate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by index and drop duplicate date rows (keeping the last occurrence)."""
    df = df.sort_index()
    duplicated = df.index.duplicated(keep="last")
    if duplicated.any():
        logger.info("Dropped %d duplicate date row(s).", int(duplicated.sum()))
        df = df[~duplicated]
    return df


def _empty_ohlcv_frame() -> pd.DataFrame:
    """Return an empty frame carrying the canonical OHLCV columns."""
    return pd.DataFrame(columns=OHLCV_FIELDS)


def _normalize_ohlcv_frame(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """Normalize one symbol's raw OHLCV frame to canonical columns/index.

    - renames lowercase/aliased columns to Open/High/Low/Close/Volume
    - drops non-OHLCV columns
    - coerces the index to a sorted, tz-naive DatetimeIndex and drops duplicate dates
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_ohlcv_frame()
    df = frame.copy()
    df.columns = [str(col) for col in df.columns]
    rename = {col: _FIELD_ALIASES[col.lower().replace(" ", "")] for col in df.columns
              if col.lower().replace(" ", "") in _FIELD_ALIASES}
    df = df.rename(columns=rename)
    keep = [col for col in OHLCV_FIELDS if col in df.columns]
    if not keep:
        logger.warning("%s: no recognizable OHLCV columns found; returning frame unchanged.", symbol)
        return df
    if len(keep) < len(OHLCV_FIELDS):
        missing = [col for col in OHLCV_FIELDS if col not in keep]
        logger.warning("%s: fetched data missing column(s) %s.", symbol, missing)
    df = df[keep]
    df = _normalize_date_index(df)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.isna().any():
        df = df[df.index.notna()]
    return _deduplicate_and_sort(df)


def _coerce_symbol_frames(
    raw: Any,
    symbols: list[str],
    normalizer: Callable[[pd.DataFrame | None, str], pd.DataFrame],
    context: str,
) -> dict[str, pd.DataFrame]:
    """Coerce a fetcher's raw output into ``{symbol: normalized DataFrame}``.

    Accepts:
      - dict[str, pd.DataFrame] keyed by symbol (keys matched case-insensitively)
      - a yfinance-style wide DataFrame with 2-level MultiIndex columns
        (field-major as produced by ``yf.download(..., group_by="column")``,
        or symbol-major as produced by ``group_by="ticker"``)
      - a single flat DataFrame when exactly one symbol was requested
    """
    frames: dict[str, pd.DataFrame] = {}

    if isinstance(raw, dict):
        lookup = {str(key).strip().lower(): value for key, value in raw.items()}
        for symbol in symbols:
            frame = lookup.get(symbol.strip().lower())
            if frame is None:
                logger.warning("%s: no data returned for symbol %r.", context, symbol)
                frames[symbol] = normalizer(None, symbol)
            else:
                frames[symbol] = normalizer(frame, symbol)
        return frames

    if isinstance(raw, pd.DataFrame):
        if isinstance(raw.columns, pd.MultiIndex) and raw.columns.nlevels == 2:
            level0 = {str(col) for col in raw.columns.get_level_values(0)}
            level1 = {str(col) for col in raw.columns.get_level_values(1)}
            if level0 & set(OHLCV_FIELDS):
                # field-major: level 0 = fields, level 1 = symbols
                for symbol in symbols:
                    if symbol in level1:
                        frames[symbol] = normalizer(raw.xs(symbol, axis=1, level=1), symbol)
                    else:
                        logger.warning("%s: no data returned for symbol %r.", context, symbol)
                        frames[symbol] = normalizer(None, symbol)
                return frames
            # symbol-major: level 0 = symbols, level 1 = fields
            for symbol in symbols:
                if symbol in level0:
                    frames[symbol] = normalizer(raw[symbol], symbol)
                else:
                    logger.warning("%s: no data returned for symbol %r.", context, symbol)
                    frames[symbol] = normalizer(None, symbol)
            return frames
        if len(symbols) == 1:
            frames[symbols[0]] = normalizer(raw, symbols[0])
            return frames
        logger.warning(
            "%s: fetcher returned a single flat frame for %d symbols; cannot split it. "
            "Return a dict keyed by symbol instead.",
            context,
            len(symbols),
        )
        return {symbol: normalizer(None, symbol) for symbol in symbols}

    raise TypeError(
        f"{context}: unsupported fetcher return type {type(raw).__name__}; "
        "expected dict[str, pd.DataFrame] or pd.DataFrame."
    )


# ---------------------------------------------------------------------------
# Default fetchers (network-bound; inject fakes in tests)
# ---------------------------------------------------------------------------

def _yfinance_ohlcv_fetcher(symbols: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Default OHLCV fetcher backed by ``yfinance.download``."""
    import yfinance as yf  # imported lazily so importing this module stays cheap/offline-safe

    logger.info("Downloading OHLCV for %d symbol(s) via yfinance (%s to %s)...", len(symbols), start, end)
    return yf.download(
        symbols,
        start=start,
        end=end,
        interval=interval,
        group_by="column",
        auto_adjust=True,
        progress=False,
    )


def _yfinance_fundamentals_fetcher(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Default fundamentals fetcher backed by ``yfinance.Ticker(...).info``."""
    import yfinance as yf  # imported lazily so importing this module stays cheap/offline-safe

    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:  # noqa: BLE001 - yfinance raises a wide variety of errors
            logger.warning("Fundamentals fetch failed for %s: %s", symbol, exc)
            results[symbol] = pd.DataFrame(columns=FUNDAMENTAL_FIELDS)
            continue
        row = {field: info.get(field, float("nan")) for field in FUNDAMENTAL_FIELDS}
        results[symbol] = pd.DataFrame([row], columns=FUNDAMENTAL_FIELDS)
    return results


def _default_sentiment_fetcher(symbols: list[str], window_days: int = 7) -> dict[str, pd.DataFrame]:
    """Placeholder sentiment fetcher: no network calls, no API keys, no fake data.

    It deliberately returns empty frames and warns. To enable sentiment,
    inject a custom fetcher into :func:`fetch_sentiment` that implements
    ``(symbols, window_days) -> dict[str, pd.DataFrame]`` with columns
    ``news_sentiment_score`` and ``article_count``.
    """
    logger.warning(
        "No sentiment provider configured: returning empty frames for %d symbol(s) "
        "(window_days=%d). Inject a custom fetcher into fetch_sentiment(...) to enable "
        "sentiment data; the default path requires no API key and makes no network calls.",
        len(symbols),
        window_days,
    )
    return {symbol: pd.DataFrame(columns=SENTIMENT_FIELDS) for symbol in symbols}


# ---------------------------------------------------------------------------
# Public fetch API
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    fetcher: Callable[..., Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV history for ``symbols`` and return ``{symbol: DataFrame}``.

    Each returned DataFrame has (a subset of) the canonical columns
    Open/High/Low/Close/Volume, a sorted tz-naive DatetimeIndex, and
    duplicate date rows dropped. Symbols without data map to an empty frame
    with the canonical columns.

    Args:
        symbols: Symbols to fetch (normalized via :func:`normalize_symbol` by callers).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: Bar interval (default "1d").
        fetcher: Injectable fetcher ``(symbols, start, end, interval) -> dict | DataFrame``.
            Defaults to a yfinance-backed fetcher (requires network).
    """
    cleaned = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        logger.warning("fetch_ohlcv: empty symbol list; nothing to fetch.")
        return {}
    fetch = fetcher if fetcher is not None else _yfinance_ohlcv_fetcher
    raw = fetch(cleaned, start, end, interval)
    return _coerce_symbol_frames(raw, cleaned, _normalize_ohlcv_frame, "fetch_ohlcv")


def _normalize_fundamentals_frame(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """Normalize one symbol's fundamentals to the canonical column set."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_FIELDS)
    df = frame.copy()
    for field in FUNDAMENTAL_FIELDS:
        if field not in df.columns:
            logger.warning("%s: fundamentals data missing column %r; filling with NaN.", symbol, field)
            df[field] = float("nan")
    return df[FUNDAMENTAL_FIELDS]


def fetch_fundamentals(
    symbols: list[str],
    fetcher: Callable[..., Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch point-in-time fundamentals for ``symbols``.

    Returns ``{symbol: DataFrame}`` where each frame carries exactly the
    columns in ``FUNDAMENTAL_FIELDS`` (missing values are NaN). Symbols
    without data map to an empty frame with the canonical columns.

    Args:
        symbols: Tickers to fetch.
        fetcher: Injectable fetcher ``(symbols) -> dict[str, pd.DataFrame]``.
            Defaults to a yfinance ``Ticker.info`` fetcher (requires network).
    """
    cleaned = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        logger.warning("fetch_fundamentals: empty symbol list; nothing to fetch.")
        return {}
    fetch = fetcher if fetcher is not None else _yfinance_fundamentals_fetcher
    raw = fetch(cleaned)
    return _coerce_symbol_frames(raw, cleaned, _normalize_fundamentals_frame, "fetch_fundamentals")


def _normalize_sentiment_frame(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """Normalize one symbol's sentiment frame to the canonical columns."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=SENTIMENT_FIELDS)
    df = frame.copy()
    for field in SENTIMENT_FIELDS:
        if field not in df.columns:
            logger.warning("%s: sentiment data missing column %r; filling with NaN.", symbol, field)
            df[field] = float("nan")
    df = df[SENTIMENT_FIELDS]
    df = _normalize_date_index(df)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.isna().any():
        df = df[df.index.notna()]
    return _deduplicate_and_sort(df)


def fetch_sentiment(
    symbols: list[str],
    fetcher: Callable[..., Any] | None = None,
    window_days: int = 7,
) -> dict[str, pd.DataFrame]:
    """Fetch news sentiment for ``symbols`` over a ``window_days`` lookback window.

    Returns ``{symbol: DataFrame}`` with columns ``news_sentiment_score`` and
    ``article_count`` (date-indexed when the provider supplies dates). The
    default path requires NO API key and performs NO network calls: it returns
    empty frames with a warning until a provider fetcher is injected. It never
    fabricates sentiment values.

    Args:
        symbols: Tickers to fetch.
        fetcher: Injectable provider ``(symbols, window_days) -> dict[str, pd.DataFrame]``.
        window_days: Lookback window in days passed through to the provider.
    """
    cleaned = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        logger.warning("fetch_sentiment: empty symbol list; nothing to fetch.")
        return {}
    fetch = fetcher if fetcher is not None else _default_sentiment_fetcher
    raw = fetch(cleaned, window_days)
    return _coerce_symbol_frames(raw, cleaned, _normalize_sentiment_frame, "fetch_sentiment")


# ---------------------------------------------------------------------------
# On-disk layout (evaluator-compatible)
# ---------------------------------------------------------------------------

def pivot_ohlcv_to_fields(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert ``{symbol: frame-with-field-columns}`` to ``{field: dates-x-symbols}``.

    This is the bridge between the fetch API (one frame per symbol) and the
    on-disk evaluator layout (one CSV per field). Indexes are aligned with an
    outer join per field.
    """
    if not data:
        return {}
    seen: list[str] = []
    for frame in data.values():
        if isinstance(frame, pd.DataFrame):
            for col in frame.columns:
                col = str(col)
                if col not in seen:
                    seen.append(col)
    canonical = [c for group in (OHLCV_FIELDS, FUNDAMENTAL_FIELDS, SENTIMENT_FIELDS) for c in group]
    ordered = [c for c in canonical if c in seen] + [c for c in seen if c not in canonical]

    fields: dict[str, pd.DataFrame] = {}
    for col in ordered:
        series_map = {str(symbol): frame[col].rename(str(symbol)) for symbol, frame in data.items()
                      if isinstance(frame, pd.DataFrame) and col in frame.columns}
        fields[col] = pd.concat(series_map, axis=1) if series_map else pd.DataFrame()
    return fields


def _coerce_field_frames(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return ``data`` as a field-keyed mapping, pivoting symbol-keyed input.

    Field-keyed input (keys like "Open"/"marketCap") is passed through.
    Symbol-keyed input (frames whose columns are fields) is pivoted via
    :func:`pivot_ohlcv_to_fields`. Anything else raises ``ValueError``.
    """
    if not data:
        return {}
    keys = [str(key) for key in data]
    if all(key in _KNOWN_FIELD_NAMES for key in keys):
        return {str(key): value for key, value in data.items()}
    values = list(data.values())
    if all(isinstance(value, pd.DataFrame) for value in values) and all(
        set(str(col) for col in value.columns) & _KNOWN_FIELD_NAMES for value in values
    ):
        logger.info("Detected symbol-keyed data; pivoting to field-keyed layout.")
        return pivot_ohlcv_to_fields(data)
    raise ValueError(
        "write_dataset_to_csv expects either {field: dates-x-tickers DataFrame} "
        "(e.g. keys 'Open', 'Close', ...) or {symbol: frame-with-field-columns} "
        "(e.g. the output of fetch_ohlcv/fetch_fundamentals/fetch_sentiment)."
    )


def write_dataset_to_csv(
    data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    fields: list[str] | None = None,
) -> Path:
    """Write field-keyed data in the evaluator-compatible CSV layout.

    Produces ``<output_dir>/<Field>.csv`` files where each file has dates as
    the index (written as the first column with header "Date") and tickers as
    columns — exactly what ``signal_evaluator.load_stock_data()`` reads with
    ``pd.read_csv(path, index_col=0, parse_dates=True)``.

    Args:
        data: Either ``{field: dates-x-tickers DataFrame}`` or symbol-keyed
            frames (auto-pivoted, e.g. the output of ``fetch_ohlcv``).
        output_dir: Directory to write into (created if missing).
        fields: Optional subset of fields to write; all available by default.

    Returns:
        The output directory as a ``Path``.

    Raises:
        KeyError: If an explicitly requested field is not present in ``data``.
        ValueError: If ``data`` matches neither accepted shape.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = _coerce_field_frames(data)
    requested = list(fields) if fields is not None else list(frames.keys())
    for field in requested:
        if field not in frames:
            raise KeyError(f"Field {field!r} not present in data. Available: {sorted(frames)}")
        frame = frames[field]
        if frame.empty:
            logger.warning("Skipping %s.csv: no rows of data.", field)
            continue
        out = frame.copy()
        if out.index.name is None:
            out.index.name = "Date"
        csv_path = output / f"{field}.csv"
        out.to_csv(csv_path)
        logger.info("Saved %s with shape %s", csv_path.name, out.shape)
    return output


def write_dataset_manifest(
    output_dir: str | Path,
    dataset_name: str,
    symbols: list[str],
    source: str,
    fetched_at: str | None = None,
) -> Path:
    """Write a ``manifest.json`` describing a downloaded dataset.

    Records the dataset name, symbol list, source, a UTC timestamp (or the
    provided ``fetched_at``), and row/column counts for every ``<Field>.csv``
    found in ``output_dir``. Used later by the dashboard.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for csv_path in sorted(output.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path, index_col=0)
        except Exception as exc:  # noqa: BLE001 - manifest must survive unreadable files
            logger.warning("Manifest: could not read %s (%s); skipping.", csv_path.name, exc)
            continue
        counts[csv_path.stem] = {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
    manifest = {
        "dataset": dataset_name,
        "symbols": [str(symbol) for symbol in symbols],
        "source": source,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "row_column_counts": counts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Signal-input helpers
# ---------------------------------------------------------------------------

def combine_fundamentals_with_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Broadcast fundamentals onto each symbol's OHLCV frame as signal inputs.

    Produces per-symbol DataFrames with the OHLCV columns first, followed by
    every column in ``FUNDAMENTAL_FIELDS``:

    - If a symbol's fundamentals frame is date-indexed, values are aligned to
      the OHLCV dates and forward-filled (dates before the first fundamental
      observation stay NaN).
    - Otherwise (single-snapshot frames, the default), the first non-null
      value of each fundamental column is broadcast across all dates.
    - Symbols without fundamentals get all-NaN fundamental columns, so the
      per-symbol schema stays consistent for cross-sectional evaluation.
    """
    combined: dict[str, pd.DataFrame] = {}
    for symbol, frame in ohlcv.items():
        out = frame.copy()
        for field in FUNDAMENTAL_FIELDS:
            if field not in out.columns:
                out[field] = float("nan")
        raw = fundamentals.get(symbol)
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            fundamentals_df = _normalize_date_index(raw.copy()).sort_index()
            if isinstance(fundamentals_df.index, pd.DatetimeIndex):
                for field in FUNDAMENTAL_FIELDS:
                    if field in fundamentals_df.columns:
                        out[field] = fundamentals_df[field].reindex(out.index, method="ffill")
            else:
                for field in FUNDAMENTAL_FIELDS:
                    if field in fundamentals_df.columns:
                        valid = fundamentals_df[field].dropna()
                        out[field] = valid.iloc[0] if len(valid) else float("nan")
        else:
            logger.info("combine_fundamentals_with_ohlcv: no fundamentals for %r; using NaN.", symbol)
        combined[symbol] = out
    return combined
