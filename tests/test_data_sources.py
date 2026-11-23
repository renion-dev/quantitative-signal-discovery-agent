# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the multi-asset data layer (data_sources + download_market_data CLI).

All fetchers are fake Python callables — no network access anywhere.
"""

import json
import logging

import pandas as pd
import pytest

from signal_discovery_workflow import data_sources, download_market_data

REQUIRED_DATASETS = [
    "sp500",
    "nasdaq100",
    "sectors",
    "forex_major",
    "crypto_spot",
    "crypto_futures",
    "fundamentals",
    "sentiment",
]
REGISTRY_KEYS = {
    "name",
    "asset_class",
    "description",
    "fields",
    "source",
    "requires_network",
    "default_symbols",
}


# ---------------------------------------------------------------------------
# Helpers: fake fetchers (plain callables, no network)
# ---------------------------------------------------------------------------

def make_ohlcv_fetcher(rows: int = 4, lowercase: bool = True):
    """Build a fake OHLCV fetcher returning deterministic frames per symbol."""

    def fetcher(symbols, start, end, interval):
        index = pd.date_range(start, periods=rows, freq="D")
        frames = {}
        for i, symbol in enumerate(symbols):
            base = 100.0 + 10 * i
            columns = {
                ("open" if lowercase else "Open"): [base + j for j in range(rows)],
                ("high" if lowercase else "High"): [base + j + 1 for j in range(rows)],
                ("low" if lowercase else "Low"): [base + j - 1 for j in range(rows)],
                ("close" if lowercase else "Close"): [base + j + 0.5 for j in range(rows)],
                ("volume" if lowercase else "Volume"): [1000.0 * (j + 1) for j in range(rows)],
            }
            frames[symbol] = pd.DataFrame(columns, index=index)
        return frames

    return fetcher


def make_fundamentals_fetcher():
    """Build a fake fundamentals fetcher returning single-snapshot frames."""

    def fetcher(symbols):
        frames = {}
        for i, symbol in enumerate(symbols):
            row = {
                "marketCap": 1e12 + i,
                "trailingPE": 25.0 + i,
                "priceToBook": 5.0 + i,
                "profitMargins": 0.2 + i / 100,
                "returnOnEquity": 0.3 + i / 100,
                "totalRevenue": 1e11 + i,
                "enterpriseValue": 1.1e12 + i,
            }
            frames[symbol] = pd.DataFrame([row], columns=data_sources.FUNDAMENTAL_FIELDS)
        return frames

    return fetcher


def make_sentiment_fetcher():
    """Build a fake sentiment provider returning date-indexed frames."""

    def fetcher(symbols, window_days):
        index = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
        frames = {}
        for i, symbol in enumerate(symbols):
            frames[symbol] = pd.DataFrame(
                {
                    "news_sentiment_score": [0.1 * (j + 1) + i for j in range(3)],
                    "article_count": [10 + j + i for j in range(3)],
                },
                index=index,
            )
        return frames

    return fetcher


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_required_presets_with_required_keys():
    for name in REQUIRED_DATASETS:
        assert name in data_sources.DATASETS, f"missing preset {name!r}"
        meta = data_sources.DATASETS[name]
        assert REGISTRY_KEYS <= set(meta), f"{name}: missing keys {REGISTRY_KEYS - set(meta)}"
        assert isinstance(meta["fields"], list) and meta["fields"]
        assert isinstance(meta["default_symbols"], list) and meta["default_symbols"]
        assert len(meta["default_symbols"]) == len(set(meta["default_symbols"]))
        assert isinstance(meta["requires_network"], bool)
        assert meta["name"] == name


def test_registry_sources_and_field_kinds():
    assert data_sources.DATASETS["sp500"]["source"] == "yfinance"
    assert data_sources.DATASETS["nasdaq100"]["source"] == "yfinance"
    assert data_sources.DATASETS["sectors"]["source"] == "yfinance"
    assert data_sources.DATASETS["forex_major"]["source"] == "yfinance"
    assert data_sources.DATASETS["crypto_spot"]["source"] == "yfinance"
    assert data_sources.DATASETS["crypto_futures"]["source"] == "binance"
    assert data_sources.DATASETS["crypto_futures"]["requires_network"] is True
    assert data_sources.DATASETS["fundamentals"]["fields"] == data_sources.FUNDAMENTAL_FIELDS
    assert data_sources.DATASETS["sentiment"]["fields"] == data_sources.SENTIMENT_FIELDS
    assert data_sources.OHLCV_FIELDS == ["Open", "High", "Low", "Close", "Volume"]


def test_sectors_preset_is_deduplicated():
    symbols = data_sources.DATASETS["sectors"]["default_symbols"]
    assert symbols.count("XLE") == 1
    assert len(symbols) == 11
    assert set(symbols) == {"XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"}


def test_get_dataset_unknown_raises_keyerror_listing_available():
    with pytest.raises(KeyError, match="sp500"):
        data_sources.get_dataset("does_not_exist")


def test_get_dataset_is_case_insensitive():
    assert data_sources.get_dataset("Crypto_Spot")["name"] == "crypto_spot"


def test_list_datasets_nonempty_and_includes_sp500():
    names = data_sources.list_datasets()
    assert names
    assert "sp500" in names
    assert set(REQUIRED_DATASETS) <= set(names)


# ---------------------------------------------------------------------------
# normalize_symbol
# ---------------------------------------------------------------------------

def test_normalize_symbol_forex():
    cases = {
        "EUR/USD": "EURUSD=X",
        "eurusd": "EURUSD=X",
        "GBPUSD=X": "GBPUSD=X",
        "usd/jpy": "USDJPY=X",
    }
    for raw, expected in cases.items():
        assert data_sources.normalize_symbol(raw, "forex_major") == expected


def test_normalize_symbol_forex_invalid_raises():
    with pytest.raises(ValueError, match="forex"):
        data_sources.normalize_symbol("not_a_pair", "forex_major")


def test_normalize_symbol_crypto_spot():
    cases = {
        "BTC": "BTC-USD",
        "btc-usd": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "solusd": "SOL-USD",
        "DOGE-USD": "DOGE-USD",
    }
    for raw, expected in cases.items():
        assert data_sources.normalize_symbol(raw, "crypto_spot") == expected


def test_normalize_symbol_crypto_futures_perp():
    cases = {"BTCUSDT": "BTCUSDT", "btcusdt": "BTCUSDT", "BTC-USDT": "BTCUSDT", "eth/usdt": "ETHUSDT"}
    for raw, expected in cases.items():
        assert data_sources.normalize_symbol(raw, "crypto_futures") == expected


def test_normalize_symbol_equity_and_unknown_dataset():
    assert data_sources.normalize_symbol(" aapl ", "sp500") == "AAPL"
    assert data_sources.normalize_symbol("nvda", "nasdaq100") == "NVDA"
    with pytest.raises(KeyError):
        data_sources.normalize_symbol("AAPL", "unknown_dataset")


# ---------------------------------------------------------------------------
# fetch_ohlcv with fake fetchers
# ---------------------------------------------------------------------------

def test_fetch_ohlcv_fake_fetcher_columns_tz_naive_sorted_dedup(caplog):
    def fake_fetcher(symbols, start, end, interval):
        index = pd.DatetimeIndex(["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-02"]).tz_localize("UTC")
        frames = {}
        for symbol in symbols:
            frames[symbol] = pd.DataFrame(
                {"open": [10.0, 20.0, 30.0, 25.0], "close": [11.0, 21.0, 31.0, 26.0], "volume": [1.0, 2.0, 3.0, 4.0]},
                index=index,
            )
        return frames

    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_ohlcv(["BTC-USD"], "2024-01-01", "2024-01-10", fetcher=fake_fetcher)

    assert set(result) == {"BTC-USD"}
    frame = result["BTC-USD"]
    assert list(frame.columns) == ["Open", "Close", "Volume"]
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert list(frame.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    # duplicate date kept the last occurrence (25.0, not 20.0)
    assert frame.loc["2024-01-02", "Open"] == 25.0


def test_fetch_ohlcv_one_frame_per_symbol_with_fake_fetcher():
    result = data_sources.fetch_ohlcv(
        ["AAPL", "MSFT"], "2024-01-01", "2024-01-10", fetcher=make_ohlcv_fetcher()
    )
    assert set(result) == {"AAPL", "MSFT"}
    for frame in result.values():
        assert list(frame.columns) == data_sources.OHLCV_FIELDS
        assert frame.index.tz is None
        assert frame.index.is_monotonic_increasing
        assert len(frame) == 4


def test_fetch_ohlcv_empty_symbol_list_returns_empty_without_calling_fetcher():
    def exploding_fetcher(symbols, start, end, interval):
        raise AssertionError("fetcher must not be called for an empty symbol list")

    assert data_sources.fetch_ohlcv([], "2024-01-01", "2024-01-10", fetcher=exploding_fetcher) == {}


def test_fetch_ohlcv_missing_symbol_warns_and_returns_canonical_empty_frame(caplog):
    def partial_fetcher(symbols, start, end, interval):
        return make_ohlcv_fetcher()(symbols[:1], start, end, interval)

    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-01-10", fetcher=partial_fetcher)

    assert set(result) == {"AAPL", "MSFT"}
    assert result["AAPL"].shape[0] == 4
    assert result["MSFT"].empty
    assert list(result["MSFT"].columns) == data_sources.OHLCV_FIELDS
    assert "MSFT" in caplog.text


def test_fetch_ohlcv_missing_field_warns_but_returns_available_columns(caplog):
    def no_volume_fetcher(symbols, start, end, interval):
        frames = make_ohlcv_fetcher()(symbols, start, end, interval)
        return {symbol: frame.drop(columns=["volume"]) for symbol, frame in frames.items()}

    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_ohlcv(["AAPL"], "2024-01-01", "2024-01-10", fetcher=no_volume_fetcher)

    frame = result["AAPL"]
    assert list(frame.columns) == ["Open", "High", "Low", "Close"]
    assert "Volume" in caplog.text


def test_fetch_ohlcv_accepts_yfinance_style_wide_multiindex_frame():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["AAPL", "MSFT"]])
    raw = pd.DataFrame([[1.0, 2.0, 10.0, 20.0]] * 3, index=index, columns=columns)

    def fetcher(symbols, start, end, interval):
        return raw

    result = data_sources.fetch_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-01-10", fetcher=fetcher)
    assert set(result) == {"AAPL", "MSFT"}
    assert list(result["AAPL"].columns) == ["Close", "Volume"]
    assert (result["MSFT"]["Volume"] == 20.0).all()


def test_fetch_ohlcv_unsupported_fetcher_output_raises_typeerror():
    with pytest.raises(TypeError):
        data_sources.fetch_ohlcv(["AAPL"], "2024-01-01", "2024-01-10", fetcher=lambda *a: "junk")


# ---------------------------------------------------------------------------
# fetch_fundamentals / combine_fundamentals_with_ohlcv
# ---------------------------------------------------------------------------

def test_fetch_fundamentals_with_fake_fetcher_returns_canonical_columns():
    result = data_sources.fetch_fundamentals(["AAPL", "MSFT"], fetcher=make_fundamentals_fetcher())
    assert set(result) == {"AAPL", "MSFT"}
    for frame in result.values():
        assert list(frame.columns) == data_sources.FUNDAMENTAL_FIELDS
        assert len(frame) == 1


def test_fetch_fundamentals_missing_symbol_returns_empty_canonical_frame(caplog):
    def partial_fetcher(symbols):
        return {"AAPL": make_fundamentals_fetcher()(["AAPL"])["AAPL"]}

    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_fundamentals(["AAPL", "MSFT"], fetcher=partial_fetcher)

    assert result["MSFT"].empty
    assert list(result["MSFT"].columns) == data_sources.FUNDAMENTAL_FIELDS
    assert "MSFT" in caplog.text


def test_fetch_fundamentals_empty_symbols_returns_empty():
    assert data_sources.fetch_fundamentals([], fetcher=make_fundamentals_fetcher()) == {}


def test_combine_fundamentals_broadcasts_snapshot_across_dates():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    ohlcv = {"AAPL": pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0], "Volume": [10.0] * 5}, index=index)}
    fundamentals = {
        "AAPL": pd.DataFrame([{"marketCap": 1e12, "trailingPE": 25.0}], columns=["marketCap", "trailingPE"])
    }

    combined = data_sources.combine_fundamentals_with_ohlcv(ohlcv, fundamentals)
    frame = combined["AAPL"]

    assert list(frame.columns)[:2] == ["Close", "Volume"]
    assert set(data_sources.FUNDAMENTAL_FIELDS) <= set(frame.columns)
    assert (frame["marketCap"] == 1e12).all()
    assert (frame["trailingPE"] == 25.0).all()
    # fundamental columns not provided by the snapshot stay NaN
    assert frame["priceToBook"].isna().all()


def test_combine_fundamentals_forward_fills_date_indexed_fundamentals():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    ohlcv = {"AAPL": pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)}
    fundamentals = {
        "AAPL": pd.DataFrame(
            {"trailingPE": [25.0, 30.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]),
        )
    }

    frame = data_sources.combine_fundamentals_with_ohlcv(ohlcv, fundamentals)["AAPL"]
    assert frame["trailingPE"].tolist() == [25.0, 25.0, 30.0, 30.0, 30.0]


def test_combine_fundamentals_symbols_without_fundamentals_get_nan_columns():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    ohlcv = {"AAPL": pd.DataFrame({"Close": [1.0, 2.0]}, index=index), "MSFT": pd.DataFrame({"Close": [3.0, 4.0]}, index=index)}
    fundamentals = {"AAPL": pd.DataFrame([{"marketCap": 1e12}], columns=["marketCap"])}

    combined = data_sources.combine_fundamentals_with_ohlcv(ohlcv, fundamentals)
    assert (combined["AAPL"]["marketCap"] == 1e12).all()
    assert combined["MSFT"]["marketCap"].isna().all()
    assert combined["MSFT"]["trailingPE"].isna().all()


# ---------------------------------------------------------------------------
# fetch_sentiment
# ---------------------------------------------------------------------------

def test_fetch_sentiment_default_provider_returns_empty_frames_without_raising(caplog):
    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_sentiment(["AAPL", "MSFT"])

    assert set(result) == {"AAPL", "MSFT"}
    for frame in result.values():
        assert frame.empty
        assert list(frame.columns) == data_sources.SENTIMENT_FIELDS
    assert "No sentiment provider configured" in caplog.text


def test_fetch_sentiment_empty_symbols_returns_empty():
    assert data_sources.fetch_sentiment([], fetcher=make_sentiment_fetcher()) == {}


def test_fetch_sentiment_with_fake_provider_returns_expected_columns():
    result = data_sources.fetch_sentiment(["AAPL", "TSLA"], fetcher=make_sentiment_fetcher(), window_days=7)
    assert set(result) == {"AAPL", "TSLA"}
    for frame in result.values():
        assert list(frame.columns) == data_sources.SENTIMENT_FIELDS
        assert frame.index.is_monotonic_increasing
        assert frame.index.tz is None
        assert len(frame) == 3


def test_fetch_sentiment_missing_symbol_and_column_degrade_gracefully(caplog):
    def partial_fetcher(symbols, window_days):
        frame = pd.DataFrame({"news_sentiment_score": [0.5]}, index=pd.DatetimeIndex(["2024-01-01"]))
        return {"AAPL": frame}

    with caplog.at_level(logging.WARNING):
        result = data_sources.fetch_sentiment(["AAPL", "MSFT"], fetcher=partial_fetcher)

    assert list(result["AAPL"].columns) == data_sources.SENTIMENT_FIELDS
    assert result["AAPL"]["article_count"].isna().all()
    assert result["MSFT"].empty
    assert list(result["MSFT"].columns) == data_sources.SENTIMENT_FIELDS
    assert "MSFT" in caplog.text


# ---------------------------------------------------------------------------
# write_dataset_to_csv / manifest / pivot
# ---------------------------------------------------------------------------

def make_field_frames():
    # No `freq` on the index: a CSV round-trip cannot preserve it, and the
    # round-trip test compares frames exactly.
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    index.name = "Date"
    tickers = ["AAPL", "MSFT"]
    return {
        field: pd.DataFrame(
            {ticker: 100.0 + j * 10 + i for j, ticker in enumerate(tickers)}, index=index
        )
        for i, field in enumerate(data_sources.OHLCV_FIELDS)
    }


def test_write_dataset_to_csv_round_trips_through_evaluator_layout(tmp_path):
    frames = make_field_frames()
    output = data_sources.write_dataset_to_csv(frames, tmp_path / "nested" / "sp500")

    assert output == tmp_path / "nested" / "sp500"
    assert output.exists()
    for field in data_sources.OHLCV_FIELDS:
        path = output / f"{field}.csv"
        assert path.exists(), f"expected evaluator-named file {path.name}"
        back = pd.read_csv(path, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(back, frames[field])


def test_write_dataset_to_csv_fields_subset(tmp_path):
    frames = make_field_frames()
    output = data_sources.write_dataset_to_csv(frames, tmp_path, fields=["Close"])

    assert (output / "Close.csv").exists()
    assert not (output / "Open.csv").exists()
    assert set(path.name for path in output.glob("*.csv")) == {"Close.csv"}


def test_write_dataset_to_csv_missing_field_raises_keyerror(tmp_path):
    frames = make_field_frames()
    with pytest.raises(KeyError, match="NotAField"):
        data_sources.write_dataset_to_csv(frames, tmp_path, fields=["NotAField"])


def test_write_dataset_to_csv_accepts_symbol_keyed_frames_and_pivots(tmp_path):
    symbol_frames = {
        "AAPL": pd.DataFrame(
            {"Close": [1.0, 2.0], "Volume": [10.0, 20.0]}, index=pd.date_range("2024-01-01", periods=2)
        ),
        "MSFT": pd.DataFrame(
            {"Close": [3.0, 4.0], "Volume": [30.0, 40.0]}, index=pd.date_range("2024-01-01", periods=2)
        ),
    }
    output = data_sources.write_dataset_to_csv(symbol_frames, tmp_path)

    back = pd.read_csv(output / "Close.csv", index_col=0, parse_dates=True)
    assert list(back.columns) == ["AAPL", "MSFT"]
    assert back["MSFT"].tolist() == [3.0, 4.0]


def test_write_dataset_to_csv_invalid_shape_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="fetch_ohlcv"):
        data_sources.write_dataset_to_csv({"weird": pd.DataFrame({"junk": [1.0]})}, tmp_path)


def test_write_dataset_to_csv_skips_empty_frames_with_warning(tmp_path, caplog):
    empty = {"Close": pd.DataFrame(columns=["AAPL"]), "Open": make_field_frames()["Open"]}
    with caplog.at_level(logging.WARNING):
        output = data_sources.write_dataset_to_csv(empty, tmp_path)
    assert not (output / "Close.csv").exists()
    assert (output / "Open.csv").exists()
    assert "Close" in caplog.text


def test_pivot_ohlcv_to_fields_returns_field_keyed_frames():
    symbol_frames = {
        "AAPL": pd.DataFrame({"Close": [1.0], "Volume": [2.0]}, index=pd.date_range("2024-01-01", periods=1)),
        "MSFT": pd.DataFrame({"Close": [3.0], "Volume": [4.0]}, index=pd.date_range("2024-01-01", periods=1)),
    }
    fields = data_sources.pivot_ohlcv_to_fields(symbol_frames)
    assert list(fields) == ["Close", "Volume"]
    assert list(fields["Close"].columns) == ["AAPL", "MSFT"]


def test_write_dataset_manifest_contents(tmp_path):
    frames = make_field_frames()
    output = data_sources.write_dataset_to_csv(frames, tmp_path)
    manifest_path = data_sources.write_dataset_manifest(
        output, "crypto_spot", ["BTC-USD", "ETH-USD"], "yfinance", fetched_at="2026-01-01T00:00:00+00:00"
    )

    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"] == "crypto_spot"
    assert manifest["symbols"] == ["BTC-USD", "ETH-USD"]
    assert manifest["source"] == "yfinance"
    assert manifest["fetched_at"] == "2026-01-01T00:00:00+00:00"
    assert manifest["row_column_counts"]["Close"] == {"rows": 5, "columns": 2}
    assert set(manifest["row_column_counts"]) == set(data_sources.OHLCV_FIELDS)


def test_write_dataset_manifest_defaults_timestamp_and_survives_empty_dir(tmp_path):
    manifest_path = data_sources.write_dataset_manifest(tmp_path / "none", "sentiment", ["AAPL"], "news")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"] == "sentiment"
    assert manifest["row_column_counts"] == {}
    assert manifest["fetched_at"]  # auto-filled UTC timestamp


# ---------------------------------------------------------------------------
# CLI: download_market_data
# ---------------------------------------------------------------------------

def test_cli_dry_run_sp500_prints_plan(capsys):
    rc = download_market_data.main(["--dataset", "sp500", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Dataset: sp500" in out
    assert "AAPL" in out  # symbols printed
    assert "Open, High, Low, Close, Volume" in out  # fields printed
    assert "data/sp500" in out  # output path printed
    assert "Requires network: yes" in out
    assert "Dry run: no data was downloaded." in out


def test_cli_dry_run_crypto_spot_prints_plan_and_normalized_symbols(capsys):
    rc = download_market_data.main(["--dataset", "crypto_spot", "--symbols", "BTC,ETH", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Dataset: crypto_spot" in out
    assert "BTC-USD, ETH-USD" in out
    assert "Output directory:" in out


def test_cli_dry_run_does_not_download_or_create_output(tmp_path, capsys):
    def exploding_fetcher(*args):
        raise AssertionError("dry-run must not fetch")

    rc = download_market_data.main(
        ["--dataset", "crypto_spot", "--dry-run", "--output", str(tmp_path / "sub")],
        ohlcv_fetcher=exploding_fetcher,
    )
    assert rc == 0
    assert not (tmp_path / "sub").exists()


def test_cli_all_presets_prints_registry_table(capsys):
    rc = download_market_data.main(["--all-presets"])
    out = capsys.readouterr().out

    assert rc == 0
    for name in REQUIRED_DATASETS:
        assert name in out
    assert "binance" in out
    assert "yfinance" in out


def test_cli_unknown_dataset_returns_error_code(capsys):
    rc = download_market_data.main(["--dataset", "nope", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "Available datasets" in out


def test_cli_end_to_end_offline_with_injected_fetchers(tmp_path, capsys):
    rc = download_market_data.main(
        [
            "--dataset",
            "crypto_spot",
            "--symbols",
            "BTC,ETH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--output",
            str(tmp_path),
            "--include-fundamentals",
        ],
        ohlcv_fetcher=make_ohlcv_fetcher(rows=5),
        fundamentals_fetcher=make_fundamentals_fetcher(),
    )
    out = capsys.readouterr().out

    assert rc == 0
    for field in data_sources.OHLCV_FIELDS:
        assert (tmp_path / f"{field}.csv").exists()
    for field in data_sources.FUNDAMENTAL_FIELDS:
        assert (tmp_path / f"{field}.csv").exists()
    close = pd.read_csv(tmp_path / "Close.csv", index_col=0, parse_dates=True)
    assert list(close.columns) == ["BTC-USD", "ETH-USD"]
    market_cap = pd.read_csv(tmp_path / "marketCap.csv", index_col=0)
    assert list(market_cap.columns) == ["BTC-USD", "ETH-USD"]
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "crypto_spot"
    assert manifest["symbols"] == ["BTC-USD", "ETH-USD"]
    assert "Manifest:" in out


def test_cli_binance_futures_delegates_without_downloading(tmp_path, capsys):
    rc = download_market_data.main(["--dataset", "crypto_futures", "--output", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "download_binance_futures" in out
    assert list(tmp_path.glob("*.csv")) == []  # nothing downloaded by this CLI


def test_cli_sentiment_without_provider_writes_nothing_and_warns(tmp_path, caplog, capsys):
    rc = download_market_data.main(["--dataset", "sentiment", "--output", str(tmp_path)])

    assert rc == 0
    assert list(tmp_path.glob("*.csv")) == []
    assert "No sentiment provider configured" in caplog.text
    assert "No data files were written" in capsys.readouterr().out


def test_cli_fundamentals_dataset_end_to_end_offline(tmp_path):
    rc = download_market_data.main(
        ["--dataset", "fundamentals", "--symbols", "AAPL,MSFT", "--output", str(tmp_path)],
        fundamentals_fetcher=make_fundamentals_fetcher(),
    )

    assert rc == 0
    assert (tmp_path / "marketCap.csv").exists()
    assert not (tmp_path / "Close.csv").exists()  # fundamentals preset fetches no OHLCV
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "fundamentals"
    assert manifest["symbols"] == ["AAPL", "MSFT"]
