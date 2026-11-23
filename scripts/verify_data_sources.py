#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for multi-asset data sources.
# Tests dataset presets resolve, symbol normalization, and CSV round-trip.

"""Verification of multi-asset data sources: presets, normalization, CSV round-trip."""

import sys
import tempfile
from pathlib import Path

# Robust import handling
try:
    from signal_discovery_workflow.data_sources import (
        get_dataset,
        list_datasets,
        normalize_symbol,
        pivot_ohlcv_to_fields,
        write_dataset_to_csv,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from signal_discovery_workflow.data_sources import (
        get_dataset,
        list_datasets,
        normalize_symbol,
        pivot_ohlcv_to_fields,
        write_dataset_to_csv,
    )

import numpy as np
import pandas as pd


def test_dataset_presets_resolve():
    """Test that every dataset preset in DATASETS can be resolved."""
    datasets = list_datasets()

    if not datasets:
        print("FAIL: list_datasets() returned empty list")
        return False

    for name in datasets:
        try:
            meta = get_dataset(name)
            if not isinstance(meta, dict):
                print(f"FAIL: get_dataset('{name}') should return dict")
                return False
            if "fields" not in meta or "source" not in meta:
                print(f"FAIL: get_dataset('{name}') missing required keys")
                return False
        except Exception as e:
            print(f"FAIL: get_dataset('{name}') raised {e}")
            return False

    print(f"PASS: All {len(datasets)} dataset presets resolve correctly")
    return True


def test_symbol_normalization_cases():
    """Test symbol normalization handles documented cases."""
    # Test cases from the docstring of normalize_symbol
    test_cases = [
        ("EUR/USD", "forex_major", "EURUSD=X"),
        ("eurusd", "forex_major", "EURUSD=X"),
        ("BTC", "crypto_spot", "BTC-USD"),
        ("btc-usd", "crypto_spot", "BTC-USD"),
        ("BTC-USDT", "crypto_futures", "BTCUSDT"),
        ("btcusdt", "crypto_futures", "BTCUSDT"),
        (" AAPL ", "sp500", "AAPL"),
        ("aapl", "sp500", "AAPL"),
    ]

    all_passed = True
    for symbol, dataset, expected in test_cases:
        try:
            result = normalize_symbol(symbol, dataset)
            if result != expected:
                print(f"FAIL: normalize_symbol('{symbol}', '{dataset}') = '{result}', expected '{expected}'")
                all_passed = False
            else:
                print(f"PASS: normalize_symbol('{symbol}', '{dataset}') = '{expected}'")
        except Exception as e:
            print(f"FAIL: normalize_symbol('{symbol}', '{dataset}') raised {e}")
            all_passed = False

    return all_passed


def test_forex_normalization_edge_cases():
    """Test forex-specific normalization edge cases."""
    # Additional forex normalization tests
    test_cases = [
        ("EURUSD=X", "forex_major", "EURUSD=X"),
        ("EURUSD", "forex_major", "EURUSD=X"),
        ("EUR-USD", "forex_major", "EURUSD=X"),
    ]

    all_passed = True
    for symbol, dataset, expected in test_cases:
        try:
            result = normalize_symbol(symbol, dataset)
            if result != expected:
                print(f"FAIL: Forex normalize_symbol('{symbol}') = '{result}', expected '{expected}'")
                all_passed = False
        except Exception as e:
            print(f"FAIL: Forex normalize_symbol('{symbol}') raised {e}")
            all_passed = False

    if all_passed:
        print("PASS: Forex normalization edge cases handled correctly")
    return all_passed


def test_crypto_normalization_edge_cases():
    """Test crypto-specific normalization edge cases."""
    test_cases = [
        ("ETH-USD", "crypto_spot", "ETH-USD"),
        ("ETH", "crypto_spot", "ETH-USD"),
        ("SOL/USD", "crypto_spot", "SOL-USD"),
        ("ADAUSDT", "crypto_futures", "ADAUSDT"),
        ("ada-usdt", "crypto_futures", "ADAUSDT"),
    ]

    all_passed = True
    for symbol, dataset, expected in test_cases:
        try:
            result = normalize_symbol(symbol, dataset)
            if result != expected:
                print(f"FAIL: Crypto normalize_symbol('{symbol}') = '{result}', expected '{expected}'")
                all_passed = False
        except Exception as e:
            print(f"FAIL: Crypto normalize_symbol('{symbol}') raised {e}")
            all_passed = False

    if all_passed:
        print("PASS: Crypto normalization edge cases handled correctly")
    return all_passed


def test_csv_round_trip():
    """Test synthetic dataset round-trips through CSV correctly."""
    # Create synthetic OHLCV data
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Create synthetic OHLCV data for 3 symbols
    np.random.seed(42)
    data = {
        "Close": pd.DataFrame(
            100 + np.random.randn(n, 3).cumsum(axis=0),
            index=dates,
            columns=["AAPL", "MSFT", "GOOG"]
        ),
        "Open": pd.DataFrame(
            98 + np.random.randn(n, 3).cumsum(axis=0),
            index=dates,
            columns=["AAPL", "MSFT", "GOOG"]
        ),
        "High": pd.DataFrame(
            102 + np.random.randn(n, 3).cumsum(axis=0),
            index=dates,
            columns=["AAPL", "MSFT", "GOOG"]
        ),
        "Low": pd.DataFrame(
            96 + np.random.randn(n, 3).cumsum(axis=0),
            index=dates,
            columns=["AAPL", "MSFT", "GOOG"]
        ),
        "Volume": pd.DataFrame(
            np.random.randint(1_000_000, 10_000_000, (n, 3)).astype(float),
            index=dates,
            columns=["AAPL", "MSFT", "GOOG"]
        ),
    }

    # Write to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write the dataset
        result_dir = write_dataset_to_csv(data, tmpdir / "test_output")

        # Read back each CSV file
        for field in ["Open", "Close", "High", "Low", "Volume"]:
            csv_path = result_dir / f"{field}.csv"

            if not csv_path.exists():
                print(f"FAIL: {field}.csv was not created")
                return False

            # Read with the same settings as the evaluator uses
            df_read = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df_original = data[field]

            # Check index
            if not df_read.index.equals(df_original.index):
                print(f"FAIL: {field}.csv index doesn't match (dates or order)")
                return False

            # Check columns
            if list(df_read.columns) != list(df_original.columns):
                print(f"FAIL: {field}.csv columns don't match")
                return False

            # Check values - use check_freq=False since CSV can't carry frequency
            df_read = df_read.set_index(df_read.index, verify_integrity=True)

            if not np.allclose(df_read.values, df_original.values, equal_nan=True):
                max_diff = np.abs(df_read.values - df_original.values).max()
                print(f"FAIL: {field}.csv values differ (max diff: {max_diff})")
                return False

    print("PASS: CSV round-trip preserves dates, columns, and values")
    return True


def test_field_pivot():
    """Test pivot_ohlcv_to_fields conversion."""
    # Create symbol-keyed data
    dates = pd.date_range("2022-01-01", periods=5, freq="B")

    symbol_data = {
        "AAPL": pd.DataFrame({
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates),
        "MSFT": pd.DataFrame({
            "Close": [200, 201, 202, 203, 204],
            "Volume": [2000, 2100, 2200, 2300, 2400],
        }, index=dates),
    }

    # Pivot to field-keyed
    fields = pivot_ohlcv_to_fields(symbol_data)

    if "Close" not in fields or "Volume" not in fields:
        print("FAIL: Pivoted data missing expected fields")
        return False

    close_df = fields["Close"]

    # Check shape: 5 rows x 2 columns
    if close_df.shape != (5, 2):
        print(f"FAIL: Close frame shape {close_df.shape}, expected (5, 2)")
        return False

    # Check columns are symbols
    if list(close_df.columns) != ["AAPL", "MSFT"]:
        print(f"FAIL: Close frame columns {list(close_df.columns)}, expected ['AAPL', 'MSFT']")
        return False

    # Check values
    expected_close = np.array([[100, 200], [101, 201], [102, 202], [103, 203], [104, 204]])
    if not np.allclose(close_df.values, expected_close):
        print("FAIL: Close values don't match expected")
        return False

    print("PASS: pivot_ohlcv_to_fields correctly converts symbol-keyed to field-keyed")
    return True


def test_dataset_case_insensitive_get():
    """Test that get_dataset is case-insensitive."""
    # Test with different cases
    test_names = ["SP500", "sp500", "Sp500", "  sp500  "]

    first_result = None
    all_passed = True

    for name in test_names:
        try:
            result = get_dataset(name)
            if first_result is None:
                first_result = result
            elif result != first_result:
                print(f"FAIL: get_dataset('{name}') returned different result")
                all_passed = False
        except Exception as e:
            print(f"FAIL: get_dataset('{name}') raised {e}")
            all_passed = False

    if all_passed:
        print("PASS: get_dataset is case-insensitive")
    return all_passed


def test_unknown_dataset_keyerror():
    """Test that unknown dataset raises KeyError with available list."""
    try:
        get_dataset("nonexistent_dataset")
        print("FAIL: Unknown dataset should raise KeyError")
        return False
    except KeyError as e:
        error_msg = str(e)
        if "Available datasets" not in error_msg:
            print(f"FAIL: KeyError should include 'Available datasets', got: {error_msg}")
            return False
        if "sp500" not in error_msg.lower():
            print(f"FAIL: Error should list available datasets, got: {error_msg}")
            return False
        print("PASS: Unknown dataset raises KeyError with available list")
        return True
    except Exception as e:
        print(f"FAIL: Unknown dataset raised {type(e).__name__}, expected KeyError")
        return False


def test_dataset_has_fields():
    """Test that OHLCV datasets have the expected fields."""

    datasets = list_datasets()
    all_passed = True

    for name in datasets:
        meta = get_dataset(name)
        if "fields" not in meta:
            print(f"FAIL: Dataset '{name}' missing 'fields' key")
            all_passed = False
            continue

        # Not all datasets need OHLCV fields - fundamentals and sentiment have different fields
        # Just verify they have some fields defined
        if not meta["fields"]:
            print(f"FAIL: Dataset '{name}' has empty fields list")
            all_passed = False

    if all_passed:
        print("PASS: All datasets have fields defined")
    return all_passed


def main():
    """Run all data sources verification tests."""
    print("=" * 60)
    print("Data Sources Verification Tests")
    print("=" * 60)

    results = [
        test_dataset_presets_resolve(),
        test_symbol_normalization_cases(),
        test_forex_normalization_edge_cases(),
        test_crypto_normalization_edge_cases(),
        test_csv_round_trip(),
        test_field_pivot(),
        test_dataset_case_insensitive_get(),
        test_unknown_dataset_keyerror(),
        test_dataset_has_fields(),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All data sources verification tests passed")
        return 0
    else:
        print("FAIL: Some data sources verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
