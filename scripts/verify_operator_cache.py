#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for operator memoization cache.
# Tests that identical calls hit the cache, different implementations don't share,
# and clear_operator_cache() empties the cache.

"""Verification of operator memoization: identical calls hit cache, different impls don't share."""

import sys
from pathlib import Path

# Robust import handling
try:
    from signal_discovery_workflow.operator_cache import (
        OperatorCache,
        data_fingerprint,
        install_cached_operators,
        memoize_operator,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from signal_discovery_workflow.operator_cache import (
        OperatorCache,
        data_fingerprint,
        install_cached_operators,
        memoize_operator,
    )

import numpy as np
import pandas as pd


def test_identical_calls_hit_cache():
    """Test that identical calls hit the cache."""
    cache = OperatorCache(max_entries=128)

    # Create a simple deterministic function
    def add_fn(df, x=1):
        return df + x

    # Wrap it
    wrapped = memoize_operator(add_fn, cache)

    # Create test data
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.random((10, 3)), columns=["A", "B", "C"])

    # First call - should be a miss
    result1 = wrapped(data.copy())
    stats1 = cache.stats()

    if stats1["hits"] != 0:
        print(f"FAIL: First call should be a miss, but hits = {stats1['hits']}")
        return False
    if stats1["misses"] != 1:
        print(f"FAIL: First call should be a miss, but misses = {stats1['misses']}")
        return False

    # Second call with identical data - should hit
    result2 = wrapped(data.copy())
    stats2 = cache.stats()

    if stats2["hits"] != 1:
        print(f"FAIL: Second identical call should be a hit, but hits = {stats2['hits']}")
        return False
    if stats2["misses"] != 1:
        print(f"FAIL: Second call should still have 1 miss, but misses = {stats2['misses']}")
        return False

    # Results should be equal
    if not np.allclose(result1.values, result2.values):
        print("FAIL: Results from identical calls should be equal")
        return False

    print("PASS: Identical calls hit the cache (hits=1, misses=1)")
    return True


def test_different_impls_dont_share():
    """Test that two DIFFERENT implementations sharing a name do NOT share cached entry."""
    cache = OperatorCache(max_entries=128)

    # Create two different implementations with the same name
    def make_add_fn(increment):
        def add_fn(df):
            return df + increment
        add_fn.__name__ = "AddOp"
        return add_fn

    add_fn_1 = make_add_fn(1)
    add_fn_2 = make_add_fn(2)  # Different implementation!

    # Wrap both
    wrapped_1 = memoize_operator(add_fn_1, cache)
    wrapped_2 = memoize_operator(add_fn_2, cache)

    # Create test data
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.random((10, 3)), columns=["A", "B", "C"])

    # Call first implementation
    result1 = wrapped_1(data.copy())

    # Call second implementation - should NOT hit first's result
    result2 = wrapped_2(data.copy())
    stats2 = cache.stats()

    # Should have 2 misses (different implementations, even with same name)
    # The key difference is the implementation fingerprint
    if stats2["misses"] != 2:
        print(f"FAIL: Different implementations should both miss, got misses = {stats2['misses']}")
        return False

    # Results should be DIFFERENT because implementations differ (2 != 1)
    if np.allclose(result1.values, result2.values):
        print("FAIL: Different implementations should produce different results when they differ")
        return False

    # Verify: result2 = result1 + 1 (since add_fn_2 adds 2, add_fn_1 adds 1)
    # Actually: result1 = data + 1, result2 = data + 2, so result2 = result1 + 1
    if not np.allclose(result2.values - result1.values, 1.0):
        print("FAIL: Results should differ by increment amount (1)")
        return False

    print("PASS: Different implementations with same name do NOT share cache entries")
    return True


def test_clear_operator_cache():
    """Test that cache.clear() empties the cache."""
    # Create a simple function and wrap it
    def test_fn(df):
        return df * 2

    cache = OperatorCache(max_entries=128)
    wrapped = memoize_operator(test_fn, cache)

    # Create test data
    data = pd.DataFrame({"X": [1.0, 2.0, 3.0]})

    # Prime the cache
    _ = wrapped(data)

    stats_before = cache.stats()
    if stats_before["entries"] == 0:
        print("FAIL: Cache should have entries after one call")
        return False

    # Clear the cache using its own clear method
    cache.clear()

    stats_after = cache.stats()
    if stats_after["entries"] != 0:
        print(f"FAIL: Cache should be empty after clear, got {stats_after['entries']} entries")
        return False
    if stats_after["hits"] != 0:
        print(f"FAIL: Cache hits should be 0 after clear, got {stats_after['hits']}")
        return False
    if stats_after["misses"] != 0:
        print(f"FAIL: Cache misses should be 0 after clear, got {stats_after['misses']}")
        return False

    print("PASS: cache.clear() empties the cache and resets counters")
    return True


def test_install_cached_operators():
    """Test install_cached_operators replaces operators with memoized wrappers."""
    # Create a namespace with some callable operators
    def OpA(df):
        return df + 1

    def OpB(df):
        return df * 2

    def helper_fn(df):
        return df - 1  # Not an operator (not in OPERATOR_NAMES typically)

    namespace = {
        "pd": pd,
        "np": np,
        "OpA": OpA,
        "OpB": OpB,
        "helper_fn": helper_fn,
        "constants": 42,  # Non-callable
    }

    cache = OperatorCache(max_entries=128)
    new_namespace = install_cached_operators(namespace, cache, only=["OpA", "OpB"])

    # Check that OpA and OpB are wrapped
    if not getattr(new_namespace["OpA"], "_operator_cache_wrapped", False):
        print("FAIL: OpA should be wrapped")
        return False

    if not getattr(new_namespace["OpB"], "_operator_cache_wrapped", False):
        print("FAIL: OpB should be wrapped")
        return False

    # Check that helper_fn and constants are unchanged
    if new_namespace["helper_fn"] is not helper_fn:
        print("FAIL: helper_fn should not be wrapped")
        return False

    if new_namespace["constants"] != 42:
        print("FAIL: constants should not be wrapped")
        return False

    print("PASS: install_cached_operators correctly wraps specified operators")
    return True


def test_fingerprint_cache():
    """Test that data fingerprint cache works."""
    from signal_discovery_workflow.operator_cache import clear_fingerprint_cache, fingerprint_memo_stats

    # Clear the fingerprint cache first
    clear_fingerprint_cache()

    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.random((10, 3)), columns=["A", "B", "C"])

    # Get fingerprint twice - second should hit memo
    fp1 = data_fingerprint(data)
    fingerprint_memo_stats()

    fp2 = data_fingerprint(data)
    stats2 = fingerprint_memo_stats()

    if stats2["hits"] < 1:
        print("FAIL: Repeating same fingerprint should hit memo")
        return False

    if fp1 != fp2:
        print("FAIL: Fingerprint should be stable")
        return False

    # Clear and check
    clear_fingerprint_cache()
    stats3 = fingerprint_memo_stats()

    if stats3["entries"] != 0:
        print("FAIL: Fingerprint cache should be empty after clear")
        return False

    print("PASS: Fingerprint cache works correctly")
    return True


def main():
    """Run all operator cache verification tests."""
    print("=" * 60)
    print("Operator Cache Verification Tests")
    print("=" * 60)

    results = [
        test_identical_calls_hit_cache(),
        test_different_impls_dont_share(),
        test_clear_operator_cache(),
        test_install_cached_operators(),
        test_fingerprint_cache(),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All operator cache verification tests passed")
        return 0
    else:
        print("FAIL: Some operator cache verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
