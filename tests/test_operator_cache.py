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

"""Tests for the operator-level memoization layer (operator_cache)."""

import functools
import gc
import inspect
import threading
import time

import numpy as np
import pandas as pd
import pytest

import signal_discovery_workflow.operator_cache as operator_cache
from signal_discovery_workflow.operator_cache import (
    OperatorCache,
    clear_fingerprint_cache,
    data_fingerprint,
    fingerprint_memo_stats,
    install_cached_operators,
    memoize_namespace_operators,
    memoize_operator,
    operator_call_key,
    set_fingerprint_memo_enabled,
)

# ---------------------------------------------------------------------------
# Helpers: realistic operators and invocation counters
# ---------------------------------------------------------------------------

def _make_frame(seed: int = 0, rows: int = 10, cols: int = 5) -> pd.DataFrame:
    """Small deterministic price-like frame (50 cells -> fingerprinted exactly)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    columns = [f"T{i:02d}" for i in range(cols)]
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (rows, cols)), axis=0))
    return pd.DataFrame(values, index=index, columns=columns)


def _ts_mean(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-period mean (mimics a calculator.json operator such as TS_Mean)."""
    return df.rolling(d).mean()


def _ts_max(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-period max (mimics TS_Max)."""
    return df.rolling(d).max()


def _counting(fn):
    """Wrap ``fn`` and return (wrapped, calls) where calls["n"] counts real computations."""
    calls = {"n": 0}

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return fn(*args, **kwargs)

    return wrapped, calls


class _Unkeyable:
    """A type data_fingerprint deliberately refuses to fingerprint."""

    def __repr__(self) -> str:  # even a working repr does not make it supported
        return "<_Unkeyable>"


@pytest.fixture
def frame():
    return _make_frame(seed=0)


# ---------------------------------------------------------------------------
# Memoization behaviour on realistic pandas operators
# ---------------------------------------------------------------------------

def test_same_call_computed_once(frame):
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped)

    first = memo(frame, 3)
    second = memo(frame, 3)

    assert calls["n"] == 1                       # underlying function ran once
    assert second is first                       # by-reference contract
    assert first.equals(frame.rolling(3).mean())  # correct pandas semantics


def test_different_window_two_computations(frame):
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped)

    r3 = memo(frame, 3)
    r5 = memo(frame, 5)

    assert calls["n"] == 2
    assert not r3.equals(r5)
    # Re-asking either window still hits the cache.
    memo(frame, 3)
    memo(frame, 5)
    assert calls["n"] == 2


def test_different_content_same_shape_two_computations(frame):
    # Fingerprint-correctness: same shape/labels, different values MUST miss.
    other = frame.copy()
    other.iloc[0, 0] += 1.0
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped)

    memo(frame, 3)
    memo(other, 3)

    assert calls["n"] == 2
    assert data_fingerprint(frame) != data_fingerprint(other)


def test_equal_content_one_computation(frame):
    # Independently built (not copied) frame with identical content: 1 computation.
    twin = pd.DataFrame(
        frame.to_numpy().copy(), index=frame.index, columns=frame.columns
    )
    assert twin is not frame
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped)

    r1 = memo(frame, 3)
    r2 = memo(twin, 3)

    assert calls["n"] == 1
    assert r2 is r1


def test_memoize_operator_idempotent(frame):
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped)
    assert memoize_operator(memo) is memo        # already wrapped -> unchanged
    memo(frame, 3)
    memo(frame, 3)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# stats() and clear()
# ---------------------------------------------------------------------------

def test_stats_hits_misses_hit_rate(frame):
    wrapped, _ = _counting(_ts_mean)
    memo = memoize_operator(wrapped)

    memo(frame, 3)   # miss
    memo(frame, 3)   # hit
    memo(frame, 4)   # miss

    stats = memo._operator_cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["entries"] == 2
    assert stats["max_entries"] == 512
    assert stats["hit_rate"] == pytest.approx(1 / 3)


def test_clear_resets_cache_and_next_call_recomputes(frame):
    cache = OperatorCache(max_entries=8)
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped, cache=cache)

    memo(frame, 3)
    memo(frame, 3)
    assert calls["n"] == 1

    cache.clear()
    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["hits"] == 0 and stats["misses"] == 0

    memo(frame, 3)
    assert calls["n"] == 2
    assert cache.stats()["misses"] == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction_with_max_entries_two(frame):
    cache = OperatorCache(max_entries=2)
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped, cache=cache)

    memo(frame, 3)                    # A
    memo(frame, 5)                    # B
    assert cache.stats()["entries"] == 2
    assert calls["n"] == 2

    memo(frame, 7)                    # C evicts the oldest (A)
    assert cache.stats()["entries"] == 2
    assert calls["n"] == 3

    memo(frame, 3)                    # A was evicted -> recompute
    assert calls["n"] == 4
    assert cache.stats()["entries"] == 2


def test_lru_keeps_recently_used_not_oldest(frame):
    # Distinguishes LRU from FIFO: re-accessing A makes it most-recently-used,
    # so inserting C must evict B, and A must still hit.
    cache = OperatorCache(max_entries=2)
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped, cache=cache)

    memo(frame, 3)                    # A
    memo(frame, 5)                    # B
    memo(frame, 3)                    # A again (hit) -> A is MRU
    assert calls["n"] == 2

    memo(frame, 7)                    # C (miss) -> evicts B (LRU), keeps A
    assert cache.stats()["entries"] == 2
    assert calls["n"] == 3

    memo(frame, 3)                    # A still cached under LRU -> hit
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Wrapper metadata and copy_on_return
# ---------------------------------------------------------------------------

def test_memoize_operator_preserves_metadata():
    def ts_rank(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
        """Rolling rank (docstring must survive wrapping)."""
        return df.rolling(d).rank()

    memo = memoize_operator(ts_rank)
    assert memo.__name__ == "ts_rank"
    assert memo.__doc__ == ts_rank.__doc__
    assert inspect.signature(memo) == inspect.signature(ts_rank)
    assert getattr(memo, "__wrapped__") is ts_rank


def test_copy_on_return_returns_distinct_equal_object(frame):
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped, copy_on_return=True)

    r1 = memo(frame, 3)
    r2 = memo(frame, 3)

    assert calls["n"] == 1
    assert r1 is not r2               # distinct objects ...
    assert r1.equals(r2)              # ... with equal values


def test_default_returns_cached_object_by_reference(frame):
    wrapped, _ = _counting(_ts_mean)
    memo = memoize_operator(wrapped)
    r1 = memo(frame, 3)
    r2 = memo(frame, 3)
    assert r1 is r2


# ---------------------------------------------------------------------------
# Exotic argument types: caching must never raise
# ---------------------------------------------------------------------------

def test_unsupported_argument_type_falls_back_to_uncached_call():
    cache = OperatorCache()
    calls = {"n": 0}

    def take_any(x):
        """Operator that accepts anything."""
        calls["n"] += 1
        return f"run-{calls['n']}"

    memo = memoize_operator(take_any, cache=cache)
    weird = _Unkeyable()

    assert memo(weird) == "run-1"     # works, no exception
    assert memo(weird) == "run-2"     # recomputed every time: cache bypassed
    # The cache was never consulted for unkeyable arguments.
    assert cache.stats() == {
        "hits": 0,
        "misses": 0,
        "entries": 0,
        "max_entries": 512,
        "hit_rate": 0.0,
    }


# ---------------------------------------------------------------------------
# install_cached_operators
# ---------------------------------------------------------------------------

def test_install_cached_operators_returns_new_dict_and_wraps_operators():
    raw_mean, mean_calls = _counting(_ts_mean)
    raw_max, max_calls = _counting(_ts_max)
    namespace = {
        "__name__": "generated_module",
        "TS_Mean": raw_mean,
        "TS_Max": raw_max,
        "pd": pd,           # module: not callable -> passthrough
        "scaling": 3,       # plain value -> passthrough
        "print": print,     # builtin -> never wrapped
        "len": len,         # builtin -> never wrapped
    }

    out = install_cached_operators(namespace)

    assert out is not namespace
    assert namespace["TS_Mean"] is raw_mean      # input dict untouched
    assert set(out) == set(namespace)
    assert getattr(out["TS_Mean"], "_operator_cache_wrapped", False) is True
    assert out["pd"] is pd
    assert out["scaling"] == 3
    assert out["print"] is print
    assert out["len"] is len

    frame = _make_frame()
    out["TS_Mean"](frame, 3)
    out["TS_Mean"](frame, 3)
    assert mean_calls["n"] == 1                  # memoized through the namespace
    out["TS_Max"](frame, 3)
    assert max_calls["n"] == 1


def test_install_cached_operators_idempotent_no_double_wrap():
    raw_mean, mean_calls = _counting(_ts_mean)
    namespace = {"TS_Mean": raw_mean}

    once = install_cached_operators(namespace)
    twice = install_cached_operators(once)

    assert twice["TS_Mean"] is once["TS_Mean"]   # same wrapper, not re-wrapped
    frame = _make_frame()
    twice["TS_Mean"](frame, 3)
    twice["TS_Mean"](frame, 3)
    assert mean_calls["n"] == 1


def test_install_cached_operators_only_filter():
    raw_mean, _ = _counting(_ts_mean)
    raw_max, _ = _counting(_ts_max)
    namespace = {"TS_Mean": raw_mean, "TS_Max": raw_max}

    out = install_cached_operators(namespace, only={"TS_Mean"})

    assert getattr(out["TS_Mean"], "_operator_cache_wrapped", False) is True
    assert out["TS_Max"] is raw_max              # excluded from wrapping


def test_install_cached_operators_skips_dunder_callables():
    sentinel = lambda: None  # noqa: E731  (callable with a dunder name)
    namespace = {"__dunder_callable__": sentinel, "TS_Mean": _ts_mean}
    out = install_cached_operators(namespace)
    assert out["__dunder_callable__"] is sentinel
    assert getattr(out["TS_Mean"], "_operator_cache_wrapped", False) is True


def test_install_cached_operators_shares_wrapper_for_same_object():
    raw_mean, _ = _counting(_ts_mean)
    namespace = {"A": raw_mean, "B": raw_mean}   # same callable under two names
    out = install_cached_operators(namespace)
    assert out["A"] is out["B"]


# ---------------------------------------------------------------------------
# memoize_namespace_operators (evaluator integration point)
# ---------------------------------------------------------------------------

def test_memoize_namespace_operators_keeps_source_and_memoizes_namespace():
    raw_mean, mean_calls = _counting(_ts_mean)
    namespace = {"TS_Mean": raw_mean, "pd": pd, "note": "not callable"}
    code = "def signal(Close):\n    return TS_Mean(Close, 5)\n"

    source, cached_ns, cache = memoize_namespace_operators(code, namespace)

    assert source == code
    assert source is code                        # the LLM's code is never rewritten
    assert cached_ns["TS_Mean"] is not raw_mean
    assert getattr(cached_ns["TS_Mean"], "_operator_cache_wrapped", False) is True
    assert cached_ns["pd"] is pd
    assert cached_ns["note"] == "not callable"
    assert namespace["TS_Mean"] is raw_mean      # input namespace untouched

    frame = _make_frame()
    cached_ns["TS_Mean"](frame, 3)
    cached_ns["TS_Mean"](frame, 3)
    assert mean_calls["n"] == 1
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert cached_ns["TS_Mean"]._operator_cache is cache


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_cache_hits_are_thread_safe(frame):
    cache = OperatorCache(max_entries=32)
    wrapped, calls = _counting(_ts_mean)
    memo = memoize_operator(wrapped, cache=cache)

    expected = memo(frame, 3)                    # warm the cache single-threaded
    assert calls["n"] == 1

    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(25):
                result = memo(frame, 3)
                assert result is expected
        except Exception as exc:  # pragma: no cover - collected below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert calls["n"] == 1                       # no thread ever recomputed
    stats = cache.stats()
    assert stats["hits"] == 8 * 25
    assert stats["misses"] == 1
    assert stats["entries"] == 1


def test_concurrent_cold_calls_all_return_correct_result(frame):
    def slow_mean(df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling mean with an artificial delay to widen the race window."""
        time.sleep(0.02)
        return df.rolling(d).mean()

    wrapped, calls = _counting(slow_mean)
    memo = memoize_operator(wrapped)

    barrier = threading.Barrier(8)
    results: list[pd.DataFrame] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait(timeout=10)
            results.append(memo(frame, 3))
        except Exception as exc:  # pragma: no cover - collected below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 8
    expected = frame.rolling(3).mean()
    for result in results:
        assert result.equals(expected)
    # Cold-race threads may each compute once (documented trade-off); the
    # cache must still hold exactly one consistent entry afterwards.
    assert calls["n"] >= 1
    assert memo._operator_cache.stats()["entries"] == 1


# ---------------------------------------------------------------------------
# OperatorCache unit behaviour
# ---------------------------------------------------------------------------

def test_cache_get_put_and_stats():
    cache = OperatorCache(max_entries=4)
    assert cache.get("a") is None                # miss on empty cache
    cache.put("a", 1)
    assert cache.get("a") == 1                   # hit
    cache.put("b", None)                         # None is a storable value
    assert cache.get("b") is None                # hit that returns the stored None

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["entries"] == 2
    assert stats["max_entries"] == 4
    assert stats["hit_rate"] == pytest.approx(2 / 3)


def test_cache_hit_rate_zero_when_no_lookups():
    assert OperatorCache().stats()["hit_rate"] == 0.0


def test_cache_put_updates_value_and_recency():
    cache = OperatorCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 9)                            # refresh value + recency
    cache.put("c", 3)                            # evicts b, not a
    assert cache.get("a") == 9
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_cache_rejects_invalid_max_entries():
    with pytest.raises(ValueError):
        OperatorCache(max_entries=0)


# ---------------------------------------------------------------------------
# operator_call_key
# ---------------------------------------------------------------------------

def test_call_key_fingerprints_frames_and_keeps_scalars(frame):
    key1 = operator_call_key("TS_Mean", (frame, 5), {})
    key2 = operator_call_key("TS_Mean", (frame.copy(), 5), {})
    assert key1 == key2                          # equal content -> same key

    assert operator_call_key("TS_Mean", (frame, 5), {}) != operator_call_key("TS_Std", (frame, 5), {})
    assert operator_call_key("TS_Mean", (frame, 5), {}) != operator_call_key("TS_Mean", (frame, 6), {})

    # Documented limitation: positional vs keyword binding are distinct keys.
    assert operator_call_key("TS_Mean", (frame, 5), {}) != operator_call_key("TS_Mean", (frame,), {"d": 5})


def test_call_key_kwargs_sorted_regardless_of_dict_order():
    a = operator_call_key("f", (), {"d": 5, "w": 2})
    b = operator_call_key("f", (), {"w": 2, "d": 5})
    assert a == b


def test_call_key_different_content_different_key():
    assert operator_call_key("f", (_make_frame(0),), {}) != operator_call_key("f", (_make_frame(1),), {})


def test_call_key_accepts_numpy_scalar_arguments(frame):
    key1 = operator_call_key("f", (frame, np.int64(5)), {})
    key2 = operator_call_key("f", (frame, np.int64(5)), {})
    assert key1 == key2


# ---------------------------------------------------------------------------
# data_fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_equal_content_equal_string():
    a = _make_frame(0)
    b = _make_frame(0)
    assert a is not b and a.equals(b)
    assert data_fingerprint(a) == data_fingerprint(b)


def test_fingerprint_detects_single_value_change():
    a = _make_frame(0)                           # 10x5 = 50 cells: all sampled (stride 1)
    b = a.copy()
    b.iloc[0, 0] += 1.0
    assert data_fingerprint(a) != data_fingerprint(b)


def test_fingerprint_detects_nan_pattern_change():
    a = _make_frame(0)
    b = a.copy()
    b.iloc[3, 2] = np.nan
    assert data_fingerprint(a) != data_fingerprint(b)


def test_fingerprint_detects_structure_changes():
    a = _make_frame(0)
    renamed = a.copy()
    renamed.columns = [column + "_x" for column in a.columns]
    reindexed = a.copy()
    reindexed.index = a.index + pd.Timedelta(days=1)
    assert data_fingerprint(a) != data_fingerprint(a.T)                  # shape
    assert data_fingerprint(a) != data_fingerprint(renamed)              # columns
    assert data_fingerprint(a) != data_fingerprint(reindexed)            # index
    assert data_fingerprint(a) != data_fingerprint(a.astype(np.int64))   # dtypes


def test_fingerprint_full_content_large_frame():
    big = _make_frame(3, rows=300, cols=8)       # 2400 cells -> full-content hashing
    assert data_fingerprint(big) == data_fingerprint(big.copy())
    assert data_fingerprint(big) != data_fingerprint(big + 1000.0)


def test_fingerprint_numpy_arrays():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert data_fingerprint(a) == data_fingerprint(b)
    c = a.copy()
    c[0, 0] = 99.0
    assert data_fingerprint(a) != data_fingerprint(c)
    assert data_fingerprint(a) != data_fingerprint(a.reshape(4, 3))      # shape folded in
    assert data_fingerprint(a) != data_fingerprint(a.astype(np.float32))  # dtype folded in


def test_fingerprint_series():
    s1 = pd.Series([1.0, 2.0, 3.0])
    s2 = pd.Series([1.0, 2.0, 3.0])
    assert data_fingerprint(s1) == data_fingerprint(s2)
    assert data_fingerprint(s1) != data_fingerprint(pd.Series([1.0, 2.0, 4.0]))
    assert data_fingerprint(s1) != data_fingerprint(pd.Series([1.0, 2.0, 3.0], index=[7, 8, 9]))


def test_fingerprint_scalars_stable_and_distinct():
    assert data_fingerprint(None) == data_fingerprint(None)
    assert data_fingerprint(5) == data_fingerprint(5)
    values = [5, 5.0, "5", True, None]
    fingerprints = {data_fingerprint(value) for value in values}
    assert len(fingerprints) == len(values)


def test_fingerprint_tuple_recursion():
    assert data_fingerprint((1, "a")) == data_fingerprint((1, "a"))
    assert data_fingerprint((1, "a")) != data_fingerprint((1, "b"))


def test_fingerprint_unsupported_type_raises():
    with pytest.raises(TypeError):
        data_fingerprint(object())


# ---------------------------------------------------------------------------
# Regression: implementation identity in the cache key (Defect 1)
# ---------------------------------------------------------------------------

def _make_ts_return(semantics: str):
    """Factory returning DIFFERENT functions that share the SAME __name__."""

    def TS_Return(df: pd.DataFrame, d: int) -> pd.DataFrame:
        if semantics == "pct":
            return df.pct_change(d)
        return df.diff(d) * 100.0

    return TS_Return


def test_same_name_different_implementation_no_shared_entry():
    data = _make_frame(0)
    cache = OperatorCache(max_entries=16)
    w_pct = memoize_operator(_make_ts_return("pct"), cache)
    w_diff = memoize_operator(_make_ts_return("diff"), cache)

    o1 = w_pct(data, 1)
    o2 = w_diff(data, 1)

    assert o1.equals(data.pct_change(1))
    assert o2.equals(data.diff(1) * 100.0)
    assert not o2.equals(o1)
    stats = cache.stats()
    assert stats["hits"] == 0       # nothing may be shared across implementations
    assert stats["misses"] == 2


# ---------------------------------------------------------------------------
# Regression: exact content fingerprint (Defect 2)
# ---------------------------------------------------------------------------

def test_large_frame_single_cell_change_changes_fingerprint():
    big = _make_frame(5, rows=10000, cols=10)        # 100,000 elements
    touched = big.copy()
    # Flat index 50000 is NOT a multiple of the old sampled stride
    # (100000 // 64 = 1562), so the old sampled fingerprint missed this change.
    touched.iloc[5000, 0] += 12345.0
    assert big.shape == touched.shape
    assert data_fingerprint(big) != data_fingerprint(touched)


# ---------------------------------------------------------------------------
# Backwards compatibility and implementation-fingerprint stability
# ---------------------------------------------------------------------------

def test_operator_call_key_three_arg_form_unchanged(frame):
    key1 = operator_call_key("TS_Return", (frame,), {})
    key2 = operator_call_key("TS_Return", (frame,), {})
    assert key1 == key2                          # deterministic without fn
    assert len(key1) == 3                        # exactly the historical shape
    assert operator_call_key("TS_Return", (_make_frame(1),), {}) != key1


def test_identical_implementations_produce_same_key_and_share_entries():
    data = _make_frame(0)
    cache = OperatorCache(max_entries=16)
    w1 = memoize_operator(_make_ts_return("pct"), cache)
    w2 = memoize_operator(_make_ts_return("pct"), cache)  # separately created, identical body

    r1 = w1(data, 1)
    r2 = w2(data, 1)
    assert r2 is r1                              # cross-wrapper reuse still works
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    # Key level: separately-created closures with identical bodies -> same key.
    a = operator_call_key("TS_Return", (data, 1), {}, fn=_make_ts_return("pct"))
    b = operator_call_key("TS_Return", (data, 1), {}, fn=_make_ts_return("pct"))
    assert a == b
    assert len(a) == 4                           # name + args + kwargs + implementation tag


def test_call_key_distinguishes_implementations_at_key_level():
    data = _make_frame(0)
    k_pct = operator_call_key("TS_Return", (data, 1), {}, fn=_make_ts_return("pct"))
    k_diff = operator_call_key("TS_Return", (data, 1), {}, fn=_make_ts_return("diff"))
    assert k_pct != k_diff
    # ... while the plain 3-arg form cannot tell them apart (documented old behaviour).
    assert operator_call_key("TS_Return", (data, 1), {}) == operator_call_key("TS_Return", (data, 1), {})


# ---------------------------------------------------------------------------
# Per-object fingerprint memo
# ---------------------------------------------------------------------------

def test_same_object_fingerprinted_once_memo_hit(monkeypatch):
    clear_fingerprint_cache()
    df = _make_frame(0)
    first = data_fingerprint(df)
    assert fingerprint_memo_stats()["misses"] >= 1

    calls = {"n": 0}
    original = operator_cache._fingerprint_frame

    def counting(frame):
        calls["n"] += 1
        return original(frame)

    monkeypatch.setattr(operator_cache, "_fingerprint_frame", counting)
    second = data_fingerprint(df)
    monkeypatch.undo()

    assert second == first                # identical string ...
    assert calls["n"] == 0                # ... served from the memo, content never re-hashed
    assert fingerprint_memo_stats()["hits"] >= 1


def test_equal_distinct_objects_same_fingerprint_with_memo():
    clear_fingerprint_cache()
    a = _make_frame(7)
    b = _make_frame(7)                    # distinct objects, equal content
    assert a is not b
    assert data_fingerprint(a) == data_fingerprint(b)   # identity must not leak into the value
    assert data_fingerprint(b) == data_fingerprint(a)   # both memoised, still equal


def test_inplace_mutation_is_stale_until_cleared():
    clear_fingerprint_cache()
    df = _make_frame(0)
    before = data_fingerprint(df)

    df.iloc[0, 0] += 999.0
    assert data_fingerprint(df) == before  # DOCUMENTED: in-place mutation is unsupported -> stale

    clear_fingerprint_cache()
    assert data_fingerprint(df) != before  # after clearing, the new content is visible


def test_weakref_cleanup_releases_memo_entry():
    clear_fingerprint_cache()
    holder = {"df": _make_frame(9)}
    fingerprint = data_fingerprint(holder["df"])
    assert fingerprint_memo_stats()["entries"] == 1

    del holder["df"]                      # last strong reference dropped
    gc.collect()
    assert fingerprint_memo_stats()["entries"] == 0   # weakref callback removed the entry

    # And the fingerprint of an equal fresh object is unaffected.
    assert data_fingerprint(_make_frame(9)) == fingerprint


def test_fingerprint_memo_bounded_and_drains():
    clear_fingerprint_cache()
    frames = [_make_frame(seed) for seed in range(80)]   # more than the 64-entry bound
    for frame in frames:
        data_fingerprint(frame)
    del frame                                            # drop the loop variable's reference
    assert fingerprint_memo_stats()["entries"] == 64      # LRU-bounded

    frames.clear()
    gc.collect()
    assert fingerprint_memo_stats()["entries"] == 0       # weakref callbacks drained everything


def test_fingerprint_memo_can_be_disabled(monkeypatch):
    clear_fingerprint_cache()
    set_fingerprint_memo_enabled(False)
    try:
        df = _make_frame(11)
        calls = {"n": 0}
        original = operator_cache._fingerprint_frame

        def counting(frame):
            calls["n"] += 1
            return original(frame)

        monkeypatch.setattr(operator_cache, "_fingerprint_frame", counting)
        data_fingerprint(df)
        data_fingerprint(df)
        assert calls["n"] == 2                            # recomputed every time, never memoised
        stats = fingerprint_memo_stats()
        assert stats["enabled"] is False
        assert stats["entries"] == 0
    finally:
        set_fingerprint_memo_enabled(True)
        clear_fingerprint_cache()
