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

"""Tests for paper-trading configuration, balance, sizing, and candidate selection."""

import pytest

from signal_discovery_workflow.paper_trading import (
    PaperConfig,
    PaperConfigError,
    classify_consensus_regime,
    classify_sma_regime,
    combine_regimes,
    free_balance,
    load_paper_config,
    position_notional_cap,
    regime_alignment,
    select_open_candidates,
    side_counts,
    size_position_qty,
)


class TestPaperConfig:
    def test_valid_file_is_loaded_and_validated(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text(
            """
max_open_positions: 5
starting_balance: 25000.5
max_position_size_pct: 25
max_position_size_amount: 1750
""",
            encoding="utf-8",
        )

        config = load_paper_config(path)

        assert config.max_open_positions == 5
        assert config.starting_balance == pytest.approx(25000.5)
        assert config.max_position_size_pct == pytest.approx(25.0)
        assert config.max_position_size_amount == pytest.approx(1750.0)
        assert config.path == path
        assert config.file_missing is False
        assert config.unknown_keys == ()

    def test_missing_file_returns_defaults_without_error(self, tmp_path):
        path = tmp_path / "missing.yml"

        config = load_paper_config(path)

        assert config.max_open_positions == 3
        assert config.starting_balance == pytest.approx(10000.0)
        assert config.max_position_size_pct is None
        assert config.max_position_size_amount is None
        assert config.file_missing is True
        assert config.path == path

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_open_positions", -1),
            ("max_open_positions", 1.5),
            ("max_open_positions", True),
            ("max_open_positions", "'3'"),
            ("starting_balance", 0),
            ("starting_balance", -100),
            ("starting_balance", ".nan"),
            ("starting_balance", ".inf"),
            ("starting_balance", "'10000'"),
            ("max_position_size_pct", 0),
            ("max_position_size_pct", -1),
            ("max_position_size_pct", ".nan"),
            ("max_position_size_pct", "'25'"),
            ("max_position_size_amount", 0),
            ("max_position_size_amount", -1000),
            ("max_position_size_amount", ".inf"),
            ("max_position_size_amount", "'1000'"),
        ],
    )
    def test_invalid_values_raise_clear_error(self, tmp_path, key, value):
        path = tmp_path / "paper-trading.yml"
        path.write_text(f"{key}: {value}\n", encoding="utf-8")

        with pytest.raises(PaperConfigError, match=key):
            load_paper_config(path)

    def test_zero_max_open_positions_is_valid(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text("max_open_positions: 0\n", encoding="utf-8")

        config = load_paper_config(path)

        assert config.max_open_positions == 0

    def test_unknown_key_warns_but_does_not_crash(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text(
            "max_open_positions: 2\nunknown_setting: ignored\n",
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="unknown_setting"):
            config = load_paper_config(path)

        assert config.max_open_positions == 2
        assert config.unknown_keys == ("unknown_setting",)
        assert config.warnings

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text("max_open_positions: [1, 2\n", encoding="utf-8")

        with pytest.raises(PaperConfigError):
            load_paper_config(path)

    def test_environment_variable_overrides_config_path(self, tmp_path, monkeypatch):
        path = tmp_path / "custom-paper.yml"
        path.write_text("max_open_positions: 4\n", encoding="utf-8")
        monkeypatch.setenv("PAPER_TRADING_CONFIG", str(path))

        config = load_paper_config()

        assert config.path == path
        assert config.max_open_positions == 4

    def test_flat_yaml_fallback_is_used_without_pyyaml(self, tmp_path, monkeypatch):
        path = tmp_path / "paper-trading.yml"
        path.write_text(
            "# fallback config\nmax_open_positions: 2\nstarting_balance: 5000\n"
            "max_position_size_pct: null\nmax_position_size_amount: 250 # USDT\n",
            encoding="utf-8",
        )
        real_import = __import__

        def import_without_pyyaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("PyYAML intentionally unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", import_without_pyyaml)

        config = load_paper_config(path)

        assert config.max_open_positions == 2
        assert config.starting_balance == pytest.approx(5000.0)
        assert config.max_position_size_pct is None
        assert config.max_position_size_amount == pytest.approx(250.0)


class TestFreeBalance:
    def test_starting_balance_only(self):
        config = PaperConfig(starting_balance=10000.0)

        assert free_balance(config, {}) == pytest.approx(10000.0)

    def test_realized_pnl_is_added(self):
        config = PaperConfig(starting_balance=10000.0)

        balance = free_balance(config, {"cumulative_closed_pnl": 125.5})

        assert balance == pytest.approx(10125.5)

    def test_open_notional_is_subtracted(self):
        config = PaperConfig(starting_balance=10000.0)
        state = {
            "positions": [
                {"symbol": "BTCUSDT", "entry": 100.0, "qty": 2.0},
                {"symbol": "ETHUSDT", "entry": 50.0, "qty": 1.5},
            ]
        }

        balance = free_balance(config, state)

        assert balance == pytest.approx(9725.0)

    def test_legacy_state_without_new_fields_is_supported(self):
        config = PaperConfig(starting_balance=8000.0)
        state = {"positions": [{"symbol": "BTCUSDT", "entry": 100.0, "qty": 1.0}]}

        assert free_balance(config, state) == pytest.approx(7900.0)

    def test_junk_values_are_ignored(self):
        config = PaperConfig(starting_balance=10000.0)
        state = {
            "cumulative_closed_pnl": "not-a-number",
            "positions": [
                {"symbol": "BAD", "entry": "bad", "qty": 1.0},
                {"symbol": "ALSO_BAD", "entry": 10.0, "qty": None},
                "not-a-position",
            ],
        }

        assert free_balance(config, state) == pytest.approx(10000.0)


class TestPositionNotionalCap:
    def test_percentage_only(self):
        config = PaperConfig(max_position_size_pct=25.0)

        assert position_notional_cap(config, 10000.0) == pytest.approx(2500.0)

    def test_fixed_amount_only(self):
        config = PaperConfig(max_position_size_amount=1200.0)

        assert position_notional_cap(config, 10000.0) == pytest.approx(1200.0)

    def test_stricter_of_percentage_and_amount_wins(self):
        config = PaperConfig(max_position_size_pct=25.0, max_position_size_amount=1200.0)

        assert position_notional_cap(config, 10000.0) == pytest.approx(1200.0)

    def test_percentage_can_be_stricter_than_amount(self):
        config = PaperConfig(max_position_size_pct=10.0, max_position_size_amount=1200.0)

        assert position_notional_cap(config, 10000.0) == pytest.approx(1000.0)

    def test_no_limits_returns_none(self):
        assert position_notional_cap(PaperConfig(), 10000.0) is None


class TestSizePositionQty:
    def test_cap_above_one_unit_price_keeps_legacy_quantity(self):
        quantity, reason = size_position_qty(1.0, 100.0, 200.0)

        assert quantity == pytest.approx(1.0)
        assert reason is None
        assert quantity * 100.0 <= 200.0

    def test_cap_below_one_unit_price_reduces_to_fractional_quantity(self):
        quantity, reason = size_position_qty(1.0, 100.0, 50.0)

        assert quantity == pytest.approx(0.5)
        assert reason is None
        assert quantity * 100.0 <= 50.0

    def test_rounding_down_never_exceeds_cap(self):
        quantity, reason = size_position_qty(1.0, 3.0, 1.0)

        assert quantity == pytest.approx(0.33333333)
        assert reason is None
        assert quantity * 3.0 <= 1.0

    def test_base_quantity_is_rounded_down_to_eight_decimals(self):
        quantity, reason = size_position_qty(1.123456789, 10.0, None)

        assert quantity == pytest.approx(1.12345678)
        assert reason is None

    def test_quantity_below_minimum_returns_skip_reason(self):
        quantity, reason = size_position_qty(1.0, 1.0, 0.0000000001)

        assert quantity == 0.0
        assert reason == "position_notional_below_minimum"

    def test_no_cap_keeps_legacy_quantity(self):
        quantity, reason = size_position_qty(1.0, 5000.0, None)

        assert quantity == pytest.approx(1.0)
        assert reason is None

    def test_cap_never_increases_legacy_base_quantity(self):
        quantity, reason = size_position_qty(0.25, 100.0, 10000.0)

        assert quantity == pytest.approx(0.25)
        assert reason is None


class TestCandidateSelection:
    def test_same_side_skip_does_not_consume_a_slot(self):
        config = PaperConfig(max_open_positions=3)
        open_positions = [
            {"symbol": "A", "side": "long", "entry": 100.0, "qty": 1.0},
        ]

        selected, skipped = select_open_candidates(
            {"A": "long", "B": "long", "C": "long"},
            open_positions,
            {"A": 100.0, "B": 100.0, "C": 100.0},
            config,
        )

        assert [candidate["symbol"] for candidate in selected] == ["B", "C"]
        assert skipped == [
            {"symbol": "A", "side": "long", "reason": "already_open_same_side"},
        ]

    def test_missing_price_is_skipped_without_consuming_a_slot(self):
        config = PaperConfig(max_open_positions=2)

        selected, skipped = select_open_candidates(
            {"A": "long", "B": "long", "C": "long"},
            [],
            {"B": 100.0, "C": 100.0},
            config,
        )

        assert [candidate["symbol"] for candidate in selected] == ["B", "C"]
        assert skipped == [
            {"symbol": "A", "side": "long", "reason": "price_unavailable"},
        ]

    def test_hard_position_cap_is_never_exceeded(self):
        config = PaperConfig(max_open_positions=2)
        open_positions = [
            {"symbol": "EXISTING", "side": "long", "entry": 100.0, "qty": 1.0},
        ]

        selected, skipped = select_open_candidates(
            {"A": "long", "B": "long", "C": "long"},
            open_positions,
            {"A": 100.0, "B": 100.0, "C": 100.0},
            config,
        )

        assert len(open_positions) + len(selected) == 2
        assert skipped == [
            {"symbol": "B", "side": "long", "reason": "position_limit_reached"},
            {"symbol": "C", "side": "long", "reason": "position_limit_reached"},
        ]

    def test_opposite_side_is_a_separate_position_and_consumes_a_slot(self):
        config = PaperConfig(max_open_positions=2)
        open_positions = [
            {"symbol": "A", "side": "long", "entry": 100.0, "qty": 1.0},
        ]

        selected, skipped = select_open_candidates(
            {"A": "short", "B": "long"},
            open_positions,
            {"A": 100.0, "B": 100.0},
            config,
        )

        assert [candidate["symbol"] for candidate in selected] == ["A"]
        assert selected[0]["side"] == "short"
        assert skipped == [
            {"symbol": "B", "side": "long", "reason": "position_limit_reached"},
        ]

    def test_zero_max_open_positions_skips_every_new_candidate(self):
        config = PaperConfig(max_open_positions=0)

        selected, skipped = select_open_candidates(
            {"A": "long"},
            [],
            {"A": 100.0},
            config,
        )

        assert selected == []
        assert skipped == [
            {"symbol": "A", "side": "long", "reason": "position_limit_reached"},
        ]

    def test_percentage_limit_uses_free_balance_after_each_opening(self):
        config = PaperConfig(max_open_positions=3, max_position_size_pct=25.0)

        selected, skipped = select_open_candidates(
            {"A": "long", "B": "long"},
            [],
            {"A": 3000.0, "B": 3000.0},
            config,
        )

        assert skipped == []
        assert len(selected) == 2
        assert selected[0]["sizing_cap"] == pytest.approx(2500.0)
        first_notional = selected[0]["notional"]
        free_after_first = 10000.0 - first_notional
        assert selected[1]["sizing_cap"] == pytest.approx(free_after_first * 0.25)
        assert selected[1]["notional"] <= selected[1]["sizing_cap"]
        assert selected[1]["qty"] < selected[0]["qty"]

    def test_sizing_below_minimum_is_reported_as_a_skip(self):
        config = PaperConfig(max_open_positions=3, max_position_size_amount=0.0000000001)

        selected, skipped = select_open_candidates(
            {"A": "long"},
            [],
            {"A": 1000.0},
            config,
        )

        assert selected == []
        assert skipped == [
            {
                "symbol": "A",
                "side": "long",
                "reason": "position_notional_below_minimum",
            },
        ]


class TestRegimeConfig:
    """Configuration of the per-side cap and the market-regime filter."""

    def test_defaults_keep_the_new_behaviour_off(self, tmp_path):
        config = load_paper_config(tmp_path / "missing.yml")

        assert config.max_positions_per_side is None
        assert config.market_regime_filter is False
        assert config.regime_size_reduction == pytest.approx(0.5)
        assert config.regime_symbol == "BTCUSDT"
        assert config.regime_fast_sma == 20
        assert config.regime_slow_sma == 60

    def test_new_keys_are_loaded(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text(
            """
max_open_positions: 3
max_positions_per_side: 2
market_regime_filter: true
regime_size_reduction: 0.25
regime_symbol: ETHUSDT
regime_fast_sma: 10
regime_slow_sma: 30
""",
            encoding="utf-8",
        )

        config = load_paper_config(path)

        assert config.max_positions_per_side == 2
        assert config.market_regime_filter is True
        assert config.regime_size_reduction == pytest.approx(0.25)
        assert config.regime_symbol == "ETHUSDT"
        assert config.regime_fast_sma == 10
        assert config.regime_slow_sma == 30

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_positions_per_side", -1),
            ("max_positions_per_side", 1.5),
            ("market_regime_filter", "maybe"),
            ("regime_size_reduction", 0),
            ("regime_size_reduction", 1.5),
            ("regime_symbol", ""),
            ("regime_fast_sma", 0),
            ("regime_slow_sma", -5),
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, key, value):
        path = tmp_path / "paper-trading.yml"
        path.write_text(f"{key}: {value}\n", encoding="utf-8")

        with pytest.raises(PaperConfigError):
            load_paper_config(path)

    def test_fast_window_must_be_smaller_than_the_slow_window(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text("regime_fast_sma: 60\nregime_slow_sma: 20\n", encoding="utf-8")

        with pytest.raises(PaperConfigError):
            load_paper_config(path)

    def test_unknown_keys_still_only_warn(self, tmp_path):
        path = tmp_path / "paper-trading.yml"
        path.write_text("max_positions_per_side: 2\nnot_a_key: 5\n", encoding="utf-8")

        with pytest.warns(UserWarning):
            config = load_paper_config(path)

        assert config.max_positions_per_side == 2

    def test_yaml_booleans_are_read_consistently(self, tmp_path):
        """The fallback parser and PyYAML must agree on YAML 1.1 boolean spellings."""

        path = tmp_path / "paper-trading.yml"
        path.write_text("market_regime_filter: no\n", encoding="utf-8")

        config = load_paper_config(path)

        assert config.market_regime_filter is False


class TestSmaRegime:
    """Classification of the market regime from two simple moving averages."""

    def test_uptrend_is_bull(self):
        closes = [100.0 + index for index in range(80)]

        result = classify_sma_regime(closes, 20, 60)

        assert result["regime"] == "bull"
        assert result["fast_sma"] > result["slow_sma"]
        assert result["samples"] == 80

    def test_downtrend_is_bear(self):
        closes = [200.0 - index for index in range(80)]

        result = classify_sma_regime(closes, 20, 60)

        assert result["regime"] == "bear"
        assert result["fast_sma"] < result["slow_sma"]

    def test_flat_market_is_sideways(self):
        result = classify_sma_regime([100.0] * 80, 20, 60)

        assert result["regime"] == "sideways"
        assert result["fast_sma"] == pytest.approx(100.0)
        assert result["slow_sma"] == pytest.approx(100.0)

    def test_too_few_samples_are_unknown(self):
        result = classify_sma_regime([100.0] * 59, 20, 60)

        assert result["regime"] == "unknown"
        assert result["fast_sma"] is None
        assert result["slow_sma"] is None
        assert result["samples"] == 59

    def test_non_finite_input_is_unknown(self):
        closes = [100.0 + index for index in range(80)]
        closes[10] = float("nan")

        result = classify_sma_regime(closes, 20, 60)

        assert result["regime"] == "unknown"

    def test_exact_window_boundary_uses_only_available_samples(self):
        closes = [100.0] * 60

        result = classify_sma_regime(closes, 20, 60)

        assert result["regime"] == "sideways"
        assert result["samples"] == 60


class TestConsensusRegime:
    """Classification of the consensus regime from the candidate balance and strength."""

    def test_buy_dominant_consensus_is_bull(self):
        summary = {
            "top_buy": [{"symbol": "A", "net": 3}, {"symbol": "B", "net": 2}],
            "top_sell": [{"symbol": "C", "net": -1}],
        }

        result = classify_consensus_regime(summary)

        assert result["regime"] == "bull"
        assert result["buy_count"] == 2
        assert result["sell_count"] == 1
        assert result["buy_strength"] == pytest.approx(5.0)
        assert result["sell_strength"] == pytest.approx(1.0)

    def test_sell_dominant_consensus_is_bear(self):
        summary = {
            "top_buy": [{"symbol": "A", "net": 1}],
            "top_sell": [{"symbol": "B", "net": -3}, {"symbol": "C", "net": -2}],
        }

        result = classify_consensus_regime(summary)

        assert result["regime"] == "bear"

    def test_equal_strength_is_neutral(self):
        summary = {
            "top_buy": [{"symbol": "A", "net": 2}],
            "top_sell": [{"symbol": "B", "net": -2}],
        }

        result = classify_consensus_regime(summary)

        assert result["regime"] == "neutral"

    def test_empty_summary_is_neutral(self):
        result = classify_consensus_regime({"top_buy": [], "top_sell": []})

        assert result["regime"] == "neutral"
        assert result["buy_count"] == 0
        assert result["sell_count"] == 0

    def test_missing_summary_is_neutral(self):
        assert classify_consensus_regime(None)["regime"] == "neutral"


class TestCombineRegimes:
    def test_agreement_produces_a_directional_regime(self):
        assert combine_regimes("bull", "bull")["combined"] == "bullish"
        assert combine_regimes("bear", "bear")["combined"] == "bearish"

    def test_disagreement_and_neutral_sources_stay_neutral(self):
        assert combine_regimes("bull", "bear")["combined"] == "neutral"
        assert combine_regimes("bull", "neutral")["combined"] == "neutral"
        assert combine_regimes("sideways", "bull")["combined"] == "neutral"
        assert combine_regimes("unknown", "bear")["combined"] == "neutral"
        assert combine_regimes(None, None)["combined"] == "neutral"

    def test_inputs_are_reported_back(self):
        result = combine_regimes("bull", "bull")

        assert result["btc"] == "bull"
        assert result["consensus"] == "bull"


class TestRegimeAlignment:
    def test_alignment_follows_the_combined_regime(self):
        assert regime_alignment("long", "bullish") == "aligned"
        assert regime_alignment("short", "bullish") == "counter"
        assert regime_alignment("short", "bearish") == "aligned"
        assert regime_alignment("long", "bearish") == "counter"

    def test_neutral_regime_has_no_alignment(self):
        assert regime_alignment("long", "neutral") == "neutral"
        assert regime_alignment("long", None) == "neutral"


class TestSideCap:
    """The per-side concentration cap."""

    def test_third_position_on_one_side_is_skipped(self):
        config = PaperConfig(max_open_positions=3)
        desired = {"A": "long", "B": "long", "C": "long", "D": "short"}

        selected, skipped = select_open_candidates(
            desired,
            [],
            {"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0},
            config,
            max_positions_per_side=2,
        )

        assert [(item["symbol"], item["side"]) for item in selected] == [
            ("A", "long"),
            ("B", "long"),
            ("D", "short"),
        ]
        assert skipped == [{"symbol": "C", "side": "long", "reason": "side_limit_reached"}]

    def test_open_positions_on_the_same_side_count_towards_the_cap(self):
        config = PaperConfig(max_open_positions=3)
        open_positions = [
            {"symbol": "A", "side": "long", "entry": 10.0, "qty": 1.0},
            {"symbol": "B", "side": "long", "entry": 10.0, "qty": 1.0},
        ]

        selected, skipped = select_open_candidates(
            {"C": "long", "D": "short"},
            open_positions,
            {"C": 10.0, "D": 10.0},
            config,
            max_positions_per_side=2,
        )

        assert [item["symbol"] for item in selected] == ["D"]
        assert skipped == [{"symbol": "C", "side": "long", "reason": "side_limit_reached"}]

    def test_a_null_side_cap_preserves_the_legacy_behaviour(self):
        config = PaperConfig(max_open_positions=3)

        selected, skipped = select_open_candidates(
            {"A": "long", "B": "long", "C": "long"},
            [],
            {"A": 10.0, "B": 10.0, "C": 10.0},
            config,
            max_positions_per_side=None,
        )

        assert len(selected) == 3
        assert skipped == []

    def test_side_counts_report_the_open_positions(self):
        positions = [
            {"symbol": "A", "side": "long"},
            {"symbol": "B", "side": "short"},
            {"symbol": "C", "side": "long"},
            {"symbol": "D", "side": "unknown"},
        ]

        assert side_counts(positions) == {"long": 2, "short": 1}


class TestRegimeAwareSelection:
    """Direction-aware ordering and sizing."""

    def test_regime_aligned_candidates_are_evaluated_first(self):
        config = PaperConfig(max_open_positions=1)
        desired = {"LONG": "long", "SHORT": "short"}

        selected, skipped = select_open_candidates(
            desired,
            [],
            {"LONG": 10.0, "SHORT": 10.0},
            config,
            regime="bearish",
        )

        assert [item["symbol"] for item in selected] == ["SHORT"]
        assert skipped == [{"symbol": "LONG", "side": "long", "reason": "position_limit_reached"}]

    def test_a_neutral_regime_preserves_the_supplied_order(self):
        config = PaperConfig(max_open_positions=1)
        desired = {"LONG": "long", "SHORT": "short"}

        selected, _ = select_open_candidates(
            desired,
            [],
            {"LONG": 10.0, "SHORT": 10.0},
            config,
            regime="neutral",
        )

        assert [item["symbol"] for item in selected] == ["LONG"]

    def test_a_missing_regime_preserves_the_supplied_order(self):
        config = PaperConfig(max_open_positions=1)

        selected, _ = select_open_candidates(
            {"LONG": "long", "SHORT": "short"},
            [],
            {"LONG": 10.0, "SHORT": 10.0},
            config,
        )

        assert [item["symbol"] for item in selected] == ["LONG"]

    def test_counter_regime_signals_are_opened_at_half_size(self):
        config = PaperConfig(max_open_positions=3)

        selected, _ = select_open_candidates(
            {"A": "long"},
            [],
            {"A": 100.0},
            config,
            regime="bearish",
            regime_size_reduction=0.5,
        )

        assert len(selected) == 1
        assert selected[0]["qty"] == pytest.approx(0.5)
        assert selected[0]["size_multiplier"] == pytest.approx(0.5)
        assert selected[0]["regime_alignment"] == "counter"
        assert selected[0]["regime_combined"] == "bearish"

    def test_regime_aligned_signals_keep_the_full_size(self):
        config = PaperConfig(max_open_positions=3)

        selected, _ = select_open_candidates(
            {"A": "short"},
            [],
            {"A": 100.0},
            config,
            regime="bearish",
            regime_size_reduction=0.5,
        )

        assert selected[0]["qty"] == pytest.approx(1.0)
        assert selected[0]["size_multiplier"] == pytest.approx(1.0)

    def test_the_notional_cap_still_applies_after_the_reduction(self):
        config = PaperConfig(max_open_positions=3, max_position_size_amount=100.0)

        selected, _ = select_open_candidates(
            {"A": "long"},
            [],
            {"A": 100.0},
            config,
            regime="bearish",
            regime_size_reduction=0.5,
        )

        assert selected[0]["qty"] == pytest.approx(0.5)
        assert selected[0]["notional"] <= selected[0]["sizing_cap"]

    def test_a_counter_regime_size_that_rounds_to_zero_is_skipped(self):
        config = PaperConfig(max_open_positions=3)

        selected, skipped = select_open_candidates(
            {"A": "long"},
            [],
            {"A": 100.0},
            config,
            regime="bearish",
            regime_size_reduction=0.00000000001,
        )

        assert selected == []
        assert skipped == [
            {
                "symbol": "A",
                "side": "long",
                "reason": "regime_reduced_size_below_minimum",
            },
        ]
