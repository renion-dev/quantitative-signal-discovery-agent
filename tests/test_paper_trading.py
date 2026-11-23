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
    free_balance,
    load_paper_config,
    position_notional_cap,
    select_open_candidates,
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
