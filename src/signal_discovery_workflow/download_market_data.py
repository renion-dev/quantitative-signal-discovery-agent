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
Download any registered dataset into the evaluator-compatible CSV layout.

Thin CLI on top of :mod:`signal_discovery_workflow.data_sources` (no
duplicated download/normalization logic). Writes
``<output>/<Field>.csv`` files (dates x tickers) plus a ``manifest.json``
— the same layout ``signal_evaluator.load_stock_data()`` consumes.

Usage:
    python -m signal_discovery_workflow.download_market_data --all-presets
    python -m signal_discovery_workflow.download_market_data --dataset crypto_spot --dry-run
    python -m signal_discovery_workflow.download_market_data --dataset sectors --start 2015-01-01
    python -m signal_discovery_workflow.download_market_data --dataset sp500 --include-fundamentals
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from signal_discovery_workflow import data_sources
from signal_discovery_workflow.data_sources import (
    fetch_fundamentals,
    fetch_ohlcv,
    fetch_sentiment,
    write_dataset_manifest,
    write_dataset_to_csv,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the downloader CLI."""
    parser = argparse.ArgumentParser(
        prog="download_market_data",
        description="Download a registered dataset into the evaluator-compatible CSV layout.",
    )
    parser.add_argument("--dataset", default="sp500", help="Dataset preset name (see --all-presets).")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol override (normalized per dataset). Defaults to the preset universe.",
    )
    parser.add_argument("--start", default="2012-01-01", help="Start date YYYY-MM-DD. Default: 2012-01-01.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD. Default: today.")
    parser.add_argument("--interval", default="1d", help="Bar interval (OHLCV only). Default: 1d.")
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_ROOT}/<dataset>",
    )
    parser.add_argument(
        "--include-fundamentals",
        action="store_true",
        help="Also fetch fundamentals for the chosen symbols (network required).",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Also fetch sentiment for the chosen symbols (needs a configured provider; "
        "the default provider makes no network calls and writes nothing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan (dataset, symbols, fields, output, network need) and exit.",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Print the registry table of all dataset presets and exit.",
    )
    return parser


def _print_registry_table() -> None:
    """Print a table of every dataset preset in the registry."""
    headers = ["dataset", "source", "asset_class", "network", "fields", "description"]
    rows = []
    for name in data_sources.list_datasets():
        meta = data_sources.get_dataset(name)
        rows.append([
            name,
            str(meta["source"]),
            str(meta["asset_class"]),
            "yes" if meta["requires_network"] else "no",
            ", ".join(meta["fields"]),
            str(meta["description"]),
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _print_plan(meta: dict[str, Any], symbols: list[str], args: argparse.Namespace, output: Path) -> None:
    """Print the download plan for a dataset."""
    print(f"Dataset: {meta['name']}")
    print(f"Description: {meta['description']}")
    print(
        f"Asset class: {meta['asset_class']} | Source: {meta['source']} | "
        f"Requires network: {'yes' if meta['requires_network'] else 'no'}"
    )
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"Fields: {', '.join(meta['fields'])}")
    print(f"Range: {args.start} to {args.end or 'today'} | Interval: {args.interval}")
    print(f"Output directory: {output}")


def main(
    argv: list[str] | None = None,
    *,
    ohlcv_fetcher: Any = None,
    fundamentals_fetcher: Any = None,
    sentiment_fetcher: Any = None,
) -> int:
    """Run the downloader CLI. Returns a process exit code.

    The ``*_fetcher`` keyword arguments mirror the injectable fetchers of
    :mod:`signal_discovery_workflow.data_sources`, enabling fully offline
    end-to-end runs (tests pass fake fetchers here; ``--dry-run`` never
    fetches anything).
    """
    args = build_parser().parse_args(argv)

    if args.all_presets:
        _print_registry_table()
        return 0

    try:
        meta = data_sources.get_dataset(args.dataset)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 2

    raw_symbols = (
        [part.strip() for part in args.symbols.split(",")] if args.symbols else list(meta["default_symbols"])
    )
    symbols: list[str] = []
    for raw in raw_symbols:
        if not raw:
            continue
        try:
            symbols.append(data_sources.normalize_symbol(raw, meta["name"]))
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
    if not symbols:
        print("ERROR: no symbols to download. Pass --symbols or pick a preset with default_symbols.")
        return 2

    output = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / meta["name"]
    _print_plan(meta, symbols, args, output)

    if args.dry_run:
        print("Dry run: no data was downloaded.")
        return 0

    if meta["source"] == "binance":
        print(
            "This dataset is provided by download_binance_futures.py (Binance USDT-M futures, "
            "network required). Run it directly, e.g.:\n"
            f"  python -m signal_discovery_workflow.download_binance_futures --output {output}"
        )
        print("Dry run: no data was downloaded.")
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat()
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    fields = set(meta["fields"])

    if fields & set(data_sources.OHLCV_FIELDS):
        frames = fetch_ohlcv(
            symbols, args.start, end, interval=args.interval, fetcher=ohlcv_fetcher
        )
        write_dataset_to_csv(frames, output)
    if args.include_fundamentals or meta["name"] == "fundamentals":
        fundamentals = fetch_fundamentals(symbols, fetcher=fundamentals_fetcher)
        write_dataset_to_csv(fundamentals, output)
    if args.include_sentiment or meta["name"] == "sentiment":
        sentiment = fetch_sentiment(symbols, fetcher=sentiment_fetcher)
        write_dataset_to_csv(sentiment, output)

    written = sorted(path.name for path in output.glob("*.csv"))
    if written:
        print(f"Wrote {len(written)} field file(s) to {output}: {', '.join(written)}")
    else:
        print("No data files were written (all fetched frames were empty).")
    manifest_path = write_dataset_manifest(output, meta["name"], symbols, meta["source"], fetched_at)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
