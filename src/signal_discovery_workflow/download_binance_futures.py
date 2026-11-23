"""Download Binance USDT-M futures OHLCV data for the signal discovery workflow.

Saves Open/High/Low/Close/Volume CSVs compatible with the workflow
(index = dates, columns = symbols).
"""
import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "binance_futures"
DATA_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
BASE = "https://fapi.binance.com"


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def top_symbols(n: int = 60) -> list[str]:
    info = _get(f"{BASE}/fapi/v1/exchangeInfo")
    syms = [s["symbol"] for s in info["symbols"]
            if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    # rank by 24h quote volume
    tickers = _get(f"{BASE}/fapi/v1/ticker/24hr")
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in tickers}
    syms.sort(key=lambda s: vol.get(s, 0), reverse=True)
    return syms[:n]


def fetch_klines(symbol: str, interval: str = "1d", limit: int = 1500) -> pd.DataFrame:
    url = f"{BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = _get(url)
    df = pd.DataFrame(data, columns=["open_time", "Open", "High", "Low", "Close", "Volume",
                                     "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
    df = df.set_index("date")
    for c in DATA_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[DATA_FIELDS]


def download_binance_futures(symbols_n: int = 60, interval: str = "1d",
                            output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    syms = top_symbols(symbols_n)
    logger.info(f"Downloading {len(syms)} Binance futures symbols...")
    frames = {f: {} for f in DATA_FIELDS}
    for s in syms:
        try:
            df = fetch_klines(s, interval=interval)
            for f in DATA_FIELDS:
                frames[f][s] = df[f]
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"{s}: {e}")
    results = {}
    for f in DATA_FIELDS:
        out = pd.DataFrame(frames[f])
        out.index = pd.to_datetime(out.index)
        # drop symbols with any missing values to keep a clean panel
        out = out.dropna(axis=1, how="any")
        out.to_csv(output_path / f"{f}.csv")
        results[f] = out
    logger.info(f"Saved {len(results[DATA_FIELDS[0]].columns)} complete symbols to {output_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=60)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    download_binance_futures(args.symbols, args.interval, args.output)


if __name__ == "__main__":
    main()
