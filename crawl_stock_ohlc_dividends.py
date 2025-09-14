#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch daily OHLC (Open, High, Low, Close) and Dividends for a given ticker and save as CSV.
- Defaults to QYLD when no ticker is provided (interactive prompt default or CLI default).
- Uses yfinance (installs automatically if not present).
- Supports optional start/end date, adjusted prices, and output file path.
Usage examples:
  - Interactive (will prompt): python crawl_stock_ohlc_dividends.py
  - CLI with defaults:         python crawl_stock_ohlc_dividends.py --ticker QYLD
  - With dates:                python crawl_stock_ohlc_dividends.py -t AAPL --start 2015-01-01 --end 2025-09-06
  - Adjusted prices:           python crawl_stock_ohlc_dividends.py -t MSFT --adjusted
  - Custom output path:        python crawl_stock_ohlc_dividends.py -t SPY -o spy_daily.csv
"""
from __future__ import annotations

import argparse
import sys
import subprocess
import importlib
import time
from typing import Optional
from datetime import datetime

def ensure_package(pkg: str):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[info] '{pkg}' not found. Installing...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
        return importlib.import_module(pkg)

# Ensure dependencies
yf = ensure_package("yfinance")
import pandas as pd

def fetch_ohlc_dividends(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    adjusted: bool = False,
    retries: int = 3,
    retry_wait: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch daily OHLC and Dividends using yfinance.
    - ticker: e.g., "QYLD"
    - start/end: "YYYY-MM-DD" strings or None for full history
    - adjusted: whether to auto-adjust prices for dividends/splits
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Dividend
    """
    ticker = ticker.strip().upper()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=adjusted,
                actions=True,
                progress=False,
                interval="1d",
                group_by="ticker",
                threads=True,
            )
            # Handle MultiIndex columns when multiple tickers, though we only pass one
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten and pick the first level matching our ticker if present
                if ticker in df.columns.levels[0]:
                    df = df[ticker]
                else:
                    # Fallback: pick the first top-level
                    df = df.droplevel(0, axis=1)

            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")
            # Normalize columns and add Dividend
            expected_cols = ["Open", "High", "Low", "Close"]
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns {missing} in downloaded data. Columns: {list(df.columns)}")

            # Dividends column may be absent if none; create if needed
            if "Dividends" in df.columns:
                dividends = df["Dividends"].fillna(0)
            else:
                dividends = pd.Series(0, index=df.index, name="Dividends")

            out = df[expected_cols].copy()
            out["Dividend"] = dividends
            out.index = pd.to_datetime(out.index)
            out.index.name = "Date"
            out.reset_index(inplace=True)

            # Sort by date ascending
            out.sort_values("Date", inplace=True)
            return out
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = retry_wait * attempt
                print(f"[warn] Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

def main():
    parser = argparse.ArgumentParser(description="Fetch daily OHLC and dividends, save to CSV (default ticker: QYLD).")
    parser.add_argument("-t", "--ticker", type=str, default=None, help="Ticker symbol (e.g., QYLD). If omitted, you'll be prompted (default=QYLD).")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--adjusted", action="store_true", help="Use adjusted prices (auto-adjust for dividends/splits).")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="Output CSV path. Default: <TICKER>_daily.csv")

    args = parser.parse_args()

    # If ticker not provided via CLI, prompt interactively with default QYLD
    if not args.ticker:
        try:
            user_in = input("티커를 입력하세요 (기본=QYLD): ").strip()
        except EOFError:
            user_in = ""
        ticker = user_in or "QYLD"
    else:
        ticker = args.ticker

    # Basic validation for dates
    for label, val in [("start", args.start), ("end", args.end)]:
        if val:
            try:
                datetime.strptime(val, "%Y-%m-%d")
            except ValueError:
                print(f"[error] Invalid {label} date format: '{val}'. Use YYYY-MM-DD.", file=sys.stderr)
                sys.exit(2)

    try:
        df = fetch_ohlc_dividends(
            ticker=ticker,
            start=args.start,
            end=args.end,
            adjusted=args.adjusted,
        )
    except Exception as e:
        print(f"[error] Failed to fetch data for '{ticker}': {e}", file=sys.stderr)
        sys.exit(1)

    outfile = args.outfile or f"{ticker.upper()}_daily.csv"
    try:
        df.to_csv(outfile, index=False, encoding="utf-8")
        print(f"[ok] Saved {len(df):,} rows to '{outfile}'.")
    except Exception as e:
        print(f"[error] Failed to save CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
