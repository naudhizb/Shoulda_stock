#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily OHLC + inline event columns (no Type column).
- One row per trading day.
- Columns: Date, Open, High, Low, Close, Dividend, SplitRatio, SplitText
- On non-event days, Dividend / SplitRatio / SplitText are blank.
- Defaults to QYLD if no ticker provided.
- If no start/end given, fetch FULL history via period='max'.

Usage:
  python crawl_stock_ohlc_events_wide.py
  python crawl_stock_ohlc_events_wide.py -t AAPL --start 2015-01-01 --end 2025-09-06 --adjusted -o aapl_wide.csv
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
import numpy as np

def split_text_from_ratio(ratio: float) -> str:
    """
    Convert numeric split factor to human-readable '2:1' or '1:5'.
    yfinance 'Stock Splits' is post/pre factor: 2.0 for 2:1 split, 0.5 for 1:2 reverse split.
    """
    if ratio is None or ratio == 0:
        return ""
    if ratio >= 1:
        return f"{int(round(ratio))}:1"
    else:
        inv = int(round(1.0 / ratio))
        return f"1:{inv}"

def fetch_events_wide(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    adjusted: bool = False,
    retries: int = 3,
    retry_wait: float = 1.5,
) -> pd.DataFrame:
    """
    Download daily OHLC and inline action columns: Dividend, SplitRatio, SplitText.
    """
    ticker = ticker.strip().upper()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            use_period = (start is None and end is None)
            df = yf.download(
                tickers=ticker,
                start=None if use_period else start,
                end=None if use_period else end,
                period="max" if use_period else None,   # full history when no dates
                auto_adjust=adjusted,
                actions=True,
                progress=False,
                interval="1d",
                group_by="ticker",
                threads=True,
            )
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[0]:
                    df = df[ticker]
                else:
                    df = df.droplevel(0, axis=1)

            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            # Validate columns
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column '{col}'. Columns: {list(df.columns)}")

            # Base price dataframe
            out = df[["Open", "High", "Low", "Close"]].copy()

            # Dividend column: only fill on event dates, blank elsewhere
            if "Dividends" in df.columns:
                div = df["Dividends"].copy().replace(0, np.nan)
                out["Dividend"] = div
            else:
                out["Dividend"] = np.nan

            # Split columns
            if "Stock Splits" in df.columns:
                sp = df["Stock Splits"].copy().replace(0, np.nan)
                out["SplitRatio"] = sp
                out["SplitText"] = sp.apply(lambda v: split_text_from_ratio(v) if pd.notna(v) else "")
            else:
                out["SplitRatio"] = np.nan
                out["SplitText"] = ""

            # Finalize
            out.index = pd.to_datetime(out.index)
            out.index.name = "Date"
            out.reset_index(inplace=True)
            out.sort_values("Date", inplace=True)

            return out[["Date", "Open", "High", "Low", "Close", "Dividend", "SplitRatio", "SplitText"]]
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = retry_wait * attempt
                print(f"[warn] Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

def main():
    parser = argparse.ArgumentParser(description="Fetch daily OHLC with inline Dividend/Split columns (no Type column).")
    parser.add_argument("-t", "--ticker", type=str, default=None, help="Ticker symbol (e.g., QYLD). If omitted, you'll be prompted (default=QYLD).")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--adjusted", action="store_true", help="Use adjusted prices (auto-adjust for dividends/splits).")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="Output CSV path. Default: <TICKER>_daily_wide.csv")

    args = parser.parse_args()

    # Interactive ticker default
    if not args.ticker:
        try:
            user_in = input("티커를 입력하세요 (기본=QYLD): ").strip()
        except EOFError:
            user_in = ""
        ticker = user_in or "QYLD"
    else:
        ticker = args.ticker

    # Validate dates
    for label, val in [("start", args.start), ("end", args.end)]:
        if val:
            try:
                datetime.strptime(val, "%Y-%m-%d")
            except ValueError:
                print(f"[error] Invalid {label} date format: '{val}'. Use YYYY-MM-DD.", file=sys.stderr)
                sys.exit(2)

    try:
        df = fetch_events_wide(
            ticker=ticker,
            start=args.start,
            end=args.end,
            adjusted=args.adjusted,
        )
    except Exception as e:
        print(f"[error] Failed to fetch data for '{ticker}': {e}", file=sys.stderr)
        sys.exit(1)

    outfile = args.outfile or f"{ticker.upper()}_daily_wide.csv"
    try:
        df.to_csv(outfile, index=False, encoding="utf-8")
        print(f"[ok] Saved {len(df):,} rows to '{outfile}'.")
    except Exception as e:
        print(f"[error] Failed to save CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
