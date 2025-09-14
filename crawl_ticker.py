#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch daily OHLC and also emit separate rows for Dividends and Stock Splits.
- Defaults to QYLD when no ticker is provided (interactive prompt default or CLI default).
- Uses yfinance (installs automatically if not present).
- If no start/end is provided, downloads FULL history via period='max'.
- Supports optional start/end date, adjusted prices, and output file path.

Output CSV columns:
  Date, Type, Open, High, Low, Close, Dividend, SplitRatio, SplitText
Where:
  - Type ∈ {"PRICE","DIVIDEND","SPLIT"}
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
    Convert numeric split factor to a human-readable text like '2:1' or '1:5'.
    yfinance 'Stock Splits' is typically the post/pre factor: e.g., 2.0 for 2:1 split, 0.5 for 1:2 reverse split.
    """
    if ratio is None or ratio == 0:
        return ""
    if ratio >= 1:
        return f"{int(round(ratio))}:1"
    else:
        inv = int(round(1.0 / ratio))
        return f"1:{inv}"

def fetch_with_events(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    adjusted: bool = False,
    retries: int = 3,
    retry_wait: float = 1.5,
) -> pd.DataFrame:
    """
    Download daily OHLC plus actions; expand dividends and splits to separate event rows.
    - If start/end are None, falls back to period='max' to fetch full history.
    """
    ticker = ticker.strip().upper()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # Use full history when start/end unspecified
            use_period = (start is None and end is None)

            df = yf.download(
                tickers=ticker,
                start=None if use_period else start,
                end=None if use_period else end,
                period="max" if use_period else None,  # <-- full history
                auto_adjust=adjusted,
                actions=True,
                progress=False,
                interval="1d",
                group_by="ticker",
                threads=True,
            )

            # Flatten in case of MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[0]:
                    df = df[ticker]
                else:
                    df = df.droplevel(0, axis=1)

            if df.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            # Ensure expected columns
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column '{col}' in downloaded data. Columns: {list(df.columns)}")

            # PRICE rows
            price = df[["Open", "High", "Low", "Close"]].copy()
            price["Type"] = "PRICE"
            price["Dividend"] = np.nan
            price["SplitRatio"] = np.nan
            price["SplitText"] = ""
            price.index.name = "Date"
            price.reset_index(inplace=True)

            # DIVIDEND rows
            if "Dividends" in df.columns:
                div = df[df["Dividends"].fillna(0) != 0]["Dividends"].copy()
                if not div.empty:
                    div_df = div.to_frame().reset_index().rename(columns={"Dividends": "Dividend"})
                    div_df["Type"] = "DIVIDEND"
                    for col in ["Open", "High", "Low", "Close"]:
                        div_df[col] = np.nan
                    div_df["SplitRatio"] = np.nan
                    div_df["SplitText"] = ""
                else:
                    div_df = None
            else:
                div_df = None

            # SPLIT rows
            if "Stock Splits" in df.columns:
                sp = df[df["Stock Splits"].fillna(0) != 0]["Stock Splits"].copy()
                if not sp.empty:
                    sp_df = sp.to_frame().reset_index().rename(columns={"Stock Splits": "SplitRatio"})
                    sp_df["Type"] = "SPLIT"
                    sp_df["SplitText"] = sp_df["SplitRatio"].apply(split_text_from_ratio)
                    for col in ["Open", "High", "Low", "Close"]:
                        sp_df[col] = np.nan
                    sp_df["Dividend"] = np.nan
                else:
                    sp_df = None
            else:
                sp_df = None

            # Concatenate without empty frames (prevents FutureWarning)
            frames = [price]
            if div_df is not None:
                frames.append(div_df)
            if sp_df is not None:
                frames.append(sp_df)

            out = pd.concat(frames, ignore_index=True)
            out = out[["Date","Type","Open","High","Low","Close","Dividend","SplitRatio","SplitText"]]
            out.sort_values(["Date","Type"], inplace=True)
            out.reset_index(drop=True, inplace=True)
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
    parser = argparse.ArgumentParser(description="Fetch daily OHLC with separate rows for dividends and splits (default ticker: QYLD).")
    parser.add_argument("-t", "--ticker", type=str, default=None, help="Ticker symbol (e.g., QYLD). If omitted, you'll be prompted (default=QYLD).")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--adjusted", action="store_true", help="Use adjusted prices (auto-adjust for dividends/splits).")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="Output CSV path. Default: <TICKER>_daily_with_events.csv")

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
        df = fetch_with_events(
            ticker=ticker,
            start=args.start,
            end=args.end,
            adjusted=args.adjusted,
        )
    except Exception as e:
        print(f"[error] Failed to fetch data for '{ticker}': {e}", file=sys.stderr)
        sys.exit(1)

    outfile = args.outfile or f"{ticker.upper()}_daily_with_events.csv"
    try:
        df.to_csv(outfile, index=False, encoding="utf-8")
        print(f"[ok] Saved {len(df):,} rows to '{outfile}'.")
    except Exception as e:
        print(f"[error] Failed to save CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
