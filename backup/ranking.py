#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def _try_import_yf():
    try:
        import yfinance as yf
        return yf
    except Exception:
        return None

def parse_args():
    p = argparse.ArgumentParser(description="주식 차트 변화 영상 생성기 (normalized compare)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", type=str, help="야후 파이낸스 티커(쉼표 여러 개 가능)")
    src.add_argument("--csv", type=str, help="로컬 CSV (Date,Open,High,Low,Close,Volume)")

    p.add_argument("--start", type=str, help="시작일 (YYYY-MM-DD)")
    p.add_argument("--end", type=str, help="종료일 (YYYY-MM-DD)")
    p.add_argument("--period", type=str, default="1y")
    p.add_argument("--interval", type=str, default="1d")

    p.add_argument("--kind", type=str, choices=["line", "candle", "compare"], default="candle")
    p.add_argument("--ma", type=str, default="")
    p.add_argument("--show-volume", action="store_true")

    p.add_argument("--title", type=str, default="")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--trailing", type=int, default=0)
    p.add_argument("--out", type=str, default="stock.mp4")

    p.add_argument("--preview", type=str, default="")
    p.add_argument("--log", action="store_true")

    return p.parse_args()

def load_data_from_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    needed = {"Date", "Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV에 필수 컬럼 누락. 필요: {needed}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df

def load_single_ticker_yf(ticker, start, end, period, interval):
    yf = _try_import_yf()
    if yf is None:
        raise ImportError("yfinance 미설치. pip install yfinance 후 다시 시도하거나 --csv 를 사용하세요.")
    if start or end:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"{ticker} 데이터가 비어있습니다.")
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def load_multi_tickers_yf(tickers, start, end, period, interval):
    series = []
    for t in tickers:
        d = load_single_ticker_yf(t, start, end, period, interval)
        s = d["Close"].astype(float); s.name = t
        series.append(s)
    if not series:
        raise ValueError("다운로드된 티커 데이터가 없습니다.")
    df_close = pd.concat(series, axis=1).sort_index()
    df_close = df_close.dropna(axis=1, how="all")
    if df_close.shape[1] == 0:
        raise ValueError("모든 티커가 유효한 Close 데이터를 제공하지 않았습니다.")
    return df_close

def add_moving_averages(df, ma_str):
    if not ma_str:
        return df, []
    ma_list = []
    for tok in ma_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
            if n >= 2:
                ma_list.append(n)
        except ValueError:
            pass
    for n in ma_list:
        if "Close" in df.columns:
            df[f"MA{n}"] = df["Close"].rolling(n).mean()
    return df, ma_list

def compute_limits_price(df_like):
    lows = df_like["Low"] if "Low" in df_like.columns else df_like["Close"]
    highs = df_like["High"] if "High" in df_like.columns else df_like["Close"]
    vals_min = np.nanmin(np.asarray(lows.values, dtype=float))
    vals_max = np.nanmax(np.asarray(highs.values, dtype=float))
    pad = (vals_max - vals_min) * 0.05 if vals_max > vals_min else 1.0
    return vals_min - pad, vals_max + pad

def compute_limits_from_norm(df_norm):
    vals = np.asarray(df_norm.values, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return 99.0, 101.0
    vals_min = np.nanmin(vals[mask])
    vals_max = np.nanmax(vals[mask])
    pad = (vals_max - vals_min) * 0.08 if vals_max > vals_min else 1.0
    return vals_min - pad, vals_max + pad

def render_candle_manual(ax_price, ax_vol, df_part, ma_list):
    xs = mdates.date2num(df_part.index.to_pydatetime())
    width = (xs[1] - xs[0]) * 0.6 if len(xs) >= 2 else 0.6
    for x, (o, h, l, c) in zip(xs, df_part[["Open","High","Low","Close"]].values):
        if np.isnan([o,h,l,c]).any():
            continue
        ax_price.vlines(x, l, h, linewidth=1.0, color='white', alpha=0.8)
        lower = min(o, c); height = abs(c - o)
        if c >= o:
            # Green for up candles
            face = (0.0, 0.8, 0.0, 0.9); edge = (0.0, 1.0, 0.0, 1.0)
        else:
            # Red for down candles
            face = (0.8, 0.0, 0.0, 0.9); edge = (1.0, 0.0, 0.0, 1.0)
        if height == 0: height = (h - l) * 0.01
        rect = Rectangle((x - width/2, lower), width, height, facecolor=face, edgecolor=edge, linewidth=0.7)
        ax_price.add_patch(rect)
    for n in ma_list:
        col = f"MA{n}"
        if col in df_part.columns:
            ax_price.plot(df_part.index, df_part[col], linewidth=2.0, label=col, color='yellow', alpha=0.8)
    if ma_list:
        ax_price.legend(loc="upper left", fontsize=10, frameon=False, labelcolor='white')
    if ("Volume" in df_part.columns) and (ax_vol is not None):
        ax_vol.bar(df_part.index, df_part["Volume"], width=0.8, align="center", color='gray', alpha=0.6)
        ax_vol.set_ylabel("Volume", fontsize=9, color='white')

def render_line(ax_price, ax_vol, df_part, ma_list, show_volume=False):
    """Render line chart with dark theme"""
    # Main price line
    ax_price.plot(df_part.index, df_part["Close"], linewidth=2.5, color='yellow', alpha=0.9, label='Close')
    
    # Moving averages
    for n in ma_list:
        col = f"MA{n}"
        if col in df_part.columns:
            ax_price.plot(df_part.index, df_part[col], linewidth=2.0, label=col, alpha=0.8)
    
    if ma_list:
        ax_price.legend(loc="upper left", fontsize=10, frameon=False, labelcolor='white')
    
    # Volume
    if show_volume and ("Volume" in df_part.columns) and (ax_vol is not None):
        ax_vol.bar(df_part.index, df_part["Volume"], width=0.8, align="center", color='gray', alpha=0.6)
        ax_vol.set_ylabel("Volume", fontsize=9, color='white')

def render_compare_from_norm(ax_price, df_norm_part):
    # baseline with white color
    ax_price.plot(df_norm_part.index, np.full(len(df_norm_part.index), 100.0), 
                 linewidth=0.8, alpha=0.3, color='white')
    plotted = False
    colors = ['yellow', 'lime', 'cyan', 'magenta', 'orange']  # Neon colors
    for i, col in enumerate(df_norm_part.columns):
        s = df_norm_part[col].astype(float)
        n_valid = s.notna().sum()
        if n_valid == 0:
            continue
        color = colors[i % len(colors)]
        if n_valid == 1:
            last_idx = s.last_valid_index()
            ax_price.scatter([last_idx], [s.loc[last_idx]], s=15, label=col, 
                           color=color, edgecolors='white', linewidth=1)
            plotted = True
        else:
            ax_price.plot(s.index, s, linewidth=2.5, label=col, color=color, alpha=0.9)
            last_idx = s.last_valid_index()
            if last_idx is not None:
                ax_price.scatter([last_idx], [s.loc[last_idx]], s=12, 
                               color=color, edgecolors='white', linewidth=1)
            plotted = True
    if plotted:
        ax_price.legend(loc="upper left", fontsize=11, frameon=False, labelcolor='white')
    ax_price.set_ylabel("Index = 100", fontsize=11, color='white')

def main():
    args = parse_args()

    tickers = []
    label = None
    if args.ticker:
        tickers = [t.strip() for t in args.ticker.split(",") if t.strip()]
        label = ",".join(tickers) if tickers else None

    multi = len(tickers) >= 2

    if args.csv:
        if multi:
            raise ValueError("--csv 모드에서는 다중 티커 비교를 지원하지 않습니다.")
        df = load_data_from_csv(args.csv)
        label = os.path.basename(args.csv)
        mode = args.kind
        if args.log:
            print("[CSV] rows:", len(df)); print(df.head()); print(df.tail()); print(df.info())
    else:
        if multi:
            # 원시 Close
            df_close = load_multi_tickers_yf(tickers, args.start, args.end, args.period, args.interval)
            if args.log:
                print("[TICKERS]", tickers, "rows:", len(df_close))
                print(df_close.head()); print(df_close.tail()); print(df_close.info())
            # 정규화 테이블 생성
            cols = []
            for col in df_close.columns:
                s = df_close[col].astype(float)
                base_idx = s.first_valid_index()
                if base_idx is None:
                    continue
                base = s.loc[base_idx]
                if not np.isfinite(base) or base == 0:
                    continue
                scaled = (s / base) * 100.0
                scaled.name = col
                cols.append(scaled)
            if not cols:
                raise ValueError("정규화할 유효 데이터가 없습니다.")
            df_norm = pd.concat(cols, axis=1).sort_index()
            base_index = df_norm.index
            mode = "compare"
        else:
            df = load_single_ticker_yf(tickers[0], args.start, args.end, args.period, args.interval) if tickers else None
            mode = args.kind

    # 타이틀
    if args.title.strip():
        title_text = args.title.strip()
    else:
        if args.csv:
            first = df.index[0] if len(df) else ""
            last  = df.index[-1] if len(df) else ""
            title_text = f"{label}  [{first:%Y-%m-%d} ~ {last:%Y-%m-%d}]"
        elif multi:
            first = base_index[0] if len(base_index) else ""
            last  = base_index[-1] if len(base_index) else ""
            title_text = f"{label}  [{first:%Y-%m-%d} ~ {last:%Y-%m-%d}] (Compare)"
        else:
            first = df.index[0] if len(df) else ""
            last  = df.index[-1] if len(df) else ""
            title_text = f"{label}  [{first:%Y-%m-%d} ~ {last:%Y-%m-%d}]"

    # 이동평균(단일)
    if not multi and mode in ("line","candle"):
        df, ma_list = add_moving_averages(df.copy(), args.ma)
    else:
        ma_list = []

    # Figure & Axes - Dark Theme
    fig_h_inches = args.height / args.dpi
    fig_w_inches = args.width / args.dpi
    fig = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=args.dpi, facecolor='black')

    if (mode == "candle") or (mode == "line" and args.show_volume and not multi):
        grid = fig.add_gridspec(5, 1, hspace=0.0)
        ax_price = fig.add_subplot(grid[:4, 0], facecolor='black')
        ax_vol   = fig.add_subplot(grid[4, 0], sharex=ax_price, facecolor='black')
    else:
        ax_price = fig.add_subplot(1,1,1, facecolor='black')
        ax_vol   = None

    # Dark theme styling for axes
    ax_price.set_facecolor('black')
    ax_price.tick_params(colors='white', labelsize=10)
    ax_price.spines['bottom'].set_color('white')
    ax_price.spines['top'].set_color('white')
    ax_price.spines['right'].set_color('white')
    ax_price.spines['left'].set_color('white')
    ax_price.xaxis.label.set_color('white')
    ax_price.yaxis.label.set_color('white')
    
    # Title with neon effect
    ax_price.set_title(title_text, fontsize=16, pad=20, color='white', 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='red', alpha=0.8))
    
    # Grid with subtle styling
    ax_price.grid(True, alpha=0.2, linewidth=0.5, color='gray')
    if ax_vol is not None:
        ax_vol.set_facecolor('black')
        ax_vol.tick_params(colors='white', labelsize=9)
        ax_vol.spines['bottom'].set_color('white')
        ax_vol.spines['top'].set_color('white')
        ax_vol.spines['right'].set_color('white')
        ax_vol.spines['left'].set_color('white')
        ax_vol.grid(True, alpha=0.15, linewidth=0.4, color='gray')

    # y-limit
    if multi:
        y_min, y_max = compute_limits_from_norm(df_norm)
    else:
        y_min, y_max = compute_limits_price(df)
    ax_price.set_ylim(y_min, y_max)

    # frames
    step = max(1, int(args.step))
    if multi:
        N = len(base_index)
    else:
        N = len(df)
    if N < 1:
        raise ValueError("그릴 데이터가 없습니다.")
    frames = list(range(1, N+1, step))
    if frames[-1] != N:
        frames.append(N)

    metadata = dict(artist="stock_video.py", comment="Generated by matplotlib + ffmpeg")
    writer = FFMpegWriter(fps=args.fps, metadata=metadata, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

    trailing = max(0, int(args.trailing))

    with writer.saving(fig, args.out, dpi=args.dpi):
        for idx in frames:
            ax_price.cla()
            if ax_vol is not None:
                ax_vol.cla()

            # Reapply dark theme styling in animation loop
            ax_price.set_facecolor('black')
            ax_price.tick_params(colors='white', labelsize=10)
            ax_price.spines['bottom'].set_color('white')
            ax_price.spines['top'].set_color('white')
            ax_price.spines['right'].set_color('white')
            ax_price.spines['left'].set_color('white')
            ax_price.xaxis.label.set_color('white')
            ax_price.yaxis.label.set_color('white')
            
            ax_price.set_title(title_text, fontsize=16, pad=20, color='white', 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='red', alpha=0.8))
            ax_price.grid(True, alpha=0.2, linewidth=0.5, color='gray')
            if ax_vol is not None:
                ax_vol.set_facecolor('black')
                ax_vol.tick_params(colors='white', labelsize=9)
                ax_vol.spines['bottom'].set_color('white')
                ax_vol.spines['top'].set_color('white')
                ax_vol.spines['right'].set_color('white')
                ax_vol.spines['left'].set_color('white')
                ax_vol.set_ylabel("Volume", fontsize=9, color='white')
                ax_vol.grid(True, alpha=0.15, linewidth=0.4, color='gray')

            ax_price.set_ylim(y_min, y_max)

            if multi:
                part_index = base_index[max(0, idx - trailing):idx] if trailing > 0 else base_index[:idx]
                df_norm_part = df_norm.loc[part_index]
                if len(df_norm_part) >= 2:
                    ax_price.set_xlim(df_norm_part.index[0], df_norm_part.index[-1])
                render_compare_from_norm(ax_price, df_norm_part)

                curr_time = df_norm_part.index[-1]
                dt_txt = getattr(curr_time, "strftime", lambda *_: str(curr_time))("%Y-%m-%d %H:%M")
                ax_price.text(0.99, 0.98, dt_txt, ha="right", va="top", transform=ax_price.transAxes, 
                            fontsize=12, color='white', alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', edgecolor='yellow', alpha=0.7))

            else:
                df_part = df.iloc[max(0, idx - trailing):idx] if trailing > 0 else df.iloc[:idx]
                if len(df_part.index) >= 2:
                    ax_price.set_xlim(df_part.index[0], df_part.index[-1])

                if mode == "line":
                    render_line(ax_price, ax_vol, df_part, ma_list, show_volume=args.show_volume)
                else:
                    render_candle_manual(ax_price, ax_vol, df_part, ma_list)

                curr_time = df_part.index[-1]
                dt_txt = getattr(curr_time, "strftime", lambda *_: str(curr_time))("%Y-%m-%d %H:%M")
                ax_price.text(0.99, 0.98, dt_txt, ha="right", va="top", transform=ax_price.transAxes, 
                            fontsize=12, color='white', alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', edgecolor='yellow', alpha=0.7))

            writer.grab_frame()

        if args.preview.strip():
            fig.savefig(args.preview.strip(), dpi=args.dpi, bbox_inches="tight")

    print(f"[완료] 저장됨: {args.out}")
    if args.preview.strip():
        print(f"[미리보기 저장]: {args.preview.strip()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("중단됨.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(2)
