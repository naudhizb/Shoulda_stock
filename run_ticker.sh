#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_tickers.sh <tickers.txt> <script.py> [extra args passed to script.py...]
if [ $# -lt 2 ]; then
  echo "Usage: $0 <tickers.txt> <script.py> [--args...]"
  echo "Example: $0 tickers.txt crawl_stock_ohlc_events_wide.py --adjusted --start 2015-01-01"
  exit 1
fi

LIST="$1"; shift
SCRIPT="$1"; shift

if [ ! -f "$LIST" ]; then
  echo "[error] ticker list not found: $LIST" >&2
  exit 2
fi
if [ ! -f "$SCRIPT" ]; then
  echo "[error] python script not found: $SCRIPT" >&2
  exit 2
fi

PY="${PYTHON:-python3}"

while IFS= read -r line || [ -n "$line" ]; do
  # strip comments (# ...) and whitespace
  t="${line%%#*}"
  t="$(echo "$t" | tr -d '[:space:]')"
  [ -z "$t" ] && continue

  echo "==> Running for ticker: $t"
  "$PY" "$SCRIPT" -t "$t" "$@" || {
    echo "[warn] script failed for ticker: $t" >&2
  }
done < "$LIST"

echo "[ok] batch finished."
