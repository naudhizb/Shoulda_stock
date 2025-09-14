@echo off
setlocal enabledelayedexpansion

REM Usage: run_tickers.bat <tickers.txt> <script.py> [extra args...]
if "%~2"=="" (
  echo Usage: %~nx0 tickers.txt script.py [extra args...]
  echo Example: %~nx0 tickers.txt crawl_stock_ohlc_events_wide.py --adjusted --start 2015-01-01
  exit /b 1
)

set "LIST=%~1"
set "SCRIPT=%~2"
shift
shift

if not exist "%LIST%" (
  echo [error] ticker list not found: %LIST% 1>&2
  exit /b 2
)
if not exist "%SCRIPT%" (
  echo [error] python script not found: %SCRIPT% 1>&2
  exit /b 2
)

set "PY=%PYTHON%"
if "%PY%"=="" set "PY=python"

for /f "usebackq tokens=* delims=" %%L in ("%LIST%") do (
  set "LINE=%%L"
  for /f "tokens=1 delims=#" %%A in ("!LINE!") do (
    set "T=%%A"
    set "T=!T: =!"
    if not "!T!"=="" (
      echo ==> Running for ticker: !T!
      "%PY%" "%SCRIPT%" -t "!T!" %*
      if errorlevel 1 echo [warn] script failed for !T! 1>&2
    )
  )
)

echo [ok] batch finished.
endlocal
