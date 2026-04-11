"""stock_rally_v10 — Daten laden (Pipeline-Modul)."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.helpers import _strip_tz


def load_stock_data(tickers=None, start=None, end=None):
    """
    Single bulk yfinance download with threads=False to avoid data corruption bug.
    Returns a DataFrame with columns [Date, close, volume, ticker, company].
    """
    if tickers is None:
        tickers = cfg.ALL_TICKERS
    if start is None:
        start = cfg.START_DATE
    if end is None:
        end = cfg.END_DATE
    print(
        f'Downloading {len(tickers)} tickers from {start} to {end} … '
        f'(yfinance Bulk-Download kann einige Minuten dauern)',
        flush=True,
    )
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        threads=False,   # CRITICAL: threads=True corrupts multi-ticker data
        progress=len(tickers) > 3,
    )

    frames = []
    n_t = len(tickers)
    for ti, ticker in enumerate(tickers, start=1):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close  = raw['Close'][ticker].dropna()
                volume = raw['Volume'][ticker].reindex(close.index).fillna(0)
            else:
                close  = raw['Close'].dropna()
                volume = raw['Volume'].reindex(close.index).fillna(0)

            if len(close) < 100:
                print(f'  Skipping {ticker}: only {len(close)} rows', flush=True)
                continue

            df = pd.DataFrame({'close': close, 'volume': volume})
            df.index = _strip_tz(df.index)
            df = df.reset_index().rename(columns={'index': 'Date', 'Price': 'Date'})
            if 'Date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'Date'})
            df['Date']    = _strip_tz(df['Date'])
            df['ticker']  = ticker
            df['company'] = cfg.COMPANY_NAMES.get(ticker, ticker)
            frames.append(df)
        except Exception as e:
            print(f'  Error {ticker}: {e}', flush=True)
        if ti == 1 or ti == n_t or ti % 25 == 0:
            print(f'  … {ti}/{n_t} Ticker nach Download verarbeitet', flush=True)

    result = pd.concat(frames, ignore_index=True)
    result['Date'] = pd.to_datetime(result['Date'])
    print(
        f'Loaded {result["ticker"].nunique()} tickers, {len(result):,} rows.',
        flush=True,
    )
    return result