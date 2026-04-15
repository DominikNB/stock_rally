"""stock_rally_v10 — Daten laden (Pipeline-Modul)."""
from __future__ import annotations

import time

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
    frames = []
    n_t = len(tickers)
    bsz = int(getattr(cfg, "YF_DOWNLOAD_BATCH_SIZE", 20) or 20)
    bsz = max(1, bsz)
    sleep_s = float(getattr(cfg, "YF_DOWNLOAD_BATCH_SLEEP_SEC", 1.0) or 0.0)
    print(
        f'Downloading {n_t} tickers from {start} to {end} … '
        f'(Batch={bsz}, Sleep={sleep_s:.1f}s, yfinance threads=False)',
        flush=True,
    )
    ti = 0
    for bi in range(0, n_t, bsz):
        batch = tickers[bi : bi + bsz]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            threads=False,   # CRITICAL: threads=True corrupts multi-ticker data
            progress=n_t > 3 and bi == 0,
            group_by="ticker",
        )
        for ticker in batch:
            ti += 1
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    # yfinance liefert je nach group_by entweder (Field, Ticker) oder (Ticker, Field)
                    cols = raw.columns
                    if (ticker, 'Close') in cols and (ticker, 'Volume') in cols:
                        close = raw[(ticker, 'Close')].dropna()
                        volume = raw[(ticker, 'Volume')].reindex(close.index).fillna(0)
                    elif ('Close', ticker) in cols and ('Volume', ticker) in cols:
                        close = raw[('Close', ticker)].dropna()
                        volume = raw[('Volume', ticker)].reindex(close.index).fillna(0)
                    else:
                        raise KeyError(f"Close/Volume-Spalten fehlen für {ticker}")
                else:
                    close = raw['Close'].dropna()
                    volume = raw['Volume'].reindex(close.index).fillna(0)

                if len(close) < 100:
                    print(f'  Skipping {ticker}: only {len(close)} rows', flush=True)
                    continue

                df = pd.DataFrame({'close': close, 'volume': volume})
                df.index = _strip_tz(df.index)
                df = df.reset_index().rename(columns={'index': 'Date', 'Price': 'Date'})
                if 'Date' not in df.columns:
                    df = df.rename(columns={df.columns[0]: 'Date'})
                df['Date'] = _strip_tz(df['Date'])
                df['ticker'] = ticker
                df['company'] = cfg.COMPANY_NAMES.get(ticker, ticker)
                frames.append(df)
            except Exception as e:
                print(f'  Error {ticker}: {e}', flush=True)
            if ti == 1 or ti == n_t or ti % 25 == 0:
                print(f'  … {ti}/{n_t} Ticker nach Download verarbeitet', flush=True)
        if sleep_s > 0 and bi + bsz < n_t:
            time.sleep(sleep_s)

    if not frames:
        raise ValueError("Kein Ticker lieferte ausreichend Kursdaten (>=100 Zeilen).")
    result = pd.concat(frames, ignore_index=True)
    result['Date'] = pd.to_datetime(result['Date'])
    print(
        f'Loaded {result["ticker"].nunique()} tickers, {len(result):,} rows.',
        flush=True,
    )
    return result