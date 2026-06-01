"""News-Repair Validierung: kleines Zeitfenster, kein BQ-Scan, kein Optuna.

Pipeline-Schritte, die wir hier nachbauen:
  1. News-Cache laden (data/news_gdelt_cache.pkl, bereits vorhanden).
  2. Mini-df_base mit ~7 reprasentativen Tickern + Date-Range bauen.
  3. df["sector"] mit normalize_sector_key korrekt setzen (wie assemble_features das nun tut).
  4. _export_news_shards_for_grid aufrufen -> baut neue Shards in tmp-Ordner.
  5. Shard pro Ticker inspizieren: news_macro_* und news_sec_* Fill-Rate.

Erwartung nach Fix:
  - Tech-Ticker (AAPL, MSFT) und Finance-Ticker (JPM) sehen jetzt 50-90 % nonzero News.
  - Crypto-Ticker (BTC-USD) bleibt wie vorher.
  - news_macro_* Spalten sind vorhanden und gefuellt fuer ALLE Ticker.
"""
import pickle
import os
import sys
import shutil
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Kleines Zeitfenster + kein BQ-Refetch.
os.environ.setdefault("PYTHONUTF8", "1")

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10 import features as feat_mod

print(f"[Validate] normalize_sector_key verfuegbar: {hasattr(cfg, 'normalize_sector_key')}")
print(f"[Validate] Beispiel: 'Consumer Cyclical' -> {cfg.normalize_sector_key('Consumer Cyclical')!r}")
print(f"[Validate] Beispiel: 'Technology'        -> {cfg.normalize_sector_key('Technology')!r}")
print(f"[Validate] Beispiel: 'Utilities'         -> {cfg.normalize_sector_key('Utilities')!r}")

# 1. News-Cache laden.
cache_path = "data/news_gdelt_cache.pkl"
print(f"\n[Validate] Lade News-Cache: {cache_path}")
with open(cache_path, "rb") as f:
    payload = pickle.load(f)
news_df = payload["df"]
print(f"  Cache shape: {news_df.shape}")
print(f"  Channels:    {sorted(news_df['channel'].unique())}")

# 2. Mini-df_base bauen: 7 Ticker, je 250 Geschaeftstage in 2024.
test_tickers = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financial Services",
    "JNJ": "Healthcare",
    "XOM": "Energy",
    "BTC-USD": "crypto",
    "ETH-USD": "Cryptocurrency",  # bewusst nicht-normalisierte Schreibweise zum Testen
}
dates = pd.bdate_range("2024-01-02", "2024-12-31")
rows = []
for t, yahoo_sector in test_tickers.items():
    for d in dates:
        rows.append({"Date": d, "ticker": t, "gics_sector": yahoo_sector})
df_base = pd.DataFrame(rows)
print(f"\n[Validate] df_base shape: {df_base.shape}, Ticker: {sorted(df_base['ticker'].unique())}")

# 3. Sektor analog zu assemble_features setzen (vor dem Fix: roh; nach dem Fix: normalisiert).
df_base["sector_raw"] = df_base["gics_sector"].astype(str)
df_base["sector"] = df_base["sector_raw"].map(cfg.normalize_sector_key)
print("\n[Validate] Sector-Mapping (nach Normalisierung):")
print(df_base[["ticker", "sector_raw", "sector"]].drop_duplicates().to_string(index=False))

# 4. tmp Shard-Verzeichnis.
tmp_shard_dir = "data/_tmp_news_validate"
shutil.rmtree(tmp_shard_dir, ignore_errors=True)
os.makedirs(tmp_shard_dir, exist_ok=True)
cfg.FEATURE_SHARD_DIR = os.path.abspath(tmp_shard_dir)

# kein Reuse erzwingen.
cfg.NEWS_SHARDS_REUSE_SAME_CALENDAR_DAY = False

# News-Cache an feat_mod weitergeben.
news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.normalize()

# Klein-Modus: nur EIN feat_tag pro Lauf, damit Test schnell ist.
# _news_export_tags_for_mode liest cfg.NEWS_*-Settings. Wir setzen ein klar definiertes Triple.
cfg._FEATURE_NEWS_SHARDS_ACTIVE = False
print(f"\n[Validate] Starte _export_news_shards_for_grid (Shard-Dir: {tmp_shard_dir!r}) ...")
feat_mod._export_news_shards_for_grid(df_base, news_df)

# 5. Shard inspizieren.
manifest = getattr(cfg, "NEWS_SHARD_MANIFEST", {}) or {}
print(f"\n[Validate] Manifest tags: {list(manifest.keys())}")
if not manifest:
    print("FEHLER: kein Manifest erzeugt — siehe Logs oben")
    sys.exit(1)

# Greife auf ersten Shard zu.
first_tag = next(iter(manifest))
first_path = manifest[first_tag]
print(f"[Validate] Lese Shard {first_tag!r}: {first_path}")
shard = pd.read_parquet(first_path)
all_news = [c for c in shard.columns if str(c).startswith("news_")]
macro_cols = [c for c in shard.columns if str(c).startswith("news_macro_")]
sec_cols = [c for c in shard.columns if str(c).startswith("news_sec_")]
print(f"  total news_*:     {len(all_news)}")
print(f"  news_macro_*:     {len(macro_cols)}")
print(f"  news_sec_*:       {len(sec_cols)}")

print("\n[Validate] Fill-Rate (nonzero share) je Ticker auf REPRAESENTATIVEN news_* Spalten:")
probe_cols = []
for prefix in ("news_macro_", "news_sec_"):
    for c in shard.columns:
        if str(c).startswith(prefix) and ("_tone" in c or "_vol" in c):
            probe_cols.append(c)
            if len([x for x in probe_cols if str(x).startswith(prefix)]) >= 2:
                break
print(f"  Probe Spalten: {probe_cols}")
print()
print(f"  {'ticker':<10} {'sector':<25}  " + "  ".join(f"{c[:30]:<32}" for c in probe_cols))
for t in sorted(df_base["ticker"].unique()):
    sub = shard[shard["ticker"] == t]
    sec_t = sub["sector"].iloc[0] if len(sub) and "sector" in sub.columns else ""
    cells = []
    for c in probe_cols:
        if c not in sub.columns:
            cells.append("MISSING".ljust(32))
            continue
        v = pd.to_numeric(sub[c], errors="coerce")
        # Sentinel auch ausblenden.
        sentinel = float(getattr(cfg, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
        nz = float(((v.fillna(0).abs() > 1e-12) & (v != sentinel)).mean())
        cells.append(f"nonzero={nz:.2%}".ljust(32))
    print(f"  {t:<10} {sec_t:<25}  " + "  ".join(cells))

print()
print("[Validate] FERTIG. Erwarte: AAPL/MSFT/JPM/JNJ/XOM zeigen jetzt >50 % nonzero auf news_sec_*; "
      "news_macro_* > 50 % nonzero fuer ALLE Ticker.")
