"""stock_rally_v10 — Feature-Matrix (Pipeline-Modul)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.equity_classification import (
    _ticker_symbol_str,
    add_yahoo_gics_feature_columns,
    build_classification_cache,
)

# Pro News-Fenster-Tripel (cfg.news_feat_tag = mom_w|vol_ma|tone_roll) ein Shard — nicht GKG-„Tag“.
# Wert: DataFrame mit MultiIndex (Date normiert, ticker); vermeidet pro Optuna-Trial ein ``shard.copy()``.
_NEWS_SHARD_FRAME_CACHE: dict[str, pd.DataFrame] = {}


def _news_fill_column_list() -> list[str]:
    """Spalten für ``assemble_features``-Fill (News): Optuna-Superset oder volle ``all_news_model``-Liste."""
    mode = str(getattr(cfg, "FEATURE_ASSEMBLE_NEWS_FILL", "optuna_union")).strip().lower()
    if mode in ("optuna_union", "optuna", "union"):
        return cfg.optuna_training_news_column_union()
    if mode in ("all_news_model", "all", "legacy"):
        return cfg.all_news_model_cols()
    if mode in ("none", "skip", "off"):
        return []
    raise ValueError(
        f"cfg.FEATURE_ASSEMBLE_NEWS_FILL={mode!r} — nutze 'optuna_union', 'all_news_model' oder 'none'"
    )


def _apply_news_fill(df: pd.DataFrame, fill_news: list[str], *, log_prefix: str = "") -> None:
    """News-Spalten: fehlende als 0 anlegen; vorhandene per **einem** fillna (keine tausend Einzelzuweisungen).

    Der frühere Loop pro Spalte triggert oft Block-Konsolidierung / hohen RAM — Batch ist günstiger.
    """
    if not fill_news:
        return
    want = list(fill_news)
    colset = set(df.columns)
    missing = [c for c in want if c not in colset]
    present = [c for c in want if c in colset]
    if missing:
        z = np.float32(0.0)
        for c in missing:
            df[c] = z
    if present:
        df[present] = df[present].fillna(0.0)
    if log_prefix:
        print(
            f"{log_prefix}News-Fill: {len(present):,} Spalten fillna (Batch), "
            f"{len(missing):,} neu als 0 — Liste hatte {len(want):,} Namen.",
            flush=True,
        )


def _assign_join_columns(
    df: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str] | str,
) -> None:
    """Weist Spalten aus ``right`` per Left-Join auf ``on`` zu — **ohne** ``df.merge`` auf der breiten Matrix.

    Ein vollständiges ``df.merge(klein)`` lässt pandas alle linken Blöcke mit dem Join-Ergebnis
    konkatenieren (~GB bei ~400k Zeilen × tausende Spalten) und kann ``MemoryError`` auslösen.

    Auch ``df[keys].merge(right)`` kann intern große Blöcke konsolidieren — daher bevorzugt:
    ``right.set_index(keys).reindex(MultiIndex aus df[keys])`` (kein ``merge``).
    """
    if right is None or right.empty:
        return
    keys = [on] if isinstance(on, str) else list(on)
    if not all(k in df.columns for k in keys):
        raise KeyError(f"_assign_join_columns: fehlende Keys {keys}")
    extra = [c for c in right.columns if c not in keys]
    if not extra:
        return
    r = right.drop_duplicates(subset=keys, keep="last").copy()
    if "Date" in keys:
        r["Date"] = pd.to_datetime(r["Date"]).dt.normalize()
    key_df = df[keys].copy()
    if "Date" in keys:
        key_df["Date"] = pd.to_datetime(key_df["Date"]).dt.normalize()
    idx = pd.MultiIndex.from_frame(key_df)
    r_ix = r.set_index(keys)
    if r_ix.index.has_duplicates:
        r_ix = r_ix[~r_ix.index.duplicated(keep="last")]
    aligned = r_ix.reindex(idx)
    if len(aligned) != len(df):
        raise RuntimeError(
            f"_assign_join_columns: reindex Länge {len(aligned)} != df {len(df)}"
        )
    for c in extra:
        ser = aligned[c]
        if pd.api.types.is_float_dtype(ser.dtype):
            df[c] = ser.astype(np.float32).to_numpy()
        else:
            df[c] = ser.to_numpy()


def _compute_news_block(ch_df, mom_w, vol_ma_w, tone_roll, col_prefix):
    # ch_df: Date, tone, vol (one row per day)
    if ch_df is None or ch_df.empty:
        return pd.DataFrame()
    g = ch_df.sort_values("Date").copy()
    dates = g["Date"].values
    ts = pd.Series(g["tone"].values, dtype=float)
    vs = pd.Series(g["vol"].values, dtype=float)
    if tone_roll > 0:
        ts = ts.rolling(window=tone_roll, min_periods=1).mean()
    tone_mom = ts - ts.shift(int(mom_w))
    vol_ma_roll = vs.rolling(window=int(vol_ma_w), min_periods=max(1, int(vol_ma_w) // 2)).mean()
    vol_spike = vs / (vol_ma_roll + 1e-10)
    out = pd.DataFrame({"Date": dates})
    out[f"{col_prefix}tone"] = ts.values
    out[f"{col_prefix}vol"] = vs.values
    out[f"{col_prefix}tone_mom"] = tone_mom.fillna(0.0).values
    out[f"{col_prefix}vol_spike"] = vol_spike.fillna(0.0).values
    for lag in (1, 3, 5):
        out[f"{col_prefix}tone_l{lag}"] = ts.shift(lag).fillna(0.0).values
        out[f"{col_prefix}vol_l{lag}"] = vs.shift(lag).fillna(0.0).values
    for zw in [w for w in cfg.NEWS_EXTRA_ZSCORE_WINDOWS if int(w) > 0]:
        min_p = max(3, int(zw) // 2)
        rt = ts.rolling(window=int(zw), min_periods=min_p)
        rv = vs.rolling(window=int(zw), min_periods=min_p)
        tone_z = (ts - rt.mean()) / (rt.std() + 1e-10)
        vol_z = (vs - rv.mean()) / (rv.std() + 1e-10)
        sw = f'_w{int(zw)}'
        out[f"{col_prefix}tone_z{sw}"] = tone_z.fillna(0.0).values
        out[f"{col_prefix}vol_z{sw}"] = vol_z.fillna(0.0).values
        # Schock/Δ/Impact auf der Z-Skala (tägliche Kanalserie — kein groupby ticker nötig)
        prev_roll = tone_z.shift(1).rolling(window=3, min_periods=1).mean()
        shock = (tone_z - prev_roll).fillna(0.0)
        dz1 = tone_z.diff().fillna(0.0)
        impact = (tone_z * vol_z.clip(lower=0.0)).fillna(0.0)
        out[f"{col_prefix}tone_z{sw}_shock"] = shock.to_numpy(dtype=np.float64)
        out[f"{col_prefix}tone_z{sw}_dz1"] = dz1.to_numpy(dtype=np.float64)
        out[f"{col_prefix}tone_z{sw}_x_volz_pos"] = impact.to_numpy(dtype=np.float64)
    if True in cfg.NEWS_EXTRA_TONE_ACCEL_OPTIONS:
        accel = ts.diff().diff()
        out[f"{col_prefix}tone_accel"] = accel.fillna(0.0).values
    vs_np = np.maximum(np.asarray(vs, dtype=np.float64), 0.0)
    ts_np = np.asarray(ts, dtype=np.float64)
    out[f"{col_prefix}tone_x_log1p_artcount"] = (ts_np * np.log1p(vs_np)).astype(np.float64)
    out[f"{col_prefix}tone_d1"] = ts.diff().fillna(0.0).to_numpy(dtype=np.float64)
    return out


def _merge_gcam_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """GCAM pro Key nur als Sektor-Serien (Macro bleibt V2-Basis ``news_macro_{tag}_*``)."""
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        if cn not in s.columns:
            continue
        mid = f"{tag}_{cn}"
        sec_parts = []
        for sector in cfg.TICKERS_BY_SECTOR.keys():
            sub = s[s["channel"] == sector]
            if sub.empty:
                continue
            sec = sub[["Date", "vol", cn]].copy()
            sec = sec.rename(columns={cn: "tone"})
            sec["tone"] = sec["tone"].fillna(0.0)
            blk_s = _compute_news_block(
                sec, mom_w, vol_ma, tone_roll, f"news_sec_{mid}_"
            )
            if blk_s.empty:
                continue
            blk_s = blk_s.copy()
            blk_s["sector"] = sector
            sec_parts.append(blk_s)
        if sec_parts:
            sec_wide = pd.concat(sec_parts, ignore_index=True)
            _assign_join_columns(df, sec_wide, ["Date", "sector"])
    return df


def _merge_gcam_anchor_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """GCAM nur aus Anker-Zeilen (Spalten anchor_*); gleiche Keys, nur Sektor-Kanäle."""
    if not getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False):
        return df
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        ac = f"anchor_{cn}"
        if ac not in s.columns or "anchor_vol" not in s.columns:
            continue
        mid = f"{tag}_anchor_{cn}"
        sec_parts = []
        for sector in cfg.TICKERS_BY_SECTOR.keys():
            sub = s[s["channel"] == sector]
            if sub.empty:
                continue
            sec = sub[["Date", "anchor_vol", ac]].copy()
            sec = sec.rename(columns={"anchor_vol": "vol", ac: "tone"})
            sec["tone"] = sec["tone"].fillna(0.0)
            blk_s = _compute_news_block(
                sec, mom_w, vol_ma, tone_roll, f"news_sec_{mid}_"
            )
            if blk_s.empty:
                continue
            blk_s = blk_s.copy()
            blk_s["sector"] = sector
            sec_parts.append(blk_s)
        if sec_parts:
            sec_wide = pd.concat(sec_parts, ignore_index=True)
            _assign_join_columns(df, sec_wide, ["Date", "sector"])
    return df


def _merge_gcam_div_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """Anker-GCAM minus breites Sektor-GCAM (Divergenz)."""
    if not getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False):
        return df
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        ac = f"anchor_{cn}"
        if (
            ac not in s.columns
            or cn not in s.columns
            or "anchor_vol" not in s.columns
        ):
            continue
        mid = f"{tag}_div_{cn}"
        sec_parts = []
        for sector in cfg.TICKERS_BY_SECTOR.keys():
            sub = s[s["channel"] == sector]
            if sub.empty:
                continue
            sec = sub[["Date", "anchor_vol", cn, ac]].copy()
            sec["tone"] = (sec[ac] - sec[cn]).fillna(0.0)
            sec["vol"] = sec["anchor_vol"].fillna(0.0)
            sec = sec[["Date", "vol", "tone"]]
            blk_s = _compute_news_block(
                sec, mom_w, vol_ma, tone_roll, f"news_sec_{mid}_"
            )
            if blk_s.empty:
                continue
            blk_s = blk_s.copy()
            blk_s["sector"] = sector
            sec_parts.append(blk_s)
        if sec_parts:
            sec_wide = pd.concat(sec_parts, ignore_index=True)
            _assign_join_columns(df, sec_wide, ["Date", "sector"])
    return df


def _merge_anchor_quality_idx_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """Mittelwert ausgewählter Anker-GCAM-Metriken (LMT-Kern) als eine Zeitreihe."""
    if not getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False):
        return df
    acols = [
        f"anchor_{cfg.gcam_series_colname(k)}"
        for k in cfg.GCAM_ANCHOR_QUALITY_MEAN_KEYS
    ]
    if "anchor_vol" not in s.columns or any(c not in s.columns for c in acols):
        return df
    mid = f"{tag}_anchor_quality_idx"
    sec_parts = []
    for sector in cfg.TICKERS_BY_SECTOR.keys():
        sub = s[s["channel"] == sector]
        if sub.empty:
            continue
        sec = sub[["Date", "anchor_vol"] + acols].copy()
        sec["tone"] = sec[acols].mean(axis=1, skipna=True).fillna(0.0)
        sec["vol"] = sec["anchor_vol"].fillna(0.0)
        sec = sec[["Date", "vol", "tone"]]
        blk_s = _compute_news_block(
            sec, mom_w, vol_ma, tone_roll, f"news_sec_{mid}_"
        )
        if blk_s.empty:
            continue
        blk_s = blk_s.copy()
        blk_s["sector"] = sector
        sec_parts.append(blk_s)
    if sec_parts:
        sec_wide = pd.concat(sec_parts, ignore_index=True)
        _assign_join_columns(df, sec_wide, ["Date", "sector"])
    return df


def _apply_news_merges_for_one_tag(
    df: pd.DataFrame,
    s: pd.DataFrame,
    mom_w: int,
    vol_ma: int,
    tone_roll: int,
    tag: str,
) -> None:
    """News-Merges für genau ein (mom_w, vol_ma, tone_roll)-Tripel — inplace auf ``df``."""
    macro = s[s["channel"] == "macro"][["Date", "tone", "vol"]]
    blk = _compute_news_block(macro, mom_w, vol_ma, tone_roll, f"news_macro_{tag}_")
    if not blk.empty:
        _assign_join_columns(df, blk, "Date")
    sec_parts = []
    for sector in cfg.TICKERS_BY_SECTOR.keys():
        sec = s[s["channel"] == sector][["Date", "tone", "vol"]]
        blk_s = _compute_news_block(sec, mom_w, vol_ma, tone_roll, f"news_sec_{tag}_")
        if blk_s.empty:
            continue
        blk_s = blk_s.copy()
        blk_s["sector"] = sector
        sec_parts.append(blk_s)
    if sec_parts:
        sec_wide = pd.concat(sec_parts, ignore_index=True)
        _assign_join_columns(df, sec_wide, ["Date", "sector"])
    _merge_gcam_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag)
    if cfg.NEWS_ANCHOR_ORG_FILTER:
        _merge_gcam_anchor_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag)
        _merge_gcam_div_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag)
        _merge_anchor_quality_idx_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag)


def _add_cross_macro_minus_sec_for_tag(df: pd.DataFrame, tag: str) -> None:
    if True not in cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
        return
    mc = f"news_macro_{tag}_tone"
    sc = f"news_sec_{tag}_tone"
    if mc in df.columns and sc in df.columns:
        df[f"news_cross_{tag}_macro_minus_sec_tone"] = df[mc] - df[sc]


def _write_news_shard_table(shard: pd.DataFrame, fpath_parquet: str) -> str:
    """Schreibt Parquet oder Pickle-Fallback; gibt den tatsächlichen Pfad zurück."""
    try:
        shard.to_parquet(fpath_parquet, index=False)
        return fpath_parquet
    except Exception:
        fpath_pkl = os.path.splitext(fpath_parquet)[0] + ".pkl"
        shard.to_pickle(fpath_pkl)
        return fpath_pkl


def _export_news_shards_for_grid(df_base: pd.DataFrame, s: pd.DataFrame) -> None:
    """Pro News-Grid-Tripel eine schmale Tabelle auf Platte — ``df_features`` bleibt ohne News-Spalten."""
    shard_dir = getattr(cfg, "FEATURE_SHARD_DIR", None) or os.path.join(
        os.getcwd(), "data", "feature_shards_news"
    )
    os.makedirs(shard_dir, exist_ok=True)
    _mw = list(cfg.NEWS_MOM_WINDOWS)
    _vma = list(cfg.NEWS_VOL_MA_WINDOWS)
    _tr = list(cfg.NEWS_TONE_ROLL_WINDOWS)
    _n_news = len(_mw) * len(_vma) * len(_tr)
    print(
        f"assemble_features [News → Platte]: schreibe {_n_news} Shard-Dateien nach {shard_dir!r} "
        f"(nur news_*-Spalten pro Fenster-Tripel feat_tag=mom_w|vol_ma|tone_roll; "
        f"df_features enthält weiter keine news_*).",
        flush=True,
    )
    manifest: dict[str, str] = {}
    _i_news = 0
    for mom_w in _mw:
        for vol_ma in _vma:
            for tone_roll in _tr:
                _i_news += 1
                tag = cfg.news_feat_tag(mom_w, vol_ma, tone_roll)
                print(
                    f"  assemble_features [News-Shard {_i_news}/{_n_news}]: feat_tag={tag} "
                    f"(Fenster-Tripel, kein GKG-Theme) — news_*-Merges → Datei …",
                    flush=True,
                )
                df_w = df_base.copy()
                _apply_news_merges_for_one_tag(df_w, s, mom_w, vol_ma, tone_roll, tag)
                if True in cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                    _add_cross_macro_minus_sec_for_tag(df_w, tag)
                news_cols = [c for c in df_w.columns if str(c).startswith("news_")]
                shard = df_w[["Date", "ticker", "sector"] + news_cols].copy()
                fpath = os.path.join(shard_dir, f"news_tag_{tag}.parquet")
                written = _write_news_shard_table(shard, fpath)
                manifest[tag] = os.path.basename(written)
                del df_w
    manifest_path = os.path.join(shard_dir, "news_shards_manifest.json")
    abs_manifest = {tag: os.path.join(shard_dir, fname) for tag, fname in manifest.items()}
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"manifest_version": 1, "shard_dir": os.path.abspath(shard_dir), "tags": manifest},
            fp,
            indent=2,
        )
    cfg.FEATURE_SHARD_DIR = os.path.abspath(shard_dir)
    cfg.NEWS_SHARD_MANIFEST = abs_manifest
    cfg._FEATURE_NEWS_SHARDS_ACTIVE = True
    _NEWS_SHARD_FRAME_CACHE.clear()
    print(
        f"assemble_features [News → Platte]: fertig — manifest {manifest_path!r}; "
        f"df_features bleibt bei {len(df_base.columns)} Spalten (nur Technik + Kontext, keine news_*).",
        flush=True,
    )


def _apply_news_regime_tone_interactions(df: pd.DataFrame, tag: str) -> None:
    """Tone × Regime (z. B. VVIX/VIX) — nach Shard-Merge; Spalten kommen aus ``augment_df_macro_regime_and_vol``."""
    if not getattr(cfg, "USE_NEWS_SENTIMENT", False) or not getattr(
        cfg, "NEWS_ADD_REGIME_TONE_INTERACTIONS", False
    ):
        return
    for base_col in getattr(cfg, "NEWS_REGIME_INTERACTION_BASE_COLS", ()):
        if base_col not in df.columns:
            continue
        m = np.nan_to_num(
            pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        safe = str(base_col).replace(".", "_")
        for prefix in ("news_macro", "news_sec"):
            tcol = f"{prefix}_{tag}_tone"
            if tcol not in df.columns:
                continue
            t = np.nan_to_num(
                pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            df[f"{tcol}_x_{safe}"] = (t * m).astype(np.float32)


def merge_news_shard_into_df(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Lädt News-Spalten für ``feat_tag`` (Fenster-Tripel-String, vgl. ``cfg.news_feat_tag``) an (Date, ticker)."""
    if not getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False):
        return df
    # Bereits gemerged (z. B. Phase 12 → 13): nur Regime-Interaktionen nachziehen.
    if f"news_macro_{tag}_tone" in df.columns:
        _apply_news_regime_tone_interactions(df, tag)
        return df
    manifest = getattr(cfg, "NEWS_SHARD_MANIFEST", None) or {}
    path = manifest.get(tag)
    if not path or not os.path.isfile(path):
        alt = os.path.join(getattr(cfg, "FEATURE_SHARD_DIR", "") or "", f"news_tag_{tag}.parquet")
        path = alt if os.path.isfile(alt) else None
        if not path:
            pkl = os.path.splitext(alt)[0] + ".pkl"
            path = pkl if os.path.isfile(pkl) else None
    if not path:
        raise FileNotFoundError(
            f"merge_news_shard_into_df: kein Shard für feat_tag={tag!r} (Fenster-Tripel) — "
            f"assemble_features [News → Platte] gelaufen und NEWS_SHARD_MANIFEST gesetzt?"
        )
    if tag not in _NEWS_SHARD_FRAME_CACHE:
        if path.endswith(".parquet"):
            shard = pd.read_parquet(path)
        else:
            shard = pd.read_pickle(path)
        dt = pd.to_datetime(shard["Date"]).dt.normalize()
        s_ix = shard.set_index([dt, shard["ticker"]])
        if s_ix.index.has_duplicates:
            s_ix = s_ix[~s_ix.index.duplicated(keep="last")]
        _NEWS_SHARD_FRAME_CACHE[tag] = s_ix
    s_ix = _NEWS_SHARD_FRAME_CACHE[tag]
    kdf = df[["Date", "ticker"]].copy()
    kdf["Date"] = pd.to_datetime(kdf["Date"]).dt.normalize()
    # Shard enthält u. a. ticker/sector — nur echte news_* (und ähnliche) mergen, nie Schlüsselspalten.
    _shard_skip = {"sector", "ticker", "Date"}
    news_cols = [c for c in s_ix.columns if c not in _shard_skip]
    idx = pd.MultiIndex.from_frame(kdf)
    aligned = s_ix.reindex(idx)
    for c in news_cols:
        ser = aligned[c]
        if pd.api.types.is_float_dtype(ser.dtype):
            df[c] = ser.astype(np.float32).fillna(0.0).to_numpy()
        else:
            df[c] = pd.to_numeric(ser, errors="coerce").fillna(0.0).to_numpy()
    _apply_news_regime_tone_interactions(df, tag)
    return df


def merge_news_shard_from_best_params(df: pd.DataFrame, best_params: dict) -> pd.DataFrame:
    """Wie ``merge_news_shard_into_df`` mit ``feat_tag`` aus ``best_params`` (news_mom_w / vol_ma / tone_roll)."""
    if not getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False):
        return df
    tag = cfg.news_feat_tag(
        int(best_params["news_mom_w"]),
        int(best_params["news_vol_ma"]),
        int(best_params["news_tone_roll"]),
    )
    return merge_news_shard_into_df(df, tag)


def assemble_features(df, sentiment_df=None, meta_only=False):
    # Baut df_features: technische Indikatoren kommen aus der Indikator-Pipeline; hier Kontext + optional News.
    # meta_only=True: ein News-Fenster-Tripel aus cfg.best_params in df (OOS) — kein Shard-Export.
    df = df.copy()
    print(
        f"assemble_features: Start — Eingang: Kurs + target + technische Indikatoren "
        f"({len(df):,} Zeilen × {df['ticker'].nunique()} Ticker). "
        f"Hier werden ergänzt: Kontext (Sektor, BTC, Breadth, …) und optional News-Features.",
        flush=True,
    )

    _gics_syms = sorted({_ticker_symbol_str(x) for x in df["ticker"].unique()} - {""})
    print(
        f"assemble_features: Yahoo GICS — yfinance für {len(_gics_syms)} Symbole (Sektor + Branche) …",
        flush=True,
    )
    _gics_cache = build_classification_cache(
        _gics_syms, progress_every=max(1, len(_gics_syms) // 10) if len(_gics_syms) > 1 else 1
    )
    add_yahoo_gics_feature_columns(df, _gics_cache)
    # Primär nach Yahoo-Sector zuordnen (2-stufige Yahoo-Klassifikation ist bereits in gics_* enthalten).
    # Fallback auf Legacy-Map nur, wenn Yahoo für ein Symbol keinen Sector liefert.
    _legacy_sector = df["ticker"].map(cfg.TICKER_TO_SECTOR)
    _gics_sector = df["gics_sector"].astype(str).str.strip()
    _gics_sector = _gics_sector.mask(_gics_sector.eq(""), np.nan)
    df["sector"] = _gics_sector.fillna(_legacy_sector).fillna("unknown")
    _sector_labels = {s: i for i, s in enumerate(sorted({str(x) for x in df["sector"].dropna().unique()}))}
    df["sector_id"] = df["sector"].map(_sector_labels).astype(float)
    # Laufzeit-Konsistenz für alle späteren Phasen (Website/Reports/Filter).
    cfg.SECTOR_LABELS = _sector_labels
    cfg.TICKER_TO_SECTOR = {
        t: (v.get("gics_sector") or cfg.TICKER_TO_SECTOR.get(t) or "unknown")
        for t, v in _gics_cache.items()
    }
    print(
        f"assemble_features: Sector-Mapping = Yahoo (fallback legacy), "
        f"{len(_sector_labels)} Sektoren im Lauf.",
        flush=True,
    )
    df["month"] = df["Date"].dt.month.astype(float)

    btc = df[df["ticker"] == "BTC-USD"][["Date", "momentum_20d"]].rename(
        columns={"momentum_20d": "btc_momentum"}
    )
    if btc.empty:
        print(
            "assemble_features: kein BTC-USD im aktuellen Ticker-Set — "
            "btc_momentum / btc_momentum_z_w* werden mit 0.0 gefüllt.",
            flush=True,
        )
        df["btc_momentum"] = np.float32(0.0)
        for _zw in cfg.BTC_MOMENTUM_Z_WINDOWS:
            df[f"btc_momentum_z_w{int(_zw)}"] = np.float32(0.0)
    else:
        btc = btc.sort_values("Date").copy()
        _assign_cols = ["Date", "btc_momentum"]
        for _zw in cfg.BTC_MOMENTUM_Z_WINDOWS:
            _zw = int(_zw)
            _min_p = max(10, _zw // 3)
            _roll = btc["btc_momentum"].rolling(_zw, min_periods=_min_p)
            _cn = f"btc_momentum_z_w{_zw}"
            btc[_cn] = (
                (btc["btc_momentum"] - _roll.mean()) / (_roll.std() + 1e-10)
            ).astype(float)
            _assign_cols.append(_cn)
        _assign_join_columns(df, btc[_assign_cols], "Date")
        df["btc_momentum"] = df["btc_momentum"].fillna(0.0)
        for _zw in cfg.BTC_MOMENTUM_Z_WINDOWS:
            _cn = f"btc_momentum_z_w{int(_zw)}"
            df[_cn] = df[_cn].fillna(0.0)

    if meta_only:
        _swm = cfg.__dict__["sma_w"]
        _rwm = cfg.__dict__["rsi_w"]
        sma_loop = [_swm]
        rsi_loop = [_rwm]
    else:
        sma_loop = list(cfg.SMA_WINDOWS)
        rsi_loop = list(cfg.RSI_WINDOWS)

    for sw in sma_loop:
        breadth = (
            df.groupby("Date")[f"sma_ratio_{sw}"]
            .apply(lambda x: (x > 1.0).mean())
            .rename(f"market_breadth_{sw}")
        )
        bdf = breadth.reset_index().sort_values("Date")
        _bcols = ["Date", f"market_breadth_{sw}"]
        for _br_zw in cfg.MARKET_BREADTH_Z_WINDOWS:
            _br_zw = int(_br_zw)
            _br_min_p = max(10, _br_zw // 3)
            _r = bdf[f"market_breadth_{sw}"].rolling(_br_zw, min_periods=_br_min_p)
            _zn = f"market_breadth_z_{sw}_w{_br_zw}"
            bdf[_zn] = (
                (bdf[f"market_breadth_{sw}"] - _r.mean()) / (_r.std() + 1e-10)
            ).astype(float)
            _bcols.append(_zn)
        _assign_join_columns(df, bdf[_bcols], "Date")
        df[f"market_breadth_{sw}"] = df[f"market_breadth_{sw}"].fillna(0.5)
        for _br_zw in cfg.MARKET_BREADTH_Z_WINDOWS:
            _zn = f"market_breadth_z_{sw}_w{int(_br_zw)}"
            df[_zn] = df[_zn].fillna(0.0)

    for rw in rsi_loop:
        df[f"sector_avg_rsi_{rw}"] = df.groupby(["Date", "sector"])[f"rsi_{rw}"].transform("mean")

    for _mw in cfg.REL_MOMENTUM_WINDOWS:
        _mw = int(_mw)
        _mcol = f"momentum_{_mw}d"
        if _mcol not in df.columns:
            df = df.sort_values(["ticker", "Date"])
            df[_mcol] = df.groupby("ticker", sort=False)["close"].pct_change(_mw)
        df[f"rel_momentum_{_mw}d"] = df[_mcol] - df.groupby(["Date", "sector"])[
            _mcol
        ].transform("mean")
        df[f"rel_momentum_{_mw}d"] = df[f"rel_momentum_{_mw}d"].fillna(0.0)

    print(
        "assemble_features: Kontext-Features fertig. Als Nächstes: News-Features …",
        flush=True,
    )

    if cfg.USE_NEWS_SENTIMENT:
        if sentiment_df is None or sentiment_df.empty:
            print(
                "assemble_features [News]: Keine News-Daten vorhanden. Lauf geht weiter; News-Features werden mit 0.0 belegt.",
                flush=True,
            )
            if meta_only:
                for c in cfg.FEAT_COLS:
                    if str(c).startswith("news_"):
                        df[c] = 0.0
            else:
                _fill_news = _news_fill_column_list()
                print(
                    f"assemble_features [News]: Fill-Liste aktiv ({len(_fill_news):,} News-Spalten).",
                    flush=True,
                )
                if _fill_news:
                    _apply_news_fill(
                        df,
                        _fill_news,
                        log_prefix="assemble_features [News ohne Daten]: ",
                    )
        else:
            s = sentiment_df.copy()
            s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
            if meta_only:
                bp = cfg.best_params
                mom_w = bp["news_mom_w"]
                vol_ma = bp["news_vol_ma"]
                tone_roll = bp["news_tone_roll"]
                tag = cfg.news_feat_tag(mom_w, vol_ma, tone_roll)
                macro = s[s["channel"] == "macro"][["Date", "tone", "vol"]]
                blk = _compute_news_block(macro, mom_w, vol_ma, tone_roll, f"news_macro_{tag}_")
                if not blk.empty:
                    _assign_join_columns(df, blk, "Date")
                sec_parts = []
                for sector in cfg.TICKERS_BY_SECTOR.keys():
                    sec = s[s["channel"] == sector][["Date", "tone", "vol"]]
                    blk_s = _compute_news_block(sec, mom_w, vol_ma, tone_roll, f"news_sec_{tag}_")
                    if blk_s.empty:
                        continue
                    blk_s = blk_s.copy()
                    blk_s["sector"] = sector
                    sec_parts.append(blk_s)
                if sec_parts:
                    sec_wide = pd.concat(sec_parts, ignore_index=True)
                    _assign_join_columns(df, sec_wide, ["Date", "sector"])
                use_cross = bool(
                    bp.get(
                        "news_extra_macro_sec_diff",
                        cfg.SEED_PARAMS.get("news_extra_macro_sec_diff"),
                    )
                )
                if use_cross and True in cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                    mc = f"news_macro_{tag}_tone"
                    sc = f"news_sec_{tag}_tone"
                    if mc in df.columns and sc in df.columns:
                        df[f"news_cross_{tag}_macro_minus_sec_tone"] = df[mc] - df[sc]
                df = _merge_gcam_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag)
                if cfg.NEWS_ANCHOR_ORG_FILTER:
                    df = _merge_gcam_anchor_news_for_tag(
                        df, s, mom_w, vol_ma, tone_roll, tag
                    )
                    df = _merge_gcam_div_news_for_tag(
                        df, s, mom_w, vol_ma, tone_roll, tag
                    )
                    df = _merge_anchor_quality_idx_news_for_tag(
                        df, s, mom_w, vol_ma, tone_roll, tag
                    )
                for c in cfg.FEAT_COLS:
                    if not str(c).startswith("news_"):
                        continue
                    if c not in df.columns:
                        df[c] = 0.0
                    else:
                        df[c] = df[c].fillna(0.0)
                _apply_news_regime_tone_interactions(df, tag)
            else:
                # Immer Shard-Export: keine monolithische Wide-Matrix (tausende news_*-Spalten) im RAM.
                _export_news_shards_for_grid(df, s)

    if not meta_only:
        from lib.stock_rally_v10.extended_base_features import augment_df_macro_regime_and_vol

        df = augment_df_macro_regime_and_vol(df)

    if meta_only:
        _rw = cfg.__dict__["rsi_w"]
        _bw = cfg.__dict__["bb_w"]
        _sw = cfg.__dict__["sma_w"]
        _bp = getattr(cfg, "best_params", None) or {}
        _sp = cfg.SEED_PARAMS
        worst_case_cols = cfg.build_technical_cols(
            _rw,
            _bw,
            _sw,
            btc_momentum_z_window=int(
                _bp.get("btc_momentum_z_window", _sp.get("btc_momentum_z_window", 60))
            ),
            market_breadth_z_window=int(
                _bp.get("market_breadth_z_window", _sp.get("market_breadth_z_window", 60))
            ),
            rel_momentum_window=int(
                _bp.get("rel_momentum_window", _sp.get("rel_momentum_window", 20))
            ),
        )
    else:
        worst_case_cols = [
            c for c in cfg.all_model_tech_col_names_for_assemble_dropna() if c in df.columns
        ]
    all_indicator_cols = [c for c in worst_case_cols if c in df.columns]
    df = df.dropna(subset=all_indicator_cols)

    _suffix = " [meta_only]" if meta_only else ""
    _has_news_cols = any(str(c).startswith("news_") for c in df.columns)
    _shards = getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False)
    if _shards and not meta_only:
        _news_where = (
            "News (news_*): nur in Shard-Dateien unter FEATURE_SHARD_DIR, nicht in df_features."
        )
    elif _has_news_cols:
        _news_where = "News (news_*): in df_features (meta_only oder Ein-Tripel-Pfad)."
    else:
        _news_where = "News: keine news_* in dieser Tabelle (ausgeschaltet oder nur Platzhalter)."
    print(
        f"assemble_features: Ende{_suffix} — df_features = technische Indikatoren + Kontext; {_news_where} "
        f"Shape {df.shape}, Zielrate {df['target'].mean():.1%}.",
        flush=True,
    )
    # reset_index(copy) konsolidiert alle Spaltenblöcke — bei ~7k Features × ~400k Zeilen
    # oft >10 GiB RAM (MemoryError). RangeIndex reicht für Downstream.
    df.index = pd.RangeIndex(len(df))
    return df

