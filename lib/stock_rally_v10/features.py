"""stock_rally_v10 — Feature-Matrix (Pipeline-Modul)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg

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
    if True in cfg.NEWS_EXTRA_TONE_ACCEL_OPTIONS:
        accel = ts.diff().diff()
        out[f"{col_prefix}tone_accel"] = accel.fillna(0.0).values
    return out


def _merge_gcam_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """GCAM-Tagesmittel wie „tone“ pro Kanal; Vol bleibt Artikelzahl (wie V2Tone-Pfad)."""
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        if cn not in s.columns:
            continue
        mid = f"{tag}_{cn}"
        msub = s[s["channel"] == "macro"]
        if not msub.empty:
            macro = msub[["Date", "vol", cn]].copy()
            macro = macro.rename(columns={cn: "tone"})
            macro["tone"] = macro["tone"].fillna(0.0)
            blk = _compute_news_block(
                macro, mom_w, vol_ma, tone_roll, f"news_macro_{mid}_"
            )
            if not blk.empty:
                df = df.merge(blk, on="Date", how="left")
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
            df = df.merge(sec_wide, on=["Date", "sector"], how="left")
    return df


def _merge_gcam_interaction_news_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """Differenzen aus GCAM-Basisspalten (z. B. fin. Spread, Confidence) wie eigenes „tone“-Serie."""
    for _slug, k_hi, k_lo in cfg.GCAM_INTERACTION_DIFFS:
        cn_a = cfg.gcam_series_colname(k_hi)
        cn_b = cfg.gcam_series_colname(k_lo)
        if cn_a not in s.columns or cn_b not in s.columns:
            continue
        mid = f"{tag}_gcam_{_slug}"
        msub = s[s["channel"] == "macro"]
        if not msub.empty:
            macro = msub[["Date", "vol", cn_a, cn_b]].copy()
            macro["tone"] = (macro[cn_a] - macro[cn_b]).fillna(0.0)
            macro = macro[["Date", "vol", "tone"]]
            blk = _compute_news_block(
                macro, mom_w, vol_ma, tone_roll, f"news_macro_{mid}_"
            )
            if not blk.empty:
                df = df.merge(blk, on="Date", how="left")
        sec_parts = []
        for sector in cfg.TICKERS_BY_SECTOR.keys():
            sub = s[s["channel"] == sector]
            if sub.empty:
                continue
            sec = sub[["Date", "vol", cn_a, cn_b]].copy()
            sec["tone"] = (sec[cn_a] - sec[cn_b]).fillna(0.0)
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
            df = df.merge(sec_wide, on=["Date", "sector"], how="left")
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
            df = df.merge(sec_wide, on=["Date", "sector"], how="left")
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
            df = df.merge(sec_wide, on=["Date", "sector"], how="left")
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
        df = df.merge(sec_wide, on=["Date", "sector"], how="left")
    return df


def _merge_gkg_theme_triples_for_tag(df, s, mom_w, vol_ma, tone_roll, tag):
    """Theme-Split-Kanäle ``sector##alias`` aus ``GKG_THEME_SQL_TRIPLES`` wie V2Tone-Pfad."""
    for ch, alias, _ in cfg._gkg_theme_triples():
        mid = f"{tag}_th_{alias}"
        cache_ch = f"{ch}##{alias}"
        sub = s[s["channel"] == cache_ch]
        if sub.empty:
            continue
        if ch == "macro":
            macro = sub[["Date", "tone", "vol"]].copy()
            blk = _compute_news_block(
                macro, mom_w, vol_ma, tone_roll, f"news_macro_{mid}_"
            )
            if not blk.empty:
                df = df.merge(blk, on="Date", how="left")
            continue
        sec_parts = []
        for sector in cfg.TICKERS_BY_SECTOR.keys():
            if sector != ch:
                continue
            sec = sub[["Date", "tone", "vol"]].copy()
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
            df = df.merge(sec_wide, on=["Date", "sector"], how="left")
    return df


def assemble_features(df, sentiment_df=None, meta_only=False):
    # Assemble feature matrix: sector/month, btc, breadth, optional news sentiment, dropna on technical worst-case
    # meta_only=True: nur Fenster rsi_w/bb_w/sma_w + ein News-Tripel aus cfg.best_params (wie cfg.FEAT_COLS) — für OOS.
    df = df.copy()

    df["sector"] = df["ticker"].map(cfg.TICKER_TO_SECTOR)
    df["sector_id"] = df["sector"].map(cfg.SECTOR_LABELS).astype(float)
    df["month"] = df["Date"].dt.month.astype(float)

    btc = df[df["ticker"] == "BTC-USD"][["Date", "momentum_20d"]].rename(
        columns={"momentum_20d": "btc_momentum"}
    )
    df = df.merge(btc, on="Date", how="left")
    df["btc_momentum"] = df["btc_momentum"].fillna(0.0)

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
        df = df.merge(breadth, on="Date", how="left")
        df[f"market_breadth_{sw}"] = df[f"market_breadth_{sw}"].fillna(0.5)

    for rw in rsi_loop:
        df[f"sector_avg_rsi_{rw}"] = df.groupby(["Date", "sector"])[f"rsi_{rw}"].transform("mean")

    if cfg.USE_NEWS_SENTIMENT:
        if sentiment_df is None or sentiment_df.empty:
            print(
                "Warning: cfg.USE_NEWS_SENTIMENT but no news sentiment data — news features set to 0."
            )
            if meta_only:
                for c in cfg.FEAT_COLS:
                    if str(c).startswith("news_"):
                        df[c] = 0.0
            else:
                for c in cfg.all_news_model_cols():
                    df[c] = 0.0
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
                    df = df.merge(blk, on="Date", how="left")
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
                    df = df.merge(sec_wide, on=["Date", "sector"], how="left")
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
                df = _merge_gcam_interaction_news_for_tag(
                    df, s, mom_w, vol_ma, tone_roll, tag
                )
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
                df = _merge_gkg_theme_triples_for_tag(
                    df, s, mom_w, vol_ma, tone_roll, tag
                )
                for c in cfg.FEAT_COLS:
                    if not str(c).startswith("news_"):
                        continue
                    if c not in df.columns:
                        df[c] = 0.0
                    else:
                        df[c] = df[c].fillna(0.0)
            else:
                for mom_w in cfg.NEWS_MOM_WINDOWS:
                    for vol_ma in cfg.NEWS_VOL_MA_WINDOWS:
                        for tone_roll in cfg.NEWS_TONE_ROLL_WINDOWS:
                            tag = cfg.news_feat_tag(mom_w, vol_ma, tone_roll)
                            macro = s[s["channel"] == "macro"][["Date", "tone", "vol"]]
                            blk = _compute_news_block(
                                macro, mom_w, vol_ma, tone_roll, f"news_macro_{tag}_"
                            )
                            if not blk.empty:
                                df = df.merge(blk, on="Date", how="left")
                            sec_parts = []
                            for sector in cfg.TICKERS_BY_SECTOR.keys():
                                sec = s[s["channel"] == sector][["Date", "tone", "vol"]]
                                blk_s = _compute_news_block(
                                    sec, mom_w, vol_ma, tone_roll, f"news_sec_{tag}_"
                                )
                                if blk_s.empty:
                                    continue
                                blk_s = blk_s.copy()
                                blk_s["sector"] = sector
                                sec_parts.append(blk_s)
                            if sec_parts:
                                sec_wide = pd.concat(sec_parts, ignore_index=True)
                                df = df.merge(sec_wide, on=["Date", "sector"], how="left")
                            df = _merge_gcam_news_for_tag(
                                df, s, mom_w, vol_ma, tone_roll, tag
                            )
                            df = _merge_gcam_interaction_news_for_tag(
                                df, s, mom_w, vol_ma, tone_roll, tag
                            )
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
                            df = _merge_gkg_theme_triples_for_tag(
                                df, s, mom_w, vol_ma, tone_roll, tag
                            )
                if True in cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                    for mom_w in cfg.NEWS_MOM_WINDOWS:
                        for vol_ma in cfg.NEWS_VOL_MA_WINDOWS:
                            for tone_roll in cfg.NEWS_TONE_ROLL_WINDOWS:
                                tag = cfg.news_feat_tag(mom_w, vol_ma, tone_roll)
                                mc = f"news_macro_{tag}_tone"
                                sc = f"news_sec_{tag}_tone"
                                if mc in df.columns and sc in df.columns:
                                    df[f"news_cross_{tag}_macro_minus_sec_tone"] = df[mc] - df[sc]
                for c in cfg.all_news_model_cols():
                    if c not in df.columns:
                        df[c] = 0.0
                    else:
                        df[c] = df[c].fillna(0.0)

    if meta_only:
        _rw = cfg.__dict__["rsi_w"]
        _bw = cfg.__dict__["bb_w"]
        _sw = cfg.__dict__["sma_w"]
        worst_case_cols = cfg.build_technical_cols(_rw, _bw, _sw)
    else:
        worst_case_cols = cfg.build_technical_cols(21, 25, 70)
    all_indicator_cols = [c for c in worst_case_cols if c in df.columns]
    df = df.dropna(subset=all_indicator_cols)

    _suffix = " (meta_only)" if meta_only else ""
    print(
        f"Feature matrix assembled{_suffix}. Shape: {df.shape}, "
        f"positive rate: {df['target'].mean():.1%}"
    )
    return df.reset_index(drop=True)

