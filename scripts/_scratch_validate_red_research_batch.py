"""
Rot-Research-Batch: IS/OOS-Tests für Makro-GDELT, Anker-Firmen-News, GICS, Events, Rot-ML.

  python scripts/_scratch_validate_red_research_batch.py

Schreibt: data/_scratch_red_research_batch.json

Nicht erneut getestet (bekannt wertlos): RS/Alpha-Chips, 0/1/2-Kombos, sec-minus-macro,
Liquidity, VIX-Chips als Badge.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.red_signal_quality import calibrate_gld_ret5_median_red_ref
from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.anchor_tickers import load_anchor_schedule, tickers_for_sector_on_date
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

MASTER = ROOT / "data" / "master_complete.csv"
NEWS_SHARD = ROOT / "data" / "feature_shards_news" / "news_tag_3_20_10.parquet"
GICS_CACHE = ROOT / "data" / "equity_classification_cache.json"
ANCHOR_SCHED = ROOT / "data" / "sector_anchor_quarters.json"
OUT = ROOT / "data" / "_scratch_red_research_batch.json"

TAG = "3_20_10"
MIN_N = 20
MIN_DELTA = 0.001
VIX_RED = 20.0

DEFENSIVE_GICS = {
    "consumer defensive",
    "healthcare",
    "utilities",
}
CYCLICAL_GICS = {
    "basic materials",
    "communication services",
    "consumer cyclical",
    "energy",
    "financial services",
    "industrials",
    "real estate",
    "technology",
}

NEWS_COLS = [
    f"news_macro_{TAG}_vol_spike",
    f"news_macro_{TAG}_tone_accel",
    f"news_macro_{TAG}_vol_z_w10",
    f"news_macro_{TAG}_tone_z_w10_shock",
    f"news_sec_{TAG}_gcam_c18_158_tone",
    f"news_sec_{TAG}_gcam_c18_158_vol_spike",
    f"news_sec_{TAG}_anchor_quality_idx_tone",
    f"news_sec_{TAG}_anchor_quality_idx_vol_spike",
]


@dataclass
class Result:
    group: str
    id: str
    label: str
    n_tune_hi: int
    n_tune_lo: int
    d_tune_pp: float | None
    n_fin_hi: int
    n_fin_lo: int
    d_fin_pp: float | None
    d_fin_raw_pp: float | None
    oos_ok: bool
    implement: bool
    fill_fin: float | None = None
    note: str = ""


def _assign_dataset(mc: pd.DataFrame) -> pd.DataFrame:
    raw = yf.download(
        "SPY",
        start=mc["Date"].min(),
        end=(mc["Date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        progress=False,
    )
    cal = np.sort(pd.to_datetime(raw.index).tz_localize(None).normalize().unique())
    _, meta, thr, final = _split_calendar_four_way(
        cal,
        float(cfg.TIME_SPLIT_FRAC_BASE),
        float(cfg.TIME_SPLIT_FRAC_META),
        float(cfg.TIME_SPLIT_FRAC_THRESHOLD),
        int(cfg.TIME_PURGE_TRADING_DAYS),
    )
    mt = {pd.Timestamp(d) for d in meta} | {pd.Timestamp(d) for d in thr}
    fin = {pd.Timestamp(d) for d in final}

    def _lab(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d).normalize()
        if d in fin:
            return "FINAL"
        if d in mt:
            return "META+THR"
        return "OTHER"

    mc = mc.copy()
    mc["dataset"] = mc["Date"].map(_lab)
    return mc


def _delta(hi: pd.DataFrame, lo: pd.DataFrame) -> tuple[float | None, int, int]:
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return None, len(hi), len(lo)
    return float(hi["ret"].mean() - lo["ret"].mean()), len(hi), len(lo)


def _eval(
    tune: pd.DataFrame,
    fin: pd.DataFrame,
    group: str,
    cid: str,
    label: str,
    mask_fn,
    *,
    note: str = "",
    fill_col: str | None = None,
) -> Result:
    def _run(sub: pd.DataFrame):
        s = sub.dropna(subset=["ret"])
        try:
            hi, lo = mask_fn(s)
            hi, lo = hi.dropna(subset=["ret"]), lo.dropna(subset=["ret"])
            return _delta(hi, lo)
        except Exception:
            return None, 0, 0

    dt, nth, ntl = _run(tune)
    df, nfh, nfl = _run(fin)
    dfr = None
    if fin is not None and len(fin):
        try:
            hi, lo = mask_fn(fin.dropna(subset=["ret"]))
            hi, lo = hi.dropna(subset=["ret"]), lo.dropna(subset=["ret"])
            if len(hi) and len(lo):
                dfr = float(100 * (hi["ret"].mean() - lo["ret"].mean()))
        except Exception:
            pass
    oos = (
        dt is not None
        and df is not None
        and dt * df > 0
        and abs(dt) >= MIN_DELTA
        and abs(df) >= MIN_DELTA
    )
    impl = bool(oos and dt > 0 and df > 0)
    fill_fin = None
    if fill_col and fill_col in fin.columns:
        fill_fin = float(fin[fill_col].notna().mean())
    return Result(
        group=group,
        id=cid,
        label=label,
        n_tune_hi=nth,
        n_tune_lo=ntl,
        d_tune_pp=round(100 * dt, 3) if dt is not None else None,
        n_fin_hi=nfh,
        n_fin_lo=nfl,
        d_fin_pp=round(100 * df, 3) if df is not None else None,
        d_fin_raw_pp=round(dfr, 3) if dfr is not None else None,
        oos_ok=bool(oos),
        implement=impl,
        fill_fin=fill_fin,
        note=note,
    )


def _merge_news_shard(red: pd.DataFrame) -> pd.DataFrame:
    if not NEWS_SHARD.is_file():
        print(f"Warnung: News-Shard fehlt: {NEWS_SHARD}")
        return red
    usecols = ["Date", "ticker"] + [c for c in NEWS_COLS if c]
    import pyarrow.parquet as pq

    avail = set(pq.ParquetFile(NEWS_SHARD).schema.names)
    usecols = [c for c in usecols if c in avail]
    shard = pd.read_parquet(NEWS_SHARD, columns=usecols)
    shard["Date"] = pd.to_datetime(shard["Date"]).dt.normalize()
    keys = red[["Date", "ticker"]].drop_duplicates()
    merged = keys.merge(shard.drop_duplicates(subset=["Date", "ticker"]), on=["Date", "ticker"], how="left")
    out = red.merge(merged, on=["Date", "ticker"], how="left", suffixes=("", "_sh"))
    return out


def _attach_gics(red: pd.DataFrame) -> pd.DataFrame:
    if not GICS_CACHE.is_file():
        red["gics_sector_key"] = ""
        return red
    obj = json.loads(GICS_CACHE.read_text(encoding="utf-8"))
    rows = obj.get("rows") or {}
    gics = {k: (v or {}).get("gics_sector_key", "") for k, v in rows.items()}
    red = red.copy()
    red["gics_sector_key"] = red["ticker"].astype(str).map(gics).fillna("")
    return red


def _attach_anchor_flag(red: pd.DataFrame) -> pd.DataFrame:
    sched = load_anchor_schedule(ANCHOR_SCHED) or []
    red = red.copy()
    flags = []
    for _, r in red.iterrows():
        sec = str(r.get("sector") or "")
        d = pd.Timestamp(r["Date"]).date()
        anc = tickers_for_sector_on_date(sched, sec, d)
        flags.append(str(r["ticker"]) in set(anc))
    red["is_sector_anchor"] = flags
    return red


def _calibrate_thresholds(tune: pd.DataFrame) -> dict[str, float]:
    gref = calibrate_gld_ret5_median_red_ref()
    tone_col = f"news_macro_{TAG}_tone"
    spike_col = f"news_macro_{TAG}_vol_spike"
    tone = pd.to_numeric(tune.get(tone_col), errors="coerce")
    spike = pd.to_numeric(tune.get(spike_col), errors="coerce")
    gld = pd.to_numeric(tune.get("gld_ret_5d"), errors="coerce")
    return {
        "gld_ref": gref,
        "macro_tone_med": float(tone.median()) if tone.notna().sum() >= 30 else float("nan"),
        "macro_tone_p20": float(tone.quantile(0.20)) if tone.notna().sum() >= 30 else float("nan"),
        "macro_spike_med": float(spike.median()) if spike.notna().sum() >= 30 else float("nan"),
        "macro_spike_p80": float(spike.quantile(0.80)) if spike.notna().sum() >= 30 else float("nan"),
        "gld_med": float(gld.median()) if gld.notna().sum() >= 30 else float("nan"),
    }


def _risk_off_flags(s: pd.DataFrame, thr: dict[str, float]) -> pd.DataFrame:
    tone = pd.to_numeric(s[f"news_macro_{TAG}_tone"], errors="coerce")
    spike = pd.to_numeric(s[f"news_macro_{TAG}_vol_spike"], errors="coerce")
    gld = pd.to_numeric(s["gld_ret_5d"], errors="coerce")
    out = s.copy()
    out["f_gld_high"] = (gld >= thr["gld_ref"]).astype("float")
    out["f_macro_tone_neg"] = (tone < thr["macro_tone_med"]).astype("float")
    out["f_macro_spike"] = (spike > thr["macro_spike_med"]).astype("float")
    out["f_macro_tone_ext_neg"] = (tone < thr["macro_tone_p20"]).astype("float")
    out.loc[gld.isna(), "f_gld_high"] = np.nan
    out.loc[tone.isna(), "f_macro_tone_neg"] = np.nan
    out.loc[tone.isna(), "f_macro_tone_ext_neg"] = np.nan
    out.loc[spike.isna(), "f_macro_spike"] = np.nan
    out["risk_off_stack"] = out[["f_gld_high", "f_macro_tone_neg", "f_macro_spike"]].sum(axis=1, min_count=1)
    return out


def _ml_rot_model(
    tune: pd.DataFrame,
    fin: pd.DataFrame,
    feat_cols: list[str],
    *,
    model: str = "linreg",
) -> Result:
    tr = tune.dropna(subset=["ret"] + feat_cols).copy()
    te = fin.dropna(subset=["ret"] + feat_cols).copy()
    if len(tr) < 80 or len(te) < 30:
        return Result(
            group="5_rot_ml",
            id=f"{model}_{len(feat_cols)}feat",
            label=f"{model} rot-only ({len(feat_cols)} feat), pred>=median",
            n_tune_hi=0,
            n_tune_lo=0,
            d_tune_pp=None,
            n_fin_hi=0,
            n_fin_lo=0,
            d_fin_pp=None,
            d_fin_raw_pp=None,
            oos_ok=False,
            implement=False,
            note="zu wenig Train/Test nach dropna",
        )
    X_tr = tr[feat_cols].astype(float).values
    y_tr = tr["ret"].astype(float).values
    if model == "linreg":
        pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    else:
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("m", RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=30, random_state=42)),
        ])
    pipe.fit(X_tr, y_tr)
    for name, sub in [("tune", tr), ("fin", te)]:
        sub["_pred"] = pipe.predict(sub[feat_cols].astype(float).values)
    med_t = tr["_pred"].median()
    med_f = te["_pred"].median()
    hi_t, lo_t = tr[tr["_pred"] >= med_t], tr[tr["_pred"] < med_t]
    hi_f, lo_f = te[te["_pred"] >= med_f], te[te["_pred"] < med_f]
    dt, nth, ntl = _delta(hi_t, lo_t)
    df, nfh, nfl = _delta(hi_f, lo_f)
    oos = (
        dt is not None
        and df is not None
        and dt * df > 0
        and abs(dt) >= MIN_DELTA
        and abs(df) >= MIN_DELTA
    )
    return Result(
        group="5_rot_ml",
        id=f"{model}_{len(feat_cols)}feat",
        label=f"{model} rot-only ({', '.join(feat_cols[:4])}{'…' if len(feat_cols)>4 else ''})",
        n_tune_hi=nth,
        n_tune_lo=ntl,
        d_tune_pp=round(100 * dt, 3) if dt is not None else None,
        n_fin_hi=nfh,
        n_fin_lo=nfl,
        d_fin_pp=round(100 * df, 3) if df is not None else None,
        d_fin_raw_pp=round(100 * df, 3) if df is not None else None,
        oos_ok=bool(oos),
        implement=bool(oos and dt > 0 and df > 0),
        note="fit META+THR rot, hi=pred>=median",
    )


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    red = _merge_news_shard(red)
    red = _attach_gics(red)
    red = _attach_anchor_flag(red)
    tune = red[red["dataset"] == "META+THR"].copy()
    fin = red[red["dataset"] == "FINAL"].copy()
    thr = _calibrate_thresholds(tune)
    red = _risk_off_flags(red, thr)
    tune = red[red["dataset"] == "META+THR"].copy()
    fin = red[red["dataset"] == "FINAL"].copy()

    print(f"Rot: tune={len(tune)} final={len(fin)}")
    print(f"Schwellen: gld_ref={100*thr['gld_ref']:+.2f}%  macro_tone_med={thr['macro_tone_med']:.3f}  "
          f"macro_spike_med={thr['macro_spike_med']:.3f}")
    corr = tune[["f_gld_high", "f_macro_tone_neg", "f_macro_spike"]].astype(float).corr()
    print(f"Korrelation Risk-off-Komponenten (tune):\n{corr.round(2).to_string()}\n")

    results: list[Result] = []

    def add(r: Result) -> None:
        results.append(r)
        mark = "OOS-OK" if r.oos_ok else "fail"
        impl = " *" if r.implement else ""
        ts = f"{r.d_tune_pp:+.2f}pp" if r.d_tune_pp is not None else "n/a"
        fs = f"{r.d_fin_pp:+.2f}pp" if r.d_fin_pp is not None else (
            f"{r.d_fin_raw_pp:+.2f}pp (n<{MIN_N})" if r.d_fin_raw_pp is not None else "n/a"
        )
        print(f"[{mark}{impl}] [{r.group}] {r.id}")
        print(f"       {r.label}")
        print(f"       tune {ts} n={r.n_tune_hi}/{r.n_tune_lo} | fin {fs} n={r.n_fin_hi}/{r.n_fin_lo}")
        if r.note:
            print(f"       note: {r.note}")

    tone_c = f"news_macro_{TAG}_tone"
    spike_c = f"news_macro_{TAG}_vol_spike"
    gcam_c = f"news_sec_{TAG}_gcam_c18_158_tone"
    gcam_sp = f"news_sec_{TAG}_gcam_c18_158_vol_spike"
    anc_c = f"news_sec_{TAG}_anchor_quality_idx_tone"
    anc_sp = f"news_sec_{TAG}_anchor_quality_idx_vol_spike"

    # --- 1 Makro-GDELT ---
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_tone_low_med",
        "Makro-Tone unter Rot-Median (global kalibriert)",
        lambda s: (s[pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"]],
                   s[pd.to_numeric(s[tone_c], errors="coerce") >= thr["macro_tone_med"]]),
        fill_col=tone_c,
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_tone_ext_neg_p20",
        "Makro-Tone unter 20%-Quantil (rot tune)",
        lambda s: (s[pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_p20"]],
                   s[pd.to_numeric(s[tone_c], errors="coerce") >= thr["macro_tone_p20"]]),
        fill_col=tone_c,
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_vol_spike_high",
        "Makro-Vol-Spike über Median",
        lambda s: (s[pd.to_numeric(s[spike_c], errors="coerce") > thr["macro_spike_med"]],
                   s[pd.to_numeric(s[spike_c], errors="coerce") <= thr["macro_spike_med"]]),
        fill_col=spike_c,
        note="hi=Spike hoch — erwartet schlechter wenn Risk-off",
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_vol_spike_low",
        "Makro-Vol-Spike unter Median (ruhige News)",
        lambda s: (s[pd.to_numeric(s[spike_c], errors="coerce") <= thr["macro_spike_med"]],
                   s[pd.to_numeric(s[spike_c], errors="coerce") > thr["macro_spike_med"]]),
        fill_col=spike_c,
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_tone_z_shock_neg",
        "Makro-Tone-Z-Shock (w10) unter Median",
        lambda s: (
            s[pd.to_numeric(s[f"news_macro_{TAG}_tone_z_w10_shock"], errors="coerce")
            < s[f"news_macro_{TAG}_tone_z_w10_shock"].median()],
            s[pd.to_numeric(s[f"news_macro_{TAG}_tone_z_w10_shock"], errors="coerce")
            >= s[f"news_macro_{TAG}_tone_z_w10_shock"].median()],
        ),
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "risk_off_stack0",
        "Risk-off-Stack=0 (kein GLD+/Tone-/Spike+)",
        lambda s: (s[s["risk_off_stack"] == 0], s[s["risk_off_stack"] >= 1]),
        note="Stack aus GLD hoch, Makro-Tone neg, Vol-Spike",
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "risk_off_stack_le1",
        "Risk-off-Stack <=1 vs >=2",
        lambda s: (s[s["risk_off_stack"] <= 1], s[s["risk_off_stack"] >= 2]),
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "gld_low_only",
        "GLD unter global ref (Referenz)",
        lambda s: (s[pd.to_numeric(s["gld_ret_5d"], errors="coerce") < thr["gld_ref"]],
                   s[pd.to_numeric(s["gld_ret_5d"], errors="coerce") >= thr["gld_ref"]]),
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_low_AND_gld_low",
        "Makro-Tone niedrig UND GLD niedrig",
        lambda s: (
            s[(pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])
              & (pd.to_numeric(s["gld_ret_5d"], errors="coerce") < thr["gld_ref"])],
            s[~((pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])
                & (pd.to_numeric(s["gld_ret_5d"], errors="coerce") < thr["gld_ref"]))],
        ),
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "macro_low_OR_gld_low",
        "Makro-Tone niedrig ODER GLD niedrig",
        lambda s: (
            s[(pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])
              | (pd.to_numeric(s["gld_ret_5d"], errors="coerce") < thr["gld_ref"])],
            s[~((pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])
                | (pd.to_numeric(s["gld_ret_5d"], errors="coerce") < thr["gld_ref"]))],
        ),
    ))
    add(_eval(
        tune, fin, "1_macro_gdelt", "gcam_c18_158_tone_low",
        "Sektor-GCAM Angst c18.158 Tone unter Median",
        lambda s: (
            s[pd.to_numeric(s[gcam_c], errors="coerce") < s[gcam_c].median()],
            s[pd.to_numeric(s[gcam_c], errors="coerce") >= s[gcam_c].median()],
        ),
        fill_col=gcam_c,
        note="GCAM nur sektoral im Shard",
    ))

    # --- 2 Firmen-News (Anker-Proxy) ---
    add(_eval(
        tune, fin, "2_firm_news", "anchor_tone_pos",
        "Anker-Quality-Tone > 0 (Sektor-Large-Cap-Proxy)",
        lambda s: (s[pd.to_numeric(s[anc_c], errors="coerce") > 0],
                   s[pd.to_numeric(s[anc_c], errors="coerce") <= 0]),
        fill_col=anc_c,
    ))
    add(_eval(
        tune, fin, "2_firm_news", "anchor_tone_ge_med",
        "Anker-Quality-Tone >= Median",
        lambda s: (
            s[pd.to_numeric(s[anc_c], errors="coerce") >= s[anc_c].median()],
            s[pd.to_numeric(s[anc_c], errors="coerce") < s[anc_c].median()],
        ),
        fill_col=anc_c,
    ))
    add(_eval(
        tune, fin, "2_firm_news", "anchor_calm",
        "Anker-Tone positiv UND kein Vol-Spike",
        lambda s: (
            s[(pd.to_numeric(s[anc_c], errors="coerce") > 0)
              & (pd.to_numeric(s[anc_sp], errors="coerce") <= s[anc_sp].median())],
            s[~((pd.to_numeric(s[anc_c], errors="coerce") > 0)
                & (pd.to_numeric(s[anc_sp], errors="coerce") <= s[anc_sp].median()))],
        ),
        fill_col=anc_c,
    ))
    add(_eval(
        tune, fin, "2_firm_news", "anchor_scandal_wave",
        "Anker-Tone negativ UND Vol-Spike hoch",
        lambda s: (
            s[(pd.to_numeric(s[anc_c], errors="coerce") < 0)
              & (pd.to_numeric(s[anc_sp], errors="coerce") > s[anc_sp].median())],
            s[~((pd.to_numeric(s[anc_c], errors="coerce") < 0)
                & (pd.to_numeric(s[anc_sp], errors="coerce") > s[anc_sp].median()))],
        ),
        fill_col=anc_c,
    ))
    add(_eval(
        tune, fin, "2_firm_news", "anchor_ticker_only_calm",
        "Nur Sektor-Anker-Ticker: Tone+ & kein Spike",
        lambda s: (
            s[s["is_sector_anchor"]
              & (pd.to_numeric(s[anc_c], errors="coerce") > 0)
              & (pd.to_numeric(s[anc_sp], errors="coerce") <= s[anc_sp].median())],
            s[~(s["is_sector_anchor"]
                 & (pd.to_numeric(s[anc_c], errors="coerce") > 0)
                 & (pd.to_numeric(s[anc_sp], errors="coerce") <= s[anc_sp].median()))],
        ),
        note=f"Anker-Anteil tune={100*tune.is_sector_anchor.mean():.1f}% fin={100*fin.is_sector_anchor.mean():.1f}%",
    ))
    add(_eval(
        tune, fin, "2_firm_news", "gcam_fear_spike_low",
        "GCAM c18.158 Vol-Spike unter Median",
        lambda s: (
            s[pd.to_numeric(s[gcam_sp], errors="coerce") <= s[gcam_sp].median()],
            s[pd.to_numeric(s[gcam_sp], errors="coerce") > s[gcam_sp].median()],
        ),
        fill_col=gcam_sp,
    ))

    # --- 3 GICS Branchen-Regime ---
    gics_fill = float(red["gics_sector_key"].astype(str).str.len().gt(0).mean())
    print(f"\nGICS aus Cache: fill={100*gics_fill:.1f}%\n")

    add(_eval(
        tune, fin, "3_gics", "gics_defensive",
        "GICS defensiv (Consumer Defensive/Healthcare/Utilities)",
        lambda s: (
            s[s["gics_sector_key"].astype(str).str.lower().isin(DEFENSIVE_GICS)],
            s[~s["gics_sector_key"].astype(str).str.lower().isin(DEFENSIVE_GICS)],
        ),
        note=f"gics fill tune={100*(tune.gics_sector_key!='').mean():.0f}%",
    ))
    add(_eval(
        tune, fin, "3_gics", "gics_cyclical",
        "GICS zyklisch vs rest",
        lambda s: (
            s[s["gics_sector_key"].astype(str).str.lower().isin(CYCLICAL_GICS)],
            s[~s["gics_sector_key"].astype(str).str.lower().isin(CYCLICAL_GICS)],
        ),
    ))
    add(_eval(
        tune, fin, "3_gics", "defensive_x_macro_low",
        "Defensiv UND Makro-Tone niedrig",
        lambda s: (
            s[s["gics_sector_key"].astype(str).str.lower().isin(DEFENSIVE_GICS)
              & (pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])],
            s[~(s["gics_sector_key"].astype(str).str.lower().isin(DEFENSIVE_GICS)
                 & (pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"]))],
        ),
    ))
    add(_eval(
        tune, fin, "3_gics", "cyclical_x_riskoff0",
        "Zyklisch UND Risk-off-Stack=0",
        lambda s: (
            s[s["gics_sector_key"].astype(str).str.lower().isin(CYCLICAL_GICS) & (s["risk_off_stack"] == 0)],
            s[~(s["gics_sector_key"].astype(str).str.lower().isin(CYCLICAL_GICS) & (s["risk_off_stack"] == 0))],
        ),
    ))

    # --- 4 Event-Kontext ---
    add(_eval(
        tune, fin, "4_events", "no_macro_event_2bd",
        "Kein Makro-Event ±2 Handelstage",
        lambda s: (s[~s["macro_event_within_2bd"].astype(bool)], s[s["macro_event_within_2bd"].astype(bool)]),
    ))
    add(_eval(
        tune, fin, "4_events", "macro_event_AND_spike",
        "Makro-Event ±2bd UND Makro-Vol-Spike hoch",
        lambda s: (
            s[s["macro_event_within_2bd"].astype(bool)
              & (pd.to_numeric(s[spike_c], errors="coerce") > thr["macro_spike_med"])],
            s[~(s["macro_event_within_2bd"].astype(bool)
                 & (pd.to_numeric(s[spike_c], errors="coerce") > thr["macro_spike_med"]))],
        ),
    ))
    add(_eval(
        tune, fin, "4_events", "macro_event_AND_tone_neg",
        "Makro-Event ±2bd UND Makro-Tone negativ (unter Median)",
        lambda s: (
            s[s["macro_event_within_2bd"].astype(bool)
              & (pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"])],
            s[~(s["macro_event_within_2bd"].astype(bool)
                 & (pd.to_numeric(s[tone_c], errors="coerce") < thr["macro_tone_med"]))],
        ),
    ))
    add(_eval(
        tune, fin, "4_events", "earnings_clear",
        "Earnings jenseits 15 Handelstage (Swing-frei)",
        lambda s: (
            s[s["earnings_beyond_swing_15b"].astype(str).str.lower().isin(("true", "1", "yes"))],
            s[~s["earnings_beyond_swing_15b"].astype(str).str.lower().isin(("true", "1", "yes"))],
        ),
    ))
    add(_eval(
        tune, fin, "4_events", "no_event_no_spike",
        "Kein Event ±2bd UND Vol-Spike niedrig",
        lambda s: (
            s[(~s["macro_event_within_2bd"].astype(bool))
              & (pd.to_numeric(s[spike_c], errors="coerce") <= thr["macro_spike_med"])],
            s[~((~s["macro_event_within_2bd"].astype(bool))
                & (pd.to_numeric(s[spike_c], errors="coerce") <= thr["macro_spike_med"]))],
        ),
    ))

    # --- 5 Rot-Conditional ML ---
    ml_feats_gdelt = [
        tone_c,
        spike_c,
        "gld_ret_5d",
        "regime_vix_z_20d",
        "vix3m_vix_ratio",
        f"news_sec_{TAG}_anchor_quality_idx_tone",
        gcam_c,
    ]
    ml_feats_base = ["prob", "regime_vix_z_20d", "gld_ret_5d"]
    ml_feats_mix = ["prob"] + ml_feats_gdelt

    add(_ml_rot_model(tune, fin, ml_feats_gdelt, model="linreg"))
    add(_ml_rot_model(tune, fin, ml_feats_gdelt, model="rf"))
    add(_ml_rot_model(tune, fin, ml_feats_base, model="linreg"))
    add(_ml_rot_model(tune, fin, ml_feats_mix, model="linreg"))

    add(_eval(
        tune, fin, "5_rot_ml", "prob_ge_median",
        "Baseline: Modell-prob >= Median (rot)",
        lambda s: (s[pd.to_numeric(s["prob"], errors="coerce") >= s["prob"].median()],
                   s[pd.to_numeric(s["prob"], errors="coerce") < s["prob"].median()]),
    ))

    passed = [r for r in results if r.implement]
    best = sorted(
        [r for r in results if r.oos_ok],
        key=lambda r: (r.d_fin_pp or -999),
        reverse=True,
    )

    payload = {
        "n_tune": len(tune),
        "n_final": len(fin),
        "thresholds": thr,
        "gics_cache_fill": gics_fill,
        "news_shard_fill_fin": {
            c: float(fin[c].notna().mean()) if c in fin.columns else 0.0 for c in NEWS_COLS[:4]
        },
        "risk_off_corr_tune": corr.round(4).to_dict(),
        "passed_ids": [r.id for r in passed],
        "best_oos": [asdict(r) for r in best[:8]],
        "results": [asdict(r) for r in results],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")
    print(f"OOS-implementierbar ({len(passed)}): {[r.id for r in passed]}")


if __name__ == "__main__":
    main()
