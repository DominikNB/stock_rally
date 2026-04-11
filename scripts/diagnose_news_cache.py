"""
QC für den News-Pickle-Cache (Abdeckung, Lücken, Stabilität, Glättung).

Nutzt dieselbe Key-Auflösung wie die Pipeline (ohne BQ-Exploration, nur Datei/Manual).

  python scripts/diagnose_news_cache.py

Optionen: --pickle, --jump-k, --roll-window, --smoothing-rolls, --allow-bq-explore
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.news import _load_news_cache, resolve_gkg_gcam_metric_keys


def _gcam_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if str(c).startswith("gcam_"))


def _coverage_block(sub: pd.DataFrame, channel: str, gcam_cols: list[str]) -> dict:
    n = len(sub)
    if n == 0:
        return {"channel": channel, "n_rows": 0}
    vol = sub["vol"].astype(float)
    had = vol > 0
    n_had = int(had.sum())
    out = {
        "channel": channel,
        "n_rows": n,
        "pct_vol_gt0": round(100.0 * n_had / n, 2) if n else 0.0,
    }
    for gc in gcam_cols:
        if gc not in sub.columns:
            continue
        g = sub[gc].astype(float)
        non_na = g.notna()
        nz = (g != 0) & non_na
        out[f"{gc}_pct_non_na"] = round(100.0 * non_na.mean(), 2) if n else 0.0
        out[f"{gc}_pct_nonzero"] = round(100.0 * nz.sum() / n, 2) if n else 0.0
        if n_had > 0:
            sub_h = g[had]
            na_on_news = sub_h[sub_h.isna()]
            out[f"{gc}_pct_na_given_vol_gt0"] = round(100.0 * len(na_on_news) / n_had, 2)
            nz_on_news = ((sub_h != 0) & sub_h.notna()).sum()
            out[f"{gc}_pct_nonzero_given_vol_gt0"] = round(100.0 * nz_on_news / n_had, 2)
        else:
            out[f"{gc}_pct_na_given_vol_gt0"] = None
            out[f"{gc}_pct_nonzero_given_vol_gt0"] = None
    return out


def _jump_rate(series: pd.Series, k: float = 3.0) -> float:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return float("nan")
    d = s.diff().abs()
    d = d.iloc[1:]
    if d.empty:
        return 0.0
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    thresh = q3 + k * iqr if iqr > 0 else d.quantile(0.99)
    return float((d > thresh).mean())


def _stability_for_channel(
    sub: pd.DataFrame,
    gcam_cols: list[str],
    roll_win: int,
    jump_k: float,
) -> pd.DataFrame:
    sub = sub.sort_values("Date")
    rows = []
    for col in ["tone"] + [c for c in gcam_cols if c in sub.columns]:
        s = sub[col].astype(float)
        med_roll_std = (
            s.rolling(roll_win, min_periods=max(3, roll_win // 2)).std().median()
        )
        jr = _jump_rate(s, jump_k)
        rows.append(
            {
                "col": col,
                "med_roll_std_w": float(med_roll_std) if pd.notna(med_roll_std) else np.nan,
                "jump_rate": jr,
            }
        )
    return pd.DataFrame(rows)


def _corr_gcam(sub: pd.DataFrame, gcam_cols: list[str]) -> pd.DataFrame | None:
    use = [c for c in gcam_cols if c in sub.columns]
    if len(use) < 2:
        return None
    m = sub[use].astype(float)
    # nur Tage mit News, um stilles0/NaN-Rauschen zu reduzieren
    m = m.loc[sub["vol"].astype(float) > 0]
    if len(m) < 30:
        return None
    c = m.corr()
    return c


def _smoothing_table(sub: pd.DataFrame, rolls: list[int]) -> pd.DataFrame:
    sub = sub.sort_values("Date")
    t = sub["tone"].astype(float)
    rows = []
    for r in rolls:
        if r <= 0:
            ts = t
            label = 0
        else:
            ts = t.rolling(r, min_periods=1).mean()
            label = r
        d = ts.diff()
        rows.append(
            {
                "news_tone_roll": label,
                "var_first_diff": float(d.var()) if len(d) > 1 else np.nan,
                "mean_abs_diff": float(d.abs().mean()) if len(d) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="News-Cache Diagnose")
    ap.add_argument("--pickle", type=str, default=None, help="Override NEWS_CACHE_FILE")
    ap.add_argument("--jump-k", type=float, default=3.0, help="IQR-Faktor für Sprung-Rate")
    ap.add_argument("--roll-window", type=int, default=20, help="Fenster Rolling-Std (Handelstage)")
    ap.add_argument(
        "--smoothing-rolls",
        type=str,
        default="0,3,7,14",
        help="Komma-getrennte news_tone_roll Kandidaten für Makro",
    )
    ap.add_argument(
        "--allow-bq-explore",
        action="store_true",
        help="BigQuery für GCAM-Exploration zulassen (nur wenn cfg.GKG_GCAM_EXPLORE_KEYS=True)",
    )
    args = ap.parse_args()

    resolve_gkg_gcam_metric_keys(allow_bigquery_refresh=bool(args.allow_bq_explore))

    path = args.pickle or cfg.NEWS_CACHE_FILE
    df, meta = _load_news_cache(path)
    gcam_cols = _gcam_columns(df)
    keys_cfg = list(cfg._gkg_gcam_keys_clean())
    keys_meta = list(meta.get("gcam_keys") or [])

    print("=== News-Cache Diagnose ===")
    print(f"Pickle: {path}")
    print(f"Zeilen: {len(df)}  |  GCAM-Spalten im DF: {gcam_cols}")
    print(f"cfg GKG_GCAM_METRIC_KEYS / resolved: {keys_cfg}")
    print(f"cfg GKG_GCAM_MUST_HAVE_KEYS: {list(cfg._gkg_gcam_must_have_keys_clean())}")
    print(f"meta gcam_keys: {keys_meta}")
    if not gcam_cols and not keys_cfg:
        print(
            "(Hinweis: keine GCAM-Spalten — Keys setzen: data/gkg_gcam_metric_keys.json, "
            "cfg.GKG_GCAM_METRIC_KEYS, oder einmalig cfg.GKG_GCAM_EXPLORE_KEYS=True + Pipeline; "
            "Diagnose mit --allow-bq-explore nur wenn Exploration gewünscht.)"
        )
    print()

    if df.empty:
        print("Cache leer — nichts zu prüfen.")
        return

    channels = sorted(df["channel"].astype(str).unique())
    macro_ch = "macro"
    sector_like = [c for c in channels if "##" not in c and c != macro_ch]

    cov_rows = [_coverage_block(df[df["channel"] == ch], ch, gcam_cols) for ch in channels]
    cov = pd.DataFrame(cov_rows)
    print("--- Abdeckung pro Kanal (vol>0 = Artikeltag; GCAM-NaN vs. echte Null) ---")
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 200)
    print(cov.to_string(index=False))
    print()

    if macro_ch in channels:
        msub = df[df["channel"] == macro_ch].sort_values("Date")
        s_cov = []
        for ch in sector_like:
            sub = df[df["channel"] == ch]
            if sub.empty:
                continue
            vol_p = 100.0 * (sub["vol"].astype(float) > 0).mean()
            s_cov.append({"sector_channel": ch, "pct_vol_gt0": round(vol_p, 2), "n": len(sub)})
        print("--- Makro vs. Sektor-Kanäle (nur Artikel-Abdeckung) ---")
        m_vol_p = 100.0 * (msub["vol"].astype(float) > 0).mean()
        print(f"macro pct_vol_gt0: {m_vol_p:.2f}%  (n={len(msub)})")
        if s_cov:
            print(pd.DataFrame(s_cov).to_string(index=False))
        print()

    print("--- Stabilität (Makro): Rolling-Std, Sprung-Rate ---")
    msub = df[df["channel"] == macro_ch] if macro_ch in channels else df.iloc[0:0]
    if not msub.empty:
        stab = _stability_for_channel(msub, gcam_cols, args.roll_window, args.jump_k)
        print(stab.to_string(index=False))
        print(
            f"(jump_rate: Anteil |Δ| > Q3 + {args.jump_k}·IQR der ersten Differenzen)\n"
        )
        cmat = _corr_gcam(msub, gcam_cols)
        if cmat is not None:
            print("--- Korrelation GCAM (nur vol>0, Makro) ---")
            print(cmat.round(3).to_string())
            hi = []
            for i in range(len(cmat.columns)):
                for j in range(i + 1, len(cmat.columns)):
                    hi.append((cmat.iloc[i, j], cmat.columns[i], cmat.columns[j]))
            hi.sort(reverse=True, key=lambda x: abs(x[0]))
            print("\nTop |Korrelation|:")
            for v, a, b in hi[:8]:
                print(f"  {a} vs {b}: {v:.3f}")
        print()

    sample_sectors = sector_like[: min(5, len(sector_like))]
    for ch in sample_sectors:
        sub = df[df["channel"] == ch]
        if sub.empty:
            continue
        print(f"--- Stabilität (Sektor {ch!r}, Auszug) ---")
        print(
            _stability_for_channel(sub, gcam_cols, args.roll_window, args.jump_k).to_string(
                index=False
            )
        )
        print()

    rolls = [int(x.strip()) for x in args.smoothing_rolls.split(",") if x.strip()]
    if macro_ch in channels and not msub.empty:
        print("--- Glättung Makro-Ton (Varianz / mittlere |Δ| der geglätteten Serie) ---")
        print(_smoothing_table(msub, rolls).to_string(index=False))
        print(
            "\nHinweis: Kleine var_first_diff bei höherem roll → weniger Rauschen, "
            "aber stärker verzögerte Reaktion."
        )

    print("\n=== Ende ===")


if __name__ == "__main__":
    main()
