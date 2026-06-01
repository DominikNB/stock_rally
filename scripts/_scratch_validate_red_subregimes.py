"""
Innerhalb VIX < 20 (rote Ampel): Zusatzinfos testen, ob gute vs. schwache Signale trennbar sind.

Daten:
  - data/_scratch_meta_thr_final_signals.csv (ret, vix, dist_high, dataset)
  - data/master_complete.csv (angereicherte Spalten, v. a. FINAL-OOS)
  - Yahoo: VIX-Z (20d), VIX3M/VIX-Ratio

Regel: Nur Splits behalten, bei denen META+THRESHOLD und FINAL dieselbe Richtung
(hoher vs. niedriger Bucket) zeigen. Kein Filter-Vorschlag ohne FINAL-Bestaetigung.

  python scripts/_scratch_validate_red_subregimes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIGNALS_CSV = ROOT / "data" / "_scratch_meta_thr_final_signals.csv"
MASTER_CSV = ROOT / "data" / "master_complete.csv"
VIX_AMPEL_YELLOW = 20.0
GOOD_RET = 0.02
MIN_N = 30


def _vix_z_vs_20d(close: pd.Series, end_d: pd.Timestamp) -> float:
    end_d = pd.Timestamp(end_d).normalize()
    s = close.dropna().sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    sub = s[s.index <= end_d]
    if len(sub) < 20:
        return float("nan")
    w = sub.tail(20).astype(float)
    st = w.std(ddof=0)
    if not np.isfinite(st) or st == 0:
        return float("nan")
    return float((w.iloc[-1] - w.mean()) / st)


def _load_vix_series(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start="2014-01-01", progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"].squeeze()
    else:
        s = raw["Close"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def _stats(df: pd.DataFrame, label: str = "") -> str:
    if len(df) < 5:
        return f"{label}n={len(df):4d} (zu wenig)"
    r = df["ret"].astype(float)
    return (
        f"{label}n={len(df):4d}  mean={100*r.mean():+.3f}%  "
        f"win={100*(r > 0).mean():.1f}%  good>{100*GOOD_RET:.0f}%={100*(r > GOOD_RET).mean():.1f}%"
    )


def _compare_split(
    df: pd.DataFrame,
    col: str,
    hi_mask: pd.Series,
    title: str,
) -> dict[str, object] | None:
    """hi_mask=True = vermutlich 'besseres' Sub-Regime innerhalb rot."""
    tune = df[df["dataset"].isin(["META", "THRESHOLD"])]
    final = df[df["dataset"] == "FINAL"]
    out: dict[str, object] = {"title": title, "col": col}

    def _delta(sub: pd.DataFrame) -> tuple[float, int, int]:
        if sub.empty:
            return float("nan"), 0, 0
        hi = sub[hi_mask.loc[sub.index]]
        lo = sub[~hi_mask.loc[sub.index]]
        if len(hi) < MIN_N or len(lo) < MIN_N:
            return float("nan"), len(hi), len(lo)
        return float(hi["ret"].mean() - lo["ret"].mean()), len(hi), len(lo)

    d_tune, n_hi_t, n_lo_t = _delta(tune)
    d_fin, n_hi_f, n_lo_f = _delta(final)
    out.update(
        {
            "delta_tune": d_tune,
            "delta_final": d_fin,
            "n_hi_tune": n_hi_t,
            "n_lo_tune": n_lo_t,
            "n_hi_final": n_hi_f,
            "n_lo_final": n_lo_f,
        }
    )
    oos_ok = (
        np.isfinite(d_tune)
        and np.isfinite(d_fin)
        and d_tune * d_fin > 0
        and abs(d_tune) >= 0.001
    )
    out["oos_ok"] = bool(oos_ok)
    return out


def _print_split(df: pd.DataFrame, col: str, bins, labels: list[str], title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)
    for src_name, sub in [("META+THR", df[df["dataset"].isin(["META", "THRESHOLD"])]), ("FINAL", df[df["dataset"] == "FINAL"])]:
        d = sub.dropna(subset=[col, "ret"]).copy()
        print(f"  [{src_name}] rot-Signale n={len(d)}")
        if d.empty:
            continue
        d["_b"] = pd.cut(d[col], bins=bins, labels=labels, right=False)
        for lab in labels:
            g = d[d["_b"] == lab]
            print(f"    {lab:18} {_stats(g, '  ')}")


def main() -> None:
    if not SIGNALS_CSV.is_file():
        raise SystemExit(f"Fehlt: {SIGNALS_CSV}")

    sig = pd.read_csv(SIGNALS_CSV)
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["ret"] = pd.to_numeric(sig["ret"], errors="coerce")
    sig["vix"] = pd.to_numeric(sig["vix"], errors="coerce")
    sig["dist_high"] = pd.to_numeric(sig["dist_high"], errors="coerce")
    sig = sig.dropna(subset=["ret", "vix"])

    # Yahoo VIX / VIX3M
    print("Lade ^VIX und ^VIX3M …", flush=True)
    vix_s = _load_vix_series("^VIX")
    vix3m_s = _load_vix_series("^VIX3M")
    z_map: dict[pd.Timestamp, float] = {}
    ratio_map: dict[pd.Timestamp, float] = {}
    for d in sig["Date"].unique():
        td = pd.Timestamp(d)
        z_map[td] = _vix_z_vs_20d(vix_s, td)
        v = vix_s.asof(td) if len(vix_s) else np.nan
        v3 = vix3m_s.asof(td) if len(vix3m_s) else np.nan
        if np.isfinite(v) and np.isfinite(v3) and v > 0:
            ratio_map[td] = float(v3 / v)
        else:
            ratio_map[td] = np.nan
    sig["vix_z"] = sig["Date"].map(z_map)
    sig["vix3m_div_vix"] = sig["Date"].map(ratio_map)

    if MASTER_CSV.is_file():
        mc = pd.read_csv(MASTER_CSV)
        mc["Date"] = pd.to_datetime(mc["Date"]).dt.normalize()
        mc["ticker"] = mc["ticker"].astype(str).str.strip()
        extra_cols = [
            c
            for c in mc.columns
            if c
            not in {
                "ticker",
                "Date",
                "company",
                "sector",
                "classification_standard",
                "gics_sector",
                "gics_industry",
                "gics_sector_key",
                "gics_industry_key",
            }
        ]
        sig = sig.merge(
            mc[["ticker", "Date"] + extra_cols],
            on=["ticker", "Date"],
            how="left",
            suffixes=("", "_mc"),
        )
        print(f"Merge master_complete: {sig[extra_cols[0]].notna().sum() if extra_cols else 0} Treffer", flush=True)

    red = sig[sig["vix"] < VIX_AMPEL_YELLOW].copy()
    pool_tune = sig[sig["dataset"].isin(["META", "THRESHOLD"])]
    pool_fin = sig[sig["dataset"] == "FINAL"]

    print("\n" + "=" * 72)
    print(f"ROT (VIX < {VIX_AMPEL_YELLOW:.0f}) — Baseline")
    print("=" * 72)
    print(_stats(red, "ALLE: "))
    print(_stats(pool_tune[pool_tune["vix"] < VIX_AMPEL_YELLOW], "META+THR: "))
    print(_stats(pool_fin[pool_fin["vix"] < VIX_AMPEL_YELLOW], "FINAL:   "))

  # Anteil „guter“ Signale in rot vs. gelb/grün
    for name, sub in [("META+THR", pool_tune), ("FINAL", pool_fin)]:
        r = sub[sub["vix"] < VIX_AMPEL_YELLOW]
        g = sub[sub["vix"] >= VIX_AMPEL_YELLOW]
        if len(r) and len(g):
            print(
                f"\n  [{name}] good>{100*GOOD_RET:.0f}%:  rot={100*(r['ret']>GOOD_RET).mean():.1f}%  "
                f"nicht-rot={100*(g['ret']>GOOD_RET).mean():.1f}%"
            )

    candidates: list[dict[str, object]] = []

    # 1) VIX absolut innerhalb rot
    _print_split(
        red,
        "vix",
        bins=[0, 14, 17, VIX_AMPEL_YELLOW],
        labels=["<14", "14-17", "17-20"],
        title="VIX-Level innerhalb rot",
    )

    # 2) VIX-Z
    _print_split(
        red,
        "vix_z",
        bins=[-99, 0, 0.5, 99],
        labels=["z<0 (unter 20d-Mittel)", "0<=z<0.5", "z>=0.5"],
        title="VIX-Z (20 Handelstage)",
    )
    candidates.append(
        _compare_split(
            red.dropna(subset=["vix_z"]),
            "vix_z",
            red["vix_z"].fillna(0) >= 0,
            "VIX-Z >= 0 vs < 0 (innerhalb rot)",
        )
    )

    # 3) dist_from_high
    _print_split(
        red.dropna(subset=["dist_high"]),
        "dist_high",
        bins=[-1.0, -0.10, -0.05, -0.02, 0.01],
        labels=["<-10%", "-10..-5%", "-5..-2%", "-2..0%"],
        title="Abstand zum 20d-Hoch (dist_high)",
    )
    candidates.append(
        _compare_split(
            red.dropna(subset=["dist_high"]),
            "dist_high",
            red["dist_high"] >= -0.05,
            "dist_high >= -5% (nicht weit unter Hoch)",
        )
    )
    candidates.append(
        _compare_split(
            red.dropna(subset=["dist_high"]),
            "dist_high",
            red["dist_high"] >= -0.02,
            "dist_high >= -2%",
        )
    )

    # 4) VIX3M / VIX
    if red["vix3m_div_vix"].notna().any():
        med = float(red["vix3m_div_vix"].median())
        _print_split(
            red.dropna(subset=["vix3m_div_vix"]),
            "vix3m_div_vix",
            bins=[0, med, 10],
            labels=[f"<=Med({med:.2f})", f">Med"],
            title="Term-Structure VIX3M/VIX",
        )
        candidates.append(
            _compare_split(
                red.dropna(subset=["vix3m_div_vix"]),
                "vix3m_div_vix",
                red["vix3m_div_vix"] >= med,
                f"VIX3M/VIX >= Median ({med:.2f})",
            )
        )

    # 5) master_complete Spalten (nur wo gemerged — schwerpunkt FINAL)
    mc_cols = [
        ("regime_vix_z_20d", [-99, 0, 0.5, 99], ["z<0", "0-0.5", "z>=0.5"]),
        ("regime_vix_ret_5d", [-99, 0, 99], ["VIX-5d fallend", "VIX-5d steigend"]),
        ("regime_vix_ret_1d", [-99, 0, 99], ["VIX-1d fallend", "VIX-1d steigend"]),
        ("dist_from_20d_high_pct", [-1, -0.05, -0.02, 0.01], ["<-5%", "-5..-2%", "-2..0%"]),
        ("ret_vs_spy_5d", [-99, 0, 99], ["vs SPY 5d schwach", "vs SPY 5d stark"]),
        ("volume_zscore_20d", [-99, 0, 99], ["Vol-Z niedrig", "Vol-Z hoch"]),
        ("prob", [0, 0.75, 0.85, 1.01], ["prob<0.75", "0.75-0.85", ">=0.85"]),
    ]
    for col, bins, labels in mc_cols:
        if col not in red.columns:
            continue
        sub = red.dropna(subset=[col, "ret"])
        if len(sub) < MIN_N:
            continue
        _print_split(sub, col, bins=bins, labels=labels, title=f"master_complete: {col}")

        if col == "regime_vix_z_20d":
            candidates.append(
                _compare_split(sub, col, sub[col] >= 0, f"{col} >= 0 (innerhalb rot)")
            )
        if col == "prob":
            candidates.append(
                _compare_split(sub, col, sub[col] >= 0.80, f"prob >= 0.80 (innerhalb rot, nur mit MC)")
            )
        if col == "regime_vix_ret_5d":
            candidates.append(
                _compare_split(sub, col, sub[col] >= 0, "VIX steigt über 5d (regime_vix_ret_5d>=0)")
            )

    # Kombinationen (nur wenn Einzeltests nahelegen)
    if red["vix_z"].notna().any() and red["dist_high"].notna().any():
        m = (red["vix"] < 17) & (red["vix_z"] >= 0) & (red["dist_high"] >= -0.05)
        print("\n" + "-" * 72)
        print("KOMBI (explorativ): VIX<17 & vix_z>=0 & dist_high>=-5%")
        print("-" * 72)
        for name, sub in [("META+THR", pool_tune), ("FINAL", pool_fin)]:
            r = sub[sub["vix"] < VIX_AMPEL_YELLOW]
            g = r[m.loc[r.index]]
            b = r[~m.loc[r.index]]
            print(f"  [{name}] SUB-GUT:  {_stats(g)}")
            print(f"  [{name}] SUB-REST: {_stats(b)}")

    print("\n" + "=" * 72)
    print("SPLIT-RICHTUNG: hi-Bucket minus lo-Bucket (mean ret), OOS = gleiches Vorzeichen META+THR & FINAL")
    print("=" * 72)
    rows = [c for c in candidates if c]
    rows.sort(key=lambda x: (not x.get("oos_ok", False), -abs(float(x.get("delta_tune") or 0))))
    for r in rows:
        mark = "OOS-OK" if r.get("oos_ok") else "------"
        print(
            f"  [{mark}] {r['title'][:55]:55}  "
            f"d_tune={100*float(r['delta_tune'] or 0):+.2f}pp (n_hi/lo={r['n_hi_tune']}/{r['n_lo_tune']})  "
            f"d_fin={100*float(r['delta_final'] or 0):+.2f}pp (n_hi/lo={r['n_hi_final']}/{r['n_lo_final']})"
        )

    print("\n" + "=" * 72)
    print("FAZIT")
    print("=" * 72)
    ok = [r for r in rows if r.get("oos_ok")]
    if ok:
        print("  Auf FINAL bestaetigte Richtungen (Kandidaten fuer Kontext-Chips, nicht Ampel-Farben):")
        for r in ok[:8]:
            print(f"    - {r['title']}")
    else:
        print("  Kein einfacher Split innerhalb rot mit klarer FINAL-Bestaetigung in diesem Lauf.")
    print(
        "\n  Hinweis: Viele „gute“ Signale in rot bleiben erwartbar — rot beschreibt das Marktregime,\n"
        "  nicht die Signalqualitaet. Ziel ist Anreicherung, nicht rot eliminieren."
    )


if __name__ == "__main__":
    main()
