"""
Ampel v2 validieren: VIX-Unterbaender + VIX-Z auf META/THRESHOLD vs. FINAL.

Nutzt gespeicherte Signale aus data/_scratch_meta_thr_final_signals.csv
(VIX pro Signal aus Pipeline-Reproduktion). VIX-Z wird per ^VIX-Yahoo
am Signaltag nachgerechnet (gleiche Logik wie signal_extra_filters).

  python scripts/_scratch_validate_ampel_v2.py
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


def _load_vix_series() -> pd.Series:
    print("Lade ^VIX (Yahoo) für VIX-Z …", flush=True)
    raw = yf.download("^VIX", start="2014-01-01", progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit("Keine VIX-Daten.")
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"].squeeze()
    else:
        s = raw["Close"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def classify_ampel_v1(vix: float) -> str:
    if vix < 20:
        return "rot"
    if vix < 25:
        return "gelb"
    return "gruen"


def classify_ampel_v2(vix: float, vix_z: float, *, z_hi: float = 0.5) -> str:
    """Feinere Stufen (Kalibrierung aus explorativer Analyse)."""
    if vix >= 25:
        return "1_gruen"
    if vix >= 20:
        return "2_gelb"
    if vix >= 17:
        if np.isfinite(vix_z) and vix_z >= z_hi:
            return "3b_rot_oben_rel"
        return "3_rot_oben"
    if vix >= 14:
        return "4_rot_mitte"
    return "5_rot_tief"


def _report_block(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if df.empty:
        print("  (leer)")
        return
    r = df["ret"].astype(float)
    print(
        f"  GESAMT: n={len(df):4d}  mean={100*r.mean():+.3f}%  "
        f"median={100*r.median():+.3f}%  win={100*(r > 0).mean():.1f}%  "
        f"good(>2%)={100*(r > 0.02).mean():.1f}%"
    )


def _tier_table(df: pd.DataFrame, col: str, order: list[str]) -> None:
    for lab in order:
        g = df[df[col] == lab]
        if len(g) < 5:
            print(f"    {lab:22}: n={len(g):4d}  (zu wenig)")
            continue
        rr = g["ret"].astype(float)
        print(
            f"    {lab:22}: n={len(g):4d}  mean={100*rr.mean():+.3f}%  "
            f"median={100*rr.median():+.3f}%  win={100*(rr > 0).mean():.1f}%  "
            f"good={100*(rr > 0.02).mean():.1f}%"
        )


def main() -> None:
    if not SIGNALS_CSV.is_file():
        raise SystemExit(f"Fehlt {SIGNALS_CSV} — zuerst scripts/_scratch_vix_regime_meta_threshold.py laufen lassen.")

    sig = pd.read_csv(SIGNALS_CSV)
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["vix"] = pd.to_numeric(sig["vix"], errors="coerce")
    sig["ret"] = pd.to_numeric(sig["ret"], errors="coerce")
    sig = sig.dropna(subset=["ret", "vix"])

    vix_close = _load_vix_series()
    z_by_date: dict[pd.Timestamp, float] = {}
    for d in sig["Date"].unique():
        z_by_date[pd.Timestamp(d)] = _vix_z_vs_20d(vix_close, d)
    sig["vix_z"] = sig["Date"].map(z_by_date)
    n_z = sig["vix_z"].notna().sum()
    print(f"Signale: {len(sig):,}  |  VIX-Z berechnet: {n_z:,} ({100*n_z/len(sig):.1f}%)")

    sig["amp_v1"] = sig["vix"].map(classify_ampel_v1)
    sig["amp_v2"] = [
        classify_ampel_v2(v, z) for v, z in zip(sig["vix"], sig["vix_z"], strict=True)
    ]

    tune = sig[sig["dataset"].isin(["META", "THRESHOLD"])].copy()
    final = sig[sig["dataset"] == "FINAL"].copy()
    pooled_tune = tune

    v2_order = [
        "1_gruen",
        "2_gelb",
        "3_rot_oben",
        "3b_rot_oben_rel",
        "4_rot_mitte",
        "5_rot_tief",
    ]

    # --- Ampel v1 Referenz ---
    _report_block(pooled_tune, "META + THRESHOLD (Tuning) — Ampel v1")
    for lab in ["rot", "gelb", "gruen"]:
        _tier_table(pooled_tune, "amp_v1", [lab])

    _report_block(final, "FINAL (OOS-Check) — Ampel v1")
    for lab in ["rot", "gelb", "gruen"]:
        _tier_table(final, "amp_v1", [lab])

    # --- Ampel v2 ---
    _report_block(pooled_tune, "META + THRESHOLD (Tuning) — Ampel v2")
    _tier_table(pooled_tune, "amp_v2", v2_order)

    _report_block(final, "FINAL (OOS-Check) — Ampel v2")
    _tier_table(final, "amp_v2", v2_order)

    # --- Nur META (groesse Stichprobe) ---
    meta = sig[sig["dataset"] == "META"]
    _report_block(meta, "Nur META — Ampel v2")
    _tier_table(meta, "amp_v2", v2_order)

    # --- Rot-Unterbaender ohne Z (Kernhypothese) ---
    print("\n" + "=" * 72)
    print("ROT-UNTERTEILUNG (nur VIX absolut) — META+THR vs. FINAL")
    print("=" * 72)
    rot_bins = [
        ("rot_<14", lambda v: v < 14),
        ("rot_14_17", lambda v: (v >= 14) & (v < 17)),
        ("rot_17_20", lambda v: (v >= 17) & (v < 20)),
    ]
    for label, fn in rot_bins:
        print(f"\n  [{label}]")
        for name, sub in [("META+THR", pooled_tune), ("FINAL", final)]:
            rsub = sub[sub["amp_v1"] == "rot"]
            g = rsub[rsub["vix"].map(fn)]
            if len(g) < 5:
                print(f"    {name:8}: n={len(g)}")
                continue
            rr = g["ret"]
            print(
                f"    {name:8}: n={len(g):4d}  mean={100*rr.mean():+.3f}%  "
                f"win={100*(rr > 0).mean():.1f}%  good={100*(rr > 0.02).mean():.1f}%"
            )

    # --- Z-Schwelle Sensitivitaet (nur auf Tuning, nur rot 17-20) ---
    print("\n" + "=" * 72)
    print("SENSITIVITAET vix_z (nur rot 17-20) — nur META+THRESHOLD")
    print("=" * 72)
    r17 = pooled_tune[(pooled_tune["amp_v1"] == "rot") & (pooled_tune["vix"] >= 17)]
    for z_thr in [0.0, 0.3, 0.5, 0.8, 1.0]:
        lo = r17[r17["vix_z"] < z_thr]
        hi = r17[r17["vix_z"] >= z_thr]
        if len(lo) >= 5:
            print(
                f"  z<{z_thr}: n={len(lo):3d} mean={100*lo.ret.mean():+.2f}% "
                f"win={100*(lo.ret>0).mean():.0f}%"
            )
        if len(hi) >= 5:
            print(
                f"  z>={z_thr}: n={len(hi):3d} mean={100*hi.ret.mean():+.2f}% "
                f"win={100*(hi.ret>0).mean():.0f}%"
            )

    print("\n" + "=" * 72)
    print("FAZIT-KRITERIEN")
    print("=" * 72)
    print("  Ampel v2 ist nutzbar wenn auf FINAL dieselbe ORDNUNG gilt wie META+THR:")
    print("  1_gruen > 2_gelb > 4_rot_mitte > 3_rot_oben / 5_rot_tief")
    print("  (exakte Werte duerfen abweichen — Richtung muss stimmen)")


if __name__ == "__main__":
    main()
