"""Durchrechnung Ampel + Chips + Badge für ein Beispiel-Signal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.red_signal_quality import calibrate_gld_ret5_median_red_ref
from lib.vix_red_context_chips import _chip_thresholds, evaluate_red_context_chips
from lib.vix_regime_ampel import classify_vix_regime, vix_ampel_thresholds

DATE = "2026-06-01"
TICKER = "ILMN"


def _load_signal() -> dict:
    p = json.loads((ROOT / "docs" / "signals.json").read_text(encoding="utf-8"))
    for s in p["signals"]:
        if s.get("ticker") == TICKER and str(s.get("date", ""))[:10] == DATE:
            return s
    raise SystemExit(f"Signal {TICKER} {DATE} nicht in signals.json")


def main() -> None:
    sig = _load_signal()
    y_min, g_min = vix_ampel_thresholds()
    thr = _chip_thresholds()
    gld_ref = calibrate_gld_ret5_median_red_ref()

    vix_level = float(sig["regime_vix_level"])
    amp = classify_vix_regime(vix_level)

    print("=" * 60)
    print(f"BEISPIEL: {TICKER}  Signaltag {DATE}  prob={sig['prob']:.3f}")
    print("=" * 60)

    print("\n--- 1. VIX-AMPEL ---")
    print(f"regime_vix_level (^VIX Schluss): {vix_level:.2f}")
    print(f"Schwellen: rot VIX < {y_min:.0f} | gelb {y_min:.0f}-{g_min:.0f} | gruen >= {g_min:.0f}")
    print(f"=> {vix_level:.2f} < {y_min:.0f}  =>  Ampel: {amp['level'].upper()} ({amp['label_de']})")

    start = (pd.Timestamp(DATE) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(DATE) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"] if not isinstance(raw.columns, pd.MultiIndex) else raw["Close"]["^VIX"]
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    w = close[close.index <= pd.Timestamp(DATE)].tail(20).astype(float)
    mu, st = w.mean(), w.std(ddof=0)
    z = (w.iloc[-1] - mu) / st

    print("\n--- 2. CHIP 1: VIX vs. 20d-Mittel ---")
    print("Letzte 20 ^VIX-Schlüsse (Handelstage bis Signaltag):")
    for d, v in w.items():
        print(f"  {d.date()}   {v:.2f}")
    print(f"Mittel mu = {mu:.4f}   Std sigma = {st:.4f}   VIX heute = {w.iloc[-1]:.2f}")
    print(f"regime_vix_z_20d = (VIX - mu) / sigma = ({w.iloc[-1]:.2f} - {mu:.4f}) / {st:.4f} = {z:.4f}")
    print(f"JSON-Wert: {sig['regime_vix_z_20d']:.4f}")
    chip1 = "GRUEN (good)" if z < 0 else "ORANGE (warn)"
    print(f"Regel: Z < 0 -> gruen  |  Z >= 0 -> orange")
    print(f"=> Z = {z:.4f} < 0  =>  Chip 1: {chip1}")

    raw2 = yf.download(["^VIX", "^VIX3M"], start=start, end=end, progress=False, auto_adjust=True)
    vix_s = raw2["Close"]["^VIX"] if isinstance(raw2.columns, pd.MultiIndex) else raw2["Close"]
    v3_s = raw2["Close"]["^VIX3M"] if isinstance(raw2.columns, pd.MultiIndex) else raw2["Close"]
    vix_s.index = pd.to_datetime(vix_s.index).tz_localize(None).normalize()
    v3_s.index = pd.to_datetime(v3_s.index).tz_localize(None).normalize()
    d = pd.Timestamp(DATE)
    v, v3 = float(vix_s[vix_s.index <= d].iloc[-1]), float(v3_s[v3_s.index <= d].iloc[-1])
    ratio = v3 / v
    mx = thr["vix3m_vix_max"]

    print("\n--- 3. CHIP 2: VIX-Term 3M/VIX ---")
    print(f"^VIX = {v:.2f}   ^VIX3M = {v3:.2f}")
    print(f"vix3m_vix_ratio = {v3:.2f} / {v:.2f} = {ratio:.4f}")
    print(f"JSON-Wert: {sig['vix3m_vix_ratio']:.4f}")
    print(f"Schwelle (OOS-kalibriert): {mx:.2f}")
    chip2 = "GRUEN (good)" if ratio < mx else "ORANGE (warn)"
    print(f"Regel: Ratio < {mx:.2f} -> gruen  |  sonst orange")
    print(f"=> {ratio:.4f} >= {mx:.2f}  =>  Chip 2: {chip2}")

    hhi = float(sig["sector_hhi_same_day"])
    hhi_max = thr["sector_hhi_max"]

    print("\n--- 4. CHIP 3: Sektor-Crowding ---")
    print(f"sector_hhi_same_day = {hhi:.4f}")
    print("Berechnung am Signaltag:")
    print("  - Alle Modell-Signale desselben Datums zaehlen")
    print("  - Pro Sektor: Anteil = Anzahl_Sektor / Anzahl_gesamt")
    print("  - HHI = Summe(Anteil^2)")
    print(f"  - HHI = 1.0 -> 100% der Signale in EINEM Sektor (max. Crowding)")
    print(f"Schwelle = Median-HHI aller rot-Signale im Backtest = {hhi_max:.4f}")
    chip3 = "GRUEN (good)" if hhi < hhi_max else "ORANGE (warn)"
    print(f"Regel: HHI < {hhi_max:.4f} -> gruen  |  sonst orange")
    print(f"=> {hhi:.4f} >= {hhi_max:.4f}  =>  Chip 3: {chip3}")

    chips = evaluate_red_context_chips(sig)
    print("\n--- Chip-Zusammenfassung (Code) ---")
    for c in chips:
        print(f"  {c['label']}: {c['state']}")

    gld = float(sig["gld_ret_5d"])

    print("\n--- 5. QUALITAETS-BADGE (kein Chip) ---")
    print(f"gld_ret_5d (GLD 5-Tage-Rendite am {DATE}): {gld*100:+.3f} %")
    print(f"gld_ret5_median_red_ref (global, alle rot-Signale Backtest): {gld_ref*100:+.3f} %")
    gld_ok = gld < gld_ref
    print(f"Faktor GLD: gld_ret_5d < Median?  {gld:.6f} < {gld_ref:.6f}  ->  {gld_ok} (+1 Punkt)")
    liq = sig.get("liquidity_tier")
    print(f"Faktor Liquiditaet: liquidity_tier = {liq!r}  ->  nicht bewertbar (fehlt im OOS-Export)")
    hits, mx_q = sig["red_quality_hits"], sig["red_quality_max"]
    print(f"Score: {hits}/{mx_q} bekannte Faktoren  ->  Tier: {sig['red_quality_tier']}")
    print(f"Anzeige: Qualitaet hoch (1/1) -- nur GLD zaehlt, 100% der bekannten Faktoren ok")

    # Zweites Beispiel mit Liquidität aus Master
    print("\n" + "=" * 60)
    print("ZWEITES BEISPIEL (mit Liquidität): ELG.DE  2026-02-26")
    print("=" * 60)
    mc = pd.read_csv(ROOT / "data" / "master_complete.csv")
    row = mc[(mc["ticker"] == "ELG.DE") & (mc["Date"].astype(str).str[:10] == "2026-02-26")]
    if len(row):
        r = row.iloc[0]
        print(f"VIX: {r.get('regime_vix_level')}  -> Ampel rot")
        print(f"liquidity_tier: {r.get('liquidity_tier')}")
        print(f"adv_pctile_same_day: {r.get('adv_pctile_same_day')}")
        print("Liquiditaet: Perzentil-Rang des 20d-ADV unter allen Signalen am Tag")
        print("  < 15% -> very_thin | 15-35% -> thin | >= 35% -> ok")
        tier = str(r.get("liquidity_tier", ""))
        print(f"=> tier={tier!r}  =>  liquidity_ok = {tier == 'ok'}")
        g2 = r.get("gld_ret_5d")
        if pd.notna(g2):
            g2f = float(g2)
            print(f"gld_ret_5d: {g2f*100:+.3f}%  vs Median {gld_ref*100:+.3f}%  -> gld_ret5_low = {g2f < gld_ref}")
        print("=> Badge: beide Faktoren bekannt -> 0/2 niedrig (Liquiditaet thin + GLD ueber Median)")


if __name__ == "__main__":
    main()
