"""Kontext-Ampel auf docs/signals.json neu berechnen (master_complete vix3m merge)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.signal_context_tier import attach_signal_context_tier, context_tier_counts


def main() -> None:
    signals_path = ROOT / "docs" / "signals.json"
    master_path = ROOT / "data" / "master_complete.csv"
    data = json.loads(signals_path.read_text(encoding="utf-8"))
    signals = list(data.get("signals") or [])
    if not signals:
        raise SystemExit(f"Keine Signale in {signals_path}")

    ratio_map: dict[tuple[str, str], float] = {}
    if master_path.is_file():
        mc = pd.read_csv(master_path, usecols=["ticker", "Date", "vix3m_vix_ratio"], low_memory=False)
        mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for _, r in mc.dropna(subset=["Date"]).iterrows():
            v = pd.to_numeric(r.get("vix3m_vix_ratio"), errors="coerce")
            if pd.notna(v):
                ratio_map[(str(r["ticker"]), str(r["Date"])[:10])] = float(v)

    for sig in signals:
        key = (str(sig.get("ticker", "")), str(sig.get("date", ""))[:10])
        if key in ratio_map:
            sig["vix3m_vix_ratio"] = ratio_map[key]
        attach_signal_context_tier(sig)

    data["signals"] = signals
    signals_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = context_tier_counts(signals)
    print(
        f"OK {signals_path}: "
        f"grün={counts['green']} orange={counts['orange']} "
        f"gelb={counts['yellow_plain']}+{counts['yellow_risk']} rot={counts['red']}"
    )


if __name__ == "__main__":
    main()
