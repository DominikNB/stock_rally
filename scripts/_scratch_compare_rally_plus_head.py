"""Einmal-Skript: rally_plus_entry — target vs rally für wenige Ticker / HEAD_FRACTION."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import _create_target_one_ticker_fixed_bands

TICKERS = ["MSFT", "NVDA", "SAP.DE"]
START = "2022-01-01"
END = "2024-06-01"
FRACS = [0.10, 0.40, 0.99]

if cfg.OPT_OPTIMIZE_Y_TARGETS:
    raise SystemExit("OPT_OPTIMIZE_Y_TARGETS muss False sein für feste Band-Regel.")

_orig_label = cfg.FIXED_Y_LABEL_MODE
_orig_head = cfg.FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION
_orig_ov = cfg.FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE

cfg.FIXED_Y_LABEL_MODE = "rally_plus_entry"
cfg.FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE = "union"

print("Download", TICKERS, START, "…", END)
df = load_stock_data(tickers=TICKERS, start=START, end=END)
if df is None or len(df) == 0:
    raise SystemExit("Keine Daten")

print(
    f"FIXED_Y: w=[{cfg.FIXED_Y_WINDOW_MIN},{cfg.FIXED_Y_WINDOW_MAX}] "
    f"rt={cfg.FIXED_Y_RALLY_THRESHOLD} max_dip={cfg.FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION} "
    f"n_pre={cfg.FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS} overlap={cfg.FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE}\n"
)

for tk in TICKERS:
    sub = df[df["ticker"] == tk].sort_values("Date").reset_index(drop=True)
    if len(sub) < 200:
        print(f"{tk}: zu wenig Zeilen ({len(sub)}), überspringe")
        continue
    print(f"=== {tk}  n={len(sub)} ===")
    for f in FRACS:
        cfg.FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION = f
        rally, target = _create_target_one_ticker_fixed_bands(sub)
        r1 = float(rally.mean())
        t1 = float(target.mean())
        n_r = int(rally.sum())
        n_t = int(target.sum())
        rally_not_target = int(((rally == 1) & (target == 0)).sum())
        print(
            f"  HEAD_FRACTION={f:.2f}  mean(rally)={r1:.4f}  mean(target)={t1:.4f}  "
            f"#rally={n_r}  #target={n_t}  #rally&~target={rally_not_target}"
        )

cfg.FIXED_Y_LABEL_MODE = _orig_label
cfg.FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION = _orig_head
cfg.FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE = _orig_ov
print("cfg FIXED_Y_* wiederhergestellt.")
