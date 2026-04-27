from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _metrics(df: pd.DataFrame, name: str) -> dict[str, float]:
    n = float(len(df))
    if n == 0:
        return {
            "variant": name,
            "n_signals": 0.0,
            "win_rate_ret4": np.nan,
            "mean_ret4": np.nan,
            "mean_ret6": np.nan,
            "mean_ret8": np.nan,
            "mean_ret_mean5": np.nan,
            "days_covered": 0.0,
        }
    ret4 = _safe_num(df, "ret_4d")
    ret6 = _safe_num(df, "ret_6d")
    ret8 = _safe_num(df, "ret_8d")
    retm = _safe_num(df, "ret_mean_5")
    d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return {
        "variant": name,
        "n_signals": n,
        "win_rate_ret4": float((ret4 > 0).mean()),
        "mean_ret4": float(ret4.mean()),
        "mean_ret6": float(ret6.mean()),
        "mean_ret8": float(ret8.mean()),
        "mean_ret_mean5": float(retm.mean()),
        "days_covered": float(d.nunique()),
    }


def main() -> None:
    p = Path("data/master_complete.csv")
    if not p.is_file():
        raise SystemExit(f"Missing file: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise SystemExit("master_complete.csv is empty.")

    # Core fields used by variants
    for c in ("bb_width_20", "ret_1d_signal_day", "volume_zscore_20d", "ret_vs_sector_5d"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    baseline = df.copy()

    # Dynamic squeeze threshold (cross-sectional, robust in current sample)
    bw_thr = float(df["bb_width_20"].quantile(0.30)) if "bb_width_20" in df.columns else np.nan

    # Variant B: global squeeze + positive day breakout
    var_b = df.copy()
    if np.isfinite(bw_thr):
        var_b = var_b[(var_b["bb_width_20"] <= bw_thr) & (var_b["ret_1d_signal_day"] > 0.0)]
    else:
        var_b = var_b.iloc[0:0]

    # Variant C: B + volume confirmation
    var_c = var_b.copy()
    if "volume_zscore_20d" in var_c.columns:
        var_c = var_c[var_c["volume_zscore_20d"] > 0.5]
    else:
        var_c = var_c.iloc[0:0]

    # Variant D: C + relative strength vs sector
    var_d = var_c.copy()
    if "ret_vs_sector_5d" in var_d.columns:
        var_d = var_d[var_d["ret_vs_sector_5d"] > 0.0]
    else:
        var_d = var_d.iloc[0:0]

    # Variant E: ticker-relative squeeze (<= own 30% quantile) + up day
    var_e = df.copy()
    if "bb_width_20" in var_e.columns:
        _q = var_e.groupby("ticker")["bb_width_20"].transform(lambda s: s.quantile(0.30))
        var_e = var_e[(var_e["bb_width_20"] <= _q) & (var_e["ret_1d_signal_day"] > 0.0)]
    else:
        var_e = var_e.iloc[0:0]

    # Variant F: E + volume confirmation
    var_f = var_e.copy()
    if "volume_zscore_20d" in var_f.columns:
        var_f = var_f[var_f["volume_zscore_20d"] > 0.5]
    else:
        var_f = var_f.iloc[0:0]

    # Variant G: F + relative strength vs sector
    var_g = var_f.copy()
    if "ret_vs_sector_5d" in var_g.columns:
        var_g = var_g[var_g["ret_vs_sector_5d"] > 0.0]
    else:
        var_g = var_g.iloc[0:0]

    rows = [
        _metrics(baseline, "A_baseline"),
        _metrics(var_b, "B_squeeze+upday"),
        _metrics(var_c, "C_B+volume_z>0.5"),
        _metrics(var_d, "D_C+ret_vs_sector_5d>0"),
        _metrics(var_e, "E_ticker_q30+upday"),
        _metrics(var_f, "F_E+volume_z>0.5"),
        _metrics(var_g, "G_F+ret_vs_sector_5d>0"),
    ]
    out = pd.DataFrame(rows)

    print("Squeeze variant comparison (on data/master_complete.csv)\n")
    print(f"Rows total: {len(df):,} | Date span: {df['Date'].min()} .. {df['Date'].max()}")
    print(f"bb_width_20 30% quantile threshold: {bw_thr:.8f}" if np.isfinite(bw_thr) else "bb_width_20 missing")
    print("\nMetrics:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

