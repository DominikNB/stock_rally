from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def main() -> None:
    sig_raw = json.loads(Path("docs/signals.json").read_text(encoding="utf-8"))
    sig = pd.DataFrame(sig_raw if isinstance(sig_raw, list) else sig_raw.get("signals", []))
    sig["date"] = pd.to_datetime(sig["date"], errors="coerce")

    mc = pd.read_csv("data/master_complete.csv")
    mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce")
    for c in ["ret_4d", "dist_from_20d_high_pct", "momentum_20d", "runup_3d", "runup_5d"]:
        if c in mc.columns:
            mc[c] = pd.to_numeric(mc[c], errors="coerce")
        else:
            mc[c] = np.nan

    j = sig.merge(
        mc[
            [
                "ticker",
                "Date",
                "ret_4d",
                "dist_from_20d_high_pct",
                "momentum_20d",
                "runup_3d",
                "runup_5d",
            ]
        ],
        left_on=["ticker", "date"],
        right_on=["ticker", "Date"],
        how="left",
    )

    # Fallback: if runup columns are missing in current master snapshot, compute true
    # runup_3d/runup_5d from yfinance closes per ticker/date.
    j["r3"] = j["runup_3d"]
    j["r5"] = j["runup_5d"]
    miss = j["r3"].isna() | j["r5"].isna()
    if miss.any():
        tickers = sorted(j.loc[miss, "ticker"].dropna().astype(str).unique().tolist())
        dmin = pd.Timestamp(j.loc[miss, "date"].min()) - pd.Timedelta(days=20)
        dmax = pd.Timestamp(j.loc[miss, "date"].max()) + pd.Timedelta(days=3)
        close_map: dict[str, pd.Series] = {}
        for t in tickers:
            try:
                df = yf.download(
                    t,
                    start=dmin.strftime("%Y-%m-%d"),
                    end=dmax.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if df is None or len(df) == 0:
                    continue
                c = df["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
                close_map[t] = c.astype(float).sort_index()
            except Exception:
                continue
        r3_vals = []
        r5_vals = []
        for _, r in j.iterrows():
            t = str(r["ticker"])
            d = pd.Timestamp(r["date"]).normalize()
            s = close_map.get(t)
            rv3 = r["r3"]
            rv5 = r["r5"]
            if s is not None and (not np.isfinite(rv3) or not np.isfinite(rv5)):
                sub = s[s.index <= d]
                if len(sub) >= 6:
                    c0 = float(sub.iloc[-1])
                    c3 = float(sub.iloc[-4])
                    c5 = float(sub.iloc[-6])
                    if np.isfinite(c0) and np.isfinite(c3) and c3 != 0.0:
                        rv3 = c0 / c3 - 1.0
                    if np.isfinite(c0) and np.isfinite(c5) and c5 != 0.0:
                        rv5 = c0 / c5 - 1.0
            r3_vals.append(rv3)
            r5_vals.append(rv5)
        j["r3"] = pd.to_numeric(pd.Series(r3_vals), errors="coerce")
        j["r5"] = pd.to_numeric(pd.Series(r5_vals), errors="coerce")
    # final safety fallback
    j["r3"] = j["r3"].where(j["r3"].notna(), 0.15 * j["momentum_20d"])
    j["r5"] = j["r5"].where(j["r5"].notna(), 0.25 * j["momentum_20d"])

    base_ready = j["ret_4d"].notna()
    print(
        "BASE",
        f"n={len(j)}",
        f"ready={int(base_ready.sum())}",
        f"hit={float((j.loc[base_ready, 'ret_4d'] > 0).mean()):.4f}",
        f"mean={float(j.loc[base_ready, 'ret_4d'].mean()):.5f}",
    )

    tests = [
        (0.05, 0.08, -0.01),
        (0.06, 0.10, -0.01),
        (0.07, 0.12, -0.005),
        (0.08, 0.14, 0.0),
    ]
    rows = []
    for t3, t5, dh in tests:
        veto = ((j["r3"] > t3) | (j["r5"] > t5)) & (j["dist_from_20d_high_pct"] > dh)
        k = j.loc[~veto].copy()
        ready = k["ret_4d"].notna()
        rows.append(
            {
                "t3": t3,
                "t5": t5,
                "dist20h_gt": dh,
                "kept": len(k),
                "dropped": int(veto.sum()),
                "ready": int(ready.sum()),
                "hit": float((k.loc[ready, "ret_4d"] > 0).mean()) if ready.any() else np.nan,
                "mean_ret4": float(k.loc[ready, "ret_4d"].mean()) if ready.any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    print("\nRUNUP VETO SCENARIOS")
    print(out.to_string(index=False))

    recent = j.sort_values("date", ascending=False).head(15).copy()
    rdy = recent["ret_4d"].notna()
    print(
        "\nTOP15 baseline",
        f"ready={int(rdy.sum())}",
        f"hit={float((recent.loc[rdy, 'ret_4d'] > 0).mean()) if rdy.any() else np.nan}",
        f"mean={float(recent.loc[rdy, 'ret_4d'].mean()) if rdy.any() else np.nan}",
    )
    for t3, t5, dh in tests[:2]:
        veto = ((recent["r3"] > t3) | (recent["r5"] > t5)) & (recent["dist_from_20d_high_pct"] > dh)
        k = recent.loc[~veto]
        rr = k["ret_4d"].notna()
        print(
            f"TOP15 t3={t3} t5={t5} dh={dh}: "
            f"kept={len(k)} dropped={int(veto.sum())} ready={int(rr.sum())} "
            f"hit={(float((k.loc[rr, 'ret_4d'] > 0).mean()) if rr.any() else np.nan)} "
            f"mean={(float(k.loc[rr, 'ret_4d'].mean()) if rr.any() else np.nan)}"
        )

    car = recent[recent["ticker"].astype(str).str.upper().eq("CARR")]
    if car.empty:
        car = j[j["ticker"].astype(str).str.upper().eq("CARR")].sort_values("date", ascending=False).head(3)
    print(f"\nCARRIER rows found: {len(car)}")
    if len(car):
        cols = ["ticker", "date", "prob", "r3", "r5", "dist_from_20d_high_pct", "ret_4d"]
        print(car[cols].to_string(index=False))


if __name__ == "__main__":
    main()

