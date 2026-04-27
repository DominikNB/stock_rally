from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fold_bounds(dates: pd.Series, n_folds: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Index(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    chunks = np.array_split(u, n_folds + 1)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(1, len(chunks)):
        if len(chunks[i]) == 0:
            continue
        out.append((pd.Timestamp(chunks[i][0]), pd.Timestamp(chunks[i][-1])))
    return out


def _metrics(ret: pd.Series) -> dict:
    r = _num(ret).dropna()
    if r.empty:
        return {"n": 0, "hit": np.nan, "mean": np.nan}
    return {"n": int(len(r)), "hit": float((r > 0).mean()), "mean": float(r.mean())}


def main() -> None:
    df = pd.read_csv("data/master_complete.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("ret_2d", "ret_4d"):
        df[c] = _num(df[c])
    df = df.dropna(subset=["Date", "ret_2d", "ret_4d"]).sort_values("Date").reset_index(drop=True)

    # Strategy A: baseline immediate entry, evaluate ret_4d
    df["ret_A_immediate"] = df["ret_4d"]

    # Strategy B: wait 2 days, then only participate if early momentum positive
    # Proxy continuation from day2-close to day4-close:
    # (1+ret_4d)/(1+ret_2d)-1
    # (approximation for delayed entry due to unavailable day2-open aligned returns)
    cont = (1.0 + df["ret_4d"]) / (1.0 + df["ret_2d"]) - 1.0
    df["ret_B_wait2_confirm"] = np.where(df["ret_2d"] > 0, cont, np.nan)

    # Strategy C: immediate entry + day2 risk management:
    # if weak after 2 days -> exit at day2; else hold to day4
    df["ret_C_manage_day2"] = np.where(df["ret_2d"] <= 0, df["ret_2d"], df["ret_4d"])

    folds = _fold_bounds(df["Date"], n_folds=5)
    lines: list[str] = []
    lines.append("# ret_2d Timing Strategy Test")
    lines.append("A=immediate_entry(ret_4d)")
    lines.append("B=wait_2d_then_confirm(ret2>0), return proxy=(1+ret4)/(1+ret2)-1")
    lines.append("C=immediate_entry_with_day2_risk_management")
    lines.append("")

    agg = {"A": [], "B": [], "C": []}
    for i, (s, e) in enumerate(folds, 1):
        te = df[(df["Date"] >= s) & (df["Date"] <= e)].copy()
        if len(te) < 60:
            continue
        a = _metrics(te["ret_A_immediate"])
        b = _metrics(te["ret_B_wait2_confirm"])
        c = _metrics(te["ret_C_manage_day2"])
        agg["A"].append(a)
        agg["B"].append(b)
        agg["C"].append(c)
        lines.append(f"## Fold {i} [{s.date()} .. {e.date()}]")
        lines.append(f"- A: n={a['n']} hit={a['hit']:.3f} mean={a['mean']:.4f}")
        lines.append(f"- B: n={b['n']} hit={b['hit']:.3f} mean={b['mean']:.4f}")
        lines.append(f"- C: n={c['n']} hit={c['hit']:.3f} mean={c['mean']:.4f}")
        lines.append("")

    def _avg(key: str, field: str) -> float:
        vals = [x[field] for x in agg[key] if np.isfinite(x[field])]
        return float(np.mean(vals)) if vals else float("nan")

    def _sum_n(key: str) -> int:
        return int(sum(x["n"] for x in agg[key]))

    lines.append("## Aggregate (fold averages)")
    for k in ("A", "B", "C"):
        lines.append(
            f"- {k}: hit={_avg(k, 'hit'):.3f}, mean={_avg(k, 'mean'):.4f}, n_total={_sum_n(k)}"
        )
    lines.append("")
    lines.append("## Deltas vs A")
    lines.append(
        f"- B-A: hit={_avg('B','hit') - _avg('A','hit'):+.3f}, mean={_avg('B','mean') - _avg('A','mean'):+.4f}, "
        f"coverage_n={_sum_n('B')}/{_sum_n('A')}"
    )
    lines.append(
        f"- C-A: hit={_avg('C','hit') - _avg('A','hit'):+.3f}, mean={_avg('C','mean') - _avg('A','mean'):+.4f}, "
        f"coverage_n={_sum_n('C')}/{_sum_n('A')}"
    )

    out = Path("data/ret2d_timing_strategies.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

