from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    feature: str
    op: str
    value: float

    def label(self) -> str:
        return f"{self.feature} {self.op} {self.value:.6g}"


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _apply_rule(df: pd.DataFrame, r: Rule) -> pd.Series:
    x = _safe_num(df[r.feature])
    if r.op == "<=":
        return x <= r.value
    if r.op == ">=":
        return x >= r.value
    raise ValueError(f"unknown op: {r.op}")


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "hit4": np.nan, "mean_ret4": np.nan, "days": 0, "spd": 0.0}
    days = df["Date"].dt.normalize().nunique()
    return {
        "n": int(len(df)),
        "hit4": float((df["ret_4d"] > 0).mean()),
        "mean_ret4": float(df["ret_4d"].mean()),
        "days": int(days),
        "spd": float(len(df) / max(1, days)),
    }


def _score(train_df: pd.DataFrame) -> float:
    m = _metrics(train_df)
    if m["n"] < 40:
        return -1e9
    # keep enough flow: at least 40% of baseline signals-per-day
    b = _metrics(BASE_TRAIN_REF)
    if m["spd"] < 0.4 * b["spd"]:
        return -1e9
    # primary: return, secondary: hit-rate
    return m["mean_ret4"] + 0.05 * (m["hit4"] - b["hit4"])


def _candidate_rules(train: pd.DataFrame) -> list[Rule]:
    feats = [
        "market_ret_1d",
        "sector_ret_1d",
        "alpha_mkt_5d",
        "alpha_sec_5d",
        "cluster_mean_corr_60d",
        "signals_same_day",
        "dist_ma200_pct",
        "ret_vs_sector_5d",
        "ret_1d_signal_day",
        "volatility_20d",
        "bdays_to_next_earnings",
    ]
    out: list[Rule] = []
    for f in feats:
        if f not in train.columns:
            continue
        x = _safe_num(train[f]).dropna()
        if len(x) < 120 or x.nunique() < 8:
            continue
        qs = x.quantile([0.25, 0.5, 0.75]).to_dict()
        for qv in qs.values():
            out.append(Rule(f, "<=", float(qv)))
            out.append(Rule(f, ">=", float(qv)))
    return out


def _best_guardrail(train: pd.DataFrame) -> tuple[list[Rule], dict]:
    cands = _candidate_rules(train)
    best_rules: list[Rule] = []
    best_train = train
    best_sc = _score(train)

    # single-rule search
    for r in cands:
        d = train[_apply_rule(train, r)].copy()
        sc = _score(d)
        if sc > best_sc:
            best_sc = sc
            best_rules = [r]
            best_train = d

    # two-rule conjunction search
    top_single = sorted(
        ((r, _score(train[_apply_rule(train, r)].copy())) for r in cands),
        key=lambda x: x[1],
        reverse=True,
    )[:20]
    pair_pool = [r for r, _ in top_single]
    for r1, r2 in combinations(pair_pool, 2):
        m1 = _apply_rule(train, r1)
        m2 = _apply_rule(train, r2)
        d = train[m1 & m2].copy()
        sc = _score(d)
        if sc > best_sc:
            best_sc = sc
            best_rules = [r1, r2]
            best_train = d

    return best_rules, _metrics(best_train)


def _apply_rules(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    if not rules:
        return df.copy()
    m = pd.Series(True, index=df.index)
    for r in rules:
        m &= _apply_rule(df, r).fillna(False)
    return df[m].copy()


def _fold_bounds(dates: pd.Series, n_folds: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Index(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    chunks = np.array_split(u, n_folds + 1)  # first chunk for warmup train
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(1, len(chunks)):
        if len(chunks[i]) == 0:
            continue
        folds.append((pd.Timestamp(chunks[i][0]), pd.Timestamp(chunks[i][-1])))
    return folds


def main() -> None:
    src = Path("data/master_complete.csv")
    df = pd.read_csv(src)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["ret_4d"] = _safe_num(df["ret_4d"])
    df = df.dropna(subset=["Date", "ret_4d"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    folds = _fold_bounds(df["Date"], n_folds=5)
    lines: list[str] = []
    lines.append("# Guardrail Walk-Forward Test")
    lines.append(f"rows={len(df)}, days={df['Date'].dt.normalize().nunique()}, folds={len(folds)}")
    lines.append("")

    res = []
    global BASE_TRAIN_REF
    for i, (te_s, te_e) in enumerate(folds, 1):
        train = df[df["Date"] < te_s].copy()
        test = df[(df["Date"] >= te_s) & (df["Date"] <= te_e)].copy()
        if len(train) < 200 or len(test) < 60:
            continue
        BASE_TRAIN_REF = train.copy()
        base_te = _metrics(test)
        rules, tr_m = _best_guardrail(train)
        filt_te = _metrics(_apply_rules(test, rules))
        res.append((i, te_s, te_e, rules, base_te, filt_te, tr_m))

    # aggregate
    base_all = {"n": 0, "hit4": [], "mean_ret4": [], "spd": []}
    filt_all = {"n": 0, "hit4": [], "mean_ret4": [], "spd": []}
    for i, te_s, te_e, rules, b, f, tr in res:
        lines.append(f"## Fold {i} [{te_s.date()} .. {te_e.date()}]")
        lines.append(f"train_best_rules: {' AND '.join([r.label() for r in rules]) if rules else '(none)'}")
        lines.append(f"train_after_filter: n={tr['n']} hit4={tr['hit4']:.3f} mean_ret4={tr['mean_ret4']:.4f} spd={tr['spd']:.3f}")
        lines.append(f"test_baseline:     n={b['n']} hit4={b['hit4']:.3f} mean_ret4={b['mean_ret4']:.4f} spd={b['spd']:.3f}")
        lines.append(f"test_with_filter:  n={f['n']} hit4={f['hit4']:.3f} mean_ret4={f['mean_ret4']:.4f} spd={f['spd']:.3f}")
        lines.append("")
        base_all["n"] += b["n"]
        filt_all["n"] += f["n"]
        base_all["hit4"].append(b["hit4"])
        base_all["mean_ret4"].append(b["mean_ret4"])
        base_all["spd"].append(b["spd"])
        filt_all["hit4"].append(f["hit4"])
        filt_all["mean_ret4"].append(f["mean_ret4"])
        filt_all["spd"].append(f["spd"])

    if res:
        lines.append("## Aggregate (fold averages)")
        lines.append(
            f"- baseline: hit4={np.nanmean(base_all['hit4']):.3f}, mean_ret4={np.nanmean(base_all['mean_ret4']):.4f}, spd={np.nanmean(base_all['spd']):.3f}, n_total={base_all['n']}"
        )
        lines.append(
            f"- filtered: hit4={np.nanmean(filt_all['hit4']):.3f}, mean_ret4={np.nanmean(filt_all['mean_ret4']):.4f}, spd={np.nanmean(filt_all['spd']):.3f}, n_total={filt_all['n']}"
        )
        lines.append(
            f"- delta:    hit4={np.nanmean(filt_all['hit4']) - np.nanmean(base_all['hit4']):+.3f}, "
            f"mean_ret4={np.nanmean(filt_all['mean_ret4']) - np.nanmean(base_all['mean_ret4']):+.4f}, "
            f"spd={np.nanmean(filt_all['spd']) - np.nanmean(base_all['spd']):+.3f}"
        )
    else:
        lines.append("No valid folds.")

    out = Path("data/guardrail_walkforward_test.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    BASE_TRAIN_REF = pd.DataFrame()
    main()

