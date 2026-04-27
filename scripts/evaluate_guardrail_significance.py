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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _apply_rule(df: pd.DataFrame, r: Rule) -> pd.Series:
    x = _num(df[r.feature])
    return x <= r.value if r.op == "<=" else x >= r.value


def _metrics(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) == 0:
        return np.nan, np.nan
    return float((df["ret_4d"] > 0).mean()), float(df["ret_4d"].mean())


def _fold_bounds(dates: pd.Series, n_folds: int = 5):
    u = pd.Index(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    chunks = np.array_split(u, n_folds + 1)
    out = []
    for i in range(1, len(chunks)):
        if len(chunks[i]) == 0:
            continue
        out.append((pd.Timestamp(chunks[i][0]), pd.Timestamp(chunks[i][-1])))
    return out


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
        x = _num(train[f]).dropna()
        if len(x) < 120 or x.nunique() < 8:
            continue
        for q in x.quantile([0.25, 0.5, 0.75]).values:
            out.append(Rule(f, "<=", float(q)))
            out.append(Rule(f, ">=", float(q)))
    return out


def _score(train_ref: pd.DataFrame, d: pd.DataFrame) -> float:
    if len(d) < 40:
        return -1e9
    ref_days = train_ref["Date"].dt.normalize().nunique()
    d_days = d["Date"].dt.normalize().nunique()
    ref_spd = len(train_ref) / max(1, ref_days)
    d_spd = len(d) / max(1, d_days)
    if d_spd < 0.4 * ref_spd:
        return -1e9
    hit, mr = _metrics(d)
    ref_hit, _ = _metrics(train_ref)
    return mr + 0.05 * (hit - ref_hit)


def _best_rules(train: pd.DataFrame) -> list[Rule]:
    cands = _candidate_rules(train)
    best_rules: list[Rule] = []
    best_sc = _score(train, train)
    single_sc = []
    for r in cands:
        d = train[_apply_rule(train, r)].copy()
        sc = _score(train, d)
        single_sc.append((r, sc))
        if sc > best_sc:
            best_sc = sc
            best_rules = [r]
    pair_pool = [r for r, _ in sorted(single_sc, key=lambda x: x[1], reverse=True)[:20]]
    for r1, r2 in combinations(pair_pool, 2):
        d = train[_apply_rule(train, r1) & _apply_rule(train, r2)].copy()
        sc = _score(train, d)
        if sc > best_sc:
            best_sc = sc
            best_rules = [r1, r2]
    return best_rules


def _apply_rules(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    if not rules:
        return df.copy()
    m = pd.Series(True, index=df.index)
    for r in rules:
        m &= _apply_rule(df, r).fillna(False)
    return df[m].copy()


def main() -> None:
    rng = np.random.default_rng(42)
    df = pd.read_csv("data/master_complete.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["ret_4d"] = _num(df["ret_4d"])
    df = df.dropna(subset=["Date", "ret_4d"]).sort_values("Date").reset_index(drop=True)
    folds = _fold_bounds(df["Date"], n_folds=5)

    fold_rows = []
    for i, (te_s, te_e) in enumerate(folds, 1):
        train = df[df["Date"] < te_s].copy()
        test = df[(df["Date"] >= te_s) & (df["Date"] <= te_e)].copy()
        if len(train) < 200 or len(test) < 60:
            continue
        rules = _best_rules(train)
        ftest = _apply_rules(test, rules)
        n_f = len(ftest)
        if n_f < 5:
            continue
        hit_b, ret_b = _metrics(test)
        hit_f, ret_f = _metrics(ftest)

        # placebo: random subsets same size as filtered
        B = 1500
        idx = np.arange(len(test))
        rand_hit = np.empty(B, dtype=float)
        rand_ret = np.empty(B, dtype=float)
        for b in range(B):
            pick = rng.choice(idx, size=n_f, replace=False)
            r = test.iloc[pick]
            rand_hit[b], rand_ret[b] = _metrics(r)
        p_hit = float((rand_hit >= hit_f).mean())
        p_ret = float((rand_ret >= ret_f).mean())
        fold_rows.append(
            {
                "fold": i,
                "n_test": len(test),
                "n_filtered": n_f,
                "base_hit4": hit_b,
                "filt_hit4": hit_f,
                "base_ret4": ret_b,
                "filt_ret4": ret_f,
                "p_hit_vs_random_same_n": p_hit,
                "p_ret_vs_random_same_n": p_ret,
            }
        )

    fr = pd.DataFrame(fold_rows)
    if fr.empty:
        raise SystemExit("No valid folds.")

    # combine fold deltas and compare against random
    obs_dhit = (fr["filt_hit4"] - fr["base_hit4"]).mean()
    obs_dret = (fr["filt_ret4"] - fr["base_ret4"]).mean()
    agg_p_hit = float((fr["p_hit_vs_random_same_n"] < 0.05).mean())
    agg_p_ret = float((fr["p_ret_vs_random_same_n"] < 0.05).mean())

    lines = []
    lines.append("# Guardrail Significance Check")
    for _, r in fr.iterrows():
        lines.append(
            f"fold {int(r['fold'])}: n_test={int(r['n_test'])}, n_filt={int(r['n_filtered'])}, "
            f"hit {r['base_hit4']:.3f}->{r['filt_hit4']:.3f}, ret {r['base_ret4']:.4f}->{r['filt_ret4']:.4f}, "
            f"p_hit={r['p_hit_vs_random_same_n']:.4f}, p_ret={r['p_ret_vs_random_same_n']:.4f}"
        )
    lines.append("")
    lines.append(f"mean_delta_hit4={obs_dhit:+.3f}")
    lines.append(f"mean_delta_ret4={obs_dret:+.4f}")
    lines.append(f"share_folds_p_hit_lt_0.05={agg_p_hit:.2f}")
    lines.append(f"share_folds_p_ret_lt_0.05={agg_p_ret:.2f}")
    lines.append("Interpretation: low p-values mean guardrail beats random subset of same size.")

    out = Path("data/guardrail_significance_test.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

