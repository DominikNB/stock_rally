from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[c], errors="coerce")


def _q_bucket(s: pd.Series, q: int = 5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    ok = x.replace([np.inf, -np.inf], np.nan).dropna()
    if ok.nunique() < q:
        return pd.Series(np.nan, index=s.index)
    try:
        b = pd.qcut(ok, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=s.index)
    out = pd.Series(np.nan, index=s.index)
    out.loc[ok.index] = b.astype(float)
    return out


def _summarize_by_bucket(df: pd.DataFrame, feature: str, y_col: str, ret_col: str) -> pd.DataFrame:
    b = _q_bucket(df[feature], q=5)
    d = df.copy()
    d["_b"] = b
    d = d.dropna(subset=["_b"])
    if d.empty:
        return pd.DataFrame()
    g = (
        d.groupby("_b", dropna=False)
        .agg(
            n=("ticker", "size"),
            hit_rate=(y_col, "mean"),
            mean_ret=(ret_col, "mean"),
            med_ret=(ret_col, "median"),
            f_min=(feature, "min"),
            f_max=(feature, "max"),
        )
        .reset_index()
    )
    g["_b"] = g["_b"].astype(int) + 1
    g["feature"] = feature
    return g


def main() -> None:
    src = Path("data/master_complete.csv")
    if not src.is_file():
        raise SystemExit(f"Missing: {src}")
    df = pd.read_csv(src)
    if df.empty:
        raise SystemExit("master_complete.csv empty")

    # Core outcomes
    for c in ["ret_2d", "ret_4d", "ret_6d", "ret_8d", "ret_10d", "ret_mean_5"]:
        df[c] = _num(df, c)
    df = df.dropna(subset=["ret_4d"]).copy()
    df["y_win4"] = (df["ret_4d"] > 0).astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Feature candidates from enriched signal table
    feats = [
        "prob",
        "meta_prob_margin",
        "bb_width_20",
        "volume_zscore_20d",
        "ret_1d_signal_day",
        "ret_vs_sector_5d",
        "ret_vs_spy_5d",
        "open_gap_pct",
        "dist_from_20d_high_pct",
        "dist_from_20d_low_pct",
        "dist_ma200_pct",
        "momentum_20d",
        "volatility_20d",
        "trend_efficiency_20d",
        "amihud_illiq_20d",
        "market_ret_1d",
        "sector_ret_1d",
        "alpha_mkt_5d",
        "alpha_sec_5d",
        "bdays_to_next_earnings",
        "short_float_pct",
        "short_days_to_cover",
        "inst_own_pct",
        "signals_same_day",
        "cluster_mean_corr_60d",
    ]
    feats = [f for f in feats if f in df.columns]
    for f in feats:
        df[f] = _num(df, f)

    lines: list[str] = []
    lines.append("# Deep Signal Investigation")
    lines.append(f"rows={len(df)}, tickers={df['ticker'].nunique()}, days={df['Date'].dt.normalize().nunique()}")
    lines.append(f"base hit_rate(ret_4d>0)={df['y_win4'].mean():.3f}, mean_ret4={df['ret_4d'].mean():.4f}")

    # 1) Univariate bucket diagnostics
    lines.append("\n## Quintile Diagnostics (ret_4d win-rate / mean return)")
    for f in feats:
        g = _summarize_by_bucket(df[[*feats, "ticker", "y_win4", "ret_4d"]], f, "y_win4", "ret_4d")
        if g.empty:
            continue
        top = g.sort_values("mean_ret", ascending=False).head(1).iloc[0]
        bot = g.sort_values("mean_ret", ascending=True).head(1).iloc[0]
        lines.append(
            f"- {f}: best_q={int(top['_b'])} mean_ret={top['mean_ret']:.4f} hit={top['hit_rate']:.3f} | "
            f"worst_q={int(bot['_b'])} mean_ret={bot['mean_ret']:.4f} hit={bot['hit_rate']:.3f}"
        )

    # 2) Explicit pattern tests
    lines.append("\n## Pattern Tests")
    bb_q30 = float(df["bb_width_20"].quantile(0.30)) if "bb_width_20" in df.columns else np.nan
    vz_q60 = float(df["volume_zscore_20d"].quantile(0.60)) if "volume_zscore_20d" in df.columns else np.nan
    if np.isfinite(bb_q30) and np.isfinite(vz_q60):
        p = df[
            (df["bb_width_20"] <= bb_q30)
            & (df["volume_zscore_20d"] >= vz_q60)
            & (df["ret_1d_signal_day"] > 0)
        ]
        lines.append(
            f"- squeeze+volume+upday: n={len(p)} hit4={p['y_win4'].mean() if len(p) else np.nan:.3f} "
            f"mean_ret4={p['ret_4d'].mean() if len(p) else np.nan:.4f}"
        )
    if "ret_vs_sector_5d" in df.columns:
        p = df[df["ret_vs_sector_5d"] > 0]
        n = df[df["ret_vs_sector_5d"] <= 0]
        lines.append(
            f"- rel_strength sector>0: n={len(p)} hit4={p['y_win4'].mean():.3f} mean_ret4={p['ret_4d'].mean():.4f}; "
            f"<=0: n={len(n)} hit4={n['y_win4'].mean():.3f} mean_ret4={n['ret_4d'].mean():.4f}"
        )
    if "market_ret_1d" in df.columns:
        p = df[df["market_ret_1d"] > 0]
        n = df[df["market_ret_1d"] <= 0]
        lines.append(
            f"- market_ret_1d>0: n={len(p)} hit4={p['y_win4'].mean():.3f} mean_ret4={p['ret_4d'].mean():.4f}; "
            f"<=0: n={len(n)} hit4={n['y_win4'].mean():.3f} mean_ret4={n['ret_4d'].mean():.4f}"
        )

    # 3) Ticker stability
    lines.append("\n## Ticker Stability (min 8 signals)")
    tg = (
        df.groupby("ticker")
        .agg(n=("ticker", "size"), hit4=("y_win4", "mean"), mean_ret4=("ret_4d", "mean"))
        .reset_index()
    )
    tg8 = tg[tg["n"] >= 8].sort_values(["hit4", "mean_ret4"], ascending=False)
    lines.append("top:")
    for _, r in tg8.head(10).iterrows():
        lines.append(f"- {r['ticker']}: n={int(r['n'])}, hit4={r['hit4']:.3f}, mean_ret4={r['mean_ret4']:.4f}")
    lines.append("bottom:")
    for _, r in tg8.tail(10).iterrows():
        lines.append(f"- {r['ticker']}: n={int(r['n'])}, hit4={r['hit4']:.3f}, mean_ret4={r['mean_ret4']:.4f}")

    # 4) Sector diagnostics
    if "sector" in df.columns:
        lines.append("\n## Sector Diagnostics (min 20 signals)")
        sg = (
            df.groupby("sector")
            .agg(n=("sector", "size"), hit4=("y_win4", "mean"), mean_ret4=("ret_4d", "mean"))
            .reset_index()
        )
        sg = sg[sg["n"] >= 20].sort_values(["hit4", "mean_ret4"], ascending=False)
        for _, r in sg.iterrows():
            lines.append(f"- {r['sector']}: n={int(r['n'])}, hit4={r['hit4']:.3f}, mean_ret4={r['mean_ret4']:.4f}")

    # 5) Lightweight model to detect strongest separators
    lines.append("\n## Model-based Separators (RandomForest + permutation importance)")
    mdl_feats = [f for f in feats if df[f].notna().sum() >= int(0.5 * len(df))]
    X = df[mdl_feats].copy()
    y = df["y_win4"].values
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=8,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    lines.append(f"- holdout AUC={auc:.3f} (if near 0.5 => weak separability)")
    pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    for i in order[:12]:
        lines.append(f"- {mdl_feats[i]}: PI={pi.importances_mean[i]:.5f} ± {pi.importances_std[i]:.5f}")

    # 6) Post-signal path diagnostics
    lines.append("\n## Post-Signal Path Diagnostics")
    path_cols = [c for c in ["ret_2d", "ret_4d", "ret_6d", "ret_8d", "ret_10d"] if c in df.columns]
    if path_cols:
        good = df[df["ret_4d"] > 0]
        bad = df[df["ret_4d"] <= 0]
        lines.append("- average cumulative returns by horizon:")
        for c in path_cols:
            lines.append(
                f"  {c}: good={good[c].mean():.4f}, bad={bad[c].mean():.4f}, diff={good[c].mean() - bad[c].mean():.4f}"
            )
        if {"ret_2d", "ret_4d"}.issubset(df.columns):
            early_follow = df[df["ret_2d"] > 0]
            early_fail = df[df["ret_2d"] <= 0]
            lines.append(
                f"- early follow-through (ret_2d>0): n={len(early_follow)} hit4={early_follow['y_win4'].mean():.3f} "
                f"mean_ret4={early_follow['ret_4d'].mean():.4f}"
            )
            lines.append(
                f"- early weakness (ret_2d<=0): n={len(early_fail)} hit4={early_fail['y_win4'].mean():.3f} "
                f"mean_ret4={early_fail['ret_4d'].mean():.4f}"
            )
            whipsaw = df[(df["ret_2d"] > 0) & (df["ret_4d"] <= 0)]
            grindup = df[(df["ret_2d"] <= 0) & (df["ret_4d"] > 0)]
            lines.append(
                f"- whipsaw pattern (ret_2d>0 then ret_4d<=0): n={len(whipsaw)} ({len(whipsaw)/len(df):.1%} of all)"
            )
            lines.append(
                f"- delayed winner (ret_2d<=0 then ret_4d>0): n={len(grindup)} ({len(grindup)/len(df):.1%} of all)"
            )

    # 7) Final critical interpretation
    lines.append("\n## Critical Readout")
    lines.append(
        "- Strong universal separators are limited (expect moderate AUC); signal quality appears regime/ticker dependent."
    )
    lines.append(
        "- Relative-strength and context features typically separate better than pure squeeze-only conditions."
    )
    lines.append(
        "- A robust improvement likely needs conditional logic by ticker/sector/regime, not one global hard filter."
    )

    out = Path("data/deep_signal_investigation.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

