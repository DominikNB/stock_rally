"""
Vergleich „gute“ vs. „schlechte“ historische Signale anhand der **10-Handelstage-Rendite ab Entry**.

**Bezug ist immer der Entry-Tag** (Spalte ``entry_date`` in der CSV), nicht der Signaltag (``Date``).
Gleiche Logik wie ``build_holdout_signals_master``:

  • Signaltag ``Date``: Tag des Modell-Treffers.
  • **Entry:** erster Handelstag **nach** ``Date``; Einstiegspreis = **Open an diesem Entry-Tag**.
  • **ret_10d:** Close am **10. Handelstag nach Entry** / Open(Entry) − 1.
  • Auswertung nur, wenn ``ret_10d`` gesetzt ist.

**Stichprobe:** Sortierung nach Entry-Datum (fallback ``Date``), aufsteigend. Pro Gruppe
(**gut** = ret_10d > 0, **schlecht** = ret_10d ≤ 0) bilden die **frühesten**
``--analysis-frac`` (Standard 0.7) die Analyse-Stichprobe; der Rest ist ``split=holdout``.

  python -m holdout.analyze_holdout_good_bad_10d
  python -m holdout.analyze_holdout_good_bad_10d --analysis-frac 0.7 --export-split data/holdout_10d_split.csv
  python -m holdout.analyze_holdout_good_bad_10d --analysis-frac 1 --no-export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "master_complete.csv"
DEFAULT_EXPORT = ROOT / "data" / "holdout_10d_split.csv"
RET_COL = "ret_10d"


def _sort_timestamp(df: pd.DataFrame) -> pd.Series:
    """Chronologie für Split: entry_date, sonst Date."""
    sig = pd.to_datetime(df["Date"], errors="coerce")
    if "entry_date" in df.columns:
        ent = pd.to_datetime(df["entry_date"], errors="coerce")
        return ent.where(ent.notna(), sig)
    return sig


def stratified_chronological_split(
    df: pd.DataFrame,
    *,
    sort_ts: pd.Series,
    is_good: pd.Series,
    frac: float,
) -> pd.Series:
    """Pro Gruppe (gut / schlecht): früheste int(n*frac) Zeilen → 'analysis', Rest → 'holdout'."""
    out = pd.Series(index=df.index, dtype=object)
    f = float(frac)
    if f >= 1.0 - 1e-12:
        out.loc[:] = "analysis"
        return out
    if f <= 0:
        out.loc[:] = "holdout"
        return out

    for mask in (is_good, ~is_good):
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        sub_order = sort_ts.loc[idx].sort_values(kind="mergesort")
        ordered_idx = sub_order.index.to_numpy()
        n = len(ordered_idx)
        n_a = min(n, max(0, int(n * f)))
        out.loc[ordered_idx[:n_a]] = "analysis"
        out.loc[ordered_idx[n_a:]] = "holdout"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Gute vs. schlechte Signale (ret_10d); chronologische Analyse-Stichprobe je Gruppe."
        )
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="master_complete.csv")
    ap.add_argument(
        "--analysis-frac",
        type=float,
        default=0.7,
        help="Anteil der frühesten Zeilen je Gruppe (gut/schlecht) für die Tabellen (Standard: 0.7).",
    )
    ap.add_argument(
        "--export-split",
        type=Path,
        default=DEFAULT_EXPORT,
        metavar="CSV",
        help=f"Export aller gültigen Zeilen inkl. signal_class, split (Standard: {DEFAULT_EXPORT.name}).",
    )
    ap.add_argument("--no-export", action="store_true", help="Keine CSV schreiben.")
    args = ap.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        print(f"Datei fehlt: {path}", file=sys.stderr)
        sys.exit(1)

    mc = pd.read_csv(path)
    if RET_COL not in mc.columns:
        print(f"Spalte {RET_COL!r} fehlt in {path}", file=sys.stderr)
        sys.exit(1)
    if "Date" not in mc.columns:
        print("Spalte 'Date' fehlt.", file=sys.stderr)
        sys.exit(1)

    r = pd.to_numeric(mc[RET_COL], errors="coerce")
    usable = mc.loc[r.notna()].copy()
    usable["_r10"] = r.loc[r.notna()].astype(float)

    n_all = len(mc)
    n_ok = len(usable)
    print(
        f"Quelle: {path}\n"
        f"  Metrik: {RET_COL} = Open(Entry) → Close(Entry+10 Handelstage); "
        f"Entry = erster Handelstag nach Signaltag Date.\n"
        f"  Zeilen gesamt: {n_all}\n"
        f"  Mit gültigem {RET_COL}: {n_ok}\n"
        f"  Ausgeschlossen: {n_all - n_ok}\n",
        flush=True,
    )
    if usable.empty:
        print("Keine auswertbaren Zeilen.", file=sys.stderr)
        sys.exit(2)

    sort_ts = _sort_timestamp(usable)
    is_good = usable["_r10"] > 0
    usable["signal_class"] = is_good.map({True: "good", False: "bad"})
    usable["split"] = stratified_chronological_split(
        usable, sort_ts=sort_ts, is_good=is_good, frac=args.analysis_frac
    )

    analysis = usable[usable["split"] == "analysis"].copy()

    n_pos_all = int(is_good.sum())
    n_neg_all = int((~is_good).sum())
    n_pos_a = int((analysis["signal_class"] == "good").sum())
    n_neg_a = int((analysis["signal_class"] == "bad").sum())
    n_pos_h = n_pos_all - n_pos_a
    n_neg_h = n_neg_all - n_neg_a

    sort_name = "entry_date (sonst Date)" if "entry_date" in usable.columns else "Date"
    print(
        f"Stichprobe (--analysis-frac={args.analysis_frac:g}; Sortierung: {sort_name}, "
        f"aufsteigend, getrennt gut/schlecht):\n"
        f"  Gewinner (ret_10d>0): gesamt={n_pos_all}  → Analyse={n_pos_a}  Holdout={n_pos_h}\n"
        f"  Verlierer (ret_10d≤0): gesamt={n_neg_all}  → Analyse={n_neg_a}  Holdout={n_neg_h}\n"
        f"\n── Tabellen: nur Analyse-Stichprobe (split=analysis) ──\n",
        flush=True,
    )

    analysis["win"] = (analysis["_r10"] > 0).astype(int)
    pos = analysis[analysis["win"] == 1]
    neg = analysis[analysis["win"] == 0]

    def _block(title: str, sub: pd.DataFrame) -> None:
        print(f"── {title} (n={len(sub)}) ──")
        if sub.empty:
            print("  (leer)\n")
            return
        x = sub["_r10"]
        print(
            f"  {RET_COL}: Mittel={x.mean():+.2%}  Median={x.median():+.2%}  "
            f"Min={x.min():+.2%}  Max={x.max():+.2%}"
        )
        if "prob" in sub.columns:
            p = pd.to_numeric(sub["prob"], errors="coerce").dropna()
            if len(p):
                print(
                    f"  prob:      Mittel={p.mean():.4f}  Median={p.median():.4f}  Std={p.std():.4f}"
                )
        if "ret_mean_5" in sub.columns:
            m5 = pd.to_numeric(sub["ret_mean_5"], errors="coerce").dropna()
            if len(m5):
                print(
                    f"  ret_mean_5: Mittel={m5.mean():+.2%}  Median={m5.median():+.2%}"
                )
        print()

    _block(f"Positiv ({RET_COL} > 0)", pos)
    _block(f"Null oder negativ ({RET_COL} ≤ 0)", neg)

    if "sector" in analysis.columns and len(analysis):
        print("── Anteil ret_10d > 0 nach Sektor (n ≥ 3); nur Analyse ──")
        g = analysis.groupby("sector", dropna=False).agg(
            n=("_r10", "count"),
            anteil_pos=("win", "mean"),
            median_ret=("_r10", "median"),
        )
        g = g[g["n"] >= 3].sort_values("anteil_pos", ascending=True)
        print(g.to_string(), end="\n\n")

    if "prob" in analysis.columns:
        a = pd.to_numeric(pos["prob"], errors="coerce").dropna()
        b = pd.to_numeric(neg["prob"], errors="coerce").dropna()
        if len(a) >= 5 and len(b) >= 5:
            print(
                "── prob (Mittelwert), nur Analyse ──\n"
                f"  Gewinner: n={len(a)}  Mittel={a.mean():.4f}\n"
                f"  Verlierer: n={len(b)}  Mittel={b.mean():.4f}\n"
            )

    if not args.no_export:
        export_path = Path(args.export_split)
        drop_internal = [c for c in ("_r10", "win") if c in usable.columns]
        out_df = usable.drop(columns=drop_internal, errors="ignore")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(export_path, index=False)
        print(
            f"Geschrieben: {export_path}  ({len(out_df)} Zeilen; Analyse: df[df.split=='analysis'])",
            flush=True,
        )


if __name__ == "__main__":
    main()
