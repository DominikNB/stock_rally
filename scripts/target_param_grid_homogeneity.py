"""
Gittersuche: FIXED_Y-Target-Parameter vs. „Homogenität“ der Positivklasse.

- Positivrate >= 15% (MIN_POS_RATE) sonst kein Score-Ranking (Zeile: pass_min_rate=0).
- „Innere“ Homogenität: niedrige Streuung (CV) der qualifizierenden Trade-Rendite/MDD und
  RSI/RV an Positives.
- Uniqueness: Cohen d (Positives vs. zufällige Nicht-Target-Tage, i>=20) für RSI und RV20;
  U=1-1/(1+mean(|d|)). Kombinierter Score: weight_inner*H_in + weight_uniqueness*U.

Nur sinnvoll mit FIXED_Y_LABEL_MODE=entry_direct (wird forciert), damit target==1
eindeutig dem Signaltag t0 zugeordnet ist.

Ausführung aus dem Projektroot:
  .venv/Scripts/python.exe scripts/target_param_grid_homogeneity.py --quick
  .venv/Scripts/python.exe scripts/target_param_grid_homogeneity.py -j 8
  (Standard: parallele Prozesse bis min(CPU, 8, Gittergröße); -j 1 = sequentiell)
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import create_target, override_fixed_y_config_for_grid

MIN_POS_RATE = 0.15


def _gpoint_max_dip_frac(gpoint: dict[str, object]) -> float:
    """Wie in target: f in [0,1], legacy-bool True→0, False→1."""
    if "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION" in gpoint:
        return max(0.0, min(float(gpoint["FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION"]), 1.0))
    if "FIXED_Y_REQUIRE_NO_DIP_BELOW_ENTRY_UNTIL_THRESHOLD" in gpoint:
        return 0.0 if bool(gpoint["FIXED_Y_REQUIRE_NO_DIP_BELOW_ENTRY_UNTIL_THRESHOLD"]) else 1.0
    return max(0.0, min(float(cfg.fixed_y_max_dip_below_entry_fraction()), 1.0))


# Prozess-Pool: DataFrame + Snapshots (nur im Child gesetzt)
_GRID_DF: pd.DataFrame | None = None
_GRID_SNAP: dict[str, object] | None = None
_GRID_BASE: dict[str, object] | None = None


def _init_grid_worker(
    df: pd.DataFrame,
    snap: dict[str, object],
    base_fixed: dict[str, object],
) -> None:
    """Einmal pro Worker: Kursmatrix im Speicher halten (kein erneutes Laden pro Gitterpunkt)."""
    global _GRID_DF, _GRID_SNAP, _GRID_BASE
    _GRID_DF = df
    _GRID_SNAP = snap
    _GRID_BASE = base_fixed


def _eval_grid_point(
    item: tuple[int, dict[str, object], float, dict[str, Any]],
) -> dict[str, object]:
    """Ein Gitterpunkt: Target bauen + Homogenität (muss auf Top-Level liegen für Spawn/Windows)."""
    i, gpoint, min_pos, h_opts = item
    if _GRID_DF is None or _GRID_SNAP is None or _GRID_BASE is None:
        raise RuntimeError("Grid-Worker nicht initialisiert")
    ovr = {**_GRID_SNAP, **_GRID_BASE, **gpoint}
    with override_fixed_y_config_for_grid(**ovr):
        dft = create_target(_GRID_DF, quiet=True)
    w_lo = int(gpoint["FIXED_Y_WINDOW_MIN"])
    w_hi = int(gpoint["FIXED_Y_WINDOW_MAX"])
    rt = float(gpoint["FIXED_Y_RALLY_THRESHOLD"])
    st = bool(gpoint["FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY"])
    mdf = _gpoint_max_dip_frac(gpoint)
    hm = _homogeneity_block(
        dft, w_lo, w_hi, rt, mdf, st, h_opts, grid_idx=i
    )
    pr = float(hm["pos_rate"])
    h_comb = float(hm["homogeneity_combined"])
    pass_min = bool(pr + 1e-12 >= min_pos)
    ok_col = f"pos_rate>={min_pos:.0%}"
    return {
        "idx": i,
        "FIXED_Y_RALLY_THRESHOLD": rt,
        "FIXED_Y_WINDOW_MIN": w_lo,
        "FIXED_Y_WINDOW_MAX": w_hi,
        "FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY": st,
        "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION": mdf,
        "n_rows": len(dft),
        "n_pos": int(hm["n_pos"]),
        "pos_rate": pr,
        ok_col: int(pass_min),
        "n_pos_path_metrics": int(hm["n_pos_path_used"]),
        "inner_homogeneity": float(hm["inner_homogeneity"]),
        "uniqueness": hm["uniqueness"],
        "homogeneity_score": h_comb,
        "n_neg_sampled": int(hm["n_neg_sampled"]),
        "cohen_d_rsi": hm["cohen_d_rsi"],
        "cohen_d_rv20": hm["cohen_d_rv20"],
        "cv_return": hm["cv_return"],
        "cv_mdd": hm["cv_mdd"],
        "cv_rsi": hm["cv_rsi"],
        "cv_rv20": hm["cv_rv20"],
        "row_label": _grid_row_name(gpoint),
    }


def _rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.astype(float).diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    ma_u = up.ewm(alpha=1.0 / n, adjust=False).mean()
    ma_d = down.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ma_u / (ma_d + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _first_qualifying_trade(
    open_: np.ndarray,
    close: np.ndarray,
    signal_idx: int,
    *,
    w_lo: int,
    w_hi: int,
    rt: float,
    max_dip: float,
    strict: bool,
) -> tuple[int, float, float] | None:
    """Erstes w in [w_lo,w_hi] wie in target._create_target_one_ticker_fixed_bands; None wenn keines."""
    entry_idx = signal_idx + 1
    n = len(close)
    if entry_idx >= n:
        return None
    o = float(open_[entry_idx])
    md = max(0.0, min(float(max_dip), 1.0))
    for w in range(int(w_lo), int(w_hi) + 1):
        exit_idx = entry_idx + int(w)
        if exit_idx >= n:
            break
        c_ex = float(close[exit_idx])
        if not (np.isfinite(o) and np.isfinite(c_ex)) or o == 0.0:
            continue
        ret_trade = c_ex / o - 1.0
        if ret_trade < rt:
            continue
        floor = o * (1.0 - md)
        c_path = close[entry_idx : exit_idx + 1]
        if len(c_path) == 0 or bool(np.any(c_path < floor - 1e-9)):
            continue
        if strict:
            bad = False
            for j in range(entry_idx + 1, exit_idx + 1):
                prev_c = float(close[j - 1])
                cur_c = float(close[j])
                if (
                    not np.isfinite(prev_c)
                    or not np.isfinite(cur_c)
                    or prev_c == 0.0
                    or (cur_c / prev_c - 1.0) <= 0.0
                ):
                    bad = True
                    break
            if bad:
                continue
        path = close[entry_idx : exit_idx + 1]
        mdd = float(np.min(path) / o - 1.0)
        return int(w), float(ret_trade), mdd
    return None


def _snapshot_all_fixed_y() -> dict[str, object]:
    import lib.stock_rally_v10.config as cmod

    return {k: getattr(cmod, k) for k in dir(cmod) if k.startswith("FIXED_Y_")}


def _build_grid(quick: bool) -> list[dict[str, object]]:
    if quick:
        # 2×2×2 = 8 Punkte (rt × Fenster × strict); f=0.01 wie Default-cfg
        g: list[dict[str, object]] = []
        for rt, w, st in itertools.product(
            (0.05, 0.07),
            ((4, 6), (4, 8)),
            (False, True),
        ):
            g.append(
                {
                    "FIXED_Y_RALLY_THRESHOLD": float(rt),
                    "FIXED_Y_WINDOW_MIN": w[0],
                    "FIXED_Y_WINDOW_MAX": w[1],
                    "FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY": st,
                    "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION": 0.01,
                }
            )
        return g
    rts = [0.04, 0.05, 0.06, 0.07, 0.08]
    # w_min >= 4: mit realistischer Mindest-Haltedauer (keine 2–3-Tage-Strohfeuer)
    wins = [(4, 6), (4, 8), (5, 10), (5, 12)]
    out: list[dict[str, object]] = []
    for rt, w, st, mdf in itertools.product(
        rts, wins, (False, True), (0.0, 0.01, 1.0)
    ):
        out.append(
            {
                "FIXED_Y_RALLY_THRESHOLD": float(rt),
                "FIXED_Y_WINDOW_MIN": w[0],
                "FIXED_Y_WINDOW_MAX": w[1],
                "FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY": st,
                "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION": float(mdf),
            }
        )
    return out


def _pooled_stdev(x1: np.ndarray, x2: np.ndarray) -> float:
    a = np.asarray(x1, dtype=np.float64).ravel()
    b = np.asarray(x2, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))
    n1, n2 = a.size, b.size
    return float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2 + 1e-12)))


def _cohen_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """Mittelwertunterschied / gepoolte SD (Positives vs. Zufalls-Negative, gleiche Kennzahl)."""
    a = np.asarray(x_pos, dtype=np.float64).ravel()
    b = np.asarray(x_neg, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    ps = _pooled_stdev(a, b)
    if not np.isfinite(ps) or ps < 1e-12:
        return 0.0 if abs(m1 - m2) < 1e-9 else float("nan")
    return (m1 - m2) / ps


def _homogeneity_block(
    df: pd.DataFrame,
    w_lo: int,
    w_hi: int,
    rt: float,
    max_dip: float,
    strict: bool,
    h_opts: dict[str, Any],
    *,
    grid_idx: int = 0,
) -> dict[str, Any]:
    """
    Innere Homogenität: niedrige rel. Streuung (CV) der Pfad- und Vortags-Kennwerte **unter Positives**.

    Uniqueness: Cohen d (RSI, RV20 am Signaltag) Positives vs. zufällig gezogene
    Nicht-Target-Eintrittstage (i>=20), gleiche pro-Ticker-Struktur. Hohes d
    = Positives heben sich von typischen Nicht-Events ab.

    homogeneity_combined = weight_inner * H_inner + weight_uniqueness * U (L2-Norm der Gewichte nachträglich).
    """
    neg_mult = float(h_opts.get("neg_sample_mult", 2.0))
    w_in = float(h_opts.get("weight_inner", 0.5))
    w_un = float(h_opts.get("weight_uniqueness", 0.5))
    s_w = w_in + w_un
    if s_w > 0:
        w_in, w_un = w_in / s_w, w_un / s_w
    else:
        w_in, w_un = 0.5, 0.5

    seed = int(h_opts.get("random_seed", 42)) + 1_000_003 * int(grid_idx)
    rng = np.random.default_rng(seed)

    n = len(df)
    pos = df["target"].to_numpy() if "target" in df.columns else np.zeros(0)
    pos_rate = float(np.mean(pos)) if n else 0.0
    n_pos = int(np.sum(pos))
    rets: list[float] = []
    mdds: list[float] = []
    rsis: list[float] = []
    rvs: list[float] = []
    rsi_pos_ge20: list[float] = []
    rv_pos_ge20: list[float] = []
    neg_cands: list[tuple[str, int]] = []
    per_t_rsi: dict[str, np.ndarray] = {}
    per_t_dlr: dict[str, np.ndarray] = {}

    for tck, g in df.groupby("ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if "open" in g.columns:
            open_ = pd.to_numeric(g["open"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        else:
            open_ = close.copy()
        tcol = g["target"].to_numpy() if "target" in g.columns else np.zeros(len(g))
        cser = g["close"].astype(float)
        rsi = _rsi_ewm(cser, 14).to_numpy()
        lr = np.log(np.maximum(close, 1e-12))
        dlr = np.diff(lr, prepend=lr[0])
        per_t_rsi[tck] = rsi
        per_t_dlr[tck] = dlr
        for i in range(len(g)):
            if i >= 20 and int(tcol[i]) == 0:
                neg_cands.append((tck, i))
            if int(tcol[i]) != 1:
                continue
            m = _first_qualifying_trade(
                open_,
                close,
                i,
                w_lo=w_lo,
                w_hi=w_hi,
                rt=rt,
                max_dip=max_dip,
                strict=strict,
            )
            if m is None:
                continue
            _, rtr, mdd = m
            rets.append(rtr)
            mdds.append(mdd)
            if i >= 14 and np.isfinite(rsi[i]):
                rsis.append(float(rsi[i]))
            if i >= 20:
                rv_ = float(np.std(dlr[i - 19 : i + 1]))
                rvs.append(rv_)
                rsi_pos_ge20.append(float(rsi[i]))
                rv_pos_ge20.append(rv_)

    def _empty_nan() -> dict[str, Any]:
        nan = float("nan")
        return {
            "n_pos": n_pos,
            "pos_rate": pos_rate,
            "inner_homogeneity": nan,
            "uniqueness": nan,
            "homogeneity_combined": nan,
            "n_pos_path_used": 0,
            "n_neg_sampled": 0,
            "cohen_d_rsi": nan,
            "cohen_d_rv20": nan,
            "cv_return": nan,
            "cv_mdd": nan,
            "cv_rsi": nan,
            "cv_rv20": nan,
        }

    if len(rets) < 2:
        return _empty_nan()

    def _cv(x: list[float]) -> float:
        a = np.array(x, dtype=np.float64)
        if a.size < 2:
            return float("nan")
        return float(np.std(a) / (abs(float(np.mean(a))) + 1e-9))

    cv_r = _cv(rets)
    cv_m = _cv(mdds)
    cv_rsi = _cv(rsis) if len(rsis) > 1 else float("nan")
    cv_rv = _cv(rvs) if len(rvs) > 1 else float("nan")
    cvs: list[float] = [cv_r, cv_m]
    if len(rsis) > 1 and np.isfinite(cv_rsi):
        cvs.append(cv_rsi)
    if len(rvs) > 1 and np.isfinite(cv_rv):
        cvs.append(cv_rv)
    mean_cvs = float(np.mean(cvs))
    h_inner = 1.0 / (1.0 + mean_cvs)
    n_path = len(rets)

    n_neg = 0
    d_rsi = float("nan")
    d_rv = float("nan")
    rsi_neg: list[float] = []
    rv_neg: list[float] = []

    if len(neg_cands) >= 2 and len(rsi_pos_ge20) >= 2 and len(rv_pos_ge20) >= 2:
        n_want = int(min(len(neg_cands), max(200, int(neg_mult * max(1, n_pos)), 2 * n_pos, 1)))
        n_want = min(n_want, 50_000, len(neg_cands))
        pick = rng.choice(len(neg_cands), size=n_want, replace=False)
        for j in pick:
            tck, i = neg_cands[int(j)]
            rsi = per_t_rsi[tck]
            dlr = per_t_dlr[tck]
            rsi_neg.append(float(rsi[i]))
            rv_neg.append(float(np.std(dlr[i - 19 : i + 1])))
        n_neg = len(rsi_neg)
        if n_neg >= 2:
            a_rsi = np.array(rsi_pos_ge20, dtype=np.float64)
            b_rsi = np.array(rsi_neg, dtype=np.float64)
            a_rv = np.array(rv_pos_ge20, dtype=np.float64)
            b_rv = np.array(rv_neg, dtype=np.float64)
            d_rsi = _cohen_d(a_rsi, b_rsi)
            d_rv = _cohen_d(a_rv, b_rv)
            mabs: list[float] = []
            if np.isfinite(d_rsi):
                mabs.append(abs(float(d_rsi)))
            if np.isfinite(d_rv):
                mabs.append(abs(float(d_rv)))
            if mabs:
                mean_d = float(np.mean(mabs))
                u = 1.0 - 1.0 / (1.0 + mean_d)
            else:
                u = float("nan")
        else:
            u = float("nan")
    else:
        u = float("nan")
        n_neg = 0

    if np.isfinite(u):
        h_comb = w_in * h_inner + w_un * u
    else:
        h_comb = h_inner

    return {
        "n_pos": n_pos,
        "pos_rate": pos_rate,
        "inner_homogeneity": h_inner,
        "uniqueness": u,
        "homogeneity_combined": h_comb,
        "n_pos_path_used": n_path,
        "n_neg_sampled": n_neg,
        "cohen_d_rsi": d_rsi,
        "cohen_d_rv20": d_rv,
        "cv_return": cv_r,
        "cv_mdd": cv_m,
        "cv_rsi": cv_rsi,
        "cv_rv20": cv_rv,
    }


def _grid_row_name(row: dict[str, object]) -> str:
    return (
        f"rt={row.get('FIXED_Y_RALLY_THRESHOLD')!s} w=[{row.get('FIXED_Y_WINDOW_MIN')},"
        f"{row.get('FIXED_Y_WINDOW_MAX')}] strict={row.get('FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY')!s} "
        f"max_dip={row.get('FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION')!s}"
    )


def _print_task_line(
    *,
    row: dict[str, object],
    n_total: int,
    min_pos: float,
    ok_col: str,
    done: int,
    t_start: float,
) -> None:
    """Eine erledigte Gitterzeile; bei parallelen Läufen done/n_total = Fortschritt, ETA ~ Restzeit."""
    h = row["homogeneity_score"]
    h_in = row.get("inner_homogeneity", h)
    uu = row.get("uniqueness", float("nan"))
    pr = float(row["pos_rate"])
    h_s = f"{float(h):.4f}" if isinstance(h, (int, float)) and np.isfinite(h) else "nan"
    h_in_s = f"{float(h_in):.4f}" if isinstance(h_in, (int, float)) and np.isfinite(h_in) else "nan"
    u_s = f"{float(uu):.4f}" if isinstance(uu, (int, float)) and np.isfinite(uu) else "nan"
    idx1 = int(row["idx"]) + 1
    eta = ""
    if done >= 1 and done < n_total:
        elapsed = time.perf_counter() - t_start
        rem_s = (n_total - done) * (elapsed / done)
        if rem_s >= 120:
            eta = f"  ~{rem_s / 60.0:.1f} min Rest"
        elif rem_s >= 1:
            eta = f"  ~{int(rem_s)} s Rest"
    print(
        f"  {done}/{n_total}{eta}  Gitter [#{idx1}] {row['row_label']}  "
        f"pos={pr:.1%}  H_in={h_in_s}  uniqueness={u_s}  H_comb={h_s}  "
        f"ok>={min_pos:.0%}={bool(row.get(ok_col))}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Gitter-Homogenität fester Y-Target-Regeln")
    ap.add_argument("--quick", action="store_true", help="Kleines Gitter (8 Punkte)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "data" / "target_param_grid_homogeneity.csv",
        help="CSV-Ausgabe",
    )
    ap.add_argument("--max-tickers", type=int, default=None, help="Nur die ersten N Ticker (Schnelltest)")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Parallele Prozesse (Default: min(CPU, 8, Gitterpunkte); 1 = rein sequentiell)",
    )
    ap.add_argument(
        "--min-pos-rate",
        type=float,
        default=MIN_POS_RATE,
        help=(
            f"Mindest-Positivrate 0–1 (Default {MIN_POS_RATE:.2f} = 15 Prozent); "
            "unter dieser Rate gilt der Gitterpunkt als ungeeignet"
        ),
    )
    ap.add_argument(
        "--neg-sample-mult",
        type=float,
        default=2.0,
        help="Nicht-Target-Stichprobe: mindestens max(200, n_pos*mult) Eintrittstage (i>=20), max 50k",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Basis für Reproduzierbarkeit (pro Gitterpunkt wird offset addiert)",
    )
    ap.add_argument(
        "--weight-inner",
        type=float,
        default=0.5,
        help="Gewicht innere Homogenität H_in in homogeneity_combined (mit weight-uniqueness normalisiert)",
    )
    ap.add_argument(
        "--weight-uniqueness",
        type=float,
        default=0.5,
        help="Gewicht Uniqueness U (Streuung vs. zufällige Nicht-Labels)",
    )
    args = ap.parse_args()
    min_pos = float(args.min_pos_rate)
    if not (0.0 < min_pos < 1.0):
        print("--min-pos-rate muss zwischen 0 und 1 liegen.", file=sys.stderr)
        return 2

    tickers = list(cfg.ALL_TICKERS)
    if args.max_tickers is not None:
        tickers = tickers[: max(1, int(args.max_tickers))]

    print(f"Lade Kurse: {len(tickers)} Ticker …", flush=True)
    df_raw = load_stock_data(tickers=tickers)
    snap = _snapshot_all_fixed_y()
    base_fixed = {
        "OPT_OPTIMIZE_Y_TARGETS": False,
        "FIXED_Y_LABEL_MODE": "entry_direct",
    }
    grid = _build_grid(args.quick)
    ok_col = f"pos_rate>={min_pos:.0%}"

    if args.jobs is None:
        n_cpu = int(os.cpu_count() or 4)
        jobs = max(1, min(len(grid), n_cpu, 8))
    else:
        jobs = max(1, int(args.jobs))

    print(
        f"Gitter: {len(grid)} Punkte, {jobs} parallele Worker (Kursmatrix pro Prozess einmal im Speicher) …",
        flush=True,
    )
    print(
        "  Log pro Zeile: pos_rate, H_in (innere Homogenität), uniqueness (vs. Zufalls-Nicht-Labels), "
        "H_comb (gewichtete Kombination), Mindest-Positivrate.",
        flush=True,
    )

    h_run_opts: dict[str, Any] = {
        "neg_sample_mult": float(args.neg_sample_mult),
        "random_seed": int(args.random_seed),
        "weight_inner": float(args.weight_inner),
        "weight_uniqueness": float(args.weight_uniqueness),
    }
    tasks: list[tuple[int, dict[str, object], float, dict[str, Any]]] = [
        (i, gpoint, min_pos, h_run_opts) for i, gpoint in enumerate(grid)
    ]

    t_grid0 = time.perf_counter()
    n_grid = len(grid)
    if jobs == 1:
        _init_grid_worker(df_raw, snap, base_fixed)
        rows_out = []
        for t in tasks:
            rows_out.append(_eval_grid_point(t))
            _print_task_line(
                row=rows_out[-1],
                n_total=n_grid,
                min_pos=min_pos,
                ok_col=ok_col,
                done=len(rows_out),
                t_start=t_grid0,
            )
    else:
        rows_by_idx: dict[int, dict[str, object]] = {}
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_grid_worker,
            initargs=(df_raw, snap, base_fixed),
        ) as ex:
            futs = {ex.submit(_eval_grid_point, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                row = fut.result()
                rows_by_idx[idx] = row
                done += 1
                _print_task_line(
                    row=row,
                    n_total=n_grid,
                    min_pos=min_pos,
                    ok_col=ok_col,
                    done=done,
                    t_start=t_grid0,
                )
        rows_out = [rows_by_idx[i] for i in range(n_grid)]

    res = pd.DataFrame(rows_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    print(f"\nGespeichert: {args.output}", flush=True)
    good = res[(res[ok_col] == 1) & np.isfinite(res["homogeneity_score"])].copy()
    if len(good):
        good = good.sort_values("homogeneity_score", ascending=False)
        print(
            f"\nTop-10 (nur {ok_col}, höchste homogeneity_combined = H_in·w_in + U·w_u):",
            flush=True,
        )
        for _, r in good.head(10).iterrows():
            uu = r.get("uniqueness", float("nan"))
            u_str = f"{float(uu):.4f}" if isinstance(uu, (int, float)) and np.isfinite(uu) else "nan"
            print(
                f"  H={r['homogeneity_score']:.4f}  H_in={r['inner_homogeneity']:.4f}  "
                f"U={u_str}  pos={r['pos_rate']:.1%}  {r['row_label']}",
                flush=True,
            )
    else:
        print(
            f"\nKein Gitterpunkt erfüllt {ok_col} (und endliche H) — Gitter/Thresholds lockern.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
