#!/usr/bin/env python3
"""
10 Visualisierungen zur **aktuellen Base-Optuna (Phase-1)-Zielfunktion** — gleiche
Verzweigungslogik wie ``optuna_train._score_tp_precision_fold`` (ohne Import dieses Moduls).

Ausführen aus dem Projektroot::

    .venv\\Scripts\\python scripts/plot_base_phase1_objective_tour.py

Ausgabe: ``reports/base_phase1_objective/objective_tour.pdf``
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg

OUT_DIR = _ROOT / "reports" / "base_phase1_objective"

# Spiegel von optuna_train (Zeilen 24–27)
PREC_GATE = float(cfg.OPT_MIN_PRECISION_BASE)
MAX_CFP = 4


def score_fold_counts(n_tp: int, n_sig: int, max_cfp: int) -> float:
    """Entspricht _score_tp_precision_fold ohne n_sig==0- und ohne _apply_filters_cv."""
    if max_cfp > MAX_CFP:
        return float(-2.0 - (max_cfp - MAX_CFP) * 0.1)
    if n_sig == 0:
        return float("nan")
    precision = float(n_tp) / float(n_sig)
    if precision >= PREC_GATE:
        return float(n_tp)
    return float(precision - 1.0)


def score_no_signals_branch(mean_pos: float, mean_neg: float, n_pos: int, n_neg: int) -> float:
    """Entspricht optuna_train Zeilen 696–707 (n_sig==0)."""
    p = np.clip(
        np.concatenate([np.full(n_pos, mean_pos), np.full(n_neg, mean_neg)]),
        1e-7,
        1.0 - 1e-7,
    ).astype(np.float64)
    yy = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int8)
    pos_m = yy == 1
    neg_m = yy == 0
    if pos_m.any() and neg_m.any():
        return float(np.mean(p[pos_m]) - np.mean(p[neg_m]))
    if pos_m.any():
        return float(np.mean(p[pos_m]) - 0.5)
    if neg_m.any():
        return float(0.5 - np.mean(p[neg_m]))
    return 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "objective_tour.pdf"

    with PdfPages(pdf_path) as pdf:
        # --- 1: Score vs n_tp bei festem n_sig ---
        fig, ax = plt.subplots(figsize=(8, 5))
        n_sig_fix = 40
        n_tps = np.arange(0, n_sig_fix + 1)
        scores = [score_fold_counts(int(nt), n_sig_fix, 0) for nt in n_tps]
        ax.plot(n_tps, scores, "b-", lw=2, label="Fold-Score")
        ax.axvline(
            PREC_GATE * n_sig_fix,
            color="orange",
            ls="--",
            lw=2,
            label=f"Gate: n_tp ≥ {PREC_GATE:.0%}·n_sig = {PREC_GATE * n_sig_fix:.0f}",
        )
        ax.set_xlabel("n_tp (bei n_sig=40, max_cfp≤4)")
        ax.set_ylabel("Score")
        ax.set_title("Über Gate → Score = n_tp; darunter → precision − 1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 2: Score vs Präzision (n_sig=50, n_tp = prec * n_sig) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        n_sig = 50
        prec = np.linspace(0.0, 1.0, 501)
        n_tp_f = prec * n_sig
        sc = []
        for nt in n_tp_f:
            sc.append(score_fold_counts(int(round(nt)), n_sig, 0))
        ax.plot(prec, sc, lw=2, color="darkgreen")
        ax.axvline(PREC_GATE, color="k", ls=":", label=f"Gate precision = {PREC_GATE:.0%}")
        ax.set_xlabel("Precision (= n_tp / n_sig, n_sig=50)")
        ax.set_ylabel("Score")
        ax.set_title("Knick am Gate: links precision−1, rechts n_tp (hier bis 50)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 3: Heatmap n_tp × n_sig ---
        fig, ax = plt.subplots(figsize=(9, 7))
        n_sig_max = 50
        n_tp_max = 40
        mat = np.full((n_tp_max + 1, n_sig_max), np.nan)
        for nt in range(n_tp_max + 1):
            for j, ns in enumerate(range(1, n_sig_max + 1)):
                mat[nt, j] = score_fold_counts(nt, ns, max_cfp=0)
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdYlGn", extent=(0.5, n_sig_max + 0.5, -0.5, n_tp_max + 0.5))
        ax.set_xlabel("n_sig")
        ax.set_ylabel("n_tp")
        ax.set_title(f"Fold-Score (max_cfp=0); Gate precision ≥ {PREC_GATE:.0%}")
        fig.colorbar(im, ax=ax, label="Score")
        xs = np.arange(1, n_sig_max + 1, dtype=float)
        ax.plot(xs, PREC_GATE * xs, "c--", lw=2, label="Gate n_tp = p·n_sig")
        ax.legend(loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 4: max_cfp Strafe ---
        fig, ax = plt.subplots(figsize=(8, 5))
        mc = np.arange(0, 16)
        sc = []
        for m in mc:
            if m > MAX_CFP:
                sc.append(float(-2.0 - (m - MAX_CFP) * 0.1))
            else:
                sc.append(score_fold_counts(25, 40, int(m)))
        ax.plot(mc, sc, "mo-", lw=2, ms=6)
        ax.axvline(MAX_CFP + 0.5, color="gray", ls="--", label=f"Limite max_cfp={MAX_CFP}")
        ax.set_xlabel("max_consec_fp")
        ax.set_ylabel("Score")
        ax.set_title("Strafe wenn max_consec_fp > 4 (Zusatz zu sonstigem Fold-Score)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 5: Keine Signale — mean_pos vs mean_neg ---
        fig, ax = plt.subplots(figsize=(8, 6))
        g1 = np.linspace(0.05, 0.95, 45)
        g2 = np.linspace(0.05, 0.95, 45)
        Z = np.zeros((len(g1), len(g2)))
        for i, a in enumerate(g1):
            for j, b in enumerate(g2):
                Z[i, j] = score_no_signals_branch(a, b, n_pos=80, n_neg=320)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[g2[0], g2[-1], g1[0], g1[-1]],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Mittlere Prob. (Neg-Klasse)")
        ax.set_ylabel("Mittlere Prob. (Pos-Klasse)")
        ax.set_title("Zweig n_signals==0: mean(p|y=1) − mean(p|y=0)")
        fig.colorbar(im, ax=ax, label="Score")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 6: Nur positive Labels (n_sig=0-Zweig) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        mp = np.linspace(0.01, 0.99, 200)
        sc = mp - 0.5
        ax.plot(mp, sc, lw=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("mean(p | nur positives y)")
        ax.set_ylabel("mean(p) − 0.5")
        ax.set_title("n_signals==0, nur positive Stichprobe")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 7: Nur negative Labels ---
        fig, ax = plt.subplots(figsize=(8, 5))
        mn = np.linspace(0.01, 0.99, 200)
        sc = 0.5 - mn
        ax.plot(mn, sc, lw=2, color="darkred")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("mean(p | nur negatives y)")
        ax.set_ylabel("0.5 − mean(p)")
        ax.set_title("n_signals==0, nur negative Stichprobe")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 8: Randfälle n_tp / n_sig (Säulen) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        cases = [
            (0, 20, 0, "0 TP, viele Sig."),
            (10, 100, 0, "10% prec"),
            (50, 100, 0, "50% prec (knapp unter Gate)"),
            (50, 99, 0, "~50.5% prec"),
            (25, 40, 6, "guter prec, max_cfp=6"),
        ]
        xs = np.arange(len(cases))
        ys = [score_fold_counts(a, b, c) for a, b, c, _ in cases]
        ax.bar(xs, ys, color="steelblue")
        ax.set_xticks(xs)
        ax.set_xticklabels([d for *_, d in cases], rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Beispiel-Kombinationen (Zähl-Logik)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 9: Study-Ziel = Mittel über Folds (Spielzeug) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        rng = np.random.default_rng(42)
        n_folds = 6
        fold_scores = []
        for _ in range(n_folds):
            nt = int(rng.integers(0, 15))
            ns = int(rng.integers(5, 40))
            mcf = int(rng.choice([0, 1, 2, 3, 5, 6]))
            fold_scores.append(score_fold_counts(nt, ns, mcf))
        x = np.arange(1, n_folds + 1)
        ax.bar(x, fold_scores, color="teal", alpha=0.85)
        ax.axhline(
            np.mean(fold_scores),
            color="crimson",
            ls="--",
            lw=2,
            label=f"Mittel = {np.mean(fold_scores):.3f}",
        )
        ax.set_xlabel("Fold (Spielzeug)")
        ax.set_ylabel("Fold-Score")
        ax.set_title("Optuna maximiert: np.mean(fold_scores) über Walk-Forward-Folds")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- 10: Text-Zusammenfassung ---
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            "Base Phase 1 — Zielfunktion (Kurzüberblick)",
            "",
            f"Konstanten: OPT_MIN_PRECISION_BASE = {PREC_GATE:.4f}, max_consec_fp ≤ {MAX_CFP}",
            "",
            "1) Wenn max_consec_fp > 4:",
            "     score = −2 − 0.1·(max_consec_fp − 4)",
            "",
            "2) Wenn n_signals == 0:",
            "     score aus Modell-Probs (Separation Pos/Neg, bzw. Fallback ±0.5)",
            "",
            f"3) Sonst precision = n_tp / n_signals:",
            f"     Wenn precision ≥ {PREC_GATE:.0%}:  score = n_tp",
            "     Sonst:                        score = precision − 1",
            "",
            "Optuna maximiert den Mittelwert dieser Fold-Scores (mean über Walk-Forward-Folds).",
            "Die echten Folds nutzen nested threshold + CV-Filter — hier nur die reine Zähl-Algebra.",
            "",
            f"PDF: {pdf_path}",
        ]
        y = 0.98
        for ln in lines:
            ax.text(0.04, y, ln, transform=ax.transAxes, fontsize=11, family="monospace", va="top")
            y -= 0.035
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Geschrieben: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
