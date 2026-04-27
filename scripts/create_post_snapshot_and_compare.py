from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_count_signals_json(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for k in ("signals", "rows", "data"):
            if isinstance(payload.get(k), list):
                return len(payload[k])
    return None


def _summarize_master_complete(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    out["rows"] = int(len(df))
    if "ticker" in df.columns:
        out["unique_tickers"] = int(df["ticker"].nunique())
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        out["date_min"] = str(d.min())
        out["date_max"] = str(d.max())
        out["unique_days"] = int(d.dt.normalize().nunique())
    if "ret_4d" in df.columns:
        r4 = pd.to_numeric(df["ret_4d"], errors="coerce")
        out["ret4_mean"] = float(r4.mean())
        out["ret4_win_rate"] = float((r4 > 0).mean())
    return out


def _latest_pre_snapshot_dir(root: Path) -> Path | None:
    base = root / "data" / "model_snapshots"
    if not base.exists():
        return None
    cands = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("pre_meta_optimization_")],
        key=lambda p: p.name,
    )
    return cands[-1] if cands else None


def main() -> None:
    root = Path(".")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "data" / "model_snapshots" / f"post_meta_optimization_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "scoring_artifacts": root / "models" / "scoring_artifacts.joblib",
        "signals_json": root / "docs" / "signals.json",
        "index_html": root / "docs" / "index.html",
        "analysis_llm_last_txt": root / "docs" / "analysis_llm_last.txt",
        "analysis_llm_last_html": root / "docs" / "analysis_llm_last.html",
        "master_complete_csv": root / "data" / "master_complete.csv",
        "master_daily_update_csv": root / "data" / "master_daily_update.csv",
    }
    copied = {}
    for k, src in files.items():
        copied[k] = _copy_if_exists(src, out_dir / src.name)

    post = {
        "snapshot_dir": str(out_dir.as_posix()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "copied_files": copied,
        "signals_json_count": _load_count_signals_json(out_dir / "signals.json"),
        "master_complete": _summarize_master_complete(out_dir / "master_complete.csv"),
    }

    pre_dir = _latest_pre_snapshot_dir(root)
    comp: dict = {"pre_snapshot_dir": str(pre_dir.as_posix()) if pre_dir else None}
    if pre_dir:
        pre_signals_n = _load_count_signals_json(pre_dir / "signals.json")
        pre_mc = _summarize_master_complete(pre_dir / "master_complete.csv")
        post_mc = post["master_complete"]
        comp["pre_signals_json_count"] = pre_signals_n
        comp["post_signals_json_count"] = post["signals_json_count"]
        if pre_signals_n is not None and post["signals_json_count"] is not None:
            comp["delta_signals_json_count"] = int(post["signals_json_count"] - pre_signals_n)
        for k in ("rows", "unique_tickers", "unique_days", "ret4_mean", "ret4_win_rate"):
            if k in pre_mc and k in post_mc:
                comp[f"pre_{k}"] = pre_mc[k]
                comp[f"post_{k}"] = post_mc[k]
                if isinstance(pre_mc[k], (int, float)) and isinstance(post_mc[k], (int, float)):
                    comp[f"delta_{k}"] = post_mc[k] - pre_mc[k]

    summary = {"post": post, "comparison_vs_latest_pre": comp}
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPost snapshot + comparison saved: {out_dir}")


if __name__ == "__main__":
    main()

