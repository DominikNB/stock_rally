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


def main() -> None:
    root = Path(".")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "data" / "model_snapshots" / f"pre_meta_optimization_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    candidates = {
        "scoring_artifacts": root / "models" / "scoring_artifacts.joblib",
        "signals_json": root / "docs" / "signals.json",
        "index_html": root / "docs" / "index.html",
        "analysis_llm_last_txt": root / "docs" / "analysis_llm_last.txt",
        "analysis_llm_last_html": root / "docs" / "analysis_llm_last.html",
        "master_complete_csv": root / "data" / "master_complete.csv",
        "master_daily_update_csv": root / "data" / "master_daily_update.csv",
    }

    for key, src in candidates.items():
        dst = out_dir / src.name
        copied[key] = _copy_if_exists(src, dst)

    summary: dict = {
        "snapshot_dir": str(out_dir.as_posix()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "copied_files": copied,
    }

    mc = out_dir / "master_complete.csv"
    if mc.exists():
        df = pd.read_csv(mc)
        if "Date" in df.columns:
            d = pd.to_datetime(df["Date"], errors="coerce")
            summary["master_complete_date_min"] = str(d.min())
            summary["master_complete_date_max"] = str(d.max())
            summary["master_complete_unique_days"] = int(d.dt.normalize().nunique())
        summary["master_complete_rows"] = int(len(df))
        if "ret_4d" in df.columns:
            r4 = pd.to_numeric(df["ret_4d"], errors="coerce")
            summary["master_complete_ret4_mean"] = float(r4.mean())
            summary["master_complete_ret4_win_rate"] = float((r4 > 0).mean())
        if "ticker" in df.columns:
            summary["master_complete_unique_tickers"] = int(df["ticker"].nunique())

    sj = out_dir / "signals.json"
    if sj.exists():
        try:
            payload = json.loads(sj.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                summary["signals_json_count"] = int(len(payload))
            elif isinstance(payload, dict):
                for k in ("signals", "rows", "data"):
                    if isinstance(payload.get(k), list):
                        summary["signals_json_count"] = int(len(payload[k]))
                        break
        except Exception as exc:
            summary["signals_json_parse_error"] = str(exc)

    (out_dir / "snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSnapshot saved: {out_dir}")


if __name__ == "__main__":
    main()

