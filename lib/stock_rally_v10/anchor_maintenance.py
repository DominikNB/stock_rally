"""Pflegt die quartalsweise Anchor-Schedule bei Universe-Änderungen."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from lib.stock_rally_v10.anchor_tickers import build_quarter_snapshot_rows, quarter_bounds


def _quarter_start(d: date) -> date:
    q = (d.month - 1) // 3
    m0 = q * 3 + 1
    return date(d.year, m0, 1)


def _next_quarter(d: date) -> date:
    m = d.month + 3
    y = d.year
    if m > 12:
        y += 1
        m -= 12
    return date(y, m, 1)


def _quarter_points(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        return []
    cur = _quarter_start(start_date)
    out: list[date] = []
    while cur <= end_date:
        out.append(cur)
        cur = _next_quarter(cur)
    return out


def _normalize_universe(tickers_by_sector: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sector, tickers in tickers_by_sector.items():
        s = str(sector).strip()
        tset = sorted({str(t).strip() for t in (tickers or []) if str(t).strip()})
        out[s] = tset
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_schedule_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = obj.get("anchors")
    if not isinstance(rows, list):
        return None
    return [r for r in rows if isinstance(r, dict)]


def _save_snapshot(path: Path, universe: dict[str, list[str]]) -> None:
    payload = {"version": 1, "sectors": universe}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_schedule(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"version": 1, "anchors": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _diff_summary(prev: dict[str, list[str]] | None, curr: dict[str, list[str]]) -> str:
    if prev is None:
        return "kein Snapshot vorhanden"
    prev_keys = set(prev.keys())
    curr_keys = set(curr.keys())
    added_sectors = sorted(curr_keys - prev_keys)
    removed_sectors = sorted(prev_keys - curr_keys)
    added_tickers = 0
    removed_tickers = 0
    for s in sorted(curr_keys & prev_keys):
        p = set(prev.get(s, []))
        c = set(curr.get(s, []))
        added_tickers += len(c - p)
        removed_tickers += len(p - c)
    return (
        f"+Sektoren={len(added_sectors)} -Sektoren={len(removed_sectors)} "
        f"+Ticker={added_tickers} -Ticker={removed_tickers}"
    )


def _schedule_needs_current_quarter(
    schedule_path: Path,
    sectors: set[str],
    end_d: date,
) -> tuple[bool, str]:
    rows = _load_schedule_rows(schedule_path)
    if not rows:
        return True, "Schedule leer/ungültig"
    cur_q = _quarter_start(end_d).isoformat()
    seen: set[str] = set()
    for r in rows:
        if str(r.get("period_start") or "") != cur_q:
            continue
        s = str(r.get("sector") or "").strip()
        if s:
            seen.add(s)
    missing = sorted(sectors - seen)
    if missing:
        return True, f"aktuelles Quartal {cur_q} unvollständig (fehlend: {missing[:3]}{'…' if len(missing) > 3 else ''})"
    return False, ""


def ensure_anchor_schedule_current(cfg_mod) -> None:
    """
    Prüft bei jedem Training die Universe-Signatur und baut bei Änderungen
    die komplette Quartals-Schedule seit START_DATE neu.
    """
    enabled = bool(getattr(cfg_mod, "NEWS_ANCHOR_ORG_FILTER", False))
    if not enabled:
        print("[Anchors] Skip: NEWS_ANCHOR_ORG_FILTER=False", flush=True)
        return

    schedule_path = Path(getattr(cfg_mod, "NEWS_ANCHOR_SCHEDULE_PATH", Path("data") / "sector_anchor_quarters.json"))
    snapshot_path = schedule_path.with_name("sector_anchor_universe_snapshot.json")
    curr_universe = _normalize_universe(getattr(cfg_mod, "TICKERS_BY_SECTOR", {}) or {})
    curr_sectors = set(curr_universe.keys())
    start_s = str(getattr(cfg_mod, "START_DATE", "")).strip() or "2015-01-01"
    end_s = str(getattr(cfg_mod, "END_DATE", "")).strip()
    start_d = date.fromisoformat(start_s[:10])
    end_d = date.fromisoformat(end_s[:10]) if end_s else date.today()
    prev_payload = _load_snapshot(snapshot_path)
    prev_universe = (prev_payload or {}).get("sectors") if isinstance(prev_payload, dict) else None
    schedule_exists = schedule_path.is_file()
    changed = prev_universe != curr_universe
    needs_q, why_q = _schedule_needs_current_quarter(schedule_path, curr_sectors, end_d)
    if schedule_exists and not changed and not needs_q:
        print("[Anchors] Universe unverändert — Schedule bleibt unverändert.", flush=True)
        return

    print(
        "[Anchors] Rebuild erforderlich: "
        + ("Schedule fehlt; " if not schedule_exists else "")
        + _diff_summary(prev_universe if isinstance(prev_universe, dict) else None, curr_universe)
        + (f"; {why_q}" if needs_q and why_q else ""),
        flush=True,
    )
    q_dates = _quarter_points(start_d, end_d)
    if not q_dates:
        print(f"[Anchors] Kein Quartal im Bereich {start_d}..{end_d}", flush=True)
        return

    top_n = int(getattr(cfg_mod, "NEWS_ANCHOR_TOP_N", 3) or 3)
    sleep_s = float(getattr(cfg_mod, "NEWS_ANCHOR_FETCH_SLEEP_SEC", 0.05) or 0.05)
    cap_cache: dict[str, float | None] = {}

    try:
        import yfinance as yf
    except ImportError:
        print("[Anchors] yfinance fehlt — Anchor-Rebuild übersprungen.", flush=True)
        return

    def fetch_cap(sym: str):
        s = str(sym).strip()
        if s in cap_cache:
            return cap_cache[s]
        time.sleep(max(0.0, sleep_s))
        try:
            t = yf.Ticker(s)
            cap = t.fast_info.get("market_cap")
            if cap is None or cap <= 0:
                info = t.info or {}
                cap = info.get("marketCap")
            v = float(cap) if cap else None
        except Exception:
            v = None
        cap_cache[s] = v
        return v

    rows: list[dict[str, Any]] = []
    n_q = len(q_dates)
    print(
        f"[Anchors] Baue Quartale {q_dates[0]}..{q_dates[-1]} ({n_q} Quartale), Top-{top_n} pro Sektor …",
        flush=True,
    )
    for i, qd in enumerate(q_dates, start=1):
        q0, q1 = quarter_bounds(qd)
        if i == 1 or i == n_q or i % max(1, n_q // 10) == 0:
            print(f"[Anchors]   Quartal {i}/{n_q}: {q0}..{q1} (exkl.)", flush=True)
        rows.extend(
            build_quarter_snapshot_rows(
                curr_universe,
                getattr(cfg_mod, "COMPANY_NAMES", {}),
                quarter_end=qd,
                top_n=top_n,
                fetch_cap=fetch_cap,
            )
        )

    _write_schedule(schedule_path, rows)
    _save_snapshot(snapshot_path, curr_universe)
    print(
        f"[Anchors] Schedule aktualisiert: {schedule_path} ({len(rows)} Zeilen); "
        f"Snapshot: {snapshot_path}",
        flush=True,
    )

