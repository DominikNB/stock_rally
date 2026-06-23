"""
Gemeinsame Pfade und CSV-Lese-Logik für Website-KI-Analysen (Gemini).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"
PROMPT_FILE = DOCS / "website_analysis_prompt.txt"
DAILY_CSV = ROOT / "data" / "master_daily_update.csv"
COMPLETE_CSV = ROOT / "data" / "master_complete.csv"
HOLDOUT_CSV = ROOT / "data" / "holdout_signals.csv"
OUT_TXT = DOCS / "analysis_llm_last.txt"
OUT_HTML = DOCS / "analysis_llm_last.html"
OUT_RUN_META = DOCS / "analysis_llm_last_run_meta.json"
INDEX_HTML = DOCS / "index.html"
_LLM_TZ = ZoneInfo("Europe/Berlin")
NO_SIGNALS_TODAY_MSG = "Für heute keine Signale."


def analysis_expect_signal_date() -> str:
    """Scoring-Tag für die KI-Analyse: Env ANALYSIS_EXPECT_SIGNAL_DATE oder heute (Europe/Berlin)."""
    raw = os.environ.get("ANALYSIS_EXPECT_SIGNAL_DATE", "").strip()
    if raw:
        try:
            import pandas as pd

            return pd.to_datetime(raw).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now(_LLM_TZ).date().isoformat()


def master_daily_latest_signal_date() -> str | None:
    """YYYY-MM-DD des jüngsten Signaltags in master_daily_update.csv."""
    import pandas as pd

    if not DAILY_CSV.is_file():
        return None
    df = pd.read_csv(DAILY_CSV)
    if df.empty or "Date" not in df.columns:
        return None
    return pd.to_datetime(df["Date"]).max().strftime("%Y-%m-%d")


def _business_days_after(start: str, end: str) -> int:
    """Handelstage (Mo–Fr) strikt nach ``start`` bis einschließlich ``end``."""
    import pandas as pd

    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e <= s:
        return 0
    return len(pd.bdate_range(s + pd.Timedelta(days=1), e))


def llm_analysis_target_signaltag(scoring_day: str | None = None) -> str | None:
    """
    Signaltag für Gemini:
    - Treffer am Scoring-Tag → Scoring-Tag
    - Kein Treffer heute, jüngster CSV-Tag höchstens 1 Handelstag zurück → jüngster CSV-Tag
    - Sonst None → Kurzmeldung „Für heute keine Signale.“
    """
    scoring = (scoring_day or analysis_expect_signal_date()).strip()[:10]
    csv_d = master_daily_latest_signal_date()
    if not csv_d:
        return None
    if csv_d >= scoring:
        return scoring
    if _business_days_after(csv_d, scoring) <= 1:
        return csv_d
    return None


def llm_analysis_signaltag() -> str | None:
    if not OUT_RUN_META.is_file():
        return None
    try:
        meta = json.loads(OUT_RUN_META.read_text(encoding="utf-8"))
        return str(meta.get("signaltag_csv") or "")[:10] or None
    except Exception:
        return None


def llm_analysis_is_stale(expected_signal_date: str | None = None) -> bool:
    """True wenn KI-Metadaten nicht zum aktuellen Scoring-/Signaltag passen."""
    scoring = (expected_signal_date or analysis_expect_signal_date()).strip()[:10]
    target = llm_analysis_target_signaltag(scoring)
    llm_d = llm_analysis_signaltag()
    try:
        meta = json.loads(OUT_RUN_META.read_text(encoding="utf-8")) if OUT_RUN_META.is_file() else {}
    except Exception:
        meta = {}
    meta_scoring = str(meta.get("scoring_day") or meta.get("signaltag_csv") or "")[:10]
    if target is None:
        if meta.get("no_signals_for_scoring_day") and meta_scoring == scoring:
            return False
        return True
    if meta.get("no_signals_for_scoring_day"):
        return True
    if not llm_d:
        return True
    return llm_d != target


def ensure_daily_csv_red_context_columns() -> None:
    """Ergänzt master_daily_update.csv um vix_regime_ampel / red_context_llm falls fehlend."""
    import pandas as pd

    if not DAILY_CSV.is_file():
        return
    df = pd.read_csv(DAILY_CSV)
    if df.empty or "red_context_llm" in df.columns:
        return
    from lib.vix_red_context_chips import attach_red_context_llm_columns

    attach_red_context_llm_columns(df).to_csv(DAILY_CSV, index=False)


def patch_index_html_from_llm_analysis() -> bool:
    """
    Übernimmt den KI-Body aus analysis_llm_last.html in docs/index.html
    (ohne vollen Phase-17-Lauf). Für GitHub Pages nach Gemini-Lauf.
    """
    import re

    if not OUT_HTML.is_file():
        return False
    if not INDEX_HTML.is_file():
        print(f"Hinweis: {INDEX_HTML} fehlt — Index-Patch übersprungen.", file=sys.stderr)
        return False
    last = OUT_HTML.read_text(encoding="utf-8")
    m = re.search(
        r'<div class="analysis-llm-body prose-analysis">(.*)</div>\s*</div>\s*$',
        last,
        flags=re.DOTALL,
    )
    if not m:
        print("Hinweis: analysis-llm-body in analysis_llm_last.html nicht gefunden.", file=sys.stderr)
        return False
    inner = m.group(1)
    html = INDEX_HTML.read_text(encoding="utf-8")
    pat = (
        r'(<div class="analysis-llm-body prose-analysis">).*?'
        r'(</div>\s*\n\s*</div>\s*\n\s*<div class="section signals-recent-section">)'
    )
    new_html, n = re.subn(pat, r"\1" + inner + r"\2", html, count=1, flags=re.DOTALL)
    if n != 1:
        print(f"Hinweis: index.html KI-Block nicht gefunden (n={n}).", file=sys.stderr)
        return False
    INDEX_HTML.write_text(new_html, encoding="utf-8")
    print(f"Website: {INDEX_HTML} KI-Block aktualisiert ({len(inner):,} Zeichen).", flush=True)
    return True


def read_signals_for_latest_day():
    """Liest gefilterte Zeilen zum aktuellsten Signaltag (wie bisher)."""
    import pandas as pd

    if DAILY_CSV.is_file():
        df = pd.read_csv(DAILY_CSV)
        if df.empty:
            return "—", pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df[df["Date"] == latest].copy()
    elif COMPLETE_CSV.is_file():
        df = pd.read_csv(COMPLETE_CSV)
        if df.empty:
            return "—", pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df[df["Date"] == latest].copy()
    elif HOLDOUT_CSV.is_file():
        df = pd.read_csv(HOLDOUT_CSV)
        if df.empty:
            return "—", pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df[df["Date"] == latest].copy()
    else:
        print(
            f"Fehlt: {DAILY_CSV}, {COMPLETE_CSV} oder {HOLDOUT_CSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    _exp = analysis_expect_signal_date()
    target = llm_analysis_target_signaltag(_exp)
    if target is None:
        return _exp, pd.DataFrame()
    if target != latest:
        sub = df[df["Date"] == target].copy()
        latest = target

    from lib.signal_extra_filters import ordered_llm_daily_columns
    from lib.stock_rally_v10.equity_classification import CLASSIFICATION_COLUMN_KEYS

    sub = sub.reindex(columns=ordered_llm_daily_columns(CLASSIFICATION_COLUMN_KEYS))

    if "prob" in sub.columns and "threshold_used" in sub.columns:
        _prob = pd.to_numeric(sub["prob"], errors="coerce")
        _thr = pd.to_numeric(sub["threshold_used"], errors="coerce")
        if _thr.notna().any():
            sub = sub[_prob >= _thr]
    max_rows = int(os.environ.get("ANALYSIS_MAX_TICKER_ROWS", "40"))
    if len(sub) > max_rows:
        sub = (
            sub.nlargest(max_rows, "prob")
            if "prob" in sub.columns
            else sub.head(max_rows)
        )
    return latest, sub


def _inline_md(s: str) -> str:
    """Fettdruck **…**, `code`, Rest HTML-escaped. Keine verschachtelten Tags."""
    import html
    import re

    if not s:
        return ""
    chunks = re.split(r"(\*\*.+?\*\*|`[^`]+`)", s, flags=re.DOTALL)
    out: list[str] = []
    for ch in chunks:
        if ch.startswith("**") and ch.endswith("**") and len(ch) >= 4:
            out.append("<strong>" + html.escape(ch[2:-2]) + "</strong>")
        elif ch.startswith("`") and ch.endswith("`") and len(ch) >= 2:
            out.append('<code class="analysis-code">' + html.escape(ch[1:-1]) + "</code>")
        else:
            out.append(html.escape(ch))
    return "".join(out)


def llm_answer_to_fragment_html(text: str) -> str:
    """
    Wandelt gängige Markdown-Strukturen in sicheres HTML um (Überschriften, Listen,
    Absätze, Trennlinien, Zitate). Für Lesbarkeit auf der Website.
    """
    import re

    raw = (text or "").strip()
    if not raw:
        return "<p class=\"analysis-empty\">(Keine Antwort)</p>"

    lines = raw.splitlines()
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def flush_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush_lists()
            i += 1
            continue

        if re.match(r"^[-*_]{3,}\s*$", s):
            flush_lists()
            out.append('<hr class="analysis-hr">')
            i += 1
            continue

        if s.startswith("#### "):
            flush_lists()
            out.append(f"<h4>{_inline_md(s[5:])}</h4>")
            i += 1
            continue
        if s.startswith("### "):
            flush_lists()
            out.append(f"<h4>{_inline_md(s[4:])}</h4>")
            i += 1
            continue
        if s.startswith("## "):
            flush_lists()
            out.append(f"<h3>{_inline_md(s[3:])}</h3>")
            i += 1
            continue
        if s.startswith("# ") and not s.startswith("##"):
            flush_lists()
            out.append(f"<h3>{_inline_md(s[2:])}</h3>")
            i += 1
            continue

        if s.startswith("> "):
            flush_lists()
            q_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                q_lines.append(lines[i].strip()[2:])
                i += 1
            q_html = "<br>\n".join(_inline_md(x) for x in q_lines)
            out.append(f'<blockquote class="analysis-bq">{q_html}</blockquote>')
            continue

        if re.match(r"^[-*•]\s+", s) or (len(s) > 1 and s[0] == "•" and s[1] in " \t"):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append('<ul class="analysis-ul">')
                in_ul = True
            if re.match(r"^[-*•]\s+", s):
                content = re.sub(r"^[-*•]\s+", "", s)
            else:
                content = s[1:].lstrip()
            out.append(f"<li>{_inline_md(content)}</li>")
            i += 1
            continue

        nm = re.match(r"^(\d+)[.)]\s+", s)
        if nm:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append('<ol class="analysis-ol">')
                in_ol = True
            content = s[nm.end() :]
            out.append(f"<li>{_inline_md(content)}</li>")
            i += 1
            continue

        if in_ul or in_ol:
            flush_lists()

        para_lines: list[str] = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if (
                nxt.startswith("#")
                or re.match(r"^[-*•]\s+", nxt)
                or (len(nxt) > 1 and nxt[0] == "•" and nxt[1] in " \t")
                or re.match(r"^\d+[.)]\s+", nxt)
                or nxt.startswith("> ")
                or re.match(r"^[-*_]{3,}\s*$", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        joined = "\n".join(para_lines)
        inner = _inline_md(joined).replace("\n", "<br>\n")
        out.append(f"<p>{inner}</p>")

    flush_lists()
    return "\n".join(out)


def analysis_text_to_html(text: str, provider_line: str) -> str:
    import html

    inner = llm_answer_to_fragment_html(text)
    lead_esc = html.escape(provider_line)
    return (
        f'<div class="section analysis-llm">\n'
        f'  <h2>KI-Analyse (automatisch)</h2>\n'
        f'  <p class="prompt-lead">{lead_esc}; keine Anlageberatung.</p>\n'
        f'  <div class="analysis-llm-body prose-analysis">{inner}</div>\n'
        f"</div>\n"
    )
