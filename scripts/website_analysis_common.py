"""
Gemeinsame Pfade und CSV-Lese-Logik für Website-KI-Analysen (Gemini).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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


def read_signals_for_latest_day():
    """Liest gefilterte Zeilen zum aktuellsten Signaltag (wie bisher)."""
    import pandas as pd

    if DAILY_CSV.is_file():
        df = pd.read_csv(DAILY_CSV)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df.copy()
    elif COMPLETE_CSV.is_file():
        df = pd.read_csv(COMPLETE_CSV)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df[df["Date"] == latest].copy()
    elif HOLDOUT_CSV.is_file():
        df = pd.read_csv(HOLDOUT_CSV)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        latest = df["Date"].max()
        sub = df[df["Date"] == latest].copy()
    else:
        print(
            f"Fehlt: {DAILY_CSV}, {COMPLETE_CSV} oder {HOLDOUT_CSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    from lib.signal_extra_filters import ordered_llm_daily_columns
    from lib.stock_rally_v10.equity_classification import CLASSIFICATION_COLUMN_KEYS

    sub = sub.reindex(columns=ordered_llm_daily_columns(CLASSIFICATION_COLUMN_KEYS))

    if "prob" in sub.columns and "threshold_used" in sub.columns:
        sub = sub[
            pd.to_numeric(sub["prob"], errors="coerce")
            >= pd.to_numeric(sub["threshold_used"], errors="coerce")
        ]
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
