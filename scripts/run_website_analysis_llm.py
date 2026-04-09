"""
Führt docs/website_analysis_prompt.txt mit OpenAI Chat Completions aus.

Voraussetzung: Umgebungsvariable OPENAI_API_KEY
Optional: OPENAI_MODEL (Default: gpt-4o-mini), ANALYSIS_MAX_TICKER_ROWS (Default: 40)

Eingabe: data/meta_holdout_signals.csv (sonst data/holdout_signals.csv, weniger Spalten)
Filter: nur Zeilen mit max(Date) = aktuellster Signaltag; prob >= threshold_used

Ausgabe:
  docs/analysis_llm_last.txt   — Rohtext der Antwort
  docs/analysis_llm_last.html — HTML-Snippet (<div class="section">…</div>) für index.html

python scripts/run_website_analysis_llm.py
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROMPT_FILE = DOCS / "website_analysis_prompt.txt"
META_CSV = ROOT / "data" / "meta_holdout_signals.csv"
HOLDOUT_CSV = ROOT / "data" / "holdout_signals.csv"
OUT_TXT = DOCS / "analysis_llm_last.txt"
OUT_HTML = DOCS / "analysis_llm_last.html"


def _read_signals():
    import pandas as pd

    if META_CSV.is_file():
        df = pd.read_csv(META_CSV)
    elif HOLDOUT_CSV.is_file():
        df = pd.read_csv(HOLDOUT_CSV)
    else:
        print(f"Fehlt: {META_CSV} oder {HOLDOUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    latest = df["Date"].max()
    sub = df[df["Date"] == latest].copy()
    if "prob" in sub.columns and "threshold_used" in sub.columns:
        sub = sub[pd.to_numeric(sub["prob"], errors="coerce") >= pd.to_numeric(sub["threshold_used"], errors="coerce")]
    max_rows = int(os.environ.get("ANALYSIS_MAX_TICKER_ROWS", "40"))
    if len(sub) > max_rows:
        sub = sub.nlargest(max_rows, "prob") if "prob" in sub.columns else sub.head(max_rows)
    return latest, sub


def _openai_chat(api_key: str, model: str, system: str, user: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.35")),
        "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", "4096")),
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unerwartete API-Antwort: {payload!r}") from e


def _txt_to_simple_html(text: str) -> str:
    import html

    blocks = text.split("\n\n")
    parts = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        esc = html.escape(b)
        if len(b) < 200 and b.isupper():
            parts.append(f"<h3>{esc}</h3>")
        else:
            parts.append(f"<p>{esc.replace(chr(10), '<br>')}</p>")
    inner = "\n".join(parts) if parts else f"<p>{html.escape(text)}</p>"
    return (
        f'<div class="section analysis-llm">\n'
        f'  <h2>KI-Analyse (automatisch)</h2>\n'
        f'  <p class="prompt-lead">Generiert per OpenAI; keine Anlageberatung.</p>\n'
        f'  <div class="analysis-llm-body">{inner}</div>\n'
        f"</div>\n"
    )


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY nicht gesetzt — überspringe LLM-Analyse.", file=sys.stderr)
        sys.exit(0)

    if not PROMPT_FILE.is_file():
        print(f"Fehlt: {PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    latest, sub = _read_signals()
    if sub.empty:
        print(f"Keine Signale für aktuellsten Tag {latest}.", file=sys.stderr)
        sys.exit(0)

    # Tabellarische Übergabe (kompakt)
    csv_buf = sub.to_csv(index=False)
    user_msg = (
        f"Signaltag (aktuellster Tag in den Daten): {latest}\n"
        f"Anzahl Treffer (Meta ≥ Schwelle, ggf. gekappt): {len(sub)}\n\n"
        f"--- Tabelle (CSV) ---\n{csv_buf}"
    )

    system = (
        "Du bist ein vorsichtiger Finanz-Research-Assistent. Antworte auf Deutsch. "
        "Nutze öffentlich zugängliches Wissen; wenn dir konkrete Tages-News fehlen, "
        "sage das transparent und arbeite mit Plausibilität und den CSV-Daten. "
        "Halte dich an die Output-Vorgaben im Nutzer-Prompt (inkl. Pflicht-Fazit)."
    )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"OpenAI Anfrage ({model}), Signaltag {latest}, Zeilen {len(sub)} …", flush=True)
    try:
        answer = _openai_chat(api_key, model, system, prompt + "\n\n---\n\n" + user_msg)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err}", file=sys.stderr)
        sys.exit(1)

    DOCS.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(answer, encoding="utf-8")
    OUT_HTML.write_text(_txt_to_simple_html(answer), encoding="utf-8")
    print(f"Geschrieben: {OUT_TXT}, {OUT_HTML}")


if __name__ == "__main__":
    main()
