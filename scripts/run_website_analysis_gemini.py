"""
Website-Analyse mit Gemini: CSV-Upload (File API) + Grounding über Google Search.

Nutzt das Paket ``google-genai`` (Google Gen AI SDK). API-Key: GEMINI_API_KEY
in .env / config/secrets.env (siehe config/env.example).

Optional: GEMINI_MODEL (Default: gemini-2.5-flash — mit oder ohne Präfix ``models/``)

Eingabe: bevorzugt data/master_daily_update.csv (gleiche Tabellenlogik wie website_analysis_common).

Ausgabe: docs/analysis_llm_last.txt, docs/analysis_llm_last.html

Aktualisiert danach automatisch den KI-Block in docs/index.html (GitHub Pages).

python scripts/run_website_analysis_gemini.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config.load_env import load_project_env

from lib.website_rally_prompt import load_rally_prompt_injection

from website_analysis_common import (
    DOCS,
    NO_SIGNALS_TODAY_MSG,
    OUT_HTML,
    OUT_TXT,
    PROMPT_FILE,
    analysis_expect_signal_date,
    analysis_scoring_run_end,
    analysis_text_to_html,
    build_llm_steering_block,
    build_scoring_news_mandate_block,
    ensure_daily_csv_red_context_columns,
    master_daily_latest_signal_date,
    patch_index_html_from_llm_analysis,
    read_signals_for_latest_day,
)

OUT_RUN_META = DOCS / "analysis_llm_last_run_meta.json"
_CHECK_NUMERIC_COLS = (
    "adv_20d_local",
    "liquidity_tier",
    "short_float_pct",
    "short_days_to_cover",
    "open_gap_pct",
    "volume_zscore_20d",
    "inst_own_pct",
    "market_ret_1d",
    "sector_ret_1d",
)


def _build_data_availability_block(sub) -> str:
    """Kompakter, deterministischer Hinweis für das LLM: welche Kernspalten sind je Treffer befüllt."""
    import pandas as pd

    lines: list[str] = [
        "=== Datenverfuegbarkeit (aus CSV, fuer den Daten-Check) ===",
        (
            "Regel: Behaupte fehlende Kennzahlen nur dann, wenn der jeweilige CSV-Zellwert "
            "fuer den konkreten Treffer wirklich NaN/leer ist. Wenn Wert vorhanden: nicht als fehlend darstellen."
        ),
    ]
    for _, r in sub.iterrows():
        t = str(r.get("ticker", ""))
        d = str(r.get("Date", ""))
        miss: list[str] = []
        have: list[str] = []
        for c in _CHECK_NUMERIC_COLS:
            v = r.get(c, None)
            if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
                miss.append(c)
            else:
                have.append(c)
        lines.append(
            f"- {t} @ {d}: verfügbar={len(have)}/{len(_CHECK_NUMERIC_COLS)}; "
            f"missing=[{', '.join(miss)}]"
        )
    lines.append("=== Ende Datenverfuegbarkeit ===")
    return "\n".join(lines) + "\n\n"


def _model_id_for_genai(raw: str) -> str:
    s = raw.strip()
    if s.startswith("models/"):
        return s[len("models/") :]
    return s


def _wait_file_ready(client: genai.Client, name: str, timeout_s: float = 120.0) -> None:
    t0 = time.monotonic()
    f = client.files.get(name=name)
    while f.state == types.FileState.PROCESSING:
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError("Datei-Upload: Verarbeitung dauert zu lange.")
        time.sleep(2)
        f = client.files.get(name=name)
    if f.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Datei-Upload fehlgeschlagen: state={f.state}")


def _retryable_api_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code == 429:
            return True
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code in (429, 500, 502, 503, 504):
            return True
    s = f"{type(exc).__name__} {exc!s}".lower()
    if any(x in s for x in ("timeout", "timed out", "connection", "temporar", "rate", "429", "503")):
        return True
    return False


def _generate_content_retry(
    client: genai.Client,
    *,
    model: str,
    contents: list,
    config: types.GenerateContentConfig | None,
    label: str,
    max_attempts: int = 4,
) -> types.GenerateContentResponse:
    delay_s = 5.0
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last = e
            if attempt < max_attempts - 1 and _retryable_api_error(e):
                print(
                    f"Gemini {label}: Versuch {attempt + 1}/{max_attempts} fehlgeschlagen "
                    f"({type(e).__name__}) — erneut in {delay_s:.0f}s …",
                    flush=True,
                )
                time.sleep(delay_s)
                delay_s = min(delay_s * 1.8, 120.0)
            else:
                break
    assert last is not None
    raise last


def _response_text(response: types.GenerateContentResponse) -> str:
    try:
        return (response.text or "").strip()
    except ValueError:
        parts_out: list[str] = []
        for c in response.candidates or []:
            if not c.content or not c.content.parts:
                continue
            for p in c.content.parts:
                if hasattr(p, "text") and p.text:
                    parts_out.append(p.text)
        return "".join(parts_out).strip()


def main() -> None:
    load_project_env(ROOT)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "GEMINI_API_KEY nicht gesetzt — setze ihn in .env (Projektroot), siehe config/env.example.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not PROMPT_FILE.is_file():
        print(f"Fehlt: {PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)

    model_raw = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    model_name = _model_id_for_genai(model_raw)

    client = genai.Client(
        api_key=api_key,
        # HttpOptions.timeout ist Millisekunden (600000 ms = 10 Minuten).
        http_options=types.HttpOptions(timeout=600_000),
    )

    prompt_static = PROMPT_FILE.read_text(encoding="utf-8")
    rally_block = load_rally_prompt_injection(ROOT).rstrip()
    prompt_base = rally_block + "\n\n---\n\n" + prompt_static
    expect_day = analysis_expect_signal_date()
    latest, sub = read_signals_for_latest_day()
    if sub.empty:
        DOCS.mkdir(parents=True, exist_ok=True)
        msg_plain = NO_SIGNALS_TODAY_MSG
        csv_latest = master_daily_latest_signal_date() or (latest if latest != "—" else None)
        provider = (
            f"Kein Gemini-Aufruf — keine Meta-Hits für den Scoring-Tag {expect_day} "
            f"(letzter Signaltag in CSV: {csv_latest or '—'})."
        )
        OUT_TXT.write_text(msg_plain + "\n", encoding="utf-8")
        OUT_HTML.write_text(
            analysis_text_to_html(msg_plain, provider),
            encoding="utf-8",
        )
        _run_meta = {
            "signaltag_csv": expect_day,
            "scoring_day": expect_day,
            "no_signals_for_scoring_day": True,
            "latest_signal_date_in_csv": csv_latest,
        }
        try:
            OUT_RUN_META.write_text(
                json.dumps(_run_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        print(
            f"Keine Signale für LLM-Auswertung (Scoring-Tag {expect_day}; "
            f"letzter CSV-Signaltag: {latest}). "
            f"Geschrieben: {OUT_TXT}, {OUT_HTML}, {OUT_RUN_META}",
            flush=True,
        )
        patch_index_html_from_llm_analysis()
        return

    tmp_path: str | None = None
    uploaded: types.File | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="meta_hits_")
        os.close(fd)
        sub.to_csv(tmp_path, index=False)

        print(
            f"Gemini: lade CSV ({len(sub)} Zeilen, Signaltag {latest}) …",
            flush=True,
        )
        uploaded = client.files.upload(
            file=tmp_path,
            config=types.UploadFileConfig(
                display_name="meta_hits_daily.csv",
                mime_type="text/csv",
            ),
        )
        _wait_file_ready(client, uploaded.name)

        _steering, _run_meta = build_llm_steering_block(
            latest,
            scoring_day=expect_day,
            scoring_end=analysis_scoring_run_end(),
        )
        if latest != expect_day:
            _run_meta["analysis_lag_one_session"] = True
        _news_mandate = build_scoring_news_mandate_block(latest, expect_day, _run_meta)
        _avail = _build_data_availability_block(sub)
        print(
            f"Gemini Steuerblock: Zusatzfenster {_run_meta['zusatzfenster_start_europe_berlin']} "
            f"… {_run_meta['zusatzfenster_end_europe_berlin_inclusive']} (Europe/Berlin)",
            flush=True,
        )

        _tickers = [str(r.get("ticker", "")).strip() for _, r in sub.iterrows() if str(r.get("ticker", "")).strip()]
        _sectors: dict[str, list[str]] = {}
        for _, r in sub.iterrows():
            _t = str(r.get("ticker", "")).strip()
            if not _t:
                continue
            _sec = str(r.get("sector") or r.get("gics_sector") or "—").strip()
            _sectors.setdefault(_sec, []).append(_t)
        _sector_lines = "\n".join(
            f"  - {_sec}: {', '.join(sorted(set(ts)))}" for _sec, ts in sorted(_sectors.items())
        )
        _n_red = 0
        if "vix_regime_ampel" in sub.columns:
            _n_red = int(
                sub["vix_regime_ampel"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("red")
                .sum()
            )
        _red_hint = ""
        if _n_red:
            _red_hint = (
                f"VIX-Ampel rot bei {_n_red} Treffer(n): Spalte **red_context_llm** pro Zeile lesen und "
                "in Makro, Daten-Check, Vergleich und Fazit **mit** News/RS/Liquidität **verweben** — "
                "**keine** Chip-Farbliste für den Nutzer.\n"
            )
        user_block = (
            f"Signaltag (aktuellster Tag in den Daten): {latest}\n"
            f"Anzahl Treffer (Meta ≥ Schwelle): {len(sub)}\n"
            f"Alle Ticker dieses Signaltags: {', '.join(_tickers)}\n"
            f"Nach Sektor/Cluster:\n{_sector_lines}\n\n"
            f"{_red_hint}"
            f"{_news_mandate}"
            f"{_steering}"
            f"{_avail}"
            "Die angehängte CSV enthält die Tabellenzeilen zu diesen Treffern. "
            "Recherchiere im Web (Google Search, Pflicht) zu News, Makro und Kontext bis einschließlich "
            "Scoring-Lauf — Hauptfenster und Zusatzfenster laut Steuerblock. "
            "Das **Fazit** wertet **alle** Informationen bis zum Scoring-Lauf **zusammen** aus "
            "(eine Prosa, keine Trennung Haupt-/Zusatzfenster). "
            "Vergleiche die Treffer **untereinander in Prosa** (Abschnitt „Vergleich der Signale“) — "
            "der Nutzer soll keine Tabellen selbst abgleichen. "
            "Nutze CSV-Spalten inkl. vix_regime_ampel / red_context_llm wie im Prompt.\n\n"
            "--- Prompt (Vorgaben) ---\n"
            f"{prompt_base}"
        )

        contents: list = [user_block, uploaded]
        cfg_search = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        cfg_plain = types.GenerateContentConfig()

        use_search = os.environ.get("GEMINI_NO_SEARCH", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        )
        allow_no_search = os.environ.get("GEMINI_ALLOW_NO_SEARCH_FALLBACK", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        _search_fallback_used = False
        try:
            if use_search:
                print(
                    f"Gemini generate_content ({model_name}) mit Google Search …",
                    flush=True,
                )
                response = _generate_content_retry(
                    client,
                    model=model_name,
                    contents=contents,
                    config=cfg_search,
                    label="mit Google Search",
                )
            else:
                print(
                    "GEMINI_NO_SEARCH gesetzt — ohne Google Search.",
                    flush=True,
                )
                response = _generate_content_retry(
                    client,
                    model=model_name,
                    contents=contents,
                    config=cfg_plain,
                    label="ohne Google Search",
                )
        except Exception as first_exc:
            if use_search:
                print(
                    f"Gemini mit Google Search fehlgeschlagen ({type(first_exc).__name__}: {first_exc}). "
                    "Erneuter Versuch mit Google Search …",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    response = _generate_content_retry(
                        client,
                        model=model_name,
                        contents=contents,
                        config=cfg_search,
                        label="Google Search Wiederholung nach Fehler",
                    )
                except Exception as retry_exc:
                    if not allow_no_search:
                        import traceback

                        traceback.print_exc()
                        print(
                            f"Erster Fehler (mit Search): {first_exc!r}; "
                            f"Wiederholung: {retry_exc!r}",
                            file=sys.stderr,
                        )
                        raise
                    print(
                        f"Google Search erneut fehlgeschlagen ({retry_exc}). "
                        "Fallback ohne Google Search (GEMINI_ALLOW_NO_SEARCH_FALLBACK).",
                        file=sys.stderr,
                        flush=True,
                    )
                    response = _generate_content_retry(
                        client,
                        model=model_name,
                        contents=contents,
                        config=cfg_plain,
                        label="Fallback ohne Google Search",
                    )
                    _search_fallback_used = True
            else:
                import traceback

                traceback.print_exc()
                raise

        if not response.candidates:
            print("Gemini: leere Antwort (candidates).", file=sys.stderr)
            sys.exit(1)
        answer = _response_text(response)
        if not answer and use_search and not _search_fallback_used:
            print(
                "Gemini: keine Text-Antwort mit Google Search (ev. Safety-Block) — "
                "erneuter Versuch mit Google Search …",
                file=sys.stderr,
                flush=True,
            )
            response = _generate_content_retry(
                client,
                model=model_name,
                contents=contents,
                config=cfg_search,
                label="Google Search Wiederholung",
            )
            answer = _response_text(response)
        if not answer and use_search and not _search_fallback_used and allow_no_search:
            print(
                "Gemini: keine Text-Antwort mit Google Search — "
                "Fallback ohne Google Search (GEMINI_ALLOW_NO_SEARCH_FALLBACK). "
                "Zusatzfenster-Nachrichten können unvollständig sein.",
                file=sys.stderr,
                flush=True,
            )
            response = _generate_content_retry(
                client,
                model=model_name,
                contents=contents,
                config=cfg_plain,
                label="Fallback ohne Google Search (leere Search-Antwort)",
            )
            answer = _response_text(response)
            _search_fallback_used = True
        if not answer:
            print(
                "Gemini: keine Text-Antwort — Google Search ist für Nachrichten bis zum "
                "Scoring-Lauf erforderlich (kein Fallback ohne Search).",
                file=sys.stderr,
            )
            sys.exit(1)

        DOCS.mkdir(parents=True, exist_ok=True)
        OUT_TXT.write_text(answer, encoding="utf-8")
        if _search_fallback_used:
            _run_meta["search_fallback_used"] = True
            provider = (
                "Generiert per Gemini (File-Upload; Google Search fehlgeschlagen — "
                "Analyse ohne Web-Grounding)"
            )
        elif use_search:
            provider = "Generiert per Gemini (File-Upload + Google Search Grounding)"
        else:
            provider = "Generiert per Gemini (File-Upload; ohne Google Search)"

        OUT_HTML.write_text(
            analysis_text_to_html(
                answer,
                provider,
            ),
            encoding="utf-8",
        )
        try:
            OUT_RUN_META.write_text(
                json.dumps(_run_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"Geschrieben: {OUT_TXT}, {OUT_HTML}, {OUT_RUN_META}")
        patch_index_html_from_llm_analysis()
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
