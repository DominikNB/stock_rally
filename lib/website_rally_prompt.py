"""
Dynamischer Textblock für die Website-KI: erklärt die Rally-/Label-Logik aus
``models/scoring_artifacts.joblib`` (best_params + Signal-Filter), damit das LLM
kurzfristige Impulse vs. langfristige Erholungen unterscheidet.
"""
from __future__ import annotations

import numbers
from pathlib import Path


def _fmt_param(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        return f"{float(v):.6g}"
    return str(v)


def load_rally_prompt_injection(root: Path | str) -> str:
    """
    Liefert einen deutschsprachigen Markdown-Block mit den aktuellen Rally-Parametern.
    Quelle: ``root/models/scoring_artifacts.joblib`` (nach Training / Optuna identisch
    mit den festen SEED/FIXED_Y-Werten, sofern Artefakt danach gespeichert wurde).
    """
    root = Path(root)
    art = root / "models" / "scoring_artifacts.joblib"
    hdr = (
        "## Rally-Zieldefinition (automatisch aus dem Scoring-Modell)\n\n"
        "**Fokus:** Das System soll **kurzfristige Kurs-Rallies** und **Impulse** "
        "identifizieren — typischerweise im **Swing-Rahmen weniger Handelstage** "
        "(siehe „Haltedauer“ im Prompt), **nicht** langfristige "
        "Trendwenden, mehrmonatige Erholungen oder „Buy & Hold“-Narrative. "
        "Deine Einordnung soll diese **kurze** Anlaufzeit im Kopf behalten.\n\n"
    )
    if not art.is_file():
        return (
            hdr
            + "*Hinweis:* `models/scoring_artifacts.joblib` fehlt — keine konkreten "
            "Zahlenwerte. Nach vollständigem Training (Artefakt speichern) erscheinen "
            "hier die exakten Label-Parameter.\n"
        )

    import joblib

    b = joblib.load(art)
    bp = b.get("best_params")
    if not isinstance(bp, dict):
        bp = {}

    rw = bp.get("return_window")
    rt = bp.get("rally_threshold")
    ld = bp.get("lead_days")
    ed = bp.get("entry_days")
    mt = bp.get("min_rally_tail_days")

    cd = b.get("CONSECUTIVE_DAYS")
    scd = b.get("SIGNAL_COOLDOWN_DAYS")

    lines = [
        hdr,
        "Trainings-Labels („Rally“) wurden aus den **folgenden Parametern** gebildet "
        "(Optuna-Optimierung oder feste Vorgaben — im Artefakt der Stand beim Speichern):\n",
    ]
    if rw is not None:
        lines.append(
            f"- **return_window** = {_fmt_param(rw)} Handelstage: "
            f"Im Label wird ein Fenster dieser Länge betrachtet."
        )
        lines.append(
            f"**Earnings & ähnliche Termine (verbindlich):** Der trainierte "
            f"**Rally-Maximalhorizont** ist **{_fmt_param(rw)} Handelstage** "
            f"(``return_window``). **Nur** wenn in der CSV **bdays_to_next_earnings** "
            f"**≤ {_fmt_param(rw)}** ist (und nicht NaN), darfst du **Earnings** als "
            f"**gewichtetes** inhaltliches Argument (Treiber, Risiko, „Vor-den-Zahlen“-Rally, "
            f"Binary-Event) für **diese** Meta-Treffer-These nutzen. "
            f"Liegt das nächste Earnings **weiter entfernt** (>**{_fmt_param(rw)}** Handelstage) "
            f"oder fehlt der Termin: **kein** Earnings-Kernnarrativ — höchstens **ein kurzer** "
            f"Hinweis, dass der Termin **außerhalb** des Modell-Horizonts liegt und für die "
            f"trainierte Rally-Länge daher **nicht** als Erklärung dient."
        )
    else:
        lines.append(
            "- **return_window:** *(nicht im Artefakt.)* **Earnings:** Keine "
            "**return_window**-basierte Grenze — **Earnings** nicht als Kernnarrativ, wenn "
            "**bdays_to_next_earnings** offensichtlich **weit** (> 15 Handelstage) liegt; "
            "**Binary-Event** (<5 Handelstage) höchstens **kurz** erwähnen."
        )
    if rt is not None:
        lines.append(
            f"- **rally_threshold** = {_fmt_param(rt)}: "
            f"Mindest-**kumulative** Rendite (über das Fenster) für die „grüne“ Rally-Phase."
        )
    if ld is not None:
        lines.append(
            f"- **lead_days** = {_fmt_param(ld)}: Tage **vor** Rally-Start, "
            f"an denen bereits ein positives Label gesetzt werden kann (Vorlauf)."
        )
    if ed is not None:
        lines.append(
            f"- **entry_days** = {_fmt_param(ed)}: Länge des **Einstiegsfensters** "
            f"am Beginn einer Rally-Phase."
        )
    if mt is not None:
        lines.append(
            f"- **min_rally_tail_days** = {_fmt_param(mt)}: "
            f"Mindestlänge der zusammenhängenden Rally-Phase (grünes Band)."
        )

    if not any(x is not None for x in (rw, rt, ld, ed, mt)):
        lines.append(
            "- *(Keine Rally-Kernparameter in `best_params` gefunden — älteres Artefakt.)*"
        )

    if cd is not None or scd is not None:
        lines.append("")
        lines.append("**Signal-Logik (Meta-Treffer, nicht Label):**")
        if cd is not None:
            lines.append(
                f"- **CONSECUTIVE_DAYS** = {int(cd)}: Anforderung an aufeinanderfolgende Treffer."
            )
        if scd is not None:
            lines.append(
                f"- **SIGNAL_COOLDOWN_DAYS** = {int(scd)}: Mindestabstand zwischen Signalen."
            )

    lines.append("")
    lines.append(
        "**Konsequenz für die Textantwort:** Schwerpunkt auf **Nachrichten und Ereignisse** "
        "(Unternehmens-, Branchen-, Wirtschafts- und Marktmeldungen), die kurzfristige Impulse "
        "typischerweise tragen; **technische Indikatoren** nur ergänzend — sie sind im Modell "
        "bereits über Kursdaten abgebildet. **Langfristige** Themen nur, wenn sie den **nahen** "
        "Verlauf plausibel stützen — nicht als „Recovery“-Story über viele Monate."
    )

    return "\n".join(lines)
