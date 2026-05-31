"""VIX-Ampel-Filter und Nutzer-Legende für die statische OOS-Website (docs/index.html)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def _guide_thresholds() -> tuple[float, float, float, float]:
    from lib.vix_regime_ampel import vix_ampel_thresholds
    from lib.vix_red_context_chips import _chip_thresholds

    y_min, g_min = vix_ampel_thresholds()
    chip = _chip_thresholds()
    return y_min, g_min, float(chip["vix3m_vix_max"]), float(chip["sector_hhi_max"])


def ampel_counts(signals: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c = Counter(str(s.get("vix_regime_ampel") or "unknown").strip().lower() for s in signals)
    return {
        "all": len(signals),
        "red": int(c.get("red", 0)),
        "yellow": int(c.get("yellow", 0)),
        "green": int(c.get("green", 0)),
        "unknown": int(c.get("unknown", 0)),
    }


def website_ampel_filter_css_block() -> str:
    return """
        .ampel-filter-bar{margin-bottom:14px;padding:14px 16px}
        .ampel-filter-bar h2{font-size:.95em;margin-bottom:10px}
        .ampel-filter-btns{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
        .ampel-filter-btn{font:inherit;cursor:pointer;border-radius:8px;padding:8px 14px;font-size:.82em;font-weight:600;border:1px solid #37474f;background:#0d1117;color:#b0bec5;transition:background .15s,border-color .15s,transform .1s}
        .ampel-filter-btn:hover{border-color:#546e7a;color:#eceff1}
        .ampel-filter-btn.is-active{transform:translateY(-1px);color:#fff}
        .ampel-filter-btn--all.is-active{background:#263238;border-color:#90a4ae}
        .ampel-filter-btn--red.is-active{background:#3d1f1f;border-color:#c62828;color:#ef9a9a}
        .ampel-filter-btn--yellow.is-active{background:#3d3520;border-color:#f9a825;color:#fff59d}
        .ampel-filter-btn--green.is-active{background:#1b3d24;border-color:#43a047;color:#a5d6a7}
        .ampel-filter-status{font-size:.78em;color:#78909c;margin:0 0 8px;line-height:1.45}
        .ampel-filter-list{max-height:min(52vh,520px);overflow:auto;border:1px solid #2d2d4e;border-radius:8px;background:#0d1117}
        .ampel-filter-list[hidden]{display:none}
        .ampel-filter-table{width:100%;border-collapse:collapse;font-size:.78em}
        .ampel-filter-table th,.ampel-filter-table td{padding:6px 10px;text-align:left;border-bottom:1px solid #1e2a3a}
        .ampel-filter-table th{position:sticky;top:0;background:#16213e;color:#81d4fa;font-weight:600;z-index:1}
        .ampel-filter-table tr:hover td{background:#12121f}
        .ampel-filter-table .col-date{color:#90a4ae;white-space:nowrap}
        .ampel-filter-table .col-ticker{color:#81d4fa;font-weight:600;white-space:nowrap}
        .ampel-filter-table .col-prob{color:#a5d6a7;white-space:nowrap}
        .ampel-filter-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
        .ampel-filter-dot--red{background:#ef5350}
        .ampel-filter-dot--yellow{background:#ffca28}
        .ampel-filter-dot--green{background:#66bb6a}
        .vix-user-guide{margin-bottom:14px;padding:14px 16px}
        .vix-user-guide h2{font-size:.95em;margin-bottom:8px}
        .vix-user-guide details{margin-top:10px;border:1px solid #2d2d4e;border-radius:8px;background:#0d1117}
        .vix-user-guide details summary{padding:10px 12px;font-size:.85em;color:#81d4fa;cursor:pointer;list-style:none}
        .vix-user-guide details[open] summary{border-bottom:1px solid #2d2d4e;margin-bottom:0}
        .vix-user-guide .guide-body{padding:12px 14px 14px;font-size:.8em;color:#b0bec5;line-height:1.55}
        .vix-user-guide .guide-body p{margin:0 0 10px}
        .vix-user-guide .guide-body h3{font-size:.88em;color:#eceff1;margin:14px 0 6px}
        .vix-user-guide .guide-body h3:first-child{margin-top:0}
        .vix-user-guide .guide-chip{margin:10px 0;padding:10px 12px;border-radius:8px;border:1px solid #37474f;background:#12121f}
        .vix-user-guide .guide-chip h4{font-size:.84em;color:#81d4fa;margin:0 0 6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
        .vix-user-guide .chip-swatch{font-size:.68em;padding:2px 7px;border-radius:6px;font-weight:500}
        .vix-user-guide .chip-swatch--good{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .vix-user-guide .chip-swatch--warn{background:#3d2a1a;color:#ffcc80;border:1px solid #ef6c00}
        .vix-user-guide .chip-swatch--na{background:#263238;color:#78909c;border:1px solid #455a64}
        .vix-user-guide ul{margin:6px 0 0 1.1em;padding:0}
        .vix-user-guide li{margin:4px 0}
        .vix-user-guide strong{color:#eceff1}
        .vix-user-guide code{font-size:.92em;color:#90caf9}
"""


def website_vix_guide_html() -> str:
    """Ausführliche Legende: VIX, Ampel, Rot-Chips (Berechnung + Bedeutung)."""
    y_min, g_min, vix3m_max, hhi_max = _guide_thresholds()
    return f"""
      <div class="section vix-user-guide" id="vix-user-guide">
        <h2>VIX-Ampel &amp; Kontext-Chips — Was bedeutet das?</h2>
        <p class="section-lead">
          Kurz erklärt für die Signalkarten unten. <strong>Kein zweites Scoring</strong> und
          <strong>kein Filter</strong> — nur Einordnung des Marktumfelds am Signaltag.
        </p>

        <details open>
          <summary><strong>Was ist der VIX?</strong> und die drei Ampel-Farben</summary>
          <div class="guide-body">
            <p>
              Der <strong>VIX</strong> („Volatility Index“, oft CBOE VIX, Symbol <code>^VIX</code>)
              misst die <strong>erwartete Schwankung</strong> des US-Aktienmarkts (S&amp;P-500-Optionen),
              nicht den Kurs selbst. Hoher VIX = mehr Angst/Unsicherheit, niedriger VIX = ruhigeres Umfeld.
              Pro Signal zeigen wir den <strong>VIX-Schlusskurs am Signaltag</strong> (Yahoo Finance).
            </p>
            <p>
              Die <strong>Ampel</strong> ordnet dieses Niveau in <strong>drei Regime</strong> ein
              (historisch auf Trainings-/Testdaten kalibriert — Signale werden <em>nicht</em> ausgeschlossen):
            </p>
            <ul>
              <li><strong style="color:#a5d6a7">Grün</strong> — VIX <strong>≥ {g_min:.0f}</strong>:
                historisch das <strong>stärkste</strong> Gesamtregime für die Modell-Signale im Schnitt.</li>
              <li><strong style="color:#fff59d">Gelb</strong> — VIX <strong>{y_min:.0f} bis unter {g_min:.0f}</strong>:
                mittleres Regime.</li>
              <li><strong style="color:#ef9a9a">Rot</strong> — VIX <strong>&lt; {y_min:.0f}</strong>:
                historisch <strong>schwächeres</strong> Gesamtregime — viele Einzeltreffer sind trotzdem möglich,
                aber das Umfeld war im Backtest im Schnitt weniger günstig.</li>
            </ul>
            <p>
              An jeder Karte: kleine <strong>Skala + drei Lichter</strong> (rot/gelb/grün) und der konkrete
              VIX-Wert (z. B. „VIX 15,3“). Der Filter oben gruppiert alle OOS-Signale nach dieser Farbe.
            </p>
          </div>
        </details>

        <details>
          <summary><strong>Die vier Zusatz-Chips</strong> (nur bei Ampel <em>rot</em>)</summary>
          <div class="guide-body">
            <p>
              Wenn die Ampel <strong>rot</strong> ist, erscheinen unter dem Kopf der Karte bis zu
              <strong>vier Chips</strong>. Sie sind aus OOS-Validierung (META+THRESHOLD + FINAL) als
              <strong>Zusatz-Kontext</strong> gewählt — sie ersetzen <em>nicht</em> die Modell-Wahrscheinlichkeit
              (<code>prob</code>) und schließen kein Signal aus.
            </p>
            <p>
              <span class="chip-swatch chip-swatch--good">Grün</span> = in rot historisch <strong>günstiger</strong> ·
              <span class="chip-swatch chip-swatch--warn">Orange</span> = eher <strong>Vorsicht</strong> ·
              <span class="chip-swatch chip-swatch--na">Grau</span> = Daten fehlen → Chip ignorieren.
            </p>

            <div class="guide-chip">
              <h4>1. VIX vs. 20d-Mittel</h4>
              <p><strong>Was:</strong> Liegt der VIX <em>unter</em> oder <em>über</em> seinem eigenen
              20-Handelstage-Mittel am Signaltag?</p>
              <p><strong>Berechnung:</strong> Z-Score <code>regime_vix_z_20d</code> =
              (VIX heute − Mittel der letzten 20 Börsentage) / Standardabweichung dieser 20 Tage.
              <strong>Grün</strong> wenn Z &lt; 0 (VIX unter Mittel), <strong>Orange</strong> wenn Z ≥ 0.</p>
              <p><strong>Was wir daraus sagen:</strong> In rot war ein <strong>relativ entspannter</strong> VIX
              (unter dem kurzfristigen Mittel) historisch etwas günstiger — trotz niedrigem absolutem Niveau (&lt;20).</p>
            </div>

            <div class="guide-chip">
              <h4>2. VIX-Term 3M/VIX</h4>
              <p><strong>Was:</strong> Ist die <strong>Terminstruktur</strong> der Volatilität eher entspannt
              oder angespannt? (Kurzlaufiger VIX vs. 3-Monats-VIX.)</p>
              <p><strong>Berechnung:</strong> Ratio <code>vix3m_vix_ratio</code> = VIX 3 Monate (<code>^VIX3M</code>)
              / VIX Spot (<code>^VIX</code>) am Signaltag.
              <strong>Grün</strong> wenn Ratio &lt; <strong>{vix3m_max:.2f}</strong>,
              <strong>Orange</strong> wenn ≥ {vix3m_max:.2f} (hohe Ratio = Kurzlaufiges Stressniveau hoch vs. längere Frist).</p>
              <p><strong>Was wir daraus sagen:</strong> In rot war eine <strong>entspannte</strong> Termstruktur
              historisch günstiger; <strong>Angespannt</strong> → eher skeptisch bei reiner Mitläufer-Story.</p>
            </div>

            <div class="guide-chip">
              <h4>3. Sektor-Crowding</h4>
              <p><strong>Was:</strong> Sind am selben Tag viele Meta-Treffer im <strong>gleichen Sektor</strong>
              („alle kaufen Tech“) oder ist der Tag breiter gestreut?</p>
              <p><strong>Berechnung:</strong> <code>sector_hhi_same_day</code> = Herfindahl-Index der Sektoranteile
              unter allen Meta-Hits am Signaltag: Summe (Anteil Sektor)².
              1,0 = nur ein Sektor; niedrig = viele Sektoren.
              <strong>Grün</strong> wenn HHI &lt; <strong>{hhi_max:.2f}</strong>,
              <strong>Orange</strong> wenn ≥ {hhi_max:.2f}.</p>
              <p><strong>Was wir daraus sagen:</strong> Wenig Crowding spricht eher für eine <strong>Einzel-Titel-Idee</strong>;
              viel Crowding → gemeinsame Sektor-/Marktbewegung, im Vergleich der Signale genauer abgrenzen.</p>
            </div>

            <div class="guide-chip">
              <h4>4. News Sektor vs. Makro</h4>
              <p><strong>Was:</strong> Ist der <strong>Ton der Sektor-News</strong> positiver als der
              <strong>Makro-News</strong>-Ton (aus der News-Pipeline des Modells)?</p>
              <p><strong>Berechnung:</strong> Differenz Sektor-Ton minus Makro-Ton
              (<code>news_sec_*_tone</code> − <code>news_macro_*_tone</code>, gleicher Tag).
              <strong>Grün</strong> wenn Differenz &gt; 0, <strong>Orange</strong> wenn ≤ 0.
              <strong>Grau</strong>, wenn keine News-Scores verfügbar.</p>
              <p><strong>Was wir daraus sagen:</strong> In rot war ein <strong>sektorgetriebener</strong> News-Hintergrund
              historisch etwas günstiger als reine Makro-Dominanz — immer mit den <strong>belegten</strong> Meldungen
              aus der KI-/eigenen Recherche abgleichen.</p>
            </div>

            <p style="margin-top:12px">
              <strong>Nutzung:</strong> Zuerst Signal (Setup, News, Kurs) bewerten — dann Chips als
              <strong>Zusatz-Risiko-/Kontext-Check</strong>. Mehr grüne Chips → etwas mehr Überzeugung in rot;
              viele orange → vorsichtiger (Size, Stop). Keine Anlageberatung.
            </p>
          </div>
        </details>
      </div>"""


def website_ampel_filter_html(counts: Mapping[str, int]) -> str:
    n_all = int(counts.get("all", 0))
    n_red = int(counts.get("red", 0))
    n_yellow = int(counts.get("yellow", 0))
    n_green = int(counts.get("green", 0))
    return website_vix_guide_html() + f"""
      <div class="section ampel-filter-bar" id="ampel-filter">
        <h2>OOS-Signale nach VIX-Ampel</h2>
        <p class="section-lead">
          Alle <strong>{n_all}</strong> FINAL-OOS-Signale (aus <code>signals.json</code>).
          Charts unten werden mitgefiltert; die Tabelle listet jeden Treffer der gewählten Farbe.
        </p>
        <div class="ampel-filter-btns" role="group" aria-label="VIX-Ampel Filter">
          <button type="button" class="ampel-filter-btn ampel-filter-btn--all is-active" data-ampel="all">Alle ({n_all})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--red" data-ampel="red">Rot ({n_red})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--yellow" data-ampel="yellow">Gelb ({n_yellow})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--green" data-ampel="green">Grün ({n_green})</button>
        </div>
        <p class="ampel-filter-status" id="ampel-filter-status" aria-live="polite"></p>
        <div class="ampel-filter-list" id="ampel-filter-list" hidden>
          <table class="ampel-filter-table">
            <thead><tr>
              <th>Datum</th><th>Ticker</th><th>Unternehmen</th><th>Sektor</th><th>prob</th>
            </tr></thead>
            <tbody id="ampel-filter-tbody"></tbody>
          </table>
        </div>
      </div>"""


def website_ampel_filter_js_block() -> str:
    return """
    <script id="ampel-filter-js">
    (function () {
      var active = "all";
      var signals = [];
      var btns = document.querySelectorAll(".ampel-filter-btn");
      var statusEl = document.getElementById("ampel-filter-status");
      var listEl = document.getElementById("ampel-filter-list");
      var tbody = document.getElementById("ampel-filter-tbody");

      function ampelFromCard(card) {
        var el = card.querySelector("[class*='vix-ampel--']");
        if (el) {
          var m = el.className.match(/vix-ampel--(red|yellow|green)/);
          if (m) return m[1];
        }
        var lights = card.querySelectorAll(".vix-light.is-active");
        for (var i = 0; i < lights.length; i++) {
          var lm = lights[i].className.match(/vix-light--(red|yellow|green)/);
          if (lm) return lm[1];
        }
        var pre = (card.getAttribute("data-vix-ampel") || "").toLowerCase();
        if (pre === "red" || pre === "yellow" || pre === "green") return pre;
        return "unknown";
      }

      function initCardAmpelAttrs() {
        document.querySelectorAll(".sig-card").forEach(function (card) {
          card.dataset.vixAmpel = ampelFromCard(card);
        });
      }

      function filterCards(ampel) {
        var shown = 0, total = 0;
        document.querySelectorAll(".sig-card").forEach(function (card) {
          total += 1;
          var a = card.dataset.vixAmpel || "unknown";
          var ok = ampel === "all" || a === ampel;
          card.style.display = ok ? "" : "none";
          if (ok) shown += 1;
        });
        return { shown: shown, total: total };
      }

      function esc(s) {
        return String(s == null ? "" : s)
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
      }

      function renderList(ampel) {
        if (!tbody || !listEl) return;
        if (ampel === "all") {
          listEl.hidden = true;
          tbody.innerHTML = "";
          return;
        }
        var rows = signals.filter(function (s) {
          return (s.vix_regime_ampel || "").toLowerCase() === ampel;
        });
        rows.sort(function (a, b) {
          return (b.date || "").localeCompare(a.date || "") || (a.ticker || "").localeCompare(b.ticker || "");
        });
        var dot = "ampel-filter-dot ampel-filter-dot--" + ampel;
        tbody.innerHTML = rows.map(function (s) {
          var sec = s.gics_sector || s.sector || "—";
          var prob = typeof s.prob === "number" ? s.prob.toFixed(3) : (s.prob || "—");
          return "<tr><td class=\"col-date\">" + esc(s.date) + "</td><td class=\"col-ticker\">"
            + "<span class=\"" + dot + "\" aria-hidden=\"true\"></span>" + esc(s.ticker)
            + "</td><td>" + esc(s.company || s.ticker) + "</td><td>" + esc(sec)
            + "</td><td class=\"col-prob\">" + esc(prob) + "</td></tr>";
        }).join("");
        listEl.hidden = rows.length === 0;
      }

      function apply(ampel) {
        active = ampel;
        btns.forEach(function (b) {
          b.classList.toggle("is-active", b.getAttribute("data-ampel") === ampel);
        });
        initCardAmpelAttrs();
        var cards = filterCards(ampel);
        renderList(ampel);
        var n = signals.filter(function (s) {
          return (s.vix_regime_ampel || "").toLowerCase() === ampel;
        }).length;
        if (ampel === "all") {
          statusEl.textContent = "Alle OOS-Signale: " + signals.length + " gesamt; "
            + cards.shown + " Chart-Karten sichtbar.";
        } else {
          statusEl.textContent = "Ampel " + ampel + ": " + n + " OOS-Signale in der Liste; "
            + cards.shown + " von " + cards.total + " Chart-Karten passen.";
        }
      }

      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          apply(btn.getAttribute("data-ampel") || "all");
        });
      });

      function boot() {
        fetch("signals.json", { cache: "no-store" })
          .then(function (r) {
            if (!r.ok) throw new Error("signals.json " + r.status);
            return r.json();
          })
          .then(function (data) {
            signals = (data && data.signals) ? data.signals : (Array.isArray(data) ? data : []);
            apply(active);
          })
          .catch(function () {
            initCardAmpelAttrs();
            apply(active);
            if (statusEl) statusEl.textContent += " (signals.json nicht geladen — nur Chart-Filter.)";
          });
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
      } else {
        boot();
      }
    })();
    </script>"""
