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
        <h2>VIX-Ampel &amp; Kontext-Chips — Erklärung</h2>
        <p class="section-lead">
          Für jede Signalkarte: Was die Begriffe <strong>bedeuten</strong>, wie die Zahl
          <strong>berechnet</strong> wird und wie Sie den Chip <strong>deuten</strong> können.
          Kein zweites Scoring — nur Kontext am Signaltag.
        </p>

        <details open>
          <summary><strong>VIX &amp; Ampel</strong> (rot / gelb / grün)</summary>
          <div class="guide-body">
            <p><strong>Was ist der VIX?</strong><br>
            Der VIX (CBOE Volatility Index, Yahoo: <code>^VIX</code>) ist ein Index aus
            <strong>S&amp;P-500-Optionspreisen</strong>. Er misst, wie stark der Markt
            <strong>kurzfristige Kursausschläge</strong> erwartet — nicht die Richtung (hoch/runter).
            VIX <strong>hoch</strong> ≈ viel Angst/Unsicherheit; VIX <strong>niedrig</strong> ≈ ruhigeres Umfeld.
            Auf der Karte: <strong>VIX-Schlusskurs am Signaltag</strong> (z. B. 15,3).</p>
            <p><strong>Was ist die Ampel?</strong><br>
            Wir ordnen nur dieses eine VIX-Niveau am Signaltag in <strong>drei Stufen</strong> ein
            (aus Backtest/OOS kalibriert — <em>kein</em> Ausschluss von Signalen):</p>
            <ul>
              <li><strong style="color:#ef9a9a">Rot</strong> — VIX <strong>unter {y_min:.0f}</strong>
                (z. B. 15): historisch schwächeres Gesamtregime für die Modell-Signale.</li>
              <li><strong style="color:#fff59d">Gelb</strong> — VIX zwischen <strong>{y_min:.0f}</strong>
                und <strong>{g_min:.0f}</strong>: mittleres Regime.</li>
              <li><strong style="color:#a5d6a7">Grün</strong> — VIX <strong>{g_min:.0f} oder höher</strong>:
                historisch stärkstes Regime im Schnitt.</li>
            </ul>
            <p><strong>So deuten:</strong> Die Ampel beschreibt das <strong>große Markt-Wetter</strong>.
            Ein rot markiertes Signal heißt nicht „schlecht“, sondern: Im Backtest war dieses
            Umfeld im Schnitt weniger günstig — die drei Chips unten verfeinern das nur bei rot.</p>
          </div>
        </details>

        <details open>
          <summary><strong>Chip 1:</strong> VIX vs. 20d-Mittel</summary>
          <div class="guide-body">
            <div class="guide-chip">
              <p><strong>Was der Name meint:</strong><br>
              „20d-Mittel“ = <strong>Mittelwert des VIX</strong> über die letzten
              <strong>20 Handelstage</strong> vor dem Signaltag (einfacher Durchschnitt der Schlusskurse).
              Der Chip vergleicht: Ist der VIX <strong>heute eher unter oder über</strong> diesem
              kurzfristigen Normalniveau?</p>
              <p><strong>So wird gerechnet:</strong><br>
              1) Mittel = Durchschnitt(VIX-Tag 1 … Tag 20 bis Signaltag)<br>
              2) Streuung = Standardabweichung derselben 20 Tage<br>
              3) <code>regime_vix_z_20d</code> = (VIX am Signaltag − Mittel) ÷ Streuung
              (ein Z-Score, „wie viele Standardabweichungen vom Mittel“)</p>
              <p><strong>Chip-Farbe:</strong><br>
              <span class="chip-swatch chip-swatch--good">Grün</span> wenn Z <strong>&lt; 0</strong>
              (VIX <em>unter</em> dem 20-Tage-Mittel — kurzfristig relativ ruhiger).<br>
              <span class="chip-swatch chip-swatch--warn">Orange</span> wenn Z <strong>≥ 0</strong>
              (VIX <em>über</em> dem Mittel — kurzfristig angespannter).<br>
              <span class="chip-swatch chip-swatch--na">Grau</span> wenn zu wenig VIX-Historie.</p>
              <p><strong>Beispiel:</strong> VIX heute 15,3; 20-Tage-Mittel 16,0 → Z negativ → grüner Chip:
              trotz niedrigem absolutem VIX (&lt;20, Ampel rot) ist der VIX <em>relativ</em> gefallen.</p>
              <p><strong>So deuten:</strong> In der Ampel-Stufe „rot“ war „VIX unter seinem eigenen
              20-Tage-Schnitt“ im Backtest etwas günstiger — ein kleines Plus für Überzeugung,
              kein Freifahrtschein.</p>
            </div>
          </div>
        </details>

        <details>
          <summary><strong>Chip 2:</strong> VIX-Term 3M/VIX</summary>
          <div class="guide-body">
            <div class="guide-chip">
              <p><strong>Was der Name meint:</strong><br>
              „Term“ = <strong>Zeitstruktur</strong> der erwarteten Volatilität.
              <strong>VIX</strong> (Spot) ≈ erwartete Schwankung für die <strong>nächsten ~30 Tage</strong>.
              <strong>VIX 3M</strong> (<code>^VIX3M</code>) ≈ erwartete Schwankung für etwa
              <strong>3 Monate</strong>. Die Ratio vergleicht: Ist die <strong>kurzfristige</strong>
              Angst höher oder niedriger als die <strong>längerfristige</strong>?</p>
              <p><strong>So wird gerechnet:</strong><br>
              <code>vix3m_vix_ratio</code> = Schlusskurs <code>^VIX3M</code> ÷ Schlusskurs
              <code>^VIX</code> (beide am Signaltag, Yahoo Finance).</p>
              <p><strong>Chip-Farbe:</strong><br>
              <span class="chip-swatch chip-swatch--good">Grün</span> wenn Ratio <strong>&lt; {vix3m_max:.2f}</strong>
              (3-Monats-Vola nicht deutlich über Spot — Terminstruktur eher „entspannt“).<br>
              <span class="chip-swatch chip-swatch--warn">Orange</span> wenn Ratio <strong>≥ {vix3m_max:.2f}</strong>
              (Spot-VIX hoch vs. 3M — kurzfristiger Stress dominiert).<br>
              <span class="chip-swatch chip-swatch--na">Grau</span> wenn VIX3M oder VIX fehlt.</p>
              <p><strong>Beispiel:</strong> VIX = 18, VIX3M = 20 → Ratio 1,11 → grün.
              VIX = 18, VIX3M = 23 → Ratio 1,28 → orange.</p>
              <p><strong>So deuten:</strong> Orange = Markt ist <strong>kurzfristig</strong> nervöser als
              der längere Horizont; bei reiner Mitläufer-Story eher skeptisch. Grün = kein akuter
              „Stress-Spike“ in der Kurve.</p>
            </div>
          </div>
        </details>

        <details>
          <summary><strong>Chip 3:</strong> Sektor-Crowding</summary>
          <div class="guide-body">
            <div class="guide-chip">
              <p><strong>Was der Name meint:</strong><br>
              „Crowding“ = <strong>gedrängte Signale</strong>: Am selben Signaltag feuern viele
              Meta-Treffer gleichzeitig — vor allem im <strong>selben Sektor</strong> (z. B. fünf
              Tech-Titel). Dann bewegt sich eher der <strong>ganze Sektor</strong>, nicht unbedingt
              eine Einzelstory.</p>
              <p><strong>So wird gerechnet:</strong><br>
              Nur Treffer mit gleichem Kalenderdatum (<code>signals_same_day</code>).
              Pro Sektor: Anteil = (Anzahl Signale in diesem Sektor) ÷ (alle Signale dieses Tages).<br>
              <code>sector_hhi_same_day</code> = Summe über alle Sektoren von (Anteil)²
              (Herfindahl-Index, HHI).<br>
              • HHI nahe <strong>1,0</strong> = fast alles in einem Sektor.<br>
              • HHI <strong>niedrig</strong> (z. B. 0,2) = viele Sektoren beteiligt.</p>
              <p><strong>Chip-Farbe:</strong><br>
              <span class="chip-swatch chip-swatch--good">Grün</span> wenn HHI <strong>&lt; {hhi_max:.2f}</strong>
              (wenig Sektor-Bündelung).<br>
              <span class="chip-swatch chip-swatch--warn">Orange</span> wenn HHI <strong>≥ {hhi_max:.2f}</strong>
              (viel Crowding).<br>
              <span class="chip-swatch chip-swatch--na">Grau</span> wenn nur ein Treffer am Tag (HHI nicht sinnvoll).</p>
              <p><strong>Beispiel:</strong> 10 Signale am Tag: 8× Tech, 1× Finance, 1× Health →
              hoher HHI → orangener Chip. 10 Signale auf 6 verschiedene Sektoren verteilt → niedriger HHI → grün.</p>
              <p><strong>So deuten:</strong> Grün spricht für eine <strong>einzelne</strong> Idee;
              orange: prüfen, ob Ihre Story wirklich <strong>ticker-spezifisch</strong> ist oder nur
              Sektor-/Markt-Mitlauf (Vergleich mit anderen Treffern des Tages).</p>
            </div>
          </div>
        </details>

        <details open>
          <summary><strong>Rot-Qualität</strong> (OOS-validiertes Badge)</summary>
          <div class="guide-body">
            <p><strong>Was ist das?</strong><br>
            Zusätzlich zu den drei Kontext-Chips erscheint bei <strong>roter Ampel</strong> ein
            Qualitäts-Badge (<em>hoch / mittel / niedrig</em>). Es nutzt nur Faktoren, die in
            META+THR und FINAL-OOS historisch stabil waren — kein zweites Modell-Scoring.</p>
            <div class="guide-chip">
              <h4><span class="red-quality-badge red-quality-badge--hoch" style="font-size:.75em">Qualität hoch (2/2)</span></h4>
              <p><strong>Faktor 1 — Liquidität ok:</strong><br>
              <code>liquidity_tier == ok</code> (ADV-Perzentil am Signaltag nicht „dünn“).
              OOS: +1,0 pp vs. dünnere Titel im rot-Regime (FINAL n≈111).</p>
              <p><strong>Faktor 2 — Gold schwach 5d:</strong><br>
              <code>gld_ret_5d</code> (GLD, Yahoo) liegt <strong>unter dem Tages-Median</strong>
              aller Signale am selben Datum. Kein Flucht-in-Gold / Risk-off-Kontext.
              OOS: +1,0 pp (FINAL).</p>
              <p><strong>Stufen:</strong> 2/2 = hoch, 1/2 = mittel, 0/2 = niedrig.
              Fehlende Daten → Badge ggf. ausgeblendet.</p>
              <p><strong>Nicht im Score (nur Tooltip):</strong> Alpha vs. Markt/Sektor
              (<code>alpha_sec_5d</code> / <code>alpha_mkt_5d</code>) — in IS oft positiv,
              FINAL nicht stabil genug für einen Filter.</p>
            </div>
            <p><strong>So deuten:</strong> Das Badge sortiert <em>innerhalb</em> rot nach
            historisch robusteren Merkmalen — es ersetzt weder die Ampel noch die Chips und
            ist <strong>keine Anlageberatung</strong>.</p>
          </div>
        </details>

        <p class="section-lead" style="margin-top:10px">
          <strong>Alle drei Chips</strong> erscheinen nur, wenn die Ampel <strong>rot</strong> ist.
          <span class="chip-swatch chip-swatch--good">Grün</span>/<span class="chip-swatch chip-swatch--warn">Orange</span>:
          aus OOS-Tests in rot historisch günstiger/ungünstiger — <strong>keine Anlageberatung</strong>.
        </p>
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
    # HTML-Strings in JS mit einfachen Anführungszeichen (kein \" in index.html nötig).
    return """
    <script id="ampel-filter-js">
    (function () {
      var active = 'all';
      var signals = [];
      var btns = [];
      var statusEl = null;
      var listEl = null;
      var tbody = null;
      var bound = false;

      function ampelFromCard(card) {
        var lights = card.querySelectorAll('.vix-light.is-active');
        var i, lm;
        for (i = 0; i < lights.length; i++) {
          lm = lights[i].className.match(/vix-light--(red|yellow|green)/);
          if (lm) return lm[1];
        }
        var pre = (card.getAttribute('data-vix-ampel') || '').toLowerCase();
        if (pre === 'red' || pre === 'yellow' || pre === 'green') return pre;
        return 'unknown';
      }

      function initCardAmpelAttrs() {
        document.querySelectorAll('.sig-card').forEach(function (card) {
          card.dataset.vixAmpel = ampelFromCard(card);
        });
      }

      function filterCards(ampel) {
        var shown = 0, total = 0;
        document.querySelectorAll('.sig-card').forEach(function (card) {
          total += 1;
          var a = card.dataset.vixAmpel || 'unknown';
          var ok = ampel === 'all' || a === ampel;
          card.style.display = ok ? '' : 'none';
          if (ok) shown += 1;
        });
        return { shown: shown, total: total };
      }

      function esc(s) {
        return String(s == null ? '' : s)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      }

      function renderList(ampel) {
        if (!tbody || !listEl) return;
        if (ampel === 'all') {
          listEl.hidden = true;
          tbody.innerHTML = '';
          return;
        }
        var rows = signals.filter(function (s) {
          return (s.vix_regime_ampel || '').toLowerCase() === ampel;
        });
        rows.sort(function (a, b) {
          return (b.date || '').localeCompare(a.date || '') ||
            (a.ticker || '').localeCompare(b.ticker || '');
        });
        var dot = 'ampel-filter-dot ampel-filter-dot--' + ampel;
        var html = [];
        rows.forEach(function (s) {
          var sec = s.gics_sector || s.sector || '—';
          var prob = typeof s.prob === 'number' ? s.prob.toFixed(3) : (s.prob || '—');
          html.push(
            '<tr><td class="col-date">' + esc(s.date) + '</td>' +
            '<td class="col-ticker"><span class="' + dot + '" aria-hidden="true"></span>' +
            esc(s.ticker) + '</td><td>' + esc(s.company || s.ticker) + '</td><td>' +
            esc(sec) + '</td><td class="col-prob">' + esc(prob) + '</td></tr>'
          );
        });
        tbody.innerHTML = html.join('');
        listEl.hidden = rows.length === 0;
      }

      function apply(ampel) {
        active = ampel;
        btns.forEach(function (b) {
          b.classList.toggle('is-active', b.getAttribute('data-ampel') === ampel);
        });
        initCardAmpelAttrs();
        var cards = filterCards(ampel);
        renderList(ampel);
        if (!statusEl) return;
        var n = signals.filter(function (s) {
          return (s.vix_regime_ampel || '').toLowerCase() === ampel;
        }).length;
        if (ampel === 'all') {
          statusEl.textContent = 'Alle OOS-Signale: ' + signals.length + ' gesamt; ' +
            cards.shown + ' Chart-Karten sichtbar.';
        } else {
          statusEl.textContent = 'Ampel ' + ampel + ': ' + n + ' OOS-Signale in der Liste; ' +
            cards.shown + ' von ' + cards.total + ' Chart-Karten passen.';
        }
      }

      function bindUi() {
        var root = document.getElementById('ampel-filter');
        if (!root) return;
        statusEl = document.getElementById('ampel-filter-status');
        listEl = document.getElementById('ampel-filter-list');
        tbody = document.getElementById('ampel-filter-tbody');
        btns = root.querySelectorAll('.ampel-filter-btn');
        if (!bound) {
          btns.forEach(function (btn) {
            btn.addEventListener('click', function () {
              apply(btn.getAttribute('data-ampel') || 'all');
            });
          });
          bound = true;
        }
      }

      function boot() {
        bindUi();
        fetch('signals.json', { cache: 'no-store' })
          .then(function (r) {
            if (!r.ok) throw new Error('signals.json ' + r.status);
            return r.json();
          })
          .then(function (data) {
            signals = (data && data.signals) ? data.signals : (Array.isArray(data) ? data : []);
            apply(active);
          })
          .catch(function () {
            initCardAmpelAttrs();
            apply(active);
            if (statusEl) {
              statusEl.textContent = (statusEl.textContent || '') +
                ' (signals.json nicht geladen — nur Chart-Filter.)';
            }
          });
      }

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
      } else {
        boot();
      }
    })();
    </script>"""
