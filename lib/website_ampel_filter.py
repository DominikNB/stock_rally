"""Kontext-Ampel-Filter und Nutzer-Legende für die statische OOS-Website (docs/index.html)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from lib.signal_context_tier import context_tier_counts, context_vix_green_min


def ampel_counts(signals: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return context_tier_counts(signals)


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
        .context-user-guide{margin-bottom:14px;padding:14px 16px}
        .context-user-guide h2{font-size:.95em;margin-bottom:8px}
        .context-user-guide details{margin-top:10px;border:1px solid #2d2d4e;border-radius:8px;background:#0d1117}
        .context-user-guide details summary{padding:10px 12px;font-size:.85em;color:#81d4fa;cursor:pointer;list-style:none}
        .context-user-guide details[open] summary{border-bottom:1px solid #2d2d4e;margin-bottom:0}
        .context-user-guide .guide-body{padding:12px 14px 14px;font-size:.8em;color:#b0bec5;line-height:1.55}
        .context-user-guide .guide-body p{margin:0 0 10px}
        .context-user-guide .guide-body h3{font-size:.88em;color:#eceff1;margin:14px 0 6px}
        .context-user-guide .guide-body h3:first-child{margin-top:0}
        .context-user-guide ul{margin:6px 0 0 1.1em;padding:0}
        .context-user-guide li{margin:4px 0}
        .context-user-guide strong{color:#eceff1}
        .context-user-guide code{font-size:.92em;color:#90caf9}
"""


def website_context_guide_html() -> str:
    """Kurze Legende zur neuen Kontext-Ampel (Makro + VIX)."""
    vix_min = context_vix_green_min()
    return f"""
      <div class="section context-user-guide" id="context-user-guide">
        <h2>Kontext-Ampel — Erklärung</h2>
        <p class="section-lead">
          Jede Signalkarte erhält eine <strong>Kontext-Ampel</strong> (grün / gelb / rot).
          Das ist <em>kein</em> zweites Modell und <em>kein</em> Ausschluss — nur eine
          OOS-validierte Einordnung des Marktumfelds am Signaltag.
        </p>

        <details open>
          <summary><strong>Die drei Stufen</strong></summary>
          <div class="guide-body">
            <h3 style="color:#ef9a9a">Rot — Makro-Risiko</h3>
            <p>
              Ein wichtiger Makro-Termin (z.&nbsp;B. FOMC, CPI, NFP) liegt innerhalb von
              <strong>±2 Handelstagen</strong> um den Signaltag
              (<code>macro_event_within_2bd = True</code>).
              Historisch waren die durchschnittlichen OOS-Renditen in diesem Umfeld schwächer.
            </p>
            <h3 style="color:#a5d6a7">Grün — Kontext gut</h3>
            <p>
              <strong>Kein</strong> Makro-Event in ±2 Handelstagen <strong>und</strong>
              VIX am Signaltag <strong>≥ {vix_min:.0f}</strong>
              (<code>regime_vix_level</code>, Yahoo <code>^VIX</code>).
              Historisch das stärkste OOS-Regime im Schnitt.
            </p>
            <h3 style="color:#fff59d">Gelb — Standard</h3>
            <p>
              Alles andere: kein Makro-Warnsignal, aber VIX unter {vix_min:.0f} — oder
              unvollständige Kalender-/VIX-Daten. Das Modell-Signal bleibt unverändert gültig.
            </p>
          </div>
        </details>

        <details>
          <summary><strong>Was die Ampel nicht ist</strong></summary>
          <div class="guide-body">
            <ul>
              <li>Kein Ersatz für die Meta-Wahrscheinlichkeit (<code>prob</code>) — höhere
                  <code>prob</code> bedeutet nicht automatisch bessere Rendite.</li>
              <li>Keine Anlageberatung — nur historischer Kontext aus dem Backtest.</li>
              <li>Kein Filter im Modell — alle Signale bleiben sichtbar; Sie filtern nur die Ansicht.</li>
            </ul>
          </div>
        </details>

        <p class="section-lead" style="margin-top:10px">
          Sortierung der Karten: <strong>neuestes Signaldatum zuerst</strong>.
          Filter oben: Alle / Grün / Gelb / Rot.
        </p>
      </div>"""


def website_ampel_filter_html(counts: Mapping[str, int]) -> str:
    n_all = int(counts.get("all", 0))
    n_red = int(counts.get("red", 0))
    n_yellow = int(counts.get("yellow", 0))
    n_green = int(counts.get("green", 0))
    return website_context_guide_html() + f"""
      <div class="section ampel-filter-bar" id="ampel-filter">
        <h2>OOS-Signale nach Kontext-Ampel</h2>
        <p class="section-lead">
          Alle <strong>{n_all}</strong> FINAL-OOS-Signale (aus <code>signals.json</code>).
          Charts werden mitgefiltert; die Tabelle listet jeden Treffer der gewählten Farbe.
        </p>
        <div class="ampel-filter-btns" role="group" aria-label="Kontext-Ampel Filter">
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
      var active = 'all';
      var signals = [];
      var btns = [];
      var statusEl = null;
      var listEl = null;
      var tbody = null;
      var bound = false;

      function tierFromSignal(s) {
        var t = (s.context_tier || s.vix_regime_ampel || '').toLowerCase();
        if (t === 'red' || t === 'yellow' || t === 'green') return t;
        return 'unknown';
      }

      function ampelFromCard(card) {
        var pre = (card.getAttribute('data-vix-ampel') || card.getAttribute('data-context-tier') || '').toLowerCase();
        if (pre === 'red' || pre === 'yellow' || pre === 'green') return pre;
        var badge = card.querySelector('.context-tier');
        if (badge) {
          var m = badge.className.match(/context-tier--(red|yellow|green)/);
          if (m) return m[1];
        }
        return 'unknown';
      }

      function initCardAmpelAttrs() {
        document.querySelectorAll('.sig-card').forEach(function (card) {
          var a = ampelFromCard(card);
          card.dataset.vixAmpel = a;
          card.dataset.contextTier = a;
        });
      }

      function filterCards(ampel) {
        var shown = 0, total = 0;
        document.querySelectorAll('.sig-card').forEach(function (card) {
          total += 1;
          var a = card.dataset.vixAmpel || card.dataset.contextTier || 'unknown';
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
          return tierFromSignal(s) === ampel;
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
          return tierFromSignal(s) === ampel;
        }).length;
        if (ampel === 'all') {
          statusEl.textContent = 'Alle OOS-Signale: ' + signals.length + ' gesamt; ' +
            cards.shown + ' Chart-Karten sichtbar.';
        } else {
          statusEl.textContent = 'Kontext ' + ampel + ': ' + n + ' OOS-Signale in der Liste; ' +
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
