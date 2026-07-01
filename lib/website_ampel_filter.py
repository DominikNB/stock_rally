"""Kontext-Ampel-Filter und Nutzer-Legende für die statische OOS-Website (docs/index.html)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from lib.signal_context_tier import (
    context_tier_counts,
    context_vix3m_ratio_yellow_risk_min,
    context_vix_green_min,
    context_vix_macro_orange_min,
)


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
        .ampel-filter-btn--orange.is-active{background:#3d2a14;border-color:#fb8c00;color:#ffcc80}
        .ampel-filter-btn--yellow.is-active{background:#3d3520;border-color:#f9a825;color:#fff59d}
        .ampel-filter-btn--green.is-active{background:#1b3d24;border-color:#43a047;color:#a5d6a7}
        .ampel-filter-btn--trade.is-active{background:#1a3320;border-color:#66bb6a;color:#c8e6c9}
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
        .ampel-filter-dot--orange{background:#ffa726}
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
    """Kurze Legende zur Kontext-Ampel (Makro × VIX, OOS Jun 2026)."""
    vix_min = context_vix_green_min()
    macro_min = context_vix_macro_orange_min()
    ratio_min = context_vix3m_ratio_yellow_risk_min()
    return f"""
      <div class="section context-user-guide" id="context-user-guide">
        <h2>Kontext-Ampel — Erklärung</h2>
        <p class="section-lead">
          Jede Signalkarte erhält eine <strong>Kontext-Ampel</strong> mit
          <strong>fünf Stufen</strong> (grün / orange / rot / gelb / gelb-risiko).
          Das ist <em>kein</em> zweites Modell und <em>kein</em> Ausschluss — nur eine
          OOS-validierte Einordnung des Marktumfelds am Signaltag.
        </p>

        <details open>
          <summary><strong>Die fünf Stufen (OOS-validiert)</strong></summary>
          <div class="guide-body">
            <h3 style="color:#a5d6a7">1. Grün — Kontext gut</h3>
            <p>
              <strong>Kein</strong> Makro-Event in ±2 Handelstagen <strong>und</strong>
              VIX ≥ <strong>{vix_min:.0f}</strong>.
              Historisch stärkstes OOS-Regime (~+4,6&nbsp;%, Hit &gt;2,5&nbsp;% ~63&nbsp;%).
            </p>
            <h3 style="color:#ffcc80">2. Orange — Makro + hohes Vol</h3>
            <p>
              Makro-Termin in ±2 Handelstagen, aber VIX ≥ <strong>{macro_min:.0f}</strong>.
              Historisch noch tradbar mit Vorsicht (~+4,7&nbsp;%) — nicht pauschal meiden.
            </p>
            <h3 style="color:#ef9a9a">3. Rot — Makro-Risiko (niedriges Vol)</h3>
            <p>
              Makro-Termin in ±2 Handelstagen bei VIX <strong>&lt; {macro_min:.0f}</strong>.
              Historisch schwächstes Regime (~−1,5&nbsp;%, Hit ~20&nbsp;%).
            </p>
            <h3 style="color:#fff59d">4. Gelb — Standard</h3>
            <p>
              Kein Makro-Warnsignal, VIX unter {vix_min:.0f}, vix3m/vix unter {ratio_min:.2f}.
              Schwächeres Regime als Grün, Modell-Signal bleibt gültig.
            </p>
            <h3 style="color:#fff59d">5. Gelb-Risiko — Vol-Struktur</h3>
            <p>
              Wie Gelb (kein Makro, VIX &lt; {vix_min:.0f}), zusätzlich
              vix3m/vix ≥ <strong>{ratio_min:.2f}</strong> (steile Terminstruktur / Contango).
              Historisch das schwächste Nicht-Makro-Regime (~+1,2&nbsp;%, Hit ~37&nbsp;%).
              Auf den Karten als Badge <strong>„Gelb-Risiko“</strong>.
            </p>
          </div>
        </details>

        <details>
          <summary><strong>Empfohlener Filter</strong></summary>
          <div class="guide-body">
            <p>
              Preset <strong>„Empfohlen“</strong> = Grün + Orange (~+4,6&nbsp;% OOS-Mittel).
              Gelb, Gelb-Risiko und Rot (niedriges Vol + Makro) historisch unterdurchschnittlich.
            </p>
          </div>
        </details>

        <p class="section-lead" style="margin-top:10px">
          Sortierung: <strong>neuestes Signaldatum zuerst</strong>.
          Filter: Alle / Empfohlen / Grün / Orange / Gelb (inkl. Gelb-Risiko) / Rot.
        </p>
      </div>"""


def website_ampel_filter_html(counts: Mapping[str, int]) -> str:
    n_all = int(counts.get("all", 0))
    n_red = int(counts.get("red", 0))
    n_orange = int(counts.get("orange", 0))
    n_yellow = int(counts.get("yellow", 0))
    n_yellow_plain = int(counts.get("yellow_plain", n_yellow))
    n_yellow_risk = int(counts.get("yellow_risk", 0))
    n_green = int(counts.get("green", 0))
    n_trade = n_green + n_orange
    yellow_label = (
        f"Gelb ({n_yellow_plain}+{n_yellow_risk})"
        if n_yellow_risk
        else f"Gelb ({n_yellow})"
    )
    return website_context_guide_html() + f"""
      <div class="section ampel-filter-bar" id="ampel-filter">
        <h2>OOS-Signale nach Kontext-Ampel</h2>
        <p class="section-lead">
          Alle <strong>{n_all}</strong> FINAL-OOS-Signale (aus <code>signals.json</code>).
          Charts werden mitgefiltert; die Tabelle listet jeden Treffer der gewählten Farbe.
        </p>
        <div class="ampel-filter-btns" role="group" aria-label="Kontext-Ampel Filter">
          <button type="button" class="ampel-filter-btn ampel-filter-btn--all is-active" data-ampel="all">Alle ({n_all})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--trade" data-ampel="trade">Empfohlen ({n_trade})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--green" data-ampel="green">Grün ({n_green})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--orange" data-ampel="orange">Orange ({n_orange})</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--yellow" data-ampel="yellow">{yellow_label}</button>
          <button type="button" class="ampel-filter-btn ampel-filter-btn--red" data-ampel="red">Rot ({n_red})</button>
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
        if (t === 'yellow_risk') return 'yellow';
        if (t === 'red' || t === 'orange' || t === 'yellow' || t === 'green') return t;
        return 'unknown';
      }

      function rawTierFromSignal(s) {
        return (s.context_tier || s.vix_regime_ampel || '').toLowerCase();
      }

      function ampelFromCard(card) {
        var pre = (card.getAttribute('data-context-tier') || card.getAttribute('data-vix-ampel') || '').toLowerCase();
        if (pre === 'yellow_risk') return 'yellow';
        if (pre === 'red' || pre === 'orange' || pre === 'yellow' || pre === 'green') return pre;
        var badge = card.querySelector('.context-tier');
        if (badge) {
          var m = badge.className.match(/context-tier--(red|orange|yellow_risk|yellow|green)/);
          if (m) {
            return m[1] === 'yellow_risk' ? 'yellow' : m[1];
          }
        }
        return 'unknown';
      }

      function rawTierFromCard(card) {
        var pre = (card.getAttribute('data-context-tier') || '').toLowerCase();
        if (pre) return pre;
        var badge = card.querySelector('.context-tier');
        if (badge) {
          var m = badge.className.match(/context-tier--(red|orange|yellow_risk|yellow|green)/);
          if (m) return m[1];
        }
        return ampelFromCard(card);
      }

      function matchesFilter(cardTier, ampel) {
        if (ampel === 'all') return true;
        if (ampel === 'trade') return cardTier === 'green' || cardTier === 'orange';
        if (ampel === 'yellow') return cardTier === 'yellow' || cardTier === 'yellow_risk';
        return cardTier === ampel;
      }

      function initCardAmpelAttrs() {
        document.querySelectorAll('.sig-card').forEach(function (card) {
          var raw = rawTierFromCard(card);
          card.dataset.contextTier = raw;
          card.dataset.vixAmpel = raw === 'yellow_risk' ? 'yellow' : raw;
        });
      }

      function filterCards(ampel) {
        var shown = 0, total = 0;
        document.querySelectorAll('.sig-card').forEach(function (card) {
          total += 1;
          var raw = rawTierFromCard(card);
          var ok = matchesFilter(raw, ampel);
          card.style.display = ok ? '' : 'none';
          if (ok) shown += 1;
        });
        return { shown: shown, total: total };
      }

      function esc(s) {
        return String(s == null ? '' : s)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      }

      function signalMatchesFilter(s, ampel) {
        var raw = rawTierFromSignal(s);
        return matchesFilter(raw, ampel);
      }

      function renderList(ampel) {
        if (!tbody || !listEl) return;
        if (ampel === 'all') {
          listEl.hidden = true;
          tbody.innerHTML = '';
          return;
        }
        var rows = signals.filter(function (s) {
          return signalMatchesFilter(s, ampel);
        });
        rows.sort(function (a, b) {
          return (b.date || '').localeCompare(a.date || '') ||
            (a.ticker || '').localeCompare(b.ticker || '');
        });
        var dotAmpel = ampel === 'trade' ? 'green' : ampel;
        var dot = 'ampel-filter-dot ampel-filter-dot--' + dotAmpel;
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
          return signalMatchesFilter(s, ampel);
        }).length;
        if (ampel === 'all') {
          statusEl.textContent = 'Alle OOS-Signale: ' + signals.length + ' gesamt; ' +
            cards.shown + ' Chart-Karten sichtbar.';
        } else if (ampel === 'trade') {
          statusEl.textContent = 'Empfohlen (Grün+Orange): ' + n + ' OOS-Signale; ' +
            cards.shown + ' von ' + cards.total + ' Chart-Karten sichtbar.';
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
