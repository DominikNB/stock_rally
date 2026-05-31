"""VIX-Ampel-Filter für die statische OOS-Signal-Website (docs/index.html)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


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
"""


def website_ampel_filter_html(counts: Mapping[str, int]) -> str:
    n_all = int(counts.get("all", 0))
    n_red = int(counts.get("red", 0))
    n_yellow = int(counts.get("yellow", 0))
    n_green = int(counts.get("green", 0))
    return f"""
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
        if (card.dataset.vixAmpel) return card.dataset.vixAmpel;
        var el = card.querySelector("[class*='vix-ampel--']");
        if (!el) return "unknown";
        var m = el.className.match(/vix-ampel--(red|yellow|green)/);
        return m ? m[1] : "unknown";
      }

      function initCardAmpelAttrs() {
        document.querySelectorAll(".sig-card").forEach(function (card) {
          if (!card.dataset.vixAmpel) card.dataset.vixAmpel = ampelFromCard(card);
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
