"""Emit full Meta-SHAP markdown table rows."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "meta_feature_shap_report.json"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else None

p = json.loads(REPORT.read_text(encoding="utf-8"))
lines = []
for r in p["shap_mean_abs_sorted"]:
    feat = r["feature"]
    kind = "Base-`prob`" if feat.endswith("_prob") else "Roh (Top-K)"
    lines.append(
        f"| {r['rank']} | {r['mean_abs_shap']:.6f} | {kind} | `{feat}` |"
    )
text = "\n".join(lines) + "\n"
if out:
    out.write_text(text, encoding="utf-8")
else:
    sys.stdout.buffer.write(text.encode("utf-8"))
