"""Emit top-100 SHAP markdown table rows to stdout or a path."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
p = json.loads(Path("models/base_feature_shap_report.json").read_text(encoding="utf-8"))
lines = [
    f"| {r['rank']} | {r['mean_abs_shap']:.6f} | {r['feature_display']} | `{r['feature_raw']}` |"
    for r in p["shap_mean_abs_sorted"][:100]
]
text = "\n".join(lines) + "\n"
if out:
    out.write_text(text, encoding="utf-8")
else:
    sys.stdout.buffer.write(text.encode("utf-8"))
