import json, sys
sys.stdout.reconfigure(encoding="utf-8")
with open(r"c:\Python projects\stock_rally\stock_rally_v10.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
cell = nb["cells"][13]
print("".join(cell["source"]))
