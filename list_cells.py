import json, sys
sys.stdout.reconfigure(encoding="utf-8")
with open(r"c:\Python projects\stock_rally\stock_rally_v10.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])[:90].replace("\n", " ")
    print(i, cell["cell_type"], src[:85])
