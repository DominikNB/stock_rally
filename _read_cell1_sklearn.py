import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('stock_rally_v10.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
src1 = ''.join(nb['cells'][1]['source'])
idx = src1.find('sklearn')
print(repr(src1[max(0,idx-5):idx+200]))
