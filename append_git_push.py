import json

NB_PATH = r'c:\Python projects\stock_rally\stock_rally_v10.ipynb'

GIT_BLOCK = '''
# ─────────────────────────────────────────────────────────────────────────────
# 6.  Auto-push to GitHub Pages
# ─────────────────────────────────────────────────────────────────────────────
import subprocess as _sp, os as _os

_repo = r'c:\\Python projects\\stock_rally'

def _git(*args):
    r = _sp.run(['git'] + list(args), cwd=_repo, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(f'[git] {r.stderr.strip()}')
    return r.returncode

print('\\nPushing docs/ to GitHub Pages ...')
from pathlib import Path as _Path
_git_docs = [
    'docs/index.html', 'docs/signals.json', 'docs/website_analysis_prompt.txt',
    'docs/analysis_llm_last.html', 'docs/analysis_llm_last.txt',
]
_git_to_add = [p for p in _git_docs if (_Path(_repo) / p.replace('/', _os.sep)).is_file()]
if not _git_to_add:
    print('No docs files to stage — skipping.')
else:
    _git('add', '--', *_git_to_add)

# Only commit if there are staged changes
_diff = _sp.run(['git', 'diff', '--cached', '--quiet'], cwd=_repo)
if _diff.returncode != 0:
    import datetime as _dt
    _msg = f'Daily signals {_dt.date.today()}'
    _git('commit', '-m', _msg)
    rc = _git('push')
    if rc == 0:
        print('Push successful — website will update in ~1 min.')
    else:
        print('Push failed. Check git credentials / network.')
else:
    print('No changes in docs/ — nothing to push.')
'''

nb    = json.load(open(NB_PATH, encoding='utf-8'))
cells = nb['cells']

# Find Cell 17
target = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'Cell 17' in src and 'full history edition' in src and c['cell_type'] == 'code':
        target = i
        break

if target is None:
    print('ERROR: Cell 17 (full history edition) not found'); exit(1)

src = ''.join(cells[target]['source'])
new_src = src + GIT_BLOCK
cells[target]['source'] = [ln + '\n' for ln in new_src.splitlines()]
cells[target]['source'][-1] = cells[target]['source'][-1].rstrip('\n')

json.dump(nb, open(NB_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Appended git-push block to cell {target}. Done.')
