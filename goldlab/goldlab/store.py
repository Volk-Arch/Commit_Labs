"""ХРАНИЛИЩЕ ПРОГОНОВ: runs/<id>.json — параметры, область, числа (с эталонами и статусом), графики, раскраска мира."""
import json, os, time, glob
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, (np.floating,)): o = float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, float) and not np.isfinite(o): return None
    return o

def save(res, tag=''):
    os.makedirs(ROOT, exist_ok=True); rid = time.strftime('%Y%m%d-%H%M%S') + '-' + res['exp'] + (('-' + tag) if tag else '')
    res = _clean({**res, 'id': rid, 'saved': time.strftime('%Y-%m-%d %H:%M:%S')})
    with open(os.path.join(ROOT, rid + '.json'), 'w') as f: json.dump(res, f, ensure_ascii=False)
    return rid

def headline(r):
    st = [n['status'] for n in r.get('numbers', []) if n.get('status')]
    return {'id': r['id'], 'exp': r['exp'], 'saved': r.get('saved'), 'seconds': r.get('seconds'), 'params': r.get('params'),
            'ok': st.count('ok'), 'near': st.count('near'), 'off': st.count('off')}

def list_runs():
    out = []
    for fn in sorted(glob.glob(os.path.join(ROOT, '*.json')), reverse=True):
        try:
            with open(fn) as f: out.append(headline(json.load(f)))
        except Exception: pass
    return out

def load(rid):
    with open(os.path.join(ROOT, rid + '.json')) as f: return json.load(f)
