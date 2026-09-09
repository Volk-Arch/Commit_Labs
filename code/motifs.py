import numpy as np
from itertools import permutations

def canon(sub):
    """Канонический вид маленького порядка: минимум по всем перестановкам."""
    n = len(sub)
    best = None
    for p in permutations(range(n)):
        q = sub[np.ix_(p, p)]
        key = tuple(q.flatten().astype(np.uint8))
        if best is None or key < best: best = key
    return best

def motif_spectrum(M, size=4, samples=6000, rs=None):
    """Частоты всех индуцированных подпорядков заданного размера."""
    r = rs if rs is not None else np.random.default_rng(0)
    N = len(M)
    Mb = np.ascontiguousarray(M, dtype=bool)
    cnt = {}
    for _ in range(samples):
        idx = r.choice(N, size, replace=False)
        sub = Mb[np.ix_(idx, idx)]
        k = canon(sub)
        cnt[k] = cnt.get(k, 0) + 1
    tot = sum(cnt.values())
    return {k: v/tot for k, v in cnt.items()}

def compare(specs, names):
    keys = set()
    for s in specs: keys |= set(s.keys())
    keys = sorted(keys, key=lambda k: -max(s.get(k, 0) for s in specs))
    return keys
