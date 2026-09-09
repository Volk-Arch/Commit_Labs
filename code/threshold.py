"""Порог расфазировки: при каком объёме интервала разброс действия
достигает периода."""
import numpy as np
import sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from bdaction import bd_action


def phase_threshold(R, rs, vmax=30, want=100, tries=200000, dim=4):
    """Наименьший объём интервала, где разброс действия > 2π."""
    N = len(R)
    byv = {}
    for _ in range(tries):
        i, j = (int(v) for v in rs.integers(0, N, 2))
        if not R[i, j]:
            continue
        mid = np.flatnonzero(R[:, j] & R[i, :])
        v = len(mid)
        if v < 3 or v > vmax:
            continue
        l = byv.setdefault(v, [])
        if len(l) >= want:
            continue
        l.append(bd_action(np.asarray(R[np.ix_(mid, mid)], dtype=bool), dim))
        if len(byv) > 5 and all(len(x) >= want for x in byv.values()):
            break
    out = None
    for v in sorted(byv):
        a = np.array(byv[v], dtype=float)
        if len(a) < 40:
            continue
        if a.std() > 2*np.pi:
            out = v
            break
    return out, {v: (len(x), float(np.std(x))) for v, x in byv.items()
                 if len(x) >= 40}
