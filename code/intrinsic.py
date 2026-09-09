import numpy as np

def interval_pairs(R, rs, want=400, lo=15, hi=400, tries=60000):
    """Собираем интервалы: пары (i,j) и события между ними.
    Всё структурно — никаких координат."""
    N = len(R)
    out = []
    for _ in range(tries):
        i, j = (int(v) for v in rs.integers(0, N, 2))
        if not R[i, j]: continue
        mid = np.flatnonzero(R[:, j] & R[i, :])
        m = len(mid)
        if m < lo or m > hi: continue
        out.append((i, j, mid))
        if len(out) >= want: break
    return out

def interval_stats(R, L, i, j, mid):
    """Две величины внутри интервала, обе из структуры:
       материя  = сколько событий внутри (объём)
       кривизна = доля прямых связей внутри
    """
    m = len(mid)
    sub = R[np.ix_(mid, mid)]
    subL = L[np.ix_(mid, mid)]
    tot = sub.sum()
    if tot == 0: return None
    return m, float(subL.sum()/tot), float(tot/(m*(m-1)/2))
