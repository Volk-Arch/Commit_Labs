"""Самые длинные цепочки в 2+1 с местным правилом конусов (общий инструмент).
События — точки (t, x, y); «раньше» только между близкими: 0 < Δt < EPS, |Δx| < EPS·2
и разность внутри местного конуса metric_ok(середина, Δ). Дальнее — цепочкой.
dist[y] = длина самой длинной цепочки от A до y, pred — предыдущий шаг."""
import numpy as np, math
from scipy.spatial import cKDTree

def local_edges(P, metric_ok, eps, slab=6.0):
    """рёбра i→j (P отсортированы по t), по слоям времени — чтобы хватало памяти."""
    t = P[:, 0]; I, J = [], []
    t0 = t[0]
    while t0 <= t[-1]:
        a = np.searchsorted(t, t0); b = np.searchsorted(t, t0 + slab); c = np.searchsorted(t, t0 + slab + eps)
        if b > a:
            sub = P[a:c]; tree = cKDTree(sub)
            pr = tree.query_pairs(r=eps * 1.5, output_type='ndarray')
            if len(pr):
                i, j = pr[:, 0] + a, pr[:, 1] + a
                sw = t[i] > t[j]; i2 = np.where(sw, j, i); j2 = np.where(sw, i, j)
                core = i2 < b
                i2, j2 = i2[core], j2[core]
                d = P[j2] - P[i2]; mid = 0.5 * (P[j2] + P[i2])
                k = (d[:, 0] > 0) & (d[:, 0] < eps) & metric_ok(mid, d)
                I.append(i2[k]); J.append(j2[k])
        t0 += slab
    return np.concatenate(I), np.concatenate(J)

def longest_from(P, src, ei, ej):
    """самая длинная цепочка из src во все события (DAG по времени). P отсортированы по t."""
    n = len(P); dist = np.full(n, -1, np.int64); pred = np.full(n, -1, np.int64); dist[src] = 0
    o = np.argsort(ej, kind='stable'); ei, ej = ei[o], ej[o]
    starts = np.searchsorted(ej, np.arange(n)); ends = np.searchsorted(ej, np.arange(n), side='right')
    for y in range(src + 1, n):
        a, b = starts[y], ends[y]
        if a == b: continue
        cand = dist[ei[a:b]]; k = cand.argmax()
        if cand[k] >= 0: dist[y] = cand[k] + 1; pred[y] = ei[a + k]
    return dist, pred

def path(pred, y):
    out = [y]
    while pred[out[-1]] >= 0: out.append(pred[out[-1]])
    return out[::-1]

def longest_to(P, dst, ei, ej):
    """самая длинная цепочка из каждого события в dst (обратный проход)."""
    n = len(P); d = np.full(n, -1, np.int64); d[dst] = 0
    o = np.argsort(ei, kind='stable'); ei, ej = ei[o], ej[o]
    starts = np.searchsorted(ei, np.arange(n)); ends = np.searchsorted(ei, np.arange(n), side='right')
    for x in range(dst - 1, -1, -1):
        a, b = starts[x], ends[x]
        if a == b: continue
        c = d[ej[a:b]].max()
        if c >= 0: d[x] = c + 1
    return d
