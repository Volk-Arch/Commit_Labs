import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'sectors.py'), encoding='utf-8').read())

def odd_scale(U, margin_min=0.0):
    """Внутренняя характеристика события: какая шкала выбивается из остальных.
    Возвращает (метка, отрыв). Метка -1, если отрыв меньше margin_min."""
    k, N = U.shape
    R = U.T.astype(float)/N
    dev = np.abs(R - np.median(R, axis=1, keepdims=True))
    lab = dev.argmax(axis=1)
    srt = np.sort(dev, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    lab = np.where(margin >= margin_min, lab, -1)
    return lab, margin

def longest_chain_path(R):
    """Восстанавливаем саму максимальную цепь — аналог мировой линии."""
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    D = np.zeros(n, dtype=np.int64); prev = np.full(n, -1)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Rb[:, j])
        for c in kids:
            if D[j] + 1 > D[c]: D[c] = D[j] + 1; prev[c] = j
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    end = int(D.argmax()); path = []
    cur = end
    while cur != -1: path.append(cur); cur = int(prev[cur])
    return path[::-1]
