import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/iterate.py').read())

def coarse_grain(R, nblocks, rs=None):
    """Огрубление: события группируются по глубине в блоки, и мы смотрим
    только на связи МЕЖДУ блоками. Микроструктура внутри блока теряется."""
    r = rs if rs is not None else np.random.default_rng(0)
    Rf = np.asarray(R, dtype=np.float32)
    D = proper_time(Rf)
    hi = max(D.max(), 1)
    blk = np.minimum((D*nblocks//(hi+1)).astype(int), nblocks-1)
    M = np.zeros((nblocks, nblocks))
    Rb = np.asarray(R, dtype=bool)
    for a in range(nblocks):
        ia = np.flatnonzero(blk == a)
        if len(ia) == 0: continue
        for b in range(nblocks):
            ib = np.flatnonzero(blk == b)
            if len(ib) == 0: continue
            M[b, a] = Rb[np.ix_(ib, ia)].mean()      # доля связей из блока a в блок b
    return M, blk

def entropy_of(M):
    """Насколько огрублённая картина «размазана»: энтропия распределения связей."""
    p = M.ravel(); p = p[p > 0]; p = p/p.sum()
    return float(-(p*np.log(p)).sum())

def coarse_evolution(R, nblocks, steps=8):
    """Огрубляем, потом итерируем УЖЕ ОГРУБЛЁННУЮ картину.
    Вопрос: теряется ли информация, и одинаково ли вперёд и назад?"""
    M, _ = coarse_grain(R, nblocks)
    hist = [entropy_of(M)]
    X = M.copy()
    for _ in range(steps):
        X = X @ M                                    # распространение по огрублённой сетке
        X = X/max(X.sum(), 1e-12)
        hist.append(entropy_of(X))
    return np.array(hist)
