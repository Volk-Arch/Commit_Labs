import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'intrinsic.py'), encoding='utf-8').read())

def partial_order_rel(U, own):
    """У события i свой набор шкал own[i] (булев вектор длины k).
    Сравниваем по ОБЩИМ шкалам пары. Транзитивность не гарантирована — меряем."""
    k, N = U.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        shared = own[:, m][None, :] & own[:, m][:, None]
        lt = (U[m][None, :] < U[m][:, None])
        R &= (lt | ~shared)                    # если шкала не общая — условие не мешает
    np.fill_diagonal(R, False)
    # пары без единой общей шкалы связывать нельзя
    common_any = (own[None, :, :] & own[:, None, :]).any(axis=2)
    return R & common_any

def make_sets(N, k, core, extra_frac, rs):
    """core шкал есть у всех; остальные k-core — у доли extra_frac событий."""
    own = np.zeros((N, k), dtype=bool)
    own[:, :core] = True
    for m in range(core, k):
        own[:, m] = rs.random(N) < extra_frac
    return own

def acyc(R):
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Rb[:, j])
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return seen == n

def close_with_weights(R_raw, amp=0.001, maxit=30):
    """Чиним неоднородность как чинили киральность: берём транзитивное замыкание,
    а связи, появившиеся ТОЛЬКО от замыкания, получают малый вес.
    Порядок становится транзитивным по построению; неоднородность живёт в весах."""
    R = np.ascontiguousarray(R_raw, dtype=bool).copy()
    for _ in range(maxit):
        F = R.astype(np.float32)
        nxt = R | ((F @ F) > 0)
        if nxt.sum() == R.sum(): break
        R = nxt
    W = np.where(R_raw, 1.0, np.where(R, amp, 0.0)).astype(np.float32)
    added = R & ~np.asarray(R_raw, dtype=bool)
    return R, W, added
