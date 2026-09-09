import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'intrinsic.py'), encoding='utf-8').read())

def warped_scales(N, k, warp, rs):
    """Искривление: плотность фаз неоднородна. warp=0 — равномерно."""
    U = []
    for m in range(k):
        x = rs.random(N)
        if warp: x = x**(1.0 + warp) if m == 0 else x
        U.append(np.argsort(np.argsort(x)))
    return np.array(U, dtype=np.int64)

def chirality_profile(U, R, nbins=4):
    """Доля правых связей ПО ОБЛАСТЯМ (по глубине). Меняется ли с кривизной?"""
    d, ok = directions(U, R)
    P = U.T.astype(np.float64); N = len(P)
    D3 = P[None, :, :3] - P[:, None, :3]
    cr = np.cross(np.broadcast_to(d[:, None, :3], (N, N, 3)),
                  np.broadcast_to(d[None, :, :3], (N, N, 3)))
    det = (cr*D3).sum(-1)
    dep = proper_time(R.T.astype(np.float32)); hi = max(dep.max(), 1)
    out = []
    for b in range(nbins):
        lo, up = b*hi/nbins, (b+1)*hi/nbins
        sel = (dep >= lo) & (dep < up)
        idx = np.flatnonzero(sel)
        if len(idx) < 20: out.append(np.nan); continue
        sub = R[np.ix_(idx, idx)]
        ds = det.T[np.ix_(idx, idx)]
        tot = sub.sum()
        out.append(float((sub & (ds < 0)).sum()/max(tot, 1)))
    return out
