import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'spinlabel.py'), encoding='utf-8').read())

def chiral_weighted(U, s, amp=0.0):
    """Связь НЕ УДАЛЯЕТСЯ, а получает ВЕС. Правые киральные пары имеют вес amp,
    остальные — 1. При amp=0 совпадает с удалением, при amp>0 связь остаётся.
    Порядок (кто с кем связан) при amp>0 НЕ МЕНЯЕТСЯ — значит транзитивность цела."""
    N = U.shape[1]
    R = base_rel(U)
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    par = np.sign(a-b)*np.sign(b-c)*np.sign(a-c)
    both = (s[None, :] > 0) & (s[:, None] > 0)
    W = R.astype(np.float32)
    weak = both & (par < 0) & R
    W[weak] = amp
    return R, W, weak

def weighted_mirror(U, s, amp, i=0, j=1):
    R1, W1, _ = chiral_weighted(U, s, amp)
    V = U.copy(); V[[i, j]] = V[[j, i]]
    R2, W2, _ = chiral_weighted(V, s, amp)
    return int((R1 ^ R2).sum()), float(np.abs(W1 - W2).sum())
