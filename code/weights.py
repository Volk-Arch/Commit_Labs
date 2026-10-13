"""Спасение через вес: встреча = МНОГО связей. Станет ли граф пространственным."""
import numpy as np
from tallmeet import tall, greedy_long
from meetings import ball_growth, fit_growth

N, T, Rr = 3000, 6.0, 0.45
rs = np.random.default_rng(9500)
X = tall(N, T, Rr, rs)
dt = X[:,0][None,:]-X[:,0][:,None]
dx = np.linalg.norm(X[:,None,1:]-X[None,:,1:],axis=2)
R = (dt>0)&(dt>dx); del dt, dx
A = R.astype(np.float32); LK = R & ~((A@A)>0.5); del A
seqs, owner = greedy_long(R); nh = len(seqs)
li, lj = np.nonzero(LK); ok = owner[li] != owner[lj]
W = np.zeros((nh, nh), np.int32)
for u, v in zip(owner[li[ok]], owner[lj[ok]]):
    W[u, v] += 1; W[v, u] += 1
cen = np.array([X[q][:,1:].mean(0) for q in seqs])
iu = np.triu_indices(nh, 1)
w = W[iu]; truedist = np.linalg.norm(cen[iu[0]]-cen[iu[1]], axis=1)
print(f'историй {nh}, пар {len(w)}, вес встреч: средний {w.mean():.1f}, '
      f'медиана {np.median(w):.0f}, макс {w.max()}')
print(f'доля пар вовсе без встреч: {(w==0).mean():.4f}')
print(f'корреляция ВЕСА с истинным расстоянием: '
      f'{np.corrcoef(w, truedist)[0,1]:+.4f}')
print()
print(f"{'порог':>7}{'ребёр':>9}{'степень':>9}{'связен':>9}"
      f"{'рост шара':>11}{'корр. с истинным':>18}")
for q in (0.50, 0.75, 0.90, 0.95, 0.98):
    thr = np.quantile(w[w>0], q)
    M = W >= thr
    np.fill_diagonal(M, False)
    adj = [set(np.flatnonzero(M[i]).tolist()) for i in range(nh)]
    deg = M.sum(1)
    g = ball_growth(adj, 40, np.random.default_rng(5), dmax=8)
    reach = g[-1]/nh
    keep = w >= thr
    cr = np.corrcoef(keep.astype(float), truedist)[0,1]
    print(f'{q:>7.2f}{int(M.sum()//2):>9}{deg.mean():>9.1f}'
          f'{reach:>9.2f}{fit_growth(g,1,5):>11.3f}{cr:>18.4f}')
