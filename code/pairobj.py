"""Пара как объект: нормированное прошлое против сырого объёма.

Разброс сырого объёма общего прошлого был 64 при среднем 44, а
пуассоновский дал бы 6.6 — значит разброс ПОЗИЦИОННЫЙ, а не
статистический. Он делится: нормируем на собственные прошлые пары.

  сырое:   |past(a) & past(b)|
  Жаккар:  |&| / |past(a) | past(b)|
  доля:    |&| / min(|past(a)|, |past(b)|)
"""
import numpy as np
from fixpoint import diamond, cone_order

N, T = 3000, 2.0
rs = np.random.default_rng(4321)
X = diamond(N, T, rs); R = cone_order(X)
A = R.astype(np.float32)
CP = A @ A.T
sz = A.sum(1)
inc = ~(R | R.T); np.fill_diagonal(inc, False)
mid = np.abs(X[:,0]-T/2) < 0.25*T
m = np.triu(inc & mid[:,None] & mid[None,:], 1)
ii, jj = np.nonzero(m)
k = rs.choice(len(ii), min(300000, len(ii)), replace=False)
ii, jj = ii[k], jj[k]
big = (sz[ii] >= 8) & (sz[jj] >= 8)          # у обоих есть прошлое
ii, jj = ii[big], jj[big]
dt = np.abs(X[ii,0]-X[jj,0]); dx = np.linalg.norm(X[ii,1:]-X[jj,1:],axis=1)
sep = np.sqrt(np.maximum(dx*dx-dt*dt, 0.0))
cp = CP[ii,jj]; sa = sz[ii]; sb = sz[jj]
jac = cp/np.maximum(sa+sb-cp, 1)
frac = cp/np.maximum(np.minimum(sa,sb), 1)
rk = lambda v: np.argsort(np.argsort(v))
print(f'N={N}, пар в выборке {len(ii)}')
print()
print(f"{'мера':>26}{'ранговая корр. с sep':>24}")
for nm, v in (('сырой объём', cp), ('Жаккар', jac),
              ('доля от меньшего', frac),
              ('объём^(1/4)', cp.astype(float)**0.25)):
    print(f'{nm:>26}{np.corrcoef(rk(sep), rk(v))[0,1]:>+24.3f}')
print()
print('ЖАККАР по корзинам разноса')
print(f"{'sep':>8}{'пар':>9}{'<Жаккар>':>11}{'с.к.о.':>9}"
      f"{'отн.разброс':>13}")
ed = np.quantile(sep, np.linspace(0,1,9))
for b in range(len(ed)-1):
    s=(sep>=ed[b])&(sep<ed[b+1])
    if s.sum()<50: continue
    mu=jac[s].mean(); sd=jac[s].std()
    print(f'{0.5*(ed[b]+ed[b+1]):>8.3f}{int(s.sum()):>9}{mu:>11.4f}'
          f'{sd:>9.4f}{sd/max(mu,1e-9):>13.2f}')
