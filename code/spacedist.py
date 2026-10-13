"""Порядковая мера пространственного расстояния через общее прошлое.

Все величины «вперёд/назад» монотонны по порядку и пространственные пары
не различают. Первая величина, смотрящая В СТОРОНУ, — объём общего
прошлого: он убывает с пространственным разносом.

Проверка калибровки: для НЕСРАВНИМЫХ пар сравнить |past(a) & past(b)| с
настоящим пространственным расстоянием в засеве. Если связь монотонная и
тесная — дыра в таблице закрыта.
"""
import numpy as np
from fixpoint import diamond, cone_order

N, T = 2500, 2.0
rs = np.random.default_rng(1234)
X = diamond(N, T, rs); R = cone_order(X)
A = R.astype(np.float32)
CP = (A @ A.T)            # |past(a) & past(b)|
CF = (A.T @ A)            # |future(a) & future(b)|
inc = ~(R | R.T); np.fill_diagonal(inc, False)
# берём пары, не слишком близкие к краю мира
mid = (np.abs(X[:,0]-T/2) < 0.22*T)
m = inc & mid[:,None] & mid[None,:]
m = np.triu(m, 1)
ii, jj = np.nonzero(m)
if len(ii) > 400000:
    k = rs.choice(len(ii), 400000, replace=False); ii, jj = ii[k], jj[k]
# настоящее пространственное расстояние (в системе ромба)
dt = np.abs(X[ii,0]-X[jj,0])
dx = np.linalg.norm(X[ii,1:]-X[jj,1:], axis=1)
sep = np.sqrt(np.maximum(dx*dx - dt*dt, 0.0))     # пространственный интервал
cp = CP[ii, jj]; cf = CF[ii, jj]
print(f'N={N}, несравнимых пар в выборке {len(ii)}')
print('sep = sqrt(dx^2 - dt^2), настоящий пространственный интервал')
print()
print(f"{'корзина sep':>14}{'пар':>9}{'<общее прошлое>':>18}"
      f"{'с.к.о.':>9}{'<общее будущее>':>18}{'<прош+буд>':>13}")
edges = np.quantile(sep, np.linspace(0, 1, 9))
for b in range(len(edges)-1):
    s = (sep >= edges[b]) & (sep < edges[b+1])
    if s.sum() < 50: continue
    print(f'{0.5*(edges[b]+edges[b+1]):>14.3f}{int(s.sum()):>9}'
          f'{cp[s].mean():>18.1f}{cp[s].std():>9.1f}'
          f'{cf[s].mean():>18.1f}{(cp[s]+cf[s]).mean():>13.1f}')
print()
for nm, q in (('общее прошлое', cp), ('общее будущее', cf),
              ('прошлое+будущее', cp+cf)):
    c = np.corrcoef(sep, q)[0,1]
    cs = np.corrcoef(np.argsort(np.argsort(sep)),
                     np.argsort(np.argsort(q)))[0,1]
    print(f'  {nm:>18}: корреляция с sep {c:+.3f}, '
          f'ранговая {cs:+.3f}')
