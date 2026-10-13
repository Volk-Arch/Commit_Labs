#!/usr/bin/env python3
"""БЫСТРЫЙ ОПЕРАТОР В 1+1: материя при N ~ 10^5, без матриц.

Утром материя была объявлена недостижимой (`matter-doc.md`): множитель
растёт как (eps*N)^0.206, до единицы нужно eps*N ~ 1.6e5, а машинерия
доходила до 1e4. Но это верно для матричного счёта. В 1+1 есть обход:

  * засев в диамант = точки в ЕДИНИЧНОМ КВАДРАТЕ световых координат
    (u,v), причём t=(u+v)/2, x=(u-v)/2, площадь в (t,x) равна 1/2,
    значит rho = 2N - та же нормировка, что в waveop;
  * |I(y,x)| в системе, отсчитанной от x (a = u_x-u, b = v_x-v), есть
    число точек, ДОМИНИРУЕМЫХ y, то есть a_z < a_y и b_z < b_y;
  * для резких весов (1,-2,1) нужны только счёты 0,1,2, а точка со
    счётом c лежит в слое Парето не глубже c+1 - значит достаточно ТРЁХ
    слоёв, а слой снимается бегущим минимумом, векторно.

Отсюда O(N) на точку оценки вместо O(N^2) на всё. И по закону
`matter-doc.md` множитель зависит только от eps*N, значит РЕЗКИЙ оператор
(eps=1) обязан сойтись при N ~ 1.6e5 - достижимо.

КОНТРОЛЬ, без которого ни одно большое число не считается: быстрый счёт
обязан совпасть с матричным при малом N, элемент за элементом.
"""
import math
import sys

import numpy as np

W = (1.0, -2.0, 1.0)
CSELF = -0.5


def sprinkle_uv(n, rs):
    return rs.random((n, 2))


def tx(P):
    return (P[:, 0] + P[:, 1]) / 2.0, (P[:, 0] - P[:, 1]) / 2.0


def bop_at(U, V, idx, order_u, pos):
    """вклад sum_{y<x} w_{|I|} f(y) через слои Парето. Возвращает список
    (индексы y, их веса)."""
    p = pos[idx]
    cand_i = order_u[:p]                      # u < u_x
    sel = V[cand_i] < V[idx]
    cand = cand_i[sel][::-1]                  # по убыванию u => a растёт
    if len(cand) == 0:
        return np.empty(0, int), np.empty(0)
    b = V[idx] - V[cand]
    alive = np.ones(len(cand), bool)
    layers = []
    for _ in range(3):
        bb = np.where(alive, b, np.inf)
        rmin = np.minimum.accumulate(
            np.concatenate([[np.inf], bb[:-1]]))
        lay = alive & (bb < rmin)
        if not lay.any():
            break
        layers.append(np.nonzero(lay)[0])
        alive &= ~lay
    if not layers:
        return np.empty(0, int), np.empty(0)
    ci = np.concatenate(layers)
    a_all = U[idx] - U[cand]
    cnt = np.empty(len(ci), int)
    for t, j in enumerate(ci):
        cnt[t] = int(np.count_nonzero((a_all < a_all[j]) & (b < b[j])))
    keep = cnt <= 2
    ci, cnt = ci[keep], cnt[keep]
    return cand[ci], np.array([W[c] for c in cnt])


def run(n, rs, npts=300, core=(0.3, 0.7, 0.15)):
    P = sprinkle_uv(n, rs)
    U, V = P[:, 0], P[:, 1]
    t, x = tx(P)
    order_u = np.argsort(U)
    pos = np.empty(n, int)
    pos[order_u] = np.arange(n)
    lo, hi, xm = core
    ok = np.nonzero((t > lo) & (t < hi) & (np.abs(x) < xm))[0]
    if len(ok) == 0:
        return None
    pick = rs.choice(ok, size=min(npts, len(ok)), replace=False)
    rho = 2.0 * n
    fs = {'t2': t * t, 'x2': x * x, 'tau2': t * t - x * x}
    out = {k: [] for k in fs}
    for idx in pick:
        ys, ws = bop_at(U, V, idx, order_u, pos)
        for k, f in fs.items():
            s = float((ws * f[ys]).sum()) if len(ys) else 0.0
            out[k].append(rho * (CSELF * f[idx] + s))
    return {k: (float(np.mean(v)),
                float(np.std(v) / math.sqrt(len(v)))) for k, v in out.items()}


def brute(n, rs, core=(0.3, 0.7, 0.15)):
    P = sprinkle_uv(n, rs)
    U, V = P[:, 0], P[:, 1]
    t, x = tx(P)
    R = (U[None, :] > U[:, None]) & (V[None, :] > V[:, None])
    Iv = np.rint(R.astype(np.float32) @ R.astype(np.float32)).astype(int)
    lo, hi, xm = core
    ok = (t > lo) & (t < hi) & (np.abs(x) < xm)
    rho = 2.0 * n
    res = {}
    for k, f in (('t2', t * t), ('x2', x * x), ('tau2', t * t - x * x)):
        M = np.zeros((n, n))
        for m, w in enumerate(W):
            M += w * (R.T & (Iv.T == m))
        v = rho * (CSELF * f + M @ f)
        res[k] = (float(v[ok].mean()),
                  float(v[ok].std() / math.sqrt(ok.sum())))
    return res


def main():
    print(__doc__)
    rs = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
    print('КОНТРОЛЬ: быстрый против матричного, одна и та же выборка')
    for n in (300, 600):
        s1 = np.random.default_rng(100 + n)
        s2 = np.random.default_rng(100 + n)
        a = run(n, s1, npts=10 ** 9)
        b = brute(n, s2)
        print(f'  N={n}:')
        for k in ('t2', 'x2', 'tau2'):
            print(f'    {k:>5}: быстрый {a[k][0]:+10.4f}  '
                  f'матричный {b[k][0]:+10.4f}  '
                  f'разница {abs(a[k][0] - b[k][0]):.2e}')


if __name__ == '__main__':
    main()
