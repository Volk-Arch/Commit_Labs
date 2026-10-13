#!/usr/bin/env python3
"""ПРОПАГАТОР ДЖОНСТОНА В 3+1: скорость убывания флуктуаций.

В 1+1 (`johnston.py`) скорость найдена: отн. флуктуация ~ N^{-0.556
+- 0.027}, что совпало с замеренным законом точности часов
(-0.5296 +- 0.0245) на 0.7 сигмы. Значит там шум пропагатора есть
дробовой шум счёта ОБЪЁМА.

В 3+1 конструкция другая: сумма идёт по ЗВЕНЬЯМ (матрица звеньев, не
причинности), a = sqrt(rho)/(2 pi sqrt 6), b = -m^2/rho, и совпадение с
непрерывным пропагатором лишь АСИМПТОТИЧНО при rho -> oo. Поэтому
скорость мерится САМОДОСТАТОЧНО: sigma(K)/|<K>| по засевам, без
обращения к непрерывной формуле — её систематика тогда не подмешивается.

ПРОГНОЗ ОТКРЫТ: в 1+1 шум шёл от счёта объёма (~N, отн. шум N^{-1/2}).
Здесь носитель — звенья, которых ~N^{3/2}, и закон может выйти другой.
Это и есть содержание прогона.

КОНТРОЛИ:
  1. при b = 0 ряд обрывается на первом члене: K = a L ровно;
  2. матрица звеньев обязана лежать в матрице причинности, а число
     звеньев на элемент — расти как N^{1/2} (известная асимптотика
     N^{2-2/d} при d = 4);
  3. обратный проход обязан совпасть с явным суммированием ряда;
  4. флуктуации обязаны убывать с плотностью.

  python3 johnston4.py
"""
import math

import numpy as np


def sprinkle4(n, rs, T=1.0):
    pts = []
    while len(pts) < n:
        m = int((n - len(pts)) * 5) + 64
        t = rs.random(m) * T
        v = rs.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / 3.0)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    P = np.array(pts[:n])
    P = np.vstack([np.array([[0.0, 0, 0, 0]]), P, np.array([[T, 0, 0, 0]])])
    o = np.argsort(P[:, 0])
    P = P[o]
    return P, int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))


def causal_links(P):
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    Rc = (dt > 0) & (dt ** 2 > dx2)
    F = Rc.astype(np.float32)
    two = (F @ F) > 0.5
    L = Rc & ~two
    return Rc, L


def prop_entry(L, a, b, p, q):
    """K[p,q] обратным проходом: w[j] = Phi[j,q] + b sum_i Phi[j,i] w[i].
    Порядок по убыванию времени (индексы уже отсортированы)."""
    n = L.shape[0]
    Phi = a * L.astype(np.float64)
    w = np.zeros(n)
    for j in range(n - 1, -1, -1):
        s = Phi[j, q]
        if b != 0.0:
            s += b * float(Phi[j] @ w)
        w[j] = s
    return w[p]


def prop_series(L, a, b, p, q, kmax=400):
    """КОНТРОЛЬ 3: явное суммирование ряда sum b^{n-1} Phi^n."""
    Phi = a * L.astype(np.float64)
    v = Phi[p].copy()
    tot = v[q]
    coef = 1.0
    for _ in range(kmax):
        v = v @ Phi
        if not np.any(v):
            break
        coef *= b
        tot += coef * v[q]
    return tot


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    T = 1.0
    Vd = math.pi / 24.0 * T ** 4
    mT = 5.0
    print('КОНТРОЛИ 1-3')
    P, p, q = sprinkle4(200, rs, T)
    Rc, L = causal_links(P)
    n = len(P)
    rho = n / Vd
    a = math.sqrt(rho) / (2 * math.pi * math.sqrt(6.0))
    b = -(mT / T) ** 2 / rho
    k0 = prop_entry(L, a, 0.0, p, q)
    print(f'  1) b = 0: K = {k0:.10f}, a*L[p,q] = {a*L[p,q]:.10f}  '
          f'{"ВЕРНО" if abs(k0 - a*L[p,q]) < 1e-12 else "НЕВЕРНО"}')
    print(f'  2) звенья внутри причинности: '
          f'{"ВЕРНО" if not (L & ~Rc).any() else "НЕВЕРНО"}; '
          f'звеньев на элемент {L.sum()/n:.2f}, '
          f'N^0.5 = {n**0.5:.2f}, отношение {L.sum()/n/n**0.5:.3f}')
    ksweep = prop_entry(L, a, b, p, q)
    kser = prop_series(L, a, b, p, q)
    print(f'  3) обратный проход {ksweep:.8e} против ряда {kser:.8e}, '
          f'отн. {abs(ksweep-kser)/max(abs(kser),1e-300):.2e}')
    print()
    print('ЧИСЛО ЗВЕНЬЕВ НА ЭЛЕМЕНТ ПРОТИВ N^{1/2} (контроль 2 по росту)')
    for nn in (200, 400, 800):
        P, p, q = sprinkle4(nn, rs, T)
        Rc, L = causal_links(P)
        m_ = len(P)
        print(f'  N = {m_:>5}: звеньев/элемент {L.sum()/m_:>7.2f}, '
              f'делённое на sqrt(N) {L.sum()/m_/m_**0.5:>7.4f}')
    print()
    print('ГЛАВНОЕ: скорость убывания флуктуаций в 3+1')
    print(f"{'N':>6}{'rho':>9}{'<K>':>14}{'ст.откл.':>13}"
          f"{'отн. флукт.':>14}{'засевов':>9}")
    rows = []
    for nn, nspr in ((150, 120), (300, 90), (600, 60), (900, 40),
                     (1200, 28)):
        vals = []
        for _ in range(nspr):
            P, p, q = sprinkle4(nn, rs, T)
            Rc, L = causal_links(P)
            m_ = len(P)
            rho = m_ / Vd
            a = math.sqrt(rho) / (2 * math.pi * math.sqrt(6.0))
            b = -(mT / T) ** 2 / rho
            vals.append(prop_entry(L, a, b, p, q))
        v = np.array(vals)
        mn, sd = float(v.mean()), float(v.std(ddof=1))
        rel = sd / abs(mn)
        rho0 = (nn + 2) / Vd
        rows.append((nn + 2, rho0, mn, sd, rel, nspr))
        print(f'{nn+2:>6}{rho0:>9.0f}{mn:>14.6e}{sd:>13.4e}'
              f'{rel:>14.5f}{nspr:>9}', flush=True)
    R = np.array([(r[0], r[4]) for r in rows], dtype=float)
    err = np.array([r[4] / math.sqrt(2 * (r[5] - 1)) for r in rows])
    x, y = np.log(R[:, 0]), np.log(R[:, 1])
    w = 1.0 / (err / R[:, 1]) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    be = cov @ (A.T @ W @ y)
    se = np.sqrt(np.diag(cov))
    print()
    print(f'  ЗАКОН 3+1: отн. флуктуация ~ N^a,  '
          f'a = {be[0]:.4f} ± {se[0]:.4f}')
    print(f'  1+1 (замер): −0.5558 ± 0.0267 -> отличие '
          f'{abs(be[0]+0.5558)/math.hypot(se[0],0.0267):.1f} сигмы')
    print(f'  часы (замер): −0.5296 ± 0.0245 -> отличие '
          f'{abs(be[0]+0.5296)/math.hypot(se[0],0.0245):.1f} сигмы')
    print(f'  пуассон −0.5 -> отличие {abs(be[0]+0.5)/se[0]:.1f} сигмы')
    print(f'  «по звеньям» −0.75 -> отличие '
          f'{abs(be[0]+0.75)/se[0]:.1f} сигмы')
    res = (y - A @ be) / (err / R[:, 1])
    print('  остатки (сигм):', ', '.join(f'{r:+.1f}' for r in res))


if __name__ == '__main__':
    main()
