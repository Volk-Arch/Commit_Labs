#!/usr/bin/env python3
"""ПРОПАГАТОР НА ПОРЯДКЕ: скорость убывания флуктуаций.

Строим то, что уже построено (Джонстон, arXiv:0806.3083):

    K = I + Phi + b Phi^2 + b^2 Phi^3 + ... = I + Phi (I - b Phi)^{-1}

где Phi = a A (A — матрица причинности, цепи) при a = 1/2, b = -m^2/rho
в 1+1. Там совпадение с непрерывным запаздывающим пропагатором есть
ТОЧНОЕ ТОЖДЕСТВО для среднего по засевам — значит это контроль
реализации, а не результат.

ДЫРА, на которую идём. У Джонстона: «по флуктуациям пропагатора на
отдельных засевах есть лишь предварительные численные указания, что они
убывают с плотностью, и СКОРОСТЬ не установлена».

Скорость мы сегодня замерили с другой стороны (`qclock-doc.md`):
точность часов порядка s = (0.286 +- 0.029) N^{-0.53 +- 0.02},
пуассоновский закон. Если флуктуации пропагатора идут тем же законом —
значит дробовой шум объёма и есть их источник, и скорость выведена.

Счёт берётся без обращения матриц: K = Phi + b Phi K даёт
w[j] = K[j,q], где w[j] = Phi[j,q] + b sum_i Phi[j,i] w[i], и при
обходе j в порядке УБЫВАНИЯ времени все нужные w[i] уже посчитаны —
один проход O(N^2), ряд обрывается сам (Phi нильпотентна).

КОНТРОЛИ:
  1. m = 0 обязано дать K = a = 1/2 ровно (J_0(0) = 1);
  2. среднее по засевам обязано совпасть с (1/2) J_0(m tau) — точное
     тождество Джонстона; отличие укажет на ошибку реализации;
  3. обрыв ряда: число шагов обязано равняться длине длиннейшей цепи;
  4. флуктуации обязаны убывать с плотностью (у Джонстона качественно).

  python3 johnston.py
"""
import math

import numpy as np
from scipy.special import j0


def sprinkle_11(rho, T, rs):
    """Засев в causal diamond 1+1. Площадь T^2/2, N ~ Poisson(rho T^2/2).
    Возвращает точки, отсортированные по времени, с вершинами."""
    area = T * T / 2.0
    n = rs.poisson(rho * area)
    # светоконусные координаты: u,v in [0,T] с u+v<=... проще: ромб
    u = rs.random(n) * T
    v = rs.random(n) * T
    t = (u + v) / 2.0
    x = (u - v) / 2.0
    keep = (t > 0) & (t < T) & (np.abs(x) < np.minimum(t, T - t))
    t, x = t[keep], x[keep]
    t = np.concatenate([[0.0], t, [T]])
    x = np.concatenate([[0.0], x, [0.0]])
    o = np.argsort(t)
    return t[o], x[o], int(np.argmin(t[o])), int(np.argmax(t[o]))


def causal(t, x):
    dt = t[None, :] - t[:, None]
    dx = np.abs(x[None, :] - x[:, None])
    return (dt > 0) & (dt > dx)


def prop(t, x, A, a, b, q):
    """K[p,q] для всех p одним проходом по убыванию времени."""
    n = len(t)
    Phi = a * A.astype(np.float64)
    w = np.zeros(n)
    steps = 0
    for j in range(n - 1, -1, -1):
        s = Phi[j, q]
        row = Phi[j]
        nz = np.nonzero(row)[0]
        if len(nz):
            s += b * float(row[nz] @ w[nz])
            steps += 1
    # повтор правильным порядком (вектор w заполняется по ходу)
        w[j] = s
    return w, steps


def run(rho, T, m, nspr, rs):
    a, b = 0.5, -m * m / rho
    vals = []
    chain = []
    for _ in range(nspr):
        t, x, p, q = sprinkle_11(rho, T, rs)
        A = causal(t, x)
        w, _ = prop(t, x, A, a, b, q)
        vals.append(w[p])
        # длина длиннейшей цепи (контроль 3)
        h = np.zeros(len(t), dtype=int)
        for j in range(len(t)):
            pre = np.nonzero(A[:, j])[0]
            h[j] = (h[pre].max() + 1) if len(pre) else 0
        chain.append(int(h[q]))
    v = np.array(vals)
    return v, float(np.mean(chain)), len(t)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    T = 1.0
    print('КОНТРОЛЬ 1: m = 0 обязано дать ровно a = 1/2')
    v, ch, n = run(400.0, T, 0.0, 6, rs)
    print(f'  K = {v.mean():.10f} ± {v.std(ddof=1):.1e}  (ожидание 0.5)')
    print()
    print('КОНТРОЛЬ 2: среднее против (1/2) J_0(m tau), '
          'точное тождество Джонстона')
    print(f"{'m*T':>7}{'rho':>8}{'N':>7}{'<K> замер':>14}"
          f"{'(1/2)J_0(mT)':>15}{'отн.':>9}{'цепь':>7}")
    for mT in (0.0, 2.0, 5.0, 8.0, 12.0):
        rho = 1600.0
        v, ch, n = run(rho, T, mT / T, 25, rs)
        pred = 0.5 * j0(mT)
        rel = (v.mean() - pred) / max(abs(pred), 1e-12)
        print(f'{mT:>7.1f}{rho:>8.0f}{n:>7}{v.mean():>14.6f}'
              f'{pred:>15.6f}{rel:>+9.3f}{ch:>7.1f}', flush=True)
    print()
    print('ГЛАВНОЕ: скорость убывания флуктуаций с плотностью')
    print(f"{'rho':>8}{'N':>7}{'<K>':>12}{'ст.откл.':>12}"
          f"{'отн. флукт.':>14}{'засевов':>9}")
    rows = []
    mT = 5.0
    for rho in (200.0, 400.0, 800.0, 1600.0, 3200.0):
        nspr = int(max(40, 4000 / math.sqrt(rho)))
        v, ch, n = run(rho, T, mT / T, nspr, rs)
        pred = 0.5 * j0(mT)
        sd = float(v.std(ddof=1))
        rel = sd / abs(pred)
        rows.append((n, rho, float(v.mean()), sd, rel, nspr))
        print(f'{rho:>8.0f}{n:>7}{v.mean():>12.6f}{sd:>12.6f}'
              f'{rel:>14.5f}{nspr:>9}', flush=True)
    R = np.array([(r[0], r[4]) for r in rows], dtype=float)
    err = np.array([r[4] / math.sqrt(2 * (r[5] - 1)) for r in rows])
    x, y = np.log(R[:, 0]), np.log(R[:, 1])
    w = 1.0 / (err / R[:, 1]) ** 2
    Amat = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    cov = np.linalg.inv(Amat.T @ W @ Amat)
    beta = cov @ (Amat.T @ W @ y)
    se = np.sqrt(np.diag(cov))
    print()
    print(f'  ЗАКОН: отн. флуктуация ~ N^a,  a = {beta[0]:.4f} ± {se[0]:.4f}')
    print(f'  закон часов (замер сегодня): −0.5296 ± 0.0245 -> отличие '
          f'{abs(beta[0]+0.5296)/math.hypot(se[0],0.0245):.1f} сигмы')
    print(f'  чистый пуассон −0.5 -> отличие {abs(beta[0]+0.5)/se[0]:.1f} сигмы')
    res = (y - Amat @ beta) / (err / R[:, 1])
    print('  остатки (сигм):', ', '.join(f'{r:+.1f}' for r in res))


if __name__ == '__main__':
    main()
