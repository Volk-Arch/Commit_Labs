#!/usr/bin/env python3
"""Беднота в примитиве или в правиле: конус из конечного числа направлений.

not-minkowski-doc.md показал, что правило «раньше по всем k шкалам» даёт
посет размерности не выше k, а причинность Минковского при d >= 3 имеет
неограниченную размерность. Отсюда вопрос: дело в том, что события
несут мало чисел, или в том, как эти числа сравниваются?

Проверка прямая. Берём ОДНИ И ТЕ ЖЕ четыре числа на событие и два
правила:

  покомпонентное   раньше по всем четырём шкалам      (наше)
  конусное         dt >= |dx|, то есть время больше    (Минковский)
                   пространственной нормы

и промежуточное семейство: конус, приближённый n направлениями,

  для всех u из U:  dt >= u·dx

Каждое направление даёт свой линейный порядок, поэтому размерность
порядка не выше n, а при n -> бесконечность получается Минковский. Так
видно, сколько линейных порядков нужно, чтобы приблизить причинность.

  python3 cones.py
  python3 cones.py --n 2000 --dim 4
"""
import argparse
import math
import time

import numpy as np

from core import rel_of, dimension, height, spectral, frac_pairs


def sprinkle_diamond(N, d, rs):
    """Засев в причинном ромбе Минковского: |t| + |x| <= 1."""
    pts = []
    while len(pts) < N:
        c = rs.uniform(-1, 1, (4 * N, d))
        ok = np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= 1
        pts.extend(c[ok][:N - len(pts)])
    return np.array(pts[:N])


def fibonacci_directions(n, d):
    """n направлений, более-менее равномерно на сфере в R^(d-1)."""
    m = d - 1
    rs = np.random.default_rng(0)
    if m == 1:
        return np.array([[1.0], [-1.0]])
    v = rs.normal(size=(n, m))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def cone_relation(X, dirs=None):
    """R[i,j] = True, если j раньше i.

    dirs=None — точный конус: dt >= |dx|.
    Иначе многогранный конус по заданным направлениям.
    """
    dt = X[:, None, 0] - X[None, :, 0]          # dt[i,j] = t_i - t_j
    dx = X[:, None, 1:] - X[None, :, 1:]
    if dirs is None:
        R = dt > np.linalg.norm(dx, axis=-1)
    else:
        R = dt > 0
        for u in dirs:
            R &= (dt >= dx @ u)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def stats(R):
    return (frac_pairs(R), dimension(R=R), height(R)[0], spectral(R))


def main():
    ap = argparse.ArgumentParser(
        description='Конус из конечного числа направлений')
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--dim', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    d = a.dim

    def agg(build):
        V = []
        for s in range(a.seeds):
            rs = np.random.default_rng(100 + 17 * s)
            V.append(stats(build(rs)))
        V = np.array(V)
        return V.mean(0), V.std(0) / math.sqrt(a.seeds)

    # эталон: точный конус
    ref, refsd = agg(lambda rs: cone_relation(sprinkle_diamond(a.n, d, rs)))
    print(f'd={d}, N={a.n}, семян {a.seeds}')
    print(f'ЭТАЛОН — точный конус (Минковский): доля {ref[0]:.4f}, '
          f'разм. {ref[1]:.3f}, выс. {ref[2]:.1f}, спектр {ref[3]:+.4f}')
    print()

    print('ТЕ ЖЕ ЧЕТЫРЕ ЧИСЛА, ДРУГОЕ ПРАВИЛО')
    box, boxsd = agg(lambda rs: rel_of(sprinkle_diamond(a.n, d, rs)))
    print(f'{"покомпонентное (наше)":>26}: доля {box[0]:.4f}, '
          f'разм. {box[1]:.3f}, выс. {box[2]:.1f}, спектр {box[3]:+.4f}')
    print(f'{"конусное (Минковский)":>26}: доля {ref[0]:.4f}, '
          f'разм. {ref[1]:.3f}, выс. {ref[2]:.1f}, спектр {ref[3]:+.4f}')
    print('  те же события, те же координаты — различие только в правиле')
    print()

    print('СКОЛЬКО НАПРАВЛЕНИЙ НУЖНО')
    print(f"{'направлений':>12}{'доля пар':>11}{'размерность':>13}"
          f"{'высота':>9}{'спектр':>11}{'худшее от эталона':>19}")
    for n_dir in (4, 6, 10, 20, 40, 80, 160):
        dirs = fibonacci_directions(n_dir, d)
        mu, sd = agg(lambda rs, D=dirs: cone_relation(
            sprinkle_diamond(a.n, d, rs), D))
        zs = [abs(mu[i] - ref[i]) / max(math.hypot(sd[i], refsd[i]), 1e-9)
              for i in range(4)]
        print(f'{n_dir:>12}{mu[0]:>11.4f}{mu[1]:>13.3f}{mu[2]:>9.1f}'
              f'{mu[3]:>+11.4f}{max(zs):>18.1f}σ')
    print()
    print('  ориентир: наше покомпонентное правило отстоит от эталона')
    zb = [abs(box[i] - ref[i]) / max(math.hypot(boxsd[i], refsd[i]), 1e-9)
          for i in range(4)]
    print(f'  на {max(zb):.1f}σ (худшее из четырёх)')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
