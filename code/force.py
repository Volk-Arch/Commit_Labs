#!/usr/bin/env python3
"""Куда направлена сила: градиент действия у локального объекта.

ПОВОД. weight-doc.md: в связанном ансамбле exp(-beta*S) даёт отклик на
локальный объект, но знак ОТТАЛКИВАЮЩИЙ (6.1 сигмы обеднения у трубки).
Это известная проблема конформного множителя евклидовой гравитации.
Вопрос был, что даст exp(iS).

КЛЮЧЕВОЕ СООБРАЖЕНИЕ. Наивное «перевернуть знак» неверно: exp(i*beta*S)
с вещественным S и вещественной наблюдаемой даёт вещественную часть,
ЧЁТНУЮ по beta, то есть отклик второго порядка, тогда как exp(-beta*S)
даёт первого. Это разные величины.

Зато классическое содержание exp(iS) — СТАЦИОНАРНОСТЬ, dS = 0. Значит
вопрос о знаке ставится без всякой выборки:

    КУДА НАПРАВЛЕН ГРАДИЕНТ ДЕЙСТВИЯ?

Сила на событие F = -dS/dx считается центральной разностью, а приращение
действия при переносе одного события — инкрементно, за O(N^2).

Радиальная компонента F_r < 0 значит «к оси трубки», то есть
притяжение. Контроль: те же M закреплённых событий, рассыпанные по
ромбу, плюс смещение по ВРЕМЕНИ, которое систематической радиальной
сигнатуры давать не должно.

  python3 force.py
  python3 force.py --t 2.5 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from mc import World
from weight import diamond


def tube(M, rad, T, rs):
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(rad, T - rad, M)
    X[:, 1:] = u * r[:, None]
    return X


def in_diamond(x, T):
    r = np.linalg.norm(x[1:])
    return (x[0] >= r) and (T - x[0] >= r)


def forces(w, M, T, delta, mode, rs, nsamp=220):
    """Средняя сила по выборке фоновых событий, с привязкой к радиусу."""
    N = len(w.X)
    cand = np.arange(M, N)
    rr = np.linalg.norm(w.X[cand, 1:], axis=1)
    # берём события из внутренней части мира
    keep = (np.abs(w.X[cand, 0] - T / 2) < 0.35 * T) & (rr < 0.6)
    cand = cand[keep]
    if len(cand) > nsamp:
        cand = rs.choice(cand, nsamp, replace=False)
    out = []
    for i in cand:
        i = int(i)
        x0 = w.X[i].copy()
        r = np.linalg.norm(x0[1:])
        if r < 1e-6:
            continue
        if mode == 'rad':
            e = np.zeros(4)
            e[1:] = x0[1:] / r               # наружу от оси
        else:
            e = np.array([1.0, 0.0, 0.0, 0.0])   # по времени, контроль
        xp, xm = x0 + delta * e, x0 - delta * e
        if not (in_diamond(xp, T) and in_diamond(xm, T)):
            continue
        st = w.move(i, xp)
        sp = w.S
        w.undo(st)
        st = w.move(i, xm)
        sm = w.S
        w.undo(st)
        out.append((r, -(sp - sm) / (2 * delta)))
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser(description='Градиент действия у объекта')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--masses', type=int, nargs='+',
                    default=[120, 240, 480, 960])
    ap.add_argument('--rad', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--delta', type=float, default=0.02)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--rho', type=float, default=150.0,
                    help='N = rho * T^4; масштаб дискретности '
                         'меняется как rho^(-1/4)')
    a = ap.parse_args()

    t0 = time.time()
    edges = np.array([0.0, 0.1, 0.18, 0.28, 0.42, 0.6])
    print(f'ромб T={a.t}, радиус трубки {a.rad}, '
          f'eps={a.eps}, шаг {a.delta}')
    rho = a.rho * 24 / math.pi
    print(f'массы {a.masses}; плотность {rho:.0f}, '
          f'масштаб дискретности {rho**-0.25:.3f}')
    print('F_r < 0 значит сила К ОСИ, то есть ПРИТЯЖЕНИЕ')
    print(f'засевов {a.seeds}')
    print()

    for M in a.masses:
        print(f'МАССА ТРУБКИ M = {M}')
        print(f"{'мир':>10}" + ''.join(f'{e:>12.2f}' for e in edges[1:]))
        mode = 'rad'
        res = {}
        for mk in (True, False):
            acc = [[] for _ in range(len(edges) - 1)]
            for t in range(a.seeds):
                rs = np.random.default_rng(201000 + 977 * t)
                bg = diamond(a.t, rs, n_at_t1=a.rho)
                ob = (tube(M, a.rad, a.t, rs) if mk
                      else diamond(a.t, rs, n_at_t1=M / a.t ** 4))
                w = World(np.vstack([ob, bg]), a.eps)
                F = forces(w, len(ob), a.t, a.delta, mode, rs)
                del w
                if len(F) == 0:
                    continue
                k = np.digitize(F[:, 0], edges) - 1
                for b in range(len(acc)):
                    m = k == b
                    if m.sum() > 2:
                        acc[b].append(F[m, 1].mean())
            m = np.array([np.mean(v) if v else np.nan for v in acc])
            e = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1
                          else np.nan for v in acc])
            res[mk] = (m, e)
            name = 'трубка' if mk else 'контроль'
            print(f'{name:>10}' + ''.join(f'{x:>+12.4f}' for x in m))
            print(f"{'± ошибка':>10}" + ''.join(f'{x:>12.4f}' for x in e))
        d = res[True][0] - res[False][0]
        er = np.hypot(res[True][1], res[False][1])
        print(f"{'разница':>10}" + ''.join(f'{x:>+12.4f}' for x in d))
        print(f"{'значим.':>10}"
              + ''.join(f'{(abs(x)/y if y > 1e-9 else float("nan")):>12.1f}'
                        for x, y in zip(d, er)), flush=True)
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
