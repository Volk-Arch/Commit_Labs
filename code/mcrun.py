#!/usr/bin/env python3
"""Обваливается ли exp(-beta*S): длинная цепь под нашими дискриминаторами.

Пилот mc.py показал, что <S> падает без остановки (-96 -> -389 -> -502 за
шесть развёрток), а плотность растёт в середине ОДИНАКОВО у трубки и у
рассыпанного контроля. Это похоже не на притяжение к объекту, а на
слипание вообще, то есть на обвал.

Решающая проверка — наши стандартные дискриминаторы по ходу цепи. Если
расстановка остаётся манифолдоподобной, доля пар держится около 0.10 и
размерность около 4. Если обваливается, они уедут.

  python3 mcrun.py
  python3 mcrun.py --sweeps 200 --betas 0 0.3 1
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim, height
from mc import World, box, tube, T_MAX


def diagnose(w, M):
    n = len(w.R)
    f = float(w.R.sum() / (n * (n - 1) / 2))
    r = np.linalg.norm(w.X[M:, 1:], axis=1)
    t = w.X[M:, 0]
    return (f, mm_dim(f), height(w.R)[0], float(w.S),
            float(np.std(r)), float(np.std(t)))


def main():
    ap = argparse.ArgumentParser(description='Обвал exp(-beta S)')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--l', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--sweeps', type=int, default=120)
    ap.add_argument('--step', type=float, default=0.4)
    ap.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 1.0])
    a = ap.parse_args()

    t0 = time.time()
    print(f'фон {a.n}, объект {a.m} (трубка), L={a.l}, eps={a.eps}, '
          f'шаг {a.step}')
    print('манифолдоподобие: доля пар ~0.10, размерность ~4, '
          'разбросы ~однородные')
    print()

    for beta in a.betas:
        rs = np.random.default_rng(1234)
        obj = tube(a.m, 0.05, rs)
        w = World(np.vstack([obj, box(a.n, a.l, rs)]), a.eps)
        print(f'beta = {beta}')
        print(f"{'развёртка':>10}{'доля пар':>10}{'размерн.':>10}{'высота':>8}"
              f"{'S':>11}{'разброс |x|':>13}{'разброс t':>11}{'принят':>8}")
        d = diagnose(w, a.m)
        print(f'{0:>10}{d[0]:>10.4f}{d[1]:>10.3f}{d[2]:>8.0f}{d[3]:>11.1f}'
              f'{d[4]:>13.3f}{d[5]:>11.3f}{1.0:>8.2f}', flush=True)
        acc = 0
        for sw in range(1, a.sweeps + 1):
            for _ in range(a.n):
                i = a.m + int(rs.integers(a.n))
                xi = w.X[i].copy()
                xi[0] = min(max(xi[0] + a.step * rs.normal(), 0.0), T_MAX)
                xi[1:] = np.clip(xi[1:] + a.step * rs.normal(size=3),
                                 -a.l, a.l)
                st = w.move(i, xi)
                dS = w.S - st[8]
                if dS <= 0 or rs.random() < math.exp(-beta * dS):
                    acc += 1
                else:
                    w.undo(st)
            if sw % max(1, a.sweeps // 8) == 0 or sw == a.sweeps:
                d = diagnose(w, a.m)
                print(f'{sw:>10}{d[0]:>10.4f}{d[1]:>10.3f}{d[2]:>8.0f}'
                      f'{d[3]:>11.1f}{d[4]:>13.3f}{d[5]:>11.3f}'
                      f'{acc/(sw*a.n):>8.2f}', flush=True)
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
