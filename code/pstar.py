#!/usr/bin/env python3
"""p*(N) С ПОГРЕШНОСТЬЮ. Прежний замер шёл по 2-4 выборкам, а разброс
одной выборки у перколяции достигает 0.05 при доле 0.29 — то есть корень
искался по шуму. Здесь на каждую точку много выборок, и весь поиск
корня повторяется независимо, чтобы погрешность p* была МЕРЕНОЙ.
"""
import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma

from growthst import perc_r, c2

TARGET = 2 * c2(4.0)


def rmean(n, p, rs, reps):
    return float(np.mean([perc_r(n, p, rs) for _ in range(reps)]))


def one_pstar(n, rs, reps):
    return brentq(lambda p: rmean(n, p, rs, reps) - TARGET,
                  1e-5, 0.6, xtol=2e-5)


def main():
    print(__doc__.strip())
    print()
    rs = np.random.default_rng(777)
    print(f"цель r = {TARGET:.4f}")
    print(f"{'N':>7}{'p*':>11}{'σ(p*)':>10}{'p* N':>9}"
          f"{'p* N^0.75':>11}{'выборок':>9}")
    out = {}
    for n in (20, 40, 80, 160, 320, 640, 1280):
        reps = max(6, int(1500 / n))
        est = [one_pstar(n, rs, reps) for _ in range(3)]
        p = float(np.mean(est))
        s = float(np.std(est, ddof=1) / math.sqrt(len(est)))
        out[n] = (p, s)
        print(f'{n:>7}{p:>11.6f}{s:>10.6f}{p*n:>9.3f}'
              f'{p*n**0.75:>11.3f}{reps:>9}', flush=True)
    ns = np.array(sorted(out))
    pv = np.array([out[n][0] for n in ns])
    sv = np.array([out[n][1] for n in ns])
    msk = ns >= 80
    a, b = np.polyfit(np.log(ns[msk]), np.log(pv[msk]), 1)
    print()
    print(f'подгонка p* = C N^a по N >= 80: a = {a:+.4f}, '
          f'C = {math.exp(b):.4f}')
    pred = math.exp(b) * 10000.0 ** a
    print(f'  экстраполяция на N = 10^4: p* = {pred:.6f}')
    json.dump({str(k): v for k, v in out.items()},
              open('/home/claude/data/pstar.json', 'w'), indent=1)
    print('записано в /home/claude/data/pstar.json')


if __name__ == '__main__':
    main()
