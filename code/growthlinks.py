#!/usr/bin/env python3
"""ЧЕМ ИМЕННО МЕРА РОСТА НАКАЗЫВАЕТ ПРОСТРАНСТВО-ВРЕМЯ: связями.

Вес p^L означает плату p < 1 за КАЖДУЮ связь. Если у многообразных
порядков связей существенно больше, чем у соперников, мера роста их
наказывает — независимо от значения p. Здесь это замеряется прямо.
"""
import math

import numpy as np

from growthst import sprinkle, rel_links, perc_r
from growthcmp import ln_w_st
from growthcmp2 import ln_w_chain, ln_w_layer_mc


def perc_links(n, p, rs):
    A = np.triu(rs.random((n, n)) < p, 1)
    Rc = A.copy()
    while True:
        nxt = Rc | ((Rc.astype(np.float32) @ Rc.astype(np.float32)) > 0.5)
        if (nxt == Rc).all():
            break
        Rc = nxt
    two = (Rc.astype(np.float32) @ Rc.astype(np.float32)) > 0.5
    return int(Rc.sum()), int((Rc & ~two).sum())


def main():
    print(__doc__.strip())
    print()
    rs = np.random.default_rng(606)
    print('ЧИСЛО СВЯЗЕЙ при одинаковой доле отношений r = 0.1 (d = 4)')
    print(f"{'N':>6}{'p* замер.':>11}{'L засева':>11}{'L перколяции':>14}"
          f"{'L цепи':>9}{'отношение':>11}")
    for n in (100, 200, 400, 800):
        ps = 1.15 * n ** -0.75
        ls = []
        for _ in range(3 if n <= 400 else 1):
            P = sprinkle(n, rs)
            _, L = rel_links(P)
            ls.append(L)
        Lsp = float(np.mean(ls))
        lp = [perc_links(n, ps, rs)[1] for _ in range(3)]
        Lpc = float(np.mean(lp))
        print(f'{n:>6}{ps:>11.5f}{Lsp:>11.0f}{Lpc:>14.0f}'
              f'{n-1:>9}{Lsp/Lpc:>11.2f}', flush=True)
    print()
    print('ТОЧКА ЗАКРЫТИЯ ОКНА по p')
    PS = np.geomspace(0.003, 0.4, 36)
    for n in (160, 170, 180, 190, 200):
        best = None
        for _ in range(5):
            P = sprinkle(n, rs)
            R, L = rel_links(P)
            if best is None or L < best[1]:
                best = (R, L)
        R_st, L_st = best
        shapes = []
        for k in (2, 3, 4, 6, 10, 16, 24):
            if n // k < 2:
                continue
            a = n // k
            s = [a] * k
            s[-1] += n - a * k
            shapes.append(tuple(s))
        win = []
        for p in PS:
            vst = ln_w_st(n, L_st, R_st, float(p))
            opp = max([ln_w_chain(n, float(p))]
                      + [ln_w_layer_mc(s, float(p), rs, 12)[0]
                         for s in shapes])
            if vst > opp:
                win.append(float(p))
        txt = (f'[{min(win):.4f}, {max(win):.4f}]' if win else 'ПУСТО')
        print(f'  N = {n:>4}: окно {txt}', flush=True)


if __name__ == '__main__':
    main()
