#!/usr/bin/env python3
"""ЕСТЬ ЛИ p, ПРИ КОТОРОМ ПРОСТРАНСТВО-ВРЕМЯ ПОБИВАЕТ ВСЕХ.

Мера ОДНА, значит сравнивать надо при одном и том же p. Предыдущий
прогон сравнивал каждый сектор при его СОБСТВЕННОМ лучшем p — это
неверно (у цепи оптимум при p -> 1, у засева около 0.05). Здесь для
каждого N ищется окно по p, где пространство-время — максимум среди
трёх, и смотрится, закрывается ли оно с ростом N.
"""
import math

import numpy as np

from growthst import sprinkle, rel_links
from growthcmp import ln_w_st
from growthcmp2 import ln_w_chain, ln_w_layer_mc


def main():
    print(__doc__.strip())
    print()
    rs = np.random.default_rng(2024)
    PS = np.concatenate([np.geomspace(0.002, 0.5, 40)])
    print(f"{'N':>6}{'окно p, где побеждает засев':>32}"
          f"{'запас в окне':>14}{'главный соперник':>19}")
    for n in (80, 120, 160, 200, 240, 320):
        best = None
        for _ in range(5):
            P = sprinkle(n, rs)
            R, L = rel_links(P)
            if best is None or L < best[1]:
                best = (R, L)
        R_st, L_st = best
        ks = [k for k in (2, 3, 4, 6, 10, 16, 24) if n // k >= 2]
        shapes = []
        for k in ks:
            a = n // k
            s = [a] * k
            s[-1] += n - a * k
            shapes.append(tuple(s))
        win, gap, rival = [], -1e18, ''
        for p in PS:
            vst = ln_w_st(n, L_st, R_st, float(p))
            vch = ln_w_chain(n, float(p))
            vl = max(ln_w_layer_mc(s, float(p), rs, 12)[0] for s in shapes)
            opp = max(vch, vl)
            if vst > opp:
                win.append(float(p))
                if vst - opp > gap:
                    gap = vst - opp
                    rival = 'цепь' if vch >= vl else 'слоистый'
        if win:
            txt = f'[{min(win):.4f}, {max(win):.4f}]'
            print(f'{n:>6}{txt:>32}{gap:>14.1f}{rival:>19}', flush=True)
        else:
            print(f'{n:>6}{"ПУСТО":>32}{"—":>14}{"—":>19}', flush=True)
    print()
    print('Если окно пусто — ни одна мера из однопараметрического')
    print('семейства транзитивной перколяции не делает')
    print('пространство-время доминирующим сектором.')


if __name__ == '__main__':
    main()
