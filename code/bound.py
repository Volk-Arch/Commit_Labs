#!/usr/bin/env python3
"""ПРЕПЯТСТВИЕ С МЕХАНИЗМОМ: ЗВЕНЬЯ.

ВЫВОД, который проверяется. В последовательном росте прошлое нового
элемента `j` есть замыкание вниз случайного `m`-подмножества `S`,
поэтому ЗВЕНЬЯ элемента `j` — ровно максимальные элементы `S`, и

    #звеньев(j) = |max(S)| ≤ m.

При доле упорядоченных пар `r` данный элемент `S` максимален в `S` с
вероятностью примерно `(1 − r/2)^(m−1)`, значит в среднем

    ⟨звеньев⟩ ≈ m (1 − r/2)^(m−1)  ≤  max_m [...]  ≈  2/(e·r),

и эта граница ОТ `N` НЕ ЗАВИСИТ. А сприклингу нужно `N₀/N ~ N^(1−1/d)`,
то есть требование РАСТЁТ с размером. Если так, расхождение не
конечноразмерное, а структурное и с размером усиливается.

ДВЕ ПРОВЕРКИ:
  1. у сприклинга `N₀/N` растёт как `N^(1−1/d)` — показатель 0.75 при
     `d = 4` и 0.5 при `d = 2`;
  2. у КПР `N₀/N` при настройке на максимум ОСТАЁТСЯ ограниченным и не
     превосходит `2/(e·r)`.

Показатель берётся наклоном логарифмов, ошибка — складным ножом по
засевам.

  python3 bound.py
"""
import math
import time

import numpy as np

from grow import mm_dim, c_k, gen_percolation
from csg import gen_csg
from bd import layers
from sprinkle import sprinkle_d
from local2 import gen_loc2


def links(gen, n, seeds, root, **kw):
    out = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        nn, L = layers(A, 1)
        c1 = float(np.count_nonzero(A))
        del A
        r = c1 / (nn * (nn - 1) / 2.0)
        out.append((L[0] / nn, r))
    P = np.array(out, float)
    return (P[:, 0].mean(), P[:, 0].std(ddof=1) / math.sqrt(len(P)),
            P[:, 1].mean())


def slope(NS, YS):
    x = np.log(np.array(NS, float))
    y = np.log(np.array(YS, float))
    k = np.polyfit(x, y, 1)[0]
    return k


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    NS = (300, 600, 1200, 2400)
    print('=' * 92)
    print('ПРОВЕРКА 1. У сприклинга звенья на элемент растут как N^(1−1/d)?')
    print(f"{'d':>4}{'':>6}" + ''.join(f'{n:>12}' for n in NS)
          + f"{'показатель':>14}{'ожидание':>11}")
    for dd in (2, 3, 4):
        ys, es = [], []
        for n in NS:
            m, e, r = links(sprinkle_d, n, 6, 900 + n + dd, d=dd)
            ys.append(m); es.append(e)
        k = slope(NS, ys)
        print(f'{dd:>4}D{"":>5}' + ''.join(f'{y:>8.2f}±{e:<3.2f}'
                                           for y, e in zip(ys, es))
              + f'{k:>14.3f}{1-1/dd:>11.3f}')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 92)
    print('ПРОВЕРКА 2. У роста звенья ограничены? И держится ли 2/(e·r)?')
    print(f"{'правило':>28}{'N':>7}{'r':>8}{'звеньев/эл':>13}"
          f"{'граница 2/(e·r)':>17}{'доля границы':>14}")
    rules = [
        ('перколяция p=0.01', gen_percolation, dict(p=0.01)),
        ('перколяция p=0.0316', gen_percolation, dict(p=0.0316)),
        ('постоянное m=5', gen_csg, dict(draw_m=lambda j, rs: 5)),
        ('постоянное m=20', gen_csg, dict(draw_m=lambda j, rs: 20)),
        ('постоянное m=60', gen_csg, dict(draw_m=lambda j, rs: 60)),
        ('локальное F=1e9 m=6', gen_loc2, dict(m=6, F=10 ** 9)),
        ('локальное F=30 m=12', gen_loc2, dict(m=12, F=30)),
    ]
    for tag, gen, kw in rules:
        ys = []
        for n in NS:
            m, e, r = links(gen, n, 4, 2000 + n + len(tag), **kw)
            b = 2.0 / (math.e * r) if r > 0 else float('inf')
            ys.append(m)
            print(f'{tag:>28}{n:>7}{r:>8.4f}{m:>8.2f}±{e:<4.2f}'
                  f'{b:>17.2f}{m/b:>14.2f}')
        print(f'{"":>28}  показатель по N: {slope(NS, ys):+.3f}'
              f'   ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 92)
    print('ИТОГ: чего требует четырёхмерие и что даёт рост')
    for n in (300, 1200, 2400, 4800):
        m, e, r = links(sprinkle_d, n, 4, 5000 + n, d=4)
        print(f'    сприклинг 4D, N={n:>5}: звеньев/эл = {m:>6.2f}, '
              f'r = {r:.4f}')
    print(f'    граница роста при r = {2*c_k(2,4):.4f} (то есть d = 4): '
          f'{2.0/(math.e*2*c_k(2,4)):.2f} — и она от N не зависит')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
