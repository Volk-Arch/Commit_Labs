#!/usr/bin/env python3
"""ТРЕТИЙ ПРИБОР: обилия МАЛЫХ интервалов N_0…N_3 порознь.

Действие БД — одна знакопеременная сумма этих четырёх чисел, поэтому
оно говорит «не сходится», но не говорит ГДЕ. Здесь они печатаются
порознь, в виде отношений к числу звеньев, и это указывает на место
поломки.

ЗАЧЕМ. Из `local2.py` и `density.py` вышел закон: хорошие обилия цепей
достигаются только при `d ≲ 2`, а размерность 4 требует РАЗРЕЖЕННОГО
порядка — доля упорядоченных пар равна `2c₂(d)`, то есть 0.5 при `d = 2`
и всего 0.1 при `d = 4`. Рост умеет делать разреженность лишь
древовидностью, и подозрение, что ломается именно микроструктура.

ЧТО СЧИТАЕТСЯ. Для каждого множества: `r` — доля упорядоченных пар,
`N₀` — звенья (пары с пустым интервалом), и отношения `N₁/N₀`, `N₂/N₀`,
`N₃/N₀`. У сприклинга они зависят только от размерности, поэтому служат
эталоном, не требующим ни калибровки, ни подгонки.

  python3 layers.py
"""
import math
import time

import numpy as np

from grow import mm_dim, gen_sprinkle, gen_percolation
from csg import gen_csg
from bd import layers
from local2 import gen_loc2


def row(tag, gen, n, seeds, root, **kw):
    acc = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        nn, L = layers(A, 4)
        c1 = float(np.count_nonzero(A))
        del A
        r = c1 / (nn * (nn - 1) / 2.0)
        d = mm_dim(r / 2.0) if 0 < r < 1 else np.nan
        if L[0] == 0:
            continue
        acc.append((d, r, L[0] / nn, L[1] / L[0], L[2] / L[0], L[3] / L[0]))
    if not acc:
        print(f'{tag:>30}  пусто')
        return
    P = np.array(acc, float)
    m = np.nanmean(P, axis=0)
    e = (np.nanstd(P, axis=0, ddof=1) / math.sqrt(len(P))
         if len(P) > 1 else np.zeros(6))
    print(f'{tag:>30}{m[0]:>8.2f}{m[1]:>9.3f}{m[2]:>10.2f}'
          f'{m[3]:>9.3f}±{e[3]:<5.3f}{m[4]:>9.3f}±{e[4]:<5.3f}'
          f'{m[5]:>9.3f}±{e[5]:<5.3f}')


HDR = (f"{'что':>30}{'d':>8}{'r':>9}{'N₀/N':>10}"
       f"{'N₁/N₀':>15}{'N₂/N₀':>15}{'N₃/N₀':>15}")


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N = 1200
    print('=' * 102)
    print('ЭТАЛОН: сприклинг. Отношения зависят ТОЛЬКО от размерности.')
    print(HDR)
    for dd in (2, 3, 4, 5):
        row(f'сприклинг {dd}D', gen_sprinkle, N, 8, 300 + dd, d=dd)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 102)
    print('КПР: перколяция по всему диапазону')
    print(HDR)
    for p in (0.003, 0.01, 0.0316, 0.1, 0.3):
        row(f'перколяция p={p}', gen_percolation, N, 5, 800 + int(1e4 * p),
            p=p)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('КПР: разреженная область (там действие ближе к нулю)')
    print(HDR)
    for m in (1, 2, 3, 5, 10):
        row(f'постоянное m={m}', gen_csg, N, 5, 1100 + m,
            draw_m=(lambda j, rs, m=m: m))
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 102)
    print('ЛОКАЛЬНОЕ правило (не КПР): по ширине поверхности')
    print(HDR)
    for F in (5, 30, 200, 10 ** 9):
        row(f'локальное F={F:g} m=6', gen_loc2, N, 4,
            1500 + int(math.log1p(F) * 31), m=6, F=F)
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    for m in (2, 3, 12):
        row(f'локальное F=30 m={m}', gen_loc2, N, 4, 1700 + m, m=m, F=30)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
