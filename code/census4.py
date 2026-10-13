#!/usr/bin/env python3
"""ПЕРЕПИСЬ: две проверки на честность собственных выводов.

census3.py при широком наборе соперников дал:
  объёмные          НЕ РАЗДЕЛЯЕТ  -1308
  объёмные+граница  НЕ РАЗДЕЛЯЕТ   -896   <- архив давал РАЗДЕЛЯЕТ
  распределение     НЕ РАЗДЕЛЯЕТ  -1872
  распред.+граница  РАЗДЕЛЯЕТ     +1229

ПРОВЕРКА 1 (против своего положительного вывода). Не держится ли
последнее разделение на ПУСТОЙ ПОЛОСЕ — координате, где у всех
соперников ноль, а у засева нет. Такое разделение есть артефакт
размера N и узости семейства, а не действие: оно говорит «награждай
пары с размером интервала 162…243», чего слоистый при N=400 просто не
умеет. Проверяется выбрасыванием таких полос и пересчётом.

ПРОВЕРКА 2 (против своих отрицательных выводов). Энтропия соперников
взята как Σ a_i a_{i+1} H2(q) — число свободных двудольных матриц. Но
замыкание склеивает разные матрицы в один порядок, значит это
ЗАВЫШЕНИЕ, а завышение энтропии соперника — ровно та сторона, что
создаёт ложное «не разделяет». Проверяется умножением энтропий
соперников на коэффициент 1.0, 0.7, 0.5, 0.3: если вердикт
«не разделяет» переживает 0.3, он устойчив.
"""
import math

import numpy as np

from census1 import EDGES, feats, sprinkle_rc
from census2 import certify, make, solve, verdict
from census3 import perc, sprink_d


def build(n, rs, S4):
    pool = []
    for k in (2, 3, 4, 6, 9, 14, 22, 34, 55, 90, 140, 220):
        for q in (0.03, 0.08, 0.2, 0.45, 0.75, 0.95):
            for kind in (0, 1, 2):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    for p in (0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5):
        pool.append(perc(n, p, rs))
    for d in (2, 3, 5, 6):
        for _ in range(3):
            pool.append(sprink_d(n, d, rs, S4))
    return pool


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(31415926)
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, _ = feats(sprinkle_rc(n, rs), n)
    pool = build(n, rs, S4)
    Sj = np.array([r[3] for r in pool])
    print(f'соперников {len(pool)}, N = {n}')

    f4 = np.concatenate([ab4, b4])
    Fj = np.array([np.concatenate([r[1], r[2]]) for r in pool], float)
    sc = np.maximum(np.abs(f4), 1.0)

    print('\nПРОВЕРКА 1: пустые полосы')
    empty = [j for j in range(len(f4))
             if Fj[:, j].max() == 0 and f4[j] > 0]
    nb = len(ab4)
    names = [f'полоса m∈[{EDGES[j]}..'
             f'{EDGES[j+1] if EDGES[j+1] < 10**8 else "∞"})' if j < nb
             else ('макс. элементов' if j == nb else 'мин. элементов')
             for j in range(len(f4))]
    print('  пустые координаты: '
          + ('; '.join(f'{j} = {names[j]} (у засева {f4[j]:.0f})'
                       for j in empty) if empty else 'таких нет'))
    for lab, keep in (('все координаты', list(range(len(f4)))),
                      ('без пустых полос',
                       [j for j in range(len(f4)) if j not in empty])):
        G = (f4[None, keep] - Fj[:, keep]) / sc[keep]
        t, _ = solve(G, Sj - S4)
        print(f'  {lab} ({len(keep)}): {verdict(t)}')

    print('\nПРОВЕРКА 2: устойчивость отрицательных вердиктов к '
          'завышению энтропии соперников')
    sets = [('восемь объёмных', [r[0] for r in pool], o4),
            ('объёмные + граничный',
             [np.concatenate([r[0], r[2]]) for r in pool],
             np.concatenate([o4, b4])),
            ('распределение', [r[1] for r in pool], ab4)]
    print(f"{'набор':>22}" + ''.join(f'{f"x{a}":>12}'
                                     for a in (1.0, 0.7, 0.5, 0.3, 0.0)))
    for nm, fj, ff in sets:
        Fj2 = np.array(fj, float)
        s2 = np.maximum(np.abs(ff), 1.0)
        G = (ff[None, :] - Fj2) / s2
        line = f'{nm:>22}'
        for a in (1.0, 0.7, 0.5, 0.3, 0.0):
            t, _ = solve(G, Sj * a - S4)
            line += f'{("НЕТ" if (t is not None and t <= 0) else "ДА"):>12}'
        print(line)
    print('  («НЕТ» = разделения нет; «ДА» = есть. x0.0 это предел, '
          'где энтропия соперников вовсе отброшена,')
    print('   то есть остаётся чистое разделение счётов.)')


if __name__ == '__main__':
    main()
