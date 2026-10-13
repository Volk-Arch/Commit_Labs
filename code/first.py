#!/usr/bin/env python3
"""КОЭФФИЦИЕНТ ПЕРВОГО ПОРЯДКА: по семействам и продлением к нулю.

ЗАЧЕМ. Отклик `v` вышел разным при разной силе кривизны: по всему
набору `-0.0201`, а в окне малых `T²R_00` — втрое больше. Полосы по
сырому `R_00` при этом разделили не наклоны, а сами семейства
(`h = 0.025` против `h = 0.040`), и у сильного отклик оказался вдвое
слабее.

Это подпись ВЫСШИХ ПОРЯДКОВ, и подпись правильного знака: ряд по
`T²R_00` обрезан на первом члене, и при `T²R_00` под единицу он уже не в
силе. Тот же урок, что был с ромбами, только теперь он виден изнутри
данных, а не из сравнения с чужой формулой.

ПОЧИНКА — подгонять `v` ОТДЕЛЬНО В КАЖДОМ СЕМЕЙСТВЕ, где `T²R_00` имеет
свой характерный размер, а затем продлить к `T²R_00 -> 0`. Первый
порядок есть предел, а не среднее по всему диапазону.

КОНТРОЛЬ: плоское семейство обязано дать `v = 0` и войти в продление
как точка в нуле.

  python3 first.py
"""
import math
import pickle

import numpy as np

from invert import SCRATCH, C1, X2, X3, AU, pool


def vfit(A, sd, col):
    """v при принудительном u = 0 (в линейном R тождественно нуль, а в
    квадратичном вклад R измерен как неотличимый от нуля), с мешающими
    членами. Ошибка — складным ножом по засевам."""
    D = np.column_stack([A[:, AU], np.ones(len(A)), 1.0 / np.sqrt(A[:, C1])])
    w = A[:, C1]
    y = A[:, col]

    def sol(m):
        Dw = D[m] * w[m][:, None]
        return np.linalg.solve(D[m].T @ Dw, Dw.T @ y[m])

    us = np.unique(sd)
    c = sol(np.ones(len(A), bool))
    jk = np.array([sol(sd != u) for u in us])
    k = len(us)
    e = math.sqrt((k - 1) / k * ((jk[:, 0] - jk[:, 0].mean()) ** 2).sum())
    return c[0], e


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/curve.pkl', 'rb') as fh:
        res = pickle.load(fh)
    flat = [k for k in res if k[0] == 'flat']
    Af, sf = pool(res, flat)

    print('=' * 70)
    print('v В КАЖДОМ СЕМЕЙСТВЕ ОТДЕЛЬНО (плоское добавлено в каждое,')
    print('чтобы мешающие члены были закреплены одинаково)')
    print()
    print(f"{'семейство':>18}{'|T²R_00|':>11}{'v(X_2)':>11}{'ст.ош':>9}"
          f"{'v(X_3)':>11}{'ст.ош':>9}")
    pts = []
    for k in sorted([k for k in res if k[0] != 'flat'], key=str):
        A, sd = pool(res, [k] + flat)
        # характерный размер параметра разложения в этом семействе
        B = res[k][0]
        au = float(np.average(np.abs(B[:, AU]), weights=B[:, C1]))
        v2, e2 = vfit(A, sd, X2)
        v3, e3 = vfit(A, sd, X3)
        pts.append((au, v2, e2, v3, e3))
        print(f'{k[0]+" "+str(k[1]):>18}{au:>11.3f}{v2:>+11.4f}{e2:>9.4f}'
              f'{v3:>+11.4f}{e3:>9.4f}')
    P = np.array(pts)
    print()
    print('ПРОДЛЕНИЕ К НУЛЮ: v(T²R_00) = v_0 + k*|T²R_00|, взвешенно')
    for j, nm in ((1, 'X_2'), (3, 'X_3')):
        y, e = P[:, j], P[:, j + 1]
        D = np.column_stack([np.ones(len(P)), P[:, 0]])
        W = np.diag(1.0 / e ** 2)
        cov = np.linalg.inv(D.T @ W @ D)
        c = cov @ (D.T @ W @ y)
        s = np.sqrt(np.diag(cov))
        pred = D @ c
        chi = float((((y - pred) / e) ** 2).sum())
        print(f'  {nm}:  v_0 = {c[0]:+.4f} +- {s[0]:.4f}'
              f'   (наклон {c[1]:+.4f} +- {s[1]:.4f},'
              f'  хи² = {chi:.2f} на {len(P)-2} степ.)')
    print()
    print('ЧТЕНИЕ. Если наклон значимо ненулевой — высшие порядки есть, и')
    print('первым порядком следует считать v_0, а не среднее по всему')
    print('диапазону. Если хи² мал, линейного продления достаточно.')


if __name__ == '__main__':
    main()
