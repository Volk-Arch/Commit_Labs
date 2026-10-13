#!/usr/bin/env python3
"""ПРАВИЛЬНАЯ ПОСТАНОВКА: выполнимость без искусственной границы.

КОНТРОЛЬ 2 первого прогона провалился при N = 600: цель «цепь» вышла
невыполнимой, а цепь РАЗДЕЛИМА по построению (у неё максимум R, C3, C4
и высоты). Значит упёрлось в границу |theta| <= 1e4, которую я ввёл
руками: энтропийные зазоры растут как N², а нормированные счёты
порядка единицы, поэтому граница связывает.

Система {theta.g_j > b_j} однородна по theta лишь частично (b_j
фиксированы), поэтому при НЕОГРАНИЧЕННОМ theta задача максимизации
запаса либо НЕОГРАНИЧЕНА (значит выполнима), либо имеет оптимум <= 0
(невыполнима). Это и есть чистый критерий, без произвольных констант.

Плюс прямой вопрос: разделяет ли САМО действие Бенинказы-Доукера?
S/k_4 = N - N_0 + 9N_1 - 16N_2 + 8N_3, то есть направление
(N0,N1,N2,N3) = (-1, +9, -16, +8). Свободен только общий множитель
(это 1/hbar), и он сканируется по знаку и величине.

  python3 actionlp2.py
"""
import json
import math

import numpy as np
from scipy.optimize import linprog

FEAT = ('R', 'N0', 'N1', 'N2', 'N3', 'C3', 'C4', 'H')
BD = {'R': 0.0, 'N0': -1.0, 'N1': 9.0, 'N2': -16.0, 'N3': 8.0,
      'C3': 0.0, 'C4': 0.0, 'H': 0.0}


def load(n):
    return json.load(open('/home/claude/data/sectors.json'))[str(n)]


def rows(secs, target, scale):
    ft, St = np.array([target['f'][k] / scale[k] for k in FEAT]), target['S']
    A, b, names = [], [], []
    for s in secs:
        if s is target:
            continue
        g = ft - np.array([s['f'][k] / scale[k] for k in FEAT])
        A.append(np.concatenate([-g, [1.0]]))
        b.append(St - s['S'])
        names.append(s['name'])
    return np.array(A), np.array(b), names


def feasible(secs, target, scale):
    A, b, names = rows(secs, target, scale)
    c = np.zeros(len(FEAT) + 1)
    c[-1] = -1.0
    r = linprog(c, A_ub=A, b_ub=b,
                bounds=[(None, None)] * (len(FEAT) + 1), method='highs')
    if r.status == 3:
        return 'ВЫПОЛНИМО (задача неограничена)', None, names, A, b
    if r.status == 0:
        m = r.x[-1]
        return ('ВЫПОЛНИМО' if m > 1e-9 else 'НЕВЫПОЛНИМО'), r.x, names, A, b
    return f'статус {r.status}', None, names, A, b


def bd_scan(secs, target, scale):
    """Разделяет ли направление БД при каком-нибудь множителе?"""
    A, b, names = rows(secs, target, scale)
    u = np.array([BD[k] / scale[k] for k in FEAT])
    best = (-1e300, None)
    for c in np.concatenate([-np.geomspace(1e-4, 1e9, 160),
                             np.geomspace(1e-4, 1e9, 160)]):
        th = c * u
        res = b - A @ np.concatenate([th, [0.0]])
        m = res.min()          # запас = min_j (theta.g_j - b_j)
        if m > best[0]:
            best = (m, c)
    return best, names, A, b


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    for N in (150, 300, 600):
        secs = load(N)
        scale = {k: max(abs(s['f'][k]) for s in secs) or 1.0 for k in FEAT}
        tgt = [s for s in secs if s['name'] == 'засев d=4'][0]
        ch = [s for s in secs if s['name'] == 'цепь'][0]
        print('=' * 72)
        print(f'N = {N}')
        v, _, _, _, _ = feasible(secs, ch, scale)
        print(f'  КОНТРОЛЬ 2 (цель = цепь): {v}')
        v, x, names, A, b = feasible(secs, tgt, scale)
        print(f'  ЦЕЛЬ = засев d=4: {v}')
        if x is not None:
            th = x[:-1]
            res = b - A @ x
            idx = np.argsort(res)[:3]
            print('     связывают: ' + ', '.join(
                f'{names[i]}' for i in idx))
        (mbd, cbd), names, A, b = bd_scan(secs, tgt, scale)
        print(f'  НАПРАВЛЕНИЕ БД: лучший запас {mbd:+.5g} при множителе '
              f'{cbd:+.4g}  -> '
              f'{"РАЗДЕЛЯЕТ" if mbd > 0 else "НЕ разделяет"}')
        res = b - A @ np.concatenate([cbd * np.array(
            [BD[k] / scale[k] for k in FEAT]), [0.0]])
        idx = np.argsort(res)[:3]
        print('     хуже всех: ' + ', '.join(
            f'{names[i]} ({res[i]:+.4g})' for i in idx))
        print()


if __name__ == '__main__':
    main()
