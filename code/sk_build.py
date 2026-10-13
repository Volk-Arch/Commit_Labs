#!/usr/bin/env python3
"""ПОСТРОЕНИЕ S_k В 4d: размерность засева НЕ ограничена.

ordim.py: жадный поиск индуцированного S_k ограничен УСИЛИЕМ (при N=800
в 4d он давал 2/2/3 при малом переборе и 3/4/4 при большом), значит по
нему заключать нельзя. Контроль 1+1 при этом стоял на 2 при любом усилии.

Поэтому не поиск, а ПОСТРОЕНИЕ. Нужны пары, где a_i пространственноподобна
своему b_i и причинна всем остальным. Кладём
    a_i = (0, r u_i),   b_i = (T, -r u_i),   |u_i| = 1
Тогда
    i = j:  |dx| = 2r            нужно T < 2r        (несравнимы)
    i != j: |dx| = r|u_i + u_j|  нужно T > r|u_i+u_j| (a_i < b_j)
Окно существует, когда |u_i + u_j| < 2 при i != j, то есть когда
единичные векторы РАЗЛИЧНЫ. В трёх пространственных измерениях их
сколько угодно; в одном u = +-1, то есть ровно ДВА - вот откуда предел
S_2 в 1+1.

Если построение даёт S_k при любом k, порядковая размерность засева в 4d
НЕ ограничена, и формулировка ограничения в `tri-doc.md` неверна: надо
не «ограниченная порядковая размерность», а ВЛОЖИМОСТЬ. Оценка энтропии
при этом выживает: вложимое множество задано n точками, значит таких не
больше exp(O(d n ln n)).

КОНТРОЛЬ: та же проверка в 1+1 обязана ПРОВАЛИТЬСЯ при k >= 3.
"""
import numpy as np


def fib_sphere(k):
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi),
                     np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def check(k, U, label):
    r = 1.0
    S = np.array([[np.linalg.norm(U[i] + U[j]) for j in range(k)]
                  for i in range(k)])
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    lo, hi = r * off.max(), 2.0 * r
    if not lo < hi:
        print(f'  {label}, k={k}: окна НЕТ '
              f'(нужно {lo:.4f} < T < {hi:.4f})  => S_k не строится')
        return False
    T = 0.5 * (lo + hi)
    P = np.vstack([np.hstack([np.zeros((k, 1)), r * U]),
                   np.hstack([T * np.ones((k, 1)), -r * U])])
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    R = (dt > 0) & (dt ** 2 > dx2)
    ok_cross = all(R[i, k + j] for i in range(k) for j in range(k)
                   if i != j)
    ok_diag = all(not R[i, k + i] and not R[k + i, i] for i in range(k))
    ok_a = all(not R[i, j] for i in range(k) for j in range(k) if i != j)
    ok_b = all(not R[k + i, k + j] for i in range(k) for j in range(k)
               if i != j)
    good = ok_cross and ok_diag and ok_a and ok_b
    print(f'  {label}, k={k:>3}: T={T:.6f} в ({lo:.4f}, {hi:.4f})   '
          f'a_i<b_j: {ok_cross}, a_i||b_i: {ok_diag}, '
          f'a несравнимы: {ok_a}, b несравнимы: {ok_b}  '
          f'=> {"S_k ПОСТРОЕН" if good else "НЕ S_k"}')
    return good


def main():
    print(__doc__)
    print('4d (три пространственных измерения, векторы по Фибоначчи):')
    res = [check(k, fib_sphere(k), '4d') for k in (3, 5, 8, 12, 20)]
    print(f'  => построено при k = '
          f'{[k for k, g in zip((3, 5, 8, 12, 20), res) if g]}')

    print('\nКОНТРОЛЬ 1+1 (одно пространственное измерение, u = ±1):')
    for k in (2, 3, 4):
        U = np.array([[1.0] if i % 2 == 0 else [-1.0] for i in range(k)])
        check(k, U, '1+1')
    print('\nЧТЕНИЕ: если в 4d строится при любом k, порядковая размерность')
    print('засева НЕ ограничена, и ограничение надо формулировать через')
    print('ВЛОЖИМОСТЬ, а не через порядковую размерность.')


if __name__ == '__main__':
    main()
