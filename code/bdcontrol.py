#!/usr/bin/env python3
"""НЕЗАВИСИМЫЙ КОНТРОЛЬ ДЕЙСТВИЯ слоистого сектора.

Вчерашние `sparsecancel.py` / `sparsescale.py` считают фазу по формуле

    ph(столбец) = -|T| + sum_i w(|s_i ∩ T|),    w(0) = -1

и контроль «факторизация против прямого перечисления» совпал до 1e-14.
Но перечисление считает по ТОЙ ЖЕ формуле, значит общую ошибку формулы
оно поймать не может. Контроль, который нельзя провалить, не контроль.

Здесь действие Бенинказы–Доукера считается ОТ САМОГО ПОРЯДКА:
строится матрица отношений, для каждой СВЯЗАННОЙ пары считается
|I(x,y)| = #{z: x<z<y}, собираются N_0..N_3 и

    S/k_4 = N - N_0 + 9 N_1 - 16 N_2 + 8 N_3.

Подозрение: в сумме по строкам `w(0) = -1` начисляется парам
(слой1, слой3) с нулевым пересечением, а они НЕ СВЯЗАНЫ и в сумму по
связанным парам не входят вовсе.

  python3 bdcontrol.py
"""
import itertools
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
W = {0: -1.0, 1: 9.0, 2: -16.0, 3: 8.0}


def wof(m):
    return W.get(int(m), 0.0)


def bd_from_poset(a1, a2, a3, rows, cols):
    """S/k_4 прямо из порядка: N_m по связанным парам."""
    n = a1 + a2 + a3
    # индексы: слой1 = 0..a1-1, слой2 = a1..a1+a2-1, слой3 = a1+a2..n-1
    R = np.zeros((n, n), dtype=bool)
    for i in range(a1):
        for j in range(a2):
            if (rows[i] >> j) & 1:
                R[i, a1 + j] = True
    for k in range(a3):
        for j in range(a2):
            if (cols[k] >> j) & 1:
                R[a1 + j, a1 + a2 + k] = True
    # транзитивное замыкание (три слоя: хватает одного шага)
    Rc = R | (R.astype(np.int8) @ R.astype(np.int8) > 0)
    # контроль: ещё шаг ничего не добавляет
    Rc2 = Rc | (Rc.astype(np.int8) @ Rc.astype(np.int8) > 0)
    assert (Rc2 == Rc).all(), 'замыкание не за один шаг'
    assert not Rc.diagonal().any(), 'цикл в порядке'
    Nm = {}
    for x in range(n):
        for y in range(n):
            if not Rc[x, y]:
                continue
            m = int((Rc[x] & Rc[:, y]).sum())
            Nm[m] = Nm.get(m, 0) + 1
    S = float(n) - Nm.get(0, 0) + 9 * Nm.get(1, 0) - 16 * Nm.get(2, 0) \
        + 8 * Nm.get(3, 0)
    return S, Nm


def bd_formula(a1, a2, a3, rows, cols, w0_in_rows):
    """S/k_4 по вчерашней формуле. w0_in_rows=True воспроизводит
    вчерашний код (wof(0) = -1 в сумме по строкам)."""
    n = a1 + a2 + a3
    p12 = sum(bin(s).count('1') for s in rows)
    p23 = sum(bin(c).count('1') for c in cols)
    ph = float(n) - p12 - p23
    for s in rows:
        for c in cols:
            m = bin(s & c).count('1')
            if m == 0 and not w0_in_rows:
                continue
            ph += wof(m)
    return ph


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(20261009)
    print('СРАВНЕНИЕ: действие от порядка против двух версий формулы')
    print(f"{'слои':>10}{'от порядка':>13}{'формула w0=-1':>16}"
          f"{'формула w0=0':>15}{'вердикт':>12}")
    bad_old = bad_new = 0
    tot = 0
    for (a1, a2, a3) in ((2, 2, 2), (2, 3, 2), (3, 2, 3), (3, 4, 2),
                         (2, 5, 3), (4, 3, 3)):
        for trial in range(40):
            rows = [int(rs.integers(0, 1 << a2)) for _ in range(a1)]
            cols = [int(rs.integers(0, 1 << a2)) for _ in range(a3)]
            S0, Nm = bd_from_poset(a1, a2, a3, rows, cols)
            S_old = bd_formula(a1, a2, a3, rows, cols, True)
            S_new = bd_formula(a1, a2, a3, rows, cols, False)
            tot += 1
            if abs(S_old - S0) > 1e-9:
                bad_old += 1
            if abs(S_new - S0) > 1e-9:
                bad_new += 1
            if trial == 0:
                v = ('w0=0 верна' if abs(S_new - S0) < 1e-9 else
                     'обе неверны')
                print(f'{f"({a1},{a2},{a3})":>10}{S0:>13.1f}{S_old:>16.1f}'
                      f'{S_new:>15.1f}{v:>12}')
    print()
    print(f'  из {tot} случайных конфигураций:')
    print(f'    формула с w(0) = -1 в строках расходится с порядком: '
          f'{bad_old}')
    print(f'    формула с w(0) =  0 в строках расходится с порядком: '
          f'{bad_new}')
    print()
    if bad_new == 0 and bad_old > 0:
        print('ВЫВОД: вчерашняя формула НЕВЕРНА. Лишний вес -1 начислялся')
        print('парам (слой1, слой3) без общего элемента второго слоя, то')
        print('есть НЕ СВЯЗАННЫМ парам. Все числа пункта 4 считаны с этой')
        print('ошибкой и подлежат пересчёту.')
    elif bad_old == 0 and bad_new == 0:
        print('ВЫВОД: обе формулы совпадают с порядком (вес -1 у')
        print('несвязанных пар даёт постоянный сдвиг). Числа в силе.')
    else:
        print('ВЫВОД: расхождение иного вида, разбираться.')


if __name__ == '__main__':
    main()
