#!/usr/bin/env python3
"""МОЖЕТ ЛИ ВРАГ ВЫЙТИ В ПЛЮС по асимметрии степеней связей.

lkdeg.py нашёл различитель по ЗНАКУ: асимметрия распределения степеней
связей положительна у всего многообразного (+0.38, +0.80, +2.00 при
d = 2, 3, 4), у семейства роста (+0.69, +0.77) и у критической фазы
(+0.77), а у слоистого врага ОТРИЦАТЕЛЬНА (-0.44, -0.30). Величина от
нормировки не зависит.

Механизм: у засева длинный положительный хвост (элементы у светового
конуса многих других); у слоистого верхний слой даёт ком нулей ниже
сосредоточенной массы, отсюда левый хвост.

НО сегодня дважды выяснялось, что различитель при фиксированном N
бесполезен, если у врага есть свободный параметр: ширину он подогнал
через k (`census-doc.md`). Здесь у слоистых свободны k, q и профиль
слоёв. Сканируется вся эта свобода — в том числе состязательно, с
целью найти положительную асимметрию.

ЧТЕНИЕ: если во всём семействе асимметрия остаётся отрицательной —
различитель настоящий и устойчивый, причём по знаку, а не по величине.
Если враг выходит в плюс — ещё одна подогнанная величина, и надо
смотреть, чем он за это платит.

КОНТРОЛЬ: засев d=4 мерится тем же кодом в том же прогоне.
"""
import math

import numpy as np

from lkdeg import degstats
from sectors import sprinkle_d


def layered_prof(n, sizes, q, rs):
    off = np.cumsum([0] + list(sizes))
    Rc = np.zeros((n, n), bool)
    for i in range(len(sizes) - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sizes[i], sizes[i + 1])) < q)
    C = Rc.copy()
    for _ in range(len(sizes) + 1):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > .5)
        if (nx == C).all():
            break
        C = nx
    return C


def prof(n, k, kind, rs):
    if kind == 0:
        s = [n // k] * k
        s[-1] += n - sum(s)
    elif kind == 1:
        thin = max(1, n // (8 * k))
        s = [n // k] * k
        s[0] = s[-1] = thin
        s[k // 2] += n - sum(s)
    else:
        w = rs.random(k) ** 2 + 0.05
        s = np.maximum(1, np.rint(w / w.sum() * n).astype(int)).tolist()
        s[int(np.argmax(s))] += n - sum(s)
    return s if min(s) >= 1 and sum(s) == n else None


def main():
    print(__doc__)
    rs = np.random.default_rng(515151)
    n = 400
    ref = [degstats(sprinkle_d(n, 4, rs), n)['вперёд'][2]
           for _ in range(4)]
    print(f'КОНТРОЛЬ: засев d=4, асимметрия '
          f'{np.mean(ref):+.3f} ± {np.std(ref):.3f} (ожидали +2.0)')

    print('\nСЕТКА по слоистым: асимметрия степеней вперёд')
    print(f"{'k':>4}" + ''.join(f'{f"q={q}":>9}' for q in
                                (0.05, 0.15, 0.3, 0.5, 0.7, 0.9)))
    best = (-9.0, None)
    for k in (2, 3, 4, 6, 9, 14, 22, 40, 80):
        row = []
        for q in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9):
            vs = []
            for kind in (0, 1, 2):
                s = prof(n, k, kind, rs)
                if s is None:
                    continue
                sk = degstats(layered_prof(n, s, q, rs), n)['вперёд'][2]
                vs.append(sk)
                if sk > best[0]:
                    best = (sk, (k, q, kind))
            row.append(np.mean(vs) if vs else float('nan'))
        print(f'{k:>4}' + ''.join(f'{v:>+9.2f}' for v in row),
              flush=True)

    print(f'\n  наибольшая асимметрия по сетке: {best[0]:+.3f} '
          f'при (k, q, профиль) = {best[1]}')

    print('\nСОСТЯЗАТЕЛЬНО: случайный поиск положительной асимметрии')
    found = []
    for _ in range(260):
        k = int(rs.integers(2, 120))
        q = float(10 ** rs.uniform(-2.0, -0.003))
        kind = int(rs.integers(0, 3))
        s = prof(n, k, kind, rs)
        if s is None:
            continue
        sk = degstats(layered_prof(n, s, q, rs), n)['вперёд'][2]
        if sk > 0:
            found.append((sk, k, round(q, 4), kind))
    found.sort(reverse=True)
    print(f'  найдено положительных: {len(found)} из 260')
    for v in found[:5]:
        print(f'    асимметрия {v[0]:+.3f} при k={v[1]}, q={v[2]}, '
              f'профиль {v[3]}')
    if not found:
        print('  НИ ОДНОЙ — различитель устойчив по знаку')


if __name__ == '__main__':
    main()
