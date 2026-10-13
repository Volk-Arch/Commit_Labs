#!/usr/bin/env python3
"""РАСПРЕДЕЛЕНИЕ СТЕПЕНЕЙ СВЯЗЕЙ: форма веера, а не сумма.

Все три механизма промахиваются по ЧИСЛУ СВЯЗЕЙ, причём в разные
стороны: критическая фаза даёт на 33% больше засева, семейство роста на
78% меньше. Пространство-время стоит между ними.

Но «связей на элемент границы» уже проверено сегодня и НЕ различает
(192 у засева против 213 у слоистого врага): пересекающих связей у обоих
порядка W^2 по показателю.

Непроверенное — РАСПРЕДЕЛЕНИЕ. У засева связи живут у светового конуса:
интервал мал, а расстояние любое, отсюда L ~ N^{2-2/d} и длинные связи.
Значит у каждого элемента есть «веер» связей размера ~N^{1-2/d} с
определённой ФОРМОЙ распределения. Сумму механизмы подгоняют; форму —
вопрос открытый.

Мерится, для связей вперёд (к будущему) и назад (к прошлому):
  * средняя степень и её отношение к N^{1-2/d};
  * ОТНОСИТЕЛЬНЫЙ разброс (разброс / среднее) — форма, не масштаб;
  * асимметрия;
  * доля элементов с нулевой степенью вперёд.
Относительный разброс и асимметрия от нормировки не зависят, значит
сравнимы между ансамблями с разными суммами — именно то, что нужно.

ПРЕДРЕГИСТРАЦИЯ: если конусная структура и есть то, чего механизмам не
хватает, относительный разброс у засевов должен быть заметно иным, чем
у семейства роста и у критической фазы, ПРИ ТОМ ЧТО сумма подогнана.

КОНТРОЛЬ: все ансамбли мерятся одним кодом в одном прогоне; у засевов
d = 2, 3, 4 величина обязана зависеть от d правильным образом
(степень ~ N^{1-2/d} растёт с d).
"""
import math

import numpy as np

from csg2 import grow_rc
from d3 import tune_r
from sectors import sprinkle_d
from crit9 import Ch


def linkmat(Rc):
    F = Rc.astype(np.float32)
    return Rc & ~((F @ F) > 0.5)


def degstats(Rc, n):
    Lk = linkmat(Rc)
    fwd = Lk.sum(axis=1).astype(float)
    bwd = Lk.sum(axis=0).astype(float)
    R = float(Rc.sum())
    out = {}
    for nm, d in (('вперёд', fwd), ('назад', bwd)):
        m = d.mean()
        s = d.std()
        sk = float(((d - m) ** 3).mean() / max(s ** 3, 1e-12))
        out[nm] = (m, s / max(m, 1e-9), sk, float((d == 0).mean()))
    out['L'] = float(Lk.sum())
    out['R'] = R
    return out


def layered(n, k, q, rs):
    a = n // k
    sz = [a] * k
    sz[-1] += n - a * k
    off = np.cumsum([0] + sz)
    Rc = np.zeros((n, n), bool)
    for i in range(k - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sz[i], sz[i + 1])) < q)
    C = Rc.copy()
    for _ in range(k):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > .5)
        if (nx == C).all():
            break
        C = nx
    return C


def crit_sample(n, rs, beta=0.04, nsweep=500):
    BET = [0.0, 0.02, 0.03, 0.04, 0.06, 0.10, 0.18]
    ch = [Ch(n, b) for b in BET]
    kt = BET.index(beta)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    for it in range(nsweep):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
    return ch[kt].R.copy()


def main():
    print(__doc__)
    rs = np.random.default_rng(424242)
    n = 400
    print(f'N = {n}; относительный разброс и асимметрия от нормировки '
          f'НЕ зависят')
    print(f"\n{'ансамбль':>22}{'ст.вперёд':>11}{'отн.разбр.':>12}"
          f"{'асимм.':>9}{'нулевых':>9}{'L/R':>8}")
    rows = []
    for d in (2, 3, 4):
        vs = [degstats(sprinkle_d(n, d, rs), n) for _ in range(4)]
        g = {k: np.mean([v[k] if not isinstance(v[k], tuple) else 0
                         for v in vs]) for k in ('L', 'R')}
        f = np.array([v['вперёд'] for v in vs]).mean(axis=0)
        rows.append((f'засев d={d}', f, g['L'] / g['R']))
        print(f'{f"засев d={d}":>22}{f[0]:>11.2f}{f[1]:>12.3f}'
              f'{f[2]:>9.2f}{f[3]:>9.3f}{g["L"]/g["R"]:>8.3f}',
              flush=True)
    # семейство роста, x подогнано под долю связанных засева d=3
    ref3 = np.mean([float(sprinkle_d(n, 3, rs).sum())
                    / (n * (n - 1) / 2) for _ in range(6)])
    for a in (2, 3):
        lx = tune_r(n, a, ref3, rs)
        vs = [degstats(grow_rc(n, a, lx, rs), n) for _ in range(4)]
        f = np.array([v['вперёд'] for v in vs]).mean(axis=0)
        L = np.mean([v['L'] for v in vs]); R = np.mean([v['R'] for v in vs])
        rows.append((f'рост a={a}', f, L / R))
        print(f'{f"рост a={a}":>22}{f[0]:>11.2f}{f[1]:>12.3f}'
              f'{f[2]:>9.2f}{f[3]:>9.3f}{L/R:>8.3f}', flush=True)
    for k, q in ((3, 0.2), (5, 0.1)):
        vs = [degstats(layered(n, k, q, rs), n) for _ in range(3)]
        f = np.array([v['вперёд'] for v in vs]).mean(axis=0)
        L = np.mean([v['L'] for v in vs]); R = np.mean([v['R'] for v in vs])
        print(f'{f"слоистый k={k}":>22}{f[0]:>11.2f}{f[1]:>12.3f}'
              f'{f[2]:>9.2f}{f[3]:>9.3f}{L/R:>8.3f}', flush=True)
    Rc = crit_sample(26, rs)
    v = degstats(Rc, 26)
    f = v['вперёд']
    print(f'{"крит. фаза (N=26)":>22}{f[0]:>11.2f}{f[1]:>12.3f}'
          f'{f[2]:>9.2f}{f[3]:>9.3f}{v["L"]/v["R"]:>8.3f}')

    print('\nКОНТРОЛЬ: степень вперёд у засевов обязана расти с d')
    print('  ' + ', '.join(f'd={2+i}: {rows[i][1][0]:.2f}'
                           for i in range(3)))
    print(f'  N^(1-2/d): ' + ', '.join(
        f'd={d}: {n**(1-2/d):.2f}' for d in (2, 3, 4)))


if __name__ == '__main__':
    main()
