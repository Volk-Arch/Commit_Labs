#!/usr/bin/env python3
"""РОЖДЕНИЕ ИЗ ДВУХ: порядок не вычисляется, а растёт.

Настоящий слом, а не фильтр. В `break-doc.md` я крутил ВСЕПАРНОЕ правило
с отсечкой по окну - но это фильтр: пары всё равно считались все. Здесь
пары НЕ СЧИТАЮТСЯ ВООБЩЕ: каждое новое событие выбирает РОВНО ДВУХ
родителей, остальное доделывает транзитивность.

Тогда теорема `dim <= k` (`orthant-doc.md`) отваливается по сути:
порядок больше не задан шкалами, его вообще никто не задаёт целиком.

Почему ровно два. Один родитель даёт ДЕРЕВО, а у дерева два несравнимых
события никогда не имеют общего будущего, тогда как в
пространстве-времени любые два его имеют - дерево умирает сразу. Два -
минимум, при котором есть и ветвление, и схождение. В 1+1 это ещё и
буквально световая структура: событие есть пересечение двух нулевых
лучей.

ГЛАВНЫЙ ПРИЗНАК - рост высоты: у засева в d измерениях H ~ N^{1/d}
(то есть N^{0.25} при d=4 и N^{0.5} при 1+1), у случайного прикрепления
H ~ log N. Различает сразу.

Правила выбора родителей (носитель, а не порог):
  'все'       - равномерно среди всех прошлых;
  'граница'   - равномерно среди МАКСИМАЛЬНЫХ (текущая поверхность);
  'гранИнк'   - среди максимальных, но пара обязана быть НЕСРАВНИМОЙ;
  'окно'      - среди последних w добавленных.

КОНТРОЛИ: засев d=4 и 1+1 прогоняются теми же мерами; показатель высоты
у них обязан выйти около 0.25 и 0.50.
"""
import math

import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from orthant import prof, region, rf
from ordim import best_sk


def grow(n, rule, rs, w=40):
    """P[i] - маска прошлого элемента i (строго)."""
    P = np.zeros((n, n), bool)
    for i in range(1, n):
        if i == 1:
            par = [0]
        else:
            if rule == 'все':
                par = list(rs.choice(i, size=2, replace=False))
            elif rule == 'окно':
                lo = max(0, i - w)
                cand = np.arange(lo, i)
                par = list(rs.choice(cand, size=min(2, len(cand)),
                                     replace=False))
            else:
                anc = P[:i, :i]
                ismax = ~anc.any(axis=1)
                mx = np.nonzero(ismax)[0]
                if len(mx) < 2:
                    par = list(rs.choice(i, size=2, replace=False))
                elif rule == 'граница':
                    par = list(rs.choice(mx, size=2, replace=False))
                else:                      # гранИнк
                    par = None
                    for _ in range(30):
                        a, b = rs.choice(mx, size=2, replace=False)
                        if not P[a, b] and not P[b, a]:
                            par = [a, b]
                            break
                    if par is None:
                        par = list(rs.choice(mx, size=2, replace=False))
        m = np.zeros(n, bool)
        for p in par:
            m[p] = True
            m |= P[p]
        P[i] = m
    return P[:, :].T.copy() & np.triu(np.ones((n, n), bool), 1) | \
        (P.T & ~np.eye(n, dtype=bool))


def order_of(n, rule, rs, w=40):
    P = np.zeros((n, n), bool)
    R = np.zeros((n, n), bool)
    for i in range(1, n):
        if i == 1:
            par = [0]
        else:
            if rule == 'все':
                par = list(rs.choice(i, size=2, replace=False))
            elif rule == 'окно':
                lo = max(0, i - w)
                par = list(rs.choice(np.arange(lo, i),
                                     size=min(2, i - lo), replace=False))
            else:
                ismax = ~R[:i, :i].any(axis=1)
                mx = np.nonzero(ismax)[0]
                if len(mx) < 2:
                    par = list(rs.choice(i, size=2, replace=False))
                elif rule == 'граница':
                    par = list(rs.choice(mx, size=2, replace=False))
                else:
                    par = None
                    for _ in range(30):
                        a, b = rs.choice(mx, size=2, replace=False)
                        if not R[a, b] and not R[b, a]:
                            par = [a, b]
                            break
                    if par is None:
                        par = list(rs.choice(mx, size=2, replace=False))
        anc = np.zeros(n, bool)
        for p in par:
            anc[p] = True
            anc |= R[:, p]
        R[:, i] = anc
    return R


def height(R):
    n = R.shape[0]
    h = np.zeros(n, int)
    for x in range(n):
        pre = np.nonzero(R[:, x])[0]
        h[x] = (h[pre].max() + 1) if len(pre) else 1
    return int(h.max())


def main():
    print(__doc__)
    rs = np.random.default_rng(778899)
    NS = (100, 200, 400, 800)

    print('КОНТРОЛЬ: показатель высоты у засевов')
    for lab, mk in (('засев 4d', lambda n: sprinkle_d(n, 4, rs)),
                    ('засев 1+1', lambda n: causal(sprinkle_11(n, rs)))):
        hs = [np.mean([height(mk(n)) for _ in range(3)]) for n in NS]
        p = np.polyfit(np.log(NS), np.log(hs), 1)[0]
        print(f'  {lab:>10}: H = ' + ' '.join(f'{v:.1f}' for v in hs)
              + f'   показатель {p:.3f}  (ждём '
              + ('0.25' if '4d' in lab else '0.50') + ')', flush=True)

    print('\nРОЖДЕНИЕ ИЗ ДВУХ: показатель высоты')
    for rule in ('все', 'окно', 'граница', 'гранИнк'):
        hs = [np.mean([height(order_of(n, rule, rs)) for _ in range(3)])
              for n in NS]
        p = np.polyfit(np.log(NS), np.log(hs), 1)[0]
        print(f'  {rule:>10}: H = ' + ' '.join(f'{v:.1f}' for v in hs)
              + f'   показатель {p:.3f}', flush=True)

    n = 400
    chi4, thr4 = region(lambda: sprinkle_d(n, 4, rs), 40, rs)
    chi2, thr2 = region(lambda: causal(sprinkle_11(n, rs)), 40, rs)
    print(f'\nпрофиль при N={n} (пороги: 4d {thr4:.1f}, 1+1 {thr2:.1f})')
    print(f"{'правило':>10}{'доля':>9}{'H':>5}{'хи2 4d':>10}"
          f"{'хи2 1+1':>10}{'S_k':>5}")
    for rule in ('все', 'окно', 'граница', 'гранИнк'):
        R = order_of(n, rule, rs)
        print(f'{rule:>10}{rf(R):>9.4f}{height(R):>5}'
              f'{float(chi4(prof(R))):>10.1f}{float(chi2(prof(R))):>10.1f}'
              f'{best_sk(R, rs):>5}', flush=True)


if __name__ == '__main__':
    main()
