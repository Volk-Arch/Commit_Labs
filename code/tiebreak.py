#!/usr/bin/env python3
"""Разрыв ничьих: артефакт ли «единственная неподвижная точка — цепь».

ПОДОЗРЕНИЕ. fixpoint.py, transloop.py и coneloop.py — три разные шкалы,
все три упали в тотальный порядок. Общая деталь: ничьи разрывались
добавкой eps*i/N — ОДИНАКОВОЙ во всех компонентах и монотонной по
индексу, а индекс отсортирован по времени. Тогда ничья решается в одну
сторону сразу во всех шкалах, покомпонентное доминирование выполняется
автоматически, и порядок насыщается.

А шкалы грубые: высота 0..9, W/2 тоже около десяти значений, значит
ничьих по всем компонентам ОЧЕНЬ много.

ПРАВИЛЬНО: разрывать НЕЗАВИСИМО в каждой шкале. Тогда ничья проходит
покомпонентное доминирование с вероятностью 2^-k, а не всегда.

Сравниваем три способа на одном и том же шаге петли.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from core import frac_pairs, mm_dim, height, spectral
from fixpoint import diamond, cone_order, box_order, rel_of_int
from coneloop import cone_scales, heights_of

N, T, K = 700, 2.0, 5
rs = np.random.default_rng(3901)
X = diamond(N, T, rs)
starts = {'конус': cone_order(X), 'ящик': box_order(N, rs)}

print(f'N={N}, опор {K}; шкала v = h - W/2')
print('ЭТАЛОН конуса: доля ~0.10, разм. 4.00')
print()
print(f"{'начало':>10}{'разрыв ничьих':>22}{'доля пар':>10}{'разм.':>8}"
      f"{'высота':>8}{'спектр':>10}{'ничьих по всем':>16}")

for nm, R0 in starts.items():
    h = heights_of(R0)
    cand = np.flatnonzero((h > np.percentile(h, 25))
                          & (h < np.percentile(h, 75)))
    srcs = np.random.default_rng(77).choice(cand, K, replace=False)
    V0 = cone_scales(R0, srcs)
    # доля пар, у которых НИЧЬЯ по всем компонентам
    eq = np.ones((N, N), bool)
    for c in range(V0.shape[1]):
        eq &= V0[:, c][:, None] == V0[:, c][None, :]
    np.fill_diagonal(eq, False)
    tie = eq.sum() / (N * (N - 1))
    for how in ('общий монотонный eps*i/N',
                'независимый случайный',
                'без разрыва вовсе'):
        if how.startswith('общий'):
            V = V0 + 1e-9 * np.arange(N)[:, None] / N
        elif how.startswith('независ'):
            V = V0 + 1e-9 * np.random.default_rng(5).random(V0.shape)
        else:
            V = V0
        R = rel_of_int(V)
        print(f'{nm:>10}{how:>22}{frac_pairs(R):>10.4f}'
              f'{mm_dim(frac_pairs(R)):>8.2f}{height(R)[0]:>8.0f}'
              f'{spectral(R):>+10.4f}{tie:>16.4f}')
    print()

print('  если «общий монотонный» даёт 1.0, а «независимый» — около 0.1,')
print('  то архивный вывод о тотальном порядке был артефактом разрыва')
