#!/usr/bin/env python3
"""МЕРА ОТ РОСТА против РАВНОВЕРОЯТНОЙ: проблема энтропии в миниатюре.

ЦЕНТРАЛЬНАЯ БЕДА ОБЛАСТИ. Клейтман и Ротшильд: при равновероятном
подсчёте почти все частично упорядоченные множества на N элементах —
ТРЁХСЛОЙНЫЕ (нижний слой ~N/4, средний ~N/2, верхний ~N/4), а они не
многообразноподобны. Это «проблема энтропии»: случайный порядок — не
пространство-время.

Архив отметил (ORDER-final.md:2951), что проблема есть артефакт
РАВНОВЕРОЯТНОГО подсчёта и при мере от РОСТА её нет. Не проверялось.

ЗДЕСЬ ПРОВЕРЯЕТСЯ. Перечисляем все истории роста до N событий. История
— последовательность рождений, новое событие максимально, его прошлое —
любое вниз-замкнутое подмножество, выбор равновероятен среди них. Это
мера от роста, и она БЕЗ ПАРАМЕТРА (в отличие от CSG Райдаута-Соркина,
где есть связи t_k).

Две меры на одних и тех же порядках:
    равновероятная  каждый помеченный порядок весит одинаково
    от роста        вес = вероятность породить его ростом

И смотрим, куда каждая мера кладёт вес: на трёхслойные (Клейтман) или
на цепеподобные, и что делается с высотой и долей пар.
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/code')
from preclude import downsets


def grow_measure(N):
    """Все помеченные порядки до N с ВЕСОМ от роста и числом историй.

    Вес от роста: на каждом шаге выбор равновероятен среди вниз-замкнутых
    подмножеств, значит вероятность истории = произведение 1/D_k.
    """
    cur = {(0,): 1.0}
    cnt = {(0,): 1}
    for n in range(1, N):
        nxt, ncnt = defaultdict(float), defaultdict(int)
        for past, p in cur.items():
            ds = downsets(past, n)
            w = 1.0 / len(ds)
            for s in ds:
                key = past + (s,)
                nxt[key] += p * w
                ncnt[key] += cnt[past]
        cur, cnt = dict(nxt), dict(ncnt)
    return cur, cnt


def layers(past, n):
    """Слои по высоте: h[i] = длина макс. цепи до i."""
    h = np.zeros(n, int)
    for j in range(n):
        m, best = past[j], 0
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            best = max(best, h[i] + 1)
        h[j] = best
    return h


def stats(past, n):
    h = layers(past, n)
    rel = sum(bin(p).count('1') for p in past)
    f = rel / (n * (n - 1) / 2)
    hh = int(h.max()) + 1                      # число слоёв
    sizes = np.bincount(h, minlength=hh)
    # признак Клейтмана: ровно 3 слоя с профилем ~N/4, N/2, N/4
    kr = (hh == 3)
    return f, hh, kr, sizes


print('N: порядков | мера | <доля пар> | <слоёв> | доля 3-слойных | <высота>')
for N in (5, 6, 7):
    cur, cnt = grow_measure(N)
    tot_g = sum(cur.values())
    tot_u = len(cur)
    accg = np.zeros(4); accu = np.zeros(4)
    for past, p in cur.items():
        f, hh, kr, _ = stats(past, N)
        v = np.array([f, hh, 1.0 if kr else 0.0, hh - 1])
        accg += v * (p / tot_g)
        accu += v / tot_u
    print(f'{N}: {tot_u} помеченных порядков')
    print(f'    равновероятная  {accu[0]:.4f}   {accu[1]:.3f}   '
          f'{accu[2]:.4f}   {accu[3]:.3f}')
    print(f'    ОТ РОСТА        {accg[0]:.4f}   {accg[1]:.3f}   '
          f'{accg[2]:.4f}   {accg[3]:.3f}')

print()
print('самые вероятные порядки ПОД КАЖДОЙ мерой (N=7)')
cur, cnt = grow_measure(7)
tot_g = sum(cur.values())
byg = sorted(cur.items(), key=lambda kv: -kv[1])[:5]
print(f"{'мера от роста':>22}{'вес':>10}{'слоёв':>8}{'доля пар':>10}")
for past, p in byg:
    f, hh, kr, sz = stats(past, 7)
    print(f'{str(list(sz)):>22}{p/tot_g:>10.4f}{hh:>8}{f:>10.4f}')
print(f'\n  всего веса от роста у 5 самых вероятных: '
      f'{sum(p for _, p in byg)/tot_g:.4f}')
