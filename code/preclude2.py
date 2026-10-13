#!/usr/bin/env python3
"""Слеп ли запрет: связано ли B mod n с геометрией порядка.

ОТКУДА. preclude.py установил арифметику:
  * фаза exp(i*B) — запрета НЕТ и быть не может (минимум |сумма| по
    подмножествам 0.273 при N=5 и 0.053 при N=6, к нулю подходит, не
    достигает: exp(i*s) при целых s линейно независимы над Q);
  * фаза exp(2*pi*i*B/n) — запреты есть при каждом n, машинный нуль
    10^-15, устройство арифметическое (полная смежная подгруппа
    простого порядка p | n).

РЕШАЮЩИЙ ВОПРОС. Погашение зависит ТОЛЬКО от B mod n. Значит
«какие геометрии можно запретить» равносильно «какие геометрии попадают
в нужные вычеты». Поэтому:

    если B mod n не связано с геометрией — запрет СЛЕП,
    и никакого отбора расстановок из него не выйдет.

ЧТО МЕРИМ.
  1. Связь B mod n с геометрией: отношение дисперсии доли пар МЕЖДУ
     классами вычетов к полной (это eta^2, доля объяснённой
     дисперсии). Слепота => eta^2 около нуля.
     Контроль: то же для случайной перестановки значений B.
  2. Сам B против геометрии (без модуля) — для сравнения: действие
     геометрию знает, вопрос лишь в том, выживает ли это знание
     при взятии остатка.
  3. Все минимальные запрещённые множества: их геометрия против
     случайных множеств того же размера.

  python3 preclude2.py
  python3 preclude2.py --n 7
"""
import argparse
import math
import time

import numpy as np

from preclude import grow, bd_action, frac_pairs


def eta2(groups):
    """Доля дисперсии, объяснённая разбиением на группы."""
    allv = np.concatenate(groups)
    if len(allv) < 2 or allv.var() <= 0:
        return float('nan')
    between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return float(between / (len(allv) * allv.var()))


def main():
    ap = argparse.ArgumentParser(description='Слеп ли запрет')
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--mods', type=int, nargs='+',
                    default=[2, 3, 4, 5, 6, 8, 12, 24])
    a = ap.parse_args()

    t0 = time.time()
    rs = np.random.default_rng(1901)

    for N in range(5, a.n + 1):
        hist = grow(N)
        B = np.array([bd_action(p, N) for p, _ in hist])
        F = np.array([frac_pairs(p, N) for p, _ in hist])
        # вторая геометрическая величина: высота (длина макс. цепи)
        H = []
        for p, _ in hist:
            h = np.zeros(N, int)
            for j in range(N):
                m, best = p[j], 0
                while m:
                    i = (m & -m).bit_length() - 1
                    m &= m - 1
                    best = max(best, h[i] + 1)
                h[j] = best
            H.append(h.max())
        H = np.array(H, float)

        print(f'\nN = {N}:  {len(hist)} порядков, '
              f'B от {B.min()} до {B.max()}, '
              f'доля пар {F.mean():.4f} ± {F.std():.4f}')

        # 2. знает ли САМО действие геометрию
        print(f'   связь B с геометрией (без модуля):  '
              f'доля пар r = {np.corrcoef(B, F)[0,1]:+.4f},  '
              f'высота r = {np.corrcoef(B, H)[0,1]:+.4f}')

        # 1. выживает ли это знание при взятии остатка
        print(f"{'n':>6}{'eta2 доля пар':>16}{'контроль':>11}"
              f"{'eta2 высота':>14}{'контроль':>11}{'классов':>9}")
        for n in a.mods:
            r = B % n
            gs = [F[r == v] for v in np.unique(r) if (r == v).sum() > 1]
            hs = [H[r == v] for v in np.unique(r) if (r == v).sum() > 1]
            if len(gs) < 2:
                continue
            Bp = rs.permutation(B) % n
            gs0 = [F[Bp == v] for v in np.unique(Bp) if (Bp == v).sum() > 1]
            hs0 = [H[Bp == v] for v in np.unique(Bp) if (Bp == v).sum() > 1]
            print(f'{n:>6}{eta2(gs):>16.4f}{eta2(gs0):>11.4f}'
                  f'{eta2(hs):>14.4f}{eta2(hs0):>11.4f}{len(gs):>9}')

        # 3. запрещённые множества БЕЗ ВЫБОРА ПРЕДСТАВИТЕЛЯ.
        #
        # ПОЙМАННЫЙ АРТЕФАКТ. Первая версия брала представителя класса
        # первым по порядку перечисления и давала до 19 сигм. Первыми
        # попадаются РЕДКИЕ значения действия, а они отвечают крайним
        # геометриям (цепь, антицепь) — значимость была артефактом
        # выбора.
        #
        # ЧЕСТНО: минимальное запрещённое множество есть выбор по одной
        # истории из каждого из p вычетов, и все выборы равноправны.
        # Значит геометрия «запрещённого» = среднее по ОБЪЕДИНЕНИЮ этих
        # вычетов, с весом по числу порядков. Выбора не остаётся.
        print('   запрещённые вычеты против всего ансамбля '
              '(без выбора представителя):')
        print(f"{'n':>6}{'косетов':>9}{'размер':>8}{'доля пар':>11}"
              f"{'всего':>9}{'сигм':>7}{'контроль, сигм':>16}")
        for n in a.mods:
            r = B % n
            have = set(np.unique(r).tolist())
            cos = []
            for p in (2, 3, 5, 7):
                if n % p or p > n:
                    continue
                step = n // p
                for q in range(step):
                    cls = [(q + j * step) % n for j in range(p)]
                    if all(c in have for c in cls):
                        cos.append(cls)
                break
            if not cos:
                print(f'{n:>6}{"нет":>9}')
                continue
            sz = len(cos[0])
            mus, zs = [], []
            rp = rs.permutation(B) % n
            for cls in cos:
                m = np.isin(r, cls)
                if m.sum() < 2:
                    continue
                mus.append(F[m].mean())
                # сигмы этого косета против всего ансамбля
                se = F.std() / math.sqrt(m.sum())
                zs.append((F[m].mean() - F.mean()) / max(se, 1e-12))
                m0 = np.isin(rp, cls)
                if m0.sum() >= 2:
                    se0 = F.std() / math.sqrt(m0.sum())
                    zs.append(None)
            # контроль: те же косеты на перемешанном B
            z0 = []
            for cls in cos:
                m0 = np.isin(rp, cls)
                if m0.sum() < 2:
                    continue
                se0 = F.std() / math.sqrt(m0.sum())
                z0.append(abs((F[m0].mean() - F.mean()) / max(se0, 1e-12)))
            zz = [abs(x) for x in zs if x is not None]
            print(f'{n:>6}{len(cos):>9}{sz:>8}{np.mean(mus):>11.4f}'
                  f'{F.mean():>9.4f}{np.mean(zz):>7.1f}'
                  f'{np.mean(z0) if z0 else float("nan"):>16.1f}')

    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
