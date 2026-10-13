#!/usr/bin/env python3
"""Обратная задача: сколько критериев использует выбирающий.

Прямая задача (вероятность циклов при мажоритарном правиле) исхожена
в теории общественного выбора. Обратная — измерить число критериев по
геометрии наблюдённых циклов — в литературе не нашлась.

Основание: длина кратчайшего цикла при пороге m из k критериев есть
L = ceil(k/(k-m)) (majority-doc.md, подтверждено 18 из 18). Если
наблюдать выборы человека и найти кратчайший цикл, получаем уравнение
на (k, m). Второе уравнение даёт ЧАСТОТА циклов. Двух может хватить,
чтобы определить оба числа.

Убийца всей затеи — шум: человек ошибается, и ошибки сами рождают
3-циклы, которые могут замаскировать структурные. Здесь это и
проверяется: до какого уровня шума подпись различима.

  python3 criteria.py --options 24
  python3 criteria.py --options 40 --trials 300
"""
import argparse
import math

import numpy as np


def choices(k, m, n_opt, noise, rs):
    """Матрица выборов: C[i,j] = True, если i предпочтён j.

    Критерии — независимые случайные полезности. Правило: i лучше j,
    если i выигрывает хотя бы на m критериях. Шум переворачивает
    каждое сравнение с вероятностью noise, независимо.
    """
    U = rs.random((n_opt, k))
    wins = np.zeros((n_opt, n_opt), dtype=np.int16)
    for s in range(k):
        wins += (U[:, None, s] > U[None, :, s])
    C = wins >= m
    np.fill_diagonal(C, False)
    if noise > 0:
        flip = rs.random((n_opt, n_opt)) < noise
        flip = np.triu(flip, 1)
        flip = flip | flip.T
        C = np.where(flip, C.T, C)
        np.fill_diagonal(C, False)
    return np.asarray(C, dtype=bool)


def cycle_counts(C, max_len=6):
    """Число замкнутых путей каждой длины (следы степеней матрицы)."""
    A = np.asarray(C, dtype=np.float64)
    out = {}
    P = A.copy()
    for L in range(2, max_len + 1):
        P = P @ A
        out[L] = float(np.trace(P)) / L
    return out


def shortest_cycle(C, max_len=6):
    for L, v in sorted(cycle_counts(C, max_len).items()):
        if v > 0.5:
            return L
    return -1


def run(k, m, n_opt, trials, noise, seed):
    rs = np.random.default_rng(seed)
    shortest, c3, c4, c5 = [], [], [], []
    for _ in range(trials):
        C = choices(k, m, n_opt, noise, rs)
        cc = cycle_counts(C)
        shortest.append(shortest_cycle(C))
        c3.append(cc[3])
        c4.append(cc[4])
        c5.append(cc[5])
    sh = np.array(shortest, float)
    good = sh[sh > 0]
    return dict(
        pred=math.ceil(k / (k - m)),
        shortest=float(np.median(good)) if len(good) else float('nan'),
        any_cycle=float((sh > 0).mean()),
        c3=float(np.mean(c3)), c4=float(np.mean(c4)), c5=float(np.mean(c5)),
    )


PAIRS = [(3, 2), (4, 3), (5, 3), (5, 4), (6, 4), (6, 5), (7, 4), (7, 6)]


def main():
    ap = argparse.ArgumentParser(
        description='Измерить число критериев по циклам выбора')
    ap.add_argument('--options', type=int, default=24,
                    help='сколько вариантов предъявлено')
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()

    for noise in (0.0, 0.02, 0.05, 0.10):
        print(f'ШУМ {noise:.0%}   (вариантов {a.options}, проб {a.trials})')
        print(f"{'k':>3}{'m':>3}{'предсказ. L':>13}{'медиана L':>11}"
              f"{'хоть один цикл':>17}{'3-циклов':>11}"
              f"{'4-циклов':>11}{'5-циклов':>11}")
        for k, m in PAIRS:
            r = run(k, m, a.options, a.trials, noise, a.seed)
            print(f'{k:>3}{m:>3}{r["pred"]:>13}{r["shortest"]:>11.1f}'
                  f'{r["any_cycle"]:>17.2f}{r["c3"]:>11.1f}'
                  f'{r["c4"]:>11.1f}{r["c5"]:>11.1f}')
        print()


if __name__ == '__main__':
    main()
