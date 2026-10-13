#!/usr/bin/env python3
"""Порог вместо единогласия: «раньше хотя бы на m шкалах из k».

m = k — наше правило, строгий частичный порядок.
m < k — правило голосования. Транзитивность обязана ломаться
(парадокс Кондорсе), вопрос в том, КАК и НАСКОЛЬКО.

Меряем по каждому m:
  доля пар          — сколько пар сравнимо
  двусторонних      — пар, где и i<j и j<i (антисимметрия сломана)
  наруш. транз.     — троек i<j<l без i<l
  3-циклов          — i<j<l<i, их в порядке быть не может
  циклов/случ.      — то же в единицах случайного турнира
  согласие с рангом — какую долю отношений объясняет один линейный ранг
  общих с m=k       — сколько отношений совпало с единогласным порядком

  python3 majority.py --k 5
  python3 majority.py --k 7 --n 500
"""
import argparse
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rel_threshold(X, m):
    """R[i,j] = True, если j раньше i хотя бы на m шкалах."""
    N, k = X.shape
    votes = np.zeros((N, N), dtype=np.int16)
    for s in range(k):
        votes += (X[None, :, s] < X[:, None, s])
    R = votes >= m
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def count_transitivity_violations(R):
    A = np.asarray(R, dtype=np.float32)
    reach2 = (A @ A) > 0
    return int((reach2 & ~R).sum())


def count_3cycles(R):
    """Число направленных треугольников i->j->l->i.

    ВНИМАНИЕ: ноль здесь НЕ означает ацикличность. При пороге m=k-1
    кратчайший цикл имеет длину ровно k, трёхзвенных нет вообще.
    Для ацикличности есть has_cycle().
    """
    A = np.asarray(R, dtype=np.float32)
    return int(round(float(np.trace(A @ A @ A)) / 3.0))


def has_cycle(R):
    """Топологическая сортировка (Кан): остался ли непустой остаток."""
    A = R.copy()
    deg = A.sum(1)
    left = len(A)
    while True:
        z = np.nonzero(deg == 0)[0]
        if len(z) == 0:
            return left > 0
        deg[z] = -1
        left -= len(z)
        A[:, z] = False
        deg = np.where(deg < 0, -1, A.sum(1))


def shortest_cycle(R, cap=14):
    """Длина кратчайшего цикла, или -1 если не нашли до cap."""
    A = np.asarray(R, dtype=np.float32)
    P = A.copy()
    for L in range(2, cap + 1):
        P = P @ A
        if float(np.trace(P)) > 0.5:
            return L
    return -1


def critical_size(k):
    """Размер мира, при котором при пороге m=k-1 ожидается первый цикл.

    Вывод: вокруг цикла каждая из k шкал даёт ровно один спуск,
    в биекции с k стрелками; цикл с одним спуском на шкале —
    отсортированная последовательность, повёрнутая, таких 1 из k!.
    Значит P(цикл на k-наборе) = (k-1)! * (k!)^(1-k), и
    E[циклов] = C(N,k) * P = 1 при N = k^(1/k) * (k!)^((k-1)/k).
    """
    from math import factorial as f
    return k ** (1.0 / k) * f(k) ** ((k - 1.0) / k)


def rank_agreement(R):
    """Строим один линейный ранг по числу предшественников
    и смотрим, какую долю отношений он объясняет."""
    score = R.sum(1)                       # сколько событий раньше i
    order = np.argsort(score)              # ранние впереди
    pos = np.empty(len(R), dtype=np.int64)
    pos[order] = np.arange(len(R))
    i, j = np.nonzero(R)                   # j раньше i
    if len(i) == 0:
        return float('nan')
    agree = (pos[j] < pos[i]).sum()
    return float(agree) / len(i)


def run(N, k, seed):
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    Rfull = rel_threshold(X, k)            # единогласие = наше правило
    pairs = N * (N - 1)

    rows = []
    for m in range(k, k // 2, -1):
        R = rel_threshold(X, m)
        both = int((R & R.T).sum())
        viol = count_transitivity_violations(R)
        cyc = count_3cycles(R)
        # случайный турнир той же плотности: C(N,3) * p^3 * 2 ориентации цикла
        p = R.sum() / pairs
        exp_cyc = comb(N, 3) * (p ** 3) * 2
        shared = int((R & Rfull).sum())
        rows.append(dict(
            m=m,
            frac=R.sum() / pairs,
            both=both,
            viol=viol,
            cyc=cyc,
            acyc=not has_cycle(R),
            short=shortest_cycle(R),
            ratio=cyc / exp_cyc if exp_cyc > 0 else float('nan'),
            agree=rank_agreement(R),
            shared=shared / max(Rfull.sum(), 1),
        ))
    return rows, int(Rfull.sum())


def main():
    ap = argparse.ArgumentParser(
        description='Правило большинства вместо единогласия')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    rows, nfull = run(a.n, a.k, a.seed)
    print(f'N={a.n}, k={a.k}, единогласных отношений: {nfull}')
    print(f'критический размер при m=k-1: N ~ {critical_size(a.k):.0f}')
    print()
    print(f"{'m':>3}{'доля пар':>11}{'ациклично':>11}{'крат.цикл':>11}"
          f"{'наруш.транз.':>14}{'3-циклов':>11}{'соглас.ранг':>13}{'общих с m=k':>13}")
    for r in rows:
        print(f"{r['m']:>3}{r['frac']:>11.4f}"
              f"{('да' if r['acyc'] else 'НЕТ'):>11}"
              f"{(r['short'] if r['short'] > 0 else '-'):>11}"
              f"{r['viol']:>14}{r['cyc']:>11}{r['agree']:>13.4f}"
              f"{r['shared']:>13.4f}")
    print()
    print('  m=k — порядок при любом N. m<k — порядок только пока мир мал.')


if __name__ == '__main__':
    main()
