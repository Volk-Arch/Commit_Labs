#!/usr/bin/env python3
"""ОСНОВАНИЕ ПРОЕКТА: порядок на k шкалах — это Минковский или нет.

Базовое правило проекта: «A раньше B, если раньше на ВСЕХ k шкалах», и
k=4 выбрано по доле связанных: у порядка на k шкалах она 2^{1-k} = 0.125
при k=4, у Минковского в 4d - 0.1 (Майрхейм-Майер). Близко.

Но конус у ОРТЫ и конус у МИНКОВСКОГО - разные тела: орта задана k
линейными неравенствами, световой конус - одним квадратичным. Доля
связанных их не различает, а сегодняшний ПРОФИЛЬ ОБИЛИЙ различает то,
чего одна статистика не видит (`prostoe-doc.md`: ни подгонкой, ни
смешиванием).

Это проверка основания, а не очередная гипотеза, поэтому спрашиваем
прямо.

Попутно - нитка из `majority-doc.md`: при m = k-1 мир остаётся ацикличным
до N_крит = k^{1/k} (k!)^{(k-1)/k}, то есть при k=8 до 13890, при k=7 до
1969. Значит при N=600 ОСЛАБЛЕННЫЙ порядок (m=k-1) - законное причинное
множество, и он в 8 раз плотнее единогласного. Вопрос пользователя:
не ближе ли он к геометрии, чем строгий?

КОНТРОЛИ:
  (a) засев 4d обязан лежать в своей области (порог считается на 60
      образцах, а не на 25 - прошлый раз он вылез в 1.8 раза на одной
      выборке, и порог был шумный);
  (b) ацикличность ослабленных порядков проверяется топологической
      сортировкой, а не числом 3-циклов (эта ошибка уже была сделана
      в `majority-doc.md`).
"""
import math

import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from ratios import nm


def scales(n, k, rs):
    return np.array([rs.permutation(n) for _ in range(k)])


def thresh_order(S, m):
    """A < B, если A раньше хотя бы на m шкалах из k (и не позже нигде
    при m=k). Возвращает матрицу отношений."""
    k, n = S.shape
    cnt = np.zeros((n, n), np.int16)
    for s in S:
        pos = np.empty(n, int)
        pos[s] = np.arange(n)
        cnt += (pos[:, None] < pos[None, :]).astype(np.int16)
    return cnt >= m


def acyclic(R):
    n = R.shape[0]
    deg = R.sum(axis=0).astype(int)
    alive = np.ones(n, bool)
    left = n
    while left:
        z = np.nonzero(alive & (deg == 0))[0]
        if len(z) == 0:
            return False
        alive[z] = False
        deg -= R[z].sum(axis=0).astype(int)
        deg[z] = -1
        left -= len(z)
    return True


def closure(R):
    C = R.copy()
    for _ in range(40):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > 0.5)
        if (nx == C).all():
            break
        C = nx
    return C


def prof(R):
    v, _ = nm(R)
    return v[1:] / max(v[0], 1.0)


def region(mk, rep, rs):
    A = np.array([prof(mk()) for _ in range(rep)])
    mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1)
    return (lambda X: (((X - mu) / sd) ** 2).sum(axis=-1),
            float(np.percentile((((A - mu) / sd) ** 2).sum(axis=1), 95)))


def rf(R):
    n = R.shape[0]
    return float(R.sum()) / (n * (n - 1) / 2)


def main():
    print(__doc__)
    rs = np.random.default_rng(265358)
    n = 600
    chi4, thr4 = region(lambda: sprinkle_d(n, 4, rs), 60, rs)
    chi2, thr2 = region(lambda: causal(sprinkle_11(n, rs)), 60, rs)
    print(f'пороги при N={n}: 4d {thr4:.1f}, 1+1 {thr2:.1f}  '
          f'(по 60 образцам)')
    cs = [float(chi4(prof(sprinkle_d(n, 4, rs)))) for _ in range(5)]
    print(f'КОНТРОЛЬ (a): пять засевов 4d в своей области -> '
          + ' '.join(f'{v:.1f}' for v in cs)
          + f'  => {"ОК" if max(cs) < thr4 * 1.5 else "порог шумит"}')

    print(f'\nЕДИНОГЛАСИЕ (m=k): орта против Минковского')
    print(f"{'k':>3}{'доля связ.':>12}{'ацикл.':>8}"
          f"{'хи2 к 4d':>11}{'хи2 к 1+1':>11}")
    for k in (2, 3, 4, 5, 6):
        R = thresh_order(scales(n, k, rs), k)
        p = prof(R)
        print(f'{k:>3}{rf(R):>12.4f}{str(acyclic(R)):>8}'
              f'{float(chi4(p)):>11.1f}{float(chi2(p)):>11.1f}', flush=True)

    print(f'\nОСЛАБЛЕНИЕ (m=k−1), законное при N < N_крит:')
    print(f"{'k':>3}{'N_крит':>9}{'доля связ.':>12}{'ацикл.':>8}"
          f"{'хи2 к 4d':>11}{'хи2 к 1+1':>11}")
    for k in (7, 8, 9):
        nc = k ** (1.0 / k) * math.factorial(k) ** ((k - 1) / k)
        R = thresh_order(scales(n, k, rs), k - 1)
        ok = acyclic(R)
        R2 = closure(R) if ok else R
        p = prof(R2)
        print(f'{k:>3}{nc:>9.0f}{rf(R2):>12.4f}{str(ok):>8}'
              f'{float(chi4(p)):>11.1f}{float(chi2(p)):>11.1f}', flush=True)


if __name__ == '__main__':
    main()
