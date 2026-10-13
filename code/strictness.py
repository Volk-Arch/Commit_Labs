#!/usr/bin/env python3
"""Измеритель строгости правила выбора по наблюдённым парным выборам.

Что меряется: сколько критериев человека должны согласиться, прежде чем
он меняет решение. Существующие методы (сопряжённый анализ) оценивают
ВЕСА критериев, а не ПОРОГ агрегации.

Основание: при пороге «лучше хотя бы на m критериях из k» длина
кратчайшего цикла предпочтений равна ceil(k/(k-m)) (majority-doc.md).
Длина как измеритель негодна — шум рождает трёхзвенные циклы и обнуляет
минимум уже при 2% ошибок (criteria.py). А ЧАСТОТА циклов выживает.

Но частота — одно число, а параметров два. Второе уравнение даёт ДОЛЯ
БЕЗРАЗЛИЧИЯ: если разрешить ответ «не могу выбрать», то при пороге m
пара остаётся нерешённой, когда ни одна сторона не набрала m голосов,
то есть с вероятностью

    1 - 2 * P(Bin(k, 1/2) >= m)

Число критериев, выигранных одним вариантом у другого, при независимых
случайных полезностях распределено как Bin(k, 1/2) — отсюда формула.

Схема опыта:
  - N вариантов, все пары;
  - разрешён ответ «не могу выбрать»;
  - часть пар повторяется, чтобы померить уровень ошибок отдельно.

  python3 strictness.py --recover
  python3 strictness.py --options 30 --recover
"""
import argparse
import math
import time
from itertools import combinations

import numpy as np

# (k, m): пороговые правила, где m строго больше половины
GRID = [(k, m) for k in range(3, 9)
        for m in range(k // 2 + 1, k)]

IND = 0   # не могу выбрать
WIN = 1   # строка лучше столбца
LOS = 2   # столбец лучше строки


def responses(k, m, n_opt, noise, rs):
    """Матрица ответов с тремя исходами плюс шум."""
    U = rs.random((n_opt, k))
    wins = np.zeros((n_opt, n_opt), dtype=np.int16)
    for s in range(k):
        wins += (U[:, None, s] > U[None, :, s])
    R = np.full((n_opt, n_opt), IND, dtype=np.int8)
    R[wins >= m] = WIN
    R[(k - wins) >= m] = LOS
    np.fill_diagonal(R, IND)

    if noise > 0:
        iu = np.triu_indices(n_opt, 1)
        hit = rs.random(len(iu[0])) < noise
        pick = np.where(rs.random(len(iu[0])) < 0.5, WIN, LOS)
        vals = R[iu].copy()
        vals[hit] = pick[hit]
        R[iu] = vals
        R[(iu[1], iu[0])] = np.where(vals == WIN, LOS,
                                     np.where(vals == LOS, WIN, IND))
    return R


def features(R):
    """Два наблюдаемых числа: доля безразличия и частота 3-циклов."""
    n = len(R)
    iu = np.triu_indices(n, 1)
    ind = float((R[iu] == IND).mean())
    beats = np.asarray(R == WIN, dtype=np.float64)   # beats[i,j]: i лучше j
    tri = float(np.trace(beats @ beats @ beats)) / 3.0
    total = math.comb(n, 3)
    return ind, tri / total


def expect(k, m, n_opt, noise, reps, seed):
    """Ожидаемые признаки и их разброс для данного правила."""
    rs = np.random.default_rng(seed)
    f = np.array([features(responses(k, m, n_opt, noise, rs))
                  for _ in range(reps)])
    return f.mean(0), f.std(0)


class Estimator:
    """Таблица ожиданий по сетке правил; оценка — ближайшее правило."""

    def __init__(self, n_opt, noise, reps=60, seed=11):
        self.grid = GRID
        self.n_opt = n_opt
        self.noise = noise
        self.mu, self.sd = [], []
        for k, m in self.grid:
            mu, sd = expect(k, m, n_opt, noise, reps, seed + 13 * k + m)
            self.mu.append(mu)
            self.sd.append(np.maximum(sd, 1e-6))
        self.mu = np.array(self.mu)
        self.sd = np.array(self.sd)

    def fit(self, feat):
        """Возвращает оценку (k, m) и расстояние до неё в сигмах."""
        z = (self.mu - np.asarray(feat)) / self.sd
        d = (z ** 2).sum(1)
        i = int(np.argmin(d))
        return self.grid[i], float(np.sqrt(d[i]))

    def strictness(self, feat):
        """Оценка строгости m/k — величина, которая опознаётся надёжнее
        самой пары (k, m)."""
        z = (self.mu - np.asarray(feat)) / self.sd
        w = np.exp(-0.5 * (z ** 2).sum(1))
        if w.sum() <= 0:
            return float('nan'), float('nan')
        w /= w.sum()
        s = np.array([m / k for k, m in self.grid])
        mean = float((w * s).sum())
        sd = float(np.sqrt((w * (s - mean) ** 2).sum()))
        return mean, sd


def recover(n_opt, noise, trials, seed):
    est = Estimator(n_opt, noise, seed=seed)
    print(f'  вариантов {n_opt}, уровень ошибок {noise:.0%}, '
          f'проб на правило {trials}')
    print(f"{'правда':>10}{'строгость':>11}{'оценка пары':>14}"
          f"{'верно':>8}{'оценка строгости':>22}{'ошибка':>9}")
    errs, hits, tot = [], 0, 0
    for k, m in GRID:
        rs = np.random.default_rng(seed + 977 * k + m)
        got, ss = [], []
        for _ in range(trials):
            f = features(responses(k, m, n_opt, noise, rs))
            got.append(est.fit(f)[0])
            ss.append(est.strictness(f)[0])
        exact = sum(1 for g in got if g == (k, m))
        hits += exact
        tot += trials
        s_true = m / k
        s_hat = float(np.mean(ss))
        s_sd = float(np.std(ss))
        errs.append(abs(s_hat - s_true))
        common = max(set(got), key=got.count)
        print(f'{f"k={k}, m={m}":>10}{s_true:>11.3f}'
              f'{f"k={common[0]}, m={common[1]}":>14}'
              f'{f"{exact/trials:.0%}":>8}'
              f'{f"{s_hat:.3f} ± {s_sd:.3f}":>22}'
              f'{s_hat - s_true:>+9.3f}')
    print()
    print(f'  пара (k, m) опознана точно: {hits/tot:.0%}')
    print(f'  строгость m/k: средняя ошибка {np.mean(errs):.3f}, '
          f'худшая {np.max(errs):.3f}')
    return float(np.mean(errs)), hits / tot


def main():
    ap = argparse.ArgumentParser(
        description='Измеритель строгости правила выбора')
    ap.add_argument('--options', type=int, default=24)
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--recover', action='store_true',
                    help='проверка восстановления на известных правилах')
    ap.add_argument('--sweep', action='store_true',
                    help='сколько вариантов нужно для нужной точности')
    a = ap.parse_args()

    t0 = time.time()
    if a.sweep:
        print('СКОЛЬКО ВАРИАНТОВ НУЖНО')
        print(f"{'вариантов':>11}{'сравнений':>11}{'ошибок':>9}"
              f"{'ошибка m/k':>13}{'пара точно':>12}")
        for n_opt in (12, 16, 24, 32, 40):
            for noise in (0.05,):
                e, h = None, None
                est_err, hit = recover_quiet(n_opt, noise, 25, a.seed)
                print(f'{n_opt:>11}{math.comb(n_opt,2):>11}'
                      f'{f"{noise:.0%}":>9}{est_err:>13.3f}{hit:>12.0%}')
    else:
        print('ВОССТАНОВЛЕНИЕ ПРАВИЛА ИЗ НАБЛЮДЁННЫХ ВЫБОРОВ')
        recover(a.options, a.noise, a.trials, a.seed)
    print(f'  время {time.time()-t0:.0f} с')


def recover_quiet(n_opt, noise, trials, seed):
    est = Estimator(n_opt, noise, reps=40, seed=seed)
    errs, hits, tot = [], 0, 0
    for k, m in GRID:
        rs = np.random.default_rng(seed + 977 * k + m)
        got, ss = [], []
        for _ in range(trials):
            f = features(responses(k, m, n_opt, noise, rs))
            got.append(est.fit(f)[0])
            ss.append(est.strictness(f)[0])
        hits += sum(1 for g in got if g == (k, m))
        tot += trials
        errs.append(abs(float(np.mean(ss)) - m / k))
    return float(np.mean(errs)), hits / tot


if __name__ == '__main__':
    main()
