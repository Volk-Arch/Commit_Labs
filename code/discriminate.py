#!/usr/bin/env python3
"""Различает ли инструмент порог от взвешенной полезности.

Инструмент (strictness.py) подгоняет пару (k, m) по двум признакам:
доле безразличия и частоте 3-циклов. Если он ничего настоящего не
мерит, то подгонится к любым данным, и толку от него нет.

Поэтому здесь проверка с зубами:

  1. берём данные от ДРУГОЙ модели — стандартной взвешенной полезности
     (полезность = сумма взвешенных критериев, безразличие в полосе
     вокруг равенства). Это скалярная полезность, то есть ПОРЯДОК, и
     структурных циклов у неё нет вовсе — только шумовые;
  2. выравниваем обе модели по ДОЛЕ БЕЗРАЗЛИЧИЯ, подбирая полосу. Так
     первый признак у них совпадает, и различать приходится по второму
     и третьему, а не по лёгкому;
  3. подгоняем порог по двум признакам и ПРЕДСКАЗЫВАЕМ третий —
     частоту 4-циклов, в подгонке не участвовавшую;
  4. смотрим, у кого предсказание сходится.

Если предсказание сходится у пороговых данных и ломается у
полезностных — тест отличает модели, и у инструмента есть смысл.

  python3 discriminate.py
  python3 discriminate.py --options 32
"""
import argparse
import math
import time

import numpy as np

from strictness import (GRID, IND, WIN, LOS, responses, Estimator)


def features3(R):
    """Три признака: безразличие, 3-циклы, 4-циклы."""
    n = len(R)
    iu = np.triu_indices(n, 1)
    ind = float((R[iu] == IND).mean())
    A = np.asarray(R == WIN, dtype=np.float64)
    A2 = A @ A
    c3 = float(np.trace(A2 @ A)) / 3.0 / math.comb(n, 3)
    c4 = float(np.trace(A2 @ A2)) / 4.0 / math.comb(n, 4)
    return ind, c3, c4


def utility_responses(k, n_opt, band, noise, rs):
    """Взвешенная полезность: скалярная, значит порядок без циклов.

    Безразличие — полоса шириной band вокруг равенства полезностей.
    """
    w = rs.random(k)
    w /= w.sum()
    U = rs.random((n_opt, k)) @ w
    d = U[:, None] - U[None, :]
    R = np.full((n_opt, n_opt), IND, dtype=np.int8)
    R[d > band] = WIN
    R[d < -band] = LOS
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


def match_band(k, n_opt, target_ind, noise, rs, tries=40):
    """Подбираем полосу так, чтобы доля безразличия совпала с целевой."""
    lo, hi = 0.0, 0.5
    for _ in range(tries):
        mid = (lo + hi) / 2
        got = np.mean([features3(utility_responses(k, n_opt, mid, noise, rs))[0]
                       for _ in range(8)])
        if got < target_ind:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def predict_c4(est, feat2, n_opt, noise, seed, reps=80):
    """Подгоняем (k, m) по двум признакам, предсказываем третий."""
    (k, m), _ = est.fit(feat2)
    rs = np.random.default_rng(seed)
    c4 = [features3(responses(k, m, n_opt, noise, rs))[2] for _ in range(reps)]
    return (k, m), float(np.mean(c4)), float(np.std(c4))


def main():
    ap = argparse.ArgumentParser(
        description='Порог против взвешенной полезности')
    ap.add_argument('--options', type=int, default=24)
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    est = Estimator(a.options, a.noise, reps=60, seed=a.seed)
    print(f'вариантов {a.options}, ошибок {a.noise:.0%}, '
          f'проб {a.trials}')
    print('подгонка по двум признакам (безразличие, 3-циклы), '
          'предсказывается третий (4-циклы)')
    print()
    print(f"{'данные от':>22}{'безразличие':>13}{'3-циклы':>10}"
          f"{'4-циклы: факт':>15}{'предсказано':>18}{'расхожд.':>11}")

    rows = []
    for k, m in [(4, 3), (5, 4), (6, 4), (7, 5), (8, 6)]:
        rs = np.random.default_rng(a.seed + 31 * k + m)
        # --- пороговые данные
        f = np.array([features3(responses(k, m, a.options, a.noise, rs))
                      for _ in range(a.trials)])
        mu = f.mean(0)
        fit, p, psd = predict_c4(est, mu[:2], a.options, a.noise, a.seed + 1)
        z = abs(mu[2] - p) / max(psd, 1e-9)
        rows.append(('порог', z))
        print(f'{f"порог k={k}, m={m}":>22}{mu[0]:>13.4f}{mu[1]:>10.4f}'
              f'{mu[2]:>15.4f}{f"{p:.4f} ± {psd:.4f}":>18}{z:>11.1f}σ')

        # --- полезностные данные, выровненные по безразличию
        band = match_band(k, a.options, mu[0], a.noise, rs)
        g = np.array([features3(utility_responses(k, a.options, band,
                                                  a.noise, rs))
                      for _ in range(a.trials)])
        gm = g.mean(0)
        fit2, p2, psd2 = predict_c4(est, gm[:2], a.options, a.noise, a.seed + 2)
        z2 = abs(gm[2] - p2) / max(psd2, 1e-9)
        rows.append(('полезность', z2))
        print(f'{f"полезность k={k}":>22}{gm[0]:>13.4f}{gm[1]:>10.4f}'
              f'{gm[2]:>15.4f}{f"{p2:.4f} ± {psd2:.4f}":>18}{z2:>11.1f}σ')

    thr = [z for nm, z in rows if nm == 'порог']
    uti = [z for nm, z in rows if nm == 'полезность']
    print()
    print('ИТОГ')
    print(f'  пороговые данные:    расхождение {np.mean(thr):.1f}σ в среднем, '
          f'худшее {np.max(thr):.1f}σ')
    print(f'  полезностные данные: расхождение {np.mean(uti):.1f}σ в среднем, '
          f'худшее {np.max(uti):.1f}σ')
    if np.mean(uti) > 3 and np.mean(thr) < 3:
        print('  ТЕСТ РАЗЛИЧАЕТ: предсказание сходится у порога и '
              'ломается у полезности')
    elif np.mean(thr) > 3:
        print('  ТЕСТ НЕГОДЕН: предсказание ломается даже на своих данных')
    else:
        print('  ТЕСТ НЕ РАЗЛИЧАЕТ: обе модели проходят, '
              'инструмент не опознаёт механизм')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
