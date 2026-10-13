#!/usr/bin/env python3
"""ДЕЙСТВИЕ С МИНИМАЛЬНОЙ ДИСПЕРСИЕЙ: среднее не всё решает.

ОТКУДА ЗАДАЧА. `band.py` показал: фазовый отбор закрыт ПО ЗНАКУ —
многообразные порядки шумят сильнее слоистых при любом сглаживании
(отношение `s_м/s_л` от 5.4 до 1.5). Причина: действие БД не
самоусредняется на посевах ОДНОЙ геометрии, `σ = 2.0·N`.

Но действие БД получено обратной разработкой из Эйнштейна–Гильберта:
его СРЕДНЕЕ подогнано к `∫R`, а дисперсия в конструкцию не входила
вовсе. При этом «правильное среднее» — всего несколько линейных
условий, а свободы в коэффициентах много.

ЗАДАЧА. Среди линейных комбинаций счётов слоёв

    S_w = w₋₁·N + Σ_{m=0..K} w_m · L_m,
    L_m = число пар x ≺ y ровно с m элементами между,

найти ту, у которой дисперсия на посевах МИНИМАЛЬНА при тех же
непрерывных условиях. Если её `σ` опустится ниже слоистых, фазовый путь
открывается заново — уже с правильным знаком.

УСЛОВИЯ (все линейные по `w`, все проверяемые замером, без литературы):
  1-2. ⟨S_w⟩ = 0 на ПЛОСКОМ при ДВУХ размерах `N` — это зануляет и
       ведущий член (`⟨L_m⟩ ∝ N^{3/2}`), и подведущий (`∝ N`);
  3.   ⟨S_w⟩ = N·⟨R⟩ на КРИВОМ — нормировка отклика на кривизну.
Минимизируется `wᵀΣw` при `Aᵀw = b`; решение `w = Σ⁻¹A(AᵀΣ⁻¹A)⁻¹b`.

ГЛАВНАЯ ОПАСНОСТЬ И ЗАЩИТА ОТ НЕЁ. Такая задача легко подгоняет ШУМ
ковариации: `w` настроится на случайные особенности выборки и даст
фиктивно малую дисперсию. Поэтому `w` строится на ОДНОЙ половине
засевов, а `σ` считается на ДРУГОЙ, отложенной. Если на отложенной
дисперсия не упала — выигрыша нет, сколько бы его ни показывала
обучающая половина.

КОНТРОЛЬ: комбинация БД `(1, −1, 9, −16, 8)·(4/√6)` проходит той же
дорогой и обязана дать `s ≈ 2.0`.

  python3 design.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from curve import curvature, sprinkle as csprinkle
from ricci import causal
from uphill import gen3

K4 = 4.0 / math.sqrt(6.0)
KMAX = 16
PARTS = (0.25, 0.5, 0.25)


def vec(A):
    """(N, L_0 .. L_{KMAX-1})."""
    n = A.shape[0]
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    rel = B[A]
    out = np.zeros(KMAX + 1)
    out[0] = n
    bc = np.bincount(rel, minlength=KMAX)[:KMAX]
    out[1:] = bc
    return out


def bd_w():
    w = np.zeros(KMAX + 1)
    w[0] = 1.0
    for i, c in enumerate((-1.0, 9.0, -16.0, 8.0)):
        w[1 + i] = c
    return K4 * w


def flat_sample(n, seeds, root):
    return np.array([vec(sprinkle_d(n, 4, np.random.default_rng(root + 97 * s)))
                     for s in range(seeds)], float)


def curved_sample(par, seeds, root, L=8.5, T=4.0, E0=2.0):
    rows, Rs = [], []
    for s in range(seeds):
        X = csprinkle('quad', par, L, T, E0, np.random.default_rng(root + 97 * s))
        A = causal(X)
        rows.append(vec(A))
        Rs.append(np.mean([curvature(e, 'quad', par, E0)[0] for e in X[:, 0]])
                  * len(X))
        del A, X
    return np.array(rows, float), np.array(Rs, float)


def solve(Sig, A, b):
    Si = np.linalg.pinv(Sig)
    M = A.T @ Si @ A
    return Si @ A @ (np.linalg.pinv(M) @ b)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N1, N2 = 800, 1600
    SD = 60
    print('=' * 100)
    print('Сбор выборок', flush=True)
    F1 = flat_sample(N1, SD, 11000)
    F2 = flat_sample(N2, SD, 12000)
    print(f'  плоское: {len(F1)} при N={N1}, {len(F2)} при N={N2} '
          f'({time.time()-t0:.0f} с)', flush=True)
    Cp, Rp = curved_sample(+0.10, 30, 13000)
    Cm, Rm = curved_sample(-0.10, 30, 14000)
    print(f'  кривое: {len(Cp)} и {len(Cm)} ({time.time()-t0:.0f} с)',
          flush=True)
    LAY = np.array([vec(gen3(N2, PARTS, 0.5,
                             np.random.default_rng(15000 + 97 * s))[0])
                    for s in range(20)], float)
    print(f'  слоистое: {len(LAY)} ({time.time()-t0:.0f} с)', flush=True)
    print()

    half = SD // 2
    tr1, te1 = F1[:half], F1[half:]
    tr2, te2 = F2[:half], F2[half:]

    # отклик на кривизну: разность средних, нормированная на N⟨R⟩
    dmu = Cp.mean(axis=0) - Cm.mean(axis=0)
    dR = float(Rp.mean() - Rm.mean())
    Acon = np.column_stack([tr1.mean(axis=0), tr2.mean(axis=0), dmu])
    b = np.array([0.0, 0.0, dR])

    Sig = np.cov(np.vstack([tr2 - tr2.mean(axis=0)]).T)

    print('=' * 100)
    print('РЕЗУЛЬТАТ. s = σ(S_w)/N, и обучающая против ОТЛОЖЕННОЙ половины')
    print(f"{'комбинация':>26}{'s обуч.':>11}{'s ОТЛОЖ.':>11}"
          f"{'⟨S⟩/N плоск.':>15}{'отклик/N⟨R⟩':>14}{'s слоист.':>11}")

    def report(tag, w):
        s_tr = np.std(tr2 @ w, ddof=1) / N2
        s_te = np.std(te2 @ w, ddof=1) / N2
        mu = float(te2.mean(axis=0) @ w) / N2
        resp = float(dmu @ w) / dR if dR != 0 else float('nan')
        s_lay = np.std(LAY @ w, ddof=1) / N2
        print(f'{tag:>26}{s_tr:>11.4f}{s_te:>11.4f}{mu:>15.4f}'
              f'{resp:>14.4f}{s_lay:>11.4f}')
        return s_te, s_lay

    bd = bd_w()
    sbd, lbd = report('БД (контроль)', bd)
    for K in (4, 6, 8, 12, 16):
        idx = np.r_[0, 1:K + 1]
        Ak = Acon[idx]
        Sk = Sig[np.ix_(idx, idx)]
        wk = solve(Sk, Ak, b)
        w = np.zeros(KMAX + 1); w[idx] = wk
        report(f'мин. дисперсия, K={K}', w)
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 100)
    print(f'''ЧТЕНИЕ
   У БД: s = {sbd:.3f} на отложенной, слоистые {lbd:.3f}.
   Нужно, чтобы s многообразных упало НИЖЕ s слоистых — тогда знак
   переворачивается и полоса по `ℏ` появляется.
   Смотреть ТОЛЬКО столбец "ОТЛОЖ.": обучающая половина покажет
   выигрыш и при подгонке шума.
''')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
