#!/usr/bin/env python3
"""Пошаговый рост с настоящим действием.

ПОВОД. step-phase-doc.md: приращение действия при добавлении
МАКСИМАЛЬНОГО события равно «сумме на событие» (сверено до 4.6e-15),
действие аддитивно по шагам, и шаг обусловлен прекрасно (фаза 0.998 при
eps=0.05) там, где глобальная сумма безнадёжна (1e-304). Значит мера
должна приходить от РОСТА, а не от веса на готовых расстановках.

ПРАВИЛО. На каждом шаге новое событие x добавляется как максимальный
элемент. Его прошлое D обязано быть замкнуто вниз (если b < a и a в D,
то b в D) — иначе отношение не будет порядком. Отсюда главное:

    ТРАНЗИТИВНОСТЬ ВЫПОЛНЯЕТСЯ ПО ПОСТРОЕНИЮ.

Это тот самый пункт, на котором рассыпался лучший результат архива
(alternate.py: 41181 нарушение). Здесь он невозможен.

Приращение действия для прошлого D:

    n(b) = |{c in D : b < c}|            сколько событий между b и x
    dS   = eps + sum_{b in D} W(n(b))

ВЫБОР ПРОШЛОГО. Генерируются K кандидатов: случайное подмножество с
долей p (p берётся логарифмически равномерно, чтобы действие само
выбирало, НАСКОЛЬКО большое прошлое нужно), затем замыкание вниз. Из
кандидатов один выбирается с вероятностью ~ exp(-beta*dS).

  beta = 0   контроль: смесь транзитивных перколяций, известно
             неманифолдоподобная
  beta > 0   действие предпочитает меньшее dS
  beta < 0   обратный знак — мы знаем, что знак и есть препятствие

ПРОВЕРКА — полный набор дискриминаторов (action-weight-doc.md):
доля пар ~0.10, размерность 4.00 устойчивая, высота ~9 при N=800,
спектр ПОЛОЖИТЕЛЬНЫЙ, нарушений транзитивности ровно 0.

  python3 growth.py
  python3 growth.py --n 800 --betas -3 -1 0 1 3
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim, height, spectral, transitivity_violations, frac_pairs
from weight import w_of


def grow(N, beta, eps, K, rs, pmin_mult=0.5, pmax=0.95):
    """Вырастить порядок на N событий. R[i,j] = True, если j раньше i."""
    R = np.zeros((N, N), bool)
    wtab = w_of(np.arange(N + 2), eps)
    hist = []
    for k in range(1, N):
        Rk = R[:k, :k]
        A = Rk.astype(np.float32)
        # K кандидатов: случайные подмножества разной доли
        p = np.exp(rs.uniform(math.log(max(pmin_mult / k, 1e-4)),
                              math.log(pmax), K))
        S = (rs.random((k, K)) < p[None, :]).astype(np.float32)
        # замыкание вниз: b входит, если b в S или b раньше кого-то из S
        D = (S > 0) | ((A.T @ S) > 0)
        Df = D.astype(np.float32)
        # n(b) = число c из D, которые позже b
        n = np.rint(A.T @ Df).astype(np.int32)
        wv = wtab[np.clip(n, 0, N + 1)] * D
        dS = eps + wv.sum(0)
        if beta == 0.0:
            j = int(rs.integers(K))
        else:
            lw = -beta * dS
            lw -= lw.max()
            pr = np.exp(lw)
            pr /= pr.sum()
            j = int(rs.choice(K, p=pr))
        R[k, :k] = D[:, j]
        hist.append((int(D[:, j].sum()), float(dS[j]), float(dS.std()),
                     float(dS[j] - dS.mean())))
    return R, np.array(hist, float)


def report(R, lbl, hist, beta):
    f = frac_pairs(R)
    sel = abs(beta) * hist[:, 2].mean()          # сила отбора
    print(f'{lbl:>9}{f:>10.4f}{mm_dim(f):>11.3f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>10.4f}{transitivity_violations(R):>8}'
          f'{hist[:, 0].mean():>11.1f}{hist[:, 2].mean():>11.4f}'
          f'{sel:>10.2f}{hist[:, 3].mean():>11.4f}', flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Пошаговый рост с действием')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--k', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--betas', type=float, nargs='+',
                    default=[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
    a = ap.parse_args()

    t0 = time.time()
    print(f'рост до N={a.n}, eps={a.eps}, кандидатов на шаг {a.k}, '
          f'засевов {a.seeds}')
    print('цель (конусный Минковский при N=800): доля пар ~0.10, '
          'размерность 4.00,')
    print('высота ~9, спектр ПОЛОЖИТЕЛЬНЫЙ, нарушений транзитивности 0')
    print()
    print(f"{'beta':>9}{'доля пар':>10}{'размерн.':>11}{'высота':>8}"
          f"{'спектр':>10}{'наруш':>8}{'|прошлое|':>11}{'разбр.dS':>11}"
          f"{'сила отб.':>10}{'сдвиг dS':>11}")

    for beta in a.betas:
        fs = []
        for t in range(a.seeds):
            rs = np.random.default_rng(4400 + 17 * t)
            R, hist = grow(a.n, beta, a.eps, a.k, rs)
            if t == 0:
                fs.append(report(R, f'{beta:+.0f}', hist, beta))
            else:
                fs.append(frac_pairs(R))
            del R
        if a.seeds > 1:
            print(f"{'засевы:':>9}  {' '.join(f'{v:.4f}' for v in fs)}")

    print()
    print('ЭТАЛОНЫ ДЛЯ СРАВНЕНИЯ (N=800, из архива и наших прогонов)')
    print(f"{'конус Минковского':>22}  0.1029   3.966    9   +0.0162   0")
    print(f"{'ящик покомпонентный':>22}  0.1165   3.818   10   -0.0050   0")
    print(f"{'рост alternate (архив)':>22}  0.0951   4.060   52   -0.0008   41181")
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
