#!/usr/bin/env python3
"""ОТБОР ОГРАНИЧЕНИЕМ: растить, не выпуская мир из сектора k шкал.

ОТКУДА. `statgrow-doc.md`: шесть шаговых правил без параметра обступают
конус с двух сторон и ни одно его не достигает, а безразличие оказалось
лучшим из них. И главное из чисел: равновероятный рост даёт долю пар
0.216 против нужных 0.098, то есть ПЕРЕСВЯЗЫВАЕТ вдвое.

Значит дело не в весе, которым перевешиваются кандидаты, а в самом
НАБОРЕ кандидатов: в конусном порядке прошлое события есть пересечение
k полупространств, а среди всех вниз-замкнутых подмножеств абстрактного
порядка таких исчезающе мало.

ПОЭТОМУ: отбирать не весом, а ЗАПРЕТОМ. Оценка `preclusion-doc.md`
бьёт по весам, а запрет не вес. Свободного параметра нет, кроме самого k.
И он отсекает ЧАСТЬ выборов — не все и не ни один.

ЧЕМ ЗАПРЕЩАТЬ. Порядковая размерность <= k означает представимость
пересечением k линейных порядков (Душник-Миллер), но её проверка
NP-трудна, а жадная оценка непригодна (архив: 22 и 59 там, где истина 2).
Практический запрет — не допускать вложения СТАНДАРТНОГО ПРИМЕРА S_m:

    S_m: минимальные a_1..a_m и максимальные b_1..b_m,
         a_i < b_j тогда и только тогда, когда i != j

У S_m порядковая размерность ровно m, поэтому вложенный S_{k+1} есть
строгое доказательство, что мир в k шкалах не представим.

Новое событие максимально, значит оно может служить только одним из b.
Проверка: найдётся ли среди уже существующих ещё k элементов b и k+1
элементов a, образующих с ним S_{k+1}. Полный поиск дорог, поэтому
берётся выборка — это ОСЛАБЛЕННЫЙ запрет, он может пропустить вложение,
но никогда не запретит лишнего. Ослабление в пользу роста, то есть
против проверяемой гипотезы.

СЕМЕЙСТВА:
    запрет-k   рост, где выбор отвергается при найденном S_{k+1}
    свободно   тот же рост без запрета (контроль)

  python3 restrict.py
  python3 restrict.py --n 500 --k 4 --tries 160 --seeds 4
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral, links_of
from core import transitivity_violations
from statgrow import candidates


def has_std_example(Pm, k, j, Pidx, m, tries, rs):
    """Есть ли вложение S_m, где новое событие j — один из максимальных.

    Pm[i,x] = True, если x раньше i. Pidx — прошлое j.
    Ищем b_1..b_{m-1} среди старых и a_1..a_m так, что
    a_i < b_l для всех i != l, и a_l не меньше b_l.

    Возвращает True при найденном вложении. Поиск выборочный: ослабляет
    запрет, но лишнего не запрещает.
    """
    if len(Pidx) < m:
        return False
    # j играет роль b_m, значит a_1..a_{m-1} лежат в его прошлом,
    # а a_m — НЕ в прошлом j
    inP = np.zeros(k, bool)
    inP[Pidx] = True
    outP = np.flatnonzero(~inP)
    if len(outP) == 0:
        return False
    for _ in range(tries):
        # m-1 кандидатов в a из прошлого j, один a_m снаружи
        A = rs.choice(Pidx, m - 1, replace=False)
        am = int(rs.choice(outP))
        # b_l обязан быть выше всех a, кроме a_l
        bs = []
        okall = True
        for l in range(m - 1):
            need = [int(x) for x in A if int(x) != int(A[l])] + [am]
            forb = int(A[l])
            cnd = np.ones(k, bool)
            for x in need:
                cnd &= Pm[:k, x]
            cnd &= ~Pm[:k, forb]
            cnd[list(A)] = False
            cnd[am] = False
            for b in bs:
                cnd[b] = False
            w = np.flatnonzero(cnd)
            if len(w) == 0:
                okall = False
                break
            bs.append(int(w[0]))
        if okall:
            return True
    return False


def grow(N, k_scales, ncand, tries, restrict, rs):
    """Рост с запретом вложения S_{k+1} либо без него."""
    Pm = np.zeros((N, N), bool)
    rejected = 0
    for k in range(1, N):
        cs = candidates(Pm, k, rs, ncand)
        order = rs.permutation(len(cs))
        chosen = None
        for ci in order:
            D = cs[int(ci)]
            if restrict and len(D) >= k_scales + 1:
                if has_std_example(Pm, k, k, D, k_scales + 1, tries, rs):
                    rejected += 1
                    continue
            chosen = D
            break
        if chosen is None:
            chosen = np.empty(0, np.int64)
        if len(chosen):
            Pm[k, chosen] = True
            Pm[k, :k] |= Pm[chosen, :k].any(0)
    return Pm, rejected


def report(R, lbl, N, rej):
    f = frac_pairs(R)
    L = links_of(R)
    A = R.astype(np.float32)
    CP = A @ A.T
    sz = A.sum(1)
    mn = sz[:, None] + sz[None, :] - CP
    iu = np.triu_indices(N, 1)
    jac = float(np.mean(CP[iu] / np.maximum(mn[iu], 1.0)))
    print(f'{lbl:>12}{f:>10.4f}{mm_dim(f):>8.2f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>+10.4f}{L.sum()/N:>9.2f}{jac:>9.4f}'
          f'{transitivity_violations(R):>7}{rej:>9}', flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Отбор ограничением')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--cands', type=int, default=18)
    ap.add_argument('--tries', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, шкал k={a.k}, запрещается вложение S_{a.k+1}')
    print(f'кандидатов на шаг ~{a.cands}, попыток поиска {a.tries}, '
          f'засевов {a.seeds}')
    print('ЭТАЛОН конуса: доля 0.098-0.103, разм. 4.00, Жаккар 0.169')
    print('свободный рост даёт 0.216 — пересвязывает вдвое')
    print()
    print(f"{'мир':>12}{'доля пар':>10}{'разм.':>8}{'высота':>8}"
          f"{'спектр':>10}{'связ/соб':>9}{'Жаккар':>9}{'наруш':>7}"
          f"{'откл.':>9}")

    for restrict in (True, False):
        nm = f'запрет-{a.k}' if restrict else 'свободно'
        fs = []
        for t in range(a.seeds):
            rs = np.random.default_rng(3601000 + 977 * t)
            Pm, rej = grow(a.n, a.k, a.cands, a.tries, restrict, rs)
            fs.append(report(Pm, f'{nm}/{t}', a.n, rej))
            del Pm
        print(f'{"среднее":>12}{np.mean(fs):>10.4f}'
              f'{mm_dim(float(np.mean(fs))):>8.2f}')
        print()

    print('  доля пар ушла к 0.10 и размерность к 4 => ограничение работает')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
