#!/usr/bin/env python3
"""ДЕФИЦИТ ЭНТРОПИИ как прямая задача, без состязательной игры.

census6.py дал «дефицит ~ N^4.96» — невозможное число (энтропии не
больше N^2) с местными показателями 6.43 и 3.60. Причина: Σλ·C в тот
круг, когда игра СЛУЧАЙНО перевернулась, есть промах переворота, а не
минимакс. Протокол из 6 кругов по 60 выборок не сходится, и при больших
N промах больше. Это та же болезнь, что уже трижды за неделю: число
оказалось числом протокола.

Здесь по двойственности задача ставится прямо, на λ, без θ и без игры:

    дефицит(N) = max ( Σ λ_j S_j - S_засев )
                 при   Σ λ_j f_j = f_засев,   Σ λ_j = 1,   λ >= 0

Это ровно утверждение о выпуклости: если система РАЗРЕШИМА, засев
лежит внутри оболочки счётов соперников, и никакое линейное действие
его не выделит; значение задачи и есть то, что обязана побить вогнутая
добавка. Если система НЕразрешима — засев вне оболочки, линейное
действие существует.

Протокол один и тот же при каждом N и задан до прогона: плотная семья
по k (малые абсолютные плюс один большой), тонкая сетка по q, три вида
профиля; координаты — распределение интервалов по лог-полосам плюс
граничные счёты (самый сильный набор из переписи).

КОНТРОЛИ:
  1. невязка равенств |Σλ_j f_j - f_засев| / |f_засев| — если она не
     машинная, значение бессмысленно;
  2. вырожденность: сколько соперников несут вес (если один — это не
     смесь);
  3. ВЕРХНЯЯ ГРАНИЦА: дефицит не может превосходить max_j S_j; печатаю
     отношение, и если оно близко к 1, значение упирается в границу
     семьи, а не в геометрию.
"""
import math

import numpy as np
from scipy.optimize import linprog

from census1 import feats, sprinkle_rc
from census2 import make
from census3 import perc, sprink_d


def family(n, rs, S4):
    pool = []
    ks = [2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, max(31, n // 8),
          max(32, n // 4)]
    for k in ks:
        for q in (0.02, 0.05, 0.09, 0.14, 0.2, 0.28, 0.38, 0.5, 0.64,
                  0.8, 0.93):
            for kind in (0, 1, 2):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    for p in (0.005, 0.015, 0.04, 0.1, 0.3):
        pool.append(perc(n, p, rs))
    for d in (2, 3, 5):
        pool.append(sprink_d(n, d, rs, S4))
    return pool


def deficit(n, rs):
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, _ = feats(sprinkle_rc(n, rs), n)
    f4 = np.concatenate([ab4, b4])
    sc = np.maximum(np.abs(f4), 1.0)
    pool = family(n, rs, S4)
    F = np.array([np.concatenate([m[1], m[2]]) for m in pool], float)
    S = np.array([m[3] for m in pool])
    A = np.vstack([(F / sc).T, np.ones((1, len(pool)))])
    b = np.concatenate([f4 / sc, [1.0]])
    r = linprog(-S, A_eq=A, b_eq=b, bounds=[(0, None)] * len(pool),
                method='highs')
    if r.status != 0:
        return dict(n=n, feasible=False, npool=len(pool))
    lam = r.x
    resid = float(np.abs(F.T @ lam - f4).max() / max(np.abs(f4).max(), 1))
    return dict(n=n, feasible=True, npool=len(pool),
                deficit=float(S @ lam - S4), resid=resid,
                support=int((lam > 1e-9).sum()), Smax=float(S.max()),
                top=[(round(float(lam[i]), 3), pool[i][4])
                     for i in np.argsort(lam)[::-1][:3] if lam[i] > 1e-4])


def main():
    print(__doc__)
    rs = np.random.default_rng(27182818)
    rows = []
    print(f"{'N':>6}{'семья':>7}{'разрешимо':>11}{'дефицит':>11}"
          f"{'невязка':>10}{'носитель':>10}{'деф/Smax':>10}"
          f"{'деф/N':>9}{'деф/N²':>10}")
    for n in (120, 180, 260, 380, 540, 760):
        d = deficit(n, rs)
        if not d['feasible']:
            print(f'{n:>6}{d["npool"]:>7}{"НЕТ":>11}'
                  f'{"(засев вне оболочки)":>31}')
            continue
        rows.append((n, d['deficit']))
        print(f'{n:>6}{d["npool"]:>7}{"да":>11}{d["deficit"]:>11.2f}'
              f'{d["resid"]:>10.1e}{d["support"]:>10}'
              f'{d["deficit"]/d["Smax"]:>10.4f}'
              f'{d["deficit"]/n:>9.3f}{d["deficit"]/n**2:>10.5f}',
              flush=True)
        print('        смесь: ' + '; '.join(f'λ={a} {b}'
                                            for a, b in d['top']))
    if len(rows) >= 3:
        Ns = np.array([r[0] for r in rows], float)
        D = np.array([r[1] for r in rows])
        p = np.polyfit(np.log(Ns), np.log(D), 1)[0]
        print(f'\n  дефицит ~ N^{p:.3f}')
        print('  местные: ' + ', '.join(
            f'{int(Ns[i])}->{int(Ns[i+1])}: '
            f'{math.log(D[i+1]/D[i])/math.log(Ns[i+1]/Ns[i]):.3f}'
            for i in range(len(Ns) - 1)))
        print('\n  ЧТО ЭТО ЗНАЧИТ ДЛЯ ДВЕРИ ВОГНУТОСТИ:')
        print(f'  показатель {p:.2f}. Вогнутая добавка с ограниченным')
        print('  вкладом на элемент даёт разрыв по Йенсену порядка N.')
        print('  Если показатель дефицита больше 1 — такая добавка не')
        print('  догонит; если около 1 или меньше — догонит.')


if __name__ == '__main__':
    main()
