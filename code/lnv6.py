#!/usr/bin/env python3
"""ЕСТЬ ЛИ ДЕЙСТВИЕ БД В ЛОГАРИФМЕ ОБЪЁМА ВЛОЖЕНИЙ.

По вечерней картине (`kin11-doc.md`, дополнение) мера засева в 4d есть
равномерная мера на ВЛОЖИМЫХ множествах, умноженная на exp(ln V), где V -
объём вложений класса. Днём же действие БД оказалось бесполезным для
отбора (`crit-geom-doc.md`), а любой локальный вес бессилен против
энтропии (`tri-doc.md`).

Вопрос, соединяющий обе половины: не есть ли ЛОКАЛЬНАЯ ЧАСТЬ ln V
действием БД? Тогда действие не фундаментально и не бесполезно - оно
локальная часть энтропии вложений.

Действие БД: S = k4 (N - N_0 + 9 N_1 - 16 N_2 + 8 N_3). При ln p = -b*S + c
коэффициенты при (N_0, N_1, N_2, N_3) обязаны быть пропорциональны
(+1, -9, +16, -8). Число элементов N здесь постоянно (=6) и уходит в
константу.

КОНТРОЛИ:
  (a) предсказание на ОТЛОЖЕННОЙ трети классов - иначе это подгонка;
  (b) только классы с частотой >= 150, иначе ln p есть шум;
  (c) две посадки - с полным числом отношений R и без него: R в БД не
      входит, и если он нужен, это отклонение от БД.
"""
import math
import time

import numpy as np

from perm2iso import canon
from emb4d import pts_4d, order_4d

N, M, MINC = 6, 150000, 150
BD = np.array([1.0, -9.0, 16.0, -8.0])


def feats(R):
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    v = [float(R.sum())]
    for m in range(4):
        v.append(float((R & (Iv == m)).sum()))
    return np.array(v)


def fit(X, y, rs, lab):
    idx = rs.permutation(len(y))
    tr, te = idx[: len(idx) * 2 // 3], idx[len(idx) * 2 // 3:]
    c, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    pr = X[te] @ c
    r2 = 1 - ((y[te] - pr) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    cf, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f'  {lab}: R² на отложенной трети = {r2:.4f}')
    return cf


def main():
    print(__doc__)
    rs = np.random.default_rng(778899)
    t0 = time.time()
    tal, fe = {}, {}
    for P in pts_4d(M, N, rs):
        R = order_4d(P)
        k = canon(R)
        if k is None:
            continue
        tal[k] = tal.get(k, 0) + 1
        if k not in fe:
            fe[k] = feats(R)
    keys = [k for k in tal if tal[k] >= MINC]
    cov = sum(tal[k] for k in keys) / M
    print(f'n={N}, выборка {M}, классов {len(tal)}, с частотой ≥{MINC}: '
          f'{len(keys)} (покрывают {cov:.4f} меры), {time.time()-t0:.0f}с')

    y = np.array([math.log(tal[k] / M) for k in keys])
    F = np.array([fe[k] for k in keys])
    print(f'  ln p от {y.min():.2f} до {y.max():.2f}')

    X1 = np.column_stack([np.ones(len(keys)), F[:, 1:]])
    c1 = fit(X1, y, rs, 'только N_0..N_3 (как в БД)')
    g = c1[1:5]
    sc = g[0] / BD[0]
    print(f'    коэффициенты: ' + ', '.join(f'N_{m} {g[m]:+.4f}'
                                            for m in range(4)))
    print(f'    отношение к БД (+1,−9,+16,−8), нормировано по N_0: '
          + ', '.join(f'{g[m] / BD[m] / sc:+.3f}' for m in range(4)))

    X2 = np.column_stack([np.ones(len(keys)), F])
    c2 = fit(X2, y, rs, 'с полным числом отношений R')
    print(f'    коэффициент при R: {c2[1]:+.4f};  '
          + ', '.join(f'N_{m} {c2[2 + m]:+.4f}' for m in range(4)))
    print('\nЧТЕНИЕ: если все четыре отношения около 1.00 — локальная часть')
    print('ln V и есть действие БД. Если нет — печатается, чем оно стало.')


if __name__ == '__main__':
    main()
