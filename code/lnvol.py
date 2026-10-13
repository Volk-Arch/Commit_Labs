#!/usr/bin/env python3
"""ЕСТЬ ЛИ В ln V ДЕЙСТВИЕ БД: сходятся ли две половины дня.

Единственный открытый объект после сегодняшнего (`kin11-doc.md`,
дополнение): в 4d мера засева есть равномерная мера на ВЛОЖИМЫХ
множествах, умноженная на exp(ln V), где V - объём вложений класса.
Днём же (`crit-geom-doc.md`) действие БД оказалось бесполезным для
отбора геометрии, а (`tri-doc.md`) любой локальный вес бессилен против
энтропии.

Вопрос, который соединил бы обе половины: не является ли ЛОКАЛЬНОЕ
ПРИБЛИЖЕНИЕ ln V действием БД? Тогда действие не фундаментально и не
бесполезно - оно локальная часть энтропии вложений.

Проверка прямая. Для малых n перечисляем классы засева выборкой,
оцениваем вероятность p(класс), и гоним регрессию
    ln p  ~  a0 + a1*R + b0*N_0 + b1*N_1 + b2*N_2 + b3*N_3
где R - число отношений, N_m - обилия интервалов. Действие БД есть
S = k4 (N - N_0 + 9 N_1 - 16 N_2 + 8 N_3), значит если ln p = -beta*S +
const, коэффициенты (b0, b1, b2, b3) обязаны быть пропорциональны
(+1, -9, +16, -8).

КОНТРОЛИ:
  (a) предсказание на ОТЛОЖЕННОЙ половине классов - иначе это подгонка;
  (b) оценка вероятности только по классам, встреченным достаточно часто
      (иначе ln p - шум);
  (c) печатается R^2 и отношения коэффициентов к БД.
"""
import math

import numpy as np

from perm2iso import canon
from emb4d import pts_4d, order_4d

MM = 4


def feats(R):
    n = R.shape[0]
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    v = [float(R.sum())]
    for m in range(MM):
        v.append(float((R & (Iv == m)).sum()))
    return np.array(v)


def main():
    print(__doc__)
    rs = np.random.default_rng(31459)
    for n, M, minc in ((6, 120000, 60), (7, 160000, 40)):
        tally, fe = {}, {}
        for P in pts_4d(M, n, rs):
            R = order_4d(P)
            k = canon(R)
            if k is None:
                continue
            tally[k] = tally.get(k, 0) + 1
            if k not in fe:
                fe[k] = feats(R)
        keys = [k for k in tally if tally[k] >= minc]
        print(f'\n=== n={n}: классов {len(tally)}, '
              f'с частотой ≥{minc}: {len(keys)} '
              f'(покрывают {sum(tally[k] for k in keys) / M:.3f} меры) ===')
        X = np.array([np.concatenate([[1.0], fe[k]]) for k in keys])
        y = np.array([math.log(tally[k] / M) for k in keys])
        idx = rs.permutation(len(keys))
        tr, te = idx[: len(idx) * 2 // 3], idx[len(idx) * 2 // 3:]
        c, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred = X[te] @ c
        ss = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        cf, *_ = np.linalg.lstsq(X, y, rcond=None)
        print(f'  R² на ОТЛОЖЕННОЙ трети: {ss:.4f}')
        print(f'  коэффициенты: конст {cf[0]:+.3f}, R {cf[1]:+.4f}, '
              + ', '.join(f'N_{m} {cf[2 + m]:+.4f}' for m in range(MM)))
        bd = np.array([1.0, -9.0, 16.0, -8.0])
        got = cf[2:2 + MM]
        sc = got[0] / bd[0] if abs(bd[0]) > 0 else 1.0
        print(f'  БД ждёт пропорции (+1, −9, +16, −8); '
              f'отношения к ним: '
              + ', '.join(f'{got[m] / bd[m] / sc:+.2f}' for m in range(MM)))
        print(f'  (все четыре около 1.00 => локальная часть ln V = действие БД)',
              flush=True)


if __name__ == '__main__':
    main()
