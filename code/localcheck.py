#!/usr/bin/env python3
"""ЛОКАЛЬНАЯ ПРОВЕРКА: скольких событий хватает, чтобы отказать.

Переворот всего дня. Отбор весом проигрывает энтропии (`tri-doc.md`:
n^{3/2} против n^2). Отбор кинематическим ограничением работает, но
формулируется внутри порядка лишь в 1+1 (`kin11-doc.md`). Третий путь:
ЛОКАЛЬНАЯ ПРОВЕРКА НА МЕСТЕ - подойти к событию и спросить «сколько у
тебя предшественников и какие у них интервалы?», и если ответ неверный,
не принять. Это ограничение, а не вес, значит приговор энтропии его не
касается; и оно проверяемо без знания всего мира.

Профиль обилий - это и есть такой вопрос: N_m считает интервалы размера
m, величина локальная. Неимитируемый различитель дня (`prostoe-doc.md`)
уже имеет форму «предъяви свою окрестность».

Вопрос, без которого конструкции не будет: СКОЛЬКИХ событий хватает?
У одного элемента предшественников мало, вектор шумный. Поэтому меряем
долю отказов как функцию числа событий g, по которым проверка
усредняется. Наименьшее g, при котором засев проходит, а остальные нет,
и есть ответ на «докуда считать» с другого конца.

КОНТРОЛЬ: вторая независимая партия засевов обязана проходить почти
целиком при любом g (порог - 95-й процентиль по первой партии).
"""
import math

import numpy as np

from sectors import sprinkle_d
from lkdeg import layered
from orthant import scales, thresh_order
from waveop import causal, sprinkle_11

N = 600
MM = 9


def percell(R):
    """для каждого элемента x: вектор n_m(x), m=0..8."""
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    out = np.zeros((R.shape[0], MM))
    for m in range(MM):
        out[:, m] = (R.T & (Iv.T == m)).sum(axis=1)
    return out


def pooled(V, g, rs, reps=400):
    """суммы векторов по случайным группам из g элементов."""
    n = V.shape[0]
    idx = rs.integers(0, n, size=(reps, g))
    S = V[idx].sum(axis=1)
    tot = S.sum(axis=1, keepdims=True)
    return S / np.maximum(tot, 1.0)


def main():
    print(__doc__)
    rs = np.random.default_rng(246813)
    Vt = np.vstack([percell(sprinkle_d(N, 4, rs)) for _ in range(6)])
    Vc = np.vstack([percell(sprinkle_d(N, 4, rs)) for _ in range(6)])
    ens = (('враг k=3 q=0.2', lambda: layered(N, 3, 0.2, rs)),
           ('враг k=13 q=0.03', lambda: layered(N, 13, 0.03, rs)),
           ('орта k=4', lambda: thresh_order(scales(N, 4, rs), 4)),
           ('засев 1+1', lambda: causal(sprinkle_11(N, rs))))
    V = {lab: np.vstack([percell(mk()) for _ in range(4)])
         for lab, mk in ens}

    print(f"{'g':>4}{'порог':>9}{'ЗАСЕВ 4d':>11}"
          + ''.join(f'{lab:>18}' for lab, _ in ens))
    for g in (1, 2, 5, 10, 30, 100):
        A = pooled(Vt, g, rs)
        mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1) + 1e-12
        chi = lambda X: (((X - mu) / sd) ** 2).sum(axis=1)
        thr = float(np.percentile(chi(A), 95))
        row = f'{g:>4}{thr:>9.1f}'
        row += f'{float((chi(pooled(Vc, g, rs)) <= thr).mean()):>11.3f}'
        for lab, _ in ens:
            row += f'{float((chi(pooled(V[lab], g, rs)) <= thr).mean()):>18.3f}'
        print(row, flush=True)
    print('\nЧТЕНИЕ: столбец ЗАСЕВ 4d обязан быть около 0.95 (контроль).')
    print('Наименьшее g, где остальные столбцы падают к нулю, и есть')
    print('размер окрестности, которой хватает для отказа.')


if __name__ == '__main__':
    main()
