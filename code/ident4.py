#!/usr/bin/env python3
"""СУЩЕСТВУЕТ ЛИ РЕШЕНИЕ ПРИ КАЖДОМ N, без экстраполяции.

ident3.py: предсказание для N=1600 провалилось на 138 и 430 сигм, но это
провал МОЕЙ экстраполяции, а не вывод. Закон выводился из двух точек, из
которых одна (N=400, невязка 14.7σ) решением не была. Мусор на входе.

Чисто: при каждом N отдельно спрашиваем, СУЩЕСТВУЕТ ли (k, q),
удовлетворяющее обоим требованиям без параметров
    (A) sum_m w_m n_m = 1/2    (существует даламбертиан)
    (B) r = r засева           (та же доля связанных),
и смотрим на невязку как функцию N. Никакой экстраполяции.

Сетка в ЕСТЕСТВЕННЫХ координатах: число слоёв k и средняя прямая
степень delta = q N / k (а не q, которое само зависит от N). Тогда
сетки при разных N сравнимы, и это та ошибка, из-за которой прежний
поиск при N=400 промахнулся.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * невязка стоит на уровне 1-2σ при всех N => у врага решение есть
    всегда, тождество имитируемо, и тип инструмента «тождество» тоже
    закрыт;
  * невязка растёт с N => область закрывается, и это вывод без
    подгонки.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from lkdeg import layered

W = (1.0, -2.0, 1.0)
KS = (8, 11, 14, 17, 21, 26, 32)
DEL = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0)


def both(Rc):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    tot = np.zeros(n)
    for m, w in enumerate(W):
        tot += w * (Rc.T & (Iv.T == m)).sum(axis=1)
    return float(tot.mean()), float(Rc.sum()) / (n * (n - 1) / 2)


def avg(mk, rep):
    v = np.array([both(mk()) for _ in range(rep)])
    return (v[:, 0].mean(), v[:, 0].std() / math.sqrt(rep),
            v[:, 1].mean(), v[:, 1].std() / math.sqrt(rep))


def main():
    print(__doc__)
    rs = np.random.default_rng(2718281)
    for n, arep, erep in ((400, 50, 6), (1600, 10, 3)):
        i0, ie0, r0, re0 = avg(lambda: causal(sprinkle_11(n, rs)), arep)
        print(f'\n=== N={n} ===')
        print(f'АНКЕР засев: тождество {i0:+.4f}±{ie0:.4f} '
              f'({(i0 - 0.5) / ie0:+.1f}σ от 1/2), r={r0:.4f}±{re0:.4f}')
        best = None
        for k in KS:
            for dl in DEL:
                q = dl * k / n
                if q >= 1.0:
                    continue
                i, ie, r, re = avg(lambda: layered(n, k, q, rs), erep)
                d = math.hypot((i - 0.5) / max(ie, 1e-6),
                               (r - r0) / max(re, 1e-6))
                if best is None or d < best[0]:
                    best = (d, k, dl, q, i, ie, r, re)
        d, k, dl, q, i, ie, r, re = best
        print(f'лучшее решение врага: k={k}, delta={dl}, q={q:.4f}')
        print(f'  тождество {i:+.3f}±{ie:.3f} '
              f'=> {(i - 0.5) / max(ie, 1e-6):+.1f}σ от 1/2')
        print(f'  r={r:.4f}±{re:.4f} '
              f'=> {(r - r0) / max(re, 1e-6):+.1f}σ от засева')
        print(f'  СОВМЕСТНАЯ НЕВЯЗКА: {d:.1f}σ', flush=True)
    print('\nЧТЕНИЕ: невязка по N. Стоит на 1-2σ => враг решает всегда.')
    print('Растёт => область закрывается.')


if __name__ == '__main__':
    main()
