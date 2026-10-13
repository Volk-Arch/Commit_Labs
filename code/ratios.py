#!/usr/bin/env python3
"""МНОГО УСЛОВИЙ БЕЗ ПАРАМЕТРОВ ПРОТИВ ДВУХ ПАРАМЕТРОВ.

Чему научил ident4.py. Два требования без параметров (тождество=1/2 и
r=r засева) враг РЕШАЕТ своими двумя параметрами: при N=1600 тождество
+1.1σ, при N=800 совместная невязка 2.8σ. Значит дело не в типе
инструмента («тождество лучше статистики»), а в АРИФМЕТИКЕ: условия
против параметров. Весь журнал читается одной строкой - восемь
маршрутов давали врагу не больше четырёх условий против трёх ручек.

Источник МНОГИХ условий без параметров получен этой же сессией с
точностью 0.3% (`nm-doc.md`): в 4d
    <N_m> -> (a/8) Gamma(m+1/2)/m! N^{3/2}
значит ОТНОШЕНИЯ
    <N_m>/<N_0> = Gamma(m+1/2)/m!/sqrt(pi) = 1, 1/2, 3/8, 5/16, ...
В отношениях сокращаются и амплитуда a, и N^{3/2}: параметров НОЛЬ, а
условий столько, сколько возьмём m. Восемь против двух.

Врагу даётся честная подгонка: его (k, q) подбирается по доле связанных
r к засеву d=4, и только потом сравниваются отношения, которых он уже
никак не касался.

КОНТРОЛЬ (анкер): у засева d=4 отношения обязаны лечь на
Gamma(m+1/2)/m!/sqrt(pi). Если нет - либо N мало, либо сломан счётчик,
и сравнивать нечего.
"""
import math

import numpy as np

from sectors import sprinkle_d
from lkdeg import layered

MMAX = 8
PRED = np.array([math.gamma(m + 0.5) / math.factorial(m) / math.sqrt(math.pi)
                 for m in range(MMAX + 1)])


def nm(Rc):
    """N_m, m = 0..MMAX, и доля связанных."""
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    out = np.array([float((Rc & (Iv == m)).sum()) for m in range(MMAX + 1)])
    return out, float(Rc.sum()) / (n * (n - 1) / 2)


def avg(mk, rep):
    N, R = [], []
    for _ in range(rep):
        a, r = nm(mk())
        N.append(a)
        R.append(r)
    N = np.array(N)
    rat = N[:, 1:] / np.maximum(N[:, :1], 1.0)
    return (rat.mean(axis=0), rat.std(axis=0) / math.sqrt(rep),
            float(np.mean(R)), float(np.std(R) / math.sqrt(rep)))


def show(lab, rat, err, pred):
    print(f'  {lab}')
    print('    m: ' + ''.join(f'{m:>9}' for m in range(1, MMAX + 1)))
    print('  ждём: ' + ''.join(f'{p:>9.4f}' for p in pred))
    print('  есть: ' + ''.join(f'{v:>9.4f}' for v in rat))
    sg = (rat - pred) / np.maximum(err, 1e-9)
    print('  сигмы:' + ''.join(f'{s:>9.1f}' for s in sg))
    chi = float((sg ** 2).sum())
    print(f'    хи2 = {chi:.1f} при {MMAX} условиях и НОЛЕ параметров',
          flush=True)
    return chi


def main():
    print(__doc__)
    rs = np.random.default_rng(6180339)
    n = 2000
    print(f'=== N = {n}, d = 4 ===')
    rat, err, r0, re0 = avg(lambda: sprinkle_d(n, 4, rs), 6)
    print(f'АНКЕР: засев d=4, r={r0:.4f}±{re0:.4f}')
    show('засев против Gamma(m+1/2)/m!/sqrt(pi)', rat, err, PRED[1:])

    print(f'\nВРАГ подгоняется по r={r0:.4f} (сетка по k, delta):')
    best = None
    for k in (4, 6, 9, 13, 18, 25):
        for dl in (1.0, 1.5, 2.0, 3.0, 4.0):
            q = dl * k / n
            if q >= 1.0:
                continue
            _, _, r, re = avg(lambda: layered(n, k, q, rs), 2)
            if best is None or abs(r - r0) < abs(best[0] - r0):
                best = (r, k, dl, q)
    r, k, dl, q = best
    print(f'  наилучшее по r: k={k}, delta={dl}, q={q:.4f} -> '
          f'r={r:.4f} (цель {r0:.4f})')
    rat2, err2, _, _ = avg(lambda: layered(n, k, q, rs), 6)
    show('тот же враг против тех же отношений', rat2, err2, PRED[1:])
    print('\nЧТЕНИЕ: у засева хи2 мал (анкер), у врага велик => восемь')
    print('условий без параметров двумя ручками не берутся.')


if __name__ == '__main__':
    main()
