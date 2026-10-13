#!/usr/bin/env python3
"""ВЫЖИВАЕТ ЛИ ПОДГОНКА ПРИ СМЕНЕ N: ЕСТЬ ЛИ У ВРАГА ЗАКОН.

ident2.py: моя оценка «область пуста» была неверна, и измерение это
показало. При q=0.05, k=10..16 размер слоя 50-80, q^2 a ~ 0.1, то есть
интервалы через слой МЕЛКИЕ, попадают в веса (1,-2,1), и погашение
работает иначе, чем в моей прикидке. Между k=10 (тождество 0.101,
r=0.493) и k=16 (0.703, 0.460) есть точка, где выполняются ОБА
требования. Область не пуста.

Остаётся исходный вопрос. Враг вправе менять параметры с размером (так
работал архивный p(n) = c n^{-alpha}), поэтому честно спрашивать не
«подгонялся ли он при одном N», а «есть ли единый ЗАКОН
масштабирования k(N), q(N)».

ПРОТОКОЛ - предсказание до замера, как в лучших заходах сессии:
  1. решаем задачу при N=400 и N=800 (оба требования);
  2. из двух решений извлекаем закон (показатели по N);
  3. ПЕЧАТАЕМ предсказание для N=1600 ДО его замера;
  4. мерим.
Если предсказанная точка проходит - у врага есть закон, и тождество
имитируемо. Если нет - закона нет, и это вывод без подгонки.

КОНТРОЛЬ: засев при тех же N обязан держать оба значения без подгонки
вообще.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from lkdeg import layered

W = (1.0, -2.0, 1.0)


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


def solve(n, r0, ks, qs, rep, rs):
    """ближайшая (k,q) к (тождество=1/2, r=r0) в единицах ошибок."""
    best = None
    for k in ks:
        for q in qs:
            i, ie, r, re = avg(lambda: layered(n, k, q, rs), rep)
            d = math.hypot((i - 0.5) / max(ie, 1e-6),
                           (r - r0) / max(re, 1e-6))
            if best is None or d < best[0]:
                best = (d, k, q, i, ie, r, re)
    return best


def main():
    print(__doc__)
    rs = np.random.default_rng(1414213)

    anch = {}
    print('АНКЕР (засев d=2, без подгонки):')
    for n, rep in ((400, 50), (800, 25), (1600, 10)):
        i, ie, r, re = avg(lambda: causal(sprinkle_11(n, rs)), rep)
        anch[n] = (i, ie, r, re)
        print(f'  N={n:>5}: тождество {i:+.4f}±{ie:.4f}, r={r:.4f}±{re:.4f}',
              flush=True)

    sols = {}
    for n, rep in ((400, 5), (800, 4)):
        ks = [6, 8, 10, 12, 14, 17, 20, 24] if n == 400 else \
             [8, 11, 14, 17, 20, 24, 29, 34]
        qs = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13]
        b = solve(n, anch[n][2], ks, qs, rep, rs)
        sols[n] = b
        print(f'\nрешение при N={n}: k={b[1]}, q={b[2]:.3f}  '
              f'(невязка {b[0]:.1f}σ)')
        print(f'  тождество {b[3]:+.3f}±{b[4]:.3f}, '
              f'r={b[5]:.3f}±{b[6]:.3f}, цель r={anch[n][2]:.3f}')
        print(f'  средняя прямая степень q*N/k = {b[2] * n / b[1]:.2f}')

    k1, q1 = sols[400][1], sols[400][2]
    k2, q2 = sols[800][1], sols[800][2]
    ak = math.log(k2 / k1) / math.log(2.0)
    aq = math.log(q2 / q1) / math.log(2.0)
    kp = k2 * 2.0 ** ak
    qp = q2 * 2.0 ** aq
    print(f'\nЗАКОН из двух точек: k ~ N^{ak:+.3f}, q ~ N^{aq:+.3f}')
    print(f'ПРЕДСКАЗАНИЕ для N=1600 ДО замера: k={kp:.1f}, q={qp:.4f}')
    print(f'  (целимся в тождество=0.5 и r={anch[1600][2]:.3f})')

    kk = max(2, int(round(kp)))
    i, ie, r, re = avg(lambda: layered(1600, kk, min(qp, 0.95), rs), 3)
    print(f'\nЗАМЕР при N=1600, k={kk}, q={min(qp, 0.95):.4f}:')
    print(f'  тождество {i:+.3f}±{ie:.3f}  '
          f'=> {(i - 0.5) / max(ie, 1e-6):+.1f}σ от 1/2')
    print(f'  r={r:.3f}±{re:.3f}  '
          f'=> {(r - anch[1600][2]) / max(re, 1e-6):+.1f}σ от засева')
    print('\nЧТЕНИЕ: предсказание прошло => у врага есть закон, тождество')
    print('имитируемо. Не прошло => закона нет.')


if __name__ == '__main__':
    main()
