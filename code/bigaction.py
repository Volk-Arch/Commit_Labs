#!/usr/bin/env python3
"""ДЕЙСТВИЕ ПРИ N = 10^4: битовые множества вместо матрицы отношений.

Матрица отношений при N = 10^4 это 10^8 элементов, а упакованная по
битам — 12.5 МБ. Промежуточные элементы считаются как
|I(x,y)| = popcount( будущее(x) & прошлое(y) ), без единого matmul.

ЗАЧЕМ. Закон `sigma = c * k4 sqrt(Q)`, Q = sum_m w_m^2 N_m, проверен на
200…3200 с c = 1.146 ± 0.028 (наклон по N всего +0.035). Двухточечный
интеграл (`nmexact2.py`) даёт Q при любом N точно. Значит остаётся
один риск: не поедет ли `c` дальше. Это и проверяется здесь.

ПРЕДРЕГИСТРАЦИЯ (числа получены ДО этого замера, из интеграла):
печатаются первыми, и сравнение идёт с ними.

КОНТРОЛЬ: тот же счёт при N = 3200 обязан совпасть со старым кодом
(`counts4` через F@F): N_m = 118961/47169/30669/22959 и c = 1.146.
"""
import math
import sys
import time

import numpy as np

from qcarrier import sprinkle4

K4 = 4.0 / math.sqrt(6.0)
WT = np.array([-1.0, 9.0, -16.0, 8.0])

if hasattr(np, 'bitwise_count'):
    popcnt = np.bitwise_count
else:
    _LUT = np.array([bin(i).count('1') for i in range(256)], np.uint8)

    def popcnt(a):
        return _LUT[a]


def nm_big(P, chunk=192, batch=24000):
    """N_0..N_3 для сприклинга P через упакованные битовые множества."""
    P = P[np.argsort(P[:, 0])].astype(np.float32)
    n = len(P)
    t, X = P[:, 0], P[:, 1:]
    fut = np.zeros((n, (n + 7) // 8), np.uint8)
    pas = np.zeros_like(fut)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dt = t[None, :] - t[s:e, None]
        d = X[None, :, :] - X[s:e, None, :]
        R = (dt > 0) & (dt * dt > (d * d).sum(-1))
        fut[s:e] = np.packbits(R, axis=1)
        pas[s:e] = np.packbits(R.T if e - s == n else
                               ((t[s:e, None] - t[None, :]) > 0) &
                               ((t[s:e, None] - t[None, :]) ** 2 >
                                ((X[s:e, None, :] - X[None, :, :]) ** 2
                                 ).sum(-1)), axis=1)
    deg = popcnt(fut).sum(axis=1, dtype=np.int64)
    tot = int(deg.sum())
    ii = np.repeat(np.arange(n), deg)
    jj = np.concatenate([np.flatnonzero(
        np.unpackbits(fut[i], count=n)) for i in range(n)])
    out = np.zeros(4, np.int64)
    for s in range(0, tot, batch):
        e = min(tot, s + batch)
        m = popcnt(fut[ii[s:e]] & pas[jj[s:e]]).sum(axis=1, dtype=np.int32)
        for k in range(4):
            out[k] += int((m == k).sum())
    return n, out, tot


def ens(n, k, rs, verbose=False):
    rows = []
    for i in range(k):
        t0 = time.time()
        nn, nm, tot = nm_big(sprinkle4(n, rs, 1.0))
        rows.append(nm)
        if verbose and i == 0:
            print(f'    первый засев: {time.time()-t0:.1f} с, '
                  f'связанных пар {tot:,}', flush=True)
    A = np.array(rows, float)
    S = K4 * (n + A @ WT)
    Q = (A * WT ** 2).sum(axis=1)
    return (float(S.std(ddof=1)), A.mean(axis=0), float(Q.mean()),
            float(S.mean()))


def main():
    print(__doc__)
    # --- предрегистрация из интеграла ---
    pre = {}
    exec(open('nmexact2.py').read().split("def main")[0], pre)
    rs0 = np.random.default_rng(2718281)
    tt = np.concatenate([pre['taus'](2_500_000, rs0) for _ in range(10)])
    M = 25_000_000
    grid = np.array([200.0, 400, 800, 1600, 3200, 10000])
    V = pre['nm'](tt, M, grid)
    Q_pre = (V * WT ** 2).sum(axis=1)
    e_full = (math.log(Q_pre[-1] / Q_pre[0])
              / math.log(grid[-1] / grid[0]))
    e_step = (math.log(Q_pre[-1] / Q_pre[-2])
              / math.log(grid[-1] / grid[-2]))
    print('ПРЕДРЕГИСТРАЦИЯ (из двухточечного интеграла, до замера):')
    print(f'  Q при N=10^4 = {Q_pre[-1]:.4g}; '
          f'предсказанная sigma = 1.146 * k4 * sqrt(Q) = '
          f'{1.146*K4*math.sqrt(Q_pre[-1]):.0f}')
    print(f'  показатель sigma в окне 200…10^4 = полпоказателя Q = '
          f'{e_full/2:.4f}   (слоистые дают ровно 1)')
    print(f'  он же на шаге 3200 -> 10^4 = {e_step/2:.4f}')
    print(f'  <N_m> при N=10^4 = '
          + ', '.join(f'{v:.0f}' for v in V[-1]), flush=True)

    rs = np.random.default_rng(99991)
    print('\nКОНТРОЛЬ: N = 3200 новым кодом против старого')
    sd, mn, Qm, Sm = ens(3200, 12, rs, verbose=True)
    old = np.array([118961, 47169, 30669, 22959], float)
    print('  <N_m> новое: ' + ', '.join(f'{v:.0f}' for v in mn))
    print('  старое:      ' + ', '.join(f'{v:.0f}' for v in old))
    print('  отношение:   ' + ', '.join(f'{v:.3f}' for v in mn / old))
    print(f'  c = sigma / (k4 sqrt(Q)) = {sd/(K4*math.sqrt(Qm)):.3f} '
          f'(старое 1.146 ± 0.028)')

    print('\nЗАМЕР при N = 10^4', flush=True)
    sd4, mn4, Qm4, Sm4 = ens(10000, 16, rs, verbose=True)
    print('  <N_m> замер:  ' + ', '.join(f'{v:.0f}' for v in mn4))
    print('  интеграл:     ' + ', '.join(f'{v:.0f}' for v in V[-1]))
    print('  отношение:    ' + ', '.join(f'{v:.3f}' for v in mn4 / V[-1]))
    print(f'  sigma(S) = {sd4:.0f}, предсказано '
          f'{1.146*K4*math.sqrt(Qm4):.0f}, '
          f'c = {sd4/(K4*math.sqrt(Qm4)):.3f} ± '
          f'{sd4/(K4*math.sqrt(Qm4))/math.sqrt(2*15):.3f}')
    print(f'  <S> = {Sm4:.0f} (ведущий член гасится тождеством весов, '
          f'рост не ожидается)')
    print(f'\n  местный показатель sigma, 3200 -> 10^4: '
          f'{math.log(sd4/sd)/math.log(10000/3200):.4f} '
          f'| предсказано {e_step/2:.4f} | слоистые 1.0')


if __name__ == '__main__':
    main()
