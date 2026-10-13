#!/usr/bin/env python3
"""ЗНАКОВАЯ поперечная шкала: разность расстояний до ДВУХ опор.

ЧЕГО НЕ ХВАТАЛО (`statgrow-doc.md`, часть вторая). Петля падала в
тотальный порядок, и причина померена:

    | |                                 наша h - W/2 | конус t - u*x |
    | разброс поперечного / времени |          0.416 |         1.413 |
    | корреляция МЕЖДУ шкалами      |        +0.9474 |       +0.2861 |
    | корреляция между поперечными  |        +0.5320 |       -0.0756 |
    | доля отрицательных поперечных |         0.0000 |        0.5057 |

W есть РАССТОЯНИЕ и всегда положительно; u*x ЗНАКОПЕРЕМЕННО.

ПОЧЕМУ РАЗНОСТЬ ДВУХ. Одна мировая линия стороны не задаёт —
поперечное пространство трёхмерно, канонической стороны нет. Сторона
появляется только относительно ВТОРОЙ линии. И тогда:

    r2^2 - r1^2 = |x1|^2 - |x2|^2 - 2*x*(x2 - x1)
    r2 - r1 = (r2^2 - r1^2) / (r1 + r2)

то есть разность расстояний ЛИНЕЙНА по x с направлением вдоль
(x1 - x2). Разность поперечных расстояний до двух опор И ЕСТЬ знаковая
проекция u*x, где u задано направлением между опорами.

ШКАЛА, без параметра, обе части в звеньях опорных цепей:

    v_pair(b) = h(b) - (W_2(b) - W_1(b)) / 2

k шкал = k ПАР опорных мировых линий, то есть k направлений u.

Опоры выбираются порядково: берутся события с МАЛЫМ взаимным
перекрытием прошлых q, то есть далёкие друг от друга. Координат не
используется.

ПОРЯДОК ДЕЙСТВИЙ. Сперва диагностики новой шкалы против конусных — если
структура не та, петлю гонять незачем. Потом петля.

  python3 signed.py
  python3 signed.py --n 700 --krefs 5 --iters 8
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral
from core import transitivity_violations
from fixpoint import diamond, cone_order, box_order, rel_of_int
from coneloop import heights_of, chain_through


def pick_pairs(R, kref, rs):
    """k пар опорных событий, далёких друг от друга по перекрытию прошлых."""
    N = len(R)
    h = heights_of(R)
    mid = np.flatnonzero((h > np.percentile(h, 20))
                         & (h < np.percentile(h, 80)))
    if len(mid) < 4 * kref:
        mid = np.arange(N)
    pool = rs.choice(mid, min(len(mid), 8 * kref), replace=False)
    A = R.astype(np.float32)
    CP = A[pool] @ A[pool].T
    sz = A[pool].sum(1)
    mn = np.minimum(sz[:, None], sz[None, :])
    q = CP / np.maximum(mn, 1.0)
    np.fill_diagonal(q, 9.0)
    pairs, used = [], set()
    flat = np.dstack(np.unravel_index(np.argsort(q, axis=None), q.shape))[0]
    for i, j in flat:
        i, j = int(i), int(j)
        if i in used or j in used:
            continue
        pairs.append((int(pool[i]), int(pool[j])))
        used.add(i); used.add(j)
        if len(pairs) >= kref:
            break
    return pairs


def signed_scales(R, pairs):
    """v[i, s] = h(i) - (W_b(i) - W_a(i))/2 для пары опор (a, b)."""
    N = len(R)
    h = heights_of(R).astype(np.float64)
    V = np.zeros((N, len(pairs)), np.float64)
    WD = np.zeros((N, len(pairs)), np.float64)
    for s, (a, b) in enumerate(pairs):
        W = []
        for v0 in (a, b):
            ch = chain_through(R, int(v0), h.astype(np.int64))
            if len(ch) < 3:
                W.append(np.zeros(N))
                continue
            rel = R[:, ch] | R[ch, :].T
            W.append((~rel).sum(1).astype(np.float64))
        d = (W[1] - W[0]) / 2.0
        WD[:, s] = d
        V[:, s] = h - d
    return V, WD, h


def diag(V, WD, h, lbl):
    k = V.shape[1]
    iu = np.triu_indices(k, 1)
    cv = np.corrcoef(V.T)[iu].mean() if k > 1 else float('nan')
    cw = np.corrcoef(WD.T)[iu].mean() if k > 1 else float('nan')
    print(f'{lbl:>24}{WD.std(0).mean()/max(h.std(),1e-9):>14.3f}'
          f'{cv:>+16.4f}{cw:>+16.4f}{(WD < 0).mean():>16.4f}')


def step(R, kref, rs):
    pairs = pick_pairs(R, kref, rs)
    V, _, _ = signed_scales(R, pairs)
    V = V + 1e-9 * rs.random(V.shape)
    return rel_of_int(V)


def report(R, lbl):
    f = frac_pairs(R)
    print(f'{lbl:>14}{f:>10.4f}{mm_dim(f):>10.2f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>+10.4f}{transitivity_violations(R):>10}',
          flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Знаковая поперечная шкала')
    ap.add_argument('--n', type=int, default=700)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--krefs', type=int, default=5)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    rs = np.random.default_rng(4101000 + a.seed)
    X = diamond(a.n, a.t, rs)
    R0 = cone_order(X)

    print(f'N={a.n}, опорных ПАР {a.krefs}')
    print('шкала v = h - (W2 - W1)/2, обе части в звеньях, параметра нет')
    print()
    print('ДИАГНОСТИКИ (считаны на конусном мире)')
    print(f"{'шкала':>24}{'попереч/время':>14}{'корр.шкал':>16}"
          f"{'корр.попереч':>16}{'доля отриц.':>16}")
    pairs = pick_pairs(R0, a.krefs, rs)
    V, WD, h = signed_scales(R0, pairs)
    diag(V, WD, h, 'знаковая (новая)')
    # для сравнения: прежняя беззнаковая h - W/2
    from coneloop import cone_scales
    srcs = np.array([p[0] for p in pairs])
    V1 = cone_scales(R0, srcs)
    hh = heights_of(R0).astype(np.float64)
    diag(V1, (hh[:, None] - V1), hh, 'беззнаковая (прежняя)')
    # эталон: настоящий конус
    u = rs.normal(size=(a.krefs, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    proj = X[:, 1:] @ u.T
    diag(X[:, 0:1] - proj, proj, X[:, 0], 'настоящий конус t-u*x')
    print()

    starts = {
        'конус': R0,
        'ящик': box_order(a.n, rs),
        'случайный': rel_of_int(rs.random((a.n, 2))),
    }
    finals = {}
    for name, Rs in starts.items():
        print(f'НАЧАЛО: {name}')
        print(f"{'итерация':>14}{'доля пар':>10}{'разм.':>10}{'высота':>8}"
              f"{'спектр':>10}{'наруш':>10}")
        R = Rs
        report(R, '0 (начало)')
        prev = None
        for it in range(1, a.iters + 1):
            Rn = step(R, a.krefs, rs)
            same = float((Rn == R).mean())
            R = Rn
            f = report(R, f'{it}')
            if prev is not None and abs(f - prev) < 1e-4 and same > 0.9999:
                print(f'   неподвижная точка на итерации {it}')
                break
            prev = f
        finals[name] = R
        print()

    print('СОШЛИСЬ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО')
    ns = list(finals)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            A, B = finals[ns[i]], finals[ns[j]]
            print(f'  {ns[i]:>10} против {ns[j]:<10} совпадение '
                  f'{float((A == B).mean()):.4f}   доли '
                  f'{frac_pairs(A):.4f} / {frac_pairs(B):.4f}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
