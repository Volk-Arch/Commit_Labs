#!/usr/bin/env python3
"""КАК РАЗМЕРНОСТЬ РАСТЁТ С N — единственная честная постановка.

Скан по разрешению прибора (`dimobs4.py`) показал, что метрику
ограничивает не прибор: при 40 полосах её ранг ПАДАЕТ с 7 до 5, и это
подпись шума — больше полос, меньше отсчётов в полосе, sigma растёт,
отклик тот же. А слоистое упирается в свой предел семейства (31 из 31)
уже при 12 полосах.

Отсюда главное: сравнение при одном N вообще НЕ про размерность.
Слоистое семейство почти детерминировано (ковариация ранга 1 — мерено
в `flip.py`), поэтому каждый его параметр читается почти без данных.
Геометрическое пуассоново-шумное, и его параметры стоят элементов.
Прошлые таблицы мерили разницу в УРОВНЕ ШУМА и принимали её за
разницу в богатстве классов.

Честная постановка одна: как число различимых направлений растёт с N у
того и у другого. Число классов идёт как `exp(dim x ln(точность))`,
значит решает именно скорость роста dim(N).

Наблюдаемая облегчена: полный `F @ F` стоит N^3, а нужен спектр
интервалов только по подвыборке строк — берётся фиксированный шаг по
времени (он же обеспечивает парность), цена падает до 600 N^2.

  python3 dimobs5.py --n 1400 [--k 12 --l 10 --nd 12 --seeds 8]
"""
import argparse
import math
import time

import numpy as np

from dimobs import gen_conf
from dimobs2 import depth, flat_pack, unpack
from dimobs4 import gen_layers_p

NDEP = 12
NROW = 600
UED = np.array([0.0, 1e-9, 0.01, 0.05, 0.15, 0.40, 1e9])


def obs(A):
    """Спектр обилий по полосам глубины, по подвыборке строк."""
    n = A.shape[0]
    h = depth(A)
    hm = max(int(h.max()), 1)
    hb = np.minimum((h * NDEP) // (hm + 1), NDEP - 1)
    st = max(1, n // NROW)
    rows = np.arange(0, n, st)
    F = A.astype(np.float32)
    B = np.rint(F[rows] @ F).astype(np.int64)
    sub = A[rows]
    ri, jj = np.nonzero(sub)
    u = B[ri, jj] / float(n)
    ub = np.minimum(np.searchsorted(UED[1:], u, side='right'),
                    len(UED) - 2)
    key = hb[rows][ri] * (len(UED) - 1) + ub
    out = np.bincount(key, minlength=NDEP * (len(UED) - 1)).astype(float)
    dens = np.bincount(hb, minlength=NDEP).astype(float)
    return np.concatenate([out / len(rows), dens / n])


def paired(gen, n, th0, steps, seeds, root):
    flat, shape = flat_pack(th0)
    base, dif = [], [[] for _ in steps]
    for s in range(seeds):
        sd = root + 7919 * s
        b = obs(gen(n, *th0, np.random.default_rng(sd)))
        base.append(b)
        for j, st in enumerate(steps):
            f2 = list(flat); f2[j] += st
            a = obs(gen(n, *unpack(f2, shape), np.random.default_rng(sd)))
            dif[j].append(a - b)
    B = np.array(base, float)
    sig = np.maximum(B.std(axis=0, ddof=1), 1.0 / n)
    cols = []
    for j in range(len(steps)):
        Dj = np.array(dif[j], float)
        mu, er = Dj.mean(axis=0), Dj.std(axis=0, ddof=1) / math.sqrt(seeds)
        cols.append(np.where(np.abs(mu) < 2.0 * er, 0.0, mu) / sig)
    J = np.column_stack(cols)
    J = J[np.abs(J).max(axis=1) > 1e-12]
    if J.size == 0:
        return np.zeros(1), 0
    sv = np.linalg.svd(J, compute_uv=False)
    return sv, int((sv > 1.0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1400)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--l', type=int, default=10)
    ap.add_argument('--nd', type=int, default=12)
    ap.add_argument('--seeds', type=int, default=8)
    g = ap.parse_args()
    global NDEP
    NDEP = g.nd
    t0 = time.time()
    N, K, L, SD = g.n, g.k, g.l, g.seeds
    parts, qs = [1.0 / L] * L, [0.5] * (L - 1)
    _, rc = paired(gen_conf, N, ([0.0] * K,), [0.04] * K, SD, 4100 + N)
    t1 = time.time()
    _, rl = paired(gen_layers_p, N, (parts, qs),
                   [0.03] * L + [0.05] * (L - 1), SD, 4200 + N)
    print(f'{N:>7}{rc:>10}/{K:<4}{rl:>10}/{2*L-1:<4}'
          f'{t1-t0:>9.0f}{time.time()-t1:>8.0f}', flush=True)


if __name__ == '__main__':
    main()
