#!/usr/bin/env python3
"""Неподвижные точки петли «порядок -> числа -> порядок».

ПОВОД. PICTURE.md: конструкция есть фильтр, а не генератор — петля
замыкается (порядок даёт числа, числа дают порядок), но ничего не
выбирает. Это установлено. А вот НЕПОДВИЖНЫЕ ТОЧКИ этой петли не
искались ни разу, хотя они были бы принципом отбора без всякого веса:
выбрана та расстановка, которая воспроизводит себя.

ОТОБРАЖЕНИЕ. По порядку R выбираются k опор; для каждого события

    v_i(b) = длиннейшая цепь от опоры r_i до b  (0, если недостижимо)

и новый порядок собирается нашим же правилом:

    R'[i,j] = True, если v(j) < v(i) ПОКОМПОНЕНТНО

Неподвижная точка: R' = R.

ЧТО ПРОВЕРЯЕТСЯ.
  1. Сходится ли итерация вообще, и к чему.
  2. Является ли конус Минковского (приближённой) неподвижной точкой.
  3. СХОДЯТСЯ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО. Если да — это настоящий принцип
     отбора; если каждое начало сидит в своей точке — это свойство
     начальных данных, и выбора нет.

  python3 fixpoint.py
  python3 fixpoint.py --iters 8 --krefs 6
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim, height, spectral, frac_pairs, transitivity_violations


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def cone_order(X, chunk=400):
    N = len(X)
    R = np.empty((N, N), bool)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        dt = X[a:b, None, 0] - X[None, :, 0]
        dx = np.linalg.norm(X[a:b, None, 1:] - X[None, :, 1:], axis=-1)
        R[a:b] = dt > dx
    np.fill_diagonal(R, False)
    return R


def box_order(N, rs):
    V = rs.random((N, 4))
    V = V[np.argsort(V.sum(1))]
    return rel_of_int(V)


def rel_of_int(V):
    """Покомпонентное строгое доминирование: R[i,j] = v(j) < v(i) по всем."""
    N = len(V)
    R = np.ones((N, N), bool)
    for m in range(V.shape[1]):
        R &= V[None, :, m] < V[:, None, m]
    np.fill_diagonal(R, False)
    return np.asarray(R, bool)


def longest_all(R, srcs):
    """L[i,s] = длиннейшая цепь от srcs[s] до i; -1 если недостижимо.

    Требует, чтобы R был ацикличен и упорядочен по времени: тогда
    R[i,j] влечёт j < i. Для надёжности делаем топологический порядок.
    """
    N, S = len(R), len(srcs)
    indeg = R.sum(1)
    order = np.argsort(indeg, kind='stable')
    pos = {int(r): s for s, r in enumerate(srcs)}
    L = np.full((N, S), -1, np.int32)
    for i in order:
        i = int(i)
        if i in pos:
            L[i, pos[i]] = max(L[i, pos[i]], 0)
        pre = np.flatnonzero(R[i])
        if len(pre):
            m = L[pre].max(0)
            L[i] = np.maximum(L[i], np.where(m >= 0, m + 1, -1))
    return L


def step(R, kref, rs):
    """Один проход петли: порядок -> числа -> порядок."""
    N = len(R)
    # опоры: события с наибольшим будущим (чтобы числа были содержательны)
    fut = R.sum(0)
    cand = np.argsort(-fut)[:max(4 * kref, 20)]
    srcs = np.array(sorted(int(c) for c in rs.choice(cand, kref,
                                                     replace=False)))
    L = longest_all(R, srcs)
    V = np.maximum(L, 0).astype(np.float64)
    # разрываем ничьи крошечным шумом по порядку индексов: иначе
    # покомпонентное СТРОГОЕ доминирование выбросит почти всё
    V = V + 1e-6 * np.arange(N)[:, None] / N
    return rel_of_int(V)


def report(R, lbl):
    f = frac_pairs(R)
    print(f'{lbl:>14}{f:>10.4f}{mm_dim(f):>10.2f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>+10.4f}{transitivity_violations(R):>10}',
          flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Неподвижные точки петли')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--krefs', type=int, default=5)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, опор {a.krefs}, итераций {a.iters}')
    print('ЭТАЛОН конуса: доля ~0.10, разм. 4.00, спектр > 0')
    print()

    rs = np.random.default_rng(401000 + a.seed)
    X = diamond(a.n, a.t, rs)
    starts = {
        'конус': cone_order(X),
        'ящик': box_order(a.n, rs),
        'случайный': rel_of_int(rs.random((a.n, 2))),
    }
    finals = {}
    for name, R0 in starts.items():
        print(f'НАЧАЛО: {name}')
        print(f"{'итерация':>14}{'доля пар':>10}{'разм.':>10}{'высота':>8}"
              f"{'спектр':>10}{'наруш':>10}")
        R = R0
        report(R, '0 (начало)')
        prev = None
        for it in range(1, a.iters + 1):
            Rn = step(R, a.krefs, rs)
            same = float((Rn == R).mean())
            R = Rn
            f = report(R, f'{it}')
            if prev is not None and abs(f - prev) < 1e-4 and same > 0.9999:
                print(f'   неподвижная точка достигнута на итерации {it}')
                break
            prev = f
        finals[name] = R
        print()

    print('СОШЛИСЬ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО')
    ns = list(finals)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            A, B = finals[ns[i]], finals[ns[j]]
            agree = float((A == B).mean())
            print(f'  {ns[i]:>10} против {ns[j]:<10} совпадение '
                  f'{agree:.4f}   доли {frac_pairs(A):.4f} / '
                  f'{frac_pairs(B):.4f}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
