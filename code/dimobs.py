#!/usr/bin/env python3
"""РАЗМЕРНОСТЬ НАБЛЮДАЕМОГО МНОГООБРАЗИЯ: сколько направлений различает порядок.

ЗАЧЕМ. Все четыре закрытые двери сводятся к одной оси: пространство-время
имеет МНОГО СТРУКТУРЫ и потому МАЛО реализаций (`e^{N lnN}` против
`e^{N²}` у слоистых). Но кратность есть свойство ОГРУБЛЕНИЯ, а не
порядка: каждый порядок есть один порядок. Если вес брать на КЛАСС, а не
на состояние, иерархия может перевернуться — у слоистого семейства
параметров мало, у метрики много.

ЧТО РЕШАЕТ. Не число параметров в определении, а число направлений,
которые порядок СПОСОБЕН РАЗЛИЧИТЬ — то есть ранг матрицы отклика
наблюдаемых, с порогом по шуму. Логарифм числа классов идёт как
`размерность × ln(точность)`, значит размерность и есть решающая
величина.

КАК МЕРИТСЯ. Для семейства с параметрами θ:
  1. J[i,j] = d<obs_i>/d theta_j — конечными разностями;
  2. Sigma = ковариация наблюдаемых по реализациям при θ;
  3. число РАЗЛИЧИМЫХ направлений = число сингулярных числе
     `Sigma^{-1/2} J delta` выше единицы, то есть таких, где изменение
     параметра на шаг `delta` виден над шумом.

ЭТО ОСТОРОЖНАЯ ОЦЕНКА СНИЗУ: наблюдаемая — лишь спектр обилий малых
интервалов, а не весь порядок. Значит настоящая размерность не меньше
измеренной.

СЕМЕЙСТВА:
  * слоистое 3 слоя: (a1, a2, q) — три параметра;
  * слоистое L слоёв: (доли и вероятности) — 2L-2 параметра, РАСТЁТ с L;
  * конформная метрика: a(eta) = 1 + sum eps_k cos(k pi (eta-eta0)/T) —
    K параметров, и K растёт пока длина волны выше масштаба дискретности.

ЕСЛИ у метрики различимых направлений больше и число их растёт с
разрешением, а у слоистых упирается — первая версия верна, и
пространство-время беднее способами СЛУЧИТЬСЯ, но богаче способами БЫТЬ.

  python3 dimobs.py
"""
import math
import time

import numpy as np

UED = np.array([0.0, 1e-9, 0.002, 0.005, 0.012, 0.03, 0.06, 0.10,
                0.16, 0.25, 0.40, 0.65, 1.0, 1e9])


def obs(A):
    """Наблюдаемая: спектр обилий интервалов в логарифмических корзинах,
    на событие. Дешева (одно умножение матриц) и не зависит от масштаба."""
    n = A.shape[0]
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    u = B[A] / float(n)
    idx = np.searchsorted(UED[1:], u, side='right')
    bc = np.bincount(idx, minlength=len(UED) - 1)[:len(UED) - 1]
    rel = float(np.count_nonzero(A)) / (n * (n - 1) / 2.0)
    return np.concatenate([[rel], bc / n])


def gen_layers(n, parts, qs, rs):
    """L слоёв, соседние связаны с вероятностями qs (L-1 чисел)."""
    p = np.array(parts, float); p = np.abs(p); p /= p.sum()
    sz = np.floor(p * n).astype(int); sz[-1] += n - sz.sum()
    sz = np.maximum(sz, 1)
    idx, c = [], 0
    for s in sz:
        idx.append(np.arange(c, min(c + s, n))); c += s
    A = np.zeros((n, n), dtype=bool)
    M = []
    for i in range(len(idx) - 1):
        q = min(max(float(qs[i]), 0.0), 1.0)
        m = rs.random((len(idx[i]), len(idx[i + 1]))) < q
        A[np.ix_(idx[i], idx[i + 1])] = m
        M.append(m)
    # замыкание: прямым произведением по цепочке слоёв
    for d in range(2, len(idx)):
        for i in range(len(idx) - d):
            acc = M[i].astype(np.float32)
            for j in range(i + 1, i + d):
                acc = acc @ M[j].astype(np.float32)
            A[np.ix_(idx[i], idx[i + d])] = acc > 0.5
    return A


def gen_conf(n, eps, rs, T=1.0, L=1.0, block=512):
    """Сприклинг конформно-плоской метрики a(eta) = 1 + sum eps_k cos(...),
    постоянная ФИЗИЧЕСКАЯ плотность: по eta плотность идёт как a^4."""
    e = np.linspace(0.0, T, 20000)
    a = np.ones_like(e)
    for k, ek in enumerate(eps, start=1):
        a += ek * np.cos(k * math.pi * (e - T / 2) / T)
    a = np.maximum(a, 0.05)
    w = a ** 4
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(e))])
    t = np.interp(rs.uniform(0, cdf[-1], n), cdf, e)
    o = np.argsort(t); t = t[o]
    X = rs.uniform(0, L, (n, 3))[o]
    sq = (X * X).sum(axis=1)
    A = np.zeros((n, n), dtype=bool)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * (X[i0:i1] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        dt = t[None, :] - t[i0:i1, None]
        A[i0:i1] = (dt > 0) & (dt * dt > d2)
    return A


def mean_obs(gen, n, theta, seeds, root, **kw):
    V = []
    for s in range(seeds):
        A = gen(n, *theta, np.random.default_rng(root + 7919 * s), **kw)
        V.append(obs(A)); del A
    return np.array(V, float)


def rank_of(gen, n, th0, steps, seeds, root, **kw):
    """Число различимых направлений: сингулярные числа Sigma^{-1/2} J
    выше единицы, где шаг параметра взят из `steps`."""
    V0 = mean_obs(gen, n, th0, seeds, root, **kw)
    mu0, Sig = V0.mean(axis=0), np.cov(V0.T)
    d = np.sqrt(np.maximum(np.diag(Sig), 1e-30)) / math.sqrt(seeds)
    cols = []
    for j, st in enumerate(steps):
        th = [list(x) if isinstance(x, (list, tuple, np.ndarray)) else x
              for x in th0]
        flat = []
        for x in th:
            flat += list(x) if isinstance(x, list) else [x]
        flat[j] += st
        # собрать обратно
        th1, c = [], 0
        for x in th0:
            if isinstance(x, (list, tuple, np.ndarray)):
                th1.append(flat[c:c + len(x)]); c += len(x)
            else:
                th1.append(flat[c]); c += 1
        V1 = mean_obs(gen, n, th1, seeds, root + 555, **kw)
        cols.append((V1.mean(axis=0) - mu0) / d)
    J = np.column_stack(cols)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv, int((sv > 1.0).sum())


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N, SD = 1200, 10
    print('=' * 92)
    print('СЛОИСТОЕ СЕМЕЙСТВО: растёт ли число различимых направлений с L?')
    print(f"{'L слоёв':>9}{'параметров':>12}{'различимых':>12}"
          f"{'сингулярные числа':>42}")
    for Lx in (3, 4, 6, 9):
        parts = [1.0 / Lx] * Lx
        qs = [0.5] * (Lx - 1)
        steps = [0.05] * Lx + [0.08] * (Lx - 1)
        sv, r = rank_of(gen_layers, N, (parts, qs), steps, SD, 3000 + Lx)
        s = '  '.join(f'{x:.1f}' for x in sv[:7])
        print(f'{Lx:>9}{2*Lx-1:>12}{r:>12}{s:>42}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 92)
    print('КОНФОРМНАЯ МЕТРИКА: растёт ли с числом модов K?')
    print(f"{'K модов':>9}{'параметров':>12}{'различимых':>12}"
          f"{'сингулярные числа':>42}")
    for K in (1, 2, 4, 8, 16):
        eps = [0.0] * K
        steps = [0.06] * K
        sv, r = rank_of(gen_conf, N, (eps,), steps, SD, 4000 + K)
        s = '  '.join(f'{x:.1f}' for x in sv[:7])
        print(f'{K:>9}{K:>12}{r:>12}{s:>42}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 92)
    print('''ЧТЕНИЕ. Если у слоистых число различимых направлений упирается
(добавление слоёв не даёт новых), а у метрики растёт с K — то
размерность наблюдаемого многообразия у пространства-времени БОЛЬШЕ, и
первая версия верна: оно беднее способами случиться, но богаче
способами быть. Оценка осторожна: наблюдаемая — лишь спектр обилий
малых интервалов, значит настоящая размерность не меньше.''')


if __name__ == '__main__':
    main()
