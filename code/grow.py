#!/usr/bin/env python3
"""ПРИБОР ДЛЯ ПОРОЖДЁННЫХ ДИНАМИКОЙ МНОЖЕСТВ: спектр обилий как мера
многообразности.

ЗАЧЕМ. Любое среднее по сприклингу готового многообразия есть
непрерывный интеграл по определению (`kinematics-doc.md`), поэтому
уравнение поля на сприклингах непроверяемо. Оно живёт только там, где
множество порождено ДИНАМИКОЙ. А литература прямо говорит, что обилия
цепей и действие БД к множествам из роста не прикладывал никто.

ПРИБОР. Для интервала считаются `C_1, C_2, C_3, C_4`. Тогда:

  1. из `C_2` берётся размерность Мирхейма–Мейера: решается
     `c_2(d) = Г(d+1)Г(d/2)/(4Г(3d/2))` относительно `d`;
  2. при ЭТОЙ `d` предсказываются `c_3(d)` и `c_4(d)` по формуле
     `c_k = Г(d+1)^(k-1) Г(d/2) Г(d) / (2^(k-1) k Г(kd/2) Г((k+1)d/2))`;
  3. отношения замер/предсказание обязаны быть ЕДИНИЦАМИ, если множество
     похоже на многообразие. Отклонение и есть КОЛИЧЕСТВЕННАЯ мера
     удалённости от многообразности — та, которой, по отчётам поиска, ни
     у кого нет.

Суть приёма: размерность подгоняется по `C_2`, а `C_3` и `C_4` остаются
ПРОВЕРКОЙ. Свободных чисел больше нет.

КОНТРОЛИ, и они идут первыми:
  * сприклинг в 4D: `d` около 4, отношения около 1;
  * сприклинг в 2D: `d` около 2, отношения около 1 — это проверяет
    зависимость формулы `c_k` от `d`, а она измерением НЕ проверялась
    (сверялось только `c_2` при d=2 и `c_2, c_3` при d=4);
  * трёхслойный порядок Клейтмана–Ротшильда: отношения обязаны уйти
    далеко от 1 — отрицательный контроль;
  * всё считается ОДНИМ и тем же кодом, чтобы сравнение было честным.

ТЕОРИИ, которые прогоняются:
  * перколяция (классический последовательный рост при постоянном `p`);
  * исходная перколяция (с началом — всё в будущем первого элемента);
  * перколяция, НАСТРОЕННАЯ по `C_2` на четырёхмерие: `p` подбирается
    так, чтобы доля упорядоченных пар дала `d = 4`. Тогда `C_3` и `C_4`
    становятся чистой проверкой.

  python3 grow.py [--n N] [--iv M]
"""
import argparse
import math
import time

import numpy as np

# ---------- спектр ----------


def c_k(k, d):
    g = math.gamma
    return (g(d + 1) ** (k - 1) * g(d / 2) * g(d)
            / (2 ** (k - 1) * k * g(k * d / 2) * g((k + 1) * d / 2)))


def mm_dim(c2, lo=0.6, hi=24.0):
    """Размерность Мирхейма–Мейера из c_2 = C_2/(C_1(C_1-1)), делением."""
    if not (c_k(2, hi) < c2 < c_k(2, lo)):
        return float('nan')
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if c_k(2, mid) > c2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spectrum(A, idx):
    """C_1..C_4 подмножества idx по матрице достижимости A."""
    S = A[np.ix_(idx, idx)]
    n = len(idx)
    if n < 5:
        return None
    past = S.sum(0).astype(np.int64)
    fut = S.sum(1).astype(np.int64)
    c2 = int(S.sum())
    c3 = int((past * fut).sum())
    c4 = int(past @ S.astype(np.int64) @ fut)
    return n, c2, c3, c4


# ---------- порождение ----------


def closure(M0):
    """Транзитивное замыкание верхнетреугольной матрицы прямых связей."""
    n = M0.shape[0]
    A = np.zeros((n, n), dtype=bool)
    for j in range(n):
        d = M0[:, j]
        if d.any():
            A[:, j] = d | A[:, d].any(axis=1)
    return A


def gen_sprinkle(n, d, rs):
    """Сприклинг в d-мерный ромб; отношение из координат."""
    if d == 2:
        U = rs.uniform(0, 1, (n, 2))          # световые координаты
        o = np.argsort(U[:, 0] + U[:, 1])
        U = U[o]
        A = (U[:, 0][:, None] < U[:, 0][None, :]) & \
            (U[:, 1][:, None] < U[:, 1][None, :])
        return A
    # d = 4: ромб собственного времени 1
    got = []
    tot = 0
    while tot < n:
        m = max(4 * (n - tot), 20000)
        t = rs.uniform(0, 1, m)
        x = rs.uniform(-0.5, 0.5, (m, 3))
        r2 = np.einsum('ij,ij->i', x, x)
        ok = (t ** 2 > r2) & ((1 - t) ** 2 > r2)
        got.append(np.column_stack([t[ok], x[ok]]))
        tot += int(ok.sum())
    X = np.vstack(got)[:n]
    X = X[np.argsort(X[:, 0])]
    sq = np.einsum('ij,ij->i', X[:, 1:], X[:, 1:])
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X[:, 1:] @ X[:, 1:].T)
    dt = X[:, 0][None, :] - X[:, 0][:, None]
    return (dt > 0) & (dt * dt > d2)


def gen_percolation(n, p, rs, originary=False):
    """Классический последовательный рост: перколяция при постоянном p."""
    M0 = (rs.uniform(0, 1, (n, n)) < p) & np.triu(np.ones((n, n), bool), 1)
    if originary:
        M0[0, 1:] = True
    return closure(M0)


def gen_kr(n, rs):
    """Трёхслойный порядок Клейтмана–Ротшильда: n/4, n/2, n/4,
    связи только между СОСЕДНИМИ слоями при p = 1/2."""
    a, b = n // 4, n // 2
    L = np.concatenate([np.zeros(a, int), np.ones(b, int),
                        2 * np.ones(n - a - b, int)])
    M0 = np.zeros((n, n), bool)
    for lo, hi in ((0, 1), (1, 2)):
        I = np.flatnonzero(L == lo)
        J = np.flatnonzero(L == hi)
        M0[np.ix_(I, J)] = rs.uniform(0, 1, (len(I), len(J))) < 0.5
    return closure(M0)


def intervals(A, want, lo, hi, rs, tries=8000):
    """Спектры интервалов I(a,b) заданного размера."""
    n = A.shape[0]
    out = []
    for _ in range(tries):
        if len(out) >= want:
            break
        a = int(rs.integers(0, n - 1))
        fut = np.flatnonzero(A[a])
        if len(fut) < lo:
            continue
        b = int(fut[rs.integers(len(fut))])
        idx = np.flatnonzero(A[a] & A[:, b])
        if not (lo <= len(idx) <= hi):
            continue
        s = spectrum(A, idx)
        if s is not None:
            out.append(s)
    return out


def report(name, rows, extra=''):
    if not rows:
        print(f'  {name:>28}: интервалов нужного размера НЕТ {extra}')
        return None
    R = np.array(rows, dtype=float)
    n = len(R)
    c1 = R[:, 0]
    h2 = R[:, 1] / (c1 * (c1 - 1))
    h3 = R[:, 2] / (c1 * (c1 - 1) * (c1 - 2))
    h4 = R[:, 3] / (c1 * (c1 - 1) * (c1 - 2) * (c1 - 3))
    d = mm_dim(float(h2.mean()))
    if not np.isfinite(d):
        print(f'  {name:>28}: C_2/C_1² = {h2.mean():.4f} ВНЕ области '
              f'размерностей {extra}')
        return None
    r3 = float(h3.mean()) / c_k(3, d)
    r4 = float(h4.mean()) / c_k(4, d)
    se = lambda v: v.std(ddof=1) / math.sqrt(n)
    e3 = se(h3) / c_k(3, d)
    e4 = se(h4) / c_k(4, d)
    print(f'  {name:>28}: C_1 {c1.mean():>6.0f}  d(MM) {d:>6.3f}'
          f'   C_3/предск {r3:>7.3f}±{e3:<6.3f}'
          f' C_4/предск {r4:>7.3f}±{e4:<6.3f}  шт {n:>4}{extra}')
    return d, r3, r4, e3, e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1400)
    ap.add_argument('--iv', type=int, default=40)
    a_ = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('формула c_k при d = 4: c_2 = %.6f, c_3 = %.3e, c_4 = %.3e'
          % (c_k(2, 4), c_k(3, 4), c_k(4, 4)))
    print('            при d = 2: c_2 = %.6f, c_3 = %.3e, c_4 = %.3e'
          % (c_k(2, 2), c_k(3, 2), c_k(4, 2)))
    print()
    LO, HI = 150, 700
    print('=' * 78)
    print('КОНТРОЛИ (идут первыми; тот же код считает всё)')
    rs = np.random.default_rng(20261007)
    for nm, gen, kw in (('сприклинг 4D', gen_sprinkle, dict(d=4)),
                        ('сприклинг 2D', gen_sprinkle, dict(d=2)),
                        ('слоистый КР (отриц.)', gen_kr, {})):
        rows = []
        for s in range(3):
            A = gen(a_.n, rs=np.random.default_rng(900 + s), **kw)
            rows += intervals(A, a_.iv, LO, HI, rs)
            del A
        report(nm, rows)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 78)
    print('РОСТ: перколяция при постоянном p')
    for p in (0.02, 0.05, 0.1, 0.2, 0.4):
        rows = []
        for s in range(3):
            A = gen_percolation(a_.n, p, np.random.default_rng(300 + s))
            rows += intervals(A, a_.iv, LO, HI, rs)
            del A
        report(f'перколяция p = {p}', rows)
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('РОСТ: исходная перколяция (с началом)')
    for p in (0.05, 0.1, 0.2):
        rows = []
        for s in range(3):
            A = gen_percolation(a_.n, p, np.random.default_rng(700 + s),
                                originary=True)
            rows += intervals(A, a_.iv, LO, HI, rs)
            del A
        report(f'исходная p = {p}', rows)
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
