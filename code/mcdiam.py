#!/usr/bin/env python3
"""exp(-beta*S) на ПРИЧИННОМ РОМБЕ: правильная геометрия.

ИСПРАВЛЕНИЕ ОШИБКИ. mcrun.py гонял цепь в ящике 4.0 x 2^3, унаследованном
из inertia.py. Этот ящик вытянут по времени, и мир в нём не
манифолдоподобен с самого начала: доля пар 0.5184, размерность 1.952,
высота 48 вместо 0.10 / 4.0 / ~9. Длинный мир вырождается в почти одну
цепь (то же, что в опыте с кадром). Для локальных разностных измерений
(вмятина) это почти безразлично, для действия — нет.

Здесь область — причинный ромб, как в weight.py, где размерность выходила
3.98. Плотность 1146 (то есть N = 150*T^4), T выбирается по N.

Дальнобойность веса относительно размера мира: для ромба
reach/T = 0.30 при N = 600, против 0.50 у прежнего ящика.

ВОПРОСЫ.
  1. Остаётся ли мир манифолдоподобным под exp(-beta*S)?
  2. Притягивает ли вес фон к локальному объекту? Эталон — beta=0,
     он же снимает все геометрические поправки формы ромба.
  3. Есть ли beta, при котором цепь устойчива (не съезжает)?

  python3 mcdiam.py
  python3 mcdiam.py --sweeps 150 --betas 0 0.1 0.3 1
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim, height
from mc import World

RHO_COEF = 150.0            # N = RHO_COEF * T^4, как в weight.py


def t_for(N):
    return (N / RHO_COEF) ** 0.25


def in_diamond(X, T):
    r = np.linalg.norm(X[..., 1:], axis=-1)
    return (X[..., 0] >= r) & (T - X[..., 0] >= r)


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        ok = in_diamond(Y, T)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    return np.vstack(out)[:N]


def axis_tube(M, rad, T, rs):
    v = rs.normal(size=(M, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(rad, T - rad, M)
    X[:, 1:] = v * r[:, None]
    return X


def diag(w):
    n = len(w.R)
    f = float(w.R.sum() / (n * (n - 1) / 2))
    return f, mm_dim(f), height(w.R)[0], float(w.S)


def chain(obj, N, T, beta, eps, sweeps, rs, step, edges, log=None):
    M = len(obj)
    w = World(np.vstack([obj, diamond(N, T, rs)]), eps)
    prof, pc, acc, out = np.zeros(len(edges) - 1), 0, 0, 0
    if log is not None:
        d = diag(w)
        log(0, d, 1.0)
    for sw in range(1, sweeps + 1):
        for _ in range(N):
            i = M + int(rs.integers(N))
            xi = w.X[i] + step * rs.normal(size=4)
            if not in_diamond(xi, T):
                out += 1
                continue                       # вне области — отказ
            st = w.move(i, xi)
            dS = w.S - st[8]
            if dS <= 0 or rs.random() < math.exp(-beta * dS):
                acc += 1
            else:
                w.undo(st)
        if sw >= sweeps // 2:
            r = np.linalg.norm(w.X[M:, 1:], axis=1)
            prof += np.bincount(np.digitize(r, edges) - 1,
                                minlength=len(edges) - 1)[:len(edges) - 1]
            pc += 1
        if log is not None and (sw % max(1, sweeps // 6) == 0
                                or sw == sweeps):
            log(sw, diag(w), acc / (sw * N))
    return prof / max(pc, 1), acc / (sweeps * N), diag(w), out / (sweeps * N)


def main():
    ap = argparse.ArgumentParser(description='exp(-beta S) на ромбе')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--rad', type=float, default=0.03)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--sweeps', type=int, default=100)
    ap.add_argument('--step', type=float, default=0.15)
    ap.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 1.0])
    a = ap.parse_args()

    t0 = time.time()
    T = t_for(a.n + a.m)
    reach = (5 * 24 / (RHO_COEF * 24 / math.pi * math.pi)) ** 0.25
    edges = np.array([0.0, 0.08, 0.16, 0.25, 0.35, 0.5, 99.0])
    print(f'ромб собственного времени T={T:.3f}, N={a.n+a.m} '
          f'({a.m} закреплено)')
    print(f'дальнобойность веса ~{reach:.3f}, то есть {reach/T:.2f} от мира')
    print(f'eps={a.eps}, шаг {a.step}, развёрток {a.sweeps}')
    print()

    print('ХОД ЦЕПИ (манифолдоподобие: доля пар ~0.10, размерность ~4)')
    for beta in a.betas:
        print(f'  beta = {beta}')
        print(f"{'развёртка':>11}{'доля пар':>10}{'размерн.':>10}"
              f"{'высота':>8}{'S':>11}{'принят':>8}")

        def log(sw, d, ac):
            print(f'{sw:>11}{d[0]:>10.4f}{d[1]:>10.3f}{d[2]:>8.0f}'
                  f'{d[3]:>11.1f}{ac:>8.2f}', flush=True)

        rs = np.random.default_rng(1234)
        chain(axis_tube(a.m, a.rad, T, rs), a.n, T, beta, a.eps,
              a.sweeps, rs, a.step, edges, log=log)
        print()

    print('ПРИТЯГИВАЕТ ЛИ — плотность фона по |x|, отнесённая к beta=0')
    print(f"{'мир':>10}{'beta':>6}{'принят':>8}{'вне обл.':>9}"
          + ''.join(f'{f"<{e}":>8}' for e in edges[1:-1]))
    for lbl, mk in (('трубка', True), ('контроль', False)):
        ref = None
        for beta in a.betas:
            ps, ac, ob = [], [], []
            for t in range(2):
                rs = np.random.default_rng(1234 + 31 * t)
                obj = (axis_tube(a.m, a.rad, T, rs) if mk
                       else diamond(a.m, T, rs))
                p, r, _, o = chain(obj, a.n, T, beta, a.eps, a.sweeps,
                                   rs, a.step, edges)
                ps.append(p)
                ac.append(r)
                ob.append(o)
            P = np.array(ps).mean(0)
            if ref is None:
                ref = P.copy()
            Q = P / np.maximum(ref, 1e-9)
            print(f'{lbl:>10}{beta:>6.1f}{np.mean(ac):>8.2f}'
                  f'{np.mean(ob):>9.2f}'
                  + ''.join(f'{v:>8.3f}' for v in Q[:len(edges) - 2]),
                  flush=True)
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
