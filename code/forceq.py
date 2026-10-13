#!/usr/bin/env python3
"""Сила против ПОРЯДКОВОЙ оси вместо координатной.

ПОВОД. Весь вечер сила мерилась против |x| — расстояния от оси трубки в
КООРДИНАТАХ, а это не порядковая величина. Профиль получался решёточным
(F ~ l^(-1.9)), и было неясно, оболочка это или артефакт линейки.

Теперь порядковая мера есть (PICTURE.md): у пары несравнимых событий есть
своё прошлое, и

    q(a,b) = |past(a) & past(b)| / min(|past(a)|, |past(b)|)

даёт пространственное расстояние с ранговой корреляцией -0.801, причём
относительный разброс 0.60 ВО ВСЕХ корзинах разноса. Больше q — ближе.

Направление смещения оставлено радиальным: трубка покоится и сама задаёт
систему отсчёта, так что «к оси» осмысленно. Менялась именно ОСЬ
графика.

  python3 forceq.py
  python3 forceq.py --seeds 12
"""
import argparse
import math
import time

import numpy as np

from mc import World
from weight import diamond


def tube(M, rad, T, rs):
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(rad, T - rad, M)
    X[:, 1:] = u * r[:, None]
    return X


def in_diamond(x, T):
    r = np.linalg.norm(x[1:])
    return (x[0] >= r) and (T - x[0] >= r)


def q_to_object(R, M):
    """q[b] = max по событиям объекта от |past&past| / min(|past|,|past|).

    Берутся только НЕСРАВНИМЫЕ пары: для связанных пространственного
    расстояния не определено.
    """
    A = R.astype(np.float32)
    CP = A[M:] @ A[:M].T                       # (фон) x (объект)
    sb = A[M:].sum(1)                          # |past| фоновых
    so = A[:M].sum(1)                          # |past| объекта
    mn = np.minimum(sb[:, None], so[None, :])
    q = CP / np.maximum(mn, 1.0)
    # несравнимость: ни фон не раньше объекта, ни объект не раньше фона
    rel = R[M:, :M] | R[:M, M:].T
    q = np.where(rel | (mn < 8), -1.0, q)
    return q.max(1)


def radial_forces(w, M, T, delta, rs, nsamp=240):
    """Радиальная сила и порядковая близость для выборки фоновых событий."""
    N = len(w.X)
    cand = np.arange(M, N)
    keep = np.abs(w.X[cand, 0] - T / 2) < 0.35 * T
    cand = cand[keep]
    if len(cand) > nsamp:
        cand = rs.choice(cand, nsamp, replace=False)
    out = []
    for i in cand:
        i = int(i)
        x0 = w.X[i].copy()
        r = np.linalg.norm(x0[1:])
        if r < 1e-6:
            continue
        e = np.zeros(4)
        e[1:] = x0[1:] / r
        xp, xm = x0 + delta * e, x0 - delta * e
        if not (in_diamond(xp, T) and in_diamond(xm, T)):
            continue
        st = w.move(i, xp)
        sp = w.S
        w.undo(st)
        st = w.move(i, xm)
        sm = w.S
        w.undo(st)
        out.append((i, r, -(sp - sm) / (2 * delta)))
    return out


def main():
    ap = argparse.ArgumentParser(description='Сила против порядковой оси')
    ap.add_argument('--t', type=float, default=1.8)
    ap.add_argument('--m', type=int, default=120)
    ap.add_argument('--rad', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--delta', type=float, default=0.06)
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    qed = np.array([0.0, 0.04, 0.08, 0.14, 0.22, 0.35, 1.01])
    print(f'ромб T={a.t}, трубка {a.m} событий радиуса {a.rad}, '
          f'eps={a.eps}')
    print('ОСЬ ПОРЯДКОВАЯ: q = |past&past|/min(|past|,|past|); '
          'больше q — ближе')
    print('F_r < 0 значит сила К ОСИ, то есть ПРИТЯЖЕНИЕ')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'мир':>10}" + ''.join(f'{f"q<{e}":>12}' for e in qed[1:]))

    res = {}
    for mk in (True, False):
        acc = [[] for _ in range(len(qed) - 1)]
        for t in range(a.seeds):
            rs = np.random.default_rng(1301000 + 977 * t)
            bg = diamond(a.t, rs)
            ob = (tube(a.m, a.rad, a.t, rs) if mk
                  else diamond(a.t, rs, n_at_t1=a.m / a.t ** 4))
            X = np.vstack([ob, bg])
            w = World(X, a.eps)
            q = q_to_object(w.R, len(ob))
            F = radial_forces(w, len(ob), a.t, a.delta, rs)
            del w
            if not F:
                continue
            for i, r, f in F:
                b = int(np.digitize(q[i - len(ob)], qed) - 1)
                if 0 <= b < len(acc):
                    acc[b].append(f)
        m = np.array([np.mean(v) if len(v) > 2 else np.nan for v in acc])
        e = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 2
                      else np.nan for v in acc])
        n = [len(v) for v in acc]
        res[mk] = (m, e)
        name = 'трубка' if mk else 'контроль'
        print(f'{name:>10}' + ''.join(f'{x:>+12.4f}' for x in m))
        print(f"{'± ошибка':>10}" + ''.join(f'{x:>12.4f}' for x in e))
        print(f"{'(событий)':>10}" + ''.join(f'{x:>12}' for x in n),
              flush=True)
    d = res[True][0] - res[False][0]
    er = np.hypot(res[True][1], res[False][1])
    print(f"{'разница':>10}" + ''.join(f'{x:>+12.4f}' for x in d))
    print(f"{'значим.':>10}"
          + ''.join(f'{(abs(x)/y if y > 1e-9 else float("nan")):>12.1f}'
                    for x, y in zip(d, er)))
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
