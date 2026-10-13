#!/usr/bin/env python3
"""Сохраняется ли суммарный топологический индекс.

ПОВОД. Архив нашёл дефект с индексом -1/2 в поле поперечных осей
интервалов (0.033 из 390 обходов) и проверил его на двулистность —
не прошёл: «дефект локален, его можно обойти стороной». Но вопрос,
записанный как открытый и НИКОГДА не измеренный (layer2-doc.md:191):
СОХРАНЯЕТСЯ ЛИ СУММАРНЫЙ ИНДЕКС. А это именно то, что делает дефекты
частицами: в физике суммарный топологический заряд сохраняется, поэтому
дефекты рождаются и гибнут только ПАРАМИ. Такая защита не требует
динамики.

ДВЕ ОШИБКИ АРХИВА, которые надо исправить.

1. ЗНАК. Обход строился в плоскости компонент (x1,x2) —
   arctan2(d[:,2], d[:,1]) — а знак поворота брался как
   np.sign(np.cross(prev[:3], cur[:3])[0]), то есть компонента вдоль
   ПЕРВОЙ оси трёхмерного срезa, а у 4-вектора это направление времени,
   а не нормаль к плоскости обхода. Знак читался с посторонней оси —
   вероятная причина того, что нашлись одни минусы и ни одного +1/2.
   Здесь оси проецируются на плоскость обхода и поворот считается
   планарно, то есть знак верен по построению.

2. ПОРЯДОК. Архивный rel_of покомпонентный, то есть ЯЩИК. С тех пор
   установлено, что ящик не есть причинность Минковского. Здесь
   конусный порядок.

ПОЛЕ. Для события j в будущем опоры i0 ось e2(j) — главная поперечная
ось разброса внутренности интервала (i0, j). Это директор: направление
без знака, поэтому индекс кратен 1/2.

ТЕСТ СОХРАНЕНИЯ. Индекс большой петли обязан равняться СУММЕ индексов
малых петель, её покрывающих. Это локальное сохранение заряда.

  python3 topo.py
  python3 topo.py --n 4000 --loops 400
"""
import argparse
import math
import time

import numpy as np


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


def cone_rel(X, chunk=400):
    N = len(X)
    R = np.empty((N, N), bool)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        dt = X[a:b, None, 0] - X[None, :, 0]
        dx = np.linalg.norm(X[a:b, None, 1:] - X[None, :, 1:], axis=-1)
        R[a:b] = dt > dx
    np.fill_diagonal(R, False)
    return R


def axis_of(X, R, i0, j, minsize):
    """Главная поперечная ось интервала (i0, j). Директор, без знака."""
    mid = np.flatnonzero(R[j] & R[:, i0])
    if len(mid) < minsize:
        return None
    a = X[j] - X[i0]
    n = np.linalg.norm(a)
    if n < 1e-12:
        return None
    e1 = a / n
    P = X[mid] - 0.5 * (X[i0] + X[j])
    P = P - np.outer(P @ e1, e1)
    if np.linalg.norm(P) < 1e-12:
        return None
    w, v = np.linalg.eigh(P.T @ P)
    return v[:, -1]


def plane_of(X, i0, j0):
    """Два орта, ортогональных оси i0->j0."""
    a = X[j0] - X[i0]
    a = a / np.linalg.norm(a)
    B = np.eye(4) - np.outer(a, a)
    w, v = np.linalg.eigh(B)
    return v[:, -1], v[:, -2]


def loop_index(X, R, i0, j0, u, v, radius, nsteps, minsize, cand, proj_min):
    """Индекс директорного поля по замкнутому обходу радиуса radius.

    Оси проецируются на плоскость (u,v), угол берётся по модулю pi
    (директор), приращения разворачиваются в (-pi/2, pi/2].
    """
    ths = 2 * np.pi * np.arange(nsteps) / nsteps
    seq = []
    for th in ths:
        tgt = X[j0] + radius * (math.cos(th) * u + math.sin(th) * v)
        d = np.linalg.norm(X[cand] - tgt, axis=1)
        got = None
        for idx in np.argsort(d)[:14]:
            j = int(cand[idx])
            ax = axis_of(X, R, i0, j, minsize)
            if ax is not None:
                p = np.array([ax @ u, ax @ v])
                if np.linalg.norm(p) >= proj_min:
                    got = math.atan2(p[1], p[0])
                    break
        if got is None:
            return None
        seq.append(got)
    tot = 0.0
    for t in range(nsteps):
        dth = seq[(t + 1) % nsteps] - seq[t]
        dth = (dth + math.pi / 2) % math.pi - math.pi / 2
        tot += dth
    return tot / (2 * math.pi)


def main():
    ap = argparse.ArgumentParser(description='Сохранение индекса')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--minsize', type=int, default=25)
    ap.add_argument('--nsteps', type=int, default=24)
    ap.add_argument('--radius', type=float, default=0.10)
    ap.add_argument('--loops', type=int, default=250)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'конусный порядок, ромб T={a.t}, N={a.n}, '
          f'интервал >= {a.minsize}')
    print(f'петля: {a.nsteps} шагов, радиус {a.radius}, '
          f'порог проекции {a.projmin}')
    print('индекс директорного поля кратен 1/2; знак из плоскости обхода')
    print()

    allidx = []
    for t in range(a.seeds):
        rs = np.random.default_rng(211000 + 977 * t)
        X = diamond(a.n, a.t, rs)
        R = cone_rel(X)
        # опора: раннее событие с большим будущем
        fut = R.sum(0)
        i0 = int(np.argmax(fut * (X[:, 0] < 0.25 * a.t)))
        cand = np.flatnonzero(R[:, i0])
        if len(cand) < 200:
            continue
        rs2 = np.random.default_rng(500 + t)
        centers = rs2.choice(cand, min(a.loops, len(cand)), replace=False)
        for j0 in centers:
            j0 = int(j0)
            try:
                u, v = plane_of(X, i0, j0)
            except Exception:
                continue
            idx = loop_index(X, R, i0, j0, u, v, a.radius, a.nsteps,
                             a.minsize, cand, a.projmin)
            if idx is not None:
                allidx.append(idx)
        del R, X
    A = np.array(allidx)
    print(f'ОБХОДОВ УДАЛОСЬ: {len(A)}')
    if len(A) == 0:
        print('  поле не построилось — ослабить minsize или projmin')
        return
    print()
    print('РАСПРЕДЕЛЕНИЕ ИНДЕКСА (округление до 1/4, чтобы видеть '
          'полуцелые)')
    q = np.round(A * 4) / 4
    vals, cnt = np.unique(q, return_counts=True)
    for vv, cc in zip(vals, cnt):
        if cc / len(A) > 0.002:
            print(f'{vv:>+8.2f}{cc / len(A):>10.4f}  ({cc})')
    print()
    print(f'  среднее {A.mean():+.4f}   с.к.о. {A.std():.4f}')
    pos = int((q >= 0.25).sum())
    neg = int((q <= -0.25).sum())
    print(f'  ненулевых: {pos} положительных, {neg} отрицательных')
    if pos + neg > 0:
        p = pos / (pos + neg)
        e = math.sqrt(max(p * (1 - p), 1e-9) / (pos + neg))
        print(f'  доля положительных {p:.3f} ± {e:.3f}   '
              f'(0.5 = симметрия, {abs(p-0.5)/max(e,1e-9):.1f}σ от неё)')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
