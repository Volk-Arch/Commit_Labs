#!/usr/bin/env python3
"""Отношение Клаузиуса на локальном горизонте: путь Якобсона.

ПОВОД. Выбор расстановки (динамика) у нас не решён: exp(-beta*S) даёт
неверный знак, exp(iS) глобально не считается. Но Якобсон (1995) выводит
уравнение Эйнштейна БЕЗ интеграла по историям — из одного отношения
Клаузиуса

    dQ = T dS,     T = Unruh,   S = eta * A

потребованного на ВСЕХ локальных горизонтах Риндлера. Если это
отношение выполняется, уравнение Эйнштейна следует, и никакой меры на
расстановках не нужно.

У нас для этого уже есть всё: энтропия горизонта по числу СВЯЗЕЙ через
него (Доу-Соркин, показатель 2.286 +- 0.060) и энергия как длина цепи.

ЧТО МЕРИМ. Горизонт — граница прошлого далёкого события q. Площадь —
число связей, пересекающих её наружу: (a,b) со a < b, a в past(q),
b вне past(q). Пропускаем через горизонт трубку из M событий и смотрим
приращение.

    ПРЕДСКАЗАНИЕ ЯКОБСОНА:  dA линейно по M,
    коэффициент не зависит от выбора горизонта.

Если dA насыщается по M, как насыщались вмятина, масса и сила, — путь
Якобсона здесь тоже не работает. Это счёт связей, а не градиент, то есть
величина, которая у нас работает.

КОНТРОЛЬ: те же M событий, рассыпанные по миру (не пересекающие
горизонт направленно).

  python3 jacobson.py
  python3 jacobson.py --masses 30 60 120 240 480
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
    return np.vstack(out)[:N]


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


def links_of(R):
    A = np.asarray(R, np.float32)
    return np.asarray(R & ((A @ A) == 0), bool)


def tube_at(M, rad, T, rs, t0, x0, where='cross', qt=1.45):
    """Трубка у точки (t0, x0). where: cross / inside / outside.

    ПОЗИЦИОННЫЙ КОНТРОЛЬ. Приращение площади может быть бухгалтерией:
    добавил события у горизонта — получил больше связей. Поэтому та же
    трубка ставится целиком внутрь прошлого q и целиком снаружу. Если
    dA велико только при ПЕРЕСЕЧЕНИИ, это физика.
    """
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = np.clip(t0 + (rs.random(M) - 0.5) * 0.5 * T, 0.02, T - 0.02)
    c = np.array(x0, float)
    if where == 'inside':
        c = c * 0.35                     # ближе к оси: глубоко в прошлом q
    elif where == 'outside':
        c = c * 1.55                     # дальше: вне прошлого q
    X[:, 0] = np.clip(X[:, 0], 0.02, T - 0.02)
    X[:, 1:] = c[None, :] + u * r[:, None]
    ok = np.linalg.norm(X[:, 1:], axis=1) < np.minimum(X[:, 0], T - X[:, 0])
    return X[ok]


def tube_through(M, rad, T, rs, t0, x0):
    """Трубка, идущая через точку (t0, x0) вдоль времени."""
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = np.clip(t0 + (rs.random(M) - 0.5) * 0.5 * T, 0.02, T - 0.02)
    X[:, 1:] = x0[None, :] + u * r[:, None]
    ok = np.linalg.norm(X[:, 1:], axis=1) < np.minimum(X[:, 0], T - X[:, 0])
    return X[ok]


def horizon_area(X, T, q_t, patch_c, patch_r):
    """Число связей, пересекающих границу прошлого q наружу, в заплате.

    q берётся как точка на оси в поздний момент: граница её прошлого —
    локальный горизонт, а заплата ограничивает счёт окрестностью.
    """
    R = cone_rel(X)
    L = links_of(R)
    # прошлое q: события, причинно предшествующие точке (q_t, 0)
    dt = q_t - X[:, 0]
    dx = np.linalg.norm(X[:, 1:], axis=1)
    inside = dt > dx
    # связи изнутри наружу
    cross = L & inside[None, :] & (~inside)[:, None]
    # заплата: оба конца близко к центру заплаты
    d = np.linalg.norm(X[:, 1:] - patch_c[None, :], axis=1)
    near = d < patch_r
    cross = cross & near[None, :] & near[:, None]
    return int(cross.sum()), int(inside.sum())


def main():
    ap = argparse.ArgumentParser(description='Клаузиус на горизонте')
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--qt', type=float, default=1.6)
    ap.add_argument('--tp', type=float, default=1.1,
                    help='время заплаты; обязано быть > qt/2, '
                         'иначе горизонт выходит за ромб')
    ap.add_argument('--rad', type=float, default=0.05)
    ap.add_argument('--patch', type=float, default=0.45)
    ap.add_argument('--masses', type=int, nargs='+',
                    default=[30, 60, 120, 240, 480])
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    # центр заплаты — на горизонте: там, где |x| = qt - t при t = qt/2
    # радиус горизонта в момент tp равен qt - tp; ромб даёт
    # min(tp, T-tp). Нужно qt-tp < min(tp,T-tp), то есть tp > qt/2.
    rh = a.qt - a.tp
    rmax = min(a.tp, a.t - a.tp)
    pc = np.array([rh, 0.0, 0.0])
    print(f'ромб T={a.t}, N={a.n}, горизонт = граница прошлого точки '
          f'(t={a.qt}, 0)')
    print(f'заплата радиуса {a.patch} вокруг |x|={rh:.3f} '
          f'при t={a.tp}; ромб там даёт {rmax:.3f}')
    if rh >= rmax:
        print('  ОШИБКА ГЕОМЕТРИИ: горизонт вне ромба, '
              'нужно tp > qt/2'); return
    print('ПРЕДСКАЗАНИЕ ЯКОБСОНА: dA линейно по M')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'M':>6}{'ПЕРЕСЕКАЕТ':>17}{'внутри':>17}"
          f"{'снаружи':>17}{'рассыпан':>17}")

    base = {}
    rows = []
    for M in a.masses:
        kinds = ['cross', 'inside', 'outside', 'scatter']
        acc = {kk: [] for kk in kinds}
        for t in range(a.seeds):
            rs = np.random.default_rng(301000 + 977 * t)
            bg = diamond(a.n, a.t, rs)
            A0, nin = horizon_area(bg, a.t, a.qt, pc, a.patch)
            for kk in kinds:
                if kk == 'scatter':
                    ob = diamond(M, a.t, rs)
                else:
                    ob = tube_at(M, a.rad, a.t, rs, a.tp, pc, kk, a.qt)
                if len(ob) == 0:
                    continue
                A1, _ = horizon_area(np.vstack([ob, bg]), a.t, a.qt,
                                     pc, a.patch)
                acc[kk].append(A1 - A0)
        if not acc['cross']:
            continue
        g = lambda kk: (float(np.mean(acc[kk])) if acc[kk] else float('nan'),
                        float(np.std(acc[kk]) / math.sqrt(max(len(acc[kk]), 1)))
                        if len(acc[kk]) > 1 else float('nan'))
        cells = [g(kk) for kk in kinds]
        d = cells[0][0] - cells[1][0]
        er = math.hypot(cells[0][1], cells[1][1])
        rows.append((M, d, er))
        print(f'{M:>6}' + ''.join(f'{m:>10.1f} ±{e:<6.1f}'
                                  for m, e in cells), flush=True)
        print(f"{'пересек минус внутри:':>28} {d:>8.1f} ±{er:<6.1f}"
              f"  {abs(d)/max(er,1e-9):>5.1f}σ   на событие {d/M:>7.3f}")

    if len(rows) >= 3:
        Ms = np.array([r[0] for r in rows], float)
        Ds = np.array([r[1] for r in rows], float)
        ok = np.abs(Ds) > 1e-9
        if ok.sum() >= 3:
            p = np.polyfit(np.log(Ms[ok]), np.log(np.abs(Ds[ok])), 1)[0]
            print()
            print(f'  показатель (пересек минус внутри) по массе: {p:.3f}')
            print(f'  Якобсон требует 1.000; насыщение дало бы ~0')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
