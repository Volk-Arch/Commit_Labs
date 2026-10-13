#!/usr/bin/env python3
"""ПЛОТНЫЕ интервалы: локальность через собственную размерность интервала.

ПУТЬ (каждый шаг исправлял предыдущий).

`locality.py`  связи светоподобны: собственное время ~ N^-0.2095 при
               ell ~ N^-0.25, а |Δx| ~ N^-0.0160 стоит на месте.
               Четвёрный цикл имеет поперечник 0.62 в ромбе радиуса 1.0
               при ЛЮБОМ N. Холономия по нему глобальна — потому
               перемешанный контроль весь день и выходил ей равен.
`interval.py`  ограничение интервалом помогает, но не доезжает:
               поперечник N^-0.0758, площадь N^-0.2413 против локальных
               −0.25 и −0.50. Площадь идёт как ell^1, поперечник как
               ell^0.3 — значит обход есть тонкая ЩЕПКА вдоль конуса.

ПРИЧИНА. Среди пар с заданным числом событий между ними подавляюще много
почти светоподобных: при |Δt| ~ |Δx| собственное время мало при любом
удалении, и между далёкими событиями лежит всего десяток. МАЛОЕ ЧИСЛО
СОБЫТИЙ МЕЖДУ — НЕ ЗНАЧИТ БЛИЗКО. Это та же нелокальность, из-за которой
нелокальному д'Аламбертиану нужен промежуточный масштаб.

ЧЕМ РАЗЛИЧИТЬ, НЕ ВЫХОДЯ ИЗ ПОРЯДКА. Собственной размерностью интервала,
то есть инструментом самого проекта. У плотного четырёхмерного интервала
самая длинная цепь h ~ n^(1/4); у светоподобной щепки h ~ n, она почти и
есть цепь. Признак чисто порядковый: ни координат, ни расстояний.

ЗАМЕР. Лестница N, пары с числом событий интервала в фиксированной
полосе, у каждой считается h. Деление на плотные (h мало при том же n) и
щепки (h велико). Поперечник и площадь — показателями по N в каждой
группе.

ПРЕДСКАЗАНИЯ, расходящиеся по знаку:
    плотные  ~ N^-0.25 (как ell) — локальность в порядке ЕСТЬ, и её
             достаёт собственная размерность интервала;
    щепки    ~ N^0 — они и портили все прежние замеры.
Проверка координатами: Δt/|Δx| у плотных должно быть заметно больше 1, у
щепок около 1. Координаты здесь только для ПРОВЕРКИ отбора, сам отбор их
не видит.

  python3 fatint.py
  python3 fatint.py --ns 400 800 1600 3200 6400 --card 12 20 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from response import warped_diamond


def longest(Rs):
    """Самая длинная цепь (число событий) в малом подмножестве."""
    n = len(Rs)
    order = np.argsort(Rs.sum(1), kind='stable')
    h = np.ones(n, np.int64)
    for i in order:
        i = int(i)
        pre = np.flatnonzero(Rs[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return int(h.max())


def diam_area(P):
    """Пространственный поперечник и площадь набора событий."""
    d = 0.0
    for p in range(len(P)):
        dd = np.linalg.norm(P[p + 1:, 1:] - P[p, 1:], axis=1)
        if len(dd):
            d = max(d, float(dd.max()))
    c = P[:, 1:3] - P[:, 1:3].mean(0)
    ang = np.argsort(np.arctan2(c[:, 1], c[:, 0]))
    q = c[ang]
    ar = abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) -
                          q[:, 1] * np.roll(q[:, 0], -1)))) / 2
    return d, ar


def one(N, T, clo, chi, npair, seed):
    """По каждой паре: n, h, поперечник, площадь, Δt/|Δx|."""
    rs = np.random.default_rng(7901000 + 977 * seed)
    X = warped_diamond(N, T, 0.0, 0.45, rs)
    R = cone_rel(X)                      # R[i,j] = «j раньше i»
    A = R.astype(np.float32)
    INT = A @ A                          # INT[i,j] = |I(j,i)|
    rel = np.argwhere(R)
    if len(rel) < 50:
        return None
    card = INT[rel[:, 0], rel[:, 1]]
    ok = np.flatnonzero((card >= clo) & (card <= chi))
    if len(ok) < 8:
        return None
    ok = rs.choice(ok, min(npair, len(ok)), replace=False)
    out = []
    for p in ok:
        i, j = int(rel[p, 0]), int(rel[p, 1])
        mem = np.flatnonzero(R[:, j] & R[i, :])
        sub = np.concatenate([[j], mem, [i]])
        if len(sub) < 6:
            continue
        h = longest(R[np.ix_(sub, sub)])
        d, ar = diam_area(X[sub])
        dt = abs(X[i, 0] - X[j, 0])
        dx = float(np.linalg.norm(X[i, 1:] - X[j, 1:]))
        out.append((len(sub), h, d, ar, dt / max(dx, 1e-9)))
    del R, A, INT, X
    if len(out) < 20:
        return None
    vol = math.pi / 24.0 * T ** 4
    return np.array(out), float((vol / N) ** 0.25)


def fit(ns, vals):
    ok = [k for k in range(len(vals))
          if vals[k] is not None and np.isfinite(vals[k]) and vals[k] > 1e-14]
    if len(ok) < 3:
        return float('nan'), float('nan')
    x = np.log([ns[k] for k in ok])
    y = np.log([vals[k] for k in ok])
    b, c = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    return float(b), (float(1 - ((y - b * x - c) ** 2).sum() / ss)
                      if ss > 1e-18 else 1.0)


def main():
    ap = argparse.ArgumentParser(description='Плотные интервалы')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[400, 800, 1600, 3200])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--card', type=int, nargs=2, default=[12, 20])
    ap.add_argument('--npair', type=int, default=700)
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    clo, chi = a.card
    nm = 0.5 * (clo + chi)
    print(f'лестница N = {a.ns}, ромб T={a.t}')
    print(f'интервалы с числом событий {clo}..{chi} при всех N, '
          f'засевов {a.seeds}')
    print(f'плотный четырёхмерный ожидает h ~ n^(1/4) ~ '
          f'{nm**0.25:.2f}..{2.5*nm**0.25:.1f}, щепка h ~ n ~ {nm:.0f}')
    print('отбор ЧИСТО ПОРЯДКОВЫЙ: число событий и длина цепи.')
    print('Δt/|Δx| показан только для проверки отбора')
    print()

    big = {}
    ell = []
    for N in a.ns:
        parts = []
        for t in range(a.seeds):
            r = one(N, a.t, clo, chi, a.npair, t)
            if r:
                parts.append(r[0])
                e = r[1]
        if not parts:
            print(f'{N:>7}  мало данных')
            continue
        big[N] = np.vstack(parts)
        ell.append(e)

    if len(big) < 3:
        print('  мало точек лестницы')
        return

    hs = np.concatenate([M[:, 1] for M in big.values()])
    cut = float(np.median(hs))
    print(f'  длина цепи h: медиана {cut:.1f}, разброс '
          f'{hs.min():.0f}..{hs.max():.0f}  (порог деления — медиана)')
    print()
    print(f"{'N':>7}{'ell':>9}" +
          f"{'плотн. попереч':>16}{'плотн. площадь':>16}{'плотн. Δt/Δx':>14}"
          f"{'щепка попереч':>15}{'щепка площадь':>15}{'щепка Δt/Δx':>13}"
          f"{'пар':>7}")
    cols = {k: [] for k in ('dp', 'ap', 'dt', 'ds', 'as', 'ts')}
    for N in a.ns:
        if N not in big:
            for k in cols:
                cols[k].append(None)
            continue
        M = big[N]
        f_, s_ = M[M[:, 1] <= cut], M[M[:, 1] > cut]
        if len(f_) < 20 or len(s_) < 20:
            for k in cols:
                cols[k].append(None)
            print(f'{N:>7}  группы малы ({len(f_)}/{len(s_)})')
            continue
        v = (f_[:, 2].mean(), f_[:, 3].mean(), f_[:, 4].mean(),
             s_[:, 2].mean(), s_[:, 3].mean(), s_[:, 4].mean())
        for k, x in zip(('dp', 'ap', 'dt', 'ds', 'as', 'ts'), v):
            cols[k].append(float(x))
        e = (math.pi / 24.0 * a.t ** 4 / N) ** 0.25
        print(f'{N:>7}{e:>9.4f}{v[0]:>16.5f}{v[1]:>16.5f}{v[2]:>14.2f}'
              f'{v[3]:>15.5f}{v[4]:>15.5f}{v[5]:>13.2f}{len(M):>7}',
              flush=True)

    be, _ = fit(a.ns, [(math.pi / 24.0 * a.t ** 4 / N) ** 0.25
                       for N in a.ns])
    print()
    for k, nmz in (('dp', 'плотные поперечник'), ('ap', 'плотные площадь'),
                   ('ds', 'щепки поперечник'), ('as', 'щепки площадь')):
        b, r2 = fit(a.ns, cols[k])
        print(f'  {nmz:<22} ~ N^{b:+.4f}   R^2 = {r2:.4f}')
    print(f'  {"ell":<22} ~ N^{be:+.4f}')
    bdp, _ = fit(a.ns, cols['dp'])
    bds, _ = fit(a.ns, cols['ds'])
    print()
    print(f'  плотные {bdp:+.4f} против локального {be:+.4f}; '
          f'щепки {bds:+.4f}')
    if abs(bdp - be) < 0.08:
        print('  ПЛОТНЫЕ СЖИМАЮТСЯ КАК ДИСКРЕТНОСТЬ => локальность в')
        print('  порядке есть, и достаёт её собственная размерность интервала')
    elif abs(bdp) > abs(bds) + 0.05:
        print('  плотные сжимаются заметно лучше щепок => направление верное,')
        print('  но до локального показателя не доходит')
    else:
        print('  разделение не помогло')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
