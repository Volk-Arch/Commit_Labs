#!/usr/bin/env python3
"""НЕСЁТ ЛИ РАЗРЕЖЕНИЕ ТОПОЛОГИЧЕСКИЙ ЗАРЯД.

СХОДЯТСЯ ДВЕ ВЕТКИ ДНЯ.
  delay-doc.md:  масса есть РАЗРЕЖЕНИЕ — замедляет часы (4.5σ) и
                 выгибает цепи наружу (5.0σ), плавно по глубине.
  charge-doc.md: индекс директорного поля есть сохраняющийся заряд —
                 квантован точно, масштабно-инвариантен (корр. 0.888),
                 дефекты рождаются парами (дисперсия 0.452 от
                 пуассоновской, 5.1σ).

Если это одно и то же, вещь получает сразу и массу, и защиту от
исчезновения, причём обе из одного порядка.

ПРЕДСКАЗАНИЕ, если разрежение-трубка есть дефект: это ДИСКЛИНАЦИОННАЯ
ЛИНИЯ. Значит петля, ОХВАТЫВАЮЩАЯ трубку, обязана давать стойкий
ненулевой индекс, а петля того же радиуса РЯДОМ — нуль.

ВСТРОЕННАЯ ЛОВУШКА, названная заранее. Разрежение само ухудшает выборку
поля, а хуже выборка — больше ложных дефектов (у случайного поля их
0.3406 против 0.0212). Контроль «удалено столько же случайно» от этого
не спасает: порча локальна. Поэтому положительным считается результат
ТОЛЬКО при трёх условиях разом:

  1. избыток индекса у охватывающих петель есть;
  2. НЕОДНОЗНАЧНОСТЬ рёбер не выросла (иначе это шум выборки);
  3. индекс ВЫЖИВАЕТ при огрублении (вдвое меньше шагов по петле).

Плюс два мира сравнения: без дырки вовсе и с удалением того же числа
событий случайно по ромбу.

  python3 holecharge.go.py
  python3 holecharge.go.py --rad 0.24 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel, axis_of


def unwrap_pi(d):
    return (d + math.pi / 2) % math.pi - math.pi / 2


def ring_index(X, R, i0, cand, ctr, rho, nsteps, minsize, projmin,
               e1, e2, tmid):
    """Индекс петли радиуса rho вокруг точки ctr в плоскости (e1,e2).

    Возвращает (индекс, максимальная неоднозначность, доля пропусков).
    """
    vals, miss = [], 0
    for s in range(nsteps):
        th = 2 * math.pi * s / nsteps
        tgt = np.zeros(4)
        tgt[0] = tmid
        tgt[1:] = ctr + rho * (math.cos(th) * e1 + math.sin(th) * e2)
        d = np.linalg.norm(X[cand][:, 1:] - tgt[1:], axis=1)
        d = d + 3.0 * np.abs(X[cand][:, 0] - tmid)
        got = None
        for idx in np.argsort(d)[:18]:
            j = int(cand[idx])
            ax = axis_of(X, R, i0, j, minsize)
            if ax is None:
                continue
            p = np.array([ax[1:] @ e1, ax[1:] @ e2])
            if np.linalg.norm(p) < projmin:
                continue
            got = math.atan2(p[1], p[0])
            break
        if got is None:
            miss += 1
            vals.append(None)
        else:
            vals.append(got)
    if miss > 0:
        return None, None, miss / nsteps
    tot, amb = 0.0, 0.0
    for t in range(nsteps):
        dd = unwrap_pi(vals[(t + 1) % nsteps] - vals[t])
        tot += dd
        amb = max(amb, abs(dd) / (math.pi / 2))
    return tot / (2 * math.pi), amb, 0.0


def main():
    ap = argparse.ArgumentParser(description='Заряд разрежения')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--rhos', type=float, nargs='+',
                    default=[0.32, 0.42, 0.55])
    ap.add_argument('--nsteps', type=int, default=24)
    ap.add_argument('--minsize', type=int, default=25)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    print(f'ромб T={a.t}, N={a.n}; трубка-разрежение радиуса {a.rad} '
          f'вокруг оси')
    print(f'петли в плоскости (x1,x2) при t≈T/2, шагов {a.nsteps}')
    print('ОХВАТ = петля вокруг оси; РЯДОМ = та же петля, сдвинутая '
          'на 2*rho')
    print('предсказание дисклинации: охват даёт ±1/2, рядом нуль')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'мир':>14}{'петля':>8}{'rho':>6}{'годных':>8}{'<индекс>':>11}"
          f"{'±':>8}{'|инд|>=1/2':>12}{'неоднозн':>10}"
          f"{'огрубл. совп':>14}")

    for world in ('дырка', 'без дырки', 'случайно удалено'):
        for rho in a.rhos:
            for mode in ('охват', 'рядом'):
                A, AM, RB = [], [], []
                for t in range(a.seeds):
                    rs = np.random.default_rng(3201000 + 977 * t)
                    X0 = diamond(a.n, a.t, rs)
                    rr = np.linalg.norm(X0[:, 1:], axis=1)
                    keep = rr >= a.rad
                    k = int((~keep).sum())
                    if world == 'дырка':
                        X = X0[keep]
                    elif world == 'без дырки':
                        X = X0
                    else:
                        ii = rs.choice(len(X0), len(X0) - k, replace=False)
                        X = X0[np.sort(ii)]
                    R = cone_rel(X)
                    rx = np.linalg.norm(X[:, 1:], axis=1)
                    early = (X[:, 0] < 0.22 * a.t) & (rx > a.rad)
                    if not early.any():
                        del R, X; continue
                    i0 = int(np.argmax(R.sum(0) * early))
                    cand = np.flatnonzero(R[:, i0])
                    if len(cand) < 200:
                        del R, X; continue
                    ctr = (np.zeros(3) if mode == 'охват'
                           else 2.0 * rho * e1)
                    v, am, _ = ring_index(X, R, i0, cand, ctr, rho,
                                          a.nsteps, a.minsize, a.projmin,
                                          e1, e2, 0.5 * a.t)
                    v2, _, _ = ring_index(X, R, i0, cand, ctr, rho,
                                          a.nsteps // 2, a.minsize,
                                          a.projmin, e1, e2, 0.5 * a.t)
                    del R, X
                    if v is None:
                        continue
                    A.append(v); AM.append(am)
                    if v2 is not None:
                        RB.append(abs(v - v2) < 1e-6)
                if not A:
                    print(f'{world:>14}{mode:>8}{rho:>6.2f}{"нет":>8}')
                    continue
                Aa = np.array(A)
                se = Aa.std() / math.sqrt(len(Aa))
                print(f'{world:>14}{mode:>8}{rho:>6.2f}{len(Aa):>8}'
                      f'{Aa.mean():>+11.4f}{se:>8.4f}'
                      f'{(np.abs(np.round(Aa*2)/2) >= 0.5).mean():>12.3f}'
                      f'{np.mean(AM):>10.3f}'
                      f'{(np.mean(RB) if RB else float("nan")):>14.3f}',
                      flush=True)
        print()

    print('  положительно ТОЛЬКО если: охват даёт ненулевой индекс,')
    print('  рядом нуль, неоднозначность НЕ выше, и огрубление совпадает')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
