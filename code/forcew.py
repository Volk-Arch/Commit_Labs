#!/usr/bin/env python3
"""Сила против ОКНА НЕСРАВНИМОСТИ: чья линейка задаёт профиль.

ПРОВАЛИВШАЯСЯ ПОПЫТКА (forceq.py). Ось q = |past&past|/min(|past|,|past|),
взятая как максимум по событиям объекта, насыщается: максимум выбирает не
пространственно ближайшее событие трубки, а самое сдвинутое по времени из
ещё несравнимых, то есть сидящее у светового конуса, где разнос
sqrt(dx^2-dt^2) обращается в нуль. Все события попали в одну корзину.

ЧТО РАБОТАЕТ. Трубка есть ЦЕПЬ, и события цепи, несравнимые с b,
образуют на ней интервал. Его длина

    W(b) = число событий объекта, несравнимых с b

относительна по построению. Для прямой мировой линии с плотностью lam по
собственному времени окно имеет длину 2r, значит

    W = 2 * lam * r                      (ПРЕДСКАЗАНИЕ, не подгонка)

Проверено: ранговая корреляция с r даёт +0.918...+0.982, наклон
125.6...141.0 против предсказанных 2*120/1.76 = 136.4.

ОТСЮДА НАСТОЯЩИЙ ОПЫТ. W — линейка САМОГО ОБЪЕКТА, в его единицах.
Если менять плотность трубки lam при той же геометрии, то r и W
разъезжаются. Значит можно спросить, чем задан профиль силы:

    профиль совпадает при фиксированном W  ->  линейка объекта,
                                               величина относительна
    профиль совпадает при фиксированном r  ->  линейка ФОНА,
                                               то есть масштаб
                                               дискретности, внешний

Это тот же вопрос «ярлыки есть, величины нет», но теперь измеряемый.

  python3 forcew.py
  python3 forcew.py --masses 60 120 240 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from mc import World
from weight import diamond


def chain(M, rad, T, rs):
    """Трубка как цепь: M событий на оси, упорядоченных по времени."""
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = np.sort(rs.uniform(rad, T - rad, M))
    X[:, 1:] = u * r[:, None]
    return X


def in_diamond(x, T):
    r = np.linalg.norm(x[1:])
    return (x[0] >= r) and (T - x[0] >= r)


def window(R, M):
    """W[b] = число событий объекта, несравнимых с фоновым событием b."""
    rel = R[M:, :M] | R[:M, M:].T
    return (~rel).sum(1).astype(np.float64)


def probe(w, M, T, delta, rs, nsamp=260):
    """Радиальная сила для выборки фоновых событий, с радиусом."""
    N = len(w.X)
    cand = np.arange(M, N)
    rr = np.linalg.norm(w.X[cand, 1:], axis=1)
    keep = (np.abs(w.X[cand, 0] - T / 2) < 0.3 * T) & (rr < 0.55)
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
        out.append((i - M, r, -(sp - sm) / (2 * delta)))
    return out


def binned(vals, keys, edges):
    acc = [[] for _ in range(len(edges) - 1)]
    for k, v in zip(keys, vals):
        b = int(np.digitize(k, edges) - 1)
        if 0 <= b < len(acc):
            acc[b].append(v)
    m = np.array([np.mean(x) if len(x) > 2 else np.nan for x in acc])
    e = np.array([np.std(x) / math.sqrt(len(x)) if len(x) > 2 else np.nan
                  for x in acc])
    return m, e, [len(x) for x in acc]


def main():
    ap = argparse.ArgumentParser(description='Сила против окна')
    ap.add_argument('--t', type=float, default=1.8)
    ap.add_argument('--masses', type=int, nargs='+', default=[60, 120, 240])
    ap.add_argument('--rad', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--delta', type=float, default=0.06)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    red = np.array([0.0, 0.10, 0.18, 0.28, 0.40, 0.55])
    wed = np.array([0.0, 14.0, 26.0, 40.0, 56.0, 80.0])
    print(f'ромб T={a.t}, трубка — цепь на оси радиуса {a.rad}, '
          f'eps={a.eps}, шаг {a.delta}')
    print('W = число событий трубки, несравнимых с событием; '
          'предсказано W = 2*lam*r')
    print('F_r < 0 значит сила К ОСИ, то есть ПРИТЯЖЕНИЕ')
    print(f'засевов {a.seeds}')

    tab = {}
    for M in a.masses:
        lam = M / (a.t - 2 * a.rad)
        print(f'\nМАССА M = {M},  плотность цепи lam = {lam:.1f},  '
              f'предсказано W = {2*lam:.0f}*r')
        res = {}
        slopes = []
        for mk in (True, False):
            rows = []
            for t in range(a.seeds):
                rs = np.random.default_rng(1501000 + 977 * t)
                bg = diamond(a.t, rs)
                ob = (chain(M, a.rad, a.t, rs) if mk
                      else diamond(a.t, rs, n_at_t1=M / a.t ** 4))
                w = World(np.vstack([ob, bg]), a.eps)
                W = window(w.R, len(ob))
                P = probe(w, len(ob), a.t, a.delta, rs)
                del w
                for j, r, f in P:
                    rows.append((r, W[j], f))
                if mk and len(P) > 20:
                    rr = np.array([p[1] for p in P])
                    ww = np.array([W[p[0]] for p in P])
                    slopes.append(np.polyfit(rr, ww, 1)[0])
            if not rows:
                continue
            rows = np.array(rows)
            res[mk] = {
                'r': binned(rows[:, 2], rows[:, 0], red),
                'W': binned(rows[:, 2], rows[:, 1], wed),
            }
        if slopes:
            print(f'   измеренный наклон W/r = {np.mean(slopes):.1f} '
                  f'± {np.std(slopes)/math.sqrt(len(slopes)):.1f}'
                  f'   (предсказано {2*lam:.0f})')
        for ax, ed, lbl in (('r', red, 'радиус'), ('W', wed, 'окно W')):
            print(f'   ось: {lbl}')
            print(f"{'':>12}" + ''.join(f'{e:>12.0f}' if ax == 'W'
                                        else f'{e:>12.2f}' for e in ed[1:]))
            for mk, nm in ((True, 'трубка'), (False, 'контроль')):
                m, e, n = res[mk][ax]
                print(f'{nm:>12}' + ''.join(f'{x:>+12.4f}' for x in m))
                print(f"{'±':>12}" + ''.join(f'{x:>12.4f}' for x in e))
            d = res[True][ax][0] - res[False][ax][0]
            er = np.hypot(res[True][ax][1], res[False][ax][1])
            print(f"{'разница':>12}" + ''.join(f'{x:>+12.4f}' for x in d))
            print(f"{'значим.':>12}"
                  + ''.join(f'{(abs(x)/y if y > 1e-9 else float("nan")):>12.1f}'
                            for x, y in zip(d, er)), flush=True)
            tab[(M, ax)] = d

    print('\nЧЬЯ ЛИНЕЙКА: совпадают ли профили разных масс')
    for ax, lbl in (('r', 'по радиусу (линейка ФОНА)'),
                    ('W', 'по окну W (линейка ОБЪЕКТА)')):
        print(f'  {lbl}')
        for M in a.masses:
            d = tab.get((M, ax))
            if d is None:
                continue
            print(f'{M:>8}' + ''.join(f'{x:>+12.4f}' for x in d))
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
