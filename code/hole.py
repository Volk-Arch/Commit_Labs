#!/usr/bin/env python3
"""ВЕЩЬ КАК ДЫРКА: задержка от обхода, а не от добавления.

ЦЕПОЧКА РАССУЖДЕНИЯ, вся из измерений этого дня.

1. Его идея: «вещь — это не то, что добавлено, а место, где порядок
   вынужден сделать лишний шаг, чтобы остаться самим собой».
2. delay.py померил ход часов h / V^(1/4) у трубки и получил ОБРАТНЫЙ
   знак: +0.32, то есть часы у массы идут БЫСТРЕЕ.
3. scratchpad/ladder.py нашёл причину: внутренняя высота трубки 57/78/105
   против фоновых 11.8 (отношение 4.85/6.66/8.94). У цепи длина растёт
   как lam*tau, у четырёхмерного засева как m4*rho^(1/4)*tau. Корень
   четвёртой степени и решает: сгусток в линию есть ЛЕСТНИЦА, не груз.
4. Значит «не то, что добавлено» — следствие замера, а не догадка.
5. И отношение h / V^(1/4) к плотности СЛЕПО: и h, и V тянутся как
   rho^(1/4), ratio не меняется. Оно чувствует только ФОРМУ.

Отсюда единственное, что может дать задержку: препятствие, вокруг
которого цепь обязана идти в обход. Дырка.

ЧТО МЕРИМ. Из засева вырезается трубка — область радиуса rad вокруг
оси, где событий нет. Контроль: удалено СТОЛЬКО ЖЕ событий случайно по
всему ромбу (совпадает счёт, не совпадает форма). Затем:

  а) ход часов h / V^(1/4): есть ли задержка у дырки;
  б) градиент действия: тянет ли к дырке.

ПРЕДСКАЗАНИЕ. Если идея верна, у дырки часы отстают И она притягивает.
Это же объясняло бы, почему связанный ансамбль требовал ОБЕДНЕНИЯ на
месте трубки (6.1 сигмы, weight-doc.md): действие хотело там массу, а
мы считали это болезнью знака.

  python3 hole.py
  python3 hole.py --rads 0.10 0.16 0.24 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel
from mc import World
from delay import heights


def carve(X, rad):
    """Вырезать трубку вокруг оси: вернуть маску оставшихся."""
    r = np.linalg.norm(X[:, 1:], axis=1)
    return r >= rad


def in_diamond(x, T):
    r = np.linalg.norm(x[1:])
    return (x[0] >= r) and (T - x[0] >= r)


def clock_profile(X, T, edges):
    """Остаток log h после снятия зависимости от log V, по радиусу."""
    R = cone_rel(X)
    h = heights(R).astype(np.float64)
    V = R.sum(1).astype(np.float64)
    keep = (V >= 40) & (V <= 600) & (h > 0)
    if keep.sum() < 60:
        return np.full(len(edges) - 1, np.nan)
    lv, lh = np.log(V[keep]), np.log(h[keep])
    resid = lh - np.polyval(np.polyfit(lv, lh, 1), lv)
    r = np.linalg.norm(X[keep][:, 1:], axis=1)
    out = []
    for b in range(len(edges) - 1):
        m = (r >= edges[b]) & (r < edges[b + 1])
        out.append(float(resid[m].mean()) if m.sum() > 5 else np.nan)
    return np.array(out)


def force_profile(X, T, eps, delta, rs, edges, nsamp=200):
    """Радиальный градиент действия по радиусу; <0 значит к оси."""
    w = World(X, eps)
    r = np.linalg.norm(X[:, 1:], axis=1)
    cand = np.flatnonzero((np.abs(X[:, 0] - T / 2) < 0.3 * T)
                          & (r > edges[0]) & (r < edges[-1]))
    if len(cand) > nsamp:
        cand = rs.choice(cand, nsamp, replace=False)
    acc = [[] for _ in range(len(edges) - 1)]
    for i in cand:
        i = int(i)
        x0 = w.X[i].copy()
        rr = np.linalg.norm(x0[1:])
        if rr < 1e-6:
            continue
        e = np.zeros(4)
        e[1:] = x0[1:] / rr
        xp, xm = x0 + delta * e, x0 - delta * e
        if not (in_diamond(xp, T) and in_diamond(xm, T)):
            continue
        st = w.move(i, xp); sp = w.S; w.undo(st)
        st = w.move(i, xm); sm = w.S; w.undo(st)
        b = int(np.digitize(rr, edges) - 1)
        if 0 <= b < len(acc):
            acc[b].append(-(sp - sm) / (2 * delta))
    del w
    return np.array([np.mean(v) if len(v) > 2 else np.nan for v in acc])


def main():
    ap = argparse.ArgumentParser(description='Вещь как дырка')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rads', type=float, nargs='+', default=[0.10, 0.16, 0.24])
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--delta', type=float, default=0.06)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    ced = np.array([0.0, 0.18, 0.30, 0.45, 0.65, 0.90])
    print(f'ромб T={a.t}, из засева ВЫРЕЗАНА трубка вокруг оси')
    print('контроль: удалено столько же событий случайно по всему ромбу')
    print('ход часов = остаток log h после снятия log V; '
          'ОТРИЦАТЕЛЬНЫЙ у дырки = ЗАДЕРЖКА')
    print('сила F_r < 0 = к оси, ПРИТЯЖЕНИЕ')
    print(f'засевов {a.seeds}')

    for rad in a.rads:
        print(f'\nРАДИУС ДЫРКИ {rad}')
        print(f"{'':>12}" + ''.join(f'{e:>12.2f}' for e in ced[1:]))
        res = {}
        nrem = []
        for mk in (True, False):
            C, F = [], []
            for t in range(a.seeds):
                rs = np.random.default_rng(2201000 + 977 * t)
                X0 = diamond(a.t, rs)
                keep = carve(X0, rad)
                k = int((~keep).sum())
                if mk:
                    X = X0[keep]
                else:
                    idx = rs.choice(len(X0), len(X0) - k, replace=False)
                    X = X0[np.sort(idx)]
                nrem.append(k)
                C.append(clock_profile(X, a.t, ced))
                F.append(force_profile(X, a.t, a.eps, a.delta, rs, ced))
            C = np.array(C); F = np.array(F)
            res[mk] = (np.nanmean(C, 0), np.nanstd(C, 0) / math.sqrt(len(C)),
                       np.nanmean(F, 0), np.nanstd(F, 0) / math.sqrt(len(F)))
            nm = 'дырка' if mk else 'контроль'
            print(f'{nm+" часы":>12}' + ''.join(f'{x:>+12.5f}'
                                                for x in res[mk][0]))
            print(f'{nm+" сила":>12}' + ''.join(f'{x:>+12.4f}'
                                                for x in res[mk][2]))
        print(f'   удалено событий: {np.mean(nrem):.0f}')
        for j, nm in ((0, 'часы'), (2, 'сила')):
            d = res[True][j] - res[False][j]
            er = np.hypot(res[True][j + 1], res[False][j + 1])
            fmt = '{:>+12.5f}' if j == 0 else '{:>+12.4f}'
            print(f'{"разн. "+nm:>12}' + ''.join(fmt.format(x) for x in d))
            print(f'{"значим.":>12}'
                  + ''.join(f'{(abs(x)/y if y > 1e-12 else float("nan")):>12.1f}'
                            for x, y in zip(d, er)), flush=True)
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
