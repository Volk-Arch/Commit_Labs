#!/usr/bin/env python3
"""ЧАСТИЦЫ: сколько порядка лежит ВНУТРИ историй, а сколько в пересечениях.

ЕГО ИДЕЯ. Рассматривать порядок как уже состоявшиеся частицы: отношение
`A ≺ B` есть не «A причинило B», а след того, что через оба прошла одна
история. «Порядок живёт во времени, мы находим всю цепочку времени — но
нужна ли она нам вся... то, что мы видим, есть не то, что мы думаем.»

ПОЧЕМУ ЭТО ЛОЖИТСЯ ПО ШВУ АРХИВА. Всё, что в проекте работало, — про
ЦЕПИ: часы замедляются вдоль цепи (`delay-doc.md`), цепь выгибается
вокруг разрежения (`hole.py`, `bow.py`), максимальная цепь есть
геодезическая, окно `W = 2λr` мерится вдоль. Всё, что провалилось, — про
ПОЛНОЕ отношение: холономия по циклам (`locality-doc.md`), поле,
киральность, шесть вечерних автоматов (`cells-doc.md`).

ЧТО ИЗ ЭТОГО СЧИТАЕМО. В четырёхмерии цепь КОРОТКА: самая длинная идёт
как `N^(1/4)`. А ширина (наибольшая антицепь) — как `N^(3/4)`. Если
разложить порядок на цепи-истории, то отношений ВНУТРИ истории выходит
примерно `(N/L)·C(L,2) ≈ N·L/2`, а всего их `f·N²/2`, значит доля

    внутри / всего  ~  L / (f·N)  ~  N^(1/4) / N  =  N^(-3/4)

то есть доля порядка, принадлежащая чьей-то собственной истории, должна
падать как `N^(-3/4)`, и почти весь порядок есть НЕ истории, а их
ПЕРЕСЕЧЕНИЯ.

Если так, его догадка получает точную форму: мы думаем, что порядок есть
причинение вдоль историй, а измеримо он почти целиком — запись о том,
кто с кем мог встретиться. И тогда понятно, почему шесть автоматов не
сдвинули форму: они переставляли ранги, то есть трогали ровно ту
крошечную часть, что лежит вдоль цепей.

ЧТО МЕРИМ, лестницей по N в четырёхмерном ромбе:
  L      самая длинная цепь                  ожидание N^(+0.25)
  Wmax   наибольший слой (антицепь снизу)    ожидание N^(+0.75)
  цепей  сколько историй покрывают все события
  доля   отношений внутри историй            ожидание N^(-0.75)
  n0     прямых связей на отношение — сколько порядка НЕ выводится
         транзитивностью, то есть сколько в нём нового сведения

РАЗЛОЖЕНИЕ НА ИСТОРИИ — жадное, первым подходящим: события в порядке
времени, каждое кладётся в ту историю, чей конец ему предшествует.
Разложение не минимальное, и это оговаривается; минимальное даёт теорема
Дилворта (= ширина), и `Wmax` служит ему оценкой снизу.

  python3 particles.py
  python3 particles.py --ns 250 500 1000 2000 4000
"""
import argparse
import math
import time

import numpy as np


def diamond(N, rs):
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, 1, m)
        Y[:, 1:] = rs.uniform(-0.5, 0.5, (m, 3))
        rad = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= rad) & (1 - Y[:, 0] >= rad)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def rel_of(X):
    """R[i,j] = «i раньше j». События уже упорядочены по времени."""
    dt = X[:, 0][None, :] - X[:, 0][:, None]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
    return (dt > 0) & (dt > dx)


def heights(R):
    N = len(R)
    h = np.ones(N, np.int64)
    for i in range(N):            # уже топологический порядок
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return h


def greedy_chains(R):
    """Жадное разложение на истории: первая подходящая. Концы в tails."""
    N = len(R)
    ends, lens = [], []
    for i in range(N):
        put = -1
        for c in range(len(ends)):
            if R[ends[c], i]:
                put = c
                break
        if put < 0:
            ends.append(i)
            lens.append(1)
        else:
            ends[put] = i
            lens[put] += 1
    return np.array(lens, np.int64)


def fit(ns, vals):
    ok = [k for k in range(len(vals)) if vals[k] and vals[k] > 1e-14]
    if len(ok) < 3:
        return float('nan'), float('nan')
    x = np.log([ns[k] for k in ok])
    y = np.log([vals[k] for k in ok])
    b, c = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    return float(b), (float(1 - ((y - b * x - c) ** 2).sum() / ss)
                      if ss > 1e-18 else 1.0)


def main():
    ap = argparse.ArgumentParser(description='Частицы вместо порядка')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[250, 500, 1000, 2000])
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    print(f'четырёхмерный ромб, лестница N = {a.ns}, засевов {a.seeds}')
    print('ОЖИДАНИЯ: цепь ~ N^0.25, ширина ~ N^0.75,')
    print('          доля отношений ВНУТРИ историй ~ N^-0.75')
    print()
    print(f"{'N':>7}{'цепь L':>9}{'Wmax':>8}{'историй':>10}"
          f"{'отношений':>11}{'внутри':>9}{'ДОЛЯ ВНУТРИ':>13}"
          f"{'n0 прям/отн':>13}")

    acc = {k: [] for k in ('L', 'W', 'ch', 'rel', 'ins', 'frac', 'n0')}
    for N in a.ns:
        row = {k: [] for k in acc}
        for s in range(a.seeds):
            X = diamond(N, np.random.default_rng(8100 + 37 * s))
            R = rel_of(X)
            h = heights(R)
            lens = greedy_chains(R)
            A = R.astype(np.float32)
            L = R & ~((A @ A) > 0.5)
            nrel = float(R.sum())
            ins = float((lens * (lens - 1) / 2).sum())
            row['L'].append(float(h.max()))
            row['W'].append(float(np.bincount(h).max()))
            row['ch'].append(float(len(lens)))
            row['rel'].append(nrel)
            row['ins'].append(ins)
            row['frac'].append(ins / max(nrel, 1))
            row['n0'].append(float(L.sum()) / max(nrel, 1))
            del R, L, A, X
        for k in acc:
            acc[k].append(float(np.mean(row[k])))
        print(f"{N:>7}{acc['L'][-1]:>9.1f}{acc['W'][-1]:>8.1f}"
              f"{acc['ch'][-1]:>10.1f}{acc['rel'][-1]:>11.0f}"
              f"{acc['ins'][-1]:>9.0f}{acc['frac'][-1]:>13.5f}"
              f"{acc['n0'][-1]:>13.4f}", flush=True)

    print()
    for k, nm, exp in (('L', 'самая длинная цепь', 0.25),
                       ('W', 'наибольший слой', 0.75),
                       ('ch', 'историй в разложении', 0.75),
                       ('frac', 'ДОЛЯ ВНУТРИ ИСТОРИЙ', -0.75),
                       ('n0', 'прямых на отношение', None)):
        b, r2 = fit(a.ns, acc[k])
        e = f'   ожидание {exp:+.2f}' if exp is not None else ''
        print(f'  {nm:<22} ~ N^{b:+.4f}   R² = {r2:.4f}{e}')

    print()
    print(f"  при N = {a.ns[-1]}: внутри историй лежит "
          f"{100*acc['frac'][-1]:.2f} % порядка")
    print('  доля падает => порядок есть почти целиком ПЕРЕСЕЧЕНИЯ историй,')
    print('  а не чьё-то собственное прошлое')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
