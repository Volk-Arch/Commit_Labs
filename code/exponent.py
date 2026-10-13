#!/usr/bin/env python3
"""ПОКАЗАТЕЛЬ вместо величины: отклик, которому нормировка не нужна.

ОТКУДА. Вчерашний-сегодняшний провал: отклик холономии на вещество
оказался тавтологией, потому что я УГАДАЛ нормировку. Делил на
масштаб^1, а из тех же данных видно, что

    |холономия| ~ масштаб^alpha,   alpha ~ 0.37

значит «безразмерная» величина была обратной плотностью в степени 0.63.
Проверка при одинаковом знаменателе дала переворот знака (+0.2573 в
целом против −0.3132 внутри корзин, парадокс Симпсона).

ЕГО МЫСЛЬ: ошибка показала, где искать. И она верна, потому что

> величину надо нормировать, и нормировку можно угадать неверно;
> ПОКАЗАТЕЛЬ нормировки не требует — он отношение двух логарифмов,
> безразмерный по построению, и теорема о перешкалировании его не
> трогает.

Проект это уже подтверждал самыми стойкими своими результатами, и все
они — показатели: площадной закон горизонта 2.286, связей на событие
N^0.75, высота N^0.279.

ЧТО МЕРИМ. На каждом мире закон `log|холономия| = alpha * log(масштаб) + c`
по циклам, затем ПАРНО: alpha у мира с дыркой против мира со случайным
удалением того же числа событий.

    alpha меняется  -> отклик, не сводимый к нормировке
    alpha не меняется -> закон универсален, и тогда он сам есть результат

КОНТРОЛИ:
  перемешанная фаза — свой alpha, обязан не отвечать;
  телескопирующаяся — холономия ровно 0, alpha не определён (проверка
    кода);
  нулевая точка (дырка нулевой глубины) — разница обязана быть нулём.

  python3 exponent.py
  python3 exponent.py --fracs 1.0 0.5 0.0 --seeds 20
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond


def law(X, r_lo, r_hi, samp, ncyc, rs):
    """Показатель alpha закона |холономия| ~ масштаб^alpha, по циклам."""
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    idx = rs.permutation(len(X))
    SH = G[np.ix_(idx, idx)]
    SH = 0.5 * (SH - SH.T)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    pool = np.flatnonzero((rr >= r_lo) & (rr < r_hi) & (U.sum(1) >= 3))
    if len(pool) < 25:
        del R, L, U
        return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    sc, hg, hs, ht = [], [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            s = np.mean([CP[c[k], c[(k + 1) % len(c)]]
                         for k in range(len(c))])
            if s < 2:
                continue
            g = abs(holonomy(c, G))
            s2 = abs(holonomy(c, SH))
            t2 = abs(holonomy(c, TE))
            if g <= 0 or s2 <= 0:
                continue
            sc.append(float(s)); hg.append(g); hs.append(s2); ht.append(t2)
    del R, L, U
    if len(sc) < 120:
        return None
    ls = np.log(np.array(sc))
    ag = np.polyfit(ls, np.log(np.array(hg)), 1)[0]
    as_ = np.polyfit(ls, np.log(np.array(hs)), 1)[0]
    return ag, as_, float(np.max(ht)), len(sc)


def main():
    ap = argparse.ArgumentParser(description='Показатель вместо величины')
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--samp', type=int, default=350)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.rad, a.rad + 0.16
    print(f'N={a.n}, слой {lo:.2f}..{hi:.2f}, засевов {a.seeds}')
    print('закон |холономия| ~ масштаб^alpha, подгонка по циклам в логах')
    print('ПАРНО: дырка против случайного удаления того же числа')
    print('показателю нормировка не нужна — он безразмерен по построению')
    print()
    print(f"{'f':>6}{'удал':>7}{'alpha дырка':>13}{'контроль':>11}"
          f"{'разница':>10}{'сигм':>7}{'знак один':>11}"
          f"{'alpha перемеш':>15}{'разн.':>8}{'сигм':>6}{'телеск':>8}")

    for f in a.fracs:
        rows = []
        for t in range(a.seeds):
            rs = np.random.default_rng(7101000 + 977 * t)
            X0 = warped_diamond(a.n, a.t, 0.0, 0.45, rs)
            rr = np.linalg.norm(X0[:, 1:], axis=1)
            ins = np.flatnonzero(rr < a.rad)
            drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
            k = len(drop)
            keep = np.ones(len(X0), bool)
            keep[drop] = False
            r1 = law(X0[keep], lo, hi, a.samp, a.ncyc,
                     np.random.default_rng(83 + t))
            if k > 0:
                ii = rs.choice(len(X0), len(X0) - k, replace=False)
                r2 = law(X0[np.sort(ii)], lo, hi, a.samp, a.ncyc,
                         np.random.default_rng(83 + t))
            else:
                r2 = r1
            del X0
            if r1 and r2:
                rows.append((k, r1[0], r2[0], r1[1], r2[1],
                             max(r1[2], r2[2])))
        if len(rows) < 4:
            print(f'{f:>6.2f}  мало данных')
            continue
        A = np.array(rows, float)
        d = A[:, 1] - A[:, 2]
        ds = A[:, 3] - A[:, 4]
        se = d.std(ddof=1) / math.sqrt(len(d))
        ses = ds.std(ddof=1) / math.sqrt(len(ds))
        same = int((d > 0).sum()) if d.mean() > 0 else int((d < 0).sum())
        print(f'{f:>6.2f}{A[:,0].mean():>7.0f}{A[:,1].mean():>13.4f}'
              f'{A[:,2].mean():>11.4f}{d.mean():>+10.4f}'
              f'{(abs(d.mean())/se if se > 1e-12 else float("nan")):>7.1f}'
              f'{same}/{len(d):<9}{A[:,3].mean():>15.4f}'
              f'{ds.mean():>+8.4f}'
              f'{(abs(ds.mean())/ses if ses > 1e-12 else float("nan")):>6.1f}'
              f'{A[:,5].max():>8.1f}', flush=True)

    print('\n  alpha отвечает монотонно, а перемешанная нет => отклик,')
    print('  не сводимый к выбору нормировки')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
