#!/usr/bin/env python3
"""ОТКЛИК: откликается ли холономия на НАВЕДЁННУЮ кривизну.

ЧТО УЖЕ ЕСТЬ (`connection-doc.md`). Холономия калибровочной связности
коррелирует с местным признаком кривизны: частная корреляция
−0.0780 ± 0.0105 при снятой высоте, 7.5 сигмы по 24 независимым мирам,
знак в 23 из 24. Контроли чистые: телескопирующаяся фаза даёт холономию
ровно 0.00e+00, перемешанная 1.3 сигмы.

Но это КОРРЕЛЯЦИЯ, а не уравнение. Уравнение будет, если навязать
кривизну и увидеть, что холономия откликнулась.

КАК НАВОДИТСЯ КРИВИЗНА. Архивным способом (`gravity-doc.md`,
`action-doc.md`): переменная плотность засева, «warp». Архивная
калибровка: при warp 0→8 связей на событие 15.06→23.76 (до 52 сигм), а
ДОЛЯ прямых падает 0.396→0.215, и доля информативнее, потому что не
зависит от общего роста связности.

ЗАМЕР. Сгущение в середине (гауссов бугор), число событий ТОЧНО
одинаково при всех warp. Внутри против снаружи, при совпадающей высоте.

КОНТРОЛИ, поставленные заранее:
  1. warp = 0 — разницы быть не должно вовсе (нулевая точка);
  2. телескопирующаяся фаза — холономия ровно 0 при любом warp;
  3. перемешанная фаза — отклика быть не должно;
  4. совпадающая высота — именно её отсутствие погубило замер по
     дефектам (3.0 сигмы целиком объяснились высотой).

  python3 response.py
  python3 response.py --warps 0 2 4 8 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal


def warped_diamond(N, T, w, sig, rs):
    """Засев ромба со сгущением в ЦИЛИНДРЕ радиуса sig. Ровно N событий.

    ПОЙМАННАЯ ОШИБКА. Первая версия ставила гауссов бугор сразу по
    радиусу И по времени, отчего плотная область выходила крошечной:
    сдвиг доли прямых внутри-снаружи получался 0.049 -> 0.084 при
    архивных 0.396 -> 0.215. Плюс «внутри» (середина ромба) само по себе
    особая область, и при warp = 0 разница была уже -0.0489, то есть
    нулевая точка не нулевая.

    Здесь сгущение идёт цилиндром по радиусу при всех временах, как поле
    в `gravity-doc.md`, и сравнение внутри-снаружи чисто радиальное.
    """
    out, got = [], 0
    while got < N:
        m = int((N - got) * 6) + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        Y = Y[ok]
        if len(Y) == 0:
            continue
        rr = np.linalg.norm(Y[:, 1:], axis=1)
        acc = (1.0 + w * (rr < sig)) / (1.0 + w)
        keep = rs.random(len(Y)) < acc
        Y = Y[keep]
        if len(Y) == 0:
            continue
        out.append(Y)
        got += len(Y)
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def measure(X, T, sig, samp, ncyc, rs):
    """Холономии трёх фаз, доля прямых, высота, признак «внутри»."""
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    G = antisym_causal(((A @ A.T) - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    idx = rs.permutation(len(X))
    SH = G[np.ix_(idx, idx)]
    SH = 0.5 * (SH - SH.T)
    infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
    h = np.zeros(len(X), np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    rr = np.linalg.norm(X[:, 1:], axis=1)
    inside = rr < sig
    deg = U.sum(1)
    pool = np.flatnonzero(deg >= 3)
    if len(pool) < 40:
        return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    rows = []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        rows.append((
            np.mean([abs(holonomy(c, G)) for c in cyc]),
            np.mean([abs(holonomy(c, TE)) for c in cyc]),
            np.mean([abs(holonomy(c, SH)) for c in cyc]),
            infl[v], float(h[v]), 1.0 if inside[v] else 0.0))
    del R, L, U
    return np.array(rows) if rows else None


def matched(M, col):
    """Разница внутри минус снаружи при совпадающей высоте (взвешенно)."""
    H, IN = M[:, 4], M[:, 5] > 0.5
    num, den = 0.0, 0.0
    for hv in np.unique(H):
        m = H == hv
        a, b = M[m & IN, col], M[m & ~IN, col]
        if len(a) < 3 or len(b) < 8:
            continue
        num += (a.mean() - b.mean()) * len(a)
        den += len(a)
    return num / den if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description='Отклик холономии')
    ap.add_argument('--n', type=int, default=2200)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--sig', type=float, default=0.35)
    ap.add_argument('--warps', type=float, nargs='+', default=[0, 2, 4, 8])
    ap.add_argument('--samp', type=int, default=400)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n} ТОЧНО при всех warp, сгущение радиуса {a.sig} '
          f'в середине')
    print('разница ВНУТРИ минус СНАРУЖИ при совпадающей высоте')
    print('warp = 0 — нулевая точка, разницы быть не должно')
    print(f'засевов {a.seeds}')
    print()
    print('СНАЧАЛА КАЛИБРОВКА: воспроизводит ли сгущение архивный сдвиг')
    print(f"{'warp':>6}{'доля прям. внут':>17}{'снаружи':>10}"
          f"{'сдвиг':>9}{'связей внут':>13}{'снаружи':>10}")
    for w in a.warps:
        ii, oo, li, lo = [], [], [], []
        for t in range(4):
            rs = np.random.default_rng(6101000 + 977 * t)
            X = warped_diamond(a.n, a.t, w, a.sig, rs)
            R = cone_rel(X); L = links_of(R)
            fr = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
            ln = (L.sum(1) + L.sum(0)).astype(float)
            m = np.linalg.norm(X[:, 1:], axis=1) < a.sig
            ii.append(fr[m].mean()); oo.append(fr[~m].mean())
            li.append(ln[m].mean()); lo.append(ln[~m].mean())
            del R, L, X
        print(f'{w:>6.0f}{np.mean(ii):>17.4f}{np.mean(oo):>10.4f}'
              f'{np.mean(ii)-np.mean(oo):>+9.4f}'
              f'{np.mean(li):>13.1f}{np.mean(lo):>10.1f}', flush=True)
    print('  архивный эталон: доля прямых 0.2973 внутри против 0.4462 '
          'снаружи')
    print()
    print(f"{'warp':>6}{'доля прямых':>14}{'холон. КАЛИБР':>16}{'сигм':>7}"
          f"{'холон. телеск':>15}{'холон. перемеш':>16}{'сигм':>7}")

    for w in a.warps:
        dg, dt_, ds, di = [], [], [], []
        for t in range(a.seeds):
            rs = np.random.default_rng(6001000 + 977 * t)
            X = warped_diamond(a.n, a.t, w, a.sig, rs)
            M = measure(X, a.t, a.sig, a.samp, a.ncyc, rs)
            del X
            if M is None or (M[:, 5] > 0.5).sum() < 20:
                continue
            di.append(matched(M, 3))
            dg.append(matched(M, 0))
            dt_.append(matched(M, 1))
            ds.append(matched(M, 2))
        if not dg:
            print(f'{w:>6.0f}{"мало данных":>14}')
            continue
        def rep(v):
            v = np.array([x for x in v if np.isfinite(x)])
            if len(v) < 2:
                return np.nan, np.nan
            return v.mean(), v.std(ddof=1) / math.sqrt(len(v))
        mi, si = rep(di); mg, sg = rep(dg)
        mt, st = rep(dt_); ms, ss = rep(ds)
        print(f'{w:>6.0f}{mi:>+14.4f}{mg:>+16.3f}'
              f'{(abs(mg)/sg if sg > 1e-12 else float("nan")):>7.1f}'
              f'{mt:>+15.3f}{ms:>+16.3f}'
              f'{(abs(ms)/ss if ss > 1e-12 else float("nan")):>7.1f}',
              flush=True)

    print('\n  холономия КАЛИБР растёт с warp, при warp=0 нуль,')
    print('  телескоп и перемешанная молчат => ОТКЛИК ЕСТЬ')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
