#!/usr/bin/env python3
"""Откликается ли холономия на УБЫЛЬ вещества, то есть на разрежение.

МОЯ ОШИБКА, которую он заметил. В проверке источника
(`scratchpad/dens_src.py`) я подсаживал ПЛОТНУЮ мировую линию и получил
0.0…0.9 сигмы. Но в этот же день установлено (`delay-doc.md`), что
плотная линия — не масса, а ЛЕСТНИЦА: внутренняя высота 57…105 против
фоновых 11.8, часы у неё идут быстрее, цепи загибаются внутрь, то есть
обе половины тяготения с обратным знаком. Масса — это РАЗРЕЖЕНИЕ: оно
замедляет часы (4.5σ) и выгибает цепи наружу (5.0σ), и обе величины
монотонны по глубине разрежения, без скачка на полной пустоте.

Значит источник я проверял анти-массой. Здесь — убылью.

ЗАМЕР. Внутри радиуса rad остаётся доля f событий. Нормированная
холономия (безразмерная, делённая на общее прошлое) берётся в СЛОЕ
СНАРУЖИ дырки — там, где днём жили сигналы задержки и выгиба, то есть в
пределах одной длины дискретности.

ПАРНО по засеву, три мира на каждый:
    дырка        — проверяемое
    случайно     — удалено столько же событий по всему ромбу (контроль
                   на «просто меньше событий»)
    целый        — нулевая точка

КОНТРОЛИ ФАЗЫ: телескопирующаяся (холономия обязана быть ровно 0) и
перемешанная.

  python3 holosource.py
  python3 holosource.py --fracs 1.0 0.6 0.3 0.0 --seeds 12
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond


def shell_holo(X, r_lo, r_hi, samp, ncyc, rs):
    """Нормированные холономии трёх фаз в слое r_lo..r_hi."""
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
    g, t_, s_ = [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            sc = np.mean([CP[c[k], c[(k + 1) % len(c)]]
                          for k in range(len(c))])
            sc = max(float(sc), 1.0)
            g.append(abs(holonomy(c, G)) / sc)
            t_.append(abs(holonomy(c, TE)) / sc)
            s_.append(abs(holonomy(c, SH)) / sc)
    del R, L, U
    if not g:
        return None
    return float(np.mean(g)), float(np.mean(t_)), float(np.mean(s_))


def main():
    ap = argparse.ArgumentParser(description='Холономия против разрежения')
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--samp', type=int, default=350)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.rad, a.rad + 0.16
    print(f'N={a.n}, внутри радиуса {a.rad} остаётся доля f событий')
    print(f'холономия берётся в слое {lo:.2f}..{hi:.2f} — там, где днём '
          f'жили задержка и выгиб')
    print('ПАРНО по засеву: дырка против случайного удаления того же числа')
    print('холономия нормирована на общее прошлое (безразмерна)')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'f':>6}{'удалено':>9}{'дырка':>11}{'случайно':>11}"
          f"{'разница':>11}{'сигм':>7}{'телеск':>9}{'перемеш':>10}")

    for f in a.fracs:
        D, C, T_, S_ = [], [], [], []
        nrem = []
        for t in range(a.seeds):
            rs = np.random.default_rng(6501000 + 977 * t)
            X0 = warped_diamond(a.n, a.t, 0.0, 0.45, rs)
            rr = np.linalg.norm(X0[:, 1:], axis=1)
            ins = np.flatnonzero(rr < a.rad)
            drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
            k = len(drop)
            nrem.append(k)
            keep = np.ones(len(X0), bool)
            keep[drop] = False
            r1 = shell_holo(X0[keep], lo, hi, a.samp, a.ncyc,
                            np.random.default_rng(7 + t))
            if k > 0:
                ii = rs.choice(len(X0), len(X0) - k, replace=False)
                r2 = shell_holo(X0[np.sort(ii)], lo, hi, a.samp, a.ncyc,
                                np.random.default_rng(7 + t))
            else:
                r2 = r1
            del X0
            if r1 is None or r2 is None:
                continue
            D.append(r1[0]); C.append(r2[0])
            T_.append(r1[1]); S_.append(r1[2])
        if len(D) < 3:
            print(f'{f:>6.2f}  мало данных')
            continue
        D = np.array(D); C = np.array(C)
        d = D - C
        se = d.std(ddof=1) / math.sqrt(len(d))
        print(f'{f:>6.2f}{np.mean(nrem):>9.0f}{D.mean():>11.4f}'
              f'{C.mean():>11.4f}{d.mean():>+11.4f}'
              f'{(abs(d.mean())/se if se > 1e-12 else float("nan")):>7.1f}'
              f'{np.mean(T_):>9.3f}{np.mean(S_):>10.4f}', flush=True)

    print('\n  монотонно по глубине разрежения и выше контроля => ОТКЛИК')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
