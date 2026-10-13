#!/usr/bin/env python3
"""ПОКАЗАТЕЛЬ МАСШТАБА лестницей по N — так, как взяты стойкие результаты.

ПОЧЕМУ ТАК, А НЕ ИНАЧЕ. Сначала (`exponent.py`) я мерил показатель по
циклам ВНУТРИ одного мира, беря за масштаб общее прошлое цикла. Вышло
alpha = 0.0534 против двухточечной оценки 0.37, то есть расхождение в
семь раз, и отклик 2.8σ при перемешанном контроле 2.4σ — то есть никакого
отклика.

Причина: внутри одного мира общее прошлое цикла — это почти целиком
ПОЛОЖЕНИЕ цикла в ромбе. У светового конуса общее прошлое крошечное, в
середине большое. Я подгонял показатель по положению, а не по масштабу.

Масштаб меняют укрупнением мира при ТОЙ ЖЕ геометрии. Все стойкие
показатели проекта так и взяты: связей на событие N^0.75, высота N^0.279,
площадной закон горизонта 2.286.

ЭТАП 1 — есть ли закон вообще. Лестница N, фиксированный ромб, замер в
фиксированной ОТНОСИТЕЛЬНОЙ оболочке. Подгонка log<|холономия|> по log N,
с R^2. Три фазы: калибровочная, перемешанная, телескопирующаяся.

    R^2 близко к 1 -> закон есть, alpha осмыслен
    R^2 мал        -> закона нет, и показатель тоже угадывание

ЭТАП 2 — отвечает ли закон на вещество. Та же лестница ПАРНО: дырка
против случайного удаления того же числа событий. Сравниваются alpha, а
показателю нормировка не нужна.

  python3 expladder.py
  python3 expladder.py --ns 600 1200 2400 --seeds 6 --stage2
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond


def shell_mean(X, f_lo, f_hi, samp, ncyc, rs):
    """Средние |холономии| трёх фаз в относительной оболочке + размер."""
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
    rmax = rr.max()
    pool = np.flatnonzero((rr >= f_lo * rmax) & (rr < f_hi * rmax) &
                          (U.sum(1) >= 3))
    if len(pool) < 25:
        del R, L, U, A, CP, G, SH, TE
        return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    g, s, t_, cs = [], [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            g.append(abs(holonomy(c, G)))
            s.append(abs(holonomy(c, SH)))
            t_.append(abs(holonomy(c, TE)))
            cs.append(float(np.mean([CP[c[k], c[(k + 1) % len(c)]]
                                     for k in range(len(c))])))
    del R, L, U, A, CP, G, SH, TE
    if len(g) < 60:
        return None
    return (float(np.mean(g)), float(np.mean(s)), float(np.max(t_)),
            float(np.mean(cs)), len(g))


def fit(ns, vals):
    """alpha и R^2 подгонки log vals по log ns."""
    ok = [i for i in range(len(vals)) if vals[i] is not None and vals[i] > 0]
    if len(ok) < 3:
        return float('nan'), float('nan')
    x = np.log(np.array([ns[i] for i in ok], float))
    y = np.log(np.array([vals[i] for i in ok], float))
    al, c = np.polyfit(x, y, 1)
    yp = al * x + c
    ss = ((y - y.mean()) ** 2).sum()
    return float(al), float(1 - ((y - yp) ** 2).sum() / ss) if ss > 0 else 1.0


def run_ladder(ns, T, f_lo, f_hi, samp, ncyc, seed, hole_rad=0.0, frac=1.0):
    """Один проход лестницы: средние по N для трёх фаз."""
    gg, ss, tt, cc, nn = [], [], [], [], []
    for N in ns:
        rs = np.random.default_rng(seed * 7919 + N)
        X = warped_diamond(N, T, 0.0, 0.45, rs)
        if hole_rad > 0:
            rr = np.linalg.norm(X[:, 1:], axis=1)
            rmax = rr.max()
            ins = np.flatnonzero(rr < hole_rad * rmax)
            drop = ins[rs.random(len(ins)) > frac] if len(ins) else ins
            keep = np.ones(len(X), bool)
            keep[drop] = False
            X = X[keep]
        r = shell_mean(X, f_lo, f_hi, samp, ncyc,
                       np.random.default_rng(seed * 13 + 5))
        del X
        gg.append(r[0] if r else None)
        ss.append(r[1] if r else None)
        tt.append(r[2] if r else None)
        cc.append(r[3] if r else None)
        nn.append(r[4] if r else 0)
    return gg, ss, tt, cc, nn


def main():
    ap = argparse.ArgumentParser(description='Показатель лестницей по N')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[500, 1000, 2000, 4000])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--shell', type=float, nargs=2, default=[0.45, 0.75])
    ap.add_argument('--rad', type=float, default=0.30)
    ap.add_argument('--samp', type=int, default=260)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--stage2', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.shell
    print(f'лестница N = {a.ns}, ромб T={a.t}')
    print(f'замер в ОТНОСИТЕЛЬНОЙ оболочке r/rmax в {lo}..{hi} — '
          f'геометрия одна при всех N')
    print(f'засевов {a.seeds}')
    print()
    print('ЭТАП 1: есть ли степенной закон вообще')
    print(f"{'N':>7}{'<|холон.|> калибр':>19}{'перемеш':>11}"
          f"{'телеск макс':>13}{'масштаб':>10}{'циклов':>9}")

    gs, ss_, cs_ = [], [], []
    for i, N in enumerate(a.ns):
        G_, S_, T_, C_, NN = [], [], [], [], []
        for t in range(a.seeds):
            r = run_ladder([N], a.t, lo, hi, a.samp, a.ncyc, 1000 + t)
            if r[0][0] is None:
                continue
            G_.append(r[0][0]); S_.append(r[1][0])
            T_.append(r[2][0]); C_.append(r[3][0]); NN.append(r[4][0])
        if not G_:
            print(f'{N:>7}  мало данных')
            gs.append(None); ss_.append(None); cs_.append(None)
            continue
        gs.append(float(np.mean(G_))); ss_.append(float(np.mean(S_)))
        cs_.append(float(np.mean(C_)))
        print(f'{N:>7}{np.mean(G_):>19.2f}{np.mean(S_):>11.2f}'
              f'{np.max(T_):>13.1e}{np.mean(C_):>10.1f}'
              f'{int(np.sum(NN)):>9}', flush=True)

    ag, rg = fit(a.ns, gs)
    as_, rs2 = fit(a.ns, ss_)
    ac, rc = fit(a.ns, cs_)
    print()
    print(f'  калибр:   |холономия| ~ N^{ag:+.4f}   R^2 = {rg:.4f}')
    print(f'  перемеш:  |холономия| ~ N^{as_:+.4f}   R^2 = {rs2:.4f}')
    print(f'  масштаб:  общее прошлое ~ N^{ac:+.4f}   R^2 = {rc:.4f}')
    print(f'  => безразмерно: |холономия| / масштаб^'
          f'{(ag/ac if ac and abs(ac) > 1e-9 else float("nan")):.4f}')
    print('  R^2 близко к 1 — закон есть; мал — показатель тоже угадывание')

    if not a.stage2:
        print(f'\n  время {time.time()-t0:.0f} с   (ЭТАП 2: --stage2)')
        return

    print()
    print('ЭТАП 2: отвечает ли ПОКАЗАТЕЛЬ на вещество')
    print(f'дырка радиуса {a.rad}*rmax против случайного удаления '
          f'того же числа')
    print(f"{'засев':>7}{'alpha дырка':>13}{'R^2':>7}{'alpha контр':>13}"
          f"{'R^2':>7}{'разница':>10}{'перемеш дырка':>15}{'контр':>9}")

    dd, sd = [], []
    for t in range(a.seeds):
        g1, s1, _, _, _ = run_ladder(a.ns, a.t, lo, hi, a.samp, a.ncyc,
                                     2000 + t, a.rad, 0.0)
        g2, s2, _, _, _ = run_ladder(a.ns, a.t, lo, hi, a.samp, a.ncyc,
                                     2000 + t, a.rad, 1.0)
        a1, r1 = fit(a.ns, g1)
        a2, r2 = fit(a.ns, g2)
        b1, _ = fit(a.ns, s1)
        b2, _ = fit(a.ns, s2)
        if not (np.isfinite(a1) and np.isfinite(a2)):
            continue
        dd.append(a1 - a2)
        if np.isfinite(b1) and np.isfinite(b2):
            sd.append(b1 - b2)
        print(f'{t:>7}{a1:>+13.4f}{r1:>7.3f}{a2:>+13.4f}{r2:>7.3f}'
              f'{a1-a2:>+10.4f}{b1:>+15.4f}{b2:>+9.4f}', flush=True)

    if len(dd) >= 3:
        d = np.array(dd)
        se = d.std(ddof=1) / math.sqrt(len(d))
        print(f'\n  разница показателей {d.mean():+.4f} ± {se:.4f}  '
              f'({abs(d.mean())/se if se > 1e-12 else float("nan"):.1f} сигм, '
              f'знак один {int((d > 0).sum() if d.mean() > 0 else (d < 0).sum())}'
              f'/{len(d)})')
        if len(sd) >= 3:
            s = np.array(sd)
            ses = s.std(ddof=1) / math.sqrt(len(s))
            print(f'  перемешанная        {s.mean():+.4f} ± {ses:.4f}  '
                  f'({abs(s.mean())/ses if ses > 1e-12 else float("nan"):.1f} '
                  f'сигм) — ОБЯЗАНА молчать')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
