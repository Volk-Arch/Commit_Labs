#!/usr/bin/env python3
"""Пункты 1 и 2 заново, ИСПРАВЛЕННЫМ оценщиком.

ПОЧЕМУ ЗАНОВО. Отклик на наведённое искривление (warp) я мерил средним
отношений |холономия|/масштаб по циклам и получил молчание (2.814,
2.808, 2.815 — 0.1 и 0.0 сигмы). Но потом выяснилось, что этот оценщик
съедает сигнал: у отдельного цикла знаменатель мал, у отношения тяжёлый
хвост. На разрежении та же ошибка превратила 17.3 сигмы в 2.1. Значит
отрицательный ответ по пункту 1 НЕДЕЙСТВИТЕЛЕН.

ПУНКТ 1: отклик на warp, отношение средних, парно по засеву.
ПУНКТ 2: рост с плотностью — то же отношение, при разных N.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond

SAMP, NCYC = 350, 6

def ratio(X, r_lo, r_hi, rs):
    """Отношение средних: (сумма |холономия|) / (сумма масштабов)."""
    R = cone_rel(X); L = links_of(R); U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    rr = np.linalg.norm(X[:, 1:], axis=1)
    pool = np.flatnonzero((rr >= r_lo) & (rr < r_hi) & (U.sum(1) >= 3))
    if len(pool) < 25:
        del R, L, U; return None
    pool = rs.choice(pool, min(SAMP, len(pool)), replace=False)
    sg = st = sc = 0.0; n = 0
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, NCYC, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            sg += abs(holonomy(c, G))
            st += abs(holonomy(c, TE))
            sc += np.mean([CP[c[k], c[(k + 1) % len(c)]]
                           for k in range(len(c))])
            n += 1
    del R, L, U
    return (sg / sc, st / max(sc, 1e-9), n) if n >= 20 else None

print('ПУНКТ 1: отклик на НАВЕДЁННОЕ ИСКРИВЛЕНИЕ, отношение средних')
print('слой r<0.45 (внутри сгущения), парно против warp=0')
print(f"{'warp':>6}{'холономия':>12}{'парно - warp0':>16}{'сигм':>7}"
      f"{'знак один':>11}{'телеск':>9}")
SEEDS, N, T, SIG = 20, 2200, 2.0, 0.45
base = {}
for w in (0.0, 2.0, 4.0, 8.0):
    vals, tele = [], []
    for t in range(SEEDS):
        rs = np.random.default_rng(6801000 + 977 * t)
        X = warped_diamond(N, T, w, SIG, rs)
        r = ratio(X, 0.0, SIG, np.random.default_rng(41 + t))
        del X
        if r:
            vals.append(r[0]); tele.append(r[1])
    if len(vals) < 4:
        print(f'{w:>6.0f}  мало'); continue
    v = np.array(vals)
    if w == 0.0:
        base = v.copy()
        print(f'{w:>6.0f}{v.mean():>12.4f}{0.0:>+16.4f}{"":>7}{"":>11}'
              f'{np.mean(tele):>9.3f}')
        continue
    m = min(len(v), len(base))
    d = v[:m] - base[:m]
    se = d.std(ddof=1) / math.sqrt(len(d))
    same = int((d > 0).sum()) if d.mean() > 0 else int((d < 0).sum())
    print(f'{w:>6.0f}{v.mean():>12.4f}{d.mean():>+16.4f}'
          f'{(abs(d.mean())/se if se > 1e-12 else float("nan")):>7.1f}'
          f'{same}/{m:<9}{np.mean(tele):>9.3f}', flush=True)

print()
print('ПУНКТ 2: растёт ли величина отклика с ПЛОТНОСТЬЮ')
print('отклик = холономия при полной дырке минус контроль, по засевам')
print(f"{'N':>7}{'плотность':>11}{'отклик':>10}{'±':>9}{'сигм':>7}"
      f"{'относит.':>10}")
RAD = 0.24
for N2 in (1400, 2600, 4200):
    rows = []
    for t in range(16):
        rs = np.random.default_rng(6901000 + 977 * t)
        X0 = warped_diamond(N2, T, 0.0, 0.45, rs)
        rr = np.linalg.norm(X0[:, 1:], axis=1)
        ins = np.flatnonzero(rr < RAD)
        keep = np.ones(len(X0), bool); keep[ins] = False
        r1 = ratio(X0[keep], RAD, RAD + 0.16,
                   np.random.default_rng(53 + t))
        ii = rs.choice(len(X0), len(X0) - len(ins), replace=False)
        r2 = ratio(X0[np.sort(ii)], RAD, RAD + 0.16,
                   np.random.default_rng(53 + t))
        del X0
        if r1 and r2:
            rows.append((r1[0], r2[0]))
    if len(rows) < 4:
        print(f'{N2:>7}  мало'); continue
    A = np.array(rows)
    d = A[:, 0] - A[:, 1]
    se = d.std(ddof=1) / math.sqrt(len(d))
    rho = N2 / (math.pi / 24 * T ** 4)
    print(f'{N2:>7}{rho:>11.0f}{d.mean():>+10.4f}{se:>9.4f}'
          f'{abs(d.mean())/max(se,1e-12):>7.1f}'
          f'{d.mean()/A[:,1].mean():>10.3f}', flush=True)
