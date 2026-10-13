#!/usr/bin/env python3
"""Отклик холономии, попытка третья: ПАРНО по мирам, нормированно.

ДВЕ ПРЕДЫДУЩИЕ ПОПЫТКИ И ЧТО В НИХ СЛОМАЛОСЬ.
  1) Бугор по радиусу И времени — плотная область крошечная, сдвиг доли
     прямых 0.049->0.084 при архивных 0.396->0.215.
  2) Цилиндр по радиусу — калибровка пошла (0.244 внутри против 0.428
     снаружи при warp=0, архив 0.2973/0.4462), НО нулевая точка не
     нулевая: при warp=0 разница доли прямых уже -0.1839, а холономии
     4.1 сигмы. Потому что «внутри» значит «в середине ромба», а
     середина особая сама по себе. И при фиксированном N сгущения не
     происходит: связей внутри 100.9->99.9, меняется периферия 46->56.

ПОЧИНКА. Сравнивать ДВА МИРА В ОДНОМ МЕСТЕ: тот же засев, тот же
радиальный слой, разный warp. Геометрия сокращается парной разностью.

И НОРМИРОВКА. Холономия есть разность счётов, значит растёт с
плотностью сама. Берём безразмерное:

    h_norm = |холономия| / <|общее прошлое| по звеньям цикла>

то есть отношение счётов — по той же причине, по которой архив брал
ДОЛЮ прямых, а не их число.

КОНТРОЛИ: телескопирующаяся (0 при любом warp) и перемешанная.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
sys.path.insert(0, '/tmp/claude-0/-home-claude/8f20417e-eecf-5054-8071-09c440210416/scratchpad')
from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond

N, T, SIG, SAMP, NCYC, SEEDS = 2200, 2.0, 0.45, 400, 6, 10
WARPS = [0.0, 2.0, 4.0, 8.0]

def shell_holo(X, sig, samp, ncyc, rs):
    """Нормированные холономии трёх фаз в слое r < sig, плюс доля прямых."""
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    CP = (A @ A.T)
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    idx = rs.permutation(len(X))
    SH = G[np.ix_(idx, idx)]; SH = 0.5 * (SH - SH.T)
    infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    pool = np.flatnonzero((rr < sig) & (U.sum(1) >= 3))
    if len(pool) < 30:
        del R, L, U; return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    g, t_, s_, fi = [], [], [], []
    for v in pool:
        v = int(v)
        cyc = cycles_through(U, v, ncyc, rs)
        if len(cyc) < 2:
            continue
        for c in cyc:
            # масштаб: среднее общее прошлое по звеньям цикла
            sc = np.mean([CP[c[k], c[(k + 1) % len(c)]]
                          for k in range(len(c))])
            sc = max(float(sc), 1.0)
            g.append(abs(holonomy(c, G)) / sc)
            t_.append(abs(holonomy(c, TE)) / sc)
            s_.append(abs(holonomy(c, SH)) / sc)
        fi.append(infl[v])
    del R, L, U
    if not g:
        return None
    return np.mean(g), np.mean(t_), np.mean(s_), np.mean(fi)

print(f'N={N} точно, слой r<{SIG}, засевов {SEEDS}')
print('ПАРНО: один засев, разный warp, один и тот же слой')
print('холономия НОРМИРОВАНА на общее прошлое (безразмерна)')
print()
acc = {w: {'g': [], 't': [], 's': [], 'f': []} for w in WARPS}
for t in range(SEEDS):
    for w in WARPS:
        rs = np.random.default_rng(6201000 + 977 * t)
        X = warped_diamond(N, T, w, SIG, rs)
        r = shell_holo(X, SIG, SAMP, NCYC,
                       np.random.default_rng(99 + t))
        del X
        if r is None:
            continue
        acc[w]['g'].append(r[0]); acc[w]['t'].append(r[1])
        acc[w]['s'].append(r[2]); acc[w]['f'].append(r[3])

print(f"{'warp':>6}{'доля прямых':>14}{'холон. КАЛИБР':>16}"
      f"{'парно minus warp0':>20}{'сигм':>7}{'телеск':>9}{'перемеш':>11}")
base_g = np.array(acc[WARPS[0]]['g'])
for w in WARPS:
    g = np.array(acc[w]['g'])
    if len(g) < 3:
        print(f'{w:>6.0f}  мало'); continue
    d = g - base_g[:len(g)]
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
    print(f'{w:>6.0f}{np.mean(acc[w]["f"]):>14.4f}{g.mean():>16.5f}'
          f'{d.mean():>+20.5f}'
          f'{(abs(d.mean())/se if se and se > 1e-12 else float("nan")):>7.1f}'
          f'{np.mean(acc[w]["t"]):>9.3f}{np.mean(acc[w]["s"]):>11.5f}')

print('\n  растёт парно с warp => ОТКЛИК; телескоп обязан быть 0')
