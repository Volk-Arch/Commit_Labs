#!/usr/bin/env python3
"""ВОЛНОВОДЫ: два «рельса» в кристалле. Рельс — события кристалла в узкой трубке |x − x_i| < δ вокруг
вертикальной линии (световод, держащий частицу). Оставляем только события рельсов, берём индуцированный
порядок и его звенья. Вопрос: сколько у события входов и выходов? Если ровно по два («по своему рельсу»
и «перескок»), то правило «поворот = i» на каждом событии — унитарная монета 2×2."""
import sys, numpy as np
sys.argv = sys.argv[:1]
import chiral as CH
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); X = w.X
for d in (1.0, 2.0, 3.0):
    for delta in (0.3, 0.5, 0.8):
        r1 = np.nonzero((np.abs(X[:, 1] + d / 2) < delta) & (np.abs(X[:, 0]) < 14))[0]
        r2 = np.nonzero((np.abs(X[:, 1] - d / 2) < delta) & (np.abs(X[:, 0]) < 14))[0]
        ev = np.concatenate([r1, r2]); rail = np.array([0] * len(r1) + [1] * len(r2))
        o = np.argsort(X[ev, 0]); ev, rail = ev[o], rail[o]
        Rs = R[np.ix_(ev, ev)]; Rf = Rs.astype(np.float32); L = Rs & ((Rf @ Rf) == 0)
        I, J = np.nonzero(L); same = rail[I] == rail[J]
        ins = np.array([[(L[:, k] & (rail == r)).sum() for r in (0, 1)] for k in range(len(ev))])
        outs = np.array([[(L[k, :] & (rail == r)).sum() for r in (0, 1)] for k in range(len(ev))])
        own_in = ins[np.arange(len(ev)), rail]; oth_in = ins[np.arange(len(ev)), 1 - rail]
        own_out = outs[np.arange(len(ev)), rail]; oth_out = outs[np.arange(len(ev)), 1 - rail]
        mid = np.abs(X[ev, 0]) < 10
        ok = (own_in == 1) & (oth_in == 1) & (own_out == 1) & (oth_out == 1)
        print(f'расстояние {d}, трубка ±{delta}: событий {len(r1)}+{len(r2)}; звеньев вдоль {same.sum()}, поперёк {(~same).sum()}; '
              f'«1 вход по рельсу + 1 перескок, 1+1 выход» у {ok[mid].mean():.0%}; вх=исх у {(ins.sum(1) == outs.sum(1))[mid].mean():.0%}; '
              f'вх по своему: {np.bincount(own_in[mid]).tolist()}, перескоков вх: {np.bincount(oth_in[mid]).tolist()}')
