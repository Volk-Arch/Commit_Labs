#!/usr/bin/env python3
"""Рельсы — зеркальная пара тонких цепочек (x = ±d/2). Перескок с рельса на рельс идёт со сдвигом по времени
не меньше d, поэтому пары «вход-перескок ↔ выход-перескок» совпадают, только если цепочка почти не меняется
при сдвиге на d. У квазикристалла почти-сдвиги — числа решётки с малой тенью: d = √τ·τ^k."""
import sys, math, numpy as np
sys.argv = sys.argv[:1]
import chiral as CH
TAU = (1 + 5 ** .5) / 2
w = CH.sym_world(4.0, 30.0); R = CH.build_order(w, 'proj'); X = w.X
def rails(d, delta=0.3, T=18):
    r1 = np.nonzero((np.abs(X[:, 1] + d / 2) < delta) & (np.abs(X[:, 0]) < T))[0]
    r2 = np.array([CH.img(w, k, CH.P, CH.P) for k in r1])
    ev = np.concatenate([r1, r2]); rail = np.array([0] * len(r1) + [1] * len(r2))
    o = np.argsort(X[ev, 0], kind='stable'); ev, rail = ev[o], rail[o]
    Rs = R[np.ix_(ev, ev)]; Rf = Rs.astype(np.float32); L = Rs & ((Rf @ Rf) == 0)
    return ev, rail, L
for k in range(-1, 4):
    d = math.sqrt(TAU) * TAU ** k; dp = (1 / math.sqrt(TAU)) * (-1 / TAU) ** k   # тень сдвига
    for d_ in (d, d * 1.07):
        ev, rail, L = rails(d_)
        ins, outs = L.sum(0), L.sum(1)
        mid = np.abs(X[ev, 0]) < 12
        combo = {}
        for a, b in zip(ins[mid], outs[mid]): combo[(int(a), int(b))] = combo.get((int(a), int(b)), 0) + 1
        bal = np.mean(ins[mid] == outs[mid])
        print(f'd = {d_:.3f} {"(√τ·τ^%d, тень сдвига %+.3f)" % (k, dp) if d_ == d else "(контроль ×1.07)":28s}: событий на рельсе {len(ev)//2}; '
              f'вх = исх у {bal:.0%}; (вх, исх): {dict(sorted(combo.items()))}')
