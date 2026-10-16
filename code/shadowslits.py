#!/usr/bin/env python3
"""«Мир + тень», две щели: экран-преграда (поглощающая полоса, кроме двух событий-щелей) и экран-детектор
(поглощающая полоса). Сумма по всем исходам (стенка + экран + край) должна быть ровно 1 — проверяем.
Интерференция: экран при двух открытых щелях против суммы «по одной»."""
import sys, math, numpy as np
sys.argv = ['x', '4']
exec(open('/home/claude/code/shadowclosed.py').read().split("if __name__ == '__main__':")[0])
TW, TS = -3.0, 5.0
cross = lambda e, tt: X[edges[e][0], 0] < tt <= X[edges[e][1], 0]
xcross = lambda e, tt: (lambda a, b: X[a, 1] + (X[b, 1] - X[a, 1]) * (tt - X[a, 0]) / (X[b, 0] - X[a, 0]))(edges[e][0], edges[e][1])
EW = [e for e in range(E) if cross(e, TW)]; ES = [e for e in range(E) if cross(e, TS)]
def run_abs(inj, opened):
    """стенка: каждая нога, пересекающая t = TW, поглощается, если не выходит из открытой щели и не входит в неё;
    экран: каждая нога, пересекающая t = TS, поглощается и записывается по месту пересечения"""
    blockW = {e for e in EW if edges[e][0] not in opened and edges[e][1] not in opened}; blockS = set(ES)
    amp = np.zeros(E, complex); leaked = 0.0; wallp = 0.0; scr = {}
    for k in order:
        li = sorted(ein[k], key=rap); lo = sorted(eout[k], key=rap)
        D = max(len(li), len(lo)); ext_in = D - len(li)
        vin = np.zeros(D, complex); vin[:len(li)] = amp[li]
        if k in inj: vin[len(li) if ext_in else 0] += inj[k]
        if D == 0: continue
        F = np.exp(2j * math.pi * np.outer(np.arange(D), np.arange(D)) / D) / math.sqrt(D)
        vout = F @ vin; leaked += (np.abs(vout[len(lo):]) ** 2).sum()
        for idx, e in enumerate(lo):
            a = vout[idx]
            if e in blockW: wallp += abs(a) ** 2; continue
            if e in blockS: scr[e] = a; continue
            amp[e] = a
    return scr, wallp, leaked
src = int(np.argmin(((X - np.array([-10.0, 0.0])) ** 2).sum(1)))
wall = [k for k in range(n) if reg[k] and abs(X[k, 0] + 3) < 0.45]
for d in (2.0, 4.0):
    S1 = min(wall, key=lambda k: (X[k, 1] - d / 2) ** 2 + (X[k, 0] + 3) ** 2); S2 = CH.img(w, S1, CH.P, CH.P)
    res = {}
    bins = np.linspace(-9, 9, 37)
    for name, opened in (('обе', {S1, S2}), ('1', {S1}), ('2', {S2})):
        scr, wallp, leaked = run_abs({src: 1.0}, opened)
        xsc = np.array([xcross(e, TS) for e in scr]); a = np.array(list(scr.values()))
        # детектор = ячейка экрана; разные ноги в одной ячейке — разные записи (складываем вероятности)
        Ib = np.histogram(xsc, bins, weights=np.abs(a) ** 2)[0]
        res[name] = (Ib, wallp, leaked, Ib.sum() + wallp + leaked)
    Ic, I1, I2 = res['обе'][0], res['1'][0], res['2'][0]
    inc = I1 + I2; r = (Ic - inc) / np.maximum(inc, 1e-30)
    m = inc > 0.05 * inc.max()
    print(f'щели x = ±{X[S1, 1]:.2f}: полная вероятность (стенка + экран + край) = {res["обе"][3]:.6f} / {res["1"][3]:.6f}; '
          f'на экран {res["обе"][0].sum():.3f} (через одну щель {I1.sum():.3f} + {I2.sum():.3f}); '
          f'интерференционный член (I − I₁ − I₂)/(I₁ + I₂) по экрану: от {r[m].min():+.2f} до {r[m].max():+.2f}')
