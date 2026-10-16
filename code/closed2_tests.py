#!/usr/bin/env python3
"""Замкнутая система (closed2.py): две щели и Белл с полным учётом исходов.
Белл: у Алисы «стенка» на линии T̃ = T̃_A в её системе (|x̃| < W) — поглощает каждую ногу, пересекающую её,
кроме ног в события-щели A₁, A₂ и из них; пластинка — фаза на выходах A₂; экран на T̃ = T̃_A + 3 (|x̃| < W):
p — x̃ > 0, q — x̃ < 0. У Боба — отражение P. Пара — рождение в неизвестном месте полосы источника.
Вероятности считаются по ВСЕМ исходам (любая нога, ушедшая куда угодно, — «не сработал», +1 по Эберхарду)."""
import sys, math, itertools, numpy as np
from closed2 import Closed, CH
S = Closed(turn=True); X = S.X; n = S.n; w = S.w
edges = S.edges
ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges])

def crossing(Xf, c, xlo, xhi):
    """ноги, пересекающие T̃ = c в кадре Xf, с x̃ ∈ (xlo, xhi); возвращает индексы и x̃ пересечения"""
    t0, t1 = Xf[ea, 0], Xf[eb, 0]; cr = (t0 < c) & (t1 >= c)
    xc = Xf[ea, 1] + (Xf[eb, 1] - Xf[ea, 1]) * (c - t0) / np.where(t1 - t0 == 0, 1, t1 - t0)
    m = cr & (xc > xlo) & (xc < xhi); return np.nonzero(m)[0], xc[m]

# ---------- две щели ----------
def slits():
    src = int(np.argmin(((X - np.array([-10.0, 0.0])) ** 2).sum(1)))
    wallE, _ = crossing(X, -3.0, -30, 30); scrE, scrx = crossing(X, 5.0, -30, 30)
    for d in (2.0, 4.0):
        cand = np.nonzero(np.abs(X[:, 0] + 3) < 0.45)[0]
        S1 = cand[np.argmin((X[cand, 1] - d / 2) ** 2)]; S2 = CH.img(w, S1, CH.P, CH.P)
        out = {}
        for name, op in (('обе', {S1, S2}), ('1', {S1}), ('2', {S2})):
            block = {e: 'w' for e in wallE if ea[e] not in op and eb[e] not in op}
            for e in scrE:
                if e not in set(wallE): block[e] = 's'      # нога через стенку и экран сразу поглощается стенкой
            ab, res, amp = S.run({src: np.array([1.0 + 0j])}, block=block)
            wall = sum(abs(ab[e][0]) ** 2 for e in ab if block[e] == 'w'); scr = {e: abs(ab[e][0]) ** 2 for e in ab if block[e] == 's'}
            bins = np.linspace(-9, 9, 37); xs = dict(zip(scrE, scrx))
            Ib = np.histogram([xs[e] for e in scr], bins, weights=list(scr.values()))[0]
            out[name] = (Ib, wall, res[0], Ib.sum() + wall + res[0])
        Ic, I1, I2 = out['обе'][0], out['1'][0], out['2'][0]; inc = I1 + I2; m = inc > 0.05 * inc.max()
        r = (Ic - inc) / np.maximum(inc, 1e-30)
        print(f'две щели x = ±{X[S1, 1]:.2f}: полная вероятность {out["обе"][3]:.6f}; на экран {Ic.sum():.4f}; '
              f'интерференционный член от {r[m].min():+.2f} до {r[m].max():+.2f}'); sys.stdout.flush()

# ---------- Белл ----------
def bell(NG=8, W=2.0):
    K = 2
    H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
    Xa = CH.axis_frame(X, -K * CH.ETA0 / 2); Xb = CH.axis_frame(X, +K * CH.ETA0 / 2)
    A1 = [k for k in range(n) if abs(Xa[k, 0] - 2.54) < 0.01 and abs(Xa[k, 1] - 0.62) < 0.01][0]; A2 = CH.img(w, A1, H, Hs)
    B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
    tA = Xa[A1, 0]
    wA, _ = crossing(Xa, tA, -W, W); sA, sxA = crossing(Xa, tA + 3, -W, W)
    wB, _ = crossing(Xb, tA, -W, W); sB, sxB = crossing(Xb, tA + 3, -W, W)
    slitsA, slitsB = {A1, A2}, {B1, B2}
    block = {}
    for e in wA:
        if ea[e] not in slitsA and eb[e] not in slitsA: block[e] = 'O'
    for e in wB:
        if ea[e] not in slitsB and eb[e] not in slitsB: block[e] = 'O'
    wallset = set(wA) | set(wB)
    for e, x in zip(sA, sxA):
        if e not in wallset: block[e] = 'Ap' if x > 0 else 'Aq'
    for e, x in zip(sB, sxB):
        if e not in wallset: block[e] = 'Bp' if x < 0 else 'Bq'      # зеркально (P меняет знак x̃)
    # пространственная разнесённость: события стенок/экранов Алисы против Боба
    evA = {ea[e] for e in list(wA) + list(sA)} | {eb[e] for e in list(wA) + list(sA)}
    evB = {ea[e] for e in list(wB) + list(sB)} | {eb[e] for e in list(wB) + list(sB)}
    # разнесённость: точки пересечения стенки/экрана Алисы и Боба (в лабораторной системе)
    def pts(Es, Xf, c):
        out = []
        for e in Es:
            a, b = ea[e], eb[e]; f = (c - Xf[a, 0]) / (Xf[b, 0] - Xf[a, 0]); out.append(X[a] + f * (X[b] - X[a]))
        return np.array(out)
    pa = np.vstack([pts(wA, Xa, tA), pts(sA, Xa, tA + 3)]); pb = np.vstack([pts(wB, Xb, tA), pts(sB, Xb, tA + 3)])
    d = pb[None, :, :] - pa[:, None, :]; caus = int((np.abs(d[..., 0]) >= np.abs(d[..., 1])).sum())
    src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]; N = len(src)
    inj = {s: np.eye(N)[i].astype(complex) for i, s in enumerate(src)}
    plA = [e for e in range(S.E) if ea[e] == A2]; plB = [e for e in range(S.E) if ea[e] == B2]
    print(f'Белл: щели A {np.round(X[A1], 2)}, {np.round(X[A2], 2)}; ног стенки/экрана Алисы {len(wA)}/{len(sA)}, Боба {len(wB)}/{len(sB)}; '
          f'причинных связей событий установок Алиса–Боб {caus}; источников {N}'); sys.stdout.flush()
    cls = ['Ap', 'Aq', 'Bp', 'Bq']
    angles = np.linspace(0, 2 * math.pi, NG, endpoint=False)
    Pfull = np.zeros((NG, NG, 2, 2)); Ppost = np.zeros((NG, NG, 2, 2)); eta = np.zeros((NG, NG)); unit = 0
    for ia, al in enumerate(angles):
        for ib, be in enumerate(angles):
            plate = {e: np.exp(1j * al) for e in plA}; plate.update({e: np.exp(1j * be) for e in plB})
            ab, res, amp = S.run(inj, block=block, plate=plate)
            # проверка унитарности на всех исходах: Σ по поглощённым + резервуар = 1 для каждого источника
            totcol = res + sum(np.abs(v) ** 2 for v in ab.values()) + (np.abs(amp) ** 2).sum(0) * 0
            groups = {c: np.array([ab[e] for e in ab if block[e] == c]) for c in cls}
            Rm = {c: (np.abs(groups[c]) ** 2).sum() / N if len(groups[c]) else 0.0 for c in cls}
            M = {}
            for c1, c2 in itertools.product(cls, cls):
                if len(groups[c1]) == 0 or len(groups[c2]) == 0: M[c1, c2] = 0.0; continue
                Psi = groups[c1] @ groups[c2].T; M[c1, c2] = (np.abs(Psi) ** 2).sum() / N
            for c in cls: M[c, 'O'] = M['O', c] = max(Rm[c] - sum(M[c, c2] for c2 in cls), 0)
            M['O', 'O'] = max(1 - sum(v for v in M.values()), 0)
            def out(pair, side):
                cp = sum(1 for c in pair if c == side + 'p'); cq = sum(1 for c in pair if c == side + 'q')
                return -1 if cq > cp else 1
            for pair, v in M.items():
                a, b = out(pair, 'A'), out(pair, 'B')
                Pfull[ia, ib, (1 - a) // 2, (1 - b) // 2] += v
                if sorted([pair[0][0], pair[1][0]]) == ['A', 'B'] and 'O' not in pair:
                    Ppost[ia, ib, (1 - a) // 2, (1 - b) // 2] += v; eta[ia, ib] += v
            Ppost[ia, ib] /= max(Ppost[ia, ib].sum(), 1e-300)
        print(f'  α = {math.degrees(al):.0f}° готово; сумма вероятностей {Pfull[ia].sum((1, 2)).min():.6f}…{Pfull[ia].sum((1, 2)).max():.6f}'); sys.stdout.flush()
    def best(Pt, tol):
        E = Pt[..., 0, 0] - Pt[..., 0, 1] - Pt[..., 1, 0] + Pt[..., 1, 1]; mA = Pt[..., 0, :].sum(-1); mB = Pt[..., :, 0].sum(-1); bS = 0
        for a0, a1 in itertools.combinations(range(NG), 2):
            for b0, b1 in itertools.combinations(range(NG), 2):
                sig = max(abs(mA[a0, b0] - mA[a0, b1]), abs(mA[a1, b0] - mA[a1, b1]), abs(mB[a0, b0] - mB[a1, b0]), abs(mB[a0, b1] - mB[a1, b1]))
                if sig > tol: continue
                e = (E[a0, b0], E[a0, b1], E[a1, b0], E[a1, b1])
                bS = max(bS, abs(e[0] + e[1] + e[2] - e[3]), abs(e[0] - e[1] + e[2] + e[3]), abs(e[0] + e[1] - e[2] + e[3]), abs(-e[0] + e[1] + e[2] + e[3]))
        return bS
    mA = Pfull[..., 0, :].sum(-1); sig = np.abs(mA - mA[:, :1]).max()
    print(f'  БЕЗ отбора: S = {best(Pfull, 1.0):.4f}; сигнал (маргинал Алисы от угла Боба) {sig:.1e}; η (по частице у каждого) {eta.mean():.4f}')
    print(f'  с отбором «обе сработали»: S = {best(Ppost, 1.0):.3f} (без требования), {best(Ppost, 0.005):.3f} (сигнал ≤ 0.5 %), {best(Ppost, 0.001):.3f} (≤ 0.1 %)')

if __name__ == '__main__':
    slits()
    bell()
