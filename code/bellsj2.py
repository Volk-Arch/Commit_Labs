#!/usr/bin/env python3
"""БЕЛЛ В ЯЗЫКЕ SJ С ПЛАСТИНКАМИ-ПОЛЯМИ.
Установка Алисы: два «пути» A₁, A₂ (одно время, справа), пластинка — фаза e^{iα} на звеньях, пересекающих
короткий отрезок над A₂ (в будущем A₂, вне будущего A₁); у Боба — зеркальная копия (P), фаза e^{iβ}.
Для каждой пары (α, β) мир свой: пересчитываем R и вакуум W. Детекторы — пара портов у каждого
(выбраны один раз в мире без пластинок), вероятности по Глауберу |Ψ(a, b)|², Ψ = Σ_s W(·, s)W(·, s).
Меряем: CHSH по сетке углов, сигнал; и главное для SJ — «нелокальность вакуума»: меняется ли
статистика Боба, когда Алиса крутит свою пластинку (у запаздывающего R такого быть не может)."""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from sj import sj
from bellsj import schmidt_settings
sys.argv = _a
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 8
Lw = 14.0
w = CH.sym_world(4.0, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
I, J = LK['I'], LK['J']
# пути Алисы: одно время t ≈ 0.5, x ≈ 2.5 и 4.5 (без причинной связи между собой)
A1 = int(np.argmin(((X - np.array([0.5, 2.6])) ** 2).sum(1)))
cand = [k for k in range(n) if abs(X[k, 0] - X[A1, 0]) < 0.3 and 1.5 < X[k, 1] - X[A1, 1] < 2.5 and not R[A1, k] and not R[k, A1]]
A2 = cand[int(np.argmin([abs(X[k, 1] - X[A1, 1] - 2) for k in cand]))]
def segment_cut(Ac, Aother):
    """звенья, пересекающие горизонтальный отрезок над Ac, лежащий в будущем Ac и вне будущего Aother"""
    gap = abs(X[Ac, 1] - X[Aother, 1]) - abs(X[Ac, 0] - X[Aother, 0])
    tS = X[Ac, 0] + 0.45 * gap / 2 * 2 / 2; half = tS - X[Ac, 0]
    t0, t1 = X[I, 0], X[J, 0]; cr = (t0 < tS) & (t1 >= tS)
    xc = X[I, 1] + (X[J, 1] - X[I, 1]) * (tS - t0) / np.where(t1 - t0 == 0, 1, t1 - t0)
    return cr & (np.abs(xc - X[Ac, 1]) <= half)
cutA = segment_cut(A2, A1)
B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
cutB = segment_cut(B2, B1)
src_ev = [k for k in range(n) if abs(X[k, 0] + 7) <= 0.45 and abs(X[k, 1]) <= 3.0]
S = np.array([2 * k + m for k in src_ev for m in (0, 1)])
Aev = [k for k in range(n) if 3.5 <= X[k, 0] <= 6.0 and R[A1, k] and R[A2, k] and X[k, 1] > 2.0 and np.abs(X[k]).sum() <= Lw - 1
       and not any(R[b, k] or R[k, b] for b in (B1, B2))]
Aev = [k for k in Aev if not any(R[CH.img(w, j, CH.P, CH.P), k] or R[k, CH.img(w, j, CH.P, CH.P)] for j in Aev)]
Bev = [CH.img(w, k, CH.P, CH.P) for k in Aev]
caus = sum(R[a, b] or R[b, a] for a in Aev + [A1, A2] for b in Bev + [B1, B2])
Ap = np.array([2 * k + m for k in Aev for m in (0, 1)]); Bp = np.array([2 * k + m for k in Bev for m in (0, 1)])
print(f'ε = {eps}: событий {n}; A1 {np.round(X[A1], 2)}, A2 {np.round(X[A2], 2)}; звеньев под пластинкой {cutA.sum()}/{cutB.sum()}; '
      f'портов {len(Ap)}; причинных связей Алиса–Боб {caus}'); sys.stdout.flush()
angles = np.linspace(0, 2 * math.pi, NG, endpoint=False)
res = {}
for ia, al in enumerate(angles):
    for ib, be in enumerate(angles):
        fld = np.where(cutA, np.exp(1j * al), 1.0) * np.where(cutB, np.exp(1j * be), 1.0)
        Rp, iD, lam, V, W = sj(w, LK, eps, fld)
        res[ia, ib] = {'R': Rp[np.ix_(Ap, S)] @ Rp[np.ix_(Bp, S)].T, 'W': W[np.ix_(Ap, S)] @ W[np.ix_(Bp, S)].T}
    print(f'  α = {math.degrees(al):.0f}° готово'); sys.stdout.flush()
np.save('/home/claude/data/bellsj2_res.npy', res, allow_pickle=True)
for kind in ('R', 'W'):
    # порты: лучшая пара по качеству и балансу в мире без пластинок
    C, leak, sv, sa, sb = schmidt_settings(res[0, 0][kind], nbest=40)
    for (ua, (p, q)), (ub, (r, s)) in [(sa[0], sb[0])]:
        pass
    best_overall = (0, None)
    for (ua, (p, q)) in sa[:8]:
        rb = (sb[[x[1] for x in sb].index((p, q))] if (p, q) in [x[1] for x in sb] else sb[0])[1]
        r, s = rb
        Pt = np.zeros((NG, NG, 2, 2))
        for ia in range(NG):
            for ib in range(NG):
                t = np.abs(res[ia, ib][kind][np.ix_([p, q], [r, s])]) ** 2; Pt[ia, ib] = t / t.sum()
        E = Pt[..., 0, 0] - Pt[..., 0, 1] - Pt[..., 1, 0] + Pt[..., 1, 1]; mA = Pt[..., 0, :].sum(-1); mB = Pt[..., :, 0].sum(-1)
        nonloc = np.abs(mB - mB[0:1, :]).max()     # зависимость статистики Боба от пластинки Алисы
        for tol in (0.005, 0.001):
            bS = 0
            for a0, a1 in itertools.combinations(range(NG), 2):
                for b0, b1 in itertools.combinations(range(NG), 2):
                    sig = max(abs(mA[a0, b0] - mA[a0, b1]), abs(mA[a1, b0] - mA[a1, b1]), abs(mB[a0, b0] - mB[a1, b0]), abs(mB[a0, b1] - mB[a1, b1]))
                    if sig > tol: continue
                    Sv = max(abs(E[a0, b0] + E[a0, b1] + E[a1, b0] - E[a1, b1]), abs(E[a0, b0] - E[a0, b1] + E[a1, b0] + E[a1, b1]),
                             abs(E[a0, b0] + E[a0, b1] - E[a1, b0] + E[a1, b1]), abs(-E[a0, b0] + E[a0, b1] + E[a1, b0] + E[a1, b1]))
                    bS = max(bS, Sv)
            if tol == 0.005: S5 = bS
            else: S1 = bS
        if S1 > best_overall[0] or best_overall[1] is None:
            best_overall = (S1, (ua, S5, S1, nonloc))
    ua, S5, S1, nonloc = best_overall[1]
    print(f'{"запаздывающий R (старое правило)" if kind == "R" else "SJ-вакуум W"}: C двух главных {C:.3f}; лучший делитель качество {ua:.3f}; '
          f'S = {S5:.3f} при сигнале ≤0.5 %, {S1:.3f} при ≤0.1 %; статистика Боба от пластинки Алисы меняется до {nonloc:.3f}')
