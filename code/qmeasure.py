#!/usr/bin/env python3
"""КВАНТОВАЯ МЕРА НА ИСТОРИЯХ (Соркин): Белл без отбора.

Нормируется не состояние на срезе, а мера всех историй мира — один раз. Исходы — где история кончилась:
  * поглощающие детекторы-области: у Алисы «p» (одна сторона её зеркальной оси) и «q» (зеркальная сторона),
    у Боба — их отражения P; история, коснувшаяся детектора, там и кончается (запись);
  * остальные истории кончаются на конечной границе мира (максимальные события мира, обрезанного по t ≤ T_f).
Пара: Ψ(x, y) = Σ_s ψ_s(x)ψ_s(y) (рождение в неизвестном месте полосы), мера пары исходов μ(x, y) = |Ψ(x, y)|²,
нормировка — сумма по ВСЕМ парам исходов (включая «не попала к детекторам» и «обе у одного»).
Исход Алисы: −1, если у неё в «q» частиц больше, чем в «p», иначе +1 (и «ничего не пришло» = +1, по Эберхарду).
Так CHSH считается БЕЗ пост-отбора: никакие истории не выбрасываются.
Правило — сумма историй с киральностью (лучшее для Белла): κ_c, ε = 0.5. Пластинки — фаза на звеньях,
пересекающих отрезок над вторым путём (как в fieldplate.py). Проверяем:
  1) эффективность η = доля меры «по частице у Алисы и у Боба»;
  2) S без отбора и S с отбором (только «по одной у каждого»), сигнал;
  3) согласованность меры: меняются ли вероятности, если мир продлить в будущее (T_f).
"""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from prop import propagate
sys.argv = _a

eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 8
kap = np.load('/home/claude/data/chisum_kc.npy', allow_pickle=True).item()[eps]
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R); LK = CH.prep(w, Lk, 1); X = w.X; n = w.n
I, J = LK['I'], LK['J']
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
A1 = [k for k in range(n) if abs(Xa[k, 0] - 2.54) < 0.01 and abs(Xa[k, 1] - 0.62) < 0.01][0]; A2 = CH.img(w, A1, H, Hs)
B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
# детекторы Алисы: полоса T̃ ∈ [4.5, 5.5] в её системе, |x̃| ≤ 3; p — x̃ > 0, q — зеркальные (h)
Dp = [k for k in range(n) if 4.5 <= Xa[k, 0] <= 5.5 and 0.05 < Xa[k, 1] <= 3.0]
Dq = [CH.img(w, k, H, Hs) for k in Dp]; assert min(Dq) >= 0
Ep = [CH.img(w, k, CH.P, CH.P) for k in Dp]; Eq = [CH.img(w, k, CH.P, CH.P) for k in Dq]
caus = sum(R[a, b] or R[b, a] for a in Dp + Dq for b in Ep + Eq)
sink = np.zeros(n, bool); sink[Dp + Dq + Ep + Eq] = True
src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
# пластинки: отрезки над A2 (в системе Алисы) и его отражение у Боба
def seg(Ac, Ao):
    tA = Xa[Ac, 0]; tX = tA + abs(Xa[Ac, 1] - Xa[Ao, 1]) / 2; tS = tA + 0.5 * (tX - tA); half = tS - tA
    t0, t1 = Xa[I, 0], Xa[J, 0]; cr = (t0 < tS) & (t1 >= tS)
    xc = Xa[I, 1] + (Xa[J, 1] - Xa[I, 1]) * (tS - t0) / np.where(t1 - t0 == 0, 1, t1 - t0)
    return cr & (np.abs(xc - Xa[Ac, 1]) <= half)
cutA = seg(A2, A1)
# у Боба — отражение P: звено (i, j) режется, если его образ режется у Алисы
pos = {(round(X[k, 0], 6), round(X[k, 1], 6)): k for k in range(n)}
linkid = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(I, J))}
cutB = np.zeros(len(I), bool)
for e in np.nonzero(cutA)[0]:
    a, b = CH.img(w, I[e], CH.P, CH.P), CH.img(w, J[e], CH.P, CH.P)
    if (a, b) in linkid: cutB[linkid[(a, b)]] = True
print(f'ε = {eps}, κ_c = {kap:.4f}: событий {n}; детекторов у Алисы {len(Dp)}+{len(Dq)}, у Боба {len(Ep)}+{len(Eq)}; '
      f'причинных связей детекторов Алиса–Боб {caus}; звеньев под пластинкой {cutA.sum()}/{cutB.sum()}'); sys.stdout.flush()

TFS = (6.0, 10.0, 14.0, 18.0)
def maximal(Tf):
    ok = (X[:, 0] <= Tf) & ~sink
    has_succ = np.zeros(n, bool)
    sel = ok[I] & ((X[J, 0] <= Tf))
    has_succ[I[sel]] = True
    return np.nonzero(ok & ~has_succ)[0]
MAXS = {Tf: maximal(Tf) for Tf in TFS}
classes_ev = {'Ap': Dp, 'Aq': Dq, 'Bp': Ep, 'Bq': Eq}
angles = np.linspace(0, 2 * math.pi, NG, endpoint=False)
out = {}
for ia, al in enumerate(angles):
    for ib, be in enumerate(angles):
        fld = np.where(cutA, np.exp(1j * al), 1.0) * np.where(cutB, np.exp(1j * be), 1.0)
        cols = []
        for s in src:
            for m in (0, 1):
                psi = propagate(w, LK, R, {s: np.eye(2)[m].astype(complex)}, eps=eps, unit=True, kappa=kap, field=fld, sinks=sink)
                cols.append(psi)
        cols = np.array(cols)                       # [источник, событие, компонента]
        res = {}
        for Tf in TFS:
            groups = dict(classes_ev); groups['F'] = list(MAXS[Tf])
            # Φ по группам: строки — (событие, компонента), столбцы — источники
            Phi = {g: cols[:, ev, :].reshape(len(cols), -1).T for g, ev in groups.items()}
            mu = {}
            for g1, g2 in itertools.product(groups, groups):
                Psi = Phi[g1] @ Phi[g2].T; mu[g1, g2] = (np.abs(Psi) ** 2).sum()
            res[Tf] = mu
        det = Dp + Dq + Ep + Eq
        Pd = cols[:, det, :].reshape(len(cols), -1).T; res['Psi_det'] = Pd @ Pd.T     # амплитуды пар по событиям детекторов
        out[ia, ib] = res
    print(f'  α = {math.degrees(al):.0f}° готово'); sys.stdout.flush()
np.save('/home/claude/data/qmeasure_out.npy', out, allow_pickle=True)

def outcome(g1, g2, side):
    cp = (g1 == side + 'p') + (g2 == side + 'p'); cq = (g1 == side + 'q') + (g2 == side + 'q')
    return -1 if cq > cp else 1
for Tf in TFS:
    tot = np.zeros((NG, NG)); Pab = np.zeros((NG, NG, 2, 2)); Ppost = np.zeros((NG, NG, 2, 2)); eta = np.zeros((NG, NG))
    for (ia, ib), res in out.items():
        mu = res[Tf]; T = sum(mu.values()); tot[ia, ib] = T
        for (g1, g2), v in mu.items():
            a, b = outcome(g1, g2, 'A'), outcome(g1, g2, 'B')
            Pab[ia, ib, (1 - a) // 2, (1 - b) // 2] += v / T
            one_each = sorted([g1[0], g2[0]]) == ['A', 'B']
            if one_each: Ppost[ia, ib, (1 - a) // 2, (1 - b) // 2] += v; eta[ia, ib] += v / T
        Ppost[ia, ib] /= Ppost[ia, ib].sum()
    def best(Pt, tol):
        E = Pt[..., 0, 0] - Pt[..., 0, 1] - Pt[..., 1, 0] + Pt[..., 1, 1]; mA = Pt[..., 0, :].sum(-1); mB = Pt[..., :, 0].sum(-1)
        bS = 0
        for a0, a1 in itertools.combinations(range(NG), 2):
            for b0, b1 in itertools.combinations(range(NG), 2):
                sig = max(abs(mA[a0, b0] - mA[a0, b1]), abs(mA[a1, b0] - mA[a1, b1]), abs(mB[a0, b0] - mB[a1, b0]), abs(mB[a0, b1] - mB[a1, b1]))
                if sig > tol: continue
                e = (E[a0, b0], E[a0, b1], E[a1, b0], E[a1, b1])
                bS = max(bS, abs(e[0] + e[1] + e[2] - e[3]), abs(e[0] - e[1] + e[2] + e[3]), abs(e[0] + e[1] - e[2] + e[3]), abs(-e[0] + e[1] + e[2] + e[3]))
        return bS
    # сигнал «в чистом виде»: маргинал Алисы от угла Боба
    mA = Pab[..., 0, :].sum(-1); sigA = np.abs(mA - mA[:, :1]).max()
    print(f'T_f = {Tf}: полная мера μ(Ω) от {tot.min():.3e} до {tot.max():.3e}; η (по частице у каждого) {eta.mean():.4f}; '
          f'S без отбора = {best(Pab, 1.0):.3f} (без требования сигнала), {best(Pab, 0.005):.3f} (сигнал ≤ 0.5 %); '
          f'S с отбором «по одной у каждого» = {best(Ppost, 0.005):.3f} (≤ 0.5 %), {best(Ppost, 0.001):.3f} (≤ 0.1 %); '
          f'маргинал Алисы от угла Боба до {sigA:.4f}')
