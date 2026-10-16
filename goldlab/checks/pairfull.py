"""Очередь п. 1, шаг 2: полная лестница (все порядки) для пары в поле ядра — прямая развёртка по слоям времени, без GMRES.
A = Kᵀ (F + W∘A) K, W = i·λ·[несравнимы]. Строка A[y₁] зависит только от более ранних строк ⇒ слой за слоем:
A[слой] = K[:, слой]ᵀ H,  G[слой] = F[слой] + W[слой]∘A[слой],  H[слой] = G[слой] K.  Цена ~2n³.
Измеритель — спектроскопия пика пары (как у одного тела), сдвиг пика при нескольких λ."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe, causal as Cs
L = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0; lams = [float(v) for v in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0', '0.001', '-0.001', '0.002'])]
e2 = 0.2; Z = 2.0; m = 1.0; be = e2 / 2; T0 = time.time(); lg = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
w = Wd.gold11(4.0, L); t, x, X, n = w.t, w.x, w.X, w.n; ts = -L + 10; lg('n =', n)
UfN, dUN, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4))
C = Cs.causal(X); I, J = np.nonzero(C); Phi = np.zeros((n, n), complex)
for s0 in range(0, len(I), 400000):
    sl = slice(s0, s0 + 400000); Phi[I[sl], J[sl]] = 0.5 * np.exp(1j * Pr.corner_phase_general(UfN, t[I[sl]], x[I[sl]], t[J[sl]], x[J[sl]], 1))
K = np.linalg.solve((np.eye(n) - Phi * (-m * m / 4.0)).T, Phi.T).T; del Phi, I, J; lg('K готов')
inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); src = np.abs(t - ts) < 0.45
fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb
om = np.linspace(0.8, 2.0, 2401); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); pk = []
for g_, p_ in ((gs, pa), (gs1, pb)):
    for e_, h_ in Ob.peaks(om, (lambda S: S / S.max())(Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)), 0.03):
        if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
pk = sorted(pk); xg, Ec, _, dens, vec = Fe.dirac_orbitals(lambda s: Z * e2 / 2 * np.abs(s), 1.0, m, nlev=2, N=512, Lg=60, locmax=40)
near = lambda E: min(pk, key=lambda z: abs(z[0] - E))[0]; E0, E1 = near(Ec[0]), near(Ec[1]); lg(f'E0 {E0:.4f} ({Ec[0]:.4f})  E1 {E1:.4f} ({Ec[1]:.4f})')
xc, amp = Fe.crystal_orbitals(w, pa + pb, [e for e, _ in pk], ts + 1.5, L - 4); ia, ib = [e for e, _ in pk].index(E0), [e for e, _ in pk].index(E1)
def hfun(a):
    tab = dict(zip(np.round(xc, 6), np.conj(a))); return np.array([tab.get(v, 0) for v in np.round(x, 6)])
ha, hb = hfun(amp[ia]), hfun(amp[ib])
lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v) < 1e-9)[0] for v in lv]
R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v) < 1e-9] for v in tR]
def ladder(Fm, lam):
    A = np.zeros((n, n), complex); H = np.zeros((n, n), complex)
    for Zl in layers:
        A[Zl] = K[:, Zl].T @ H
        G = Fm[Zl] + 1j * lam * inc[Zl] * A[Zl]
        if np.any(G): H[Zl] = G @ K
    return A
def readout(A, P1, P2, sgn):
    return np.array([(np.outer(P1[Z_], P2[Z_]) * A[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * A[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR])
ow = np.linspace(1.8, 3.8, 4001)
for name, Fm, P1, P2, sgn, Et in (('фермионы a∧b', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0 + E1), ('бозоны a∨b', np.outer(fa, fb) + np.outer(fb, fa), ha, hb, 1, E0 + E1),
                                  ('бозоны a·a', np.outer(fa, fa), ha, ha, 0, 2 * E0)):
    res = []
    for lam in lams:
        A = ladder(Fm, lam); s = readout(A, P1, P2, sgn); del A; S = Ob.spectrum(tR, s, ow); S /= S.max()
        p = min([q for q in Ob.peaks(ow, S, 0.05)], key=lambda z: abs(z[0] - Et)); res.append((lam, p[0], p[1], float(np.abs(s).max())))
        lg(f'{name}: λ = {lam:+.4f}  пик {p[0]:.4f} (высота {p[1]:.2f})  |s|max {np.abs(s).max():.2e}')
    l0 = [r for r in res if r[0] == 0][0][1]; sl = [(r[1] - l0) / r[0] for r in res if r[0] != 0]
    lg(f'ИТОГ {name} L={L}: наклоны dE/dλ = {", ".join(f"{v:+.2f}" for v in sl)}')
