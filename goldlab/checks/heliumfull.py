"""Физический «гелий» всеми порядками: двигатель эволюции с вершиной, нормированной сшивкой (λ = e²/2 / (2 w_a w_b)),
против непрерывной задачи двух тел (конфигурационное взаимодействие в базисе орбиталей Дирака положительной энергии)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
e2, Z, m, rho = 0.2, 2.0, 1.0, 4.0; be = e2 / 2
# ---- непрерывный эталон: КВ (без пар), базис nb орбиталей положительной энергии ----
N, Lg = 512, 64.0; xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
Dm = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)); V = np.diag(Z * e2 / 2 * np.abs(xg))
ev, vec = np.linalg.eigh(np.block([[-1j * Dm + V, m * np.eye(N)], [m * np.eye(N), 1j * Dm + V]]))
loc = lambda i: (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) @ xg ** 2
pos = [i for i in np.argsort(ev) if ev[i] > 0.5 and loc(i) < 150][:40]; print('базис КВ:', len(pos), 'состояний, E до', round(float(ev[pos[-1]]), 3), flush=True); E = ev[pos]; P = vec[:, pos]; u, v = P[:N], P[N:]
T = np.einsum('xi,xj->ijx', np.conj(u), u) + np.einsum('xi,xj->ijx', np.conj(v), v)          # T_ij(x) = ψ_i†ψ_j
Ux = -be * np.abs(xg[:, None] - xg[None])
nb = len(pos); Tf = T.reshape(nb * nb, N); M = (Tf @ Ux @ Tf.T).reshape(nb, nb, nb, nb)            # ⟨ij|V|kl⟩ = ∫∫T_ik(x)U T_jl(x') → M[i,k,j,l]
for nbas in sorted({min(10, nb), min(20, nb), nb}):
    pairsA = [(i, j) for i in range(nbas) for j in range(i + 1, nbas)]; pairsS = [(i, j) for i in range(nbas) for j in range(i, nbas)]
    def Hmat(pairs, sgn):
        n = len(pairs); H = np.zeros((n, n), complex)
        for p, (i, j) in enumerate(pairs):
            ni = 1 / np.sqrt(2) if (i != j) else 0.5
            for q, (k, l) in enumerate(pairs):
                nk = 1 / np.sqrt(2) if (k != l) else 0.5
                if i == j and sgn > 0: ni_ = 1.0 / np.sqrt(2) * np.sqrt(2) / 2
                val = M[i, k, j, l] + sgn * M[i, l, j, k] + sgn * M[j, k, i, l] + M[j, l, i, k]
                H[p, q] = (ni if i != j else 0.5) * (nk if k != l else 0.5) * val * (2 if (i == j or k == l) and sgn > 0 and not (i == j and k == l) else 1) if False else 0
        return H
    # прямая сборка через полные двухчастичные векторы (надёжнее нормировок)
    def ground(pairs, sgn):
        n = len(pairs); basis = np.zeros((n, nbas, nbas))
        for p, (i, j) in enumerate(pairs):
            basis[p, i, j] += 1; basis[p, j, i] += sgn
            basis[p] /= np.linalg.norm(basis[p])
        M2 = M[:nbas, :nbas, :nbas, :nbas].transpose(0, 2, 1, 3).reshape(nbas * nbas, nbas * nbas)   # [(i,j),(k,l)]
        B = basis.reshape(n, nbas * nbas); Vb = B @ M2 @ B.T
        H0 = np.diag([E[i] + E[j] for (i, j) in pairs]); w_ = np.linalg.eigvalsh(H0 + Vb); return w_[0], H0[0, 0] + Vb[0, 0]
    EF, EF1 = ground(pairsA, -1); EB, EB1 = ground(pairsS, +1)
    print(f'КВ базис {nbas}: фермионы {EF:.4f} (первый порядок {EF1:.4f}), бозоны оба внизу {EB:.4f} (первый порядок {EB1:.4f}); E0 + E1 = {E[0] + E[1]:.4f}, 2E0 = {2 * E[0]:.4f}', flush=True)
# ---- кристалл ----
W_ = {24: (-1.841, -1.617), 26: (-1.569, -1.496)}
for L in (24.0, 26.0):
    w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n; ts = -L + 10; UfN, _, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4))
    K, C = Fe.dirac_component_K(w, UfN, m); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); src = np.abs(t - ts) < 0.45
    fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb
    om = np.linspace(0.8, 2.0, 2401); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); pk = []
    for g_, p_ in ((gs, pa), (gs1, pb)):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)
        for e_, h_ in Ob.peaks(om, S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    pk = sorted(pk); lev = [e for e, _ in pk]; near = lambda Ex: min(lev, key=lambda z: abs(z - Ex)); E0c, E1c = near(E[0]), near(E[1])
    xc, amp = Fe.crystal_orbitals(w, pa + pb, lev, ts + 1.5, L - 4)
    def hfun(a_):
        tab = dict(zip(np.round(xc, 6), np.conj(a_))); return np.array([tab.get(v_, 0) for v_ in np.round(x, 6)])
    ha, hb = hfun(amp[lev.index(E0c)]), hfun(amp[lev.index(E1c)])
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v_) < 1e-9)[0] for v_ in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v_) < 1e-9] for v_ in tR]; ow = np.linspace(1.8, 3.8, 4001)
    wa, wb = W_[int(L)]
    for name, Fm, P1, P2, sgn, Et, lam in (('фермионы', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0c + E1c, be / (2 * wa * wb)), ('бозоны оба внизу', np.outer(fa, fa), ha, ha, 0, 2 * E0c, be / (2 * wa * wa))):
        res = {}
        for l_ in (0.0, lam):
            A = Fe.pair_ladder(K, inc, layers, Fm, l_)
            s = np.array([(np.outer(P1[Z_], P2[Z_]) * A[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * A[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR]); del A
            S = Ob.spectrum(tR, s, ow); S /= S.max(); res[l_] = min(Ob.peaks(ow, S, 0.05), key=lambda z: abs(z[0] - (Et + (-0.26 if l_ else 0))))[0]
        print(f'L={L:.0f} {name}: λ_физ = {lam:.5f}; пик без взаимодействия {res[0.0]:.4f}, с взаимодействием {res[lam]:.4f}, сдвиг {res[lam] - res[0.0]:+.4f}', flush=True)
