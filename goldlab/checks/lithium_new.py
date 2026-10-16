"""Очередь п. 2: «литий» на новой основе. Ядро +3 (поле из порядка), три электрона — компоненты Дирака по световому уголку.
Энергия: Σ уровней кристалла + Σ сдвигов трёх пар (двигатель эволюции, векторная вершина, все порядки внутри пары).
Эталон: непрерывная задача трёх тел без пар (КВ) и её «парное приближение»."""
import sys, itertools, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, observe as Ob, fermions as Fe
e2, Z, m, rho = 0.2, 3.0, 1.0, 4.0; be = e2 / 2; Fn = Z * e2 / 2
# ---------- непрерывный эталон ----------
N_, Lg = 512, 64.0; xg = (np.arange(N_) - N_ / 2) * (Lg / N_); kf = 2 * np.pi * np.fft.fftfreq(N_, d=Lg / N_)
Dm = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N_), axis=0), axis=0)); Vd = np.diag(Fn * np.abs(xg))
ev, vec = np.linalg.eigh(np.block([[-1j * Dm + Vd, m * np.eye(N_)], [m * np.eye(N_), 1j * Dm + Vd]]))
loc = lambda i: (np.abs(vec[:N_, i]) ** 2 + np.abs(vec[N_:, i]) ** 2) @ xg ** 2
pos = [i for i in np.argsort(ev) if ev[i] > 0.5 and loc(i) < 150][:14]; E = ev[pos]; u, v = vec[:N_, pos], vec[N_:, pos]; nb = len(pos)
T = (np.einsum('xi,xj->ijx', np.conj(u), u) + np.einsum('xi,xj->ijx', np.conj(v), v)).reshape(nb * nb, N_)
M = (T @ (-be * np.abs(xg[:, None] - xg[None])) @ T.T).reshape(nb, nb, nb, nb)        # M[i,k,j,l] = ∫∫T_ik U T_jl
V2 = M.transpose(0, 2, 1, 3).reshape(nb * nb, nb * nb)                                # ⟨ij|V|kl⟩
I = np.eye(nb); H1 = np.diag(E)
# трёхчастичный гамильтониан в пространстве nb³, проекция на антисимметричные
dets = list(itertools.combinations(range(nb), 3)); D = np.zeros((len(dets), nb ** 3))
for p, (a, b, c) in enumerate(dets):
    for perm, sg in ((( a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1), ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
        D[p, perm[0] * nb * nb + perm[1] * nb + perm[2]] = sg / np.sqrt(6)
V12 = np.kron(V2, I); P23 = np.eye(nb ** 3).reshape(nb, nb, nb, nb, nb, nb).transpose(0, 2, 1, 3, 5, 4).reshape(nb ** 3, nb ** 3)
P13 = np.eye(nb ** 3).reshape(nb, nb, nb, nb, nb, nb).transpose(2, 1, 0, 5, 4, 3).reshape(nb ** 3, nb ** 3)
Hd = np.real(D @ (V12 + P23 @ V12 @ P23 + P13 @ V12 @ P13) @ D.T) + np.diag([E[a] + E[b] + E[c] for a, b, c in dets])
w3 = np.linalg.eigvalsh(Hd); E3_1 = Hd[0, 0]
print(f'непр.: уровни {E[0]:.4f} {E[1]:.4f} {E[2]:.4f}; сумма {E[0] + E[1] + E[2]:.4f}; первый порядок {E3_1:.4f}; трёх тел (все порядки) {w3[0]:.4f}', flush=True)
# парное приближение в непрерывном: сдвиги пар (i,j) из задачи двух тел
def pair2(i, j):
    pairs = [(a, b) for a in range(nb) for b in range(a + 1, nb)]; B = np.zeros((len(pairs), nb, nb))
    for q, (a, b) in enumerate(pairs):
        B[q, a, b], B[q, b, a] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    B = B.reshape(len(pairs), nb * nb); H = np.diag([E[a] + E[b] for a, b in pairs]) + np.real(B @ V2 @ B.T); wv, U_ = np.linalg.eigh(H)
    k = int(np.argmax(np.abs(U_[pairs.index((i, j))]) ** 2)); return wv[k] - E[i] - E[j]
pc = {ij: pair2(*ij) for ij in ((0, 1), (0, 2), (1, 2))}
print(f'непр. сдвиги пар (все порядки): 01 {pc[(0, 1)]:+.4f}, 02 {pc[(0, 2)]:+.4f}, 12 {pc[(1, 2)]:+.4f}; парное приближение {E[0] + E[1] + E[2] + sum(pc.values()):.4f}', flush=True)
# бозоны все внизу (для подъёма Паули): первый порядок 3E0 + 3J00
J00 = np.real(M[0, 0, 0, 0]); print(f'непр. бозоны все внизу (первый порядок) {3 * E[0] + 3 * J00:.4f}', flush=True)
# ---------- кристалл ----------
for L in (24.0, 26.0):
    w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n; ts = -L + 10
    UfN, _, Uev, (_, _, FN) = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); K, C = Fe.dirac_component_K(w, UfN, m); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    srcf = [lambda s: (1 + s) * np.exp(-s * s / 2), lambda s: s * np.exp(-s * s / 8), lambda s: (1 - s * s / 3) * np.exp(-s * s / 10)]
    srcm = np.abs(t - ts) < 0.45; fs = [np.where(srcm, g(x), 0) for g in srcf]; ps = [K.T @ f for f in fs]
    om = np.linspace(0.8, 2.6, 3601); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); pk = []
    for g_, p_ in zip(srcf, ps):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)
        for e_, h_ in Ob.peaks(om, S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    lev = [e for e, _ in sorted(pk)]; near = lambda Ex: min(lev, key=lambda z: abs(z - Ex)); Ec = [near(E[k]) for k in range(3)]
    print(f'L={L:.0f}: наклон ядра {FN:.4f}; уровни кристалла {Ec[0]:.4f} {Ec[1]:.4f} {Ec[2]:.4f} (пики {", ".join(f"{e:.3f}" for e in lev)})', flush=True)
    xc, amp = Fe.crystal_orbitals(w, ps[0] + ps[1] + ps[2], lev, ts + 1.5, L - 4)
    def hfun(a_):
        tab = dict(zip(np.round(xc, 6), np.conj(a_))); return np.array([tab.get(v_, 0) for v_ in np.round(x, 6)])
    hs = [hfun(amp[lev.index(Ec[k])]) for k in range(3)]
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v_) < 1e-9)[0] for v_ in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v_) < 1e-9] for v_ in tR]; ow = np.linspace(1.8, 4.6, 5601); sh = {}
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        Fm = np.outer(fs[i], fs[j]) - np.outer(fs[j], fs[i]); nz = 0.5 * (Ec[i] + Ec[j]) - Uev; Wm = (e2 / rho ** 2) * np.outer(nz, nz) * inc; res = {}
        for on in (0, 1):
            A = np.zeros((n, n), complex); H = np.zeros((n, n), complex)
            for Zl in layers:
                A[Zl] = K[:, Zl].T @ H; G = Fm[Zl] + (1j * Wm[Zl] * A[Zl] if on else 0)
                if np.any(G): H[Zl] = G @ K
            s = np.array([(np.outer(hs[i][Z_], hs[j][Z_]) * A[np.ix_(Z_, Z_)]).sum() - (np.outer(hs[j][Z_], hs[i][Z_]) * A[np.ix_(Z_, Z_)]).sum() for Z_ in layR]); del A, H
            S = Ob.spectrum(tR, s, ow); S /= S.max(); res[on] = min(Ob.peaks(ow, S, 0.03), key=lambda z: abs(z[0] - (Ec[i] + Ec[j] + pc[(i, j)] * on)))[0]
        sh[(i, j)] = res[1] - res[0]; del Wm
        print(f'   пара {i}{j}: {res[0]:.4f} → {res[1]:.4f}, сдвиг {sh[(i, j)]:+.4f} (непр. {pc[(i, j)]:+.4f})', flush=True)
    Etot = sum(Ec) + sum(sh.values())
    print(f'ИТОГ L={L:.0f}: «литий» кристалл {Etot:.4f} (сумма уровней {sum(Ec):.4f} + пары {sum(sh.values()):+.4f}); непр.: трёх тел {w3[0]:.4f}, парное {E[0] + E[1] + E[2] + sum(pc.values()):.4f}, первый порядок {E3_1:.4f}', flush=True)
