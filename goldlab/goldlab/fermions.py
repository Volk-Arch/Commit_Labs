"""ФЕРМИОНЫ И АТОМЫ (dirac-doc). Частица со спином в 1+1 — унитарное блуждание по «световым нитям» кристалла:
каждое событие лежит на одной нити вправо и одной влево (паросочетание по порядку), в событии — монета
[[cos θ, −i sin θ], [−i sin θ, cos θ]], θ = m·Δt̄. Поле — фаза e^{−i∫V dt} на шаге нити. Две частицы — амплитуда по парам «рёбер»
с фазой взаимодействия между слоями времени. Самосогласованное поле (Хартри) — уровни и орбитали из блуждания.
Взято руками (см. experiments.HAND): правило нитей (система покоя кристалла, скорость c ≈ 0.76), монета, масса.
Без нитей и монеты — dirac-doc §12: компонента Дирака = propagate.corner_respond(..., side=±1, ins=0) (опыт lightspin)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from . import observe as Ob

def threads(X, side, cap=1.6, big=1e3):
    """нить вправо (side = +1) или влево (−1): событие → «позже»-событие с Δx·side > 0, Δt < cap; цена — собственное время."""
    n = len(X); t, x = X[:, 0], X[:, 1]; rows, cols, vals = [], [], []
    for i in range(n):
        j0 = np.searchsorted(t, t[i] + 1e-9); j1 = np.searchsorted(t, t[i] + cap)
        js = np.arange(j0, j1); dx = x[js] - x[i]; dt = t[js] - t[i]; ok = (dx * side > 1e-9) & (dt > np.abs(dx) + 1e-12); js = js[ok]
        rows += [i] * len(js); cols += list(js); vals += list(np.sqrt(dt[ok] ** 2 - dx[ok] ** 2) + 1e-3 * dt[ok] + 1.0)
    rows += list(range(n)); cols += list(range(n, 2 * n)); vals += [big] * n
    r, c = min_weight_full_bipartite_matching(coo_matrix((vals, (rows, cols)), shape=(n, 2 * n)).tocsr())
    nxt = -np.ones(n, int); nxt[r] = c; nxt[nxt >= n] = -1; return nxt

class Walk:
    def __init__(self, w, cap=1.6):
        X = w.X; self.w = w; t = X[:, 0]; n = w.n; self.R = threads(X, +1, cap); self.L = threads(X, -1, cap)
        self.pR = -np.ones(n, int); self.pR[self.R[self.R >= 0]] = np.nonzero(self.R >= 0)[0]
        self.pL = -np.ones(n, int); self.pL[self.L[self.L >= 0]] = np.nonzero(self.L >= 0)[0]
        dR = np.where(self.pR >= 0, t - t[np.maximum(self.pR, 0)], np.nan); dL = np.where(self.pL >= 0, t - t[np.maximum(self.pL, 0)], np.nan)
        both = np.c_[dR, dL]; cnt = np.sum(~np.isnan(both), 1); med = np.nanmedian(both)
        self.dtb = np.where(cnt > 0, np.nansum(np.nan_to_num(both), 1) / np.maximum(cnt, 1), med)
        sR = X[self.R[self.R >= 0]] - X[self.R >= 0]; sL = X[self.L[self.L >= 0]] - X[self.L >= 0]
        self.c = 0.5 * (np.mean(sR[:, 1] / sR[:, 0]) - np.mean(sL[:, 1] / sL[:, 0]))
        inner = np.abs(t) + np.abs(X[:, 1]) < 0.8 * np.abs(t).max()
        self.full = float((inner & (self.R >= 0) & (self.L >= 0) & (self.pR >= 0) & (self.pL >= 0)).sum() / inner.sum())
    def phases(self, Vint):
        X = self.w.X; out = []
        for nx in (self.R, self.L):
            p = np.ones(self.w.n, complex); ok = nx >= 0; p[ok] = np.exp(-1j * Vint(X[ok, 0], X[ok, 1], X[nx[ok], 0], X[nx[ok], 1])); out.append(p)
        return out
    def run(self, m, fR, fL, ph=None):
        n = self.w.n; aR = fR.astype(complex).copy(); aL = fL.astype(complex).copy(); leak = 0.0
        pR, pL = ph if ph is not None else (np.ones(n), np.ones(n)); c = np.cos(m * self.dtb); s = np.sin(m * self.dtb)
        for z in range(n):
            bR = c[z] * aR[z] - 1j * s[z] * aL[z]; bL = -1j * s[z] * aR[z] + c[z] * aL[z]
            if self.R[z] >= 0: aR[self.R[z]] += bR * pR[z]
            else: leak += abs(bR) ** 2
            if self.L[z] >= 0: aL[self.L[z]] += bL * pL[z]
            else: leak += abs(bL) ** 2
        return aR, aL, leak

def step_integral(V, k=8):
    """∫V(x(t)) dt по прямому шагу (Гаусс–Лежандр)."""
    gx, gw = np.polynomial.legendre.leggauss(k)
    def f(t0, x0, t1, x1):
        tot = 0
        for g, wg in zip(gx, gw):
            q = 0.5 * (g + 1); tot = tot + 0.5 * wg * V(x0 + q * (x1 - x0))
        return tot * (t1 - t0)
    return f

def readout(w, ts, L, xmax=9.0):
    t, x = w.t, w.x; sel = (t > ts + 2) & (t < L - 4) & (np.abs(x) < xmax); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
    def proj(h, aR, aL):
        s = np.zeros(len(tu), complex); np.add.at(s, inv, (h * (aR + aL))[sel]); return s
    return sel, tu, proj

def energies(tu, s, lo, hi, thr=0.1, npts=2601):
    """уровни ветви частицы (вращение e^{−iEt}) в полосе [lo, hi]."""
    om = np.linspace(lo, hi, npts); S = Ob.spectrum(tu, s, -om); S = S / max(S.max(), 1e-300); return Ob.peaks(om, S, thr)

def dirac_orbitals(Vfun, c, m=1.0, nlev=3, N=512, Lg=64.0, locmax=40.0):
    """непрерывный Дирак H = −i c σ_z ∂x + m σ_x + V (спектральная производная): уровни частицы и формы (u + v)."""
    xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
    D = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)); V = np.diag(Vfun(xg))
    ev, vec = np.linalg.eigh(np.block([[-1j * c * D + V, m * np.eye(N)], [m * np.eye(N), 1j * c * D + V]]))
    loc = lambda i: (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) @ xg ** 2
    idx = [i for i in np.argsort(ev) if ev[i] > 0 and loc(i) < locmax][:nlev]; shapes, dens = [], []
    for i in idx:
        f = vec[:N, i] + vec[N:, i]; f = np.real(f * np.exp(-1j * np.angle(f[np.argmax(np.abs(f))]))); shapes.append(f / np.abs(f).max())
        dens.append((np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) / h)
    return xg, ev[idx], shapes, dens, vec[:, idx]

def pair_run(W, ph, m, ts, L, U, he, ho, sgn=-1, x1=-2.0, x2=2.0, sw=3.0):
    """две частицы: амплитуда A[e₁, e₂] по рёбрам (событие + направление); U(Δx) — взаимодействие между слоями (или None).
    Старт ψ₁ ⊗ ψ₂ + sgn·ψ₂ ⊗ ψ₁ (sgn = −1 — фермионы). Чтение: «нижняя + следующая» (he ⊗ ho + sgn·ho ⊗ he) и «обе внизу» (he ⊗ he)."""
    w = W.w; t, x, n = w.t, w.x, w.n; E = 2 * n; nxt = np.empty(E, int); nxt[0::2] = W.R; nxt[1::2] = W.L
    phe = np.empty(E, complex); phe[0::2] = ph[0]; phe[1::2] = ph[1]; cz, sz = np.cos(m * W.dtb), np.sin(m * W.dtb)
    tS = np.repeat(t, 2); xS = np.repeat(x, 2); tE = np.where(nxt >= 0, t[np.maximum(nxt, 0)], np.inf); xE = np.where(nxt >= 0, x[np.maximum(nxt, 0)], 0)
    lv = np.unique(np.round(t, 9)); A = np.zeros((E, E), np.complex64); act = []; s1, s2, tt = [], [], []
    for k, tl in enumerate(lv):
        Z = np.nonzero(np.abs(t - tl) < 1e-9)[0]
        if ts + 1.5 < tl < L - 4:
            inR = np.where(W.pR[Z] >= 0, 2 * W.pR[Z], -1); inL = np.where(W.pL[Z] >= 0, 2 * W.pL[Z] + 1, -1); amp = np.zeros((len(Z), len(Z)), complex)
            for i1 in (inR, inL):
                for i2 in (inR, inL):
                    o1, o2 = i1 >= 0, i2 >= 0; blk = np.zeros((len(Z), len(Z)), complex); blk[np.ix_(o1, o2)] = A[np.ix_(i1[o1], i2[o2])]; amp += blk
            xz = x[Z]; s1.append(((np.outer(he(xz), ho(xz)) + sgn * np.outer(ho(xz), he(xz))) * amp).sum()); s2.append((np.outer(he(xz), he(xz)) * amp).sum()); tt.append(tl)
        for z in Z:
            eR = 2 * W.pR[z] if W.pR[z] >= 0 else -1; eL = 2 * W.pL[z] + 1 if W.pL[z] >= 0 else -1
            for M in (A, A.T):
                vR = M[eR].copy() if eR >= 0 else 0; vL = M[eL].copy() if eL >= 0 else 0
                if eR >= 0: M[eR] = 0
                if eL >= 0: M[eL] = 0
                M[2 * z] = (cz[z] * vR - 1j * sz[z] * vL) * phe[2 * z] if nxt[2 * z] >= 0 else 0
                M[2 * z + 1] = (-1j * sz[z] * vR + cz[z] * vL) * phe[2 * z + 1] if nxt[2 * z + 1] >= 0 else 0
        if abs(tl - ts) < 1e-9:
            f1 = np.exp(-(x[Z] - x1) ** 2 / sw); f2 = np.exp(-(x[Z] - x2) ** 2 / sw); S = np.outer(f1, f2) + sgn * np.outer(f2, f1)
            for d1 in (0, 1):
                for d2 in (0, 1):
                    e1 = 2 * Z + d1; e2 = 2 * Z + d2; A[np.ix_(e1, e2)] += (0.5 * S * np.outer(phe[e1], phe[e2])).astype(np.complex64)
        act = [e for e in act if tE[e] > tl + 1e-9] + [e for z in Z for e in (2 * z, 2 * z + 1) if nxt[e] >= 0]
        if U is not None and k + 1 < len(lv) and act:
            t1 = lv[k + 1]; tm = 0.5 * (tl + t1); ac = np.array(act); xm = xS[ac] + (xE[ac] - xS[ac]) * (tm - tS[ac]) / (tE[ac] - tS[ac])
            A[np.ix_(ac, ac)] *= np.exp(-1j * U(xm[:, None] - xm[None, :]) * (t1 - tl)).astype(np.complex64)
    return np.array(tt), np.array(s1), np.array(s2)

# ---------- «гелий» без нитей (dirac-doc §14): всё поле из порядка, электроны — компоненты Дирака по световому уголку ----------
def crystal_orbitals(w, phi, levels, t0, t1, xmax=10.0):
    """орбитали из отклика φ: для каждого естественного значения x кристалла — МНК по частотам уровней (обе ветви) + медленная часть.
    Плотность — |ψ_R(x)|² + |ψ_R(−x)|² (чётность Дирака: ψ_L(x) = ±ψ_R(−x)); набор x кристалла симметричен."""
    t, x = w.t, w.x; win = (t > t0) & (t < t1); xc = np.unique(np.round(x[win & (np.abs(x) < xmax)], 6)); amp = np.zeros((len(levels), len(xc)), complex)
    for k, xv in enumerate(xc):
        s = win & (np.abs(x - xv) < 1e-6); tt = t[s]
        if len(tt) < 6: continue
        A = np.c_[np.exp(1j * np.outer(tt, levels)), np.exp(-1j * np.outer(tt, levels)), np.ones_like(tt), tt]
        amp[:, k] = np.linalg.lstsq(A, phi[s], rcond=None)[0][:len(levels)]
    return xc, amp

def helium_order(w, e2=0.2, Z=2.0, m=1.0, log=print):
    from . import fields as Fl, propagate as Pr
    t, x, X = w.t, w.x, w.X; L = max(np.abs(t).max(), np.abs(x).max()); ts = -L + 10; be = e2 / 2; out = {}
    UN = lambda s: Z * e2 / 2 * np.abs(s); xg, Ec, _, dens, vec = dirac_orbitals(UN, 1.0, m, nlev=2, N=512, Lg=60, locmax=40); N = len(xg); h = xg[1] - xg[0]
    Kx = lambda a: -be * np.abs(a[:, None] - a[None])
    rc0, rc1 = (d / np.trapezoid(d, xg) for d in dens); Tc = (np.conj(vec[:N, 0]) * vec[:N, 1] + np.conj(vec[N:, 0]) * vec[N:, 1]) / h
    Jc = float(rc0 @ Kx(xg) @ rc1 * h * h); Kc = float(np.real(Tc @ Kx(xg) @ np.conj(Tc)) * h * h); J00c = float(rc0 @ Kx(xg) @ rc0 * h * h)
    hc = []
    for i, rr in ((0, rc1), (1, rc0)):
        Ue = np.trapezoid(rr[None] * -be * np.abs(xg[:, None] - xg[None]), xg, axis=1)
        hc.append(dirac_orbitals(lambda s: UN(s) + np.interp(s, xg, Ue), 1.0, m, nlev=2, N=512, Lg=60, locmax=40)[1][i])
    UfN, dUN, _, (_, _, FN) = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); log(f'ядро: наклон {FN:.4f}')
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); om = np.linspace(m - 0.1, m + 1.0, 2201); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8)
    def run(Uf, dU, src, thr):
        phi = Pr.corner_respond_general(X, Uf, dU, m, 4.0, np.where(np.abs(t - ts) < 0.45, src(x), 0), 1, 0.0)
        S = Ob.spectrum(t[sel], src(x[sel]) * phi[sel], om); S /= S.max(); return phi, sorted(Ob.peaks(om, S, thr)), S
    near = lambda pk, E: min(pk, key=lambda z: abs(z[0] - E))[0]
    phi, pk, S0 = run(UfN, dUN, gs, 0.03); E0, E1 = near(pk, Ec[0]), near(pk, Ec[1]); log(f'уровни в поле ядра {E0:.4f} {E1:.4f}')
    xc, amp = crystal_orbitals(w, phi, [e for e, _ in pk], ts + 1.5, L - 4); i0, i1 = [e for e, _ in pk].index(E0), [e for e, _ in pk].index(E1)
    tr = lambda f: float(np.trapezoid(f, xc)); wq = np.gradient(xc)
    def dn(ps): r = np.abs(ps) ** 2 + np.abs(ps[::-1]) ** 2; return r / tr(r), np.sqrt(tr(r))
    (r0, n0), (r1, n1) = dn(amp[i0]), dn(amp[i1]); fid = [tr(np.sqrt(r * np.interp(xc, xg, rc))) for r, rc in ((r0, rc0), (r1, rc1))]
    a0, a1 = amp[i0] / n0, amp[i1] / n1; T = (np.conj(a0) * a1 - np.conj(a0[::-1]) * a1[::-1]) * wq
    J = float((r0 * wq) @ Kx(xc) @ (r1 * wq)); K = float(np.real(T @ Kx(xc) @ np.conj(T))); J00 = float((r0 * wq) @ Kx(xc) @ (r0 * wq))
    log(f'орбитали: совпадение {fid[0]:.4f} {fid[1]:.4f}; J {J:+.4f} K {K:+.4f} J00 {J00:+.4f}'); Eh = []; cs = []; spec = [S0]
    for lev, ro, src, thr, Eexp in ((0, r1, gs, 0.03, hc[0]), (1, r0, gs1, 0.02, hc[1])):
        Uc, dUc, c, _ = Fl.order_cloud11(w, lambda s, ro=ro: np.interp(s, xc, ro, left=0, right=0), e2, ts)
        _, pk2, S2 = run(lambda a, b, Uc=Uc: UfN(a, b) + Uc(a, b), dUN + dUc, src, thr); Eh.append(near(pk2, Eexp)); cs.append(c); spec.append(S2)
        log(f'уровень {lev} в поле облака {1 - lev}: {Eh[-1]:.4f} (непр. {Eexp:.4f}), сила облака {c:.4f}')
    return dict(Ec=Ec, E=(E0, E1), FN=FN, fid=fid, J=J, K=K, J00=J00, Jc=Jc, Kc=Kc, J00c=J00c, Eh=Eh, hc=hc, cs=cs, om=om, spec=spec)

# ---------- две частицы с взаимодействием как динамика (dirac-doc §15): полная лестница прямой развёрткой по слоям ----------
def dirac_component_K(w, Uf, m=1.0, rho=4.0, side=1):
    """плотный K компоненты Дирака: световой уголок с фазой поля Uf(t, x), без вставки (dirac-doc §12)."""
    from .causal import causal
    from .propagate import corner_phase_general
    t, x, n = w.t, w.x, w.n; C = causal(w.X); I, J = np.nonzero(C); Phi = np.zeros((n, n), complex)
    for s0 in range(0, len(I), 400000):
        sl = slice(s0, s0 + 400000); Phi[I[sl], J[sl]] = 0.5 * np.exp(1j * corner_phase_general(Uf, t[I[sl]], x[I[sl]], t[J[sl]], x[J[sl]], side))
    return np.linalg.solve((np.eye(n) - Phi * (-m * m / rho)).T, Phi.T).T, C

def pair_ladder(K, inc, layers, Fm, lam):
    """A = Kᵀ(F + W∘A)K, W = i·λ·[несравнимы]; строка A[y₁] зависит только от более ранних — развёртка слоями, цена ~2n³."""
    n = K.shape[0]; A = np.zeros((n, n), complex); H = np.zeros((n, n), complex)
    for Zl in layers:
        A[Zl] = K[:, Zl].T @ H; G = Fm[Zl] + 1j * lam * inc[Zl] * A[Zl]
        if np.any(G): H[Zl] = G @ K
    return A

def pair_exchange(w, e2=0.2, Z=2.0, m=1.0, lam=0.002, log=print):
    """пара электронов в поле ядра (из порядка): сдвиг пика пары при ±λ (центральная разность) в каналах фермионы a∧b, бозоны a∨b, бозоны a·a."""
    from . import fields as Fl
    t, x, n = w.t, w.x, w.n; L = max(np.abs(t).max(), np.abs(x).max()); ts = -L + 10
    UfN, _, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); K, C = dirac_component_K(w, UfN, m); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); src = np.abs(t - ts) < 0.45
    fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb
    om = np.linspace(0.8, 2.0, 2401); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); pk = []
    for g_, p_ in ((gs, pa), (gs1, pb)):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)
        for e_, h_ in Ob.peaks(om, S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    pk = sorted(pk); xg, Ec, _, dens, vec = dirac_orbitals(lambda s: Z * e2 / 2 * np.abs(s), 1.0, m, nlev=2, N=512, Lg=60, locmax=40)
    near = lambda E: min(pk, key=lambda z: abs(z[0] - E))[0]; E0, E1 = near(Ec[0]), near(Ec[1]); levels = [e for e, _ in pk]
    xc, amp = crystal_orbitals(w, pa + pb, levels, ts + 1.5, L - 4)
    def hfun(a):
        tab = dict(zip(np.round(xc, 6), np.conj(a))); return np.array([tab.get(v, 0) for v in np.round(x, 6)])
    ha, hb = hfun(amp[levels.index(E0)]), hfun(amp[levels.index(E1)])
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v) < 1e-9)[0] for v in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v) < 1e-9] for v in tR]; ow = np.linspace(1.8, 3.8, 4001); out = {'E': (E0, E1), 'Ec': Ec}
    for key, Fm, P1, P2, sgn, Et in (('F', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0 + E1), ('B', np.outer(fa, fb) + np.outer(fb, fa), ha, hb, 1, E0 + E1), ('BB', np.outer(fa, fa), ha, ha, 0, 2 * E0)):
        Ep = {}
        for l_ in (0.0, lam, -lam):
            A = pair_ladder(K, inc, layers, Fm, l_)
            s = np.array([(np.outer(P1[Z_], P2[Z_]) * A[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * A[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR]); del A
            S = Ob.spectrum(tR, s, ow); S /= S.max(); Ep[l_] = min(Ob.peaks(ow, S, 0.05), key=lambda z: abs(z[0] - Et))[0]
        out[key] = (Ep[0.0], (Ep[lam] - Ep[-lam]) / (2 * lam), Et); log(f'{key}: пик пары {Ep[0.0]:.4f} (ожидание {Et:.4f}), dE/dλ {out[key][1]:+.3f}')
    N = len(xg); h = xg[1] - xg[0]; rc0, rc1 = (d / np.trapezoid(d, xg) for d in dens); Tc = (np.conj(vec[:N, 0]) * vec[:N, 1] + np.conj(vec[N:, 0]) * vec[N:, 1]) / h
    D = np.abs(xg[:, None] - xg[None]); be = e2 / 2; dE = Ec[1] - Ec[0]
    out['J'] = float(rc0 @ (-be * D) @ rc1 * h * h); out['K'] = float(np.real(Tc @ (-be * D) @ np.conj(Tc)) * h * h); out['J00'] = float(rc0 @ (-be * D) @ rc0 * h * h)
    out['Kret'] = float(np.real(Tc @ (-be * np.sin(dE * D) / dE) @ np.conj(Tc)) * h * h)
    return out

# ---------- нормировка вершины статистической сшивкой (dirac-doc §16) и непрерывный эталон двух тел ----------
def ci_two_body(Vfun, be, m=1.0, N=512, Lg=64.0, nmax=32, locmax=150.0):
    """непрерывный эталон: задача двух тел без пар (КВ в базисе локализованных орбиталей Дирака положительной энергии),
    взаимодействие −be|x₁ − x₂|. Возвращает (E0, E1, фермионы основное, бозоны основное, первый порядок Ф, Б)."""
    xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
    Dm = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)); V = np.diag(Vfun(xg))
    ev, vec = np.linalg.eigh(np.block([[-1j * Dm + V, m * np.eye(N)], [m * np.eye(N), 1j * Dm + V]]))
    loc = lambda i: (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) @ xg ** 2
    pos = [i for i in np.argsort(ev) if ev[i] > 0.5 * m and loc(i) < locmax][:nmax]; E = ev[pos]; u, v = vec[:N, pos], vec[N:, pos]; nb = len(pos)
    T = (np.einsum('xi,xj->ijx', np.conj(u), u) + np.einsum('xi,xj->ijx', np.conj(v), v)).reshape(nb * nb, N)
    M2 = (T @ (-be * np.abs(xg[:, None] - xg[None])) @ T.T).reshape(nb, nb, nb, nb).transpose(0, 2, 1, 3).reshape(nb * nb, nb * nb)
    out = []
    for sgn in (-1, 1):
        pairs = [(i, j) for i in range(nb) for j in range(i + (1 if sgn < 0 else 0), nb)]; B = np.zeros((len(pairs), nb, nb))
        for p, (i, j) in enumerate(pairs):
            B[p, i, j] += 1; B[p, j, i] += sgn; B[p] /= np.linalg.norm(B[p])
        B = B.reshape(len(pairs), nb * nb); H = np.diag([E[i] + E[j] for i, j in pairs]) + np.real(B @ M2 @ B.T)
        out.append((np.linalg.eigvalsh(H)[0], H[0, 0]))
    return E[0], E[1], out[0][0], out[1][0], out[0][1], out[1][1]

def pair_physical(w, e2=0.2, Z=2.0, m=1.0, rho=4.0, db=0.004, log=print):
    """(1) веса частиц w = dE/dδb (постоянная вставка на остановках) — той же машиной; (2) вершина λ = (e²/2)/(2 w_a w_b);
    (3) пара всеми порядками двигателем эволюции; сравнение с непрерывным КВ."""
    from . import fields as Fl, propagate as Pr
    t, x, n = w.t, w.x, w.n; L = max(np.abs(t).max(), np.abs(x).max()); ts = -L + 10; be = e2 / 2
    UfN, dUN, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8)
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); om = np.linspace(0.9, 2.0, 2201)
    E0c, E1c, EFc, EBc, EF1, EB1 = ci_two_body(lambda s: Z * e2 / 2 * np.abs(s), be, m)
    Ew = {}
    for d_ in (db, -db):
        me = np.sqrt(m * m - rho * d_); row = []
        for src, Et in ((gs, E0c), (gs1, E1c)):
            phi = Pr.corner_respond_general(w.X, UfN, dUN, me, rho, np.where(np.abs(t - ts) < 0.45, src(x), 0), 1, 0.0)
            S = Ob.spectrum(t[sel], src(x[sel]) * phi[sel], om); row.append(min(Ob.peaks(om, S / S.max(), 0.03), key=lambda z: abs(z[0] - Et))[0])
        Ew[d_] = row
    wa = (Ew[db][0] - Ew[-db][0]) / (2 * db); wb = (Ew[db][1] - Ew[-db][1]) / (2 * db); log(f'веса {wa:+.3f} {wb:+.3f}')
    K, C = dirac_component_K(w, UfN, m); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    src = np.abs(t - ts) < 0.45; fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb; pk = []
    for g_, p_ in ((gs, pa), (gs1, pb)):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], np.linspace(0.8, 2.0, 2401))
        for e_, h_ in Ob.peaks(np.linspace(0.8, 2.0, 2401), S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    lev = [e for e, _ in sorted(pk)]; near = lambda Ex: min(lev, key=lambda z: abs(z - Ex)); E0, E1 = near(E0c), near(E1c)
    xc, amp = crystal_orbitals(w, pa + pb, lev, ts + 1.5, L - 4)
    def hfun(a_):
        tab = dict(zip(np.round(xc, 6), np.conj(a_))); return np.array([tab.get(v_, 0) for v_ in np.round(x, 6)])
    ha, hb = hfun(amp[lev.index(E0)]), hfun(amp[lev.index(E1)])
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v_) < 1e-9)[0] for v_ in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v_) < 1e-9] for v_ in tR]; ow = np.linspace(1.8, 3.8, 4001); out = {}
    for key, Fm, P1, P2, sgn, Et, lam, ref in (('F', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0 + E1, be / (2 * wa * wb), EFc), ('BB', np.outer(fa, fa), ha, ha, 0, 2 * E0, be / (2 * wa * wa), EBc)):
        res = {}
        for l_ in (0.0, lam):
            A = pair_ladder(K, inc, layers, Fm, l_)
            s = np.array([(np.outer(P1[Z_], P2[Z_]) * A[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * A[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR]); del A
            S = Ob.spectrum(tR, s, ow); S /= S.max(); guess = Et + (ref - (E0c + E1c if key == 'F' else 2 * E0c)) * (1 if l_ else 0)
            res[l_] = min(Ob.peaks(ow, S, 0.05), key=lambda z: abs(z[0] - guess))[0]
        out[key] = (res[0.0], res[lam], lam); log(f'{key}: {res[0.0]:.4f} → {res[lam]:.4f} (λ {lam:.5f})')
    return dict(w=(wa, wb), E=(E0, E1), ci=(E0c, E1c, EFc, EBc, EF1, EB1), F=out['F'], BB=out['BB'])
