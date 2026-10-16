"""Точность измерителя пары: ансамбль сдвигов фазона (тот же закон, другие события) и две плотности ρ.
На каждом мире: уровни E0, E1; веса частиц w = dE/dδb; сдвиги пары (векторная вершина, все порядки): фермионы a∧b и
бозоны оба внизу. Эталон — непрерывная задача двух тел."""
import sys, json, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
e2, Z, m = 0.2, 2.0, 1.0; be = e2 / 2
rho = float(sys.argv[1]); nw = int(sys.argv[2]); L = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0; CH = sys.argv[4].split(',') if len(sys.argv) > 4 else ['F', 'BB']; WEIGHTS = 'w' in (sys.argv[5] if len(sys.argv) > 5 else 'w')
E0c, E1c, EFc, EBc, _, _ = Fe.ci_two_body(lambda s: Z * e2 / 2 * np.abs(s), be, m)
rs = np.random.default_rng(7); from goldlab.core import r_for_density; r = r_for_density(rho, 2)
shifts = [(0.0, 0.0)] + [tuple(r * np.sqrt(rs.uniform()) * np.array([np.cos(a), np.sin(a)])) for a in rs.uniform(0, 2 * np.pi, nw - 1)]
for c in shifts:
    T0 = time.time(); w = Wd.gold11_shift(rho, L, c); t, x, n = w.t, w.x, w.n; ts = -L + 10
    UfN, dUN, Uev, (_, _, FN) = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8)
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); om = np.linspace(0.9, 2.0, 2201); Ew = {}
    for d_ in ((0.004, -0.004) if WEIGHTS else ()):
        me = np.sqrt(m * m - rho * d_); row = []
        for src, Et in ((gs, E0c), (gs1, E1c)):
            phi = Pr.corner_respond_general(w.X, UfN, dUN, me, rho, np.where(np.abs(t - ts) < 0.45, src(x), 0), 1, 0.0)
            S = Ob.spectrum(t[sel], src(x[sel]) * phi[sel], om); row.append(min(Ob.peaks(om, S / S.max(), 0.03), key=lambda z: abs(z[0] - Et))[0])
        Ew[d_] = row
    wa = (Ew[0.004][0] - Ew[-0.004][0]) / 0.008 if WEIGHTS else None; wb = (Ew[0.004][1] - Ew[-0.004][1]) / 0.008 if WEIGHTS else None
    K, C = Fe.dirac_component_K(w, UfN, m, rho); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    src = np.abs(t - ts) < 0.45; fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb; pk = []
    omb = np.linspace(0.8, 2.0, 2401)
    for g_, p_ in ((gs, pa), (gs1, pb)):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], omb)
        for e_, h_ in Ob.peaks(omb, S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    lev = [e for e, _ in sorted(pk)]; near = lambda Ex: min(lev, key=lambda z: abs(z - Ex)); E0, E1 = near(E0c), near(E1c)
    xc, amp = Fe.crystal_orbitals(w, pa + pb, lev, ts + 1.5, L - 4)
    def hfun(a_):
        tab = dict(zip(np.round(xc, 6), np.conj(a_))); return np.array([tab.get(v_, 0) for v_ in np.round(x, 6)])
    ha, hb = hfun(amp[lev.index(E0)]), hfun(amp[lev.index(E1)])
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v_) < 1e-9)[0] for v_ in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v_) < 1e-9] for v_ in tR]; ow = np.linspace(1.8, 3.8, 4001); out = {}
    EBab = 2.6918
    for key, Fm, P1, P2, sgn, Et, Eb, dref in [z for z in (('F', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0 + E1, 0.5 * (E0 + E1), EFc - E0c - E1c), ('B', np.outer(fa, fb) + np.outer(fb, fa), ha, hb, 1, E0 + E1, 0.5 * (E0 + E1), EBab - E0c - E1c), ('BB', np.outer(fa, fa), ha, ha, 0, 2 * E0, E0, EBc - 2 * E0c)) if z[0] in CH]:
        nz = Eb - Uev; Wm = (e2 / rho ** 2) * np.outer(nz, nz) * inc; res = {}
        for on in (0, 1):
            A = np.zeros((n, n), complex); H = np.zeros((n, n), complex)
            for Zl in layers:
                A[Zl] = K[:, Zl].T @ H; G = Fm[Zl] + (1j * Wm[Zl] * A[Zl] if on else 0)
                if np.any(G): H[Zl] = G @ K
            s = np.array([(np.outer(P1[Z_], P2[Z_]) * A[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * A[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR]); del A, H
            S = Ob.spectrum(tR, s, ow); S /= S.max(); res[on] = min(Ob.peaks(ow, S, 0.05), key=lambda z: abs(z[0] - (Et + dref * on)))[0]
        out[key] = res[1] - res[0]; del Wm, Fm
    rec = dict(rho=rho, L=L, c=[round(float(c[0]), 3), round(float(c[1]), 3)], n=n, FN=FN, E0=E0, E1=E1, wa=wa, wb=wb, **{'d' + k: v for k, v in out.items()},
               ref=dict(E0=E0c, E1=E1c, dF=EFc - E0c - E1c, dBB=EBc - 2 * E0c, wa=-rho / (2 * 1.0907), wb=-rho / (2 * 1.2176)), sec=round(time.time() - T0))
    print(json.dumps(rec, ensure_ascii=False), flush=True); del K, inc
