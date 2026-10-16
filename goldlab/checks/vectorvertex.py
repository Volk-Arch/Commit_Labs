"""Вершина, одинаковая для всех состояний: калибровочная (векторная) связь. Вес частицы оказался w = −ρ/(2(E − U)) —
множитель «энергия минус потенциал», как у заряда, связанного с полем. Значит вершина пары
  W(z₁, z₂) = i·(e²/ρ²)·(E − U(x₁))·(E − U(x₂))·[z₁, z₂ несравнимы]
без единого измеренного числа (e² — заряд, ρ — плотность). E — энергия частицы в полюсе (приближение «полюсное»:
вместо оператора i∂_t берётся энергия канала; для пары a, b — средняя). Проверка: один и тот же вид для трёх каналов."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, observe as Ob, fermions as Fe
MODE = 'both'
e2, Z, m, rho = 0.2, 2.0, 1.0, 4.0; be = e2 / 2; Fn = Z * e2 / 2
# ---- непрерывный эталон (КВ), с возбуждённым симметричным состоянием a∨b ----
N_, Lg = 512, 64.0; xg = (np.arange(N_) - N_ / 2) * (Lg / N_); hh = Lg / N_; kf = 2 * np.pi * np.fft.fftfreq(N_, d=hh)
Dm = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N_), axis=0), axis=0))
ev, vec = np.linalg.eigh(np.block([[-1j * Dm + np.diag(Fn * np.abs(xg)), m * np.eye(N_)], [m * np.eye(N_), 1j * Dm + np.diag(Fn * np.abs(xg))]]))
loc = lambda i: (np.abs(vec[:N_, i]) ** 2 + np.abs(vec[N_:, i]) ** 2) @ xg ** 2
pos = [i for i in np.argsort(ev) if ev[i] > 0.5 and loc(i) < 150][:24]; E = ev[pos]; u, v = vec[:N_, pos], vec[N_:, pos]; nb = len(pos)
T = (np.einsum('xi,xj->ijx', np.conj(u), u) + np.einsum('xi,xj->ijx', np.conj(v), v)).reshape(nb * nb, N_)
M2 = (T @ (-be * np.abs(xg[:, None] - xg[None])) @ T.T).reshape(nb, nb, nb, nb).transpose(0, 2, 1, 3).reshape(nb * nb, nb * nb)
ci = {}
for sgn in (-1, 1):
    pairs = [(i, j) for i in range(nb) for j in range(i + (1 if sgn < 0 else 0), nb)]; B = np.zeros((len(pairs), nb, nb))
    for p, (i, j) in enumerate(pairs):
        B[p, i, j] += 1; B[p, j, i] += sgn; B[p] /= np.linalg.norm(B[p])
    B = B.reshape(len(pairs), nb * nb); H = np.diag([E[i] + E[j] for i, j in pairs]) + np.real(B @ M2 @ B.T); wv, U_ = np.linalg.eigh(H)
    # состояние, ближайшее к конфигурации (0,1)
    k01 = pairs.index((0, 1)); best = int(np.argmax(np.abs(U_[k01]) ** 2)); ci[sgn] = (wv[0], wv[best])
print(f'КВ: фермионы a∧b {ci[-1][0]:.4f}; бозоны a∨b {ci[1][1]:.4f}; бозоны оба внизу {ci[1][0]:.4f}  (E0 {E[0]:.4f}, E1 {E[1]:.4f})', flush=True)
for L in (24.0, 26.0):
    w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n; ts = -L + 10
    UfN, _, Uev, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); K, C = Fe.dirac_component_K(w, UfN, m); inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
    gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); src = np.abs(t - ts) < 0.45
    fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb
    om = np.linspace(0.8, 2.0, 2401); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); pk = []
    for g_, p_ in ((gs, pa), (gs1, pb)):
        S = Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)
        for e_, h_ in Ob.peaks(om, S / S.max(), 0.03):
            if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
    lev = [e for e, _ in sorted(pk)]; near = lambda Ex: min(lev, key=lambda z: abs(z - Ex)); E0, E1 = near(E[0]), near(E[1])
    xc, amp = Fe.crystal_orbitals(w, pa + pb, lev, ts + 1.5, L - 4)
    def hfun(a_):
        tab = dict(zip(np.round(xc, 6), np.conj(a_))); return np.array([tab.get(v_, 0) for v_ in np.round(x, 6)])
    ha, hb = hfun(amp[lev.index(E0)]), hfun(amp[lev.index(E1)])
    lv = np.unique(np.round(t, 9)); layers = [np.nonzero(np.abs(t - v_) < 1e-9)[0] for v_ in lv]
    R = np.nonzero(sel)[0]; tR = np.unique(np.round(t[R], 9)); layR = [R[np.abs(t[R] - v_) < 1e-9] for v_ in tR]; ow = np.linspace(1.8, 3.8, 4001)
    print(f'L={L:.0f}: E0 {E0:.4f} E1 {E1:.4f}', flush=True)
    xr = np.round(x, 6); prev = -np.ones(n, int); last = {}
    for z in np.argsort(t, kind='stable'):
        prev[z] = last.get(xr[z], -1); last[xr[z]] = z
    has = prev >= 0; dl = np.where(has, t - t[np.maximum(prev, 0)], 1.0); print(f'   шаг по столбцу: медиана {np.median(dl[has]):.3f}, макс {dl[has].max():.3f}', flush=True)
    for name, Fm, P1, P2, sgn, Et, Ebar, ref in (('фермионы a∧b', np.outer(fa, fb) - np.outer(fb, fa), ha, hb, -1, E0 + E1, 0.5 * (E0 + E1), ci[-1][0]),
                                                 ('бозоны a∨b', np.outer(fa, fb) + np.outer(fb, fa), ha, hb, 1, E0 + E1, 0.5 * (E0 + E1), ci[1][1]),
                                                 ('бозоны оба внизу', np.outer(fa, fa), ha, ha, 0, 2 * E0, E0, ci[1][0])):
        nz = Ebar - Uev; Wv = (e2 / rho ** 2) * np.outer(nz, nz) * inc        # вещественная часть; множитель i — в лестнице
        res = {}
        for on in ((0, 1, 2) if MODE == 'both' else (0, 1)):
            Am = np.zeros((n, n), complex); H = np.zeros((n, n), complex)
            for Zl in layers:
                Am[Zl] = K[:, Zl].T @ H
                if on == 1: G = Fm[Zl] + 1j * Wv[Zl] * Am[Zl]
                elif on == 2:
                    A1 = Am[Zl]; A0 = np.where(has[Zl][:, None], Am[np.maximum(prev[Zl], 0)], 0); d1 = dl[Zl][:, None]
                    X = 1j * (A1 - A0) / d1 - Uev[Zl][:, None] * (A1 + A0) / 2
                    X0 = np.where(has[None, :], X[:, np.maximum(prev, 0)], 0); Y = 1j * (X - X0) / dl[None, :] - Uev[None, :] * (X + X0) / 2
                    Y = np.where(has[None, :] & has[Zl][:, None], Y, 0); G = Fm[Zl] + 1j * (e2 / rho ** 2) * inc[Zl] * Y
                else: G = Fm[Zl]
                if np.any(G): H[Zl] = G @ K
            s = np.array([(np.outer(P1[Z_], P2[Z_]) * Am[np.ix_(Z_, Z_)]).sum() + (sgn * (np.outer(P2[Z_], P1[Z_]) * Am[np.ix_(Z_, Z_)]).sum() if sgn else 0) for Z_ in layR]); del Am, H
            S = Ob.spectrum(tR, s, ow); S /= S.max(); guess = Et + (ref - (E[0] + E[1] if 'a∧b' in name or 'a∨b' in name else 2 * E[0])) * (on > 0)
            res[on] = min(Ob.peaks(ow, S, 0.05), key=lambda z: abs(z[0] - guess))[0]
        dc = ref - (E[0] + E[1] if 'оба' not in name else 2 * E[0])
        print(f'   {name}: {res[0]:.4f} → полюс {res[1]:.4f} (сдвиг {res[1] - res[0]:+.4f}, отн. {(res[1] - res[0]) / dc:.2f})' + (f'; производная {res[2]:.4f} (сдвиг {res[2] - res[0]:+.4f}, отн. {(res[2] - res[0]) / dc:.2f})' if 2 in res else '') + f'; непрерывная задача двух тел {ref:.4f} (сдвиг {dc:+.4f})', flush=True)
