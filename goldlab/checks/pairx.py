"""Очередь п. 1, шаг 1: обмен как ДИНАМИКА (первый порядок) в паре двух электронов в поле ядра.
Двухчастичная амплитуда A(y₁, y₂) = K⊗K·src (компонента Дирака по световому уголку, поле ядра из порядка),
поправка первого порядка δA = (K⊗K)·(V ∘ A₀), вершина V(z₁, z₂) = [z₁, z₂ несравнимы] — ковариантная, из порядка (§26 maxwell-doc).
δA на парах чтения считается двумя умножениями матриц: δA_RR = K[:,R]ᵀ (V∘A₀) K[:,R].
Сдвиг уровня пары: пик |FT(s₀ + λ·δs)|² как функция λ (наклон). Нормировка вершины неизвестна ⇒ сравниваем ОТНОШЕНИЯ:
фермионы/бозоны (J − K)/(J + K), бозоны-внизу/бозоны J₀₀/(J + K) — от нормировки не зависят."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe, causal as Cs
L = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0; e2 = 0.2; Z = 2.0; m = 1.0; be = e2 / 2; T0 = time.time()
lg = lambda *a: print(f'[{time.time() - T0:5.0f} c]', *a, flush=True)
w = Wd.gold11(4.0, L); t, x, X, n = w.t, w.x, w.X, w.n; ts = -L + 10; lg('n =', n)
UfN, dUN, _, (_, _, FN) = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); lg(f'ядро: наклон {FN:.4f}')
# ---- плотный K компоненты Дирака (уголок, без вставки) ----
C = Cs.causal(X); I, J = np.nonzero(C); Phi = np.zeros((n, n), complex)
for s0 in range(0, len(I), 400000):
    sl = slice(s0, s0 + 400000); Phi[I[sl], J[sl]] = 0.5 * np.exp(1j * Pr.corner_phase_general(UfN, t[I[sl]], x[I[sl]], t[J[sl]], x[J[sl]], 1))
K = np.linalg.solve((np.eye(n) - Phi * (-m * m / 4.0)).T, Phi.T).T; del Phi; lg('K готов')
gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); src = np.abs(t - ts) < 0.45
fa, fb = np.where(src, gs(x), 0), np.where(src, gs1(x), 0); pa, pb = K.T @ fa, K.T @ fb
om = np.linspace(0.8, 2.0, 2401); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8)
pk = []
for g_, p_ in ((gs, pa), (gs1, pb)):
    for e_, h_ in Ob.peaks(om, (lambda S: S / S.max())(Ob.spectrum(t[sel], g_(x[sel]) * p_[sel], om)), 0.03):
        if all(abs(e_ - q[0]) > 0.03 for q in pk): pk.append((e_, h_))
pk = sorted(pk); lg('уровни (источники a, b):', [(round(a, 4), round(b, 2)) for a, b in pk])
xg, Ec, _, dens, vec = Fe.dirac_orbitals(lambda s: Z * e2 / 2 * np.abs(s), 1.0, m, nlev=2, N=512, Lg=60, locmax=40)
near = lambda E: min(pk, key=lambda z: abs(z[0] - E))[0]; E0, E1 = near(Ec[0]), near(Ec[1]); lg(f'E0 {E0:.4f} ({Ec[0]:.4f})  E1 {E1:.4f} ({Ec[1]:.4f})')
xc, amp = Fe.crystal_orbitals(w, pa + pb, [e for e, _ in pk], ts + 1.5, L - 4)
ia, ib = [e for e, _ in pk].index(E0), [e for e, _ in pk].index(E1)
def hfun(a):
    tab = dict(zip(np.round(xc, 6), np.conj(a))); return np.array([tab.get(v, 0) for v in np.round(x, 6)])
ha, hb = hfun(amp[ia]), hfun(amp[ib])
# ---- чтение: равновременные пары в окне ----
R = np.nonzero(sel)[0]; tR = np.round(t[R], 9); lay = [R[tR == v] for v in np.unique(tR)]; tl = np.unique(tR)
def signal(Afun, P1, P2, sgn):
    out = np.zeros(len(lay), complex)
    for k, Z_ in enumerate(lay):
        M = Afun(Z_); out[k] = (np.outer(P1[Z_], P2[Z_]) * M).sum() + sgn * (np.outer(P2[Z_], P1[Z_]) * M).sum()
    return out
inc = ~(C | C.T); np.fill_diagonal(inc, False); del C
KR = K[:, R]; posR = {r: i for i, r in enumerate(R)}
# вершина i·[несравнимы]: у K⊗K два множителя 1/(2iE), сдвиг энергии (фаза) требует множителя i
def shift(u, v, sgn, P1, P2, label, Etar_):
    A0 = np.outer(u, v) + (sgn * np.outer(v, u) if sgn != 0 else 0)
    Mv = (1j * inc * A0).astype(np.complex128); dA = KR.T @ (Mv @ KR); del Mv
    A0R = A0[np.ix_(R, R)]; idx = lambda Z_: [posR[z] for z in Z_]
    s0 = signal(lambda Z_: A0R[np.ix_(idx(Z_), idx(Z_))], P1, P2, sgn); ds = signal(lambda Z_: dA[np.ix_(idx(Z_), idx(Z_))], P1, P2, sgn)
    ob = np.linspace(-4, 4, 4001); S0b = Ob.spectrum(tl, s0, ob); Sdb = Ob.spectrum(tl, ds, ob)
    lg('   s0 пики:', [(round(e, 3), round(h, 3)) for e, h in Ob.peaks(ob, S0b / S0b.max(), 0.02)])
    lg('   δs пики:', [(round(e, 3), round(h, 3)) for e, h in Ob.peaks(ob, Sdb / Sdb.max(), 0.02)])
    ow = np.linspace(1.4, 4.0, 5201); S = Ob.spectrum(tl, s0, ow); pks = [e for e, h in Ob.peaks(ow, S / S.max(), 0.01)]
    Et = min(pks, key=lambda e: abs(e - Etar_)); tc = tl - tl.mean(); wv = Ob.tukey(tl, 0.1)
    Bs = np.exp(1j * np.outer(tc, pks)); A = np.linalg.lstsq(Bs * wv[:, None], s0 * wv, rcond=None)[0]
    Bd = np.c_[Bs, tc[:, None] * Bs]; cd = np.linalg.lstsq(Bd * wv[:, None], ds * wv, rcond=None)[0]; k = pks.index(Et)
    slope = (cd[len(pks) + k] / (1j * A[k])); res = np.linalg.norm((Bd @ cd - ds) * wv) / np.linalg.norm(ds * wv)
    lg(f'{label}: пик пары {Et:.4f} (ожидание {Etar_:.4f}); dE/dλ = {slope.real:+.4e} (мнимая часть {slope.imag:+.1e}); частот в подгонке {len(pks)}, остаток {res:.2f}')
    return Et, slope
EF, sF = shift(pa, pb, -1, ha, hb, 'фермионы a∧b', E0 + E1)
EB, sB = shift(pa, pb, +1, ha, hb, 'бозоны a∨b', E0 + E1)
EBB, sBB = shift(pa, pa, 0, ha, ha, 'бозоны a·a (оба внизу)', 2 * E0)
# ---- непрерывные предсказания (первый порядок): мгновенное ядро и ядро «несравнимых» (запаздывание: sin(ΔE|d|)/ΔE) ----
N = len(xg); h = xg[1] - xg[0]; rc0, rc1 = (d / np.trapezoid(d, xg) for d in dens); Tc = (np.conj(vec[:N, 0]) * vec[:N, 1] + np.conj(vec[N:, 0]) * vec[N:, 1]) / h
D = np.abs(xg[:, None] - xg[None]); dE = Ec[1] - Ec[0]
J = rc0 @ (-be * D) @ rc1 * h * h; K_ = np.real(Tc @ (-be * D) @ np.conj(Tc)) * h * h; J00 = rc0 @ (-be * D) @ rc0 * h * h
Kr = np.real(Tc @ (-be * np.sin(dE * D) / dE) @ np.conj(Tc)) * h * h
lg(f'непр.: J {J:+.4f} K {K_:+.4f} (с запаздыванием {Kr:+.4f}) J00 {J00:+.4f}')
lg(f'ИТОГ L={L}: наклоны Ф {sF:.3f}  Б {sB:.3f}  ББ {sBB:.3f}'); sF, sB, sBB = sF.real, sB.real, sBB.real
lg(f'ИТОГ L={L}: Ф/Б кристалл {sF / sB:.4f}; теория мгновенная {(J - K_) / (J + K_):.4f}, с запаздыванием {(J - Kr) / (J + Kr):.4f}')
lg(f'ИТОГ L={L}: (оба внизу)/Б кристалл {sBB / sB:.4f}; теория {J00 / (J + K_):.4f}, с запаздыванием {J00 / (J + Kr):.4f}')
c = 0.5 * (sF + sB) / J; lg(f'ИТОГ L={L}: калибровка по J: K кристалл = {(sB - sF) / 2 / c:+.4f} (мгновенная {K_:+.4f}, с запаздыванием {Kr:+.4f}); J00 = {sBB / c:+.4f} ({J00:+.4f})')
