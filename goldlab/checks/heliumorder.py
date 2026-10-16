"""«Гелий» без нитей: ядро +Z и два электрона, всё поле — из порядка; электроны — компоненты Дирака по световому уголку (§12–13).
(1) уровни E0, E1 в поле ядра и орбитали, вынутые из кристалла (фильтр по частоте; вторая компонента — по чётности ψ_L(x) = ±ψ_R(−x));
(2) поле облака одного электрона из порядка (½ за событие в прошлом конусе с весом плотности); сдвиг уровня другого электрона — Хартри (замер);
(3) обмен K — по орбиталям кристалла и ядру взаимодействия −(e²/2)|Δx|; пара фермионов E0 + E1 + J − K, бозонов E0 + E1 + J + K."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
e2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2; L = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0; Z = 2.0; m = 1.0; be = e2 / 2
w = Wd.gold11(4.0, L); t, x, X = w.t, w.x, w.X; ts = -L + 10; T0 = time.time(); print('n =', w.n, flush=True)
# ---------- непрерывный эталон ----------
UN = lambda s: Z * e2 / 2 * np.abs(s)
xg, Ec, shp, dens, vec = Fe.dirac_orbitals(UN, 1.0, m, nlev=2, N=512, Lg=60, locmax=40); h = xg[1] - xg[0]; N = len(xg)
def JK(r0, r1, T01, xs, hs, cfac=1.0):
    Ux = -be * cfac * np.abs(xs[:, None] - xs[None, :]); return float(r0 @ Ux @ r1 * hs * hs), float(np.real(T01 @ Ux @ np.conj(T01)) * hs * hs)
rc0, rc1 = dens; Tc = (np.conj(vec[:N, 0]) * vec[:N, 1] + np.conj(vec[N:, 0]) * vec[N:, 1]) / h
Jc, Kc = JK(rc0, rc1, Tc, xg, h); J00c = JK(rc0, rc0, Tc, xg, h)[0]; print(f'непрерывный: E0 {Ec[0]:.4f} E1 {Ec[1]:.4f}  J {Jc:+.4f} K {Kc:+.4f}', flush=True)
def hartree_cont(rho_other, lev):
    rr = rho_other / np.trapezoid(rho_other, xg); Ue = -be * np.trapezoid(rr[None] * np.abs(xg[:, None] - xg[None]), xg, axis=1)
    _, ev, _, _, _ = Fe.dirac_orbitals(lambda s: UN(s) + np.interp(s, xg, Ue), 1.0, m, nlev=2, N=512, Lg=60, locmax=40); return ev[lev] - Ec[lev]
hc0, hc1 = hartree_cont(rc1, 0), hartree_cont(rc0, 1); print(f'непрерывный Хартри (один шаг): сдвиг E0 {hc0:+.4f}, E1 {hc1:+.4f}', flush=True)
# ---------- кристалл: ядро ----------
UfN, dUN, _, (alN, beN, FN) = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); print(f'ядро из порядка: наклон {FN:.4f} (цель {Z * e2 / 2})', flush=True)
gs = lambda s: (1 + s) * np.exp(-s * s / 2); om = np.linspace(m - 0.1, m + 1.0, 2201); sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8)
gs1 = lambda s: s * np.exp(-s * s / 8)
def run(Uf, dU, side=1, src=gs, thr=0.03):
    phi = Pr.corner_respond_general(X, Uf, dU, m, 4.0, np.where(np.abs(t - ts) < 0.45, src(x), 0), side, 0.0)
    S = Ob.spectrum(t[sel], src(x[sel]) * phi[sel], om); S /= S.max(); return phi, sorted(Ob.peaks(om, S, thr), key=lambda z: z[0])
phi, pk = run(UfN, dUN); print('пики в поле ядра:', [(round(a, 4), round(b, 2)) for a, b in pk], f'{time.time() - T0:.0f} c', flush=True)
near = lambda pk, E: min(pk, key=lambda z: abs(z[0] - E))[0]
E0, E1 = near(pk, Ec[0]), near(pk, Ec[1])
# ---------- орбитали из кристалла (столбцы — естественные значения x кристалла; МНК по частотам уровней + медленная часть) ----------
win = (t > ts + 1.5) & (t < L - 4); xc = np.unique(np.round(x[win & (np.abs(x) < 10)], 6)); allE = [e for e, _ in pk]
def orbital(E):
    amp = np.zeros(len(xc), complex)
    for k, xv in enumerate(xc):
        s = win & (np.abs(x - xv) < 1e-6); tt = t[s]
        if len(tt) < 6: continue
        A = np.c_[np.exp(1j * np.outer(tt, allE)), np.exp(-1j * np.outer(tt, allE)), np.ones_like(tt), tt]
        amp[k] = np.linalg.lstsq(A, phi[s], rcond=None)[0][allE.index(E)]
    return amp
trap = lambda f: float(np.trapezoid(f, xc))
psi0, psi1 = orbital(E0), orbital(E1)
def dens_of(ps): r = np.abs(ps) ** 2 + np.abs(ps[::-1]) ** 2; return r / trap(r)
r0, r1 = dens_of(psi0), dens_of(psi1)
fid = lambda r, rc: trap(np.sqrt(r * np.interp(xc, xg, rc) / np.trapezoid(rc, xg)))
print(f'орбитали кристалла ({len(xc)} столбцов x): совпадение плотности с непрерывной 0: {fid(r0, rc0):.4f}, 1: {fid(r1, rc1):.4f}', flush=True)
# обмен по орбиталям кристалла (ψ_L(x) = p ψ_R(−x), p0·p1 = −1)
n0 = np.sqrt(trap(np.abs(psi0) ** 2 + np.abs(psi0[::-1]) ** 2)); n1 = np.sqrt(trap(np.abs(psi1) ** 2 + np.abs(psi1[::-1]) ** 2))
a0, a1 = psi0 / n0, psi1 / n1; Tcr = np.conj(a0) * a1 - np.conj(a0[::-1]) * a1[::-1]
wq = np.gradient(xc); Jcr0 = float((r0 * wq) @ (-be * np.abs(xc[:, None] - xc[None])) @ (r1 * wq)); Kcr = float(np.real((Tcr * wq) @ (-be * np.abs(xc[:, None] - xc[None])) @ np.conj(Tcr * wq))); J00 = float((r0 * wq) @ (-be * np.abs(xc[:, None] - xc[None])) @ (r0 * wq)); print(f'по орбиталям кристалла: J {Jcr0:+.4f} K {Kcr:+.4f} J00 {J00:+.4f} (непр. {J00c:+.4f})', flush=True)
# ---------- поле облака из порядка и сдвиг Хартри (динамика) ----------
res = {}
ctrl = [(0, np.interp(xc, xg, rc1) / np.trapezoid(rc1, xg), E0, 'контроль: облако непрерывное'), (1, np.interp(xc, xg, rc0) / np.trapezoid(rc0, xg), E1, 'контроль: облако непрерывное')] if len(sys.argv) > 3 else []
for lev, ro, Ebase, *lab in [(0, r1, E0), (1, r0, E1)] + ctrl:
    Uc, dUc, cfit, alc = Fl.order_cloud11(w, lambda s, ro=ro: np.interp(s, xc, ro, left=0, right=0), e2, ts)
    Uf = lambda a, b, Uc=Uc: UfN(a, b) + Uc(a, b); _, pk2 = run(Uf, dUN + dUc, 1, gs if lev == 0 else gs1, 0.02); En = near(pk2, Ebase + (Jc if True else 0))
    res.setdefault(lev, (En - Ebase, cfit)); print('   пики:', [(round(a_, 4), round(b_, 2)) for a_, b_ in pk2]); print(*lab, f'уровень {lev} в поле облака {1 - lev}: сила облака c = {cfit:.4f} (цель {be}); сдвиг {En - Ebase:+.4f}  ({time.time() - T0:.0f} c)', flush=True)
J = 0.5 * (res[0][0] + res[1][0])
print(f'ИТОГ e²={e2} L={L}: E0 {E0:.4f} ({Ec[0]:.4f}) E1 {E1:.4f} ({Ec[1]:.4f}); J кристалл (дин.) {res[0][0]:+.4f}/{res[1][0]:+.4f}, непр. Хартри {hc0:+.4f}/{hc1:+.4f}, J₁ {Jc:+.4f}; '
      f'K кристалл {Kcr:+.4f} непр. {Kc:+.4f}; пара фермионов {E0 + E1 + J - Kcr:.4f} (непр. {Ec[0] + Ec[1] + Jc - Kc:.4f}), бозонов {E0 + E1 + J + Kcr:.4f} (непр. {Ec[0] + Ec[1] + Jc + Kc:.4f}); по орбиталям: фермионы {E0 + E1 + Jcr0 - Kcr:.4f}, бозоны-основное 2E0+J00 {2 * E0 + J00:.4f} (непр. {2 * Ec[0] + J00c:.4f})', flush=True)
