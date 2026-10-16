"""ДИАГНОСТИКА: почему сила «заряда из геометрии» зависит от места старта (maxwell-doc §22, оговорка 15 окт.).
Одно поле (срез κ = 0.055, L = 16), одна SJ на каждое m; много стартов, мелкие шаги по времени.
Контроль: внешнее однородное поле той же силы (U = E·x, световой уголок) — те же старты."""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import core, worlds as Wd, propagate as Pr, fields as Fl, causal as Cs, observe as Ob
from scipy.interpolate import RegularGridInterpolator
L, kap = 16.0, 0.055; w = Wd.gold11(4.0, L); t, x, X, n = w.t, w.x, w.X, w.n; h = Fl.slice_twist(kap); T0 = time.time()
g = np.linspace(-L - 1, L + 1, 161); AT = np.zeros((161, 161)); Ax = np.zeros((161, 161))
for i, Tq in enumerate(g):
    for j, xq in enumerate(g):
        try: a_, b_ = Fl.connection(h, Tq, xq); AT[i, j], Ax[i, j] = a_[0, 1], b_[0, 1]
        except ValueError: pass
fAT = RegularGridInterpolator((g, g), AT); fAx = RegularGridInterpolator((g, g), Ax); dg = g[1] - g[0]
fF = RegularGridInterpolator((g, g), np.gradient(Ax, dg, axis=0) - np.gradient(AT, dg, axis=1))
gx, gw = np.polynomial.legendre.leggauss(6)
def seg(a0, b0, a1, b1):
    tot = 0
    for xg, wg in zip(gx, gw):
        s = 0.5 * (xg + 1); pts = np.stack([a0 + s * (a1 - a0), b0 + s * (b1 - b0)], -1); tot = tot + 0.5 * wg * (fAT(pts) * (a1 - a0) + fAx(pts) * (b1 - b0))
    return tot
C = Cs.causal(X); I, J = np.nonzero(C); ty, xy, tx, xx = t[I], x[I], t[J], x[J]; tc = 0.5 * ((ty - xy) + (tx + xx)); xc = 0.5 * ((ty + xy) - (tx - xx))
W1 = seg(ty, xy, tc, xc) + seg(tc, xc, tx, xx); Ev = fF(np.stack([t, x], -1))
print(f'поле: F в узлах среднее {Ev.mean():+.5f}, разброс {Ev.std():.1e} (−2κ² = {-2 * kap * kap:+.5f}); A_T, A_x у (−6,1): {fAT([[-6, 1]])[0]:+.3f} {fAx([[-6, 1]])[0]:+.3f}; у (−5,0): {fAT([[-5, 0]])[0]:+.3f} {fAx([[-5, 0]])[0]:+.3f}; {time.time() - T0:.0f} с', flush=True)
E0 = -Ev.mean()
def Wgeom(mq):
    Phi = np.zeros((n, n), complex); Phi[I, J] = 0.5 * np.exp(1j * mq * W1); B = -(1 - 1j * mq * (-Ev)) / 4.0
    return Pr.sj(Pr.johnston11(X, 4.0, 1.0, Phi, B))[0]
def Wext(mq):
    P, dU, U = Fl.uniform(mq * E0); Phi, B = Pr.corner_dense(X, P, dU, 1.0, 4.0); return Pr.sj(Pr.johnston11(X, 4.0, 1.0, Phi, B))[0]
starts = [(-7.0, 0.0), (-6.0, 1.0), (-6.0, 0.0), (-6.0, -1.0), (-5.0, 0.0), (-5.0, 1.0), (-4.0, 0.0)]
LT = np.array([tl for tl, _ in core.layers(X)])
def lcent(psi, times):                       # центр |ψ|² на точном слое времени (без «полосы» ±0.5)
    out = []
    for tl in times:
        S = np.abs(t - tl) < 1e-9; p = np.abs(psi[S]) ** 2; out.append((p * x[S]).sum() / p.sum())
    return np.array(out)
fF_path = lambda t0, x0, s1: np.mean(fF(np.stack([np.linspace(t0, t0 + s1, 40), np.full(40, x0)], -1)))
res = {}
for name, Wf, Afun in (('geom', Wgeom, lambda p: fAx(np.array([p]))[0]), ('ext', Wext, lambda p: 0.0)):
    Ws = {mq: Wf(mq) for mq in (0, 1, -1)}; print(f'== {name}: SJ готовы, {time.time() - T0:.0f} с', flush=True)
    for (t0, x0) in starts:
        tl = LT[(LT > t0 + 1.0) & (LT < t0 + 9.0)]; s = tl - t0; cen = {}; cs = {}
        for mq, Wp in Ws.items():
            A0 = Afun((t0, x0)); f = np.where(np.abs(t - t0) < 0.45, np.exp(-(x - x0) ** 2 / (2 * 1.2 ** 2)) * np.exp(1j * mq * A0 * (x - x0)), 0)
            psi = Wp.T @ f; cen[mq] = lcent(psi, tl); cs[mq] = Ob.centers(t, x, psi, t0 + np.arange(1.0, 9.01, 0.5))
        odd = (cen[1] - cen[-1]) / 2; oslab = (cs[1] - cs[-1]) / 2
        a, v = np.linalg.lstsq(np.c_[0.5 * s * s, s], odd, rcond=None)[0]; rr = odd - (0.5 * a * s * s + v * s)
        res[(name, t0, x0)] = (a, v, rr.std(), np.ptp(oslab - 0.5 * a * np.arange(1.0, 9.01, 0.5) ** 2))
        print(f'{name} старт ({t0:+.0f}, {x0:+.0f}): слоёв {len(tl)}; ускорение a = {a:+.5f}, v₀ = {v:+.4f}, разброс вокруг параболы {rr.std():.3f}; F на пути {fF_path(t0, x0, 8):+.5f}', flush=True)
print('\nОтношение a(срез) / a(внешнее), внешнее пересчитано на F пути (E внешнего = {:.5f}):'.format(E0))
for (t0, x0) in starts:
    ag = res[('geom', t0, x0)][0]; ae = res[('ext', t0, x0)][0] * abs(fF_path(t0, x0, 8)) / E0
    print(f'  ({t0:+.0f}, {x0:+.0f}): {abs(ag) / abs(ae):.3f}   (a срез {ag:+.5f}, a внешн. на F пути {ae:+.5f}; классика F/m = {abs(fF_path(t0, x0, 8)):.5f})')
