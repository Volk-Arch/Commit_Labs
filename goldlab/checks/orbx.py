"""извлечение орбиталей из кристалла: кэш φ и проверка способов."""
import sys, os, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
e2, L, Z, m = 0.2, 30.0, 2.0, 1.0; w = Wd.gold11(4.0, L); t, x, X = w.t, w.x, w.X; ts = -L + 10
UN = lambda s: Z * e2 / 2 * np.abs(s); xg, Ec, shp, dens, vec = Fe.dirac_orbitals(UN, 1.0, m, nlev=3, N=512, Lg=60, locmax=40); N = len(xg)
f = 'orbx_phi.npz'
if not os.path.exists(f):
    UfN, dUN, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4)); gs = lambda s: (1 + s) * np.exp(-s * s / 2)
    phi = Pr.corner_respond_general(X, UfN, dUN, m, 4.0, np.where(np.abs(t - ts) < 0.45, gs(x), 0), 1, 0.0); np.savez(f, phi=phi)
phi = np.load(f)['phi']
xb = np.arange(-10, 10.01, 0.25); xc = 0.5 * (xb[1:] + xb[:-1]); hb = 0.25
def fid(r, rc): rr = np.interp(xc, xg, rc); rr /= rr.sum() * hb; return float(np.sum(np.sqrt(np.maximum(r, 0) * rr)) * hb)
def dens_of(ps): r = np.abs(ps) ** 2 + np.abs(ps[::-1]) ** 2; return r / (r.sum() * hb)
Es = [1.0126, 1.282, 1.578, 1.9152]
for name, t0, t1 in (('окно ts+1.5..L−4', ts + 1.5, L - 4), ('окно ts+4..L−4', ts + 4, L - 4)):
    win = (t > t0) & (t < L - 4); b = np.digitize(x, xb) - 1; out = {E: np.zeros(len(xc), complex) for E in Es}
    for k in range(len(xc)):
        s = win & (b == k)
        if s.sum() < 8: continue
        tt = t[s]; A = np.exp(1j * np.outer(tt, Es)); A = np.c_[A, np.exp(-1j * np.outer(tt, Es))]   # обе ветви
        coef, *_ = np.linalg.lstsq(A, phi[s], rcond=None)
        for i, E in enumerate(Es): out[E][k] = coef[i]
    print(name, 'МНК:', [round(fid(dens_of(out[E]), dens[i]), 3) for i, E in ((0, 1.282), (1, 1.578))])
win = (t > ts + 1.5) & (t < L - 4); b = np.digitize(x, xb) - 1; amp = {}
for E in (1.282, 1.578):
    a = np.zeros(len(xc), complex)
    for k in range(len(xc)):
        s = win & (b == k)
        if s.sum() < 8: continue
        tt = t[s]; A = np.c_[np.exp(1j * np.outer(tt, Es)), np.exp(-1j * np.outer(tt, Es)), np.ones_like(tt), tt, np.exp(1j*np.outer(tt,[0.5,0.75,2.2,2.5])), np.exp(-1j*np.outer(tt,[0.5,0.75,1.0]))]; a[k] = np.linalg.lstsq(A, phi[s], rcond=None)[0][Es.index(E)]
    amp[E] = a
for i, E in ((0, 1.282), (1, 1.578)):
    r = np.abs(amp[E]) ** 2; r /= r.sum() * hb; u2 = np.abs(vec[:N, i]) ** 2; v2 = np.abs(vec[N:, i]) ** 2
    print(E, 'fid |ψ|² с |u|²', round(fid(r, u2), 3), 'с |v|²', round(fid(r, v2), 3), ' <x²> кр', round(float((r * xc ** 2).sum() * hb), 2), 'непр', round(float(np.trapezoid(dens[i] * xg ** 2, xg) / np.trapezoid(dens[i], xg)), 2))
    print('   профиль кр (x=-6..6):', np.round(np.interp(np.arange(-6, 7, 1.0), xc, r), 3))
    print('   профиль |u|²+|v|²    :', np.round(np.interp(np.arange(-6, 7, 1.0), xg, dens[i] / np.trapezoid(dens[i], xg)), 3))
print('--- подробно')
cnt = np.array([(win & (b == k)).sum() for k in range(len(xc))])
for E, i in ((1.282, 0), (1.578, 1)):
    r = np.abs(amp[E]) ** 2; r /= r.sum() * hb; u2 = np.abs(vec[:N, i]) ** 2; v2 = np.abs(vec[N:, i]) ** 2; nn = np.trapezoid(u2 + v2, xg)
    for xx in np.arange(-4, 4.01, 0.5):
        k = np.argmin(np.abs(xc - xx)); print(E, f'{xc[k]:+.3f} n={cnt[k]:4d} кр {r[k]:.3f}  |u|² {np.interp(xc[k], xg, u2) / nn:.3f} |v|² {np.interp(xc[k], xg, v2) / nn:.3f}')
