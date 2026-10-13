"""Предсказание w = −1 (одна Λ на конус) против меняющейся тёмной энергии w(a) = w0 + wa(1−a).
Данные: Pantheon+ (1590, полная ковариация, яркость свободна) + DESI DR2 BAO (13 точек)
+ сжатый реликт: угол звука 100θ* = 1.04110 ± 0.00031 и ω_m = 0.1430 ± 0.0011 (Planck 2018).
r_d и r* масштабируются с ω_m по Aubourg (2015), ω_b = 0.02237."""
import sys, math, numpy as np
sys.argv = sys.argv[:1]
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2dist
import everpresent as ep
C = 299792.458; WB = 0.02237; WR = 4.18e-5
B = np.loadtxt('/home/claude/data/bao/desi_gaussian_bao_ALL_GCcomb_mean.txt', dtype=str)
BZ = B[:, 0].astype(float); BV = B[:, 1].astype(float); BQ = B[:, 2]
BC = np.linalg.inv(np.loadtxt('/home/claude/data/bao/desi_gaussian_bao_ALL_GCcomb_cov.txt'))
def rd(wm): return 147.05 * (WB / 0.02236) ** -0.13 * (wm / 0.1432) ** -0.23
def E(z, h, wm, w0, wa):
    Om = wm / h ** 2; Or = WR / h ** 2; Ode = 1 - Om - Or
    f = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + Ode * f)
ZZ = np.concatenate([np.linspace(0, 3, 3001), np.logspace(math.log10(3.001), math.log10(1090), 3000)])
def chi(p, model):
    h, wm = p[0], p[1]
    w0, wa = (p[2], p[3]) if model == 'w0wa' else (-1.0, 0.0)
    if not (0.5 < h < 0.9 and 0.12 < wm < 0.17 and -3 < w0 < 1 and -5 < wa < 3): return 1e9
    e = E(ZZ, h, wm, w0, wa)
    if np.any(~np.isfinite(e)) or np.any(e <= 0): return 1e9
    DHh = C / (100 * h)
    DM = DHh * np.concatenate([[0], np.cumsum(0.5 * (1 / e[1:] + 1 / e[:-1]) * np.diff(ZZ))])
    r = rd(wm)
    mod = []
    for z, q in zip(BZ, BQ):
        dm = np.interp(z, ZZ, DM); dh = DHh / np.interp(z, ZZ, e)
        mod.append({'DM_over_rs': dm, 'DH_over_rs': dh, 'DV_over_rs': (z * dm ** 2 * dh) ** (1 / 3)}[q] / r)
    d = BV - np.array(mod); cb = d @ BC @ d
    rstar = r * 0.98166                                         # r*/r_d ≈ 144.39/147.09
    th = rstar / DM[-1]; cc = ((100 * th - 1.04110) / 0.00031) ** 2 + ((wm - 0.1430) / 0.0011) ** 2
    lna = ep.LNA_LATE; Esn = E(np.exp(-lna) - 1, h, wm, w0, wa)[None]
    cs = ep.chi2(ep.mu_from_E(lna, Esn))[0]
    return cb + cc + cs
def fit(model, starts):
    best = None
    for s in starts:
        r = minimize(chi, s, args=(model,), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 6000, 'maxfev': 6000})
        r = minimize(chi, r.x, args=(model,), method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-5, 'maxiter': 6000, 'maxfev': 6000})
        if best is None or r.fun < best.fun: best = r
    return best
l = fit('lcdm', [[0.68, 0.143], [0.66, 0.141], [0.70, 0.144]])
w = fit('w0wa', [[0.68, 0.143, -1, 0], [0.67, 0.143, -0.8, -0.8], [0.66, 0.142, -0.7, -1.2], [0.69, 0.143, -0.9, -0.3]])
d = l.fun - w.fun; p = chi2dist.sf(d, 2); sig = abs(__import__('scipy.stats', fromlist=['norm']).norm.isf(p / 2))
print(f'ΛCDM  (w = −1, наше предсказание): χ² = {l.fun:.2f}  при h = {l.x[0]:.4f}, ω_m = {l.x[1]:.4f}, Ω_m = {l.x[1]/l.x[0]**2:.3f}')
print(f'w0wa (меняющаяся):                 χ² = {w.fun:.2f}  при h = {w.x[0]:.4f}, ω_m = {w.x[1]:.4f}, w0 = {w.x[2]:.3f}, wa = {w.x[3]:.3f}')
print(f'Δχ² = {d:.2f} за 2 лишних параметра ⇒ p = {p:.4f}, ≈ {sig:.1f}σ в пользу меняющейся тёмной энергии')
# что даёт каждая выборка отдельно: без SN
