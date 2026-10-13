"""Меняем связь «дрожь → Λ». Модель 1 Соркина: Λ(t) = среднее по всему прошлому
конусу события t (бегущее) ⇒ Λ ~ H² во все эпохи ⇒ CMB против.
Блочное прочтение (у нас мир задан целиком: событие есть ⟺ тень в шаре; роста нет):
Λ — одно число для всего прошлого конуса наблюдателя, ΛV_набл = 8πα·ξ·√V_набл.
Внутри конуса Λ постоянна (w = −1), величина ~ 1/√V_набл ~ H0². Самосогласование:
V_набл зависит от Λ. Наблюдатель — там, где фотоны остыли до 2.7 K (a = 1),
физическая плотность вещества фиксирована (ω_m = 0.142, как по CMB)."""
import sys; sys.argv = sys.argv[:1]
import numpy as np, math
from scipy.optimize import brentq
import everpresent as ep
np.seterr(all='ignore')
OMN = 0.142 / 0.677 ** 2          # Ω_m в номинальных единицах H=67.7
OMR = ep.OM_R

def past_cone(Lam, n=6000, a0=1e-8):
    """4-объём прошлого конуса при a=1 во вселенной с постоянной Λ (единицы 1/H_ном)."""
    lna = np.linspace(math.log(a0), 0, n); a = np.exp(lna)
    H2 = OMN * a ** -3 + OMR * a ** -4 + Lam / 3
    if np.any(H2 <= 0): return np.nan, None
    H = np.sqrt(H2); dl = lna[1] - lna[0]
    dt = dl / H; t0 = 1 / (2 * H[0]); eta0 = 2 * t0 / a[0]
    eta = eta0 + np.cumsum(dt / a); eta_now = eta[-1]
    V = (4 * math.pi / 3) * np.sum(a ** 3 * dt * (eta_now - eta) ** 3)
    return V, H[-1]

def solve(alpha, xi):
    """Λ = 8πα ξ /√V(Λ): ищем корень f(Λ) = Λ√V(Λ) − 8πα ξ."""
    target = 8 * math.pi * alpha * xi
    f = lambda L: L * math.sqrt(past_cone(L)[0]) - target
    if xi > 0:
        lo, hi = 0.0, 1.0
        while f(hi) < 0: hi *= 2
        return brentq(f, lo, hi, xtol=1e-6)
    lo = 0.0; hi = -0.05                       # Λ < 0: ищем, пока вселенная доживает до a = 1
    while True:
        V, _ = past_cone(hi)
        if not np.isfinite(V): return np.nan
        if f(hi) < 0: return brentq(f, hi, lo, xtol=1e-6)
        lo, hi = hi, hi * 1.5

if __name__ == '__main__':
    c0 = min(ep.lcdm_chi2(o) for o in np.linspace(0.2, 0.45, 51))
    xis = np.linspace(-4, 4, 161); wx = np.exp(-xis ** 2 / 2); wx /= wx.sum()
    V0, _ = past_cone(3 * 0.69 * (0.677 / 0.677) ** 2)
    print(f'ΛCDM χ² = {c0:.1f}; для справки V_набл·H0⁴ в ΛCDM ≈ {V0:.3f}')
    best = None
    for alpha in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
        good1 = good4 = 0.0; OLs = []
        for xi, w in zip(xis, wx):
            if xi == 0: continue
            L = solve(alpha, xi)
            if not np.isfinite(L): continue
            V, Hn = past_cone(L)
            h0 = 67.7 * Hn; Om = OMN / Hn ** 2; OL = L / 3 / Hn ** 2
            OLs.append((OL, w))
            if not (60 < h0 < 80 and 0.2 < Om < 0.45): continue
            a = np.exp(ep.LNA_LATE); E = np.sqrt(OMN * a ** -3 + OMR * a ** -4 + L / 3)[None] / Hn
            d = ep.chi2(ep.mu_from_E(ep.LNA_LATE, E))[0] - c0
            good1 += w * (d < 1); good4 += w * (d < 4)
        OLs = np.array(OLs)
        pos = OLs[:, 0] > 0
        print(f'α = {alpha:.2f}: доля миров с Δχ²<1: {good1*100:5.1f} %, Δχ²<4: {good4*100:5.1f} %  '
              f'(Ω_Λ при ξ=1: {OLs[np.argmin(abs(xis[xis != 0][:len(OLs)] - 1)), 0]:.2f})')
        sys.stdout.flush()

    # точнее: ξ с шагом 0.005, и карта ξ → Ω_Λ, H0
    print('--- точная сетка ---')
    xs = np.linspace(0.005, 3, 600); wg = np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi) * (xs[1] - xs[0])
    for alpha in (0.02, 0.025, 0.03, 0.035, 0.04):
        g4 = g1 = pol = 0.0; table = []
        for xi, w in zip(xs, wg):
            L = solve(alpha, xi); V, Hn = past_cone(L)
            h0 = 67.7 * Hn; Om = OMN / Hn ** 2; OL = L / 3 / Hn ** 2
            pol += w * (0.6 < OL < 0.8)
            if 60 < h0 < 80 and 0.2 < Om < 0.45:
                a = np.exp(ep.LNA_LATE); E = np.sqrt(OMN * a ** -3 + OMR * a ** -4 + L / 3)[None] / Hn
                d = ep.chi2(ep.mu_from_E(ep.LNA_LATE, E))[0] - c0
                g1 += w * (d < 1); g4 += w * (d < 4)
                if d < 4: table.append((xi, OL, h0, d))
        t = np.array(table)
        print(f'α = {alpha:.3f}: Δχ²<1 {g1*100:.1f} %, Δχ²<4 {g4*100:.1f} %, 0.6<Ω_Λ<0.8 {pol*100:.1f} %;  '
              f'хорошие: ξ ∈ [{t[:,0].min():.2f}, {t[:,0].max():.2f}], Ω_Λ ∈ [{t[:,1].min():.2f}, {t[:,1].max():.2f}], H0 ∈ [{t[:,2].min():.0f}, {t[:,2].max():.0f}]')
        sys.stdout.flush()
