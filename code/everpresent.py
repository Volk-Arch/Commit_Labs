#!/usr/bin/env python3
"""«Вечно присутствующая Λ» против сверхновых Pantheon+, с наследованием и без.

Модель 1 (Ahmed–Dodelson–Greene–Sorkin; в форме arXiv:2307.13743):
  V_t = (4π/3)∫₀ᵗ dt' a³ (η_t − η_{t'})³          — 4-объём прошлого светового конуса
  Λ_t = (Λ_{t−dt}V_{t−dt} + 8πα·e_t·√(V_t − V_{t−dt})) / V_t
  H² = (8π/3)(ρ_m + ρ_r) + Λ/3                      — планковские единицы
Наше изменение — наследование по родословной (lineage-doc): e_t не независимы,
а помнят прошлое по ln V: за шаг самоподобия (Δln V = 4 ln τ) корреляция
φ = q·τ². q = 0 — модель Соркина как есть.
Сверка: χ² по Pantheon+ (z > 0.01, полная ковариация, сдвиг M вычеркнут),
сравнение с плоской ΛCDM. Считаем долю «миров» (зёрен), которые подходят
не хуже ΛCDM.
"""
import sys, math, time
import numpy as np
TAU = (1 + 5 ** .5) / 2
H0 = 1.0   # единицы 1/H0: уравнение ΛV = 8πα Σ ξ√ΔV не меняется при пересчёте масштаба
            # (Λ ~ 1/длина², V ~ длина⁴), так что планковские единицы не нужны
RHO_CR = 3 * H0 ** 2 / (8 * math.pi)
OM_R = 9.0e-5

# ---------- данные
def load():
    import pandas as pd
    df = pd.read_csv('/home/claude/data/pantheon/pp.dat', sep=r'\s+')
    n = int(open('/home/claude/data/pantheon/cov.cov').readline())
    C = np.loadtxt('/home/claude/data/pantheon/cov.cov', skiprows=1).reshape(n, n)
    m = (df.zHD.values > 0.01)
    C = C[np.ix_(m, m)]; d = df[m]
    return d.zHD.values, d.zHEL.values, d.m_b_corr.values, np.linalg.inv(C)
Z, ZH, MB, CI = load()
ONE = CI.sum()
def chi2(mu):
    """μ — модельные модули расстояния (с точностью до константы), shape (..., nSN)"""
    D = MB[None] - mu
    A = np.einsum('ij,kj->ki', CI, D)
    return np.einsum('ki,ki->k', D, A) - (A.sum(1)) ** 2 / ONE

# ---------- ΛCDM
def mu_from_E(lna, E):
    """E(a)=H/H(a=1) на сетке ln a (shape k × n); модули для Z"""
    a = np.exp(lna)
    integrand = 1 / (a ** 2 * E) * a              # da = a dln a
    # D(a) = ∫_a^1 da'/(a'² E): кумулятивно от конца
    cum = np.concatenate([np.zeros((E.shape[0], 1)), np.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(lna)[None], axis=1)], axis=1)
    Dtot = cum[:, -1:] - cum                       # ∫ от a до 1
    a_sn = 1 / (1 + Z)
    Dz = np.array([np.interp(np.log(a_sn), lna, Dtot[k]) for k in range(E.shape[0])])
    return 5 * np.log10((1 + ZH)[None] * Dz)
LNA_LATE = np.linspace(math.log(1 / 3.5), 0, 600)
def lcdm_chi2(om):
    a = np.exp(LNA_LATE); E = np.sqrt(om * a ** -3 + OM_R * a ** -4 + (1 - om - OM_R))[None]
    return chi2(mu_from_E(LNA_LATE, E))[0]

# ---------- вечно присутствующая Λ
def histories(alpha, om, q, seeds, mode='level', nstep=4000, a0=1e-5):
    lna = np.linspace(math.log(a0), 0, nstep); dl = lna[1] - lna[0]
    K = len(seeds); rs = [np.random.default_rng(s) for s in seeds]
    a = math.exp(lna[0]); H = H0 * math.sqrt(OM_R) / a ** 2
    t = np.full(K, 1 / (2 * H)); eta = 2 * t / a
    # радиационная эпоха до a0 точно: a = A√t, η = 2√t/A, S_m = ∫a³η^m dt
    A = a / math.sqrt(t[0])
    S = [np.full(K, A ** (3 - m) * 2 ** m * t[0] ** ((5 + m) / 2) / ((5 + m) / 2)) for m in range(4)]
    V = np.full(K, (32 * math.pi / 3) * t[0] ** 4 / 140)
    xi0 = np.array([r.normal() for r in rs])
    vst = (1 - q) / (1 - (q * TAU ** 2) ** 2)          # стационарная дисперсия d_k/√n_k
    Lam = 8 * math.pi * alpha * xi0 * math.sqrt(vst) / np.sqrt(V)   # накопленное до a0 (грубо; забывается за пару уровней)
    LV = 4 * math.log(TAU)
    off = np.array([r.uniform(0, LV) for r in rs])        # фаза уровней случайна
    curlev = np.floor((np.log(V) - off) / LV).astype(int); lev0 = curlev.copy()
    XL = np.stack([r.normal(size=80) for r in rs])             # монетки уровней, заранее
    nk = np.exp(off + curlev * LV) * (TAU ** 4 - 1)
    d = np.array([r.normal() for r in rs]) * np.sqrt(vst * nk); dens = d / nk
    Hs = np.zeros((K, nstep)); alive = np.ones(K, bool); i_rec = int(np.argmin(abs(lna - math.log(1 / 1101)))); OLrec = np.zeros(K)
    xi_all = np.stack([r.normal(size=nstep) for r in rs])
    for i, x in enumerate(lna):
        a = math.exp(x)
        H2 = (8 * math.pi / 3) * RHO_CR * (om * a ** -3 + OM_R * a ** -4) + Lam / 3
        alive &= H2 > 0
        Hc = np.sqrt(np.maximum(H2, 1e-300)); Hs[:, i] = Hc
        if i == i_rec: OLrec = (Lam / 3) / np.maximum(H2, 1e-300)
        dt = dl / Hc
        t += dt; eta += dt / a
        w = a ** 3 * dt
        S[0] += w; S[1] += w * eta; S[2] += w * eta ** 2; S[3] += w * eta ** 3
        Vn = (4 * math.pi / 3) * (eta ** 3 * S[0] - 3 * eta ** 2 * S[1] + 3 * eta * S[2] - S[3])
        Vn = np.maximum(Vn, V + 1e-300)
        dV = Vn - V
        if mode == 'step':                       # Соркин как в статье: новая монетка на каждом шаге
            Lam = (Lam * V + 8 * math.pi * alpha * xi_all[:, i] * np.sqrt(dV)) / Vn
        else:                                    # по уровням τ⁴ с наследованием (lineage-doc)
            lev = np.floor((np.log(Vn) - off) / LV).astype(int)
            new = lev > curlev
            while new.any():                     # вошли в новый уровень: d_k = qτ⁴d_{k-1} + √((1-q)n_k)ξ
                curlev = np.where(new, curlev + 1, curlev)
                nk = np.exp(off + curlev * LV) * (TAU ** 4 - 1)
                xi = XL[np.arange(K), np.clip(curlev - lev0, 0, XL.shape[1] - 1)]
                d = np.where(new, q * TAU ** 4 * d + np.sqrt((1 - q) * nk) * xi, d)
                dens = np.where(new, d / nk, dens)
                new = lev > curlev
            Lam = (Lam * V + 8 * math.pi * alpha * dens * dV) / Vn
        V = Vn
    E = Hs / Hs[:, -1:]
    keep = lna >= LNA_LATE[0] - 0.01
    return lna[keep], E[:, keep], alive, Hs[:, -1] / H0, Lam, OLrec

np.seterr(all='ignore')
if __name__ == '__main__':
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    variants = [('step', 0.0), ('level', 0.0), ('level', 0.2), ('level', 0.3), ('level', 0.35), ('level', 0.37)]
    if len(sys.argv) > 2: variants = [variants[int(k)] for k in sys.argv[2].split(',')]
    t0 = time.time()
    oms = np.linspace(0.20, 0.45, 26); cl = np.array([lcdm_chi2(o) for o in oms])
    om_best = oms[cl.argmin()]; c0 = cl.min()
    print(f'ΛCDM: Ω_m = {om_best:.2f}, χ² = {c0:.1f} на {len(Z)} сверхновых   [{time.time()-t0:.0f} с]')
    ALPHAS = np.arange(0.002, 0.0241, 0.002); OMS = (0.24, 0.28, 0.32, 0.36, 0.40)
    for mode, q in variants:
        best = np.full(nseed, np.inf); crash = 0; bestpar = [None] * nseed
        bestc = np.full(nseed, np.inf); recb = np.zeros(nseed); h0b = np.zeros(nseed)
        for alpha in ALPHAS:
            for om in OMS:
                lna, E, alive, h0ratio, Lam, OLrec = histories(alpha, om, q, list(range(1000, 1000 + nseed)), mode)
                c = np.full(nseed, np.inf)
                if alive.any():
                    c[alive] = chi2(mu_from_E(lna, E[alive]))
                # внешние условия (как в статье): H0 = 60…80 км/с/Мпк, 0.2 < Ω_m < 0.45
                h0 = 67.7 * h0ratio; ome = om / h0ratio ** 2
                phys = (h0 > 60) & (h0 < 80) & (ome > 0.2) & (ome < 0.45)
                cc = np.where(phys, c, np.inf)
                upd = cc < bestc; bestc[upd] = cc[upd]; recb[upd] = OLrec[upd]; h0b[upd] = h0[upd]
                for k in np.where(c < best)[0]: bestpar[k] = (alpha, om)
                best = np.minimum(best, c); crash += (~alive).sum()
        ok = np.isfinite(best); D = best - c0
        np.save(f'/home/claude/data/everpresent_{mode}_{q}.npy', D)
        vst = (1 - q) / (1 - (q * TAU ** 2) ** 2) if mode == 'level' else 1.0
        print(f'{mode:5s} q={q:.2f} (амплитуда ×{vst**.5:.2f}): миров {nseed}; живых хоть при одном (α,Ω) {ok.sum()}; '
              f'лучше ΛCDM {np.sum(D < 0)} ({np.mean(D < 0)*100:.2f} %); Δχ²<4: {np.sum(D < 4)} ({np.mean(D < 4)*100:.1f} %); '
              f'Δχ²<25: {np.mean(D < 25)*100:.1f} %; медиана {np.median(D[ok]):.0f}; лучший {D.min():+.2f} при α,Ω={bestpar[int(D.argmin())]}; '
              f'падений {crash/(nseed*len(ALPHAS)*len(OMS))*100:.0f} %   [{time.time()-t0:.0f} с]')
        Dc = bestc - c0; np.save(f'/home/claude/data/everpresent_phys_{mode}_{q}.npy', np.stack([Dc, recb, h0b]))
        g = Dc < 4
        print(f'      с условиями на H0 и Ω_m: допустимых миров {np.isfinite(Dc).sum()}; лучше ΛCDM {np.sum(Dc < 0)} ({np.mean(Dc < 0)*100:.2f} %); '
              f'Δχ²<4: {g.sum()} ({np.mean(g)*100:.1f} %); Δχ²<25: {np.mean(Dc < 25)*100:.1f} %'
              + (f'; у хороших |Ω_Λ| на рекомбинации медиана {np.median(abs(recb[g])):.3f}, H0 медиана {np.median(h0b[g]):.0f}' if g.any() else ''))
        sys.stdout.flush()
