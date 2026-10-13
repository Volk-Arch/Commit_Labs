"""Хорошие миры (лучше ΛCDM или Δχ²<4) из everpresent.py: какая у них Λ сегодня
и на рекомбинации (z≈1100) — грубая проверка того, что скажет реликтовое излучение."""
import sys; sys.argv = sys.argv[:1] + ['0']
import numpy as np, math
import everpresent as ep
np.seterr(all='ignore')
c0 = min(ep.lcdm_chi2(o) for o in np.linspace(0.2, 0.45, 26))
ALPHAS = np.arange(0.002, 0.0241, 0.002); OMS = (0.24, 0.28, 0.32, 0.36, 0.40)

def hist_full(alpha, om, q, seeds, mode):
    """тот же прогон, но возвращает Ω_Λ(a) = (Λ/3)/H² на всей сетке"""
    import types
    out = {}
    lna = np.linspace(math.log(1e-5), 0, 4000)
    # повторяем histories, записывая Λ/3H²: проще — малая обёртка с монкипатчем
    K = len(seeds); OL = np.zeros((K, len(lna)))
    orig = ep.histories
    # копия цикла через exec исходника с вставкой записи
    src = open(ep.__file__).read()
    start = src.index('def histories'); end = src.index('np.seterr')
    body = src[start:end].replace('Hs[:, i] = Hc', 'Hs[:, i] = Hc; OL[:, i] = (Lam / 3) / np.maximum(H2, 1e-300)')
    body = body.replace('def histories(', 'def hf(')
    ns = dict(ep.__dict__); ns['OL'] = OL; exec(body, ns)
    lna2, E, alive, h0r, Lam = ns['hf'](alpha, om, q, seeds, mode)
    return lna, OL, alive, h0r, E, lna2

for mode, q in (('step', 0.0), ('level', 0.0), ('level', 0.35)):
    D = np.load(f'/home/claude/data/everpresent_{mode}_{q}.npy')
    thr = 4 if (D < 4).sum() >= 5 else 25
    good = np.where(D < thr)[0]
    seeds_all = np.arange(1000, 4000)
    gs = list(seeds_all[good])
    best = np.full(len(gs), np.inf); rec = np.zeros(len(gs)); now = np.zeros(len(gs)); h0 = np.zeros(len(gs)); omeff = np.zeros(len(gs))
    for alpha in ALPHAS:
        for om in OMS:
            lna, OL, alive, h0r, E, lna2 = hist_full(alpha, om, q, gs, mode)
            c = np.full(len(gs), np.inf)
            if alive.any(): c[alive] = ep.chi2(ep.mu_from_E(lna2, E[alive]))
            better = c < best
            i_rec = np.argmin(abs(lna - math.log(1 / 1101)))
            rec[better] = OL[better, i_rec]; now[better] = OL[better, -1]; h0[better] = h0r[better]
            omeff[better] = om / h0r[better] ** 2
            best = np.minimum(best, c)
    d = best - c0
    print(f'{mode} q={q}: хороших миров (Δχ²<{thr}) {len(gs)} из 3000')
    print(f'   Ω_Λ сегодня: медиана {np.median(now):.2f};  эфф. Ω_m: медиана {np.median(omeff):.2f}')
    print(f'   |Ω_Λ| на рекомбинации: медиана {np.median(abs(rec)):.3f}, доля < 0.02: {np.mean(abs(rec) < 0.02):.2f}, < 0.005: {np.mean(abs(rec) < 0.005):.2f}')
    print(f'   H0/H0_номинал: медиана {np.median(h0):.2f}; доля в [0.9, 1.1]: {np.mean((h0 > 0.9) & (h0 < 1.1)):.2f}')
    sys.stdout.flush()
