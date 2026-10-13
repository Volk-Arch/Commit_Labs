import sys, time, math, numpy as np
sys.path.insert(0,'/home/claude'); sys.path.insert(0,'.')
from drawlab import grow, relmat, feats, sprinkle_R, NAMES

def band4(c): return c['P'] >= 0.035 * c['h']**4 and c['P'] <= 0.35 * c['h']**4
def bandL(c): return c['P'] > 0 and 1.0 <= c['L'] / math.sqrt(c['P']) <= 5.0
def share(c): return c['L'] > 0 and (c['prev_links'] == 0 or (c['links'] & c['prev_links']) != 0)
RULES = {
 'нет правила (перколяция)': lambda c: c['L'] > 0,
 'R4: 0.035 h⁴ ≤ |P| ≤ 0.35 h⁴': band4,
 'R7: 1 ≤ L/√|P| ≤ 5': bandL,
 'R4 и R7': lambda c: band4(c) and bandL(c),
 'R6: общая связь с соседом': share,
 'R4 и R6': lambda c: band4(c) and share(c),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ref = np.array([feats(sprinkle_R(N, np.random.default_rng(70+s))) for s in range(5)])
mu, sd = ref.mean(0), ref.std(0, ddof=1)
print(f'N={N}  ЗАСЕВ 4d:  ' + '  '.join(f'{k} {m:.3f}±{s:.3f}' for k, m, s in zip(NAMES, mu, sd)))
t0 = time.time()
for name, rule in RULES.items():
    best = None
    for kappa in (1, 2, 4, 8, 16, 32):
        F = []
        for s in range(2):
            past, got = grow(N, kappa, rule, np.random.default_rng(10*s+kappa))
            if past is None: F = None; break
            F.append(feats(relmat(past)))
        if F is None:
            print(f'  {name:32s} κ={kappa:2d}: рост ОСТАНОВЛЕН на n={got}'); continue
        F = np.mean(F, 0); z = (F - mu) / sd; chi = float(np.nansum(z**2))
        if best is None or chi < best[0]: best = (chi, kappa, F, z)
    if best:
        chi, kappa, F, z = best
        print(f'{name:34s} лучший κ={kappa:2d} χ²={chi:9.1f} | ' + '  '.join(f'{k} {v:.3f}({zz:+.0f})' for k, v, zz in zip(NAMES, F, z)) + f'  [{time.time()-t0:.0f}s]')
    sys.stdout.flush()
