"""Ширина «пуассоновского пика» около системы кристалла: σ/√n при малых быстротах
вдоль x, для n = 3 000 и 30 000. Если ширина ~ 1/τ₀ (размер ромба в длинах
дискретности), то для Вселенной (τ₀ ~ 10⁶¹) пик неуловимо узок."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box
rs = np.random.default_rng(21); t0 = time.time(); n = np.array([1.0, 0, 0])
for NEXP in (300, 100000):
    tau0 = (24 * NEXP / math.pi) ** 0.25; G.tau0 = tau0; line = []
    for eta in (0.0, 0.02, 0.04, 0.08, 0.15):
        c = []
        for m in range(40 if NEXP < 50000 else 24):
            p = rs.uniform(0, 500, 4); q = p + tau0 * G.boost_vec(eta, n)
            c.append(int(G.inside(points_in_box(G.interval_box(p, q, eta, n), G.r), p, q).sum()))
        line.append((eta, np.std(c, ddof=1) / math.sqrt(NEXP)))
    print(f'n={NEXP:6d} (τ₀={tau0:5.1f}, 1/τ₀={1/tau0:.3f}): ' + '  '.join(f'η={e}:{s:.2f}' for e, s in line) + f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
