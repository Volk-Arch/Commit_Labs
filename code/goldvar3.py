"""Разброс в ромбах квазикристалла до n = 100 000: в покое; с бустом η₀ = 1.0613
вдоль x (точная симметрия — обязан совпасть с покоем); с бустом 1.5 в случайном
направлении."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box
rs = np.random.default_rng(91); t0 = time.time()
ETA0 = math.acosh((1 + 5 ** .5) / 2)
rnd = np.array([.48, -.71, .52]); rnd /= np.linalg.norm(rnd)
for name, eta, n in (('покой', 0.0, np.array([1.0, 0, 0])), ('буст η₀ вдоль x', ETA0, np.array([1.0, 0, 0])), ('буст 1.5, случайно', 1.5, rnd)):
    out = []
    for nexp, M in ((3000, 40), (30000, 30), (100000, 24)):
        tau0 = (24 * nexp / math.pi) ** 0.25; G.tau0 = tau0; c = []
        for m in range(M):
            p = rs.uniform(0, 500, 4); q = p + tau0 * G.boost_vec(eta, n)
            c.append(int(G.inside(points_in_box(G.interval_box(p, q, eta, n), G.r), p, q).sum()))
        c = np.array(c); s = c.std(ddof=1); out.append((nexp, s))
        print(f'  {name:20s} n={nexp:7d}  σ/√n = {s/math.sqrt(nexp):.3f} ± {s/math.sqrt(nexp)/math.sqrt(2*(M-1)):.3f}  [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    ns, sg = np.array(out).T; print(f'  → показатель {np.polyfit(np.log(ns), np.log(sg), 1)[0]:.3f}\n')
