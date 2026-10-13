"""Кристалл + случайность: каждое событие кристалла существует с вероятностью p.
Разброс в ромбе (n ≈ 30 000) в системе кристалла и в общей системе."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box, r_for_density
rs = np.random.default_rng(5); t0 = time.time()
rnd = np.array([.48, -.71, .52]); rnd /= np.linalg.norm(rnd)
for p in (1.0, 0.99, 0.9):
    rr = r_for_density(1.0 / p)                 # плотность после прореживания = 1
    NEXP = 30000; tau0 = (24 * NEXP / math.pi) ** 0.25; G.tau0 = tau0; row = []
    for name, eta, n in (('покой', 0.0, np.array([1.0, 0, 0])), ('буст 1.5', 1.5, rnd)):
        c = []
        for m in range(30):
            a = rs.uniform(0, 500, 4); b = a + tau0 * G.boost_vec(eta, n)
            P = points_in_box(G.interval_box(a, b, eta, n), rr); P = P[rs.random(len(P)) < p]
            c.append(int(G.inside(P, a, b).sum()))
        row.append(f'{name}: σ/√n = {np.std(c, ddof=1)/math.sqrt(np.mean(c)):.2f}')
    print(f'p = {p}:  ' + '   '.join(row) + f'   (ожидание чистой случайной части √(1−p) = {math.sqrt(1-p):.2f})  [{time.time()-t0:.0f} с]'); sys.stdout.flush()
