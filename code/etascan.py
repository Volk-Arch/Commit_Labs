"""Дрожь числа событий как функция быстроты: вдоль оси x (через точную
симметрию η₀ = 1.0613) и в случайном направлении. n = 20 000, 30 ромбов на точку."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box
rs = np.random.default_rng(13); t0 = time.time()
NEXP = 20000; tau0 = (24 * NEXP / math.pi) ** 0.25; G.tau0 = tau0
rnd = np.array([.48, -.71, .52]); rnd /= np.linalg.norm(rnd)
diag = np.array([1, 1, 1.0]) / math.sqrt(3)
for name, n in (('вдоль x', np.array([1.0, 0, 0])), ('по диагонали (1,1,1)', diag), ('случайное', rnd)):
    line = []
    for eta in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0613, 1.3, 1.6, 2.1226):
        c = []
        for m in range(30):
            p = rs.uniform(0, 500, 4); q = p + tau0 * G.boost_vec(eta, n)
            c.append(int(G.inside(points_in_box(G.interval_box(p, q, eta, n), G.r), p, q).sum()))
        line.append((eta, np.std(c, ddof=1) / math.sqrt(NEXP)))
    print(f'{name:22s} ' + '  '.join(f'η={e:.2f}:{s:.2f}' for e, s in line) + f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
print('(ошибка каждого σ/√n около ±0.13·σ/√n)')
