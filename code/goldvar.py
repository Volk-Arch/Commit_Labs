"""Как растёт разброс числа событий в причинном ромбе у золотого квазикристалла 3+1.
Пуассон: σ ~ n^0.5 (основа предсказания Соркина Λ ~ 1/√V).
Гипероднородный (Соболь, hyperuniform-doc): σ ~ n^0.375. Квазикристалл: ?"""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box
rs = np.random.default_rng(31)
t0 = time.time()
for eta, n_dir in ((0.0, np.array([1., 0, 0])), (1.5, np.array([.48, -.71, .52]) / np.linalg.norm([.48, -.71, .52]))):
    ns, sg = [], []
    for nexp in (30, 100, 300, 1000, 3000):
        tau0 = (24 * nexp / math.pi) ** 0.25
        c = []
        for m in range(60 if nexp < 1000 else 30):
            p = rs.uniform(0, 500, 4); q = p + tau0 * G.boost_vec(eta, n_dir)
            G.tau0 = tau0
            box = G.interval_box(p, q, eta, n_dir)
            c.append(int(G.inside(points_in_box(box, G.r), p, q).sum()))
        c = np.array(c); ns.append(c.mean()); sg.append(c.std(ddof=1))
        print(f'  η={eta}: ожидание {nexp:5d}  среднее {c.mean():8.1f}  σ {c.std(ddof=1):7.2f}  (Пуассон {math.sqrt(nexp):6.2f})  [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    k = np.polyfit(np.log(ns), np.log(sg), 1)[0]
    print(f'  показатель σ ~ n^k:  k = {k:.3f}   (Пуассон 0.5, гипероднородный 0.375)\n')
