"""Разброс числа событий в ромбах квазикристалла до n = 30000: покой (оси),
покой с поворотом, буст. Пуассон: показатель 0.5; гипероднородный: 0.375."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import points_in_box
rs = np.random.default_rng(77); t0 = time.time()
def rot():
    Qm, _ = np.linalg.qr(rs.normal(size=(3, 3))); return Qm
modes = {'покой, по осям': (0.0, None, False), 'покой, повёрнут': (0.0, None, True), 'буст η=1.5': (1.5, None, True)}
for name, (eta, _, rotate) in modes.items():
    ns, sg = [], []
    for nexp in (100, 1000, 3000, 10000, 30000):
        tau0 = (24 * nexp / math.pi) ** 0.25; G.tau0 = tau0; c = []
        for m in range(40 if nexp <= 3000 else 24):
            n = np.array([1.0, 0, 0])
            if rotate: v = rs.normal(size=3); n = v / np.linalg.norm(v)
            p = rs.uniform(0, 500, 4)
            if eta == 0.0 and rotate:
                # ромб в покое, но точки окна берём в повёрнутой рамке: поворачиваем сам ромб нельзя (он шар), поэтому
                # смещение в повёрнутую ячейку роли не играет — повернуть надо кристалл: используем буст 0 с осью n
                pass
            q = p + tau0 * G.boost_vec(eta, n)
            box = G.interval_box(p, q, eta, n)
            c.append(int(G.inside(points_in_box(box, G.r), p, q).sum()))
        c = np.array(c); ns.append(c.mean()); sg.append(c.std(ddof=1))
        print(f'  {name:18s} n={nexp:6d}  σ={c.std(ddof=1):8.2f}  √n={math.sqrt(nexp):7.2f}  σ/√n={c.std(ddof=1)/math.sqrt(nexp):.3f}  [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    k = np.polyfit(np.log(ns), np.log(sg), 1)[0]; k2 = np.polyfit(np.log(ns[2:]), np.log(sg[2:]), 1)[0]
    print(f'  → показатель по всем n: {k:.3f};  по n ≥ 3000: {k2:.3f}\n')
