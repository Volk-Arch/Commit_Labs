import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
rs = np.random.default_rng(21)
dirs = {'(1,1,0)/√2 рациональное': np.array([1, 1, 0]) / 2 ** .5,
        '(1,2,2)/3 рациональное': np.array([1, 2, 2]) / 3.0,
        'случайное': np.array([0.48, -0.71, 0.52]) / np.linalg.norm([0.48, -0.71, 0.52])}
t0 = time.time()
for dname, n in dirs.items():
    print(f'  буст {dname}')
    for eta in (2.0, 3.0, 4.0, 4.5):
        cg, cz = [], []
        for m in range(25 if eta < 4 else 12):
            p = rs.uniform(0, 500, 4); q = p + G.tau0 * G.boost_vec(eta, n)
            box = G.interval_box(p, q, eta, n)
            cg.append(G.count_gold(p, q, box))
            cz.append(G.count_z4(p, q, box) if eta <= 3 else np.nan)
        cg, cz = np.array(cg), np.array(cz)
        zs = f'Z⁴ {np.nanmean(cz):5.1f} ± {np.nanstd(cz):5.1f} (пустых {np.nanmean(cz==0):.2f})' if eta <= 3 else 'Z⁴ —'
        print(f'    η={eta}: квазикристалл {cg.mean():5.1f} ± {cg.std():4.1f} (пустых {np.mean(cg==0):.2f}) | {zs}  [{time.time()-t0:.0f} с]')
        sys.stdout.flush()
