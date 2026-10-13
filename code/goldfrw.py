"""Кривизна как тень, честнее: скорость роста самой длинной цепи вдоль оси.
Плоское: dh/dη постоянна. Фридман с излучением (a = η): dh/dη ∝ a(η), то есть
отношение скоростей поздней и ранней половины ≈ a(2.7)/a(2.1) = 1.29.
Конечный размер искажает оба — поэтому есть плоские контроли (квазикристалл и
засев), и смотрим РАЗНИЦУ отношений «кривой минус плоский»."""
import sys, math, numpy as np
sys.argv = [sys.argv[0], '--b', '150']
import importlib.util
spec = importlib.util.spec_from_file_location('g3', 'gold3way.py')
src = open('gold3way.py').read().split("rs = np.random.default_rng(5)\nrq =")[0]
g = {}; exec(compile(src, 'g3', 'exec'), g)
frw_quasi, frw_sprinkle, order, height, r_for_density = g['frw_quasi'], g['frw_sprinkle'], g['order'], g['height'], g['r_for_density']
E1, E2, MID = g['E1'], g['E2'], g['MID']; RHO0 = 150.0
def slopes(P):
    P = P[np.argsort(P[:, 0])]; R = order(P); _, D = height(R)
    ax = np.linalg.norm(P[:, 1:], axis=1) < 0.18
    e, h = P[ax, 0], D[ax].astype(float)
    out = []
    for lo, hi in ((1.8, 2.4), (2.4, 3.0)):
        m = (e >= lo) & (e < hi); out.append(np.polyfit(e[m], h[m], 1)[0])
    return out[1] / out[0]
rs = np.random.default_rng(7)
rq = lambda eta: r_for_density(RHO0 * np.asarray(eta) ** 4)
rf = lambda eta: r_for_density(RHO0 * MID ** 4) + 0 * np.asarray(eta)
res = {}
for lab, gen in (('квазикристалл КРИВОЙ (r ∝ a)', lambda s: frw_quasi(rq, rs)),
                 ('квазикристалл плоский', lambda s: frw_quasi(rf, rs)),
                 ('засев КРИВОЙ (ρ ∝ a⁴)', lambda s: frw_sprinkle(2700, np.random.default_rng(60 + s), lambda e: (e / E2) ** 4)),
                 ('засев плоский', lambda s: frw_sprinkle(2700, np.random.default_rng(80 + s), lambda e: 1.0 + 0 * e))):
    v = np.array([slopes(gen(s)) for s in range(6)])
    res[lab] = (v.mean(), v.std(ddof=1) / math.sqrt(len(v)))
    print(f'  {lab:32s} отношение скоростей {v.mean():.3f} ± {res[lab][1]:.3f}'); sys.stdout.flush()
for a, b in (('квазикристалл КРИВОЙ (r ∝ a)', 'квазикристалл плоский'), ('засев КРИВОЙ (ρ ∝ a⁴)', 'засев плоский')):
    d = res[a][0] / res[b][0]; e = d * math.hypot(res[a][1] / res[a][0], res[b][1] / res[b][0])
    print(f'  {a.split()[0]}: кривой/плоский = {d:.3f} ± {e:.3f}   (Фридман ждёт ≈ 1.29, плоское 1)')
