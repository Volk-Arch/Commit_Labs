"""Анизотропия «собственного времени» у кристалла: помогает ли размытый край тени (скрытые фазы)?"""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from gold4 import coord_values, ST, SI
from chainspeed2 import longest_full, TAU0, ETA0
RHO = 20.0; W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
r0 = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3)
def prob(s, r, w): return np.clip((r * (1 + w) - s) / (2 * w * r), 0, 1)
# радиус, при котором средняя плотность та же: ∫P(s)·3s² ds = r0³
ss = np.linspace(0, 3 * r0, 20001)
def dens(r): return np.trapezoid(prob(ss, r, W) * 3 * ss ** 2, ss) / r0 ** 3
lo_, hi_ = 0.5 * r0, 1.5 * r0
for _ in range(60):
    mid = 0.5 * (lo_ + hi_); (lo_, hi_) = (mid, hi_) if dens(mid) < 1 else (lo_, mid)
RF = mid; RMAX = RF * (1 + W)
def fuzzy_box(lo, hi, sh, rng):
    t, tp = coord_values((lo[0] + sh[0]) / ST, (hi[0] + sh[0]) / ST, RMAX / SI); vals = [(t * ST - sh[0], (tp * SI) ** 2)]
    for k in (1, 2):
        cc, cp = coord_values(lo[k] + sh[k], hi[k] + sh[k], RMAX); vals.append((cc - sh[k], cp ** 2))
    P = np.zeros((1, 3)); Wt = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); Wt = np.repeat(Wt, len(cc))
        P[:, k] = np.tile(cc, m); Wt += np.tile(w2, m); keep = Wt <= RMAX ** 2; P, Wt = P[keep], Wt[keep]
    return P[rng.random(len(P)) < prob(np.sqrt(Wt), RF, W)]
if __name__ == '__main__':
    rng = np.random.default_rng(8); t0 = time.time()
    row = []
    for eta in (0.0, 0.53, ETA0, 1.6, 2 * ETA0):
        ls = []
        for s in range(5):
            th = rng.uniform(0, 2 * math.pi)
            B = np.array([TAU0 * math.cosh(eta), TAU0 * math.sinh(eta) * math.cos(th), TAU0 * math.sinh(eta) * math.sin(th)])
            m = np.array([0.5, TAU0 / 2 + 0.5, TAU0 / 2 + 0.5]); lo = np.minimum(0, B) - m; hi = np.maximum(0, B) + m
            P = fuzzy_box(lo, hi, rng.uniform(0, 300, 3), rng)
            d1 = P; d2 = B - P
            keep = (d1[:, 0] > 0) & (d1[:, 0] ** 2 > (d1[:, 1:] ** 2).sum(1)) & (d2[:, 0] > 0) & (d2[:, 0] ** 2 > (d2[:, 1:] ** 2).sum(1))
            ls.append(longest_full(np.vstack([[0, 0, 0], P[keep], B])) / TAU0)
        row.append(f'{eta:.2f}: {np.mean(ls):.3f}±{np.std(ls)/math.sqrt(len(ls)):.3f}')
    print(f'кристалл с краем w={W} случайно ' + '  '.join(row) + f'   [{time.time()-t0:.0f} с]')
