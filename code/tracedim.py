import sys, math, numpy as np
from scipy.special import gamma
from scipy.optimize import brentq
from traceorder import world, order
def f_of_d(d): return gamma(d + 1) * gamma(d / 2) / (4 * gamma(3 * d / 2))
def mm_dim(f): return brentq(lambda d: 2 * f_of_d(d) - f, 1.0, 12)
rs = np.random.default_rng(5); P, r = world(4.0, 22.0, rs)
Rp, Rf = order(P, False), order(P, True)
tau_phys = lambda a, b: math.sqrt(max((P[b, 0] - P[a, 0]) ** 2 - (P[b, 1] - P[a, 1]) ** 2, 0))
for lo, hi in ((2, 4), (4, 7), (7, 12), (12, 20), (20, 32)):
    out = {False: [], True: []}
    tries = 0
    while len(out[True]) < 40 and tries < 20000:
        tries += 1
        a, b = rs.integers(len(P), size=2)
        if not (Rp[a, b]): continue
        t = tau_phys(a, b)
        if not (lo <= t < hi) or abs(P[a, 0] + P[b, 0]) / 2 > 8 or abs(P[a, 1] + P[b, 1]) / 2 > 8: continue
        for full, R in ((False, Rp), (True, Rf)):
            if not R[a, b]: continue
            I = np.nonzero(R[a] & R[:, b])[0]
            if len(I) < 6: continue
            S = R[np.ix_(I, I)]; n = len(I); f = S.sum() / (n * (n - 1) / 2)
            out[full].append(mm_dim(min(max(f, 0.02), 0.99)))
    print(f'τ ∈ [{lo},{hi}): размерность по порядку — старое {np.mean(out[False]):.2f} ± {np.std(out[False]) / math.sqrt(max(1, len(out[False]))):.2f}, '
          f'новое (след) {np.mean(out[True]):.2f} ± {np.std(out[True]) / math.sqrt(max(1, len(out[True]))):.2f}   [{len(out[False])}, {len(out[True])} ромбов]'); sys.stdout.flush()
