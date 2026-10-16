import sys, math, numpy as np
from traceorder import world, order, longest
rs = np.random.default_rng(8); res = {False: [], True: []}
L = 22.0
for w in range(4):
    P, r = world(4.0, L, rs)
    Rp, Rf = order(P, False), order(P, True)
    bot = np.nonzero(P[:, 0] < -L + 3)[0]; top = np.nonzero(P[:, 0] > L - 3)[0]
    for _ in range(40):
        a, b = rs.choice(bot), rs.choice(top)
        if not (Rp[a, b] and Rf[a, b]): continue
        for full, R in ((False, Rp), (True, Rf)):
            ch, ln = longest(R, a, b)
            th = np.unwrap(np.arctan2(P[ch, 3], P[ch, 2])); h = len(th) // 2
            w1, w2 = th[h] - th[0], th[-1] - th[h]
            res[full].append((w1, w2, th[-1] - th[0], len(ch)))
for full in (False, True):
    A = np.array(res[full]); c = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    print(f'{"след" if full else "старое"}: цепочек {len(A)}, длина {A[:, 3].mean():.1f}; намотка в окне за всю цепочку: |Δθ| ≈ {np.abs(A[:, 2]).mean():.2f} рад (разброс {A[:, 2].std():.2f}); '
          f'связь намотки первой и второй половины: {c:+.3f} ± {1 / math.sqrt(len(A)):.3f}')
