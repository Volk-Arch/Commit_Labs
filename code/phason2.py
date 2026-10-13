"""Заряженная проба. Событие кристалла — точка 8-мерной решётки: (место x, тень x').
Нейтральная частица видит только x и идёт «как можно прямее» в мире. «Заряженная» видит
всю решётку: штраф за излом и в тени, λ·|Δx' − ū'|², где ū' — её внутренняя скорость
(среднее по последним K шагам). Фазон G сдвигает окно: доступные тени лежат около G·x.
Нечётный по G отклик поперёк скорости при λ > 0 и его отсутствие при λ = 0 — признак поля."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import phason as PH
from swerve import mdot, u_of, T1, T2, HALF
class PC(PH.PhasonCrystal):
    def box_s(self, lo, hi):
        c0 = 0.5 * (lo + hi); lo = lo + self.shift; hi = hi + self.shift
        if self.G is None: cen = np.zeros(4); rr = PH.RW
        else: cen = self.G @ c0; rr = PH.RW + np.abs(self.G).sum(1).max() * (0.5 * (hi - lo).max() + 0.5)
        vals = []
        t, tp = PH.cv_off(lo[0] / PH.ST, hi[0] / PH.ST, (cen[0] - rr) / PH.SI, (cen[0] + rr) / PH.SI); vals.append((t * PH.ST, tp * PH.SI))
        for k in (1, 2, 3):
            cc, cp = PH.cv_off(lo[k], hi[k], cen[k] - rr, cen[k] + rr); vals.append((cc, cp))
        P = np.zeros((1, 4)); S = np.zeros((1, 4)); W = np.zeros(1)
        for k, (cc, cp) in enumerate(vals):
            m = len(P); P = np.repeat(P, len(cc), 0); S = np.repeat(S, len(cc), 0); W = np.repeat(W, len(cc))
            P[:, k] = np.tile(cc, m); S[:, k] = np.tile(cp, m); W += np.tile((cp - cen[k]) ** 2, m); keep = W <= rr ** 2
            P, S, W = P[keep], S[keep], W[keep]
        X = P - self.shift; sh = X @ self.G.T if self.G is not None else 0.0
        k = ((S - sh) ** 2).sum(1) <= PH.RW ** 2
        return X[k], S[k]
def walk(world, u0, steps, lam, K=8):
    x = np.zeros(4); P, S = world.box_s(x - 1.5, x + 1.5); j = np.argmin(np.linalg.norm(P, axis=1)); x = P[j]; s = S[j]
    hx, hs = [x.copy()], [s.copy()]; u = u0.copy(); us = []
    for n in range(steps):
        c = x + 2.5 * u; P, S = world.box_s(c - HALF, c + HALF); d = P - x; tau2 = mdot(d, d)
        ok = (d[:, 0] > 0) & (tau2 > T1 ** 2) & (tau2 < T2 ** 2)
        if not ok.any(): return None
        d, ds, tau = d[ok], S[ok] - s, np.sqrt(tau2[ok])
        cost = mdot(d, u[None]) / tau
        if lam > 0 and len(hs) > 1:
            k0 = max(0, len(hs) - 1 - K); vbar = (hs[-1] - hs[k0]) / max(1, len(hs) - 1 - k0)
            cost = cost + lam * ((ds - vbar[None]) ** 2).sum(1)
        j = np.argmin(cost); x = x + d[j]; s = s + ds[j]; hx.append(x.copy()); hs.append(s.copy())
        D = hx[-1] - hx[max(0, len(hx) - 1 - K)]; u = D / math.sqrt(mdot(D, D)); us.append(u.copy())
    return np.array(us)
if __name__ == '__main__':
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 6); STEPS = 100; NP = int(sys.argv[1]) if len(sys.argv) > 1 else 12; t0 = time.time()
    for lam, g in ((0.0, 0.05), (1.0, 0.05), (1.0, 0.025), (2.0, 0.05)):
        for kind in ('вихрь', 'сдвиг'):
            perp, para = [], []
            for p in range(NP):
                seed = int(rng.integers(1e9)); u0 = u_of(1.2, np.array([*rng.normal(size=2), 0.3 * rng.normal()]))
                Up = walk(PC(seed, PH.Gmat(kind, g)), u0, STEPS, lam); Um = walk(PC(seed, PH.Gmat(kind, -g)), u0, STEPS, lam)
                if Up is None or Um is None: continue
                dU = 0.5 * (Up[-1] - Um[-1]); v = u0[1:3] / np.linalg.norm(u0[1:3]); ez = np.array([-v[1], v[0]])
                perp.append(float(dU[1:3] @ ez)); para.append(float(dU[1:3] @ v))
            perp, para = np.array(perp), np.array(para)
            print(f'λ={lam} g={g} {kind:5s}: нечётный отклик поперёк {perp.mean():+.4f} ± {perp.std()/math.sqrt(len(perp)):.4f}, вдоль {para.mean():+.4f} ± {para.std()/math.sqrt(len(para)):.4f} (n={len(perp)})   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
