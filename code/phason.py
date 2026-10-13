"""Первый шаг к калибровочному полю: фазон. Окно тени сдвигается плавно по миру:
событие есть ⟺ |тень − G·x| ≤ r (x — физическое положение). G — постоянный градиент.
Два вида: «вихрь» (антисимметричный в плоскости xy: w_x = g·y, w_y = −g·x — как магнитное
поле B_z) и «сдвиг» (симметричный: w_x = g·y, w_y = g·x). Частица идёт «как можно прямее»
(правило swerve, K = 8) в одном и том же кристалле с G и без G (парное сравнение).
Если фазон — поле вроде магнитного, разность скоростей растёт ⟂ скорости, меняет знак с g."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from gold4 import coord_values, ST, SI, r_for_density
from swerve import mdot, u_of, T1, T2, HALF
t0 = time.time(); RW = r_for_density(1.0)
class PhasonCrystal:
    def __init__(self, seed, G):
        self.shift = np.random.default_rng(seed).uniform(0, 400, 4); self.G = G
        self.extra = 0.0 if G is None else 1.0
    def box(self, lo, hi):
        c0 = 0.5 * (lo + hi)
        lo = lo + self.shift; hi = hi + self.shift
        if self.G is None: cen = np.zeros(4); rr = RW
        else: cen = self.G @ c0; rr = RW + np.abs(self.G).sum(1).max() * (0.5 * (hi - lo).max() + 0.5)
        vals = []
        t, tp = cv_off(lo[0] / ST, hi[0] / ST, (cen[0] - rr) / SI, (cen[0] + rr) / SI); vals.append((t * ST, tp * SI))
        for k in (1, 2, 3):
            cc, cp = cv_off(lo[k], hi[k], cen[k] - rr, cen[k] + rr); vals.append((cc, cp))
        P = np.zeros((1, 4)); S = np.zeros((1, 4)); W = np.zeros(1)
        for k, (cc, cp) in enumerate(vals):
            m = len(P); P = np.repeat(P, len(cc), 0); S = np.repeat(S, len(cc), 0); W = np.repeat(W, len(cc))
            P[:, k] = np.tile(cc, m); S[:, k] = np.tile(cp, m); W += np.tile((cp - cen[k]) ** 2, m); keep = W <= rr ** 2
            P, S, W = P[keep], S[keep], W[keep]
        X = P - self.shift
        sh = X @ self.G.T if self.G is not None else 0.0
        return X[((S - sh) ** 2).sum(1) <= RW ** 2]
TAU_ = (1 + 5 ** .5) / 2; TP_ = -1 / TAU_
def cv_off(lo, hi, slo, shi):
    """c = a + bτ, lo ≤ c ≤ hi, slo ≤ c' ≤ shi"""
    C, Cp = [], []
    for b in range(math.floor((lo - shi) / 5 ** .5) - 1, math.ceil((hi - slo) / 5 ** .5) + 2):
        alo = max(math.ceil(lo - b * TAU_), math.ceil(slo - b * TP_)); ahi = min(math.floor(hi - b * TAU_), math.floor(shi - b * TP_))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1); C.append(a + b * TAU_); Cp.append(a + b * TP_)
    return (np.concatenate(C), np.concatenate(Cp)) if C else (np.empty(0), np.empty(0))
def walk(world, u0, steps, K=8):
    x = np.zeros(4); P = world.box(x - 1.5, x + 1.5); x = P[np.argmin(np.linalg.norm(P, axis=1))]
    hist = [x.copy()]; u = u0.copy(); us = []
    for n in range(steps):
        c = x + 2.5 * u; P = world.box(c - HALF, c + HALF); d = P - x; tau2 = mdot(d, d)
        ok = (d[:, 0] > 0) & (tau2 > T1 ** 2) & (tau2 < T2 ** 2)
        if not ok.any(): return None
        d = d[ok]; tau = np.sqrt(tau2[ok]); j = np.argmin(mdot(d, u[None]) / tau)
        x = x + d[j]; hist.append(x.copy()); D = hist[-1] - hist[max(0, len(hist) - 1 - K)]
        u = D / math.sqrt(mdot(D, D)); us.append(u.copy())
    return np.array(us)
def Gmat(kind, g):
    G = np.zeros((4, 4))
    if kind == 'вихрь': G[1, 2], G[2, 1] = g, -g
    else: G[1, 2], G[2, 1] = g, g
    return G
if __name__ == '__main__':
    rng = np.random.default_rng(4); STEPS = 120; NP = 12
    for g in (0.02, 0.05):
        for kind in ('вихрь', 'сдвиг'):
            perp, para, cross = [], [], []
            for p in range(NP):
                seed = int(rng.integers(1e9)); u0 = u_of(1.2, np.array([*rng.normal(size=2), 0.3 * rng.normal()]))
                U0 = walk(PhasonCrystal(seed, None), u0, STEPS); Up = walk(PhasonCrystal(seed, Gmat(kind, g)), u0, STEPS); Um = walk(PhasonCrystal(seed, Gmat(kind, -g)), u0, STEPS)
                if U0 is None or Up is None or Um is None: continue
                # нечётная по g часть отклика — то, что меняет знак вместе с полем
                dU = 0.5 * (Up[-1] - Um[-1]); v = u0[1:3] / np.linalg.norm(u0[1:3])
                ez = np.array([-v[1], v[0]])                               # поперёк скорости в плоскости xy
                perp.append(float(dU[1:3] @ ez)); para.append(float(dU[1:3] @ v))
                cross.append(float(np.linalg.norm(Up[-1] - U0[-1])))
            perp, para = np.array(perp), np.array(para)
            print(f'{kind:5s} g={g}: нечётный отклик после {STEPS} шагов: поперёк {perp.mean():+.4f} ± {perp.std()/math.sqrt(len(perp)):.4f}, '
                  f'вдоль {para.mean():+.4f} ± {para.std()/math.sqrt(len(para)):.4f}; |изменение скорости| {np.mean(cross):.3f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
