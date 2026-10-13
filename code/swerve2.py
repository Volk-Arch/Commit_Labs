"""Кристалл подольше: насыщается ли уход скорости и не «залипает» ли частица на
выделенных скоростях (это было бы нарушение Лоренца)."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from swerve import Crystal, mdot, u_of, T1, T2, HALF
t0 = time.time(); rng = np.random.default_rng(7)
def walk2(world, u0, steps, K):
    x = np.zeros(4); P = world.box(x - 1, x + 1); x = P[np.argmin(np.linalg.norm(P, axis=1))]
    hist = [x.copy()]; u = u0.copy(); us = []
    for n in range(steps):
        c = x + 2.5 * u; P = world.box(c - HALF, c + HALF); d = P - x; tau2 = mdot(d, d)
        ok = (d[:, 0] > 0) & (tau2 > T1 ** 2) & (tau2 < T2 ** 2); d = d[ok]; tau = np.sqrt(tau2[ok])
        j = np.argmin(mdot(d, u[None]) / tau); x = x + d[j]; hist.append(x.copy())
        D = hist[-1] - hist[max(0, len(hist) - 1 - K)]; u = D / math.sqrt(mdot(D, D)); us.append(u.copy())
    return np.array(us)
if __name__ == '__main__':
    for K in (1, 8):
        for eta0 in (0.5, 1.5, 2.5):
            dev, absfin = [], []
            for p in range(24):
                u0 = u_of(eta0, rng.normal(size=3)); U = walk2(Crystal(int(rng.integers(1e9))), u0, 600, K)
                dev.append(np.arccosh(np.maximum(1, mdot(U, u0[None])))); absfin.append(math.acosh(U[-1, 0]))
            dev = np.array(dev); m2 = (dev ** 2).mean(0)
            print(f'K={K} η₀={eta0}: ⟨η²⟩ при 50/200/600 шагах {m2[49]:.3f}/{m2[199]:.3f}/{m2[-1]:.3f}; '
                  f'конечная быстрота в системе кристалла: среднее {np.mean(absfin):.2f} ± {np.std(absfin):.2f} (старт {eta0})   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
