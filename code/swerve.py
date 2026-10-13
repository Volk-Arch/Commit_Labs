"""«Виляние» частицы. Частица идёт по миру «как можно прямее»: из текущего события
выбирает следующее в будущем на собственном времени τ ∈ [2, 3] (ячейки), направление
которого ближе всего к её скорости u. Новая скорость — направление последних K шагов.
Меряем, как уходит скорость: ⟨η²⟩ (быстрота относительно исходной) от числа шагов.
Пуассон: случайное блуждание (растёт ∝ n). Кристалл: ?"""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from gold4 import points_in_box, r_for_density
t0 = time.time(); T1, T2 = 2.0, 3.0; HALF = 3.2
MINK = np.array([1., -1, -1, -1])
def mdot(a, b): return (a * b * MINK).sum(-1)

class Poisson:
    def __init__(self, seed): self.seed = seed; self.cache = {}
    def box(self, lo, hi):
        out = []
        for c in np.ndindex(*[int(math.floor(h)) - int(math.floor(l)) + 1 for l, h in zip(lo, hi)]):
            cell = tuple(int(math.floor(l)) + k for l, k in zip(lo, c))
            if cell not in self.cache:
                r = np.random.default_rng(abs(hash((self.seed,) + cell)) % (2 ** 63)); n = r.poisson(1.0)
                self.cache[cell] = np.array(cell) + r.random((n, 4))
            out.append(self.cache[cell])
        P = np.concatenate(out); return P[np.all((P >= lo) & (P <= hi), 1)]
class Crystal:
    def __init__(self, seed): self.r = r_for_density(1.0); self.shift = np.random.default_rng(seed).uniform(0, 400, 4)
    def box(self, lo, hi):
        return points_in_box([(l + s, h + s) for l, h, s in zip(lo, hi, self.shift)], self.r) - self.shift

def walk(world, u0, steps, K):
    x = np.zeros(4)
    # старт: ближайшее событие к началу
    P = world.box(x - 1, x + 1); x = P[np.argmin(np.linalg.norm(P, axis=1))]
    hist = [x.copy()]; u = u0.copy(); eta = []
    for n in range(steps):
        c = x + 2.5 * u
        P = world.box(c - HALF, c + HALF)
        d = P - x; tau2 = mdot(d, d)
        ok = (d[:, 0] > 0) & (tau2 > T1 ** 2) & (tau2 < T2 ** 2)
        if not ok.any(): return None
        d = d[ok]; tau = np.sqrt(tau2[ok]); ch = mdot(d, u[None]) / tau
        j = np.argmin(ch); x = x + d[j]; hist.append(x.copy())
        D = hist[-1] - hist[max(0, len(hist) - 1 - K)]
        u = D / math.sqrt(mdot(D, D))
        eta.append(math.acosh(max(1.0, mdot(u, u0))))
    return np.array(eta)

def u_of(eta, n): n = n / np.linalg.norm(n); return np.concatenate([[math.cosh(eta)], math.sinh(eta) * n])
if __name__ == '__main__':
    rng = np.random.default_rng(2)
    steps, npart = 150, 16
    for K in (1, 8):
        for name, mk, eta0 in (('Пуассон', Poisson, 1.5), ('кристалл, общая скорость', Crystal, 1.5), ('кристалл, покой (по оси)', Crystal, 0.0)):
            E = []
            for p in range(npart):
                u0 = u_of(eta0, rng.normal(size=3)) if eta0 > 0 else np.array([1., 0, 0, 0])
                e = walk(mk(int(rng.integers(1e9))), u0, steps, K)
                if e is not None: E.append(e)
            E = np.array(E); m2 = (E ** 2).mean(0)
            ns = np.arange(1, steps + 1); sel = ns >= 10
            beta = np.polyfit(np.log(ns[sel]), np.log(m2[sel] + 1e-30), 1)[0]
            print(f'K={K} {name:26s}: ⟨η²⟩ после 10/50/150 шагов: {m2[9]:.2e} / {m2[49]:.2e} / {m2[-1]:.2e};  рост ∝ n^{beta:.2f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
