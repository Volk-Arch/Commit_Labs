"""МИРЫ = ядро (core.crystal) + область обзора. Сама формула мира — в core.py; здесь только «где смотрим»:
1+1 — ромб |t| + |x| ≤ L (точно симметричен x → −x и бусту η₀ = 1.0613); 3+1 — ромб или цилиндр (окно опыта);
контроль — посыпка Пуассона (не кристалл). Все миры отсортированы по времени (t — столбец 0)."""
import math
import numpy as np
from .core import TAU, TAU_ as TP, S5, ST, SI, crystal, r_for_density, numbers as zt_range

ETA0 = math.acosh(TAU)

class World:
    def __init__(self, X, kind, rho, meta=None):
        o = np.argsort(X[:, 0], kind='stable'); self.X = np.ascontiguousarray(X[o]); self.kind = kind; self.rho = rho
        self.meta = meta or {}; self.n = len(self.X); self.dim = self.X.shape[1]
    @property
    def t(self): return self.X[:, 0]
    @property
    def x(self): return self.X[:, 1]
    def summary(self): return {'kind': self.kind, 'rho': self.rho, 'n': self.n, 'dim': self.dim, **self.meta}

def gold11(rho=4.0, L=12.0):
    """золотой мир 1+1 в ромбе |t| + |x| ≤ L."""
    r = r_for_density(rho, 2); X = crystal([(-L, L), (-L, L)], r)
    return World(X[np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L], 'gold11', rho, {'L': L, 'window_r': r})

def poisson11(rho=4.0, L=12.0, seed=3):
    rs = np.random.default_rng(seed); N = rs.poisson(rho * 2 * L * L)
    u = rs.uniform(-L, L, (N, 2)); X = np.stack([(u[:, 0] + u[:, 1]) / 2, (u[:, 0] - u[:, 1]) / 2], 1)
    return World(X[np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L], 'poisson11', rho, {'L': L, 'seed': seed})

# ---------- 3+1: то же правило в четырёх осях ----------
def r_for_density31(rho): return r_for_density(rho, 4)
def points_in_box31(box, r): return crystal(box, r)

def gold31(rho=1.0, shape='diamond', H=10.0, Rc=7.0, Rs=3.0):
    """3+1: 'diamond' — причинный ромб высоты H; 'cylinder' — цилиндр радиуса Rc ∩ конусы источника (радиус Rs при t = 0) и
    наблюдения (как hydro4b.py); 'tube' — ящик для опыта Кулона."""
    r = r_for_density31(rho)
    if shape == 'diamond':
        hh = H / 2; P = points_in_box31([(-hh, hh)] + [(-hh, hh)] * 3, r); P = P[np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= hh]
    elif shape == 'cylinder':
        P = points_in_box31([(-0.6, H + 0.1)] + [(-Rc, Rc)] * 3, r); rr = np.linalg.norm(P[:, 1:], axis=1)
        P = P[rr <= np.minimum(Rc, np.minimum(Rs + np.maximum(P[:, 0], 0) + 0.6, Rc + (H - P[:, 0])))]
    else:
        raise ValueError(shape)
    return World(P, 'gold31', rho, {'shape': shape, 'H': H, 'Rc': Rc, 'Rs': Rs})

def make(kind='gold11', **kw):
    return {'gold11': gold11, 'poisson11': poisson11, 'gold31': gold31}[kind](**kw)

# ---------- то же правило с другим числом вместо τ (опыт «почему золотое сечение») ----------
def unit_crystal(th, thc, rho=4.0, L=12.0):
    """события a + bθ (время × √|θ|), тень a + bθ' (× √|θ'|) в шаре; θ — единица с θθ' = −1 или дробь p/q (тогда θ' = −q/p,
    одинаковые точки сливаются). Радиус окна подбирается по счёту, чтобы плотность была ρ. Возвращает World и |тень|²."""
    ST, SI = math.sqrt(abs(th)), math.sqrt(abs(thc)); r = math.sqrt((th - thc) ** 2 * rho / math.pi)
    def nums(lo, hi, rlo, rhi):
        D = th - thc; C, Cp = [], []
        for b in range(math.floor((lo - rhi) / D) - 2, math.ceil((hi - rlo) / D) + 3):
            alo = max(math.ceil(lo - b * th), math.ceil(rlo - b * thc)); ahi = min(math.floor(hi - b * th), math.floor(rhi - b * thc))
            if ahi >= alo: a = np.arange(alo, ahi + 1, dtype=float); C.append(a + b * th); Cp.append(a + b * thc)
        return (np.concatenate(C), np.concatenate(Cp)) if C else (np.empty(0), np.empty(0))
    for _ in range(3):
        c, cp = nums(-L / ST, L / ST, -r / SI, r / SI); T, Tq = c * ST, cp * SI; d, dp = nums(-L, L, -r, r)
        i, j = np.nonzero(Tq[:, None] ** 2 + dp[None, :] ** 2 <= r * r); X = np.stack([T[i], d[j]], 1); sh = Tq[i] ** 2 + dp[j] ** 2
        _, u = np.unique(np.round(X, 7), axis=0, return_index=True); X, sh = X[u], sh[u]
        keep = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L; X, sh = X[keep], sh[keep]; r *= math.sqrt(rho / (len(X) / (2 * L * L)))
    o = np.argsort(X[:, 0], kind='stable'); return World(X[o], 'unit', rho, {'L': L, 'theta': th, 'window_r': r}), sh[o]
