"""Ящик □ из слоёв прошлого (dirac-doc §31). Слой k события x — все y ≺ x, у которых в интервале I(y, x) ровно k событий.
B φ(x) = (1/s)·Σ_k c_k Σ_{y ∈ слой k} (φ(y) − φ(x)); веса c_k — из обнуления первых моментов по гиперболоидам (для Пуассона 4 слоя дают
веса Соркина 1, −9, 16, −8; берём 5 слоёв: 1, −14.0, 41.2, −44.4, 16.2). Местный член Σ c_k N_k(x) — из счёта в самом событии
(□ константы = 0). Распространение — марш по порядку: φ(x) = (f − (1/s)Σ c φ(y)) / (M(x) − W(x)/s). Только сравнимые пары, только счёт:
ни координат, ни «кто раньше» у несравнимых. Координаты используются лишь для ускорения поиска кандидатов (окно по времени tmax)."""
import math, numpy as np
import numba

@numba.njit(parallel=True)
def _layers(X, NK, tmax, umax, lo, cnt_only, ptr, idx, lay):
    n = X.shape[0]; tot = np.zeros(n, np.int64)
    for x in numba.prange(n):
        # кандидаты: y в окне времени, y ≺ x, τ² < umax; сортировка по τ²(y, x)
        m = x - lo[x]; cy = np.empty(m, np.int64); cu = np.empty(m); q = 0
        for y in range(lo[x], x):
            dt = X[x, 0] - X[y, 0]
            if dt <= 0: continue
            u = dt * dt - (X[x, 1] - X[y, 1]) ** 2 - (X[x, 2] - X[y, 2]) ** 2 - (X[x, 3] - X[y, 3]) ** 2
            if u > 1e-12 and u < umax: cy[q] = y; cu[q] = u; q += 1
        o = np.argsort(cu[:q]); cy2 = cy[:q][o]; w = 0
        for p in range(q):
            y = cy2[p]; c = 0
            for r in range(p):                      # k внутри I(y, x) имеют τ²(k, x) < τ²(y, x)
                k = cy2[r]; dt = X[k, 0] - X[y, 0]
                if dt <= 0: continue
                if dt * dt - (X[k, 1] - X[y, 1]) ** 2 - (X[k, 2] - X[y, 2]) ** 2 - (X[k, 3] - X[y, 3]) ** 2 > 1e-12:
                    c += 1
                    if c >= NK: break
            if c < NK:
                if not cnt_only: idx[ptr[x] + w] = y; lay[ptr[x] + w] = c
                w += 1
        tot[x] = w
    return tot

def layers(X, NK=5, tmax=6.0, vmax=9.0, rho=1.0):
    """CSR: ptr, idx (y), lay (номер слоя) для каждого x. X отсортирован по времени."""
    X = np.ascontiguousarray(X, np.float64); n = len(X); umax = math.sqrt(24 * vmax / (math.pi * rho))
    lo = np.searchsorted(X[:, 0], X[:, 0] - tmax).astype(np.int64)
    z = np.zeros(1, np.int64); tot = _layers(X, NK, tmax, umax, lo, True, z, np.zeros(1, np.int32), np.zeros(1, np.int8))
    ptr = np.zeros(n + 1, np.int64); ptr[1:] = np.cumsum(tot); idx = np.empty(ptr[-1], np.int32); lay = np.empty(ptr[-1], np.int8)
    _layers(X, NK, tmax, umax, lo, False, ptr, idx, lay); return ptr, idx, lay

# ---------- средняя теория (k = 0) ----------
_AL = math.pi / 24; _UMAX = (16 / _AL) ** 0.5; _UG = np.linspace(0, _UMAX, 40001); _WG = np.full(len(_UG), _UG[1]); _WG[0] = _WG[-1] = _UG[1] / 2
def Pk_poisson(k): return lambda v: v ** k * np.exp(-v) / math.factorial(k)
def _moments(b, K): return np.array([np.sum(_WG * b * _UG ** j) for j in range(K)])
def design(P, NK):
    """c_k: обнуление моментов M0..M(NK−2) ядра Σ c_k P_k(ρV) по τ² (ρ = 1)."""
    Ms = np.array([_moments(P[k](_AL * _UG ** 2), NK) for k in range(NK)]).T; ns = np.linalg.svd(Ms[:NK - 1, :NK])[2][-1]; return ns / ns[0]
def _A(b, t):
    out = np.empty(len(t))
    for i, tt in enumerate(t):
        if tt * tt > _UMAX * 1.0001: r = np.sqrt(tt * tt - _UG) - tt + _UG / (2 * tt) + _UG ** 2 / (8 * tt ** 3); out[i] = 2 * np.pi * np.sum(_WG * b * r)
        else: out[i] = 2 * np.pi * np.sum(_WG * b * np.sqrt(np.maximum(tt * tt - _UG, 0)))
    return out
def mean_box(c, P, oms, T=80.0, N=6000, tmax=None):
    """B̂(ω) = −α + ∫e^{iωt}A(t)dt (сырое); tmax — обрезка окна по времени, как в марше."""
    t = np.linspace(1e-4, T, N); b = sum(ck * P[k](_AL * _UG ** 2) for k, ck in enumerate(c)); A = _A(b, t)
    if tmax: A = A * (t < tmax)
    win = np.exp(-(t / T * 3) ** 2); F = lambda w: np.trapezoid(np.exp(1j * w * t) * A * win, t)
    return F(0.0).real, np.array([F(w) for w in oms])
def coeffs(NK=5, tmax=None):
    """веса слоёв c, нормировка s (B = сырое/s ≈ −ω²) и голая масса m_b²(m) = −Re B̂(m) — средняя теория, Пуассон, ρ = 1."""
    P = [Pk_poisson(k) for k in range(NK)]; c = design(P, NK); F0, Fs = mean_box(c, P, np.array([0.05]), tmax=tmax); s = -((Fs[0] - F0).real) / 0.05 ** 2
    def mb2(m): F0_, Fm = mean_box(c, P, np.array([m]), tmax=tmax); return -float(((Fm[0] - F0_) / s).real)
    def im(m): F0_, Fm = mean_box(c, P, np.array([m]), tmax=tmax); return float(((Fm[0] - F0_) / s).imag)
    return c, s, mb2, im

@numba.njit
def march(ptr, idx, lay, c, s, Mloc, f):
    """спинор: Mloc (n,2,2) — местная масса/вставки, f (n,2). φ(x) = (Mloc − W/s)^{-1}(f − (1/s)Σ c φ(y))."""
    n = f.shape[0]; phi = np.zeros((n, 2), np.complex128)
    for x in range(n):
        a0 = 0j; a1 = 0j; W = 0.0
        for e in range(ptr[x], ptr[x + 1]):
            y = idx[e]; w = c[lay[e]]; W += w; a0 += w * phi[y, 0]; a1 += w * phi[y, 1]
        r0 = f[x, 0] - a0 / s; r1 = f[x, 1] - a1 / s
        m00 = Mloc[x, 0, 0] - W / s; m01 = Mloc[x, 0, 1]; m10 = Mloc[x, 1, 0]; m11 = Mloc[x, 1, 1] - W / s
        det = m00 * m11 - m01 * m10
        phi[x, 0] = (m11 * r0 - m01 * r1) / det; phi[x, 1] = (-m10 * r0 + m00 * r1) / det
    return phi
