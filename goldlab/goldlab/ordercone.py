"""Свет 3+1 из порядка (dirac-doc §30). Вес размытого конуса S1 e^{−ατ⁴} = e^{−ρV}, V — объём наименьшего причинного ромба,
содержащего обе точки. Сравнимая сторона: V читается счётом событий в интервале. Несравнимая: наименьший ромб с вершинами-событиями
(смещён и шумит). Односторонний конус (только сравнимые пары, знакопеременный вес) — средняя теория: раскачка на частотах ~ зерна."""
import math, numpy as np
from math import gamma, sqrt, pi

def causal(X):
    """плотные матрицы: U[a,b] = τ² от a к b, C[a,b] = a ≺ b, Cnt[a,b] = #{k: a ≺ k ≺ b} (мир до ~10⁴ событий)."""
    Xf = np.asarray(X, np.float32); t = Xf[:, 0]; dt = t[None, :] - t[:, None]; U = dt * dt
    for a in (1, 2, 3): U -= (Xf[None, :, a] - Xf[:, None, a]) ** 2
    C = (dt > 0) & (U > 0); Cf = C.astype(np.float32); return U, C, Cf @ Cf

def pair_stats(X, rho, inner, tcf=2.8, ntarget=250, cap=12, seed=5):
    """пары у светового конуса (|τ²| < τc²) с концами в inner: (τ², ρV из координат, n из порядка). Для несравнимых n — наименьший
    ромб I(p,q) ∋ i,j с вершинами-событиями минус 2 (сами i, j); поиск вершин — среди близких (≤ cap событий до i и j)."""
    U, C, Cnt = causal(X); t = X[:, 0]; tc2 = (tcf * rho ** -0.25) ** 2; idx = np.where(inner)[0]; out = []
    for j in np.random.default_rng(seed).choice(idx, min(len(idx), ntarget), replace=False):
        for i in idx:
            if t[i] >= t[j] or abs(U[i, j]) >= tc2: continue
            if C[i, j]: n = Cnt[i, j]
            else:
                Pm = np.where(C[:, i] & C[:, j] & (Cnt[:, i] <= cap) & (Cnt[:, j] <= cap))[0]; Qm = np.where(C[i, :] & C[j, :] & (Cnt[i, :] <= cap) & (Cnt[j, :] <= cap))[0]
                if len(Pm) == 0 or len(Qm) == 0: continue
                n = Cnt[np.ix_(Pm, Qm)].min() - 2
            u = float(U[i, j]); out.append((u, rho * pi / 24 * u * u, float(n)))
    return np.array(out)

# ---------- средняя теория (k = 0): δ(t) = ∫ядро по пространству − t; G(ω) = ∫e^{iωt}δ(t)dt, Im G < 0 — раскачка ----------
def consts(rho=1.0):
    c = sqrt(rho) / (2 * pi * sqrt(6)); al = rho * pi / 24; return c, al, sqrt(al)

def kernel_S1(rho=1.0, beta=0.115):
    c, al, sa = consts(rho); return lambda u: 0.5 * c * np.exp(-al * u * u) * np.where(u > 0, 1 - beta, 1 + beta)

def kernel_onesided(rho=1.0, bp=0.05):
    """h = (c/2)·a·e^{−v}(1 − λ√v), v = ρV, только u > 0; M0 как у S1, M1 = −(c/2)β′/α."""
    c, al, sa = consts(rho); lam = sqrt(pi) * (1 + bp) / (pi / 2 + bp); a = 2 * sqrt(pi) / (sqrt(pi) - lam)
    return (lambda u: np.where(u > 0, 0.5 * c * a * np.exp(-al * u * u) * (1 - lam * sa * np.maximum(u, 0)), 0.0)), (a, lam)

def mean_G(h, oms, rho=1.0, T=40.0, N=1200):
    c, al, sa = consts(rho); t = np.linspace(1e-6, T, N); A = np.empty(N)
    for i, tt in enumerate(t):
        uu = np.linspace(-9 / sa, min(tt * tt, 60.0), 40001); A[i] = np.trapezoid(2 * np.pi * np.sqrt(np.maximum(tt * tt - uu, 0)) * h(uu), uu)
    d = (A - t) * np.exp(-(t / T * 2.5) ** 2)
    return np.array([np.trapezoid(np.exp(1j * w * t) * d, t) for w in oms])
