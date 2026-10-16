"""Массивная частица 3+1 без цепочки по массе (dirac-doc §32). G_m = δ(τ²)/2π − θ(τ²)·m J1(mτ)/(4πτ): край конуса — один шаг по связям
(сравнимые пары с пустым интервалом, вес c = √ρ/(2π√6)), внутренность — по всем сравнимым парам, τ — из счёта интервала (таблица n → v
кристалла, τ⁴ = 24v/(πρ)) или из координат (для сравнения). Масса входит в вес, а не повторением безмассового шага — разгонять нечему.
Потенциал — цепочкой вставок V = 2EU − U² при энергии E, спин — σ·∇U. Несравнимые пары не нужны. Плотные матрицы: мир до ~1.5·10⁴ событий."""
import math, numpy as np
from scipy.special import j1
from . import atom31f as Af

def kernel(X, rho, M, mode='order', ch=1500):
    """KT[x, y] — вес от y к x. mode: 'order' (τ из счёта), 'coord' (τ из координат), 's1' (размытый конус, для сравнения)."""
    N = len(X); Xf = X.astype(np.float32); t = Xf[:, 0]; coef = math.sqrt(rho) / (2 * math.pi * math.sqrt(6)); al = rho * math.pi / 24
    def rows(a, b):
        dt = t[None, :] - t[a:b, None]; U = dt * dt
        for k in (1, 2, 3): U -= (Xf[None, :, k] - Xf[a:b, None, k]) ** 2
        return dt, U
    Cf = np.zeros((N, N), np.float32)
    for a in range(0, N, ch):
        dt, U = rows(a, min(N, a + ch)); Cf[a:a + ch] = (dt > 0) & (U > 0)
    sv = np.zeros(4001); cv = np.zeros(4001)
    for a in range(0, N, ch):
        b = min(N, a + ch); dt, U = rows(a, b); C = (dt > 0) & (U > 0); n = np.minimum(Cf[a:b] @ Cf, 4000).astype(int); v = al * U * U
        sv += np.bincount(n[C], v[C], 4001); cv += np.bincount(n[C], minlength=4001)
    vt = np.where(cv > 20, sv / np.maximum(cv, 1), np.nan); kk = np.nonzero(np.isfinite(vt))[0]; vtab = np.interp(np.arange(4001), kk, vt[kk])
    K = np.zeros((N, N), np.float32)
    for a in range(0, N, ch):
        b = min(N, a + ch); dt, U = rows(a, b); C = (dt > 0) & (U > 0); n = np.minimum(Cf[a:b] @ Cf, 4000).astype(int)
        if mode == 's1':
            v = al * U * U; K[a:b] = np.where((dt > 0) & (np.abs(U) < (2.8 * rho ** -0.25) ** 2), 0.5 * coef * np.exp(-v) * np.where(U > 0, 1 - Fz_beta(), 1 + Fz_beta()), 0)
        else:
            tau = np.sqrt(U.clip(0)) if mode == 'coord' else (24 * vtab[n] / (math.pi * rho)) ** 0.25
            tau = np.maximum(tau, 1e-6); K[a:b] = np.where(C & (n == 0), coef, 0) + np.where(C, -M * j1(M * tau) / (4 * np.pi * tau), 0)
    del Cf
    return np.ascontiguousarray(K.T), vtab

def Fz_beta(): return Af.BETA

def level_and_spin(X, rho, M, H, R, mode='order', spin=True):
    """уровень 1s (вставка при E, гармоническая инверсия) и спиновый сдвиг на одном мире."""
    X = X[np.argsort(X[:, 0])]; N = len(X); KT, _ = kernel(X, rho, M, mode); at = Af.Atom(X, rho, M, H, R)
    rr = np.linalg.norm(X[:, 1:], axis=1); Un = -Af.AL / np.sqrt(rr ** 2 + Af.A ** 2); MB = at.mb2(M) if mode == 's1' else 0.0
    def runs(b, pauli):
        Bm = (b[:, None, None] * at.eye + pauli * 1j * at.dU[:, None, None] * at.sr) / rho; phi = np.zeros((N, 2), complex); src = np.array(at.src(), complex)
        for j in range(N):
            phi[j] = KT[j, :j] @ src[:j]; src[j] += Bm[j] @ phi[j]
        return phi[:, 0]
    def lev(phi):
        if not np.all(np.isfinite(phi)) or np.abs(phi).max() > 1e12: return np.nan
        return Af.level_hinv(at, phi)
    E = Af._fixed_point(lambda E: lev(runs(-(MB + 2 * E * Un - Un * Un), 0.0)), 0.85 * M, tol=5e-4, it=6)
    Es = Af._fixed_point(lambda x: lev(runs(-(MB + 2 * x * Un - Un * Un), 1.0)), E, tol=3e-4, it=6) - E if spin else np.nan
    return E, Es
