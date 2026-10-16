"""ПОРЯДОК И СВЯЗИ. i ≺ j, если j − i во внутреннем световом конусе. Связь — пара без событий между.
1+1: «лестница» связей (двойное доминирование, как в песочнице) для полярного света; плотные C и связи через C·C;
3+1: экономные связи (τ связи золота ограничено ≤ 1.93ρ^{−¼}: z в интервале (i, j) — ближний кандидат j)."""
import numpy as np
from .core import precedes

def causal(X):
    """C[i, j] = True, если i ≺ j — отношение порядка из ядра (core.precedes)."""
    return precedes(X)

def links_dense(X, block=1500):
    """все связи плотно: C и (C·C) = 0. Возвращает (I, J) i → j."""
    C = causal(X).astype(np.float32); n = len(X); I_, J_ = [], []
    for s in range(0, n, block):
        B = C @ C[:, s:s + block]; li, lj = np.nonzero((C[:, s:s + block] > 0) & (B == 0)); I_.append(li); J_.append(lj + s)
    I, J = np.concatenate(I_), np.concatenate(J_); o = np.argsort(J, kind='stable'); return I[o], J[o]

def links_sparse(X, rho, tc_factor=2.1, progress=None):
    """экономные связи 3+1 (X отсортирован по t); проверено против плотных: −0.23 % (пары у края)."""
    tc2 = (tc_factor * rho ** -0.25) ** 2; n = len(X); I_, J_ = [], []
    for j in range(1, n):
        if progress and j % 5000 == 0: progress(j, n)
        d = X[j] - X[:j]; tau2 = d[:, 0] ** 2 - (d[:, 1:] ** 2).sum(1)
        c = np.nonzero((d[:, 0] > 0) & (tau2 > 1e-12) & (tau2 < tc2))[0]
        if len(c) == 0: continue
        Q = X[c]; e = Q[None, :, :] - Q[:, None, :]
        prec = (e[..., 0] > 0) & (e[..., 0] ** 2 - (e[..., 1:] ** 2).sum(-1) > 1e-12)
        lk = c[~prec.any(1)]; I_.append(lk); J_.append(np.full(len(lk), j))
    return np.concatenate(I_), np.concatenate(J_)

def staircase11(X):
    """связи 1+1 как «лестница» минимумов (песочница): j — непосредственный последователь i. Возвращает список (i, j)."""
    U = X[:, 0] + X[:, 1]; V = X[:, 0] - X[:, 1]; idx = np.argsort(U, kind='stable'); la, lb = [], []
    for i in range(len(X)):
        cand = idx[(U[idx] > U[i] + 1e-9) & (V[idx] > V[i] + 1e-9)]; minV = np.inf
        for j in cand:
            if V[j] < minV - 1e-9: la.append(i); lb.append(j); minV = V[j]
    return np.array(la), np.array(lb)

def time_levels(t, rho):
    """события золотого 1+1 лежат на уровнях времени; «толщина уровня» λ = ρ·Δt (мера мгновенной вершины)."""
    lv = np.unique(np.round(t, 9)); gap = np.diff(lv); half = 0.5 * (np.r_[gap[0], gap] + np.r_[gap, gap[-1]])
    return rho * half[np.searchsorted(lv, np.round(t, 9))]

def mm_dimension(X):
    """размерность Майрхейма–Мейера причинного интервала: f = R/(N(N−1)) (доля упорядоченных пар), f(d) = Γ(d+1)Γ(d/2)/(4Γ(3d/2));
    в 2D f = 1/4, в 4D f = 1/20."""
    from math import gamma
    from scipy.optimize import brentq
    n = len(X); C = causal(X); f = C.sum() / (n * (n - 1))
    g = lambda d: gamma(d + 1) * gamma(d / 2) / (4 * gamma(3 * d / 2)) - f
    try: return brentq(g, 1.01, 8.0), f
    except ValueError: return float('nan'), f

def layers31(X, rho, nmax=2, tc_factor=2.6):
    """слои 3+1: все пары i ≺ j, у которых в интервале не больше nmax событий; возвращает (I, J, n). nmax = 0 — связи.
    Интервал (i, j) целиком из «кандидатов» j (τ(k, j) ≤ τ(i, j) < τ_cut), поэтому n точно. Ускорение — numba, если есть."""
    tc2 = (tc_factor * rho ** -0.25) ** 2
    try:
        import numba
    except ImportError:
        numba = None
    if numba is None:
        I_, J_, N_ = [], [], []
        for j in range(1, len(X)):
            d = X[j] - X[:j]; tau2 = d[:, 0] ** 2 - (d[:, 1:] ** 2).sum(1); c = np.nonzero((d[:, 0] > 0) & (tau2 > 1e-12) & (tau2 < tc2))[0]
            if len(c) == 0: continue
            Q = X[c]; e = Q[None, :, :] - Q[:, None, :]; prec = (e[..., 0] > 0) & (e[..., 0] ** 2 - (e[..., 1:] ** 2).sum(-1) > 1e-12)
            nn = prec.sum(1); k = nn <= nmax; I_.append(c[k]); J_.append(np.full(k.sum(), j)); N_.append(nn[k])
        return np.concatenate(I_), np.concatenate(J_), np.concatenate(N_)
    return _layers31_nb(np.ascontiguousarray(X, dtype=np.float64), tc2, nmax)

def _layers31_nb(X, tc2, nmax):
    import numba
    global _L31
    if '_L31' not in globals():
        @numba.njit(cache=False)
        def f(X, tc2, nmax):
            n = X.shape[0]; cap = 1 << 22; I = np.empty(cap, np.int32); J = np.empty(cap, np.int32); N = np.empty(cap, np.int8); m = 0
            cand = np.empty(n, np.int64)
            for j in range(1, n):
                q = 0
                for i in range(j):
                    dt = X[j, 0] - X[i, 0]
                    if dt <= 0: continue
                    t2 = dt * dt - (X[j, 1] - X[i, 1]) ** 2 - (X[j, 2] - X[i, 2]) ** 2 - (X[j, 3] - X[i, 3]) ** 2
                    if t2 > 1e-12 and t2 < tc2: cand[q] = i; q += 1
                for a in range(q):
                    i = cand[a]; cnt = 0
                    for b in range(q):
                        k = cand[b]; dt = X[k, 0] - X[i, 0]
                        if dt <= 0: continue
                        if dt * dt - (X[k, 1] - X[i, 1]) ** 2 - (X[k, 2] - X[i, 2]) ** 2 - (X[k, 3] - X[i, 3]) ** 2 > 1e-12:
                            cnt += 1
                            if cnt > nmax: break
                    if cnt <= nmax:
                        if m >= I.shape[0]:
                            I2 = np.empty(2 * I.shape[0], np.int32); I2[:m] = I[:m]; I = I2
                            J2 = np.empty(2 * J.shape[0], np.int32); J2[:m] = J[:m]; J = J2
                            N2 = np.empty(2 * N.shape[0], np.int8); N2[:m] = N[:m]; N = N2
                        I[m] = i; J[m] = j; N[m] = cnt; m += 1
            return I[:m], J[:m], N[:m]
        _L31 = f
    return _L31(X, tc2, nmax)

LAYER_WEIGHTS = {0: (1.0,), 1: (2.0, -2.0), 2: (3.0, -7.0, 4.0)}   # выведены: нормировка, нет хвоста 1/t, нет следующего порядка (dirac-doc §23)
