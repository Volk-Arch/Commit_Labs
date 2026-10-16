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
